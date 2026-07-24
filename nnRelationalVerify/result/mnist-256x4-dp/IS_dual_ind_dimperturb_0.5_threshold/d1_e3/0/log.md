## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0005928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0025398, 1.0047827, 1.0025398, 1.0047827, -0.0014188, 0.0014188)
1: (-0.0006311, -0.0000722, -0.0006311, -0.0000722, -0.0003535, 0.0003535)
2: (-0.0096713, -0.0067094, -0.0096713, -0.0067094, -0.0018735, 0.0018735)
3: (0.0017807, 0.0031288, 0.0017807, 0.0031288, -0.0008527, 0.0008527)
4: (-0.0013440, -0.0007707, -0.0013440, -0.0007707, -0.0003626, 0.0003626)
5: (-0.0132044, -0.0094792, -0.0132044, -0.0094792, -0.0023564, 0.0023564)
6: (0.0039468, 0.0048923, 0.0039468, 0.0048923, -0.0005981, 0.0005981)
7: (0.0070738, 0.0095201, 0.0070738, 0.0095201, -0.0015474, 0.0015474)
8: (0.0041559, 0.0054424, 0.0041559, 0.0054424, -0.0008138, 0.0008138)
9: (-0.0081746, -0.0066828, -0.0081746, -0.0066828, -0.0009436, 0.0009436)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.58 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0007494, upper bound: 0.0007495

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007021, upper bound: 0.0006245
time: 0.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007021, upper bound: 0.0007021
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -0.0007021, upper bound: 0.0006245
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -0.0007021, upper bound: 0.0007021

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 1.0026388, 1.0045794, 1.0025405, 1.0047183, -0.0011436, 0.0011856
1: -0.0006064, -0.0001229, -0.0006309, -0.0000883, -0.0002849, 0.0002954
2: -0.0094027, -0.0068402, -0.0095861, -0.0067105, -0.0015655, 0.0015101
3: 0.0018403, 0.0030066, 0.0017812, 0.0030901, -0.0006873, 0.0007126
4: -0.0012920, -0.0007960, -0.0013275, -0.0007709, -0.0003030, 0.0002923
5: -0.0128666, -0.0096438, -0.0130974, -0.0094805, -0.0019690, 0.0018993
6: 0.0039885, 0.0048065, 0.0039471, 0.0048651, -0.0004821, 0.0004998
7: 0.0071819, 0.0092983, 0.0070747, 0.0094498, -0.0012472, 0.0012930
8: 0.0042127, 0.0053257, 0.0041564, 0.0054054, -0.0006559, 0.0006800
9: -0.0080393, -0.0067487, -0.0081317, -0.0066834, -0.0007885, 0.0007606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0006245
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0006245
time: 0.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 1.0025423, 1.0046903, 1.0025405, 1.0047536, -0.0014039, 0.0010662
1: -0.0006305, -0.0000953, -0.0006309, -0.0000795, -0.0003498, 0.0002657
2: -0.0095492, -0.0067126, -0.0096328, -0.0067103, -0.0014079, 0.0018538
3: 0.0017822, 0.0030733, 0.0017811, 0.0031113, -0.0008438, 0.0006408
4: -0.0013203, -0.0007713, -0.0013365, -0.0007709, -0.0002725, 0.0003588
5: -0.0130509, -0.0094832, -0.0131561, -0.0094804, -0.0017707, 0.0023316
6: 0.0039478, 0.0048533, 0.0039471, 0.0048800, -0.0005918, 0.0004494
7: 0.0070765, 0.0094193, 0.0070746, 0.0094884, -0.0015312, 0.0011628
8: 0.0041573, 0.0053894, 0.0041563, 0.0054257, -0.0008052, 0.0006115
9: -0.0081131, -0.0066844, -0.0081552, -0.0066833, -0.0007091, 0.0009337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0007021
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0007021
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0006245
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0006245
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0007021
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -0.0006244, upper bound: 0.0007021

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026388, 1.0045794, 1.0026388, 1.0045794, -0.0009912, 0.0009912
1: -0.0006064, -0.0001229, -0.0006064, -0.0001229, -0.0002470, 0.0002470
2: -0.0094027, -0.0068402, -0.0094027, -0.0068402, -0.0013088, 0.0013088
3: 0.0018403, 0.0030066, 0.0018403, 0.0030066, -0.0005957, 0.0005957
4: -0.0012920, -0.0007960, -0.0012920, -0.0007960, -0.0002533, 0.0002533
5: -0.0128666, -0.0096438, -0.0128666, -0.0096438, -0.0016461, 0.0016461
6: 0.0039885, 0.0048065, 0.0039885, 0.0048065, -0.0004178, 0.0004178
7: 0.0071819, 0.0092983, 0.0071819, 0.0092983, -0.0010810, 0.0010810
8: 0.0042127, 0.0053257, 0.0042127, 0.0053257, -0.0005685, 0.0005685
9: -0.0080393, -0.0067487, -0.0080393, -0.0067487, -0.0006592, 0.0006592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005819, upper bound: 0.0005864
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006124, upper bound: 0.0005956
time: 0.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026388, 1.0045794, 1.0025423, 1.0046903, -0.0011897, 0.0011841
1: -0.0006064, -0.0001229, -0.0006305, -0.0000953, -0.0002964, 0.0002951
2: -0.0094027, -0.0068402, -0.0095492, -0.0067126, -0.0015636, 0.0015709
3: 0.0018403, 0.0030066, 0.0017822, 0.0030733, -0.0007150, 0.0007117
4: -0.0012920, -0.0007960, -0.0013203, -0.0007713, -0.0003026, 0.0003040
5: -0.0128666, -0.0096438, -0.0130509, -0.0094832, -0.0019667, 0.0019758
6: 0.0039885, 0.0048065, 0.0039478, 0.0048533, -0.0005015, 0.0004992
7: 0.0071819, 0.0092983, 0.0070765, 0.0094193, -0.0012975, 0.0012915
8: 0.0042127, 0.0053257, 0.0041573, 0.0053894, -0.0006823, 0.0006792
9: -0.0080393, -0.0067487, -0.0081131, -0.0066844, -0.0007875, 0.0007912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005819, upper bound: 0.0005864
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006124, upper bound: 0.0005956
time: 0.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025423, 1.0046903, 1.0026388, 1.0045794, -0.0011841, 0.0011897
1: -0.0006305, -0.0000953, -0.0006064, -0.0001229, -0.0002951, 0.0002964
2: -0.0095492, -0.0067126, -0.0094027, -0.0068402, -0.0015709, 0.0015636
3: 0.0017822, 0.0030733, 0.0018403, 0.0030066, -0.0007117, 0.0007150
4: -0.0013203, -0.0007713, -0.0012920, -0.0007960, -0.0003040, 0.0003026
5: -0.0130509, -0.0094832, -0.0128666, -0.0096438, -0.0019758, 0.0019667
6: 0.0039478, 0.0048533, 0.0039885, 0.0048065, -0.0004992, 0.0005015
7: 0.0070765, 0.0094193, 0.0071819, 0.0092983, -0.0012915, 0.0012975
8: 0.0041573, 0.0053894, 0.0042127, 0.0053257, -0.0006792, 0.0006823
9: -0.0081131, -0.0066844, -0.0080393, -0.0067487, -0.0007912, 0.0007875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006653
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0006732
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025423, 1.0046903, 1.0025423, 1.0046903, -0.0010646, 0.0010646
1: -0.0006305, -0.0000953, -0.0006305, -0.0000953, -0.0002653, 0.0002653
2: -0.0095492, -0.0067126, -0.0095492, -0.0067126, -0.0014057, 0.0014057
3: 0.0017822, 0.0030733, 0.0017822, 0.0030733, -0.0006398, 0.0006398
4: -0.0013203, -0.0007713, -0.0013203, -0.0007713, -0.0002721, 0.0002721
5: -0.0130509, -0.0094832, -0.0130509, -0.0094832, -0.0017680, 0.0017680
6: 0.0039478, 0.0048533, 0.0039478, 0.0048533, -0.0004487, 0.0004487
7: 0.0070765, 0.0094193, 0.0070765, 0.0094193, -0.0011611, 0.0011611
8: 0.0041573, 0.0053894, 0.0041573, 0.0053894, -0.0006106, 0.0006106
9: -0.0081131, -0.0066844, -0.0081131, -0.0066844, -0.0007080, 0.0007080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006653
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0006732
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0005819, upper bound: 0.0005864
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0006124, upper bound: 0.0005956
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0005819, upper bound: 0.0005864
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0006124, upper bound: 0.0005956
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006653
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0006732
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006653
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0006732

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045787, 1.0026388, 1.0045794, -0.0008182, 0.0009850
1: -0.0006022, -0.0001231, -0.0006064, -0.0001229, -0.0002039, 0.0002454
2: -0.0094019, -0.0068626, -0.0094027, -0.0068402, -0.0013007, 0.0010805
3: 0.0018505, 0.0030062, 0.0018403, 0.0030066, -0.0004918, 0.0005920
4: -0.0012918, -0.0008004, -0.0012920, -0.0007960, -0.0002517, 0.0002091
5: -0.0128656, -0.0096719, -0.0128666, -0.0096438, -0.0016359, 0.0013589
6: 0.0039957, 0.0048063, 0.0039885, 0.0048065, -0.0003449, 0.0004152
7: 0.0072004, 0.0092976, 0.0071819, 0.0092983, -0.0008924, 0.0010743
8: 0.0042225, 0.0053254, 0.0042127, 0.0053257, -0.0004693, 0.0005649
9: -0.0080389, -0.0067600, -0.0080393, -0.0067487, -0.0006551, 0.0005442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0005819
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0006127
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045787, 1.0025423, 1.0046903, -0.0010486, 0.0011780
1: -0.0006022, -0.0001231, -0.0006305, -0.0000953, -0.0002613, 0.0002935
2: -0.0094019, -0.0068626, -0.0095492, -0.0067126, -0.0015555, 0.0013847
3: 0.0018505, 0.0030062, 0.0017822, 0.0030733, -0.0006303, 0.0007080
4: -0.0012918, -0.0008004, -0.0013203, -0.0007713, -0.0003011, 0.0002680
5: -0.0128656, -0.0096719, -0.0130509, -0.0094832, -0.0019564, 0.0017416
6: 0.0039957, 0.0048063, 0.0039478, 0.0048533, -0.0004420, 0.0004966
7: 0.0072004, 0.0092976, 0.0070765, 0.0094193, -0.0011437, 0.0012847
8: 0.0042225, 0.0053254, 0.0041573, 0.0053894, -0.0006015, 0.0006756
9: -0.0080389, -0.0067600, -0.0081131, -0.0066844, -0.0007834, 0.0006974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006653, upper bound: 0.0005701
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006653, upper bound: 0.0005956
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047839, 1.0026772, 1.0045784, -0.0010065, 0.0011716
1: -0.0005930, -0.0000719, -0.0005969, -0.0001232, -0.0002508, 0.0002919
2: -0.0096727, -0.0069113, -0.0094013, -0.0068910, -0.0015471, 0.0013291
3: 0.0018726, 0.0031295, 0.0018634, 0.0030059, -0.0006049, 0.0007042
4: -0.0013442, -0.0008098, -0.0012917, -0.0008058, -0.0002994, 0.0002572
5: -0.0132063, -0.0097331, -0.0128649, -0.0097076, -0.0019459, 0.0016716
6: 0.0040112, 0.0048927, 0.0040047, 0.0048061, -0.0004243, 0.0004939
7: 0.0072406, 0.0095214, 0.0072238, 0.0092972, -0.0010977, 0.0012778
8: 0.0042436, 0.0054430, 0.0042348, 0.0053251, -0.0005773, 0.0006720
9: -0.0081753, -0.0067845, -0.0080386, -0.0067743, -0.0007792, 0.0006694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006448
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006653
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046896, 1.0026388, 1.0045794, -0.0010351, 0.0011843
1: -0.0006262, -0.0000954, -0.0006064, -0.0001229, -0.0002579, 0.0002951
2: -0.0095483, -0.0067356, -0.0094027, -0.0068402, -0.0015638, 0.0013669
3: 0.0017926, 0.0030728, 0.0018403, 0.0030066, -0.0006221, 0.0007118
4: -0.0013202, -0.0007758, -0.0012920, -0.0007960, -0.0003027, 0.0002646
5: -0.0130498, -0.0095121, -0.0128666, -0.0096438, -0.0019669, 0.0017191
6: 0.0039551, 0.0048530, 0.0039885, 0.0048065, -0.0004363, 0.0004992
7: 0.0070955, 0.0094186, 0.0071819, 0.0092983, -0.0011289, 0.0012916
8: 0.0041673, 0.0053890, 0.0042127, 0.0053257, -0.0005937, 0.0006793
9: -0.0081127, -0.0066960, -0.0080393, -0.0067487, -0.0007876, 0.0006884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006448
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006732
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047839, 1.0025803, 1.0046891, -0.0008601, 0.0010096
1: -0.0005930, -0.0000719, -0.0006210, -0.0000956, -0.0002143, 0.0002516
2: -0.0096727, -0.0069113, -0.0095476, -0.0067630, -0.0013332, 0.0011357
3: 0.0018726, 0.0031295, 0.0018051, 0.0030725, -0.0005169, 0.0006068
4: -0.0013442, -0.0008098, -0.0013200, -0.0007811, -0.0002580, 0.0002198
5: -0.0132063, -0.0097331, -0.0130489, -0.0095466, -0.0016768, 0.0014284
6: 0.0040112, 0.0048927, 0.0039639, 0.0048528, -0.0003626, 0.0004256
7: 0.0072406, 0.0095214, 0.0071181, 0.0094180, -0.0009380, 0.0011011
8: 0.0042436, 0.0054430, 0.0041792, 0.0053887, -0.0004933, 0.0005791
9: -0.0081753, -0.0067845, -0.0081123, -0.0067098, -0.0006715, 0.0005720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005702, upper bound: 0.0006448
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005702, upper bound: 0.0006653
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046896, 1.0025423, 1.0046903, -0.0008887, 0.0010584
1: -0.0006262, -0.0000954, -0.0006305, -0.0000953, -0.0002214, 0.0002637
2: -0.0095483, -0.0067356, -0.0095492, -0.0067126, -0.0013976, 0.0011735
3: 0.0017926, 0.0030728, 0.0017822, 0.0030733, -0.0005341, 0.0006361
4: -0.0013202, -0.0007758, -0.0013203, -0.0007713, -0.0002705, 0.0002271
5: -0.0130498, -0.0095121, -0.0130509, -0.0094832, -0.0017578, 0.0014759
6: 0.0039551, 0.0048530, 0.0039478, 0.0048533, -0.0003746, 0.0004461
7: 0.0070955, 0.0094186, 0.0070765, 0.0094193, -0.0009692, 0.0011543
8: 0.0041673, 0.0053890, 0.0041573, 0.0053894, -0.0005097, 0.0006070
9: -0.0081127, -0.0066960, -0.0081131, -0.0066844, -0.0007039, 0.0005910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006448
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006732
time: 0.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.95 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0005819
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0006127
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0006653, upper bound: 0.0005701
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0006653, upper bound: 0.0005956
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006448
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005701, upper bound: 0.0006653
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006448
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006732
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005702, upper bound: 0.0006448
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005702, upper bound: 0.0006653
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006448
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006732

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045787, 1.0027900, 1.0046738, -0.0009943, 0.0007818
1: -0.0006022, -0.0001231, -0.0005688, -0.0000994, -0.0002477, 0.0001948
2: -0.0094019, -0.0068626, -0.0095274, -0.0070398, -0.0010324, 0.0013129
3: 0.0018505, 0.0030062, 0.0019311, 0.0030634, -0.0005976, 0.0004699
4: -0.0012918, -0.0008004, -0.0013161, -0.0008347, -0.0001998, 0.0002541
5: -0.0128656, -0.0096719, -0.0130236, -0.0098948, -0.0012985, 0.0016513
6: 0.0039957, 0.0048063, 0.0040522, 0.0048464, -0.0004191, 0.0003296
7: 0.0072004, 0.0092976, 0.0073467, 0.0094014, -0.0010844, 0.0008527
8: 0.0042225, 0.0053254, 0.0042994, 0.0053799, -0.0005703, 0.0004484
9: -0.0080389, -0.0067600, -0.0081022, -0.0068492, -0.0005200, 0.0006612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005228
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0005334
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045787, 1.0026557, 1.0045787, -0.0008148, 0.0008148
1: -0.0006022, -0.0001231, -0.0006022, -0.0001231, -0.0002030, 0.0002030
2: -0.0094019, -0.0068626, -0.0094019, -0.0068626, -0.0010759, 0.0010759
3: 0.0018505, 0.0030062, 0.0018505, 0.0030062, -0.0004897, 0.0004897
4: -0.0012918, -0.0008004, -0.0012918, -0.0008004, -0.0002082, 0.0002082
5: -0.0128656, -0.0096719, -0.0128656, -0.0096719, -0.0013532, 0.0013532
6: 0.0039957, 0.0048063, 0.0039957, 0.0048063, -0.0003435, 0.0003435
7: 0.0072004, 0.0092976, 0.0072004, 0.0092976, -0.0008886, 0.0008886
8: 0.0042225, 0.0053254, 0.0042225, 0.0053254, -0.0004673, 0.0004673
9: -0.0080389, -0.0067600, -0.0080389, -0.0067600, -0.0005419, 0.0005419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005451
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0005495
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045787, 1.0026926, 1.0047839, -0.0012310, 0.0010026
1: -0.0006022, -0.0001231, -0.0005930, -0.0000719, -0.0003067, 0.0002498
2: -0.0094019, -0.0068626, -0.0096727, -0.0069113, -0.0013240, 0.0016255
3: 0.0018505, 0.0030062, 0.0018726, 0.0031295, -0.0007399, 0.0006026
4: -0.0012918, -0.0008004, -0.0013442, -0.0008098, -0.0002563, 0.0003146
5: -0.0128656, -0.0096719, -0.0132063, -0.0097331, -0.0016652, 0.0020445
6: 0.0039957, 0.0048063, 0.0040112, 0.0048927, -0.0005189, 0.0004226
7: 0.0072004, 0.0092976, 0.0072406, 0.0095214, -0.0013426, 0.0010935
8: 0.0042225, 0.0053254, 0.0042436, 0.0054430, -0.0007060, 0.0005751
9: -0.0080389, -0.0067600, -0.0081753, -0.0067845, -0.0006668, 0.0008187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005214
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045787, 1.0025595, 1.0046896, -0.0010448, 0.0010316
1: -0.0006022, -0.0001231, -0.0006262, -0.0000954, -0.0002603, 0.0002571
2: -0.0094019, -0.0068626, -0.0095483, -0.0067356, -0.0013623, 0.0013796
3: 0.0018505, 0.0030062, 0.0017926, 0.0030728, -0.0006279, 0.0006200
4: -0.0012918, -0.0008004, -0.0013202, -0.0007758, -0.0002637, 0.0002670
5: -0.0128656, -0.0096719, -0.0130498, -0.0095121, -0.0017134, 0.0017352
6: 0.0039957, 0.0048063, 0.0039551, 0.0048530, -0.0004404, 0.0004349
7: 0.0072004, 0.0092976, 0.0070955, 0.0094186, -0.0011395, 0.0011252
8: 0.0042225, 0.0053254, 0.0041673, 0.0053890, -0.0005992, 0.0005917
9: -0.0080389, -0.0067600, -0.0081127, -0.0066960, -0.0006861, 0.0006948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005341
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005368
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047839, 1.0027900, 1.0046738, -0.0010177, 0.0010336
1: -0.0005930, -0.0000719, -0.0005688, -0.0000994, -0.0002536, 0.0002575
2: -0.0096727, -0.0069113, -0.0095274, -0.0070398, -0.0013648, 0.0013438
3: 0.0018726, 0.0031295, 0.0019311, 0.0030634, -0.0006116, 0.0006212
4: -0.0013442, -0.0008098, -0.0013161, -0.0008347, -0.0002642, 0.0002601
5: -0.0132063, -0.0097331, -0.0130236, -0.0098948, -0.0017166, 0.0016902
6: 0.0040112, 0.0048927, 0.0040522, 0.0048464, -0.0004290, 0.0004357
7: 0.0072406, 0.0095214, 0.0073467, 0.0094014, -0.0011099, 0.0011273
8: 0.0042436, 0.0054430, 0.0042994, 0.0053799, -0.0005837, 0.0005928
9: -0.0081753, -0.0067845, -0.0081022, -0.0068492, -0.0006874, 0.0006768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0005824
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0005989
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047839, 1.0026557, 1.0045787, -0.0010026, 0.0012310
1: -0.0005930, -0.0000719, -0.0006022, -0.0001231, -0.0002498, 0.0003067
2: -0.0096727, -0.0069113, -0.0094019, -0.0068626, -0.0016255, 0.0013240
3: 0.0018726, 0.0031295, 0.0018505, 0.0030062, -0.0006026, 0.0007399
4: -0.0013442, -0.0008098, -0.0012918, -0.0008004, -0.0003146, 0.0002563
5: -0.0132063, -0.0097331, -0.0128656, -0.0096719, -0.0020445, 0.0016652
6: 0.0040112, 0.0048927, 0.0039957, 0.0048063, -0.0004226, 0.0005189
7: 0.0072406, 0.0095214, 0.0072004, 0.0092976, -0.0010935, 0.0013426
8: 0.0042436, 0.0054430, 0.0042225, 0.0053254, -0.0005751, 0.0007060
9: -0.0081753, -0.0067845, -0.0080389, -0.0067600, -0.0008187, 0.0006668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046896, 1.0027900, 1.0046738, -0.0011806, 0.0009811
1: -0.0006262, -0.0000954, -0.0005688, -0.0000994, -0.0002942, 0.0002445
2: -0.0095483, -0.0067356, -0.0095274, -0.0070398, -0.0012955, 0.0015590
3: 0.0017926, 0.0030728, 0.0019311, 0.0030634, -0.0007096, 0.0005897
4: -0.0013202, -0.0007758, -0.0013161, -0.0008347, -0.0002508, 0.0003017
5: -0.0130498, -0.0095121, -0.0130236, -0.0098948, -0.0016295, 0.0019608
6: 0.0039551, 0.0048530, 0.0040522, 0.0048464, -0.0004977, 0.0004136
7: 0.0070955, 0.0094186, 0.0073467, 0.0094014, -0.0012876, 0.0010700
8: 0.0041673, 0.0053890, 0.0042994, 0.0053799, -0.0006772, 0.0005627
9: -0.0081127, -0.0066960, -0.0081022, -0.0068492, -0.0006525, 0.0007852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046896, 1.0026557, 1.0045787, -0.0010316, 0.0010448
1: -0.0006262, -0.0000954, -0.0006022, -0.0001231, -0.0002571, 0.0002603
2: -0.0095483, -0.0067356, -0.0094019, -0.0068626, -0.0013796, 0.0013623
3: 0.0017926, 0.0030728, 0.0018505, 0.0030062, -0.0006200, 0.0006279
4: -0.0013202, -0.0007758, -0.0012918, -0.0008004, -0.0002670, 0.0002637
5: -0.0130498, -0.0095121, -0.0128656, -0.0096719, -0.0017352, 0.0017134
6: 0.0039551, 0.0048530, 0.0039957, 0.0048063, -0.0004349, 0.0004404
7: 0.0070955, 0.0094186, 0.0072004, 0.0092976, -0.0011252, 0.0011395
8: 0.0041673, 0.0053890, 0.0042225, 0.0053254, -0.0005917, 0.0005992
9: -0.0081127, -0.0066960, -0.0080389, -0.0067600, -0.0006948, 0.0006861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005994
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0006131
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047839, 1.0026926, 1.0047839, -0.0008725, 0.0008725
1: -0.0005930, -0.0000719, -0.0005930, -0.0000719, -0.0002174, 0.0002174
2: -0.0096727, -0.0069113, -0.0096727, -0.0069113, -0.0011521, 0.0011521
3: 0.0018726, 0.0031295, 0.0018726, 0.0031295, -0.0005244, 0.0005244
4: -0.0013442, -0.0008098, -0.0013442, -0.0008098, -0.0002230, 0.0002230
5: -0.0132063, -0.0097331, -0.0132063, -0.0097331, -0.0014490, 0.0014490
6: 0.0040112, 0.0048927, 0.0040112, 0.0048927, -0.0003678, 0.0003678
7: 0.0072406, 0.0095214, 0.0072406, 0.0095214, -0.0009516, 0.0009516
8: 0.0042436, 0.0054430, 0.0042436, 0.0054430, -0.0005004, 0.0005004
9: -0.0081753, -0.0067845, -0.0081753, -0.0067845, -0.0005803, 0.0005803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005824
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0005989
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047839, 1.0025595, 1.0046896, -0.0008562, 0.0010687
1: -0.0005930, -0.0000719, -0.0006262, -0.0000954, -0.0002133, 0.0002663
2: -0.0096727, -0.0069113, -0.0095483, -0.0067356, -0.0014112, 0.0011306
3: 0.0018726, 0.0031295, 0.0017926, 0.0030728, -0.0005146, 0.0006423
4: -0.0013442, -0.0008098, -0.0013202, -0.0007758, -0.0002731, 0.0002188
5: -0.0132063, -0.0097331, -0.0130498, -0.0095121, -0.0017750, 0.0014220
6: 0.0040112, 0.0048927, 0.0039551, 0.0048530, -0.0003609, 0.0004505
7: 0.0072406, 0.0095214, 0.0070955, 0.0094186, -0.0009338, 0.0011656
8: 0.0042436, 0.0054430, 0.0041673, 0.0053890, -0.0004911, 0.0006130
9: -0.0081753, -0.0067845, -0.0081127, -0.0066960, -0.0007108, 0.0005694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046896, 1.0026926, 1.0047839, -0.0010687, 0.0008562
1: -0.0006262, -0.0000954, -0.0005930, -0.0000719, -0.0002663, 0.0002133
2: -0.0095483, -0.0067356, -0.0096727, -0.0069113, -0.0011306, 0.0014112
3: 0.0017926, 0.0030728, 0.0018726, 0.0031295, -0.0006423, 0.0005146
4: -0.0013202, -0.0007758, -0.0013442, -0.0008098, -0.0002188, 0.0002731
5: -0.0130498, -0.0095121, -0.0132063, -0.0097331, -0.0014220, 0.0017750
6: 0.0039551, 0.0048530, 0.0040112, 0.0048927, -0.0004505, 0.0003609
7: 0.0070955, 0.0094186, 0.0072406, 0.0095214, -0.0011656, 0.0009338
8: 0.0041673, 0.0053890, 0.0042436, 0.0054430, -0.0006130, 0.0004911
9: -0.0081127, -0.0066960, -0.0081753, -0.0067845, -0.0005694, 0.0007108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046896, 1.0025595, 1.0046896, -0.0008847, 0.0008847
1: -0.0006262, -0.0000954, -0.0006262, -0.0000954, -0.0002205, 0.0002205
2: -0.0095483, -0.0067356, -0.0095483, -0.0067356, -0.0011683, 0.0011683
3: 0.0017926, 0.0030728, 0.0017926, 0.0030728, -0.0005318, 0.0005318
4: -0.0013202, -0.0007758, -0.0013202, -0.0007758, -0.0002261, 0.0002261
5: -0.0130498, -0.0095121, -0.0130498, -0.0095121, -0.0014694, 0.0014694
6: 0.0039551, 0.0048530, 0.0039551, 0.0048530, -0.0003730, 0.0003730
7: 0.0070955, 0.0094186, 0.0070955, 0.0094186, -0.0009649, 0.0009649
8: 0.0041673, 0.0053890, 0.0041673, 0.0053890, -0.0005075, 0.0005075
9: -0.0081127, -0.0066960, -0.0081127, -0.0066960, -0.0005884, 0.0005884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005994
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0006132
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.02 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005228
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0005334
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005451
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0005495
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005214
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005341
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005368
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0005824
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0005989
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005994
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0006131
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005824
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0005989
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005994
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0006132

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047671, -0.0012350, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000762, -0.0003077, 0.0002248
2: -0.0093030, -0.0067732, -0.0096504, -0.0069113, -0.0011913, 0.0016308
3: 0.0018097, 0.0029612, 0.0018726, 0.0031193, -0.0007423, 0.0005422
4: -0.0012727, -0.0007830, -0.0013399, -0.0008098, -0.0002306, 0.0003156
5: -0.0127413, -0.0095594, -0.0131783, -0.0097331, -0.0014984, 0.0020512
6: 0.0039671, 0.0047747, 0.0040112, 0.0048856, -0.0005206, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0095030, -0.0013470, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054334, -0.0007084, 0.0005175
9: -0.0079891, -0.0067150, -0.0081641, -0.0067845, -0.0006000, 0.0008214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003326, upper bound: 0.0000195
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047839, -0.0012310, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000719, -0.0003067, 0.0002224
2: -0.0093469, -0.0068626, -0.0096727, -0.0069113, -0.0011785, 0.0016255
3: 0.0018505, 0.0029812, 0.0018726, 0.0031295, -0.0007399, 0.0005364
4: -0.0012812, -0.0008004, -0.0013442, -0.0008098, -0.0002281, 0.0003146
5: -0.0127965, -0.0096719, -0.0132063, -0.0097331, -0.0014822, 0.0020445
6: 0.0039957, 0.0047887, 0.0040112, 0.0048927, -0.0005189, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0095214, -0.0013426, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054430, -0.0007060, 0.0005119
9: -0.0080112, -0.0067600, -0.0081753, -0.0067845, -0.0005935, 0.0008187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005214
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0025595, 1.0046675, -0.0010003, 0.0008993
1: -0.0006191, -0.0001417, -0.0006262, -0.0001009, -0.0002493, 0.0002241
2: -0.0093030, -0.0067732, -0.0095191, -0.0067356, -0.0011875, 0.0013209
3: 0.0018097, 0.0029612, 0.0017926, 0.0030596, -0.0006012, 0.0005405
4: -0.0012727, -0.0007830, -0.0013145, -0.0007758, -0.0002298, 0.0002557
5: -0.0127413, -0.0095594, -0.0130131, -0.0095121, -0.0014936, 0.0016614
6: 0.0039671, 0.0047747, 0.0039551, 0.0048437, -0.0004217, 0.0003791
7: 0.0071265, 0.0092160, 0.0070955, 0.0093945, -0.0010910, 0.0009808
8: 0.0041836, 0.0052825, 0.0041673, 0.0053763, -0.0005737, 0.0005158
9: -0.0079891, -0.0067150, -0.0080980, -0.0066960, -0.0005981, 0.0006653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005175, upper bound: 0.0002463
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003281, upper bound: 0.0002610
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0025595, 1.0046896, -0.0010448, 0.0009160
1: -0.0006022, -0.0001334, -0.0006262, -0.0000954, -0.0002603, 0.0002283
2: -0.0093469, -0.0068626, -0.0095483, -0.0067356, -0.0012096, 0.0013796
3: 0.0018505, 0.0029812, 0.0017926, 0.0030728, -0.0006279, 0.0005506
4: -0.0012812, -0.0008004, -0.0013202, -0.0007758, -0.0002341, 0.0002670
5: -0.0127965, -0.0096719, -0.0130498, -0.0095121, -0.0015214, 0.0017352
6: 0.0039957, 0.0047887, 0.0039551, 0.0048530, -0.0004404, 0.0003861
7: 0.0072004, 0.0092523, 0.0070955, 0.0094186, -0.0011395, 0.0009991
8: 0.0042225, 0.0053015, 0.0041673, 0.0053890, -0.0005992, 0.0005254
9: -0.0080112, -0.0067600, -0.0081127, -0.0066960, -0.0006092, 0.0006948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006151, upper bound: 0.0005234
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006151, upper bound: 0.0005368
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0027900, 1.0046711, -0.0010171, 0.0009115
1: -0.0005930, -0.0000841, -0.0005688, -0.0001001, -0.0002534, 0.0002271
2: -0.0096083, -0.0069113, -0.0095238, -0.0070398, -0.0012036, 0.0013431
3: 0.0018726, 0.0031001, 0.0019311, 0.0030617, -0.0006113, 0.0005478
4: -0.0013318, -0.0008098, -0.0013154, -0.0008347, -0.0002330, 0.0002600
5: -0.0131252, -0.0097331, -0.0130190, -0.0098948, -0.0015138, 0.0016893
6: 0.0040112, 0.0048722, 0.0040522, 0.0048452, -0.0004288, 0.0003842
7: 0.0072406, 0.0094681, 0.0073467, 0.0093983, -0.0011093, 0.0009941
8: 0.0042436, 0.0054150, 0.0042994, 0.0053783, -0.0005834, 0.0005228
9: -0.0081429, -0.0067845, -0.0081003, -0.0068492, -0.0006062, 0.0006765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005860
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005989
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045731, -0.0009738, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001244, -0.0002427, 0.0002795
2: -0.0095653, -0.0068740, -0.0093946, -0.0068626, -0.0014812, 0.0012859
3: 0.0018556, 0.0030806, 0.0018505, 0.0030029, -0.0005853, 0.0006742
4: -0.0013235, -0.0008026, -0.0012904, -0.0008004, -0.0002867, 0.0002489
5: -0.0130712, -0.0096862, -0.0128564, -0.0096719, -0.0018630, 0.0016174
6: 0.0039993, 0.0048585, 0.0039957, 0.0048039, -0.0004105, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092916, -0.0010621, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053222, -0.0005586, 0.0006434
9: -0.0081212, -0.0067657, -0.0080352, -0.0067600, -0.0007460, 0.0006477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045787, -0.0010026, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001231, -0.0002498, 0.0002787
2: -0.0096083, -0.0069113, -0.0094019, -0.0068626, -0.0014768, 0.0013240
3: 0.0018726, 0.0031001, 0.0018505, 0.0030062, -0.0006026, 0.0006722
4: -0.0013318, -0.0008098, -0.0012918, -0.0008004, -0.0002858, 0.0002563
5: -0.0131252, -0.0097331, -0.0128656, -0.0096719, -0.0018575, 0.0016652
6: 0.0040112, 0.0048722, 0.0039957, 0.0048063, -0.0004226, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092976, -0.0010935, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053254, -0.0005751, 0.0006415
9: -0.0081429, -0.0067845, -0.0080389, -0.0067600, -0.0007438, 0.0006668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006187
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0027900, 1.0046738, -0.0011806, 0.0008537
1: -0.0006262, -0.0001053, -0.0005688, -0.0000994, -0.0002942, 0.0002127
2: -0.0094961, -0.0067356, -0.0095274, -0.0070398, -0.0011273, 0.0015590
3: 0.0017926, 0.0030491, 0.0019311, 0.0030634, -0.0007096, 0.0005131
4: -0.0013101, -0.0007758, -0.0013161, -0.0008347, -0.0002182, 0.0003017
5: -0.0129841, -0.0095121, -0.0130236, -0.0098948, -0.0014179, 0.0019608
6: 0.0039551, 0.0048363, 0.0040522, 0.0048464, -0.0004977, 0.0003599
7: 0.0070955, 0.0093754, 0.0073467, 0.0094014, -0.0012876, 0.0009311
8: 0.0041673, 0.0053663, 0.0042994, 0.0053799, -0.0006772, 0.0004897
9: -0.0080864, -0.0066960, -0.0081022, -0.0068492, -0.0005678, 0.0007852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005988
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025077, 1.0046004, 1.0026557, 1.0045613, -0.0009978, 0.0009219
1: -0.0006391, -0.0001177, -0.0006022, -0.0001274, -0.0002486, 0.0002297
2: -0.0094304, -0.0066670, -0.0093789, -0.0068626, -0.0012173, 0.0013176
3: 0.0017614, 0.0030192, 0.0018505, 0.0029957, -0.0005997, 0.0005541
4: -0.0012973, -0.0007625, -0.0012874, -0.0008004, -0.0002356, 0.0002550
5: -0.0129015, -0.0094258, -0.0128367, -0.0096719, -0.0015310, 0.0016572
6: 0.0039332, 0.0048154, 0.0039957, 0.0047989, -0.0004206, 0.0003886
7: 0.0070388, 0.0093212, 0.0072004, 0.0092786, -0.0010882, 0.0010054
8: 0.0041375, 0.0053378, 0.0042225, 0.0053154, -0.0005723, 0.0005287
9: -0.0080533, -0.0066615, -0.0080273, -0.0067600, -0.0006131, 0.0006636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004506, upper bound: 0.0002520
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002995, upper bound: 0.0002682
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026557, 1.0045787, -0.0010316, 0.0009247
1: -0.0006262, -0.0001053, -0.0006022, -0.0001231, -0.0002571, 0.0002304
2: -0.0094961, -0.0067356, -0.0094019, -0.0068626, -0.0012210, 0.0013623
3: 0.0017926, 0.0030491, 0.0018505, 0.0030062, -0.0006200, 0.0005558
4: -0.0013101, -0.0007758, -0.0012918, -0.0008004, -0.0002363, 0.0002637
5: -0.0129841, -0.0095121, -0.0128656, -0.0096719, -0.0015357, 0.0017134
6: 0.0039551, 0.0048363, 0.0039957, 0.0048063, -0.0004349, 0.0003898
7: 0.0070955, 0.0093754, 0.0072004, 0.0092976, -0.0011252, 0.0010085
8: 0.0041673, 0.0053663, 0.0042225, 0.0053254, -0.0005917, 0.0005304
9: -0.0080864, -0.0066960, -0.0080389, -0.0067600, -0.0006150, 0.0006861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0006015
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0006132
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026926, 1.0047722, -0.0008701, 0.0007147
1: -0.0005930, -0.0000841, -0.0005930, -0.0000748, -0.0002168, 0.0001781
2: -0.0096083, -0.0069113, -0.0096574, -0.0069113, -0.0009438, 0.0011490
3: 0.0018726, 0.0031001, 0.0018726, 0.0031225, -0.0005230, 0.0004296
4: -0.0013318, -0.0008098, -0.0013413, -0.0008098, -0.0001827, 0.0002224
5: -0.0131252, -0.0097331, -0.0131870, -0.0097331, -0.0011871, 0.0014451
6: 0.0040112, 0.0048722, 0.0040112, 0.0048878, -0.0003668, 0.0003013
7: 0.0072406, 0.0094681, 0.0072406, 0.0095087, -0.0009490, 0.0007795
8: 0.0042436, 0.0054150, 0.0042436, 0.0054364, -0.0004991, 0.0004099
9: -0.0081429, -0.0067845, -0.0081676, -0.0067845, -0.0004753, 0.0005787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004551, upper bound: 0.0003393
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0003563
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046767, -0.0007992, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0000987, -0.0001992, 0.0002359
2: -0.0095653, -0.0068740, -0.0095311, -0.0067356, -0.0012503, 0.0010554
3: 0.0018556, 0.0030806, 0.0017926, 0.0030650, -0.0004804, 0.0005691
4: -0.0013235, -0.0008026, -0.0013168, -0.0007758, -0.0002420, 0.0002043
5: -0.0130712, -0.0096862, -0.0130282, -0.0095121, -0.0015725, 0.0013274
6: 0.0039993, 0.0048585, 0.0039551, 0.0048475, -0.0003369, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0094044, -0.0008717, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053815, -0.0004584, 0.0005430
9: -0.0081212, -0.0067657, -0.0081040, -0.0066960, -0.0006297, 0.0005316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046881, -0.0008554, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0000958, -0.0002131, 0.0002295
2: -0.0096083, -0.0069113, -0.0095464, -0.0067356, -0.0012161, 0.0011295
3: 0.0018726, 0.0031001, 0.0017926, 0.0030720, -0.0005141, 0.0005535
4: -0.0013318, -0.0008098, -0.0013198, -0.0007758, -0.0002354, 0.0002186
5: -0.0131252, -0.0097331, -0.0130474, -0.0095121, -0.0015296, 0.0014207
6: 0.0040112, 0.0048722, 0.0039551, 0.0048524, -0.0003606, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0094170, -0.0009329, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053882, -0.0004906, 0.0005282
9: -0.0081429, -0.0067845, -0.0081117, -0.0066960, -0.0006125, 0.0005689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006187
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026926, 1.0047839, -0.0010687, 0.0007050
1: -0.0006262, -0.0001053, -0.0005930, -0.0000719, -0.0002663, 0.0001757
2: -0.0094961, -0.0067356, -0.0096727, -0.0069113, -0.0009309, 0.0014112
3: 0.0017926, 0.0030491, 0.0018726, 0.0031295, -0.0006423, 0.0004237
4: -0.0013101, -0.0007758, -0.0013442, -0.0008098, -0.0001802, 0.0002731
5: -0.0129841, -0.0095121, -0.0132063, -0.0097331, -0.0011708, 0.0017750
6: 0.0039551, 0.0048363, 0.0040112, 0.0048927, -0.0004505, 0.0002972
7: 0.0070955, 0.0093754, 0.0072406, 0.0095214, -0.0011656, 0.0007689
8: 0.0041673, 0.0053663, 0.0042436, 0.0054430, -0.0006130, 0.0004043
9: -0.0080864, -0.0066960, -0.0081753, -0.0067845, -0.0004689, 0.0007108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005988
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025077, 1.0046004, 1.0025595, 1.0046588, -0.0008113, 0.0007445
1: -0.0006391, -0.0001177, -0.0006262, -0.0001031, -0.0002022, 0.0001855
2: -0.0094304, -0.0066670, -0.0095076, -0.0067356, -0.0009831, 0.0010713
3: 0.0017614, 0.0030192, 0.0017926, 0.0030543, -0.0004876, 0.0004475
4: -0.0012973, -0.0007625, -0.0013123, -0.0007758, -0.0001903, 0.0002074
5: -0.0129015, -0.0094258, -0.0129986, -0.0095121, -0.0012365, 0.0013475
6: 0.0039332, 0.0048154, 0.0039551, 0.0048400, -0.0003420, 0.0003138
7: 0.0070388, 0.0093212, 0.0070955, 0.0093850, -0.0008849, 0.0008120
8: 0.0041375, 0.0053378, 0.0041673, 0.0053713, -0.0004653, 0.0004270
9: -0.0080533, -0.0066615, -0.0080922, -0.0066960, -0.0004951, 0.0005396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004506, upper bound: 0.0002459
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002995, upper bound: 0.0002607
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0025595, 1.0046849, -0.0008839, 0.0007265
1: -0.0006262, -0.0001053, -0.0006262, -0.0000966, -0.0002202, 0.0001810
2: -0.0094961, -0.0067356, -0.0095422, -0.0067356, -0.0009593, 0.0011671
3: 0.0017926, 0.0030491, 0.0017926, 0.0030701, -0.0005312, 0.0004366
4: -0.0013101, -0.0007758, -0.0013190, -0.0007758, -0.0001857, 0.0002259
5: -0.0129841, -0.0095121, -0.0130422, -0.0095121, -0.0012066, 0.0014679
6: 0.0039551, 0.0048363, 0.0039551, 0.0048511, -0.0003726, 0.0003062
7: 0.0070955, 0.0093754, 0.0070955, 0.0094136, -0.0009640, 0.0007923
8: 0.0041673, 0.0053663, 0.0041673, 0.0053864, -0.0005069, 0.0004167
9: -0.0080864, -0.0066960, -0.0081096, -0.0066960, -0.0004832, 0.0005878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0004451
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005186, upper bound: 0.0005577
time: 0.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.17 seconds
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005214
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005175, upper bound: 0.0002463
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0003281, upper bound: 0.0002610
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0006151, upper bound: 0.0005234
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0006151, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005860
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005989
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006187
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005988
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0004506, upper bound: 0.0002520
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0002995, upper bound: 0.0002682
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0006015
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0006132
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0004551, upper bound: 0.0003393
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0003563
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006187
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005988
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0004506, upper bound: 0.0002459
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0002995, upper bound: 0.0002607
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0004451
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0005186, upper bound: 0.0005577

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005214
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0025077, 1.0046004, -0.0009219, 0.0010290
1: -0.0006022, -0.0001334, -0.0006391, -0.0001177, -0.0002297, 0.0002564
2: -0.0093469, -0.0068626, -0.0094304, -0.0066670, -0.0013588, 0.0012173
3: 0.0018505, 0.0029812, 0.0017614, 0.0030192, -0.0005541, 0.0006185
4: -0.0012812, -0.0008004, -0.0012973, -0.0007625, -0.0002630, 0.0002356
5: -0.0127965, -0.0096719, -0.0129015, -0.0094258, -0.0017090, 0.0015310
6: 0.0039957, 0.0047887, 0.0039332, 0.0048154, -0.0003886, 0.0004338
7: 0.0072004, 0.0092523, 0.0070388, 0.0093212, -0.0010054, 0.0011223
8: 0.0042225, 0.0053015, 0.0041375, 0.0053378, -0.0005287, 0.0005902
9: -0.0080112, -0.0067600, -0.0080533, -0.0066615, -0.0006844, 0.0006131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002914, upper bound: 0.0004485
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002970, upper bound: 0.0002903
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0025595, 1.0046501, -0.0009247, 0.0009160
1: -0.0006022, -0.0001334, -0.0006262, -0.0001053, -0.0002304, 0.0002283
2: -0.0093469, -0.0068626, -0.0094961, -0.0067356, -0.0012096, 0.0012210
3: 0.0018505, 0.0029812, 0.0017926, 0.0030491, -0.0005558, 0.0005506
4: -0.0012812, -0.0008004, -0.0013101, -0.0007758, -0.0002341, 0.0002363
5: -0.0127965, -0.0096719, -0.0129841, -0.0095121, -0.0015214, 0.0015357
6: 0.0039957, 0.0047887, 0.0039551, 0.0048363, -0.0003898, 0.0003861
7: 0.0072004, 0.0092523, 0.0070955, 0.0093754, -0.0010085, 0.0009991
8: 0.0042225, 0.0053015, 0.0041673, 0.0053663, -0.0005304, 0.0005254
9: -0.0080112, -0.0067600, -0.0080864, -0.0066960, -0.0006092, 0.0006150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006165, upper bound: 0.0005271
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006283, upper bound: 0.0005368
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0027900, 1.0046202, -0.0009023, 0.0009115
1: -0.0005930, -0.0000841, -0.0005688, -0.0001127, -0.0002248, 0.0002271
2: -0.0096083, -0.0069113, -0.0094566, -0.0070398, -0.0012036, 0.0011914
3: 0.0018726, 0.0031001, 0.0019311, 0.0030311, -0.0005423, 0.0005478
4: -0.0013318, -0.0008098, -0.0013024, -0.0008347, -0.0002330, 0.0002306
5: -0.0131252, -0.0097331, -0.0129344, -0.0098948, -0.0015138, 0.0014985
6: 0.0040112, 0.0048722, 0.0040522, 0.0048237, -0.0003803, 0.0003842
7: 0.0072406, 0.0094681, 0.0073467, 0.0093428, -0.0009841, 0.0009941
8: 0.0042436, 0.0054150, 0.0042994, 0.0053492, -0.0005175, 0.0005228
9: -0.0081429, -0.0067845, -0.0080665, -0.0068492, -0.0006062, 0.0006001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0005824
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0005989
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000195, upper bound: 0.0003326
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006084
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0027900, 1.0046202, -0.0010605, 0.0008537
1: -0.0006262, -0.0001053, -0.0005688, -0.0001127, -0.0002642, 0.0002127
2: -0.0094961, -0.0067356, -0.0094566, -0.0070398, -0.0011273, 0.0014003
3: 0.0017926, 0.0030491, 0.0019311, 0.0030311, -0.0006374, 0.0005131
4: -0.0013101, -0.0007758, -0.0013024, -0.0008347, -0.0002182, 0.0002710
5: -0.0129841, -0.0095121, -0.0129344, -0.0098948, -0.0014179, 0.0017613
6: 0.0039551, 0.0048363, 0.0040522, 0.0048237, -0.0004470, 0.0003599
7: 0.0070955, 0.0093754, 0.0073467, 0.0093428, -0.0011566, 0.0009311
8: 0.0041673, 0.0053663, 0.0042994, 0.0053492, -0.0006082, 0.0004897
9: -0.0080864, -0.0066960, -0.0080665, -0.0068492, -0.0005678, 0.0007053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0025880, 1.0045038, -0.0008993, 0.0010118
1: -0.0006262, -0.0001053, -0.0006191, -0.0001417, -0.0002241, 0.0002521
2: -0.0094961, -0.0067356, -0.0093030, -0.0067732, -0.0013361, 0.0011875
3: 0.0017926, 0.0030491, 0.0018097, 0.0029612, -0.0005405, 0.0006081
4: -0.0013101, -0.0007758, -0.0012727, -0.0007830, -0.0002586, 0.0002298
5: -0.0129841, -0.0095121, -0.0127413, -0.0095594, -0.0016805, 0.0014936
6: 0.0039551, 0.0048363, 0.0039671, 0.0047747, -0.0003791, 0.0004265
7: 0.0070955, 0.0093754, 0.0071265, 0.0092160, -0.0009808, 0.0011036
8: 0.0041673, 0.0053663, 0.0041836, 0.0052825, -0.0005158, 0.0005804
9: -0.0080864, -0.0066960, -0.0079891, -0.0067150, -0.0006729, 0.0005981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002817, upper bound: 0.0005139
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002887, upper bound: 0.0003210
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026557, 1.0045372, -0.0009160, 0.0009247
1: -0.0006262, -0.0001053, -0.0006022, -0.0001334, -0.0002283, 0.0002304
2: -0.0094961, -0.0067356, -0.0093469, -0.0068626, -0.0012210, 0.0012096
3: 0.0017926, 0.0030491, 0.0018505, 0.0029812, -0.0005506, 0.0005558
4: -0.0013101, -0.0007758, -0.0012812, -0.0008004, -0.0002363, 0.0002341
5: -0.0129841, -0.0095121, -0.0127965, -0.0096719, -0.0015357, 0.0015214
6: 0.0039551, 0.0048363, 0.0039957, 0.0047887, -0.0003861, 0.0003898
7: 0.0070955, 0.0093754, 0.0072004, 0.0092523, -0.0009991, 0.0010085
8: 0.0041673, 0.0053663, 0.0042225, 0.0053015, -0.0005254, 0.0005304
9: -0.0080864, -0.0066960, -0.0080112, -0.0067600, -0.0006150, 0.0006092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005353, upper bound: 0.0005991
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005480, upper bound: 0.0006132
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000117, upper bound: 0.0003281
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026926, 1.0047350, -0.0009210, 0.0007050
1: -0.0006262, -0.0001053, -0.0005930, -0.0000841, -0.0002295, 0.0001757
2: -0.0094961, -0.0067356, -0.0096083, -0.0069113, -0.0009309, 0.0012161
3: 0.0017926, 0.0030491, 0.0018726, 0.0031001, -0.0005535, 0.0004237
4: -0.0013101, -0.0007758, -0.0013318, -0.0008098, -0.0001802, 0.0002354
5: -0.0129841, -0.0095121, -0.0131252, -0.0097331, -0.0011708, 0.0015296
6: 0.0039551, 0.0048363, 0.0040112, 0.0048722, -0.0003882, 0.0002972
7: 0.0070955, 0.0093754, 0.0072406, 0.0094681, -0.0010044, 0.0007689
8: 0.0041673, 0.0053663, 0.0042436, 0.0054150, -0.0005282, 0.0004043
9: -0.0080864, -0.0066960, -0.0081429, -0.0067845, -0.0004689, 0.0006125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.06 seconds
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005214
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0002914, upper bound: 0.0004485
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0002970, upper bound: 0.0002903
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006165, upper bound: 0.0005271
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0006283, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0005824
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0005989
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006084
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0002817, upper bound: 0.0005139
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0002887, upper bound: 0.0003210
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005353, upper bound: 0.0005991
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005480, upper bound: 0.0006132
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000195
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000195
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005215
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0025595, 1.0046501, -0.0010118, 0.0008993
1: -0.0006191, -0.0001417, -0.0006262, -0.0001053, -0.0002521, 0.0002241
2: -0.0093030, -0.0067732, -0.0094961, -0.0067356, -0.0011875, 0.0013361
3: 0.0018097, 0.0029612, 0.0017926, 0.0030491, -0.0006081, 0.0005405
4: -0.0012727, -0.0007830, -0.0013101, -0.0007758, -0.0002298, 0.0002586
5: -0.0127413, -0.0095594, -0.0129841, -0.0095121, -0.0014936, 0.0016805
6: 0.0039671, 0.0047747, 0.0039551, 0.0048363, -0.0004265, 0.0003791
7: 0.0071265, 0.0092160, 0.0070955, 0.0093754, -0.0011036, 0.0009808
8: 0.0041836, 0.0052825, 0.0041673, 0.0053663, -0.0005804, 0.0005158
9: -0.0079891, -0.0067150, -0.0080864, -0.0066960, -0.0005981, 0.0006729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005053, upper bound: 0.0002463
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002447, upper bound: 0.0002107
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0025595, 1.0046501, -0.0009247, 0.0009160
1: -0.0006022, -0.0001334, -0.0006262, -0.0001053, -0.0002304, 0.0002283
2: -0.0093469, -0.0068626, -0.0094961, -0.0067356, -0.0012096, 0.0012210
3: 0.0018505, 0.0029812, 0.0017926, 0.0030491, -0.0005558, 0.0005506
4: -0.0012812, -0.0008004, -0.0013101, -0.0007758, -0.0002341, 0.0002363
5: -0.0127965, -0.0096719, -0.0129841, -0.0095121, -0.0015214, 0.0015357
6: 0.0039957, 0.0047887, 0.0039551, 0.0048363, -0.0003898, 0.0003861
7: 0.0072004, 0.0092523, 0.0070955, 0.0093754, -0.0010085, 0.0009991
8: 0.0042225, 0.0053015, 0.0041673, 0.0053663, -0.0005304, 0.0005254
9: -0.0080112, -0.0067600, -0.0080864, -0.0066960, -0.0006092, 0.0006150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006151, upper bound: 0.0005234
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006283, upper bound: 0.0005368
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0027900, 1.0046202, -0.0009023, 0.0009115
1: -0.0005930, -0.0000841, -0.0005688, -0.0001127, -0.0002248, 0.0002271
2: -0.0096083, -0.0069113, -0.0094566, -0.0070398, -0.0012036, 0.0011914
3: 0.0018726, 0.0031001, 0.0019311, 0.0030311, -0.0005423, 0.0005478
4: -0.0013318, -0.0008098, -0.0013024, -0.0008347, -0.0002330, 0.0002306
5: -0.0131252, -0.0097331, -0.0129344, -0.0098948, -0.0015138, 0.0014985
6: 0.0040112, 0.0048722, 0.0040522, 0.0048237, -0.0003803, 0.0003842
7: 0.0072406, 0.0094681, 0.0073467, 0.0093428, -0.0009841, 0.0009941
8: 0.0042436, 0.0054150, 0.0042994, 0.0053492, -0.0005175, 0.0005228
9: -0.0081429, -0.0067845, -0.0080665, -0.0068492, -0.0006062, 0.0006001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005860
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005215, upper bound: 0.0005989
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005129, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005129, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005129, upper bound: 0.0006068
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005132, upper bound: 0.0006084
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0027900, 1.0046202, -0.0010605, 0.0008537
1: -0.0006262, -0.0001053, -0.0005688, -0.0001127, -0.0002642, 0.0002127
2: -0.0094961, -0.0067356, -0.0094566, -0.0070398, -0.0011273, 0.0014003
3: 0.0017926, 0.0030491, 0.0019311, 0.0030311, -0.0006374, 0.0005131
4: -0.0013101, -0.0007758, -0.0013024, -0.0008347, -0.0002182, 0.0002710
5: -0.0129841, -0.0095121, -0.0129344, -0.0098948, -0.0014179, 0.0017613
6: 0.0039551, 0.0048363, 0.0040522, 0.0048237, -0.0004470, 0.0003599
7: 0.0070955, 0.0093754, 0.0073467, 0.0093428, -0.0011566, 0.0009311
8: 0.0041673, 0.0053663, 0.0042994, 0.0053492, -0.0006082, 0.0004897
9: -0.0080864, -0.0066960, -0.0080665, -0.0068492, -0.0005678, 0.0007053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025077, 1.0046004, 1.0026557, 1.0045372, -0.0010290, 0.0009219
1: -0.0006391, -0.0001177, -0.0006022, -0.0001334, -0.0002564, 0.0002297
2: -0.0094304, -0.0066670, -0.0093469, -0.0068626, -0.0012173, 0.0013588
3: 0.0017614, 0.0030192, 0.0018505, 0.0029812, -0.0006185, 0.0005541
4: -0.0012973, -0.0007625, -0.0012812, -0.0008004, -0.0002356, 0.0002630
5: -0.0129015, -0.0094258, -0.0127965, -0.0096719, -0.0015310, 0.0017090
6: 0.0039332, 0.0048154, 0.0039957, 0.0047887, -0.0004338, 0.0003886
7: 0.0070388, 0.0093212, 0.0072004, 0.0092523, -0.0011223, 0.0010054
8: 0.0041375, 0.0053378, 0.0042225, 0.0053015, -0.0005902, 0.0005287
9: -0.0080533, -0.0066615, -0.0080112, -0.0067600, -0.0006131, 0.0006844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004499, upper bound: 0.0002520
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002265, upper bound: 0.0002682
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026557, 1.0045372, -0.0009160, 0.0009247
1: -0.0006262, -0.0001053, -0.0006022, -0.0001334, -0.0002283, 0.0002304
2: -0.0094961, -0.0067356, -0.0093469, -0.0068626, -0.0012210, 0.0012096
3: 0.0017926, 0.0030491, 0.0018505, 0.0029812, -0.0005506, 0.0005558
4: -0.0013101, -0.0007758, -0.0012812, -0.0008004, -0.0002363, 0.0002341
5: -0.0129841, -0.0095121, -0.0127965, -0.0096719, -0.0015357, 0.0015214
6: 0.0039551, 0.0048363, 0.0039957, 0.0047887, -0.0003861, 0.0003898
7: 0.0070955, 0.0093754, 0.0072004, 0.0092523, -0.0009991, 0.0010085
8: 0.0041673, 0.0053663, 0.0042225, 0.0053015, -0.0005254, 0.0005304
9: -0.0080864, -0.0066960, -0.0080112, -0.0067600, -0.0006150, 0.0006092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0006015
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005480, upper bound: 0.0006131
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005131, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005131, upper bound: 0.0006068
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005131, upper bound: 0.0006068
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005133, upper bound: 0.0006084
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026926, 1.0047350, -0.0009210, 0.0007050
1: -0.0006262, -0.0001053, -0.0005930, -0.0000841, -0.0002295, 0.0001757
2: -0.0094961, -0.0067356, -0.0096083, -0.0069113, -0.0009309, 0.0012161
3: 0.0017926, 0.0030491, 0.0018726, 0.0031001, -0.0005535, 0.0004237
4: -0.0013101, -0.0007758, -0.0013318, -0.0008098, -0.0001802, 0.0002354
5: -0.0129841, -0.0095121, -0.0131252, -0.0097331, -0.0011708, 0.0015296
6: 0.0039551, 0.0048363, 0.0040112, 0.0048722, -0.0003882, 0.0002972
7: 0.0070955, 0.0093754, 0.0072406, 0.0094681, -0.0010044, 0.0007689
8: 0.0041673, 0.0053663, 0.0042436, 0.0054150, -0.0005282, 0.0004043
9: -0.0080864, -0.0066960, -0.0081429, -0.0067845, -0.0004689, 0.0006125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988
time: 0.81 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.09 seconds
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005215
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005053, upper bound: 0.0002463
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0002447, upper bound: 0.0002107
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006151, upper bound: 0.0005234
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0006283, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005860
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005215, upper bound: 0.0005989
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005129, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005129, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005129, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005132, upper bound: 0.0006084
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0004499, upper bound: 0.0002520
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0002265, upper bound: 0.0002682
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0006015
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005480, upper bound: 0.0006131
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005131, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005131, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005131, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005133, upper bound: 0.0006084
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005285, upper bound: 0.0005860
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.09
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005214
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0025077, 1.0046004, -0.0009219, 0.0010290
1: -0.0006022, -0.0001334, -0.0006391, -0.0001177, -0.0002297, 0.0002564
2: -0.0093469, -0.0068626, -0.0094304, -0.0066670, -0.0013588, 0.0012173
3: 0.0018505, 0.0029812, 0.0017614, 0.0030192, -0.0005541, 0.0006185
4: -0.0012812, -0.0008004, -0.0012973, -0.0007625, -0.0002630, 0.0002356
5: -0.0127965, -0.0096719, -0.0129015, -0.0094258, -0.0017090, 0.0015310
6: 0.0039957, 0.0047887, 0.0039332, 0.0048154, -0.0003886, 0.0004338
7: 0.0072004, 0.0092523, 0.0070388, 0.0093212, -0.0010054, 0.0011223
8: 0.0042225, 0.0053015, 0.0041375, 0.0053378, -0.0005287, 0.0005902
9: -0.0080112, -0.0067600, -0.0080533, -0.0066615, -0.0006844, 0.0006131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002914, upper bound: 0.0004485
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002970, upper bound: 0.0002903
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0025595, 1.0046501, -0.0009247, 0.0009160
1: -0.0006022, -0.0001334, -0.0006262, -0.0001053, -0.0002304, 0.0002283
2: -0.0093469, -0.0068626, -0.0094961, -0.0067356, -0.0012096, 0.0012210
3: 0.0018505, 0.0029812, 0.0017926, 0.0030491, -0.0005558, 0.0005506
4: -0.0012812, -0.0008004, -0.0013101, -0.0007758, -0.0002341, 0.0002363
5: -0.0127965, -0.0096719, -0.0129841, -0.0095121, -0.0015214, 0.0015357
6: 0.0039957, 0.0047887, 0.0039551, 0.0048363, -0.0003898, 0.0003861
7: 0.0072004, 0.0092523, 0.0070955, 0.0093754, -0.0010085, 0.0009991
8: 0.0042225, 0.0053015, 0.0041673, 0.0053663, -0.0005304, 0.0005254
9: -0.0080112, -0.0067600, -0.0080864, -0.0066960, -0.0006092, 0.0006150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006165, upper bound: 0.0005271
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006283, upper bound: 0.0005368
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0027900, 1.0046202, -0.0009023, 0.0009115
1: -0.0005930, -0.0000841, -0.0005688, -0.0001127, -0.0002248, 0.0002271
2: -0.0096083, -0.0069113, -0.0094566, -0.0070398, -0.0012036, 0.0011914
3: 0.0018726, 0.0031001, 0.0019311, 0.0030311, -0.0005423, 0.0005478
4: -0.0013318, -0.0008098, -0.0013024, -0.0008347, -0.0002330, 0.0002306
5: -0.0131252, -0.0097331, -0.0129344, -0.0098948, -0.0015138, 0.0014985
6: 0.0040112, 0.0048722, 0.0040522, 0.0048237, -0.0003803, 0.0003842
7: 0.0072406, 0.0094681, 0.0073467, 0.0093428, -0.0009841, 0.0009941
8: 0.0042436, 0.0054150, 0.0042994, 0.0053492, -0.0005175, 0.0005228
9: -0.0081429, -0.0067845, -0.0080665, -0.0068492, -0.0006062, 0.0006001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0005824
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0005989
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000195, upper bound: 0.0003326
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025880, 1.0045038, -0.0008823, 0.0011490
1: -0.0006001, -0.0000922, -0.0006191, -0.0001417, -0.0002198, 0.0002863
2: -0.0095653, -0.0068740, -0.0093030, -0.0067732, -0.0015173, 0.0011651
3: 0.0018556, 0.0030806, 0.0018097, 0.0029612, -0.0005303, 0.0006906
4: -0.0013235, -0.0008026, -0.0012727, -0.0007830, -0.0002937, 0.0002255
5: -0.0130712, -0.0096862, -0.0127413, -0.0095594, -0.0019083, 0.0014653
6: 0.0039993, 0.0048585, 0.0039671, 0.0047747, -0.0003719, 0.0004844
7: 0.0072098, 0.0094327, 0.0071265, 0.0092160, -0.0009623, 0.0012532
8: 0.0042274, 0.0053964, 0.0041836, 0.0052825, -0.0005060, 0.0006590
9: -0.0081212, -0.0067657, -0.0079891, -0.0067150, -0.0007642, 0.0005868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000015, upper bound: 0.0003214
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0026557, 1.0045372, -0.0009633, 0.0011217
1: -0.0006001, -0.0000922, -0.0006022, -0.0001334, -0.0002400, 0.0002795
2: -0.0095653, -0.0068740, -0.0093469, -0.0068626, -0.0014812, 0.0012720
3: 0.0018556, 0.0030806, 0.0018505, 0.0029812, -0.0005790, 0.0006742
4: -0.0013235, -0.0008026, -0.0012812, -0.0008004, -0.0002867, 0.0002462
5: -0.0130712, -0.0096862, -0.0127965, -0.0096719, -0.0018630, 0.0015999
6: 0.0039993, 0.0048585, 0.0039957, 0.0047887, -0.0004061, 0.0004729
7: 0.0072098, 0.0094327, 0.0072004, 0.0092523, -0.0010506, 0.0012234
8: 0.0042274, 0.0053964, 0.0042225, 0.0053015, -0.0005525, 0.0006434
9: -0.0081212, -0.0067657, -0.0080112, -0.0067600, -0.0007460, 0.0006407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025880, 1.0045038, -0.0009022, 0.0012321
1: -0.0005930, -0.0000841, -0.0006191, -0.0001417, -0.0002248, 0.0003070
2: -0.0096083, -0.0069113, -0.0093030, -0.0067732, -0.0016270, 0.0011913
3: 0.0018726, 0.0031001, 0.0018097, 0.0029612, -0.0005422, 0.0007406
4: -0.0013318, -0.0008098, -0.0012727, -0.0007830, -0.0003149, 0.0002306
5: -0.0131252, -0.0097331, -0.0127413, -0.0095594, -0.0020464, 0.0014984
6: 0.0040112, 0.0048722, 0.0039671, 0.0047747, -0.0003803, 0.0005194
7: 0.0072406, 0.0094681, 0.0071265, 0.0092160, -0.0009840, 0.0013438
8: 0.0042436, 0.0054150, 0.0041836, 0.0052825, -0.0005175, 0.0007067
9: -0.0081429, -0.0067845, -0.0079891, -0.0067150, -0.0008195, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000195, upper bound: 0.0003326
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0026557, 1.0045372, -0.0008924, 0.0011184
1: -0.0005930, -0.0000841, -0.0006022, -0.0001334, -0.0002224, 0.0002787
2: -0.0096083, -0.0069113, -0.0093469, -0.0068626, -0.0014768, 0.0011785
3: 0.0018726, 0.0031001, 0.0018505, 0.0029812, -0.0005364, 0.0006722
4: -0.0013318, -0.0008098, -0.0012812, -0.0008004, -0.0002858, 0.0002281
5: -0.0131252, -0.0097331, -0.0127965, -0.0096719, -0.0018575, 0.0014822
6: 0.0040112, 0.0048722, 0.0039957, 0.0047887, -0.0003762, 0.0004714
7: 0.0072406, 0.0094681, 0.0072004, 0.0092523, -0.0009733, 0.0012198
8: 0.0042436, 0.0054150, 0.0042225, 0.0053015, -0.0005119, 0.0006415
9: -0.0081429, -0.0067845, -0.0080112, -0.0067600, -0.0007438, 0.0005935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0027900, 1.0046202, -0.0010605, 0.0008537
1: -0.0006262, -0.0001053, -0.0005688, -0.0001127, -0.0002642, 0.0002127
2: -0.0094961, -0.0067356, -0.0094566, -0.0070398, -0.0011273, 0.0014003
3: 0.0017926, 0.0030491, 0.0019311, 0.0030311, -0.0006374, 0.0005131
4: -0.0013101, -0.0007758, -0.0013024, -0.0008347, -0.0002182, 0.0002710
5: -0.0129841, -0.0095121, -0.0129344, -0.0098948, -0.0014179, 0.0017613
6: 0.0039551, 0.0048363, 0.0040522, 0.0048237, -0.0004470, 0.0003599
7: 0.0070955, 0.0093754, 0.0073467, 0.0093428, -0.0011566, 0.0009311
8: 0.0041673, 0.0053663, 0.0042994, 0.0053492, -0.0006082, 0.0004897
9: -0.0080864, -0.0066960, -0.0080665, -0.0068492, -0.0005678, 0.0007053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0025880, 1.0045038, -0.0008993, 0.0010118
1: -0.0006262, -0.0001053, -0.0006191, -0.0001417, -0.0002241, 0.0002521
2: -0.0094961, -0.0067356, -0.0093030, -0.0067732, -0.0013361, 0.0011875
3: 0.0017926, 0.0030491, 0.0018097, 0.0029612, -0.0005405, 0.0006081
4: -0.0013101, -0.0007758, -0.0012727, -0.0007830, -0.0002586, 0.0002298
5: -0.0129841, -0.0095121, -0.0127413, -0.0095594, -0.0016805, 0.0014936
6: 0.0039551, 0.0048363, 0.0039671, 0.0047747, -0.0003791, 0.0004265
7: 0.0070955, 0.0093754, 0.0071265, 0.0092160, -0.0009808, 0.0011036
8: 0.0041673, 0.0053663, 0.0041836, 0.0052825, -0.0005158, 0.0005804
9: -0.0080864, -0.0066960, -0.0079891, -0.0067150, -0.0006729, 0.0005981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002817, upper bound: 0.0005139
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002887, upper bound: 0.0003210
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026557, 1.0045372, -0.0009160, 0.0009247
1: -0.0006262, -0.0001053, -0.0006022, -0.0001334, -0.0002283, 0.0002304
2: -0.0094961, -0.0067356, -0.0093469, -0.0068626, -0.0012210, 0.0012096
3: 0.0017926, 0.0030491, 0.0018505, 0.0029812, -0.0005506, 0.0005558
4: -0.0013101, -0.0007758, -0.0012812, -0.0008004, -0.0002363, 0.0002341
5: -0.0129841, -0.0095121, -0.0127965, -0.0096719, -0.0015357, 0.0015214
6: 0.0039551, 0.0048363, 0.0039957, 0.0047887, -0.0003861, 0.0003898
7: 0.0070955, 0.0093754, 0.0072004, 0.0092523, -0.0009991, 0.0010085
8: 0.0041673, 0.0053663, 0.0042225, 0.0053015, -0.0005254, 0.0005304
9: -0.0080864, -0.0066960, -0.0080112, -0.0067600, -0.0006150, 0.0006092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005353, upper bound: 0.0005991
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005480, upper bound: 0.0006132
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000117, upper bound: 0.0003281
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025077, 1.0046004, -0.0007084, 0.0009663
1: -0.0006001, -0.0000922, -0.0006391, -0.0001177, -0.0001765, 0.0002408
2: -0.0095653, -0.0068740, -0.0094304, -0.0066670, -0.0012760, 0.0009355
3: 0.0018556, 0.0030806, 0.0017614, 0.0030192, -0.0004258, 0.0005808
4: -0.0013235, -0.0008026, -0.0012973, -0.0007625, -0.0002470, 0.0001811
5: -0.0130712, -0.0096862, -0.0129015, -0.0094258, -0.0016048, 0.0011766
6: 0.0039993, 0.0048585, 0.0039332, 0.0048154, -0.0002986, 0.0004073
7: 0.0072098, 0.0094327, 0.0070388, 0.0093212, -0.0007726, 0.0010539
8: 0.0042274, 0.0053964, 0.0041375, 0.0053378, -0.0004063, 0.0005542
9: -0.0081212, -0.0067657, -0.0080533, -0.0066615, -0.0006426, 0.0004711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: 0.0000067, upper bound: 0.0003080
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0026644, 1.0047026, 1.0025595, 1.0046501, -0.0007973, 0.0009468
1: -0.0006001, -0.0000922, -0.0006262, -0.0001053, -0.0001987, 0.0002359
2: -0.0095653, -0.0068740, -0.0094961, -0.0067356, -0.0012503, 0.0010529
3: 0.0018556, 0.0030806, 0.0017926, 0.0030491, -0.0004792, 0.0005691
4: -0.0013235, -0.0008026, -0.0013101, -0.0007758, -0.0002420, 0.0002038
5: -0.0130712, -0.0096862, -0.0129841, -0.0095121, -0.0015725, 0.0013242
6: 0.0039993, 0.0048585, 0.0039551, 0.0048363, -0.0003361, 0.0003991
7: 0.0072098, 0.0094327, 0.0070955, 0.0093754, -0.0008696, 0.0010326
8: 0.0042274, 0.0053964, 0.0041673, 0.0053663, -0.0004573, 0.0005430
9: -0.0081212, -0.0067657, -0.0080864, -0.0066960, -0.0006297, 0.0005303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025077, 1.0046004, -0.0007473, 0.0010800
1: -0.0005930, -0.0000841, -0.0006391, -0.0001177, -0.0001862, 0.0002691
2: -0.0096083, -0.0069113, -0.0094304, -0.0066670, -0.0014261, 0.0009869
3: 0.0018726, 0.0031001, 0.0017614, 0.0030192, -0.0004492, 0.0006491
4: -0.0013318, -0.0008098, -0.0012973, -0.0007625, -0.0002760, 0.0001910
5: -0.0131252, -0.0097331, -0.0129015, -0.0094258, -0.0017937, 0.0012412
6: 0.0040112, 0.0048722, 0.0039332, 0.0048154, -0.0003150, 0.0004553
7: 0.0072406, 0.0094681, 0.0070388, 0.0093212, -0.0008151, 0.0011779
8: 0.0042436, 0.0054150, 0.0041375, 0.0053378, -0.0004286, 0.0006194
9: -0.0081429, -0.0067845, -0.0080533, -0.0066615, -0.0007183, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000117, upper bound: 0.0003281
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0026926, 1.0047350, 1.0025595, 1.0046501, -0.0007050, 0.0009210
1: -0.0005930, -0.0000841, -0.0006262, -0.0001053, -0.0001757, 0.0002295
2: -0.0096083, -0.0069113, -0.0094961, -0.0067356, -0.0012161, 0.0009309
3: 0.0018726, 0.0031001, 0.0017926, 0.0030491, -0.0004237, 0.0005535
4: -0.0013318, -0.0008098, -0.0013101, -0.0007758, -0.0002354, 0.0001802
5: -0.0131252, -0.0097331, -0.0129841, -0.0095121, -0.0015296, 0.0011708
6: 0.0040112, 0.0048722, 0.0039551, 0.0048363, -0.0002972, 0.0003882
7: 0.0072406, 0.0094681, 0.0070955, 0.0093754, -0.0007689, 0.0010044
8: 0.0042436, 0.0054150, 0.0041673, 0.0053663, -0.0004043, 0.0005282
9: -0.0081429, -0.0067845, -0.0080864, -0.0066960, -0.0006125, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0025595, 1.0046501, 1.0026926, 1.0047350, -0.0009210, 0.0007050
1: -0.0006262, -0.0001053, -0.0005930, -0.0000841, -0.0002295, 0.0001757
2: -0.0094961, -0.0067356, -0.0096083, -0.0069113, -0.0009309, 0.0012161
3: 0.0017926, 0.0030491, 0.0018726, 0.0031001, -0.0005535, 0.0004237
4: -0.0013101, -0.0007758, -0.0013318, -0.0008098, -0.0001802, 0.0002354
5: -0.0129841, -0.0095121, -0.0131252, -0.0097331, -0.0011708, 0.0015296
6: 0.0039551, 0.0048363, 0.0040112, 0.0048722, -0.0003882, 0.0002972
7: 0.0070955, 0.0093754, 0.0072406, 0.0094681, -0.0010044, 0.0007689
8: 0.0041673, 0.0053663, 0.0042436, 0.0054150, -0.0005282, 0.0004043
9: -0.0080864, -0.0066960, -0.0081429, -0.0067845, -0.0004689, 0.0006125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 242

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 129

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988
time: 0.90 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.51 seconds
IS_A1_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
IS_A1_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005129
IS_A1_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005132
IS_A1_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
IS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006083, upper bound: 0.0005113
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006187, upper bound: 0.0005214
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0002914, upper bound: 0.0004485
IS_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0002970, upper bound: 0.0002903
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006165, upper bound: 0.0005271
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0006283, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0005824
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0005989
IS_A2_B1_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
IS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
IS_A2_B1_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006072
IS_A2_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0006083
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005067, upper bound: 0.0006068
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005214, upper bound: 0.0006187
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005988
IS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0002817, upper bound: 0.0005139
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0002887, upper bound: 0.0003210
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005353, upper bound: 0.0005991
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005480, upper bound: 0.0006132
IS_A2_B2_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006072
IS_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0006083
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0006068
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005218, upper bound: 0.0006187
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005824
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.51
Output dim: 0, lower bound: -0.0005372, upper bound: 0.0005988

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026926, 1.0047350, -0.0011184, 0.0008924
1: -0.0006022, -0.0001334, -0.0005930, -0.0000841, -0.0002787, 0.0002224
2: -0.0093469, -0.0068626, -0.0096083, -0.0069113, -0.0011785, 0.0014768
3: 0.0018505, 0.0029812, 0.0018726, 0.0031001, -0.0006722, 0.0005364
4: -0.0012812, -0.0008004, -0.0013318, -0.0008098, -0.0002281, 0.0002858
5: -0.0127965, -0.0096719, -0.0131252, -0.0097331, -0.0014822, 0.0018575
6: 0.0039957, 0.0047887, 0.0040112, 0.0048722, -0.0004714, 0.0003762
7: 0.0072004, 0.0092523, 0.0072406, 0.0094681, -0.0012198, 0.0009733
8: 0.0042225, 0.0053015, 0.0042436, 0.0054150, -0.0006415, 0.0005119
9: -0.0080112, -0.0067600, -0.0081429, -0.0067845, -0.0005935, 0.0007438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026644, 1.0047026, -0.0011490, 0.0008823
1: -0.0006191, -0.0001417, -0.0006001, -0.0000922, -0.0002863, 0.0002198
2: -0.0093030, -0.0067732, -0.0095653, -0.0068740, -0.0011651, 0.0015173
3: 0.0018097, 0.0029612, 0.0018556, 0.0030806, -0.0006906, 0.0005303
4: -0.0012727, -0.0007830, -0.0013235, -0.0008026, -0.0002255, 0.0002937
5: -0.0127413, -0.0095594, -0.0130712, -0.0096862, -0.0014653, 0.0019083
6: 0.0039671, 0.0047747, 0.0039993, 0.0048585, -0.0004844, 0.0003719
7: 0.0071265, 0.0092160, 0.0072098, 0.0094327, -0.0012532, 0.0009623
8: 0.0041836, 0.0052825, 0.0042274, 0.0053964, -0.0006590, 0.0005060
9: -0.0079891, -0.0067150, -0.0081212, -0.0067657, -0.0005868, 0.0007642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000015
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0026557, 1.0045372, 1.0026644, 1.0047026, -0.0011217, 0.0009633
1: -0.0006022, -0.0001334, -0.0006001, -0.0000922, -0.0002795, 0.0002400
2: -0.0093469, -0.0068626, -0.0095653, -0.0068740, -0.0012720, 0.0014812
3: 0.0018505, 0.0029812, 0.0018556, 0.0030806, -0.0006742, 0.0005790
4: -0.0012812, -0.0008004, -0.0013235, -0.0008026, -0.0002462, 0.0002867
5: -0.0127965, -0.0096719, -0.0130712, -0.0096862, -0.0015999, 0.0018630
6: 0.0039957, 0.0047887, 0.0039993, 0.0048585, -0.0004729, 0.0004061
7: 0.0072004, 0.0092523, 0.0072098, 0.0094327, -0.0012234, 0.0010506
8: 0.0042225, 0.0053015, 0.0042274, 0.0053964, -0.0006434, 0.0005525
9: -0.0080112, -0.0067600, -0.0081212, -0.0067657, -0.0006407, 0.0007460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 129

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0005067
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006068, upper bound: 0.0005067
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0025880, 1.0045038, 1.0026926, 1.0047350, -0.0012321, 0.0009022
1: -0.0006191, -0.0001417, -0.0005930, -0.0000841, -0.0003070, 0.0002248
2: -0.0093030, -0.0067732, -0.0096083, -0.0069113, -0.0011913, 0.0016270
3: 0.0018097, 0.0029612, 0.0018726, 0.0031001, -0.0007406, 0.0005422
4: -0.0012727, -0.0007830, -0.0013318, -0.0008098, -0.0002306, 0.0003149
5: -0.0127413, -0.0095594, -0.0131252, -0.0097331, -0.0014984, 0.0020464
6: 0.0039671, 0.0047747, 0.0040112, 0.0048722, -0.0005194, 0.0003803
7: 0.0071265, 0.0092160, 0.0072406, 0.0094681, -0.0013438, 0.0009840
8: 0.0041836, 0.0052825, 0.0042436, 0.0054150, -0.0007067, 0.0005175
9: -0.0079891, -0.0067150, -0.0081429, -0.0067845, -0.0006000, 0.0008195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 145
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 242

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 145

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003214, upper bound: 0.0000195
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.90 + 597.86 = 600.76 seconds
