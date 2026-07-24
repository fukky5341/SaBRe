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
Threshold: 0.00037578


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042012, -0.0041392, -0.0042012, -0.0041392, -0.0000620, 0.0000620)
1: (-0.0099914, -0.0092434, -0.0099914, -0.0092434, -0.0007479, 0.0007479)
2: (0.9644735, 0.9653709, 0.9644735, 0.9653709, -0.0008974, 0.0008974)
3: (-0.0157321, -0.0091120, -0.0157321, -0.0091120, -0.0051015, 0.0051015)
4: (-0.0000000, 0.0005035, -0.0000000, 0.0005035, -0.0005035, 0.0005035)
5: (0.0172703, 0.0180004, 0.0172703, 0.0180004, -0.0007301, 0.0007301)
6: (0.0026698, 0.0035065, 0.0026698, 0.0035065, -0.0008367, 0.0008367)
7: (-0.0054168, -0.0033513, -0.0054168, -0.0033513, -0.0020655, 0.0020655)
8: (0.0124317, 0.0137928, 0.0124317, 0.0137928, -0.0013611, 0.0013611)
9: (0.0200842, 0.0225324, 0.0200842, 0.0225324, -0.0022769, 0.0022769)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.64 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0006095, upper bound: 0.0006096

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0005968
time: 0.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006056
time: 0.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0005968
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006056

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041321, -0.0042012, -0.0041392, -0.0000625, 0.0000691
1: -0.0100124, -0.0093022, -0.0099912, -0.0092525, -0.0007599, 0.0006891
2: 0.9644482, 0.9653004, 0.9644736, 0.9653600, -0.0009119, 0.0008268
3: -0.0159183, -0.0096320, -0.0157311, -0.0091925, -0.0052861, 0.0046736
4: 0.0000396, 0.0005176, 0.0000061, 0.0005034, -0.0004639, 0.0005115
5: 0.0173103, 0.0180420, 0.0172765, 0.0180002, -0.0006899, 0.0007655
6: 0.0025902, 0.0034871, 0.0026703, 0.0035035, -0.0009133, 0.0008168
7: -0.0052820, -0.0032599, -0.0053959, -0.0033518, -0.0019302, 0.0021361
8: 0.0125386, 0.0138311, 0.0124483, 0.0137926, -0.0012540, 0.0013828
9: 0.0202765, 0.0226012, 0.0201140, 0.0225320, -0.0020968, 0.0023278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0005968
time: 0.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0005968
time: 0.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041393, -0.0042012, -0.0041392, -0.0000620, 0.0000619
1: -0.0099912, -0.0092495, -0.0099914, -0.0092434, -0.0007478, 0.0007418
2: 0.9644735, 0.9653636, 0.9644735, 0.9653709, -0.0008973, 0.0008901
3: -0.0157307, -0.0091661, -0.0157321, -0.0091120, -0.0051000, 0.0049145
4: 0.0000041, 0.0005034, -0.0000000, 0.0005035, -0.0004994, 0.0005034
5: 0.0172745, 0.0180001, 0.0172703, 0.0180004, -0.0007259, 0.0007297
6: 0.0026705, 0.0035045, 0.0026698, 0.0035065, -0.0008360, 0.0008346
7: -0.0054028, -0.0033520, -0.0054168, -0.0033513, -0.0020515, 0.0020648
8: 0.0124428, 0.0137925, 0.0124317, 0.0137928, -0.0013500, 0.0013608
9: 0.0201043, 0.0225318, 0.0200842, 0.0225324, -0.0022433, 0.0022764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0006052
time: 0.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0006057
time: 0.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0005968
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0005968
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0006052
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005968, upper bound: 0.0006057

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041321, -0.0042018, -0.0041321, -0.0000697, 0.0000697
1: -0.0100124, -0.0093022, -0.0100124, -0.0093022, -0.0007102, 0.0007102
2: 0.9644482, 0.9653004, 0.9644482, 0.9653004, -0.0008522, 0.0008522
3: -0.0159183, -0.0096320, -0.0159183, -0.0096320, -0.0048336, 0.0048336
4: 0.0000396, 0.0005176, 0.0000396, 0.0005176, -0.0004781, 0.0004781
5: 0.0173103, 0.0180420, 0.0173103, 0.0180420, -0.0007317, 0.0007317
6: 0.0025902, 0.0034871, 0.0025902, 0.0034871, -0.0008969, 0.0008969
7: -0.0052820, -0.0032599, -0.0052820, -0.0032599, -0.0020222, 0.0020222
8: 0.0125386, 0.0138311, 0.0125386, 0.0138311, -0.0012925, 0.0012925
9: 0.0202765, 0.0226012, 0.0202765, 0.0226012, -0.0021639, 0.0021639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005790, upper bound: 0.0005793
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041321, -0.0042012, -0.0041393, -0.0000625, 0.0000691
1: -0.0100124, -0.0093022, -0.0099912, -0.0092495, -0.0007628, 0.0006890
2: 0.9644482, 0.9653004, 0.9644735, 0.9653636, -0.0009154, 0.0008268
3: -0.0159183, -0.0096320, -0.0157307, -0.0091662, -0.0053202, 0.0046732
4: 0.0000396, 0.0005176, 0.0000041, 0.0005034, -0.0004638, 0.0005135
5: 0.0173103, 0.0180420, 0.0172745, 0.0180001, -0.0006898, 0.0007675
6: 0.0025902, 0.0034871, 0.0026705, 0.0035045, -0.0009143, 0.0008166
7: -0.0052820, -0.0032599, -0.0054028, -0.0033520, -0.0019300, 0.0021429
8: 0.0125386, 0.0138311, 0.0124429, 0.0137925, -0.0012539, 0.0013882
9: 0.0202765, 0.0226012, 0.0201043, 0.0225318, -0.0020966, 0.0023380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005793, upper bound: 0.0005797
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005747
time: 0.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041393, -0.0042018, -0.0041321, -0.0000691, 0.0000625
1: -0.0099912, -0.0092495, -0.0100124, -0.0093022, -0.0006890, 0.0007628
2: 0.9644735, 0.9653636, 0.9644482, 0.9653004, -0.0008268, 0.0009154
3: -0.0157307, -0.0091661, -0.0159183, -0.0096320, -0.0046732, 0.0052185
4: 0.0000041, 0.0005034, 0.0000396, 0.0005176, -0.0005135, 0.0004638
5: 0.0172745, 0.0180001, 0.0173103, 0.0180420, -0.0007675, 0.0006898
6: 0.0026705, 0.0035045, 0.0025902, 0.0034871, -0.0008166, 0.0009143
7: -0.0054028, -0.0033520, -0.0052820, -0.0032599, -0.0021429, 0.0019300
8: 0.0124428, 0.0137925, 0.0125386, 0.0138311, -0.0013883, 0.0012539
9: 0.0201043, 0.0225318, 0.0202765, 0.0226012, -0.0023245, 0.0020966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005797, upper bound: 0.0005973
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005746, upper bound: 0.0005944
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041393, -0.0042012, -0.0041393, -0.0000619, 0.0000619
1: -0.0099912, -0.0092495, -0.0099912, -0.0092495, -0.0007417, 0.0007417
2: 0.9644735, 0.9653636, 0.9644735, 0.9653636, -0.0008900, 0.0008900
3: -0.0157307, -0.0091661, -0.0157307, -0.0091661, -0.0049130, 0.0049130
4: 0.0000041, 0.0005034, 0.0000041, 0.0005034, -0.0004993, 0.0004993
5: 0.0172745, 0.0180001, 0.0172745, 0.0180001, -0.0007256, 0.0007256
6: 0.0026705, 0.0035045, 0.0026705, 0.0035045, -0.0008340, 0.0008340
7: -0.0054028, -0.0033520, -0.0054028, -0.0033520, -0.0020508, 0.0020508
8: 0.0124428, 0.0137925, 0.0124428, 0.0137925, -0.0013497, 0.0013497
9: 0.0201043, 0.0225318, 0.0201043, 0.0225318, -0.0022427, 0.0022427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005797, upper bound: 0.0005973
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005746, upper bound: 0.0006037
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.80 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005790, upper bound: 0.0005793
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005793, upper bound: 0.0005797
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005747
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005797, upper bound: 0.0005973
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005746, upper bound: 0.0005944
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005797, upper bound: 0.0005973
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 2, lower bound: -0.0005746, upper bound: 0.0006037

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041322, -0.0042018, -0.0041321, -0.0000697, 0.0000696
1: -0.0100122, -0.0093051, -0.0100124, -0.0093022, -0.0007100, 0.0007073
2: 0.9644485, 0.9652969, 0.9644482, 0.9653004, -0.0008519, 0.0008488
3: -0.0159163, -0.0096582, -0.0159183, -0.0096320, -0.0048315, 0.0048062
4: 0.0000415, 0.0005175, 0.0000396, 0.0005176, -0.0004761, 0.0004779
5: 0.0173123, 0.0180416, 0.0173103, 0.0180420, -0.0007297, 0.0007313
6: 0.0025910, 0.0034861, 0.0025902, 0.0034871, -0.0008960, 0.0008959
7: -0.0052752, -0.0032609, -0.0052820, -0.0032599, -0.0020154, 0.0020212
8: 0.0125440, 0.0138307, 0.0125386, 0.0138311, -0.0012871, 0.0012921
9: 0.0202862, 0.0226005, 0.0202765, 0.0226012, -0.0021541, 0.0021631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005726
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042020, -0.0041292, -0.0042018, -0.0041322, -0.0000698, 0.0000726
1: -0.0100211, -0.0093131, -0.0100122, -0.0093042, -0.0007169, 0.0006991
2: 0.9644377, 0.9652873, 0.9644483, 0.9652979, -0.0008602, 0.0008390
3: -0.0159954, -0.0097292, -0.0159170, -0.0096501, -0.0048967, 0.0047830
4: 0.0000469, 0.0005235, 0.0000409, 0.0005176, -0.0004706, 0.0004826
5: 0.0173178, 0.0180593, 0.0173117, 0.0180417, -0.0007240, 0.0007476
6: 0.0025571, 0.0034834, 0.0025907, 0.0034864, -0.0009292, 0.0008927
7: -0.0052568, -0.0032220, -0.0052773, -0.0032605, -0.0019964, 0.0020553
8: 0.0125586, 0.0138470, 0.0125424, 0.0138309, -0.0012722, 0.0013046
9: 0.0203125, 0.0226297, 0.0202832, 0.0226007, -0.0021325, 0.0021857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041321, -0.0042012, -0.0041394, -0.0000624, 0.0000691
1: -0.0100124, -0.0093022, -0.0099910, -0.0092525, -0.0007599, 0.0006888
2: 0.9644482, 0.9653004, 0.9644738, 0.9653602, -0.0009120, 0.0008265
3: -0.0159183, -0.0096320, -0.0157288, -0.0091922, -0.0052934, 0.0046713
4: 0.0000396, 0.0005176, 0.0000061, 0.0005032, -0.0004637, 0.0005116
5: 0.0173103, 0.0180420, 0.0172765, 0.0179997, -0.0006894, 0.0007655
6: 0.0025902, 0.0034871, 0.0026713, 0.0035035, -0.0009133, 0.0008158
7: -0.0052820, -0.0032599, -0.0053960, -0.0033529, -0.0019291, 0.0021361
8: 0.0125386, 0.0138311, 0.0124482, 0.0137922, -0.0012535, 0.0013829
9: 0.0202765, 0.0226012, 0.0201139, 0.0225311, -0.0020959, 0.0023283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041322, -0.0042014, -0.0041364, -0.0000654, 0.0000693
1: -0.0100122, -0.0093042, -0.0099998, -0.0092606, -0.0007516, 0.0006956
2: 0.9644483, 0.9652979, 0.9644632, 0.9653503, -0.0009019, 0.0008347
3: -0.0159170, -0.0096501, -0.0158068, -0.0092641, -0.0052680, 0.0047370
4: 0.0000409, 0.0005176, 0.0000116, 0.0005092, -0.0004682, 0.0005060
5: 0.0173117, 0.0180417, 0.0172820, 0.0180171, -0.0007054, 0.0007597
6: 0.0025907, 0.0034864, 0.0026379, 0.0035008, -0.0009101, 0.0008485
7: -0.0052773, -0.0032605, -0.0053774, -0.0033146, -0.0019627, 0.0021169
8: 0.0125424, 0.0138309, 0.0124630, 0.0138082, -0.0012658, 0.0013679
9: 0.0202832, 0.0226007, 0.0201405, 0.0225600, -0.0021187, 0.0023064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041394, -0.0042018, -0.0041321, -0.0000691, 0.0000624
1: -0.0099910, -0.0092525, -0.0100124, -0.0093022, -0.0006888, 0.0007599
2: 0.9644738, 0.9653602, 0.9644482, 0.9653004, -0.0008265, 0.0009120
3: -0.0157288, -0.0091922, -0.0159183, -0.0096320, -0.0046713, 0.0052934
4: 0.0000061, 0.0005032, 0.0000396, 0.0005176, -0.0005116, 0.0004637
5: 0.0172765, 0.0179997, 0.0173103, 0.0180420, -0.0007655, 0.0006894
6: 0.0026713, 0.0035035, 0.0025902, 0.0034871, -0.0008158, 0.0009133
7: -0.0053960, -0.0033529, -0.0052820, -0.0032599, -0.0021361, 0.0019291
8: 0.0124482, 0.0137922, 0.0125386, 0.0138311, -0.0013829, 0.0012535
9: 0.0201139, 0.0225311, 0.0202765, 0.0226012, -0.0023283, 0.0020959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041364, -0.0042018, -0.0041322, -0.0000693, 0.0000654
1: -0.0099998, -0.0092606, -0.0100122, -0.0093042, -0.0006956, 0.0007516
2: 0.9644632, 0.9653503, 0.9644483, 0.9652979, -0.0008347, 0.0009019
3: -0.0158068, -0.0092641, -0.0159170, -0.0096501, -0.0047370, 0.0052680
4: 0.0000116, 0.0005092, 0.0000409, 0.0005176, -0.0005060, 0.0004682
5: 0.0172820, 0.0180171, 0.0173117, 0.0180417, -0.0007597, 0.0007054
6: 0.0026379, 0.0035008, 0.0025907, 0.0034864, -0.0008485, 0.0009101
7: -0.0053774, -0.0033146, -0.0052773, -0.0032605, -0.0021169, 0.0019627
8: 0.0124630, 0.0138082, 0.0125424, 0.0138309, -0.0013679, 0.0012658
9: 0.0201405, 0.0225600, 0.0202832, 0.0226007, -0.0023064, 0.0021187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041394, -0.0042012, -0.0041393, -0.0000619, 0.0000618
1: -0.0099910, -0.0092525, -0.0099912, -0.0092495, -0.0007414, 0.0007387
2: 0.9644738, 0.9653602, 0.9644735, 0.9653636, -0.0008897, 0.0008866
3: -0.0157288, -0.0091922, -0.0157307, -0.0091661, -0.0049112, 0.0049998
4: 0.0000061, 0.0005032, 0.0000041, 0.0005034, -0.0004973, 0.0004991
5: 0.0172765, 0.0179997, 0.0172745, 0.0180001, -0.0007236, 0.0007252
6: 0.0026713, 0.0035035, 0.0026705, 0.0035045, -0.0008332, 0.0008330
7: -0.0053960, -0.0033529, -0.0054028, -0.0033520, -0.0020440, 0.0020499
8: 0.0124482, 0.0137922, 0.0124428, 0.0137925, -0.0013443, 0.0013493
9: 0.0201139, 0.0225311, 0.0201043, 0.0225318, -0.0022481, 0.0022421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006036
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006036
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041364, -0.0042012, -0.0041393, -0.0000621, 0.0000649
1: -0.0099998, -0.0092606, -0.0099910, -0.0092516, -0.0007482, 0.0007304
2: 0.9644632, 0.9653503, 0.9644738, 0.9653611, -0.0008979, 0.0008765
3: -0.0158068, -0.0092641, -0.0157294, -0.0091845, -0.0050879, 0.0049761
4: 0.0000116, 0.0005092, 0.0000055, 0.0005033, -0.0004917, 0.0005037
5: 0.0172820, 0.0180171, 0.0172759, 0.0179998, -0.0007178, 0.0007412
6: 0.0026379, 0.0035008, 0.0026710, 0.0035038, -0.0008659, 0.0008298
7: -0.0053774, -0.0033146, -0.0053980, -0.0033526, -0.0020248, 0.0020834
8: 0.0124630, 0.0138082, 0.0124466, 0.0137923, -0.0013293, 0.0013616
9: 0.0201405, 0.0225600, 0.0201111, 0.0225314, -0.0022260, 0.0022793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006037
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006037
time: 0.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.16 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005726
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005727, upper bound: 0.0005727
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005944, upper bound: 0.0005747
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005747, upper bound: 0.0005944
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006036
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006036
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006037
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 2, lower bound: -0.0005977, upper bound: 0.0006037

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041322, -0.0042018, -0.0041322, -0.0000696, 0.0000696
1: -0.0100122, -0.0093051, -0.0100122, -0.0093051, -0.0007070, 0.0007070
2: 0.9644485, 0.9652969, 0.9644485, 0.9652969, -0.0008485, 0.0008485
3: -0.0159163, -0.0096582, -0.0159163, -0.0096582, -0.0048041, 0.0048041
4: 0.0000415, 0.0005175, 0.0000415, 0.0005175, -0.0004760, 0.0004760
5: 0.0173123, 0.0180416, 0.0173123, 0.0180416, -0.0007293, 0.0007293
6: 0.0025910, 0.0034861, 0.0025910, 0.0034861, -0.0008951, 0.0008951
7: -0.0052752, -0.0032609, -0.0052752, -0.0032609, -0.0020144, 0.0020144
8: 0.0125440, 0.0138307, 0.0125440, 0.0138307, -0.0012867, 0.0012867
9: 0.0202862, 0.0226005, 0.0202862, 0.0226005, -0.0021534, 0.0021534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004815
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005783, upper bound: 0.0005784
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041322, -0.0042020, -0.0041292, -0.0000726, 0.0000698
1: -0.0100122, -0.0093051, -0.0100211, -0.0093131, -0.0006990, 0.0007160
2: 0.9644485, 0.9652969, 0.9644377, 0.9652873, -0.0008389, 0.0008592
3: -0.0159163, -0.0096582, -0.0159954, -0.0097292, -0.0047407, 0.0048866
4: 0.0000415, 0.0005175, 0.0000469, 0.0005235, -0.0004820, 0.0004706
5: 0.0173123, 0.0180416, 0.0173178, 0.0180593, -0.0007469, 0.0007238
6: 0.0025910, 0.0034861, 0.0025571, 0.0034834, -0.0008924, 0.0009289
7: -0.0052752, -0.0032609, -0.0052568, -0.0032220, -0.0020532, 0.0019960
8: 0.0125440, 0.0138307, 0.0125586, 0.0138470, -0.0013029, 0.0012721
9: 0.0202862, 0.0226005, 0.0203125, 0.0226297, -0.0021825, 0.0021273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004815
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005783, upper bound: 0.0005784
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042020, -0.0041292, -0.0042018, -0.0041322, -0.0000698, 0.0000726
1: -0.0100211, -0.0093131, -0.0100122, -0.0093051, -0.0007160, 0.0006990
2: 0.9644377, 0.9652873, 0.9644485, 0.9652969, -0.0008592, 0.0008389
3: -0.0159954, -0.0097292, -0.0159163, -0.0096582, -0.0048866, 0.0047407
4: 0.0000469, 0.0005235, 0.0000415, 0.0005175, -0.0004706, 0.0004820
5: 0.0173178, 0.0180593, 0.0173123, 0.0180416, -0.0007238, 0.0007469
6: 0.0025571, 0.0034834, 0.0025910, 0.0034861, -0.0009289, 0.0008924
7: -0.0052568, -0.0032220, -0.0052752, -0.0032609, -0.0019960, 0.0020532
8: 0.0125586, 0.0138470, 0.0125440, 0.0138307, -0.0012721, 0.0013029
9: 0.0203125, 0.0226297, 0.0202862, 0.0226005, -0.0021273, 0.0021825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 187
type: B, layer: 3, pos: 187
type: B, layer: 3, pos: 141
type: A, layer: 3, pos: 141
type: A, layer: 3, pos: 157
type: B, layer: 3, pos: 245
type: A, layer: 3, pos: 245
type: B, layer: 3, pos: 157

Time for candidate selection: 7.42 seconds

### Candidate
type: A, layer: 3, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004864, upper bound: 0.0005668
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005676, upper bound: 0.0005676
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042020, -0.0041292, -0.0042020, -0.0041292, -0.0000728, 0.0000728
1: -0.0100211, -0.0093131, -0.0100211, -0.0093131, -0.0007079, 0.0007079
2: 0.9644377, 0.9652873, 0.9644377, 0.9652873, -0.0008496, 0.0008496
3: -0.0159954, -0.0097292, -0.0159954, -0.0097292, -0.0048174, 0.0048174
4: 0.0000469, 0.0005235, 0.0000469, 0.0005235, -0.0004766, 0.0004766
5: 0.0173178, 0.0180593, 0.0173178, 0.0180593, -0.0007415, 0.0007415
6: 0.0025571, 0.0034834, 0.0025571, 0.0034834, -0.0009263, 0.0009263
7: -0.0052568, -0.0032220, -0.0052568, -0.0032220, -0.0020348, 0.0020348
8: 0.0125586, 0.0138470, 0.0125586, 0.0138470, -0.0012883, 0.0012883
9: 0.0203125, 0.0226297, 0.0203125, 0.0226297, -0.0021566, 0.0021566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 187
type: B, layer: 3, pos: 187
type: A, layer: 3, pos: 141
type: B, layer: 3, pos: 141
type: A, layer: 3, pos: 245
type: B, layer: 3, pos: 245
type: A, layer: 3, pos: 157
type: B, layer: 3, pos: 157

Time for candidate selection: 7.40 seconds

### Candidate
type: A, layer: 3, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004864, upper bound: 0.0005668
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005676, upper bound: 0.0005676
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041322, -0.0042012, -0.0041394, -0.0000624, 0.0000690
1: -0.0100122, -0.0093051, -0.0099910, -0.0092525, -0.0007597, 0.0006859
2: 0.9644485, 0.9652969, 0.9644738, 0.9653602, -0.0009117, 0.0008231
3: -0.0159163, -0.0096582, -0.0157288, -0.0091922, -0.0052914, 0.0046439
4: 0.0000415, 0.0005175, 0.0000061, 0.0005032, -0.0004617, 0.0005114
5: 0.0173123, 0.0180416, 0.0172765, 0.0179997, -0.0006874, 0.0007651
6: 0.0025910, 0.0034861, 0.0026713, 0.0035035, -0.0009125, 0.0008148
7: -0.0052752, -0.0032609, -0.0053960, -0.0033529, -0.0019223, 0.0021352
8: 0.0125440, 0.0138307, 0.0124482, 0.0137922, -0.0012481, 0.0013825
9: 0.0202862, 0.0226005, 0.0201139, 0.0225311, -0.0020862, 0.0023275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005603, upper bound: 0.0005075
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0005791
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042020, -0.0041292, -0.0042012, -0.0041394, -0.0000626, 0.0000720
1: -0.0100211, -0.0093131, -0.0099910, -0.0092525, -0.0007686, 0.0006778
2: 0.9644377, 0.9652873, 0.9644738, 0.9653602, -0.0009224, 0.0008135
3: -0.0159954, -0.0097292, -0.0157288, -0.0091922, -0.0053738, 0.0045805
4: 0.0000469, 0.0005235, 0.0000061, 0.0005032, -0.0004563, 0.0005174
5: 0.0173178, 0.0180593, 0.0172765, 0.0179997, -0.0006819, 0.0007828
6: 0.0025571, 0.0034834, 0.0026713, 0.0035035, -0.0009464, 0.0008122
7: -0.0052568, -0.0032220, -0.0053960, -0.0033529, -0.0019039, 0.0021740
8: 0.0125586, 0.0138470, 0.0124482, 0.0137922, -0.0012335, 0.0013987
9: 0.0203125, 0.0226297, 0.0201139, 0.0225311, -0.0020601, 0.0023567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005603, upper bound: 0.0005075
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0005791
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042018, -0.0041322, -0.0042014, -0.0041364, -0.0000654, 0.0000693
1: -0.0100122, -0.0093051, -0.0099998, -0.0092606, -0.0007515, 0.0006947
2: 0.9644485, 0.9652969, 0.9644632, 0.9653503, -0.0009018, 0.0008337
3: -0.0159163, -0.0096582, -0.0158068, -0.0092641, -0.0052221, 0.0047269
4: 0.0000415, 0.0005175, 0.0000116, 0.0005092, -0.0004676, 0.0005059
5: 0.0173123, 0.0180416, 0.0172820, 0.0180171, -0.0007048, 0.0007595
6: 0.0025910, 0.0034861, 0.0026379, 0.0035008, -0.0009098, 0.0008482
7: -0.0052752, -0.0032609, -0.0053774, -0.0033146, -0.0019606, 0.0021165
8: 0.0125440, 0.0138307, 0.0124630, 0.0138082, -0.0012642, 0.0013677
9: 0.0202862, 0.0226005, 0.0201405, 0.0225600, -0.0021156, 0.0023013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005300, upper bound: 0.0004502
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005935, upper bound: 0.0005739
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042020, -0.0041292, -0.0042014, -0.0041364, -0.0000657, 0.0000723
1: -0.0100211, -0.0093131, -0.0099998, -0.0092606, -0.0007605, 0.0006866
2: 0.9644377, 0.9652873, 0.9644632, 0.9653503, -0.0009125, 0.0008241
3: -0.0159954, -0.0097292, -0.0158068, -0.0092641, -0.0053025, 0.0046584
4: 0.0000469, 0.0005235, 0.0000116, 0.0005092, -0.0004622, 0.0005119
5: 0.0173178, 0.0180593, 0.0172820, 0.0180171, -0.0006993, 0.0007772
6: 0.0025571, 0.0034834, 0.0026379, 0.0035008, -0.0009437, 0.0008456
7: -0.0052568, -0.0032220, -0.0053774, -0.0033146, -0.0019423, 0.0021554
8: 0.0125586, 0.0138470, 0.0124630, 0.0138082, -0.0012496, 0.0013840
9: 0.0203125, 0.0226297, 0.0201405, 0.0225600, -0.0020894, 0.0023304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005300, upper bound: 0.0004502
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005935, upper bound: 0.0005739
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041394, -0.0042018, -0.0041322, -0.0000690, 0.0000624
1: -0.0099910, -0.0092525, -0.0100122, -0.0093051, -0.0006859, 0.0007597
2: 0.9644738, 0.9653602, 0.9644485, 0.9652969, -0.0008231, 0.0009117
3: -0.0157288, -0.0091922, -0.0159163, -0.0096582, -0.0046439, 0.0052914
4: 0.0000061, 0.0005032, 0.0000415, 0.0005175, -0.0005114, 0.0004617
5: 0.0172765, 0.0179997, 0.0173123, 0.0180416, -0.0007651, 0.0006874
6: 0.0026713, 0.0035035, 0.0025910, 0.0034861, -0.0008148, 0.0009125
7: -0.0053960, -0.0033529, -0.0052752, -0.0032609, -0.0021352, 0.0019223
8: 0.0124482, 0.0137922, 0.0125440, 0.0138307, -0.0013825, 0.0012481
9: 0.0201139, 0.0225311, 0.0202862, 0.0226005, -0.0023275, 0.0020862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005075, upper bound: 0.0005603
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005791, upper bound: 0.0005965
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041394, -0.0042020, -0.0041292, -0.0000720, 0.0000626
1: -0.0099910, -0.0092525, -0.0100211, -0.0093131, -0.0006778, 0.0007686
2: 0.9644738, 0.9653602, 0.9644377, 0.9652873, -0.0008135, 0.0009224
3: -0.0157288, -0.0091922, -0.0159954, -0.0097292, -0.0045805, 0.0053738
4: 0.0000061, 0.0005032, 0.0000469, 0.0005235, -0.0005174, 0.0004563
5: 0.0172765, 0.0179997, 0.0173178, 0.0180593, -0.0007828, 0.0006819
6: 0.0026713, 0.0035035, 0.0025571, 0.0034834, -0.0008122, 0.0009464
7: -0.0053960, -0.0033529, -0.0052568, -0.0032220, -0.0021740, 0.0019039
8: 0.0124482, 0.0137922, 0.0125586, 0.0138470, -0.0013987, 0.0012335
9: 0.0201139, 0.0225311, 0.0203125, 0.0226297, -0.0023567, 0.0020601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005075, upper bound: 0.0005603
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005791, upper bound: 0.0005965
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041364, -0.0042018, -0.0041322, -0.0000693, 0.0000654
1: -0.0099998, -0.0092606, -0.0100122, -0.0093051, -0.0006947, 0.0007515
2: 0.9644632, 0.9653503, 0.9644485, 0.9652969, -0.0008337, 0.0009018
3: -0.0158068, -0.0092641, -0.0159163, -0.0096582, -0.0047269, 0.0052221
4: 0.0000116, 0.0005092, 0.0000415, 0.0005175, -0.0005059, 0.0004676
5: 0.0172820, 0.0180171, 0.0173123, 0.0180416, -0.0007595, 0.0007048
6: 0.0026379, 0.0035008, 0.0025910, 0.0034861, -0.0008482, 0.0009098
7: -0.0053774, -0.0033146, -0.0052752, -0.0032609, -0.0021165, 0.0019606
8: 0.0124630, 0.0138082, 0.0125440, 0.0138307, -0.0013677, 0.0012642
9: 0.0201405, 0.0225600, 0.0202862, 0.0226005, -0.0023013, 0.0021156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004502, upper bound: 0.0005300
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005739, upper bound: 0.0005935
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041364, -0.0042020, -0.0041292, -0.0000723, 0.0000657
1: -0.0099998, -0.0092606, -0.0100211, -0.0093131, -0.0006866, 0.0007605
2: 0.9644632, 0.9653503, 0.9644377, 0.9652873, -0.0008241, 0.0009125
3: -0.0158068, -0.0092641, -0.0159954, -0.0097292, -0.0046584, 0.0053025
4: 0.0000116, 0.0005092, 0.0000469, 0.0005235, -0.0005119, 0.0004622
5: 0.0172820, 0.0180171, 0.0173178, 0.0180593, -0.0007772, 0.0006993
6: 0.0026379, 0.0035008, 0.0025571, 0.0034834, -0.0008456, 0.0009437
7: -0.0053774, -0.0033146, -0.0052568, -0.0032220, -0.0021554, 0.0019423
8: 0.0124630, 0.0138082, 0.0125586, 0.0138470, -0.0013840, 0.0012496
9: 0.0201405, 0.0225600, 0.0203125, 0.0226297, -0.0023304, 0.0020894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004502, upper bound: 0.0005300
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005739, upper bound: 0.0005935
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041394, -0.0042012, -0.0041394, -0.0000618, 0.0000618
1: -0.0099910, -0.0092525, -0.0099910, -0.0092525, -0.0007385, 0.0007385
2: 0.9644738, 0.9653602, 0.9644738, 0.9653602, -0.0008863, 0.0008863
3: -0.0157288, -0.0091922, -0.0157288, -0.0091922, -0.0049979, 0.0049979
4: 0.0000061, 0.0005032, 0.0000061, 0.0005032, -0.0004971, 0.0004971
5: 0.0172765, 0.0179997, 0.0172765, 0.0179997, -0.0007232, 0.0007232
6: 0.0026713, 0.0035035, 0.0026713, 0.0035035, -0.0008322, 0.0008322
7: -0.0053960, -0.0033529, -0.0053960, -0.0033529, -0.0020431, 0.0020431
8: 0.0124482, 0.0137922, 0.0124482, 0.0137922, -0.0013439, 0.0013439
9: 0.0201139, 0.0225311, 0.0201139, 0.0225311, -0.0022474, 0.0022474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005754, upper bound: 0.0005884
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006028
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0042012, -0.0041394, -0.0042014, -0.0041364, -0.0000648, 0.0000621
1: -0.0099910, -0.0092525, -0.0099998, -0.0092606, -0.0007304, 0.0007473
2: 0.9644738, 0.9653602, 0.9644632, 0.9653503, -0.0008764, 0.0008969
3: -0.0157288, -0.0091922, -0.0158068, -0.0092641, -0.0049303, 0.0050786
4: 0.0000061, 0.0005032, 0.0000116, 0.0005092, -0.0005031, 0.0004917
5: 0.0172765, 0.0179997, 0.0172820, 0.0180171, -0.0007406, 0.0007176
6: 0.0026713, 0.0035035, 0.0026379, 0.0035008, -0.0008296, 0.0008656
7: -0.0053960, -0.0033529, -0.0053774, -0.0033146, -0.0020814, 0.0020245
8: 0.0124482, 0.0137922, 0.0124630, 0.0138082, -0.0013600, 0.0013292
9: 0.0201139, 0.0225311, 0.0201405, 0.0225600, -0.0022764, 0.0022210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005807, upper bound: 0.0005555
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006028
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041364, -0.0042012, -0.0041394, -0.0000621, 0.0000648
1: -0.0099998, -0.0092606, -0.0099910, -0.0092525, -0.0007473, 0.0007304
2: 0.9644632, 0.9653503, 0.9644738, 0.9653602, -0.0008969, 0.0008764
3: -0.0158068, -0.0092641, -0.0157288, -0.0091922, -0.0050786, 0.0049303
4: 0.0000116, 0.0005092, 0.0000061, 0.0005032, -0.0004917, 0.0005031
5: 0.0172820, 0.0180171, 0.0172765, 0.0179997, -0.0007176, 0.0007406
6: 0.0026379, 0.0035008, 0.0026713, 0.0035035, -0.0008656, 0.0008296
7: -0.0053774, -0.0033146, -0.0053960, -0.0033529, -0.0020245, 0.0020814
8: 0.0124630, 0.0138082, 0.0124482, 0.0137922, -0.0013292, 0.0013600
9: 0.0201405, 0.0225600, 0.0201139, 0.0225311, -0.0022210, 0.0022764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005413, upper bound: 0.0005811
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006029
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041364, -0.0042014, -0.0041364, -0.0000651, 0.0000651
1: -0.0099998, -0.0092606, -0.0099998, -0.0092606, -0.0007392, 0.0007392
2: 0.9644632, 0.9653503, 0.9644632, 0.9653503, -0.0008870, 0.0008870
3: -0.0158068, -0.0092641, -0.0158068, -0.0092641, -0.0050102, 0.0050102
4: 0.0000116, 0.0005092, 0.0000116, 0.0005092, -0.0004976, 0.0004976
5: 0.0172820, 0.0180171, 0.0172820, 0.0180171, -0.0007351, 0.0007351
6: 0.0026379, 0.0035008, 0.0026379, 0.0035008, -0.0008629, 0.0008629
7: -0.0053774, -0.0033146, -0.0053774, -0.0033146, -0.0020628, 0.0020628
8: 0.0124630, 0.0138082, 0.0124630, 0.0138082, -0.0013452, 0.0013452
9: 0.0201405, 0.0225600, 0.0201405, 0.0225600, -0.0022500, 0.0022500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005413, upper bound: 0.0005811
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006029
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.95 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004815
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005783, upper bound: 0.0005784
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004815
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005783, upper bound: 0.0005784
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0004864, upper bound: 0.0005668
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005676, upper bound: 0.0005676
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0004864, upper bound: 0.0005668
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005676, upper bound: 0.0005676
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005603, upper bound: 0.0005075
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0005791
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005603, upper bound: 0.0005075
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0005791
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005300, upper bound: 0.0004502
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005935, upper bound: 0.0005739
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005300, upper bound: 0.0004502
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005935, upper bound: 0.0005739
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005075, upper bound: 0.0005603
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005791, upper bound: 0.0005965
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005075, upper bound: 0.0005603
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005791, upper bound: 0.0005965
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0004502, upper bound: 0.0005300
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005739, upper bound: 0.0005935
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0004502, upper bound: 0.0005300
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005739, upper bound: 0.0005935
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005754, upper bound: 0.0005884
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006028
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005807, upper bound: 0.0005555
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006028
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005413, upper bound: 0.0005811
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006029
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005413, upper bound: 0.0005811
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.95
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006029

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041366, -0.0042017, -0.0041329, -0.0000685, 0.0000651
1: -0.0099990, -0.0092961, -0.0100100, -0.0093053, -0.0006937, 0.0007139
2: 0.9644642, 0.9653077, 0.9644510, 0.9652967, -0.0008325, 0.0008566
3: -0.0158001, -0.0095785, -0.0158972, -0.0096597, -0.0046889, 0.0048654
4: 0.0000355, 0.0005087, 0.0000416, 0.0005160, -0.0004806, 0.0004670
5: 0.0173062, 0.0180156, 0.0173124, 0.0180373, -0.0007311, 0.0007031
6: 0.0026408, 0.0034891, 0.0025991, 0.0034860, -0.0008453, 0.0008899
7: -0.0052959, -0.0033179, -0.0052749, -0.0032702, -0.0020257, 0.0019569
8: 0.0125276, 0.0138068, 0.0125443, 0.0138268, -0.0012992, 0.0012625
9: 0.0202567, 0.0225575, 0.0202868, 0.0225934, -0.0021761, 0.0021098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005580, upper bound: 0.0005580
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005580, upper bound: 0.0005719
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042017, -0.0041330, -0.0042018, -0.0041322, -0.0000695, 0.0000688
1: -0.0100097, -0.0093053, -0.0100122, -0.0093051, -0.0007046, 0.0007069
2: 0.9644514, 0.9652966, 0.9644485, 0.9652969, -0.0008456, 0.0008481
3: -0.0158948, -0.0096598, -0.0159163, -0.0096582, -0.0047806, 0.0048024
4: 0.0000417, 0.0005159, 0.0000415, 0.0005175, -0.0004758, 0.0004743
5: 0.0173124, 0.0180368, 0.0173123, 0.0180416, -0.0007291, 0.0007245
6: 0.0026002, 0.0034860, 0.0025910, 0.0034861, -0.0008859, 0.0008950
7: -0.0052748, -0.0032714, -0.0052752, -0.0032609, -0.0020140, 0.0020039
8: 0.0125443, 0.0138263, 0.0125440, 0.0138307, -0.0012863, 0.0012823
9: 0.0202868, 0.0225925, 0.0202862, 0.0226005, -0.0021528, 0.0021453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005719, upper bound: 0.0005672
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005719, upper bound: 0.0005947
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0042014, -0.0041366, -0.0042020, -0.0041299, -0.0000715, 0.0000653
1: -0.0099990, -0.0092961, -0.0100190, -0.0093133, -0.0006857, 0.0007229
2: 0.9644642, 0.9653077, 0.9644402, 0.9652871, -0.0008229, 0.0008674
3: -0.0158001, -0.0095785, -0.0159767, -0.0097306, -0.0046256, 0.0049480
4: 0.0000355, 0.0005087, 0.0000470, 0.0005221, -0.0004866, 0.0004616
5: 0.0173062, 0.0180156, 0.0173179, 0.0180551, -0.0007489, 0.0006977
6: 0.0026408, 0.0034891, 0.0025652, 0.0034834, -0.0008426, 0.0009239
7: -0.0052959, -0.0033179, -0.0052565, -0.0032312, -0.0020647, 0.0019386
8: 0.0125276, 0.0138068, 0.0125589, 0.0138431, -0.0013155, 0.0012479
9: 0.0202567, 0.0225575, 0.0203130, 0.0226228, -0.0022054, 0.0020838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 187
type: A, layer: 3, pos: 187
type: A, layer: 3, pos: 141
type: B, layer: 3, pos: 157
type: A, layer: 3, pos: 245
type: B, layer: 3, pos: 245
type: B, layer: 3, pos: 141
type: A, layer: 3, pos: 157

Time for candidate selection: 5.68 seconds

### Candidate
type: B, layer: 3, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 141

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 157

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 245

## Relational analysis of IS_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004514, upper bound: 0.0004577
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 245

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 141

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 157

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5

No IS candidates found

### IS candidates at layer 7
type: B, layer: 7, pos: 144
type: A, layer: 7, pos: 144

Time for candidate selection: 21.27 seconds

### Candidate
type: B, layer: 7, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

No IS candidates found

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.93 + 181.38 = 184.31 seconds
