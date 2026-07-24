## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00640413


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040857, -0.0023094, -0.0040857, -0.0023094, -0.0017764, 0.0017764)
1: (-0.0056657, -0.0028656, -0.0056657, -0.0028656, -0.0028001, 0.0028001)
2: (0.9637213, 0.9716780, 0.9637213, 0.9716780, -0.0079567, 0.0079567)
3: (0.0225558, 0.0374076, 0.0225558, 0.0374076, -0.0122357, 0.0122357)
4: (-0.0035381, 0.0000305, -0.0035381, 0.0000305, -0.0035686, 0.0035686)
5: (0.0123002, 0.0148361, 0.0123002, 0.0148361, -0.0025359, 0.0025359)
6: (0.0021688, 0.0052458, 0.0021688, 0.0052458, -0.0030770, 0.0030770)
7: (-0.0174728, -0.0121105, -0.0174728, -0.0121105, -0.0053623, 0.0053623)
8: (0.0028671, 0.0072861, 0.0028671, 0.0072861, -0.0044190, 0.0044190)
9: (0.0016518, 0.0083735, 0.0016518, 0.0083735, -0.0067218, 0.0067218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.90 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.13 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.28 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0040851, -0.0023098, -0.0040819, -0.0023347, -0.0017504, 0.0017721
1: -0.0056444, -0.0028661, -0.0055233, -0.0028915, -0.0027529, 0.0026572
2: 0.9637231, 0.9716778, 0.9638253, 0.9716665, -0.0079435, 0.0078525
3: 0.0227438, 0.0374061, 0.0238159, 0.0373223, -0.0112937, 0.0109453
4: -0.0035380, 0.0000297, -0.0035316, -0.0000149, -0.0035231, 0.0035613
5: 0.0123006, 0.0148216, 0.0123270, 0.0147392, -0.0024386, 0.0024947
6: 0.0021695, 0.0052457, 0.0022103, 0.0052426, -0.0030731, 0.0030355
7: -0.0174724, -0.0121115, -0.0174507, -0.0121662, -0.0053061, 0.0053392
8: 0.0028674, 0.0072852, 0.0028846, 0.0072394, -0.0043720, 0.0044006
9: 0.0016526, 0.0083040, 0.0017012, 0.0079076, -0.0062549, 0.0066028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.16 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.30 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0040857, -0.0023094, -0.0040845, -0.0023108, -0.0017749, 0.0017751
1: -0.0056657, -0.0028656, -0.0056188, -0.0028671, -0.0027986, 0.0027532
2: 0.9637213, 0.9716780, 0.9637273, 0.9716773, -0.0079560, 0.0079507
3: 0.0225558, 0.0374076, 0.0229702, 0.0374027, -0.0118661, 0.0111413
4: -0.0035381, 0.0000305, -0.0035377, 0.0000279, -0.0035660, 0.0035682
5: 0.0123002, 0.0148361, 0.0123017, 0.0148042, -0.0025041, 0.0025344
6: 0.0021688, 0.0052458, 0.0021711, 0.0052456, -0.0030768, 0.0030747
7: -0.0174728, -0.0121105, -0.0174715, -0.0121136, -0.0053591, 0.0053610
8: 0.0028671, 0.0072861, 0.0028681, 0.0072834, -0.0044163, 0.0044180
9: 0.0016518, 0.0083735, 0.0016546, 0.0082203, -0.0065685, 0.0067190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.35 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.66 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0023347, -0.0040819, -0.0023347, -0.0017472, 0.0017472
1: -0.0055233, -0.0028915, -0.0055233, -0.0028915, -0.0026318, 0.0026318
2: 0.9638253, 0.9716665, 0.9638253, 0.9716665, -0.0078412, 0.0078412
3: 0.0238159, 0.0373223, 0.0238159, 0.0373223, -0.0101624, 0.0101624
4: -0.0035316, -0.0000149, -0.0035316, -0.0000149, -0.0035168, 0.0035168
5: 0.0123270, 0.0147392, 0.0123270, 0.0147392, -0.0024123, 0.0024123
6: 0.0022103, 0.0052426, 0.0022103, 0.0052426, -0.0030323, 0.0030323
7: -0.0174507, -0.0121662, -0.0174507, -0.0121662, -0.0052844, 0.0052844
8: 0.0028846, 0.0072394, 0.0028846, 0.0072394, -0.0043548, 0.0043548
9: 0.0017012, 0.0079076, 0.0017012, 0.0079076, -0.0062064, 0.0062064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.27 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040845, -0.0023108, -0.0040819, -0.0023347, -0.0017497, 0.0017711
1: -0.0056188, -0.0028671, -0.0055233, -0.0028915, -0.0027273, 0.0026562
2: 0.9637273, 0.9716773, 0.9638253, 0.9716665, -0.0079392, 0.0078520
3: 0.0229702, 0.0374027, 0.0238159, 0.0373223, -0.0112395, 0.0105203
4: -0.0035377, 0.0000279, -0.0035316, -0.0000149, -0.0035229, 0.0035595
5: 0.0123017, 0.0148042, 0.0123270, 0.0147392, -0.0024375, 0.0024773
6: 0.0021711, 0.0052456, 0.0022103, 0.0052426, -0.0030715, 0.0030353
7: -0.0174715, -0.0121136, -0.0174507, -0.0121662, -0.0053052, 0.0053370
8: 0.0028681, 0.0072834, 0.0028846, 0.0072394, -0.0043713, 0.0043988
9: 0.0016546, 0.0082203, 0.0017012, 0.0079076, -0.0062530, 0.0065191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069117
time: 1.15 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.23 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0023347, -0.0040845, -0.0023108, -0.0017711, 0.0017497
1: -0.0055233, -0.0028915, -0.0056188, -0.0028671, -0.0026562, 0.0027273
2: 0.9638253, 0.9716665, 0.9637273, 0.9716773, -0.0078520, 0.0079392
3: 0.0238159, 0.0373223, 0.0229702, 0.0374027, -0.0105203, 0.0112395
4: -0.0035316, -0.0000149, -0.0035377, 0.0000279, -0.0035595, 0.0035229
5: 0.0123270, 0.0147392, 0.0123017, 0.0148042, -0.0024773, 0.0024375
6: 0.0022103, 0.0052426, 0.0021711, 0.0052456, -0.0030353, 0.0030715
7: -0.0174507, -0.0121662, -0.0174715, -0.0121136, -0.0053370, 0.0053052
8: 0.0028846, 0.0072394, 0.0028681, 0.0072834, -0.0043988, 0.0043713
9: 0.0017012, 0.0079076, 0.0016546, 0.0082203, -0.0065191, 0.0062530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.21 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.07 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040845, -0.0023108, -0.0040845, -0.0023108, -0.0017736, 0.0017736
1: -0.0056188, -0.0028671, -0.0056188, -0.0028671, -0.0027518, 0.0027518
2: 0.9637273, 0.9716773, 0.9637273, 0.9716773, -0.0079500, 0.0079500
3: 0.0229702, 0.0374027, 0.0229702, 0.0374027, -0.0105946, 0.0105946
4: -0.0035377, 0.0000279, -0.0035377, 0.0000279, -0.0035656, 0.0035656
5: 0.0123017, 0.0148042, 0.0123017, 0.0148042, -0.0025025, 0.0025025
6: 0.0021711, 0.0052456, 0.0021711, 0.0052456, -0.0030745, 0.0030745
7: -0.0174715, -0.0121136, -0.0174715, -0.0121136, -0.0053578, 0.0053578
8: 0.0028681, 0.0072834, 0.0028681, 0.0072834, -0.0044153, 0.0044153
9: 0.0016546, 0.0082203, 0.0016546, 0.0082203, -0.0065657, 0.0065657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.25 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.47 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069117
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040819, -0.0023347, -0.0017471, 0.0015719
1: -0.0055228, -0.0030704, -0.0055233, -0.0028915, -0.0026313, 0.0024529
2: 0.9645426, 0.9715868, 0.9638253, 0.9716665, -0.0071239, 0.0077615
3: 0.0238203, 0.0367342, 0.0238159, 0.0373223, -0.0101507, 0.0095224
4: -0.0034869, -0.0003276, -0.0035316, -0.0000149, -0.0034720, 0.0032040
5: 0.0125119, 0.0147389, 0.0123270, 0.0147392, -0.0022274, 0.0024119
6: 0.0024965, 0.0052206, 0.0022103, 0.0052426, -0.0027461, 0.0030103
7: -0.0172982, -0.0125510, -0.0174507, -0.0121662, -0.0051320, 0.0048997
8: 0.0030055, 0.0069177, 0.0028846, 0.0072394, -0.0042339, 0.0040330
9: 0.0020418, 0.0079059, 0.0017012, 0.0079076, -0.0058657, 0.0062048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.19 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.14 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040819, -0.0023644, -0.0017195, 0.0015797
1: -0.0055997, -0.0030624, -0.0055232, -0.0029219, -0.0026778, 0.0024607
2: 0.9645107, 0.9715902, 0.9639469, 0.9716529, -0.0071422, 0.0076433
3: 0.0231401, 0.0367604, 0.0238171, 0.0372227, -0.0109714, 0.0097929
4: -0.0034889, -0.0003137, -0.0035240, -0.0000678, -0.0034210, 0.0032103
5: 0.0125036, 0.0147912, 0.0123583, 0.0147391, -0.0022355, 0.0024329
6: 0.0024838, 0.0052216, 0.0022588, 0.0052389, -0.0027551, 0.0029628
7: -0.0173050, -0.0125339, -0.0174248, -0.0122315, -0.0050736, 0.0048909
8: 0.0030002, 0.0069320, 0.0029051, 0.0071849, -0.0041847, 0.0040269
9: 0.0020267, 0.0081575, 0.0017589, 0.0079071, -0.0058805, 0.0063986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.35 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040845, -0.0023108, -0.0040819, -0.0025100, -0.0015745, 0.0017711
1: -0.0056188, -0.0028671, -0.0055228, -0.0030704, -0.0025484, 0.0026557
2: 0.9637273, 0.9716773, 0.9645426, 0.9715868, -0.0078595, 0.0071347
3: 0.0229702, 0.0374027, 0.0238203, 0.0367342, -0.0105994, 0.0105086
4: -0.0035377, 0.0000279, -0.0034869, -0.0003276, -0.0032101, 0.0035148
5: 0.0123017, 0.0148042, 0.0125119, 0.0147389, -0.0024372, 0.0022924
6: 0.0021711, 0.0052456, 0.0024965, 0.0052206, -0.0030495, 0.0027491
7: -0.0174715, -0.0121136, -0.0172982, -0.0125510, -0.0049205, 0.0051846
8: 0.0028681, 0.0072834, 0.0030055, 0.0069177, -0.0040496, 0.0042779
9: 0.0016546, 0.0082203, 0.0020418, 0.0079059, -0.0062513, 0.0061785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.22 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040845, -0.0023418, -0.0040839, -0.0025022, -0.0015823, 0.0017422
1: -0.0056187, -0.0028987, -0.0055997, -0.0030624, -0.0025563, 0.0027010
2: 0.9638539, 0.9716632, 0.9645107, 0.9715902, -0.0077363, 0.0071526
3: 0.0229714, 0.0372988, 0.0231401, 0.0367604, -0.0108704, 0.0113030
4: -0.0035298, -0.0000274, -0.0034889, -0.0003137, -0.0032161, 0.0034615
5: 0.0123344, 0.0148042, 0.0125036, 0.0147912, -0.0024568, 0.0023005
6: 0.0022217, 0.0052417, 0.0024838, 0.0052216, -0.0029999, 0.0027579
7: -0.0174446, -0.0121816, -0.0173050, -0.0125339, -0.0049107, 0.0051234
8: 0.0028894, 0.0072266, 0.0030002, 0.0069320, -0.0040425, 0.0042264
9: 0.0017148, 0.0082198, 0.0020267, 0.0081575, -0.0064427, 0.0061932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067878
time: 1.06 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.11 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040845, -0.0023108, -0.0017711, 0.0015745
1: -0.0055228, -0.0030704, -0.0056188, -0.0028671, -0.0026557, 0.0025484
2: 0.9645426, 0.9715868, 0.9637273, 0.9716773, -0.0071347, 0.0078595
3: 0.0238203, 0.0367342, 0.0229702, 0.0374027, -0.0105086, 0.0105994
4: -0.0034869, -0.0003276, -0.0035377, 0.0000279, -0.0035148, 0.0032101
5: 0.0125119, 0.0147389, 0.0123017, 0.0148042, -0.0022924, 0.0024372
6: 0.0024965, 0.0052206, 0.0021711, 0.0052456, -0.0027491, 0.0030495
7: -0.0172982, -0.0125510, -0.0174715, -0.0121136, -0.0051846, 0.0049205
8: 0.0030055, 0.0069177, 0.0028681, 0.0072834, -0.0042779, 0.0040496
9: 0.0020418, 0.0079059, 0.0016546, 0.0082203, -0.0061785, 0.0062513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.22 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.18 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040845, -0.0023418, -0.0017422, 0.0015823
1: -0.0055997, -0.0030624, -0.0056187, -0.0028987, -0.0027010, 0.0025563
2: 0.9645107, 0.9715902, 0.9638539, 0.9716632, -0.0071526, 0.0077363
3: 0.0231401, 0.0367604, 0.0229714, 0.0372988, -0.0113030, 0.0108704
4: -0.0034889, -0.0003137, -0.0035298, -0.0000274, -0.0034615, 0.0032161
5: 0.0125036, 0.0147912, 0.0123344, 0.0148042, -0.0023005, 0.0024568
6: 0.0024838, 0.0052216, 0.0022217, 0.0052417, -0.0027579, 0.0029999
7: -0.0173050, -0.0125339, -0.0174446, -0.0121816, -0.0051234, 0.0049107
8: 0.0030002, 0.0069320, 0.0028894, 0.0072266, -0.0042264, 0.0040425
9: 0.0020267, 0.0081575, 0.0017148, 0.0082198, -0.0061932, 0.0064427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063901
time: 1.23 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.20 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040845, -0.0023108, -0.0017736, 0.0015998
1: -0.0056184, -0.0030446, -0.0056188, -0.0028671, -0.0027513, 0.0025743
2: 0.9644390, 0.9715982, 0.9637273, 0.9716773, -0.0072383, 0.0078709
3: 0.0229746, 0.0368191, 0.0229702, 0.0374027, -0.0105826, 0.0099676
4: -0.0034933, -0.0002825, -0.0035377, 0.0000279, -0.0035212, 0.0032553
5: 0.0124852, 0.0148039, 0.0123017, 0.0148042, -0.0023191, 0.0025022
6: 0.0024552, 0.0052238, 0.0021711, 0.0052456, -0.0027904, 0.0030526
7: -0.0173202, -0.0124955, -0.0174715, -0.0121136, -0.0052066, 0.0049760
8: 0.0029881, 0.0069641, 0.0028681, 0.0072834, -0.0042953, 0.0040960
9: 0.0019926, 0.0082187, 0.0016546, 0.0082203, -0.0062276, 0.0065641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.12 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.35 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040845, -0.0023418, -0.0017446, 0.0015968
1: -0.0056918, -0.0030476, -0.0056187, -0.0028987, -0.0027931, 0.0025711
2: 0.9644510, 0.9715968, 0.9638539, 0.9716632, -0.0072122, 0.0077429
3: 0.0223248, 0.0368093, 0.0229714, 0.0372988, -0.0113953, 0.0102023
4: -0.0034926, -0.0002877, -0.0035298, -0.0000274, -0.0034652, 0.0032421
5: 0.0124883, 0.0148539, 0.0123344, 0.0148042, -0.0023159, 0.0025195
6: 0.0024600, 0.0052234, 0.0022217, 0.0052417, -0.0027817, 0.0030017
7: -0.0173177, -0.0125019, -0.0174446, -0.0121816, -0.0051361, 0.0049427
8: 0.0029901, 0.0069587, 0.0028894, 0.0072266, -0.0042364, 0.0040693
9: 0.0019983, 0.0084590, 0.0017148, 0.0082198, -0.0062215, 0.0067442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.29 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.61 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067878
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063901
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040819, -0.0025100, -0.0015719, 0.0015719
1: -0.0055228, -0.0030704, -0.0055228, -0.0030704, -0.0024524, 0.0024524
2: 0.9645426, 0.9715868, 0.9645426, 0.9715868, -0.0070442, 0.0070442
3: 0.0238203, 0.0367342, 0.0238203, 0.0367342, -0.0095106, 0.0095106
4: -0.0034869, -0.0003276, -0.0034869, -0.0003276, -0.0031593, 0.0031593
5: 0.0125119, 0.0147389, 0.0125119, 0.0147389, -0.0022270, 0.0022270
6: 0.0024965, 0.0052206, 0.0024965, 0.0052206, -0.0027241, 0.0027241
7: -0.0172982, -0.0125510, -0.0172982, -0.0125510, -0.0047473, 0.0047473
8: 0.0030055, 0.0069177, 0.0030055, 0.0069177, -0.0039121, 0.0039121
9: 0.0020418, 0.0079059, 0.0020418, 0.0079059, -0.0058641, 0.0058641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063790
time: 1.31 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.14 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040839, -0.0025022, -0.0015797, 0.0015740
1: -0.0055228, -0.0030704, -0.0055997, -0.0030624, -0.0024604, 0.0025293
2: 0.9645426, 0.9715868, 0.9645107, 0.9715902, -0.0070477, 0.0070761
3: 0.0238203, 0.0367342, 0.0231401, 0.0367604, -0.0097690, 0.0104070
4: -0.0034869, -0.0003276, -0.0034889, -0.0003137, -0.0031732, 0.0031613
5: 0.0125119, 0.0147389, 0.0125036, 0.0147912, -0.0022793, 0.0022353
6: 0.0024965, 0.0052206, 0.0024838, 0.0052216, -0.0027251, 0.0027368
7: -0.0172982, -0.0125510, -0.0173050, -0.0125339, -0.0047643, 0.0047540
8: 0.0030055, 0.0069177, 0.0030002, 0.0069320, -0.0039264, 0.0039175
9: 0.0020418, 0.0079059, 0.0020267, 0.0081575, -0.0061157, 0.0058793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
time: 1.13 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.20 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040823, -0.0026585, -0.0014255, 0.0015285
1: -0.0055991, -0.0031151, -0.0055393, -0.0032220, -0.0023771, 0.0024242
2: 0.9647219, 0.9715667, 0.9651505, 0.9715191, -0.0067972, 0.0064161
3: 0.0231453, 0.0365872, 0.0236743, 0.0362358, -0.0098112, 0.0094598
4: -0.0034757, -0.0004058, -0.0034490, -0.0005927, -0.0028830, 0.0030432
5: 0.0125581, 0.0147908, 0.0126685, 0.0147501, -0.0021921, 0.0021223
6: 0.0025681, 0.0052151, 0.0027391, 0.0052020, -0.0026339, 0.0024760
7: -0.0172601, -0.0126472, -0.0171691, -0.0128771, -0.0043831, 0.0045219
8: 0.0030358, 0.0068372, 0.0031080, 0.0066450, -0.0036092, 0.0037292
9: 0.0021270, 0.0081555, 0.0023305, 0.0079599, -0.0058330, 0.0058250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063790
time: 1.05 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.15 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040819, -0.0024473, -0.0016367, 0.0015797
1: -0.0055997, -0.0030624, -0.0055224, -0.0030064, -0.0025933, 0.0024600
2: 0.9645107, 0.9715902, 0.9642858, 0.9716153, -0.0071046, 0.0073044
3: 0.0231401, 0.0367604, 0.0238238, 0.0369447, -0.0101667, 0.0097836
4: -0.0034889, -0.0003137, -0.0035029, -0.0002157, -0.0032732, 0.0031892
5: 0.0125036, 0.0147912, 0.0124457, 0.0147386, -0.0022350, 0.0023455
6: 0.0024838, 0.0052216, 0.0023941, 0.0052285, -0.0027447, 0.0028275
7: -0.0173050, -0.0125339, -0.0173528, -0.0124133, -0.0048917, 0.0048189
8: 0.0030002, 0.0069320, 0.0029623, 0.0070328, -0.0040326, 0.0039697
9: 0.0020267, 0.0081575, 0.0019199, 0.0079046, -0.0058780, 0.0062376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
time: 1.21 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.20 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040819, -0.0025100, -0.0015745, 0.0015972
1: -0.0056184, -0.0030446, -0.0055228, -0.0030704, -0.0025480, 0.0024782
2: 0.9644390, 0.9715982, 0.9645426, 0.9715868, -0.0071477, 0.0070556
3: 0.0229746, 0.0368191, 0.0238203, 0.0367342, -0.0105902, 0.0099433
4: -0.0034933, -0.0002825, -0.0034869, -0.0003276, -0.0031657, 0.0032044
5: 0.0124852, 0.0148039, 0.0125119, 0.0147389, -0.0022537, 0.0022921
6: 0.0024552, 0.0052238, 0.0024965, 0.0052206, -0.0027654, 0.0027273
7: -0.0173202, -0.0124955, -0.0172982, -0.0125510, -0.0047693, 0.0048028
8: 0.0029881, 0.0069641, 0.0030055, 0.0069177, -0.0039296, 0.0039586
9: 0.0019926, 0.0082187, 0.0020418, 0.0079059, -0.0059133, 0.0061769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067797, upper bound: 0.0063906
time: 1.35 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067977, upper bound: 0.0068119
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040819, -0.0025100, -0.0015764, 0.0015943
1: -0.0056918, -0.0030476, -0.0055228, -0.0030704, -0.0026214, 0.0024752
2: 0.9644510, 0.9715968, 0.9645426, 0.9715868, -0.0071357, 0.0070543
3: 0.0223248, 0.0368093, 0.0238203, 0.0367342, -0.0113067, 0.0099772
4: -0.0034926, -0.0002877, -0.0034869, -0.0003276, -0.0031650, 0.0031992
5: 0.0124883, 0.0148539, 0.0125119, 0.0147389, -0.0022506, 0.0023420
6: 0.0024600, 0.0052234, 0.0024965, 0.0052206, -0.0027606, 0.0027269
7: -0.0173177, -0.0125019, -0.0172982, -0.0125510, -0.0047667, 0.0047963
8: 0.0029901, 0.0069587, 0.0030055, 0.0069177, -0.0039275, 0.0039532
9: 0.0019983, 0.0084590, 0.0020418, 0.0079059, -0.0059076, 0.0064171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067797, upper bound: 0.0063906
time: 1.39 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067977, upper bound: 0.0068119
time: 1.30 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0026329, -0.0040839, -0.0025538, -0.0015312, 0.0014510
1: -0.0056386, -0.0031959, -0.0055991, -0.0031151, -0.0025235, 0.0024032
2: 0.9650460, 0.9715307, 0.9647219, 0.9715667, -0.0065207, 0.0068088
3: 0.0227953, 0.0363215, 0.0231453, 0.0365872, -0.0107727, 0.0103037
4: -0.0034555, -0.0005471, -0.0034757, -0.0004058, -0.0030497, 0.0029286
5: 0.0126416, 0.0148177, 0.0125581, 0.0147908, -0.0021492, 0.0022596
6: 0.0026974, 0.0052052, 0.0025681, 0.0052151, -0.0025177, 0.0026371
7: -0.0171913, -0.0128210, -0.0172601, -0.0126472, -0.0045441, 0.0044392
8: 0.0030904, 0.0066919, 0.0030358, 0.0068372, -0.0037468, 0.0036561
9: 0.0022809, 0.0082850, 0.0021270, 0.0081555, -0.0058747, 0.0061580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063790, upper bound: 0.0067878
time: 1.12 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063790, upper bound: 0.0067789
time: 1.15 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024269, -0.0040839, -0.0025022, -0.0015822, 0.0016571
1: -0.0056179, -0.0029856, -0.0055997, -0.0030624, -0.0025554, 0.0026141
2: 0.9642025, 0.9716244, 0.9645107, 0.9715902, -0.0073878, 0.0071138
3: 0.0229790, 0.0370131, 0.0231401, 0.0367604, -0.0108605, 0.0105856
4: -0.0035081, -0.0001793, -0.0034889, -0.0003137, -0.0031944, 0.0033096
5: 0.0124242, 0.0148036, 0.0125036, 0.0147912, -0.0023670, 0.0022999
6: 0.0023608, 0.0052310, 0.0024838, 0.0052216, -0.0028608, 0.0027472
7: -0.0173705, -0.0123685, -0.0173050, -0.0125339, -0.0048366, 0.0049365
8: 0.0029482, 0.0070702, 0.0030002, 0.0069320, -0.0039838, 0.0040701
9: 0.0018803, 0.0082170, 0.0020267, 0.0081575, -0.0062772, 0.0061904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.16 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0068059
time: 1.40 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040844, -0.0024847, -0.0015972, 0.0015745
1: -0.0055228, -0.0030704, -0.0056184, -0.0030446, -0.0024782, 0.0025480
2: 0.9645426, 0.9715868, 0.9644390, 0.9715982, -0.0070556, 0.0071477
3: 0.0238203, 0.0367342, 0.0229746, 0.0368191, -0.0099433, 0.0105902
4: -0.0034869, -0.0003276, -0.0034933, -0.0002825, -0.0032044, 0.0031657
5: 0.0125119, 0.0147389, 0.0124852, 0.0148039, -0.0022921, 0.0022537
6: 0.0024965, 0.0052206, 0.0024552, 0.0052238, -0.0027273, 0.0027654
7: -0.0172982, -0.0125510, -0.0173202, -0.0124955, -0.0048028, 0.0047693
8: 0.0030055, 0.0069177, 0.0029881, 0.0069641, -0.0039586, 0.0039296
9: 0.0020418, 0.0079059, 0.0019926, 0.0082187, -0.0061769, 0.0059133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067797
time: 1.06 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.13 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040864, -0.0024876, -0.0015943, 0.0015764
1: -0.0055228, -0.0030704, -0.0056918, -0.0030476, -0.0024752, 0.0026214
2: 0.9645426, 0.9715868, 0.9644510, 0.9715968, -0.0070543, 0.0071357
3: 0.0238203, 0.0367342, 0.0223248, 0.0368093, -0.0099772, 0.0113067
4: -0.0034869, -0.0003276, -0.0034926, -0.0002877, -0.0031992, 0.0031650
5: 0.0125119, 0.0147389, 0.0124883, 0.0148539, -0.0023420, 0.0022506
6: 0.0024965, 0.0052206, 0.0024600, 0.0052234, -0.0027269, 0.0027606
7: -0.0172982, -0.0125510, -0.0173177, -0.0125019, -0.0047963, 0.0047667
8: 0.0030055, 0.0069177, 0.0029901, 0.0069587, -0.0039532, 0.0039275
9: 0.0020418, 0.0079059, 0.0019983, 0.0084590, -0.0064171, 0.0059076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
time: 1.19 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.25 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040850, -0.0026329, -0.0014510, 0.0015312
1: -0.0055991, -0.0031151, -0.0056386, -0.0031959, -0.0024032, 0.0025235
2: 0.9647219, 0.9715667, 0.9650460, 0.9715307, -0.0068088, 0.0065207
3: 0.0231453, 0.0365872, 0.0227953, 0.0363215, -0.0103037, 0.0107727
4: -0.0034757, -0.0004058, -0.0034555, -0.0005471, -0.0029286, 0.0030497
5: 0.0125581, 0.0147908, 0.0126416, 0.0148177, -0.0022596, 0.0021492
6: 0.0025681, 0.0052151, 0.0026974, 0.0052052, -0.0026371, 0.0025177
7: -0.0172601, -0.0126472, -0.0171913, -0.0128210, -0.0044392, 0.0045441
8: 0.0030358, 0.0068372, 0.0030904, 0.0066919, -0.0036561, 0.0037468
9: 0.0021270, 0.0081555, 0.0022809, 0.0082850, -0.0061580, 0.0058747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063790
time: 1.23 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063901
time: 1.25 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040844, -0.0024269, -0.0016571, 0.0015822
1: -0.0055997, -0.0030624, -0.0056179, -0.0029856, -0.0026141, 0.0025554
2: 0.9645107, 0.9715902, 0.9642025, 0.9716244, -0.0071138, 0.0073878
3: 0.0231401, 0.0367604, 0.0229790, 0.0370131, -0.0105856, 0.0108605
4: -0.0034889, -0.0003137, -0.0035081, -0.0001793, -0.0033096, 0.0031944
5: 0.0125036, 0.0147912, 0.0124242, 0.0148036, -0.0022999, 0.0023670
6: 0.0024838, 0.0052216, 0.0023608, 0.0052310, -0.0027472, 0.0028608
7: -0.0173050, -0.0125339, -0.0173705, -0.0123685, -0.0049365, 0.0048366
8: 0.0030002, 0.0069320, 0.0029482, 0.0070702, -0.0040701, 0.0039838
9: 0.0020267, 0.0081575, 0.0018803, 0.0082170, -0.0061904, 0.0062772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067867
time: 1.20 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068059
time: 1.32 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040844, -0.0024847, -0.0015998, 0.0015998
1: -0.0056184, -0.0030446, -0.0056184, -0.0030446, -0.0025738, 0.0025738
2: 0.9644390, 0.9715982, 0.9644390, 0.9715982, -0.0071592, 0.0071592
3: 0.0229746, 0.0368191, 0.0229746, 0.0368191, -0.0099555, 0.0099555
4: -0.0034933, -0.0002825, -0.0034933, -0.0002825, -0.0032109, 0.0032109
5: 0.0124852, 0.0148039, 0.0124852, 0.0148039, -0.0023187, 0.0023187
6: 0.0024552, 0.0052238, 0.0024552, 0.0052238, -0.0027686, 0.0027686
7: -0.0173202, -0.0124955, -0.0173202, -0.0124955, -0.0048248, 0.0048248
8: 0.0029881, 0.0069641, 0.0029881, 0.0069641, -0.0039760, 0.0039760
9: 0.0019926, 0.0082187, 0.0019926, 0.0082187, -0.0062260, 0.0062260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067802
time: 1.18 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.20 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040864, -0.0024876, -0.0015968, 0.0016017
1: -0.0056184, -0.0030446, -0.0056918, -0.0030476, -0.0025708, 0.0026472
2: 0.9644390, 0.9715982, 0.9644510, 0.9715968, -0.0071578, 0.0071472
3: 0.0229746, 0.0368191, 0.0223248, 0.0368093, -0.0101815, 0.0108476
4: -0.0034933, -0.0002825, -0.0034926, -0.0002877, -0.0032056, 0.0032101
5: 0.0124852, 0.0148039, 0.0124883, 0.0148539, -0.0023687, 0.0023157
6: 0.0024552, 0.0052238, 0.0024600, 0.0052234, -0.0027682, 0.0027638
7: -0.0173202, -0.0124955, -0.0173177, -0.0125019, -0.0048183, 0.0048222
8: 0.0029881, 0.0069641, 0.0029901, 0.0069587, -0.0039706, 0.0039740
9: 0.0019926, 0.0082187, 0.0019983, 0.0084590, -0.0064663, 0.0062203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
time: 1.27 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.49 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025380, -0.0040850, -0.0026329, -0.0014534, 0.0015470
1: -0.0056911, -0.0030990, -0.0056386, -0.0031959, -0.0024952, 0.0025396
2: 0.9646574, 0.9715739, 0.9650460, 0.9715307, -0.0068734, 0.0065280
3: 0.0223306, 0.0366401, 0.0227953, 0.0363215, -0.0102558, 0.0098936
4: -0.0034797, -0.0003776, -0.0034555, -0.0005471, -0.0029326, 0.0030778
5: 0.0125414, 0.0148534, 0.0126416, 0.0148177, -0.0022763, 0.0022118
6: 0.0025423, 0.0052171, 0.0026974, 0.0052052, -0.0026629, 0.0025197
7: -0.0172739, -0.0126126, -0.0171913, -0.0128210, -0.0044529, 0.0045787
8: 0.0030249, 0.0068662, 0.0030904, 0.0066919, -0.0036670, 0.0037758
9: 0.0020963, 0.0084568, 0.0022809, 0.0082850, -0.0061887, 0.0061759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063790
time: 1.34 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.23 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040844, -0.0024269, -0.0016595, 0.0015968
1: -0.0056918, -0.0030476, -0.0056179, -0.0029856, -0.0027062, 0.0025703
2: 0.9644510, 0.9715968, 0.9642025, 0.9716244, -0.0071734, 0.0073944
3: 0.0223248, 0.0368093, 0.0229790, 0.0370131, -0.0105829, 0.0101924
4: -0.0034926, -0.0002877, -0.0035081, -0.0001793, -0.0033133, 0.0032204
5: 0.0124883, 0.0148539, 0.0124242, 0.0148036, -0.0023153, 0.0024297
6: 0.0024600, 0.0052234, 0.0023608, 0.0052310, -0.0027710, 0.0028626
7: -0.0173177, -0.0125019, -0.0173705, -0.0123685, -0.0049491, 0.0048686
8: 0.0029901, 0.0069587, 0.0029482, 0.0070702, -0.0040801, 0.0040105
9: 0.0019983, 0.0084590, 0.0018803, 0.0082170, -0.0062187, 0.0065787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
time: 1.30 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.81 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063790
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063790
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067797, upper bound: 0.0063906
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067977, upper bound: 0.0068119
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067797, upper bound: 0.0063906
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067977, upper bound: 0.0068119
IS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063790, upper bound: 0.0067878
IS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063790, upper bound: 0.0067789
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0068059
IS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067797
IS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
IS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063790
IS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063901
IS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067867
IS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068059
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067802
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063790
IS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025620, -0.0040823, -0.0028050, -0.0012768, 0.0015204
1: -0.0055223, -0.0031235, -0.0055390, -0.0033715, -0.0021507, 0.0024156
2: 0.9647554, 0.9715630, 0.9657504, 0.9714524, -0.0066970, 0.0058126
3: 0.0238252, 0.0365597, 0.0236766, 0.0357439, -0.0083599, 0.0091793
4: -0.0034736, -0.0004204, -0.0034116, -0.0008542, -0.0026194, 0.0029912
5: 0.0125667, 0.0147385, 0.0128232, 0.0147499, -0.0021832, 0.0019154
6: 0.0025815, 0.0052141, 0.0029785, 0.0051836, -0.0026021, 0.0022356
7: -0.0172530, -0.0126652, -0.0170416, -0.0131989, -0.0040542, 0.0043765
8: 0.0030414, 0.0068222, 0.0032091, 0.0063759, -0.0033345, 0.0036131
9: 0.0021429, 0.0079041, 0.0026154, 0.0079591, -0.0058162, 0.0052887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.20 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025100, -0.0040819, -0.0025921, -0.0014898, 0.0015719
1: -0.0055228, -0.0030704, -0.0055221, -0.0031542, -0.0023686, 0.0024517
2: 0.9645426, 0.9715868, 0.9648787, 0.9715494, -0.0070068, 0.0067081
3: 0.0238203, 0.0367342, 0.0238268, 0.0364587, -0.0086593, 0.0095013
4: -0.0034869, -0.0003276, -0.0034659, -0.0004741, -0.0030127, 0.0031383
5: 0.0125119, 0.0147389, 0.0125985, 0.0147384, -0.0022265, 0.0021404
6: 0.0024965, 0.0052206, 0.0026306, 0.0052103, -0.0027138, 0.0025900
7: -0.0172982, -0.0125510, -0.0172268, -0.0127313, -0.0045670, 0.0046758
8: 0.0030055, 0.0069177, 0.0030622, 0.0067669, -0.0037614, 0.0038555
9: 0.0020418, 0.0079059, 0.0022014, 0.0079035, -0.0058617, 0.0057045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067919
time: 1.11 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0068119
time: 1.17 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040839, -0.0025538, -0.0015285, 0.0012789
1: -0.0055390, -0.0033715, -0.0055991, -0.0031151, -0.0024239, 0.0022275
2: 0.9657504, 0.9714524, 0.9647219, 0.9715667, -0.0058163, 0.0067305
3: 0.0236766, 0.0357439, 0.0231453, 0.0365872, -0.0094446, 0.0092554
4: -0.0034116, -0.0008542, -0.0034757, -0.0004058, -0.0030058, 0.0026215
5: 0.0128232, 0.0147499, 0.0125581, 0.0147908, -0.0019676, 0.0021919
6: 0.0029785, 0.0051836, 0.0025681, 0.0052151, -0.0022366, 0.0026155
7: -0.0170416, -0.0131989, -0.0172601, -0.0126472, -0.0043944, 0.0040613
8: 0.0032091, 0.0063759, 0.0030358, 0.0068372, -0.0036281, 0.0033401
9: 0.0026154, 0.0079591, 0.0021270, 0.0081555, -0.0055401, 0.0058321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
time: 1.16 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
time: 1.21 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040839, -0.0025022, -0.0015797, 0.0014919
1: -0.0055221, -0.0031542, -0.0055997, -0.0030624, -0.0024596, 0.0024455
2: 0.9648787, 0.9715494, 0.9645107, 0.9715902, -0.0067115, 0.0070387
3: 0.0238268, 0.0364587, 0.0231401, 0.0367604, -0.0097597, 0.0095966
4: -0.0034659, -0.0004741, -0.0034889, -0.0003137, -0.0031522, 0.0030147
5: 0.0125985, 0.0147384, 0.0125036, 0.0147912, -0.0021927, 0.0022348
6: 0.0026306, 0.0052103, 0.0024838, 0.0052216, -0.0025910, 0.0027265
7: -0.0172268, -0.0127313, -0.0173050, -0.0125339, -0.0046929, 0.0045738
8: 0.0030622, 0.0067669, 0.0030002, 0.0069320, -0.0038698, 0.0037668
9: 0.0022014, 0.0079035, 0.0020267, 0.0081575, -0.0059561, 0.0058768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.08 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.36 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040823, -0.0028050, -0.0012789, 0.0015285
1: -0.0055991, -0.0031151, -0.0055390, -0.0033715, -0.0022275, 0.0024239
2: 0.9647219, 0.9715667, 0.9657504, 0.9714524, -0.0067305, 0.0058163
3: 0.0231453, 0.0365872, 0.0236766, 0.0357439, -0.0092554, 0.0094446
4: -0.0034757, -0.0004058, -0.0034116, -0.0008542, -0.0026215, 0.0030058
5: 0.0125581, 0.0147908, 0.0128232, 0.0147499, -0.0021919, 0.0019676
6: 0.0025681, 0.0052151, 0.0029785, 0.0051836, -0.0026155, 0.0022366
7: -0.0172601, -0.0126472, -0.0170416, -0.0131989, -0.0040613, 0.0043944
8: 0.0030358, 0.0068372, 0.0032091, 0.0063759, -0.0033401, 0.0036281
9: 0.0021270, 0.0081555, 0.0026154, 0.0079591, -0.0058321, 0.0055401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062823
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040844, -0.0027970, -0.0012870, 0.0015306
1: -0.0055991, -0.0031151, -0.0056158, -0.0033633, -0.0022358, 0.0025007
2: 0.9647219, 0.9715667, 0.9657173, 0.9714562, -0.0067343, 0.0058494
3: 0.0231453, 0.0365872, 0.0229975, 0.0357711, -0.0086862, 0.0095118
4: -0.0034757, -0.0004058, -0.0034136, -0.0008398, -0.0026359, 0.0030078
5: 0.0125581, 0.0147908, 0.0128146, 0.0148021, -0.0022441, 0.0019762
6: 0.0025681, 0.0052151, 0.0029653, 0.0051846, -0.0026165, 0.0022498
7: -0.0172601, -0.0126472, -0.0170486, -0.0131811, -0.0040791, 0.0044015
8: 0.0030358, 0.0068372, 0.0032036, 0.0063907, -0.0033550, 0.0036337
9: 0.0021270, 0.0081555, 0.0025997, 0.0082102, -0.0060832, 0.0055559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
time: 1.34 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040819, -0.0025921, -0.0014919, 0.0015797
1: -0.0055997, -0.0030624, -0.0055221, -0.0031542, -0.0024455, 0.0024596
2: 0.9645107, 0.9715902, 0.9648787, 0.9715494, -0.0070387, 0.0067115
3: 0.0231401, 0.0367604, 0.0238268, 0.0364587, -0.0095966, 0.0097597
4: -0.0034889, -0.0003137, -0.0034659, -0.0004741, -0.0030147, 0.0031522
5: 0.0125036, 0.0147912, 0.0125985, 0.0147384, -0.0022348, 0.0021927
6: 0.0024838, 0.0052216, 0.0026306, 0.0052103, -0.0027265, 0.0025910
7: -0.0173050, -0.0125339, -0.0172268, -0.0127313, -0.0045738, 0.0046929
8: 0.0030002, 0.0069320, 0.0030622, 0.0067669, -0.0037668, 0.0038698
9: 0.0020267, 0.0081575, 0.0022014, 0.0079035, -0.0058768, 0.0059561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
time: 1.38 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067977
time: 1.31 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040839, -0.0025848, -0.0014991, 0.0015817
1: -0.0055997, -0.0030624, -0.0055989, -0.0031468, -0.0024528, 0.0025365
2: 0.9645107, 0.9715902, 0.9648491, 0.9715527, -0.0070420, 0.0067412
3: 0.0231401, 0.0367604, 0.0231467, 0.0364830, -0.0089989, 0.0098323
4: -0.0034889, -0.0003137, -0.0034678, -0.0004612, -0.0030276, 0.0031541
5: 0.0125036, 0.0147912, 0.0125908, 0.0147907, -0.0022870, 0.0022004
6: 0.0024838, 0.0052216, 0.0026188, 0.0052112, -0.0027274, 0.0026028
7: -0.0173050, -0.0125339, -0.0172331, -0.0127154, -0.0045896, 0.0046992
8: 0.0030002, 0.0069320, 0.0030572, 0.0067802, -0.0037800, 0.0038748
9: 0.0020267, 0.0081575, 0.0021873, 0.0081550, -0.0061284, 0.0059701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
time: 1.30 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
time: 1.45 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025351, -0.0040823, -0.0028050, -0.0012794, 0.0015472
1: -0.0056177, -0.0030960, -0.0055390, -0.0033715, -0.0022462, 0.0024430
2: 0.9646454, 0.9715753, 0.9657504, 0.9714524, -0.0068070, 0.0058249
3: 0.0229800, 0.0366500, 0.0236766, 0.0357439, -0.0094391, 0.0096449
4: -0.0034805, -0.0003724, -0.0034116, -0.0008542, -0.0026262, 0.0030392
5: 0.0125383, 0.0148035, 0.0128232, 0.0147499, -0.0022116, 0.0019803
6: 0.0025375, 0.0052175, 0.0029785, 0.0051836, -0.0026461, 0.0022389
7: -0.0172764, -0.0126061, -0.0170416, -0.0131989, -0.0040776, 0.0044355
8: 0.0030229, 0.0068716, 0.0032091, 0.0063759, -0.0033530, 0.0036624
9: 0.0020906, 0.0082167, 0.0026154, 0.0079591, -0.0058684, 0.0056013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.18 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.11 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040819, -0.0025921, -0.0014924, 0.0015972
1: -0.0056184, -0.0030446, -0.0055221, -0.0031542, -0.0024642, 0.0024775
2: 0.9644390, 0.9715982, 0.9648787, 0.9715494, -0.0071104, 0.0067195
3: 0.0229746, 0.0368191, 0.0238268, 0.0364587, -0.0099423, 0.0099340
4: -0.0034933, -0.0002825, -0.0034659, -0.0004741, -0.0030192, 0.0031835
5: 0.0124852, 0.0148039, 0.0125985, 0.0147384, -0.0022532, 0.0022054
6: 0.0024552, 0.0052238, 0.0026306, 0.0052103, -0.0027551, 0.0025932
7: -0.0173202, -0.0124955, -0.0172268, -0.0127313, -0.0045890, 0.0047314
8: 0.0029881, 0.0069641, 0.0030622, 0.0067669, -0.0037788, 0.0039019
9: 0.0019926, 0.0082187, 0.0022014, 0.0079035, -0.0059109, 0.0060173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067923
time: 1.08 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0068119
time: 1.19 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025380, -0.0040823, -0.0028050, -0.0012813, 0.0015443
1: -0.0056911, -0.0030990, -0.0055390, -0.0033715, -0.0023196, 0.0024400
2: 0.9646574, 0.9715739, 0.9657504, 0.9714524, -0.0067950, 0.0058236
3: 0.0223306, 0.0366401, 0.0236766, 0.0357439, -0.0101550, 0.0096748
4: -0.0034797, -0.0003776, -0.0034116, -0.0008542, -0.0026255, 0.0030339
5: 0.0125414, 0.0148534, 0.0128232, 0.0147499, -0.0022085, 0.0020302
6: 0.0025423, 0.0052171, 0.0029785, 0.0051836, -0.0026413, 0.0022386
7: -0.0172739, -0.0126126, -0.0170416, -0.0131989, -0.0040750, 0.0044290
8: 0.0030249, 0.0068662, 0.0032091, 0.0063759, -0.0033510, 0.0036570
9: 0.0020963, 0.0084568, 0.0026154, 0.0079591, -0.0058627, 0.0058414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066503, upper bound: 0.0061319
time: 1.30 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
time: 1.28 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040819, -0.0025921, -0.0014943, 0.0015942
1: -0.0056918, -0.0030476, -0.0055221, -0.0031542, -0.0025376, 0.0024745
2: 0.9644510, 0.9715968, 0.9648787, 0.9715494, -0.0070984, 0.0067182
3: 0.0223248, 0.0368093, 0.0238268, 0.0364587, -0.0106502, 0.0099679
4: -0.0034926, -0.0002877, -0.0034659, -0.0004741, -0.0030185, 0.0031782
5: 0.0124883, 0.0148539, 0.0125985, 0.0147384, -0.0022501, 0.0022554
6: 0.0024600, 0.0052234, 0.0026306, 0.0052103, -0.0027503, 0.0025928
7: -0.0173177, -0.0125019, -0.0172268, -0.0127313, -0.0045864, 0.0047249
8: 0.0029901, 0.0069587, 0.0030622, 0.0067669, -0.0037768, 0.0038965
9: 0.0019983, 0.0084590, 0.0022014, 0.0079035, -0.0059052, 0.0062575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067787
time: 1.14 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068119
time: 1.26 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040839, -0.0025538, -0.0015312, 0.0013108
1: -0.0056384, -0.0033390, -0.0055991, -0.0031151, -0.0025232, 0.0022601
2: 0.9656199, 0.9714670, 0.9647219, 0.9715667, -0.0059468, 0.0067451
3: 0.0227976, 0.0358510, 0.0231453, 0.0365872, -0.0107577, 0.0098485
4: -0.0034197, -0.0007973, -0.0034757, -0.0004058, -0.0030139, 0.0026784
5: 0.0127895, 0.0148175, 0.0125581, 0.0147908, -0.0020013, 0.0022595
6: 0.0029264, 0.0051876, 0.0025681, 0.0052151, -0.0022887, 0.0026195
7: -0.0170693, -0.0131288, -0.0172601, -0.0126472, -0.0044222, 0.0041313
8: 0.0031871, 0.0064344, 0.0030358, 0.0068372, -0.0036501, 0.0033987
9: 0.0025534, 0.0082841, 0.0021270, 0.0081555, -0.0056021, 0.0061572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0064564
time: 1.19 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0066920
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040839, -0.0025538, -0.0015331, 0.0013059
1: -0.0057107, -0.0033440, -0.0055991, -0.0031151, -0.0025956, 0.0022551
2: 0.9656399, 0.9714647, 0.9647219, 0.9715667, -0.0059268, 0.0067428
3: 0.0221572, 0.0358347, 0.0231453, 0.0365872, -0.0108165, 0.0092530
4: -0.0034185, -0.0008060, -0.0034757, -0.0004058, -0.0030127, 0.0026697
5: 0.0127946, 0.0148667, 0.0125581, 0.0147908, -0.0019962, 0.0023087
6: 0.0029344, 0.0051870, 0.0025681, 0.0052151, -0.0022808, 0.0026189
7: -0.0170651, -0.0131395, -0.0172601, -0.0126472, -0.0044179, 0.0041207
8: 0.0031905, 0.0064255, 0.0030358, 0.0068372, -0.0036467, 0.0033898
9: 0.0025629, 0.0085209, 0.0021270, 0.0081555, -0.0055927, 0.0063939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0064564
time: 1.09 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0066836
time: 1.27 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024269, -0.0040844, -0.0027970, -0.0012875, 0.0016575
1: -0.0056179, -0.0029856, -0.0056158, -0.0033633, -0.0022546, 0.0026302
2: 0.9642025, 0.9716244, 0.9657173, 0.9714562, -0.0072537, 0.0059071
3: 0.0229790, 0.0370131, 0.0229975, 0.0357711, -0.0097118, 0.0109595
4: -0.0035081, -0.0001793, -0.0034136, -0.0008398, -0.0026683, 0.0032343
5: 0.0124242, 0.0148036, 0.0128146, 0.0148021, -0.0023780, 0.0019889
6: 0.0023608, 0.0052310, 0.0029653, 0.0051846, -0.0028238, 0.0022657
7: -0.0173705, -0.0123685, -0.0170486, -0.0131811, -0.0041894, 0.0046801
8: 0.0029482, 0.0070702, 0.0032036, 0.0063907, -0.0034425, 0.0038667
9: 0.0018803, 0.0082170, 0.0025997, 0.0082102, -0.0063299, 0.0056174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0063901
time: 1.27 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0063901
time: 1.24 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024269, -0.0040839, -0.0025848, -0.0014996, 0.0016570
1: -0.0056179, -0.0029856, -0.0055989, -0.0031468, -0.0024710, 0.0026133
2: 0.9642025, 0.9716244, 0.9648491, 0.9715527, -0.0073502, 0.0067754
3: 0.0229790, 0.0370131, 0.0231467, 0.0364830, -0.0102167, 0.0105747
4: -0.0035081, -0.0001793, -0.0034678, -0.0004612, -0.0030469, 0.0032885
5: 0.0124242, 0.0148036, 0.0125908, 0.0147907, -0.0023665, 0.0022127
6: 0.0023608, 0.0052310, 0.0026188, 0.0052112, -0.0028504, 0.0026122
7: -0.0173705, -0.0123685, -0.0172331, -0.0127154, -0.0046551, 0.0048646
8: 0.0029482, 0.0070702, 0.0030572, 0.0067802, -0.0038320, 0.0040130
9: 0.0018803, 0.0082170, 0.0021873, 0.0081550, -0.0062747, 0.0060297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0063901
time: 1.48 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0067977
time: 1.57 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040844, -0.0025351, -0.0015472, 0.0012794
1: -0.0055390, -0.0033715, -0.0056177, -0.0030960, -0.0024430, 0.0022462
2: 0.9657504, 0.9714524, 0.9646454, 0.9715753, -0.0058249, 0.0068070
3: 0.0236766, 0.0357439, 0.0229800, 0.0366500, -0.0096449, 0.0094391
4: -0.0034116, -0.0008542, -0.0034805, -0.0003724, -0.0030392, 0.0026262
5: 0.0128232, 0.0147499, 0.0125383, 0.0148035, -0.0019803, 0.0022116
6: 0.0029785, 0.0051836, 0.0025375, 0.0052175, -0.0022389, 0.0026461
7: -0.0170416, -0.0131989, -0.0172764, -0.0126061, -0.0044355, 0.0040776
8: 0.0032091, 0.0063759, 0.0030229, 0.0068716, -0.0036624, 0.0033530
9: 0.0026154, 0.0079591, 0.0020906, 0.0082167, -0.0056013, 0.0058684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
time: 1.10 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040844, -0.0024847, -0.0015972, 0.0014924
1: -0.0055221, -0.0031542, -0.0056184, -0.0030446, -0.0024775, 0.0024642
2: 0.9648787, 0.9715494, 0.9644390, 0.9715982, -0.0067195, 0.0071104
3: 0.0238268, 0.0364587, 0.0229746, 0.0368191, -0.0099340, 0.0099423
4: -0.0034659, -0.0004741, -0.0034933, -0.0002825, -0.0031835, 0.0030192
5: 0.0125985, 0.0147384, 0.0124852, 0.0148039, -0.0022054, 0.0022532
6: 0.0026306, 0.0052103, 0.0024552, 0.0052238, -0.0025932, 0.0027551
7: -0.0172268, -0.0127313, -0.0173202, -0.0124955, -0.0047314, 0.0045890
8: 0.0030622, 0.0067669, 0.0029881, 0.0069641, -0.0039019, 0.0037788
9: 0.0022014, 0.0079035, 0.0019926, 0.0082187, -0.0060173, 0.0059109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0063906
time: 1.22 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0068119
time: 1.22 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040864, -0.0025380, -0.0015443, 0.0012813
1: -0.0055390, -0.0033715, -0.0056911, -0.0030990, -0.0024400, 0.0023196
2: 0.9657504, 0.9714524, 0.9646574, 0.9715739, -0.0058236, 0.0067950
3: 0.0236766, 0.0357439, 0.0223306, 0.0366401, -0.0096748, 0.0101550
4: -0.0034116, -0.0008542, -0.0034797, -0.0003776, -0.0030339, 0.0026255
5: 0.0128232, 0.0147499, 0.0125414, 0.0148534, -0.0020302, 0.0022085
6: 0.0029785, 0.0051836, 0.0025423, 0.0052171, -0.0022386, 0.0026413
7: -0.0170416, -0.0131989, -0.0172739, -0.0126126, -0.0044290, 0.0040750
8: 0.0032091, 0.0063759, 0.0030249, 0.0068662, -0.0036570, 0.0033510
9: 0.0026154, 0.0079591, 0.0020963, 0.0084568, -0.0058414, 0.0058627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
time: 1.04 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040864, -0.0024876, -0.0015942, 0.0014943
1: -0.0055221, -0.0031542, -0.0056918, -0.0030476, -0.0024745, 0.0025376
2: 0.9648787, 0.9715494, 0.9644510, 0.9715968, -0.0067182, 0.0070984
3: 0.0238268, 0.0364587, 0.0223248, 0.0368093, -0.0099679, 0.0106502
4: -0.0034659, -0.0004741, -0.0034926, -0.0002877, -0.0031782, 0.0030185
5: 0.0125985, 0.0147384, 0.0124883, 0.0148539, -0.0022554, 0.0022501
6: 0.0026306, 0.0052103, 0.0024600, 0.0052234, -0.0025928, 0.0027503
7: -0.0172268, -0.0127313, -0.0173177, -0.0125019, -0.0047249, 0.0045864
8: 0.0030622, 0.0067669, 0.0029901, 0.0069587, -0.0038965, 0.0037768
9: 0.0022014, 0.0079035, 0.0019983, 0.0084590, -0.0062575, 0.0059052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0063901
time: 1.18 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0068059
time: 1.26 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040850, -0.0027732, -0.0013108, 0.0015312
1: -0.0055991, -0.0031151, -0.0056384, -0.0033390, -0.0022601, 0.0025232
2: 0.9647219, 0.9715667, 0.9656199, 0.9714670, -0.0067451, 0.0059468
3: 0.0231453, 0.0365872, 0.0227976, 0.0358510, -0.0098485, 0.0107577
4: -0.0034757, -0.0004058, -0.0034197, -0.0007973, -0.0026784, 0.0030139
5: 0.0125581, 0.0147908, 0.0127895, 0.0148175, -0.0022595, 0.0020013
6: 0.0025681, 0.0052151, 0.0029264, 0.0051876, -0.0026195, 0.0022887
7: -0.0172601, -0.0126472, -0.0170693, -0.0131288, -0.0041313, 0.0044222
8: 0.0030358, 0.0068372, 0.0031871, 0.0064344, -0.0033987, 0.0036501
9: 0.0021270, 0.0081555, 0.0025534, 0.0082841, -0.0061572, 0.0056021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062822
time: 1.31 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062822
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040869, -0.0027780, -0.0013059, 0.0015331
1: -0.0055991, -0.0031151, -0.0057107, -0.0033440, -0.0022551, 0.0025956
2: 0.9647219, 0.9715667, 0.9656399, 0.9714647, -0.0067428, 0.0059268
3: 0.0231453, 0.0365872, 0.0221572, 0.0358347, -0.0092530, 0.0108165
4: -0.0034757, -0.0004058, -0.0034185, -0.0008060, -0.0026697, 0.0030127
5: 0.0125581, 0.0147908, 0.0127946, 0.0148667, -0.0023087, 0.0019962
6: 0.0025681, 0.0052151, 0.0029344, 0.0051870, -0.0026189, 0.0022808
7: -0.0172601, -0.0126472, -0.0170651, -0.0131395, -0.0041207, 0.0044179
8: 0.0030358, 0.0068372, 0.0031905, 0.0064255, -0.0033898, 0.0036467
9: 0.0021270, 0.0081555, 0.0025629, 0.0085209, -0.0063939, 0.0055927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
time: 1.43 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062919
time: 1.15 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040844, -0.0024269, -0.0016575, 0.0012875
1: -0.0056158, -0.0033633, -0.0056179, -0.0029856, -0.0026302, 0.0022546
2: 0.9657173, 0.9714562, 0.9642025, 0.9716244, -0.0059071, 0.0072537
3: 0.0229975, 0.0357711, 0.0229790, 0.0370131, -0.0109595, 0.0097118
4: -0.0034136, -0.0008398, -0.0035081, -0.0001793, -0.0032343, 0.0026683
5: 0.0128146, 0.0148021, 0.0124242, 0.0148036, -0.0019889, 0.0023780
6: 0.0029653, 0.0051846, 0.0023608, 0.0052310, -0.0022657, 0.0028238
7: -0.0170486, -0.0131811, -0.0173705, -0.0123685, -0.0046801, 0.0041894
8: 0.0032036, 0.0063907, 0.0029482, 0.0070702, -0.0038667, 0.0034425
9: 0.0025997, 0.0082102, 0.0018803, 0.0082170, -0.0056174, 0.0063299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
time: 1.14 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
time: 1.15 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040844, -0.0024269, -0.0016570, 0.0014996
1: -0.0055989, -0.0031468, -0.0056179, -0.0029856, -0.0026133, 0.0024710
2: 0.9648491, 0.9715527, 0.9642025, 0.9716244, -0.0067754, 0.0073502
3: 0.0231467, 0.0364830, 0.0229790, 0.0370131, -0.0105747, 0.0102167
4: -0.0034678, -0.0004612, -0.0035081, -0.0001793, -0.0032885, 0.0030469
5: 0.0125908, 0.0147907, 0.0124242, 0.0148036, -0.0022127, 0.0023665
6: 0.0026188, 0.0052112, 0.0023608, 0.0052310, -0.0026122, 0.0028504
7: -0.0172331, -0.0127154, -0.0173705, -0.0123685, -0.0048646, 0.0046551
8: 0.0030572, 0.0067802, 0.0029482, 0.0070702, -0.0040130, 0.0038320
9: 0.0021873, 0.0081550, 0.0018803, 0.0082170, -0.0060297, 0.0062747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067977
time: 1.38 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068059
time: 1.44 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040844, -0.0025351, -0.0015499, 0.0013113
1: -0.0056384, -0.0033390, -0.0056177, -0.0030960, -0.0025423, 0.0022787
2: 0.9656199, 0.9714670, 0.9646454, 0.9715753, -0.0059554, 0.0068216
3: 0.0227976, 0.0358510, 0.0229800, 0.0366500, -0.0096497, 0.0088241
4: -0.0034197, -0.0007973, -0.0034805, -0.0003724, -0.0030473, 0.0026831
5: 0.0127895, 0.0148175, 0.0125383, 0.0148035, -0.0020140, 0.0022792
6: 0.0029264, 0.0051876, 0.0025375, 0.0052175, -0.0022910, 0.0026501
7: -0.0170693, -0.0131288, -0.0172764, -0.0126061, -0.0044632, 0.0041476
8: 0.0031871, 0.0064344, 0.0030229, 0.0068716, -0.0036844, 0.0034116
9: 0.0025534, 0.0082841, 0.0020906, 0.0082167, -0.0056632, 0.0061935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
time: 1.07 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
time: 1.14 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040844, -0.0024847, -0.0015997, 0.0015140
1: -0.0056175, -0.0031321, -0.0056184, -0.0030446, -0.0025729, 0.0024863
2: 0.9647900, 0.9715592, 0.9644390, 0.9715982, -0.0068082, 0.0071202
3: 0.0229819, 0.0365314, 0.0229746, 0.0368191, -0.0099458, 0.0091076
4: -0.0034715, -0.0004355, -0.0034933, -0.0002825, -0.0031890, 0.0030578
5: 0.0125756, 0.0148033, 0.0124852, 0.0148039, -0.0022283, 0.0023182
6: 0.0025953, 0.0052130, 0.0024552, 0.0052238, -0.0026285, 0.0027578
7: -0.0172457, -0.0126837, -0.0173202, -0.0124955, -0.0047502, 0.0046365
8: 0.0030472, 0.0068067, 0.0029881, 0.0069641, -0.0039169, 0.0038186
9: 0.0021593, 0.0082159, 0.0019926, 0.0082187, -0.0060594, 0.0062233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
time: 1.18 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040864, -0.0025380, -0.0015470, 0.0013132
1: -0.0056384, -0.0033390, -0.0056911, -0.0030990, -0.0025393, 0.0023521
2: 0.9656199, 0.9714670, 0.9646574, 0.9715739, -0.0059541, 0.0068097
3: 0.0227976, 0.0358510, 0.0223306, 0.0366401, -0.0098803, 0.0097152
4: -0.0034197, -0.0007973, -0.0034797, -0.0003776, -0.0030421, 0.0026824
5: 0.0127895, 0.0148175, 0.0125414, 0.0148534, -0.0020639, 0.0022761
6: 0.0029264, 0.0051876, 0.0025423, 0.0052171, -0.0022907, 0.0026453
7: -0.0170693, -0.0131288, -0.0172739, -0.0126126, -0.0044568, 0.0041450
8: 0.0031871, 0.0064344, 0.0030249, 0.0068662, -0.0036790, 0.0034095
9: 0.0025534, 0.0082841, 0.0020963, 0.0084568, -0.0059034, 0.0061878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
time: 1.11 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
time: 1.33 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040864, -0.0024876, -0.0015968, 0.0015160
1: -0.0056175, -0.0031321, -0.0056918, -0.0030476, -0.0025700, 0.0025597
2: 0.9647900, 0.9715592, 0.9644510, 0.9715968, -0.0068068, 0.0071082
3: 0.0229819, 0.0365314, 0.0223248, 0.0368093, -0.0101718, 0.0100312
4: -0.0034715, -0.0004355, -0.0034926, -0.0002877, -0.0031837, 0.0030571
5: 0.0125756, 0.0148033, 0.0124883, 0.0148539, -0.0022782, 0.0023151
6: 0.0025953, 0.0052130, 0.0024600, 0.0052234, -0.0026281, 0.0027530
7: -0.0172457, -0.0126837, -0.0173177, -0.0125019, -0.0047437, 0.0046339
8: 0.0030472, 0.0068067, 0.0029901, 0.0069587, -0.0039115, 0.0038165
9: 0.0021593, 0.0082159, 0.0019983, 0.0084590, -0.0062996, 0.0062176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.30 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.34 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025380, -0.0040850, -0.0027732, -0.0013132, 0.0015470
1: -0.0056911, -0.0030990, -0.0056384, -0.0033390, -0.0023521, 0.0025393
2: 0.9646574, 0.9715739, 0.9656199, 0.9714670, -0.0068097, 0.0059541
3: 0.0223306, 0.0366401, 0.0227976, 0.0358510, -0.0097152, 0.0098803
4: -0.0034797, -0.0003776, -0.0034197, -0.0007973, -0.0026824, 0.0030421
5: 0.0125414, 0.0148534, 0.0127895, 0.0148175, -0.0022761, 0.0020639
6: 0.0025423, 0.0052171, 0.0029264, 0.0051876, -0.0026453, 0.0022907
7: -0.0172739, -0.0126126, -0.0170693, -0.0131288, -0.0041450, 0.0044568
8: 0.0030249, 0.0068662, 0.0031871, 0.0064344, -0.0034095, 0.0036790
9: 0.0020963, 0.0084568, 0.0025534, 0.0082841, -0.0061878, 0.0059034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
time: 1.11 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062823
time: 1.31 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025380, -0.0040869, -0.0027780, -0.0013084, 0.0015489
1: -0.0056911, -0.0030990, -0.0057107, -0.0033440, -0.0023471, 0.0026117
2: 0.9646574, 0.9715739, 0.9656399, 0.9714647, -0.0068073, 0.0059341
3: 0.0223306, 0.0366401, 0.0221572, 0.0358347, -0.0091138, 0.0099443
4: -0.0034797, -0.0003776, -0.0034185, -0.0008060, -0.0026737, 0.0030408
5: 0.0125414, 0.0148534, 0.0127946, 0.0148667, -0.0023253, 0.0020588
6: 0.0025423, 0.0052171, 0.0029344, 0.0051870, -0.0026447, 0.0022827
7: -0.0172739, -0.0126126, -0.0170651, -0.0131395, -0.0041344, 0.0044525
8: 0.0030249, 0.0068662, 0.0031905, 0.0064255, -0.0034006, 0.0036757
9: 0.0020963, 0.0084568, 0.0025629, 0.0085209, -0.0064246, 0.0058939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.31 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062919
time: 1.34 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040844, -0.0025704, -0.0015160, 0.0015968
1: -0.0056918, -0.0030476, -0.0056175, -0.0031321, -0.0025597, 0.0025700
2: 0.9644510, 0.9715968, 0.9647900, 0.9715592, -0.0071082, 0.0068068
3: 0.0223248, 0.0368093, 0.0229819, 0.0365314, -0.0100312, 0.0101718
4: -0.0034926, -0.0002877, -0.0034715, -0.0004355, -0.0030571, 0.0031837
5: 0.0124883, 0.0148539, 0.0125756, 0.0148033, -0.0023151, 0.0022782
6: 0.0024600, 0.0052234, 0.0025953, 0.0052130, -0.0027530, 0.0026281
7: -0.0173177, -0.0125019, -0.0172457, -0.0126837, -0.0046339, 0.0047437
8: 0.0029901, 0.0069587, 0.0030472, 0.0068067, -0.0038165, 0.0039115
9: 0.0019983, 0.0084590, 0.0021593, 0.0082159, -0.0062176, 0.0062996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067734
time: 1.13 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063790
time: 1.25 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040864, -0.0025727, -0.0015137, 0.0015987
1: -0.0056918, -0.0030476, -0.0056909, -0.0031344, -0.0025573, 0.0026433
2: 0.9644510, 0.9715968, 0.9647995, 0.9715582, -0.0071071, 0.0067974
3: 0.0223248, 0.0368093, 0.0223324, 0.0365237, -0.0093988, 0.0102391
4: -0.0034926, -0.0002877, -0.0034709, -0.0004396, -0.0030530, 0.0031832
5: 0.0124883, 0.0148539, 0.0125780, 0.0148533, -0.0023650, 0.0022758
6: 0.0024600, 0.0052234, 0.0025990, 0.0052127, -0.0027527, 0.0026244
7: -0.0173177, -0.0125019, -0.0172437, -0.0126887, -0.0046289, 0.0047418
8: 0.0029901, 0.0069587, 0.0030488, 0.0068025, -0.0038124, 0.0039099
9: 0.0019983, 0.0084590, 0.0021638, 0.0084561, -0.0064578, 0.0062952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
time: 1.21 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068059
time: 1.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.74 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067919
IS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0068119
IS_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
IS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
IS_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062823
IS_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
IS_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
IS_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067977
IS_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
IS_B1_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
IS_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067923
IS_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0068119
IS_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066503, upper bound: 0.0061319
IS_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067787
IS_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068119
IS_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0064564
IS_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0066920
IS_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0064564
IS_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062823, upper bound: 0.0066836
IS_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0063901
IS_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0063901
IS_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0063901
IS_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067715, upper bound: 0.0067977
IS_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
IS_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
IS_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0063906
IS_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0068119
IS_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
IS_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0063901
IS_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0068059
IS_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062822
IS_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062822
IS_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
IS_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062919
IS_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
IS_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
IS_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067977
IS_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068059
IS_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
IS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
IS_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
IS_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
IS_B2_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_B2_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
IS_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062823
IS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062919
IS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067734
IS_B2_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063790
IS_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
IS_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0068059

## BFS IS instance: IS_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027495, -0.0040823, -0.0028285, -0.0012533, 0.0013328
1: -0.0055181, -0.0033149, -0.0055390, -0.0033955, -0.0021227, 0.0022241
2: 0.9655231, 0.9714777, 0.9658464, 0.9714418, -0.0059187, 0.0056314
3: 0.0238616, 0.0359304, 0.0236768, 0.0356653, -0.0082055, 0.0085609
4: -0.0034257, -0.0007551, -0.0034056, -0.0008960, -0.0025297, 0.0026505
5: 0.0127646, 0.0147357, 0.0128479, 0.0147499, -0.0019854, 0.0018879
6: 0.0028878, 0.0051906, 0.0030168, 0.0051807, -0.0022929, 0.0021738
7: -0.0170899, -0.0130769, -0.0170212, -0.0132503, -0.0038396, 0.0039443
8: 0.0031708, 0.0064779, 0.0032253, 0.0063329, -0.0031621, 0.0032526
9: 0.0025074, 0.0078907, 0.0026610, 0.0079590, -0.0054515, 0.0052297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.04 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.45 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026449, -0.0040823, -0.0028050, -0.0012768, 0.0014374
1: -0.0055220, -0.0032081, -0.0055390, -0.0033715, -0.0021505, 0.0023309
2: 0.9650951, 0.9715253, 0.9657504, 0.9714524, -0.0063573, 0.0057749
3: 0.0238274, 0.0362813, 0.0236766, 0.0357439, -0.0083540, 0.0086574
4: -0.0034524, -0.0005685, -0.0034116, -0.0008542, -0.0025982, 0.0028431
5: 0.0126542, 0.0147384, 0.0128232, 0.0147499, -0.0020957, 0.0019152
6: 0.0027170, 0.0052037, 0.0029785, 0.0051836, -0.0024666, 0.0022252
7: -0.0171809, -0.0128473, -0.0170416, -0.0131989, -0.0039820, 0.0041943
8: 0.0030987, 0.0066699, 0.0032091, 0.0063759, -0.0032772, 0.0034607
9: 0.0023042, 0.0079033, 0.0026154, 0.0079591, -0.0056549, 0.0052879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
time: 0.99 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
time: 1.12 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040819, -0.0025921, -0.0014902, 0.0012768
1: -0.0055390, -0.0033715, -0.0055221, -0.0031542, -0.0023848, 0.0021505
2: 0.9657504, 0.9714524, 0.9648787, 0.9715494, -0.0057990, 0.0065737
3: 0.0236766, 0.0357439, 0.0238268, 0.0364587, -0.0092283, 0.0083571
4: -0.0034116, -0.0008542, -0.0034659, -0.0004741, -0.0029374, 0.0026117
5: 0.0128232, 0.0147499, 0.0125985, 0.0147384, -0.0019152, 0.0021515
6: 0.0029785, 0.0051836, 0.0026306, 0.0052103, -0.0022318, 0.0025530
7: -0.0170416, -0.0131989, -0.0172268, -0.0127313, -0.0043104, 0.0040280
8: 0.0032091, 0.0063759, 0.0030622, 0.0067669, -0.0035578, 0.0033137
9: 0.0026154, 0.0079591, 0.0022014, 0.0079035, -0.0052881, 0.0057576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040819, -0.0025921, -0.0014898, 0.0014898
1: -0.0055221, -0.0031542, -0.0055221, -0.0031542, -0.0023679, 0.0023679
2: 0.9648787, 0.9715494, 0.9648787, 0.9715494, -0.0066707, 0.0066707
3: 0.0238268, 0.0364587, 0.0238268, 0.0364587, -0.0086499, 0.0086499
4: -0.0034659, -0.0004741, -0.0034659, -0.0004741, -0.0029918, 0.0029918
5: 0.0125985, 0.0147384, 0.0125985, 0.0147384, -0.0021399, 0.0021399
6: 0.0026306, 0.0052103, 0.0026306, 0.0052103, -0.0025797, 0.0025797
7: -0.0172268, -0.0127313, -0.0172268, -0.0127313, -0.0044956, 0.0044956
8: 0.0030622, 0.0067669, 0.0030622, 0.0067669, -0.0037047, 0.0037047
9: 0.0022014, 0.0079035, 0.0022014, 0.0079035, -0.0057021, 0.0057021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065425
time: 1.16 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
time: 1.14 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040839, -0.0025781, -0.0015041, 0.0010917
1: -0.0055329, -0.0035626, -0.0055990, -0.0031399, -0.0023930, 0.0020365
2: 0.9665166, 0.9713672, 0.9648213, 0.9715557, -0.0050392, 0.0065460
3: 0.0237312, 0.0351158, 0.0231456, 0.0365057, -0.0092349, 0.0086294
4: -0.0033638, -0.0011883, -0.0034695, -0.0004491, -0.0029147, 0.0022812
5: 0.0130206, 0.0147457, 0.0125837, 0.0147908, -0.0017701, 0.0021621
6: 0.0032842, 0.0051601, 0.0026077, 0.0052121, -0.0019278, 0.0025524
7: -0.0168788, -0.0136098, -0.0172390, -0.0127005, -0.0041783, 0.0036293
8: 0.0033383, 0.0060323, 0.0030525, 0.0067927, -0.0034544, 0.0029797
9: 0.0029792, 0.0079389, 0.0021742, 0.0081554, -0.0051762, 0.0057647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062919
time: 1.17 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
time: 1.26 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040839, -0.0025538, -0.0015285, 0.0011976
1: -0.0055389, -0.0034545, -0.0055991, -0.0031151, -0.0024238, 0.0021446
2: 0.9660832, 0.9714154, 0.9647219, 0.9715667, -0.0054834, 0.0066935
3: 0.0236782, 0.0354711, 0.0231453, 0.0365872, -0.0094400, 0.0086858
4: -0.0033908, -0.0009993, -0.0034757, -0.0004058, -0.0029850, 0.0024764
5: 0.0129089, 0.0147498, 0.0125581, 0.0147908, -0.0018819, 0.0021918
6: 0.0031113, 0.0051734, 0.0025681, 0.0052151, -0.0021038, 0.0026053
7: -0.0169709, -0.0133773, -0.0172601, -0.0126472, -0.0043237, 0.0038828
8: 0.0032652, 0.0062266, 0.0030358, 0.0068372, -0.0035720, 0.0031909
9: 0.0027734, 0.0079585, 0.0021270, 0.0081555, -0.0053821, 0.0058315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0064619
time: 1.34 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0066947
time: 1.45 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040844, -0.0027970, -0.0012849, 0.0014923
1: -0.0055221, -0.0031542, -0.0056158, -0.0033633, -0.0021588, 0.0024616
2: 0.9648787, 0.9715494, 0.9657173, 0.9714562, -0.0065775, 0.0058321
3: 0.0238268, 0.0364587, 0.0229975, 0.0357711, -0.0086216, 0.0101542
4: -0.0034659, -0.0004741, -0.0034136, -0.0008398, -0.0026261, 0.0029395
5: 0.0125985, 0.0147384, 0.0128146, 0.0148021, -0.0022037, 0.0019238
6: 0.0026306, 0.0052103, 0.0029653, 0.0051846, -0.0025540, 0.0022450
7: -0.0172268, -0.0127313, -0.0170486, -0.0131811, -0.0040457, 0.0043174
8: 0.0030622, 0.0067669, 0.0032036, 0.0063907, -0.0033285, 0.0035634
9: 0.0022014, 0.0079035, 0.0025997, 0.0082102, -0.0060088, 0.0053038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
time: 1.24 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
time: 1.27 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040839, -0.0025848, -0.0014970, 0.0014918
1: -0.0055221, -0.0031542, -0.0055989, -0.0031468, -0.0023753, 0.0024447
2: 0.9648787, 0.9715494, 0.9648491, 0.9715527, -0.0066740, 0.0067003
3: 0.0238268, 0.0364587, 0.0231467, 0.0364830, -0.0089258, 0.0095858
4: -0.0034659, -0.0004741, -0.0034678, -0.0004612, -0.0030047, 0.0029936
5: 0.0125985, 0.0147384, 0.0125908, 0.0147907, -0.0021922, 0.0021476
6: 0.0026306, 0.0052103, 0.0026188, 0.0052112, -0.0025806, 0.0025915
7: -0.0172268, -0.0127313, -0.0172331, -0.0127154, -0.0045114, 0.0045019
8: 0.0030622, 0.0067669, 0.0030572, 0.0067802, -0.0037180, 0.0037097
9: 0.0022014, 0.0079035, 0.0021873, 0.0081550, -0.0059536, 0.0057162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
time: 1.39 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
time: 1.43 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025781, -0.0040822, -0.0029922, -0.0010917, 0.0015041
1: -0.0055990, -0.0031399, -0.0055329, -0.0035626, -0.0020365, 0.0023930
2: 0.9648213, 0.9715557, 0.9665166, 0.9713672, -0.0065460, 0.0050392
3: 0.0231456, 0.0365057, 0.0237312, 0.0351158, -0.0086294, 0.0092349
4: -0.0034695, -0.0004491, -0.0033638, -0.0011883, -0.0022812, 0.0029147
5: 0.0125837, 0.0147908, 0.0130206, 0.0147457, -0.0021621, 0.0017701
6: 0.0026077, 0.0052121, 0.0032842, 0.0051601, -0.0025524, 0.0019278
7: -0.0172390, -0.0127005, -0.0168788, -0.0136098, -0.0036293, 0.0041783
8: 0.0030525, 0.0067927, 0.0033383, 0.0060323, -0.0029797, 0.0034544
9: 0.0021742, 0.0081554, 0.0029792, 0.0079389, -0.0057647, 0.0051762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
time: 1.12 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
time: 1.17 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025538, -0.0040823, -0.0028863, -0.0011976, 0.0015285
1: -0.0055991, -0.0031151, -0.0055389, -0.0034545, -0.0021446, 0.0024238
2: 0.9647219, 0.9715667, 0.9660832, 0.9714154, -0.0066935, 0.0054834
3: 0.0231453, 0.0365872, 0.0236782, 0.0354711, -0.0086858, 0.0094400
4: -0.0034757, -0.0004058, -0.0033908, -0.0009993, -0.0024764, 0.0029850
5: 0.0125581, 0.0147908, 0.0129089, 0.0147498, -0.0021918, 0.0018819
6: 0.0025681, 0.0052151, 0.0031113, 0.0051734, -0.0026053, 0.0021038
7: -0.0172601, -0.0126472, -0.0169709, -0.0133773, -0.0038828, 0.0043237
8: 0.0030358, 0.0068372, 0.0032652, 0.0062266, -0.0031909, 0.0035720
9: 0.0021270, 0.0081555, 0.0027734, 0.0079585, -0.0058315, 0.0053821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062910
time: 1.36 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062922
time: 1.47 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027483, -0.0040844, -0.0028212, -0.0012625, 0.0013361
1: -0.0055916, -0.0033136, -0.0056157, -0.0033881, -0.0022035, 0.0023021
2: 0.9655182, 0.9714783, 0.9658166, 0.9714451, -0.0059268, 0.0056617
3: 0.0232114, 0.0359343, 0.0229978, 0.0356897, -0.0085206, 0.0088704
4: -0.0034260, -0.0007530, -0.0034074, -0.0008831, -0.0025430, 0.0026545
5: 0.0127633, 0.0147857, 0.0128402, 0.0148021, -0.0020388, 0.0019455
6: 0.0028858, 0.0051907, 0.0030049, 0.0051816, -0.0022957, 0.0021858
7: -0.0170910, -0.0130743, -0.0170275, -0.0132344, -0.0038566, 0.0039532
8: 0.0031700, 0.0064800, 0.0032203, 0.0063462, -0.0031762, 0.0032598
9: 0.0025051, 0.0081311, 0.0026468, 0.0082101, -0.0057050, 0.0054843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
time: 1.36 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062919
time: 1.39 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026342, -0.0040844, -0.0027970, -0.0012870, 0.0014502
1: -0.0055988, -0.0031972, -0.0056158, -0.0033633, -0.0022355, 0.0024186
2: 0.9650511, 0.9715303, 0.9657173, 0.9714562, -0.0064051, 0.0058129
3: 0.0231474, 0.0363173, 0.0229975, 0.0357711, -0.0086804, 0.0089866
4: -0.0034552, -0.0005493, -0.0034136, -0.0008398, -0.0026154, 0.0028643
5: 0.0126429, 0.0147906, 0.0128146, 0.0148021, -0.0021592, 0.0019760
6: 0.0026995, 0.0052050, 0.0029653, 0.0051846, -0.0024852, 0.0022397
7: -0.0171902, -0.0128237, -0.0170486, -0.0131811, -0.0040091, 0.0042249
8: 0.0030913, 0.0066896, 0.0032036, 0.0063907, -0.0032995, 0.0034860
9: 0.0022833, 0.0081547, 0.0025997, 0.0082102, -0.0059269, 0.0055551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 1.29 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040819, -0.0025921, -0.0014923, 0.0012849
1: -0.0056158, -0.0033633, -0.0055221, -0.0031542, -0.0024616, 0.0021588
2: 0.9657173, 0.9714562, 0.9648787, 0.9715494, -0.0058321, 0.0065775
3: 0.0229975, 0.0357711, 0.0238268, 0.0364587, -0.0101542, 0.0086216
4: -0.0034136, -0.0008398, -0.0034659, -0.0004741, -0.0029395, 0.0026261
5: 0.0128146, 0.0148021, 0.0125985, 0.0147384, -0.0019238, 0.0022037
6: 0.0029653, 0.0051846, 0.0026306, 0.0052103, -0.0022450, 0.0025540
7: -0.0170486, -0.0131811, -0.0172268, -0.0127313, -0.0043174, 0.0040457
8: 0.0032036, 0.0063907, 0.0030622, 0.0067669, -0.0035634, 0.0033285
9: 0.0025997, 0.0082102, 0.0022014, 0.0079035, -0.0053038, 0.0060088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064953
time: 1.22 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066793
time: 1.17 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040819, -0.0025921, -0.0014918, 0.0014970
1: -0.0055989, -0.0031468, -0.0055221, -0.0031542, -0.0024447, 0.0023753
2: 0.9648491, 0.9715527, 0.9648787, 0.9715494, -0.0067003, 0.0066740
3: 0.0231467, 0.0364830, 0.0238268, 0.0364587, -0.0095858, 0.0089258
4: -0.0034678, -0.0004612, -0.0034659, -0.0004741, -0.0029936, 0.0030047
5: 0.0125908, 0.0147907, 0.0125985, 0.0147384, -0.0021476, 0.0021922
6: 0.0026188, 0.0052112, 0.0026306, 0.0052103, -0.0025915, 0.0025806
7: -0.0172331, -0.0127154, -0.0172268, -0.0127313, -0.0045019, 0.0045114
8: 0.0030572, 0.0067802, 0.0030622, 0.0067669, -0.0037097, 0.0037180
9: 0.0021873, 0.0081550, 0.0022014, 0.0079035, -0.0057162, 0.0059536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065425
time: 1.26 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067141
time: 1.18 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040839, -0.0025848, -0.0014995, 0.0012870
1: -0.0056158, -0.0033633, -0.0055989, -0.0031468, -0.0024690, 0.0022356
2: 0.9657173, 0.9714562, 0.9648491, 0.9715527, -0.0058354, 0.0066071
3: 0.0229975, 0.0357711, 0.0231467, 0.0364830, -0.0095562, 0.0086835
4: -0.0034136, -0.0008398, -0.0034678, -0.0004612, -0.0029524, 0.0026280
5: 0.0128146, 0.0148021, 0.0125908, 0.0147907, -0.0019760, 0.0022113
6: 0.0029653, 0.0051846, 0.0026188, 0.0052112, -0.0022459, 0.0025658
7: -0.0170486, -0.0131811, -0.0172331, -0.0127154, -0.0043333, 0.0040520
8: 0.0032036, 0.0063907, 0.0030572, 0.0067802, -0.0035766, 0.0033335
9: 0.0025997, 0.0082102, 0.0021873, 0.0081550, -0.0055553, 0.0060228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
time: 1.37 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
time: 1.21 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027180, -0.0040823, -0.0028285, -0.0012559, 0.0013643
1: -0.0056161, -0.0032827, -0.0055390, -0.0033955, -0.0022206, 0.0022563
2: 0.9653942, 0.9714921, 0.9658464, 0.9714418, -0.0060476, 0.0056457
3: 0.0229948, 0.0360360, 0.0236768, 0.0356653, -0.0094091, 0.0090813
4: -0.0034338, -0.0006989, -0.0034056, -0.0008960, -0.0025377, 0.0027067
5: 0.0127314, 0.0148024, 0.0128479, 0.0147499, -0.0020186, 0.0019545
6: 0.0028364, 0.0051945, 0.0030168, 0.0051807, -0.0023443, 0.0021777
7: -0.0171173, -0.0130078, -0.0170212, -0.0132503, -0.0038670, 0.0040134
8: 0.0031491, 0.0065357, 0.0032253, 0.0063329, -0.0031838, 0.0033104
9: 0.0024462, 0.0082112, 0.0026610, 0.0079590, -0.0055127, 0.0055502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.19 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0026219, -0.0040823, -0.0028050, -0.0012794, 0.0014604
1: -0.0056175, -0.0031846, -0.0055390, -0.0033715, -0.0022460, 0.0023544
2: 0.9650006, 0.9715359, 0.9657504, 0.9714524, -0.0064518, 0.0057855
3: 0.0229821, 0.0363587, 0.0236766, 0.0357439, -0.0094343, 0.0091621
4: -0.0034583, -0.0005273, -0.0034116, -0.0008542, -0.0026041, 0.0028843
5: 0.0126299, 0.0148033, 0.0128232, 0.0147499, -0.0021200, 0.0019802
6: 0.0026793, 0.0052066, 0.0029785, 0.0051836, -0.0025043, 0.0022281
7: -0.0172009, -0.0127967, -0.0170416, -0.0131989, -0.0040021, 0.0042449
8: 0.0030827, 0.0067122, 0.0032091, 0.0063759, -0.0032932, 0.0035031
9: 0.0022593, 0.0082159, 0.0026154, 0.0079591, -0.0056997, 0.0056005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0061319
time: 1.16 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0062922
time: 1.32 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040819, -0.0025921, -0.0014929, 0.0013087
1: -0.0056384, -0.0033390, -0.0055221, -0.0031542, -0.0024842, 0.0021831
2: 0.9656199, 0.9714670, 0.9648787, 0.9715494, -0.0059295, 0.0065883
3: 0.0227976, 0.0358510, 0.0238268, 0.0364587, -0.0105415, 0.0089502
4: -0.0034197, -0.0007973, -0.0034659, -0.0004741, -0.0029456, 0.0026686
5: 0.0127895, 0.0148175, 0.0125985, 0.0147384, -0.0019489, 0.0022190
6: 0.0029264, 0.0051876, 0.0026306, 0.0052103, -0.0022839, 0.0025570
7: -0.0170693, -0.0131288, -0.0172268, -0.0127313, -0.0043381, 0.0040980
8: 0.0031871, 0.0064344, 0.0030622, 0.0067669, -0.0035798, 0.0033722
9: 0.0025534, 0.0082841, 0.0022014, 0.0079035, -0.0053501, 0.0060827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
time: 1.14 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066939
time: 1.25 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040819, -0.0025921, -0.0014923, 0.0015115
1: -0.0056175, -0.0031321, -0.0055221, -0.0031542, -0.0024633, 0.0023900
2: 0.9647900, 0.9715592, 0.9648787, 0.9715494, -0.0067593, 0.0066805
3: 0.0229819, 0.0365314, 0.0238268, 0.0364587, -0.0099307, 0.0091776
4: -0.0034715, -0.0004355, -0.0034659, -0.0004741, -0.0029973, 0.0030304
5: 0.0125756, 0.0148033, 0.0125985, 0.0147384, -0.0021628, 0.0022049
6: 0.0025953, 0.0052130, 0.0026306, 0.0052103, -0.0026150, 0.0025824
7: -0.0172457, -0.0126837, -0.0172268, -0.0127313, -0.0045144, 0.0045431
8: 0.0030472, 0.0068067, 0.0030622, 0.0067669, -0.0037197, 0.0037445
9: 0.0021593, 0.0082159, 0.0022014, 0.0079035, -0.0057442, 0.0060145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065425
time: 1.24 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0067141
time: 1.40 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025615, -0.0040822, -0.0029922, -0.0010942, 0.0015207
1: -0.0056911, -0.0031229, -0.0055329, -0.0035626, -0.0021285, 0.0024099
2: 0.9647532, 0.9715633, 0.9665166, 0.9713672, -0.0066140, 0.0050468
3: 0.0223310, 0.0365615, 0.0237312, 0.0351158, -0.0095294, 0.0094741
4: -0.0034737, -0.0004195, -0.0033638, -0.0011883, -0.0022855, 0.0029443
5: 0.0125662, 0.0148534, 0.0130206, 0.0147457, -0.0021796, 0.0018328
6: 0.0025806, 0.0052142, 0.0032842, 0.0051601, -0.0025795, 0.0019299
7: -0.0172535, -0.0126640, -0.0168788, -0.0136098, -0.0036437, 0.0042148
8: 0.0030411, 0.0068232, 0.0033383, 0.0060323, -0.0029912, 0.0034849
9: 0.0021419, 0.0084567, 0.0029792, 0.0079389, -0.0057970, 0.0054774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
time: 1.12 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025380, -0.0040823, -0.0028863, -0.0012001, 0.0015443
1: -0.0056911, -0.0030990, -0.0055389, -0.0034545, -0.0022366, 0.0024398
2: 0.9646574, 0.9715739, 0.9660832, 0.9714154, -0.0067580, 0.0054907
3: 0.0223306, 0.0366401, 0.0236782, 0.0354711, -0.0097082, 0.0096702
4: -0.0034797, -0.0003776, -0.0033908, -0.0009993, -0.0024804, 0.0030132
5: 0.0125414, 0.0148534, 0.0129089, 0.0147498, -0.0022084, 0.0019445
6: 0.0025423, 0.0052171, 0.0031113, 0.0051734, -0.0026311, 0.0021058
7: -0.0172739, -0.0126126, -0.0169709, -0.0133773, -0.0038965, 0.0043583
8: 0.0030249, 0.0068662, 0.0032652, 0.0062266, -0.0032018, 0.0036009
9: 0.0020963, 0.0084568, 0.0027734, 0.0079585, -0.0058622, 0.0056834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
time: 1.13 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
time: 1.44 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040819, -0.0025921, -0.0014948, 0.0013039
1: -0.0057107, -0.0033440, -0.0055221, -0.0031542, -0.0025565, 0.0021781
2: 0.9656399, 0.9714647, 0.9648787, 0.9715494, -0.0059095, 0.0065860
3: 0.0221572, 0.0358347, 0.0238268, 0.0364587, -0.0112409, 0.0089770
4: -0.0034185, -0.0008060, -0.0034659, -0.0004741, -0.0029443, 0.0026599
5: 0.0127946, 0.0148667, 0.0125985, 0.0147384, -0.0019438, 0.0022683
6: 0.0029344, 0.0051870, 0.0026306, 0.0052103, -0.0022760, 0.0025563
7: -0.0170651, -0.0131395, -0.0172268, -0.0127313, -0.0043339, 0.0040873
8: 0.0031905, 0.0064255, 0.0030622, 0.0067669, -0.0035764, 0.0033633
9: 0.0025629, 0.0085209, 0.0022014, 0.0079035, -0.0053406, 0.0063195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
time: 1.15 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040819, -0.0025921, -0.0014943, 0.0015092
1: -0.0056909, -0.0031344, -0.0055221, -0.0031542, -0.0025367, 0.0023876
2: 0.9647995, 0.9715582, 0.9648787, 0.9715494, -0.0067499, 0.0066795
3: 0.0223324, 0.0365237, 0.0238268, 0.0364587, -0.0106390, 0.0092100
4: -0.0034709, -0.0004396, -0.0034659, -0.0004741, -0.0029967, 0.0030263
5: 0.0125780, 0.0148533, 0.0125985, 0.0147384, -0.0021604, 0.0022548
6: 0.0025990, 0.0052127, 0.0026306, 0.0052103, -0.0026113, 0.0025821
7: -0.0172437, -0.0126887, -0.0172268, -0.0127313, -0.0045124, 0.0045381
8: 0.0030488, 0.0068025, 0.0030622, 0.0067669, -0.0037181, 0.0037403
9: 0.0021638, 0.0084561, 0.0022014, 0.0079035, -0.0057397, 0.0062547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065425
time: 1.33 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067141
time: 1.27 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027957, -0.0040837, -0.0027483, -0.0013367, 0.0012880
1: -0.0056383, -0.0033620, -0.0055916, -0.0033136, -0.0023247, 0.0022296
2: 0.9657123, 0.9714566, 0.9655182, 0.9714783, -0.0057660, 0.0059384
3: 0.0227978, 0.0357753, 0.0232114, 0.0359343, -0.0100532, 0.0096069
4: -0.0034139, -0.0008376, -0.0034260, -0.0007530, -0.0026610, 0.0025885
5: 0.0128133, 0.0148175, 0.0127633, 0.0147857, -0.0019724, 0.0020542
6: 0.0029633, 0.0051848, 0.0028858, 0.0051907, -0.0022274, 0.0022989
7: -0.0170497, -0.0131783, -0.0170910, -0.0130743, -0.0039754, 0.0039126
8: 0.0032027, 0.0063930, 0.0031700, 0.0064800, -0.0032774, 0.0032230
9: 0.0025972, 0.0082840, 0.0025051, 0.0081311, -0.0055338, 0.0057789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0060624
time: 1.15 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0064624
time: 1.45 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040839, -0.0026342, -0.0014508, 0.0013108
1: -0.0056384, -0.0033390, -0.0055988, -0.0031972, -0.0024412, 0.0022598
2: 0.9656199, 0.9714670, 0.9650511, 0.9715303, -0.0059104, 0.0064159
3: 0.0227976, 0.0358510, 0.0231474, 0.0363173, -0.0103757, 0.0098428
4: -0.0034197, -0.0007973, -0.0034552, -0.0005493, -0.0028704, 0.0026578
5: 0.0127895, 0.0148175, 0.0126429, 0.0147906, -0.0020011, 0.0021746
6: 0.0029264, 0.0051876, 0.0026995, 0.0052050, -0.0022786, 0.0024881
7: -0.0170693, -0.0131288, -0.0171902, -0.0128237, -0.0042456, 0.0040614
8: 0.0031871, 0.0064344, 0.0030913, 0.0066896, -0.0035024, 0.0033432
9: 0.0025534, 0.0082841, 0.0022833, 0.0081547, -0.0056013, 0.0060008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062919
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
time: 1.21 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0028018, -0.0040837, -0.0027483, -0.0013386, 0.0012820
1: -0.0057107, -0.0033682, -0.0055916, -0.0033136, -0.0023970, 0.0022234
2: 0.9657370, 0.9714540, 0.9655182, 0.9714783, -0.0057414, 0.0059357
3: 0.0221575, 0.0357550, 0.0232114, 0.0359343, -0.0101750, 0.0091025
4: -0.0034124, -0.0008484, -0.0034260, -0.0007530, -0.0026594, 0.0025777
5: 0.0128197, 0.0148667, 0.0127633, 0.0147857, -0.0019660, 0.0021034
6: 0.0029732, 0.0051840, 0.0028858, 0.0051907, -0.0022175, 0.0022982
7: -0.0170445, -0.0131916, -0.0170910, -0.0130743, -0.0039702, 0.0038993
8: 0.0032069, 0.0063819, 0.0031700, 0.0064800, -0.0032732, 0.0032119
9: 0.0026090, 0.0085208, 0.0025051, 0.0081311, -0.0055221, 0.0060157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0060624
time: 1.13 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064564
time: 1.17 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040839, -0.0026342, -0.0014527, 0.0013059
1: -0.0057107, -0.0033440, -0.0055988, -0.0031972, -0.0025135, 0.0022549
2: 0.9656399, 0.9714647, 0.9650511, 0.9715303, -0.0058904, 0.0064136
3: 0.0221572, 0.0358347, 0.0231474, 0.0363173, -0.0104838, 0.0092472
4: -0.0034185, -0.0008060, -0.0034552, -0.0005493, -0.0028691, 0.0026492
5: 0.0127946, 0.0148667, 0.0126429, 0.0147906, -0.0019960, 0.0022238
6: 0.0029344, 0.0051870, 0.0026995, 0.0052050, -0.0022707, 0.0024875
7: -0.0170651, -0.0131395, -0.0171902, -0.0128237, -0.0042414, 0.0040507
8: 0.0031905, 0.0064255, 0.0030913, 0.0066896, -0.0034991, 0.0033343
9: 0.0025629, 0.0085209, 0.0022833, 0.0081547, -0.0055919, 0.0062376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 1.07 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
time: 1.14 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040844, -0.0027970, -0.0012875, 0.0015140
1: -0.0056175, -0.0031321, -0.0056158, -0.0033633, -0.0022542, 0.0024837
2: 0.9647900, 0.9715592, 0.9657173, 0.9714562, -0.0066661, 0.0058419
3: 0.0229819, 0.0365314, 0.0229975, 0.0357711, -0.0097006, 0.0104912
4: -0.0034715, -0.0004355, -0.0034136, -0.0008398, -0.0026317, 0.0029781
5: 0.0125756, 0.0148033, 0.0128146, 0.0148021, -0.0022265, 0.0019887
6: 0.0025953, 0.0052130, 0.0029653, 0.0051846, -0.0025893, 0.0022477
7: -0.0172457, -0.0126837, -0.0170486, -0.0131811, -0.0040646, 0.0043649
8: 0.0030472, 0.0068067, 0.0032036, 0.0063907, -0.0033435, 0.0036031
9: 0.0021593, 0.0082159, 0.0025997, 0.0082102, -0.0060509, 0.0056163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.38 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066758, upper bound: 0.0062919
time: 1.12 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040844, -0.0027970, -0.0012894, 0.0015117
1: -0.0056909, -0.0031344, -0.0056158, -0.0033633, -0.0023276, 0.0024813
2: 0.9647995, 0.9715582, 0.9657173, 0.9714562, -0.0066567, 0.0058408
3: 0.0223324, 0.0365237, 0.0229975, 0.0357711, -0.0097578, 0.0098732
4: -0.0034709, -0.0004396, -0.0034136, -0.0008398, -0.0026311, 0.0029740
5: 0.0125780, 0.0148533, 0.0128146, 0.0148021, -0.0022241, 0.0020386
6: 0.0025990, 0.0052127, 0.0029653, 0.0051846, -0.0025856, 0.0022474
7: -0.0172437, -0.0126887, -0.0170486, -0.0131811, -0.0040626, 0.0043599
8: 0.0030488, 0.0068025, 0.0032036, 0.0063907, -0.0033419, 0.0035989
9: 0.0021638, 0.0084561, 0.0025997, 0.0082102, -0.0060464, 0.0058565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.27 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066758, upper bound: 0.0062919
time: 1.33 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040839, -0.0025848, -0.0014996, 0.0015135
1: -0.0056175, -0.0031321, -0.0055989, -0.0031468, -0.0024707, 0.0024668
2: 0.9647900, 0.9715592, 0.9648491, 0.9715527, -0.0067626, 0.0067102
3: 0.0229819, 0.0365314, 0.0231467, 0.0364830, -0.0102067, 0.0101135
4: -0.0034715, -0.0004355, -0.0034678, -0.0004612, -0.0030102, 0.0030323
5: 0.0125756, 0.0148033, 0.0125908, 0.0147907, -0.0022150, 0.0022125
6: 0.0025953, 0.0052130, 0.0026188, 0.0052112, -0.0026159, 0.0025942
7: -0.0172457, -0.0126837, -0.0172331, -0.0127154, -0.0045303, 0.0045494
8: 0.0030472, 0.0068067, 0.0030572, 0.0067802, -0.0037329, 0.0037495
9: 0.0021593, 0.0082159, 0.0021873, 0.0081550, -0.0059957, 0.0060286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067099
time: 1.32 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0067099
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040839, -0.0025848, -0.0015015, 0.0015112
1: -0.0056909, -0.0031344, -0.0055989, -0.0031468, -0.0025441, 0.0024645
2: 0.9647995, 0.9715582, 0.9648491, 0.9715527, -0.0067532, 0.0067091
3: 0.0223324, 0.0365237, 0.0231467, 0.0364830, -0.0102620, 0.0094906
4: -0.0034709, -0.0004396, -0.0034678, -0.0004612, -0.0030096, 0.0030282
5: 0.0125780, 0.0148533, 0.0125908, 0.0147907, -0.0022126, 0.0022624
6: 0.0025990, 0.0052127, 0.0026188, 0.0052112, -0.0026122, 0.0025939
7: -0.0172437, -0.0126887, -0.0172331, -0.0127154, -0.0045283, 0.0045444
8: 0.0030488, 0.0068025, 0.0030572, 0.0067802, -0.0037314, 0.0037453
9: 0.0021638, 0.0084561, 0.0021873, 0.0081550, -0.0059912, 0.0062688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0064738
time: 1.48 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0067099
time: 1.46 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028285, -0.0040844, -0.0027180, -0.0013643, 0.0012559
1: -0.0055390, -0.0033955, -0.0056161, -0.0032827, -0.0022563, 0.0022206
2: 0.9658464, 0.9714418, 0.9653942, 0.9714921, -0.0056457, 0.0060476
3: 0.0236768, 0.0356653, 0.0229948, 0.0360360, -0.0090813, 0.0094091
4: -0.0034056, -0.0008960, -0.0034338, -0.0006989, -0.0027067, 0.0025377
5: 0.0128479, 0.0147499, 0.0127314, 0.0148024, -0.0019545, 0.0020186
6: 0.0030168, 0.0051807, 0.0028364, 0.0051945, -0.0021777, 0.0023443
7: -0.0170212, -0.0132503, -0.0171173, -0.0130078, -0.0040134, 0.0038670
8: 0.0032253, 0.0063329, 0.0031491, 0.0065357, -0.0033104, 0.0031838
9: 0.0026610, 0.0079590, 0.0024462, 0.0082112, -0.0055502, 0.0055127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
time: 1.18 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
time: 1.20 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040844, -0.0026219, -0.0014604, 0.0012794
1: -0.0055390, -0.0033715, -0.0056175, -0.0031846, -0.0023544, 0.0022460
2: 0.9657504, 0.9714524, 0.9650006, 0.9715359, -0.0057855, 0.0064518
3: 0.0236766, 0.0357439, 0.0229821, 0.0363587, -0.0091621, 0.0094343
4: -0.0034116, -0.0008542, -0.0034583, -0.0005273, -0.0028843, 0.0026041
5: 0.0128232, 0.0147499, 0.0126299, 0.0148033, -0.0019802, 0.0021200
6: 0.0029785, 0.0051836, 0.0026793, 0.0052066, -0.0022281, 0.0025043
7: -0.0170416, -0.0131989, -0.0172009, -0.0127967, -0.0042449, 0.0040021
8: 0.0032091, 0.0063759, 0.0030827, 0.0067122, -0.0035031, 0.0032932
9: 0.0026154, 0.0079591, 0.0022593, 0.0082159, -0.0056005, 0.0056997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.08 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
time: 1.14 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040850, -0.0027732, -0.0013087, 0.0014929
1: -0.0055221, -0.0031542, -0.0056384, -0.0033390, -0.0021831, 0.0024842
2: 0.9648787, 0.9715494, 0.9656199, 0.9714670, -0.0065883, 0.0059295
3: 0.0238268, 0.0364587, 0.0227976, 0.0358510, -0.0089503, 0.0105415
4: -0.0034659, -0.0004741, -0.0034197, -0.0007973, -0.0026686, 0.0029456
5: 0.0125985, 0.0147384, 0.0127895, 0.0148175, -0.0022190, 0.0019489
6: 0.0026306, 0.0052103, 0.0029264, 0.0051876, -0.0025570, 0.0022839
7: -0.0172268, -0.0127313, -0.0170693, -0.0131288, -0.0040980, 0.0043381
8: 0.0030622, 0.0067669, 0.0031871, 0.0064344, -0.0033722, 0.0035798
9: 0.0022014, 0.0079035, 0.0025534, 0.0082841, -0.0060827, 0.0053501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0062922
time: 1.32 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066939, upper bound: 0.0062922
time: 1.16 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040844, -0.0025704, -0.0015115, 0.0014923
1: -0.0055221, -0.0031542, -0.0056175, -0.0031321, -0.0023900, 0.0024633
2: 0.9648787, 0.9715494, 0.9647900, 0.9715592, -0.0066805, 0.0067593
3: 0.0238268, 0.0364587, 0.0229819, 0.0365314, -0.0091776, 0.0099307
4: -0.0034659, -0.0004741, -0.0034715, -0.0004355, -0.0030304, 0.0029973
5: 0.0125985, 0.0147384, 0.0125756, 0.0148033, -0.0022049, 0.0021628
6: 0.0026306, 0.0052103, 0.0025953, 0.0052130, -0.0025824, 0.0026150
7: -0.0172268, -0.0127313, -0.0172457, -0.0126837, -0.0045431, 0.0045144
8: 0.0030622, 0.0067669, 0.0030472, 0.0068067, -0.0037445, 0.0037197
9: 0.0022014, 0.0079035, 0.0021593, 0.0082159, -0.0060145, 0.0057442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0067141
time: 1.58 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066940, upper bound: 0.0067141
time: 1.29 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040864, -0.0025615, -0.0015207, 0.0010942
1: -0.0055329, -0.0035626, -0.0056911, -0.0031229, -0.0024099, 0.0021285
2: 0.9665166, 0.9713672, 0.9647532, 0.9715633, -0.0050468, 0.0066140
3: 0.0237312, 0.0351158, 0.0223310, 0.0365615, -0.0094741, 0.0095294
4: -0.0033638, -0.0011883, -0.0034737, -0.0004195, -0.0029443, 0.0022855
5: 0.0130206, 0.0147457, 0.0125662, 0.0148534, -0.0018328, 0.0021796
6: 0.0032842, 0.0051601, 0.0025806, 0.0052142, -0.0019299, 0.0025795
7: -0.0168788, -0.0136098, -0.0172535, -0.0126640, -0.0042148, 0.0036437
8: 0.0033383, 0.0060323, 0.0030411, 0.0068232, -0.0034849, 0.0029912
9: 0.0029792, 0.0079389, 0.0021419, 0.0084567, -0.0054774, 0.0057970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062919
time: 1.11 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
time: 1.18 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040864, -0.0025380, -0.0015443, 0.0012001
1: -0.0055389, -0.0034545, -0.0056911, -0.0030990, -0.0024398, 0.0022366
2: 0.9660832, 0.9714154, 0.9646574, 0.9715739, -0.0054907, 0.0067580
3: 0.0236782, 0.0354711, 0.0223306, 0.0366401, -0.0096702, 0.0097082
4: -0.0033908, -0.0009993, -0.0034797, -0.0003776, -0.0030132, 0.0024804
5: 0.0129089, 0.0147498, 0.0125414, 0.0148534, -0.0019445, 0.0022084
6: 0.0031113, 0.0051734, 0.0025423, 0.0052171, -0.0021058, 0.0026311
7: -0.0169709, -0.0133773, -0.0172739, -0.0126126, -0.0043583, 0.0038965
8: 0.0032652, 0.0062266, 0.0030249, 0.0068662, -0.0036009, 0.0032018
9: 0.0027734, 0.0079585, 0.0020963, 0.0084568, -0.0056834, 0.0058622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062919
time: 1.09 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
time: 1.13 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040869, -0.0027780, -0.0013039, 0.0014948
1: -0.0055221, -0.0031542, -0.0057107, -0.0033440, -0.0021781, 0.0025565
2: 0.9648787, 0.9715494, 0.9656399, 0.9714647, -0.0065860, 0.0059095
3: 0.0238268, 0.0364587, 0.0221572, 0.0358347, -0.0089770, 0.0112409
4: -0.0034659, -0.0004741, -0.0034185, -0.0008060, -0.0026599, 0.0029443
5: 0.0125985, 0.0147384, 0.0127946, 0.0148667, -0.0022683, 0.0019438
6: 0.0026306, 0.0052103, 0.0029344, 0.0051870, -0.0025563, 0.0022760
7: -0.0172268, -0.0127313, -0.0170651, -0.0131395, -0.0040873, 0.0043339
8: 0.0030622, 0.0067669, 0.0031905, 0.0064255, -0.0033633, 0.0035764
9: 0.0022014, 0.0079035, 0.0025629, 0.0085209, -0.0063195, 0.0053406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0062919
time: 1.22 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0062919
time: 1.21 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040864, -0.0025727, -0.0015092, 0.0014943
1: -0.0055221, -0.0031542, -0.0056909, -0.0031344, -0.0023876, 0.0025367
2: 0.9648787, 0.9715494, 0.9647995, 0.9715582, -0.0066795, 0.0067499
3: 0.0238268, 0.0364587, 0.0223324, 0.0365237, -0.0092100, 0.0106390
4: -0.0034659, -0.0004741, -0.0034709, -0.0004396, -0.0030263, 0.0029967
5: 0.0125985, 0.0147384, 0.0125780, 0.0148533, -0.0022548, 0.0021604
6: 0.0026306, 0.0052103, 0.0025990, 0.0052127, -0.0025821, 0.0026113
7: -0.0172268, -0.0127313, -0.0172437, -0.0126887, -0.0045381, 0.0045124
8: 0.0030622, 0.0067669, 0.0030488, 0.0068025, -0.0037403, 0.0037181
9: 0.0022014, 0.0079035, 0.0021638, 0.0084561, -0.0062547, 0.0057397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0067099
time: 1.39 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0067099
time: 1.34 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027483, -0.0040850, -0.0027957, -0.0012880, 0.0013367
1: -0.0055916, -0.0033136, -0.0056383, -0.0033620, -0.0022296, 0.0023247
2: 0.9655182, 0.9714783, 0.9657123, 0.9714566, -0.0059384, 0.0057660
3: 0.0232114, 0.0359343, 0.0227978, 0.0357753, -0.0096069, 0.0100532
4: -0.0034260, -0.0007530, -0.0034139, -0.0008376, -0.0025885, 0.0026610
5: 0.0127633, 0.0147857, 0.0128133, 0.0148175, -0.0020542, 0.0019724
6: 0.0028858, 0.0051907, 0.0029633, 0.0051848, -0.0022989, 0.0022274
7: -0.0170910, -0.0130743, -0.0170497, -0.0131783, -0.0039126, 0.0039754
8: 0.0031700, 0.0064800, 0.0032027, 0.0063930, -0.0032230, 0.0032774
9: 0.0025051, 0.0081311, 0.0025972, 0.0082840, -0.0057789, 0.0055338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
time: 1.10 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
time: 1.28 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026342, -0.0040850, -0.0027732, -0.0013108, 0.0014508
1: -0.0055988, -0.0031972, -0.0056384, -0.0033390, -0.0022598, 0.0024412
2: 0.9650511, 0.9715303, 0.9656199, 0.9714670, -0.0064159, 0.0059104
3: 0.0231474, 0.0363173, 0.0227976, 0.0358510, -0.0098428, 0.0103757
4: -0.0034552, -0.0005493, -0.0034197, -0.0007973, -0.0026578, 0.0028704
5: 0.0126429, 0.0147906, 0.0127895, 0.0148175, -0.0021746, 0.0020011
6: 0.0026995, 0.0052050, 0.0029264, 0.0051876, -0.0024881, 0.0022786
7: -0.0171902, -0.0128237, -0.0170693, -0.0131288, -0.0040614, 0.0042456
8: 0.0030913, 0.0066896, 0.0031871, 0.0064344, -0.0033432, 0.0035024
9: 0.0022833, 0.0081547, 0.0025534, 0.0082841, -0.0060008, 0.0056013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
time: 1.18 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
time: 1.14 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027483, -0.0040869, -0.0028018, -0.0012820, 0.0013386
1: -0.0055916, -0.0033136, -0.0057107, -0.0033682, -0.0022234, 0.0023970
2: 0.9655182, 0.9714783, 0.9657370, 0.9714540, -0.0059357, 0.0057414
3: 0.0232114, 0.0359343, 0.0221575, 0.0357550, -0.0091025, 0.0101750
4: -0.0034260, -0.0007530, -0.0034124, -0.0008484, -0.0025777, 0.0026594
5: 0.0127633, 0.0147857, 0.0128197, 0.0148667, -0.0021034, 0.0019660
6: 0.0028858, 0.0051907, 0.0029732, 0.0051840, -0.0022982, 0.0022175
7: -0.0170910, -0.0130743, -0.0170445, -0.0131916, -0.0038993, 0.0039702
8: 0.0031700, 0.0064800, 0.0032069, 0.0063819, -0.0032119, 0.0032732
9: 0.0025051, 0.0081311, 0.0026090, 0.0085208, -0.0060157, 0.0055221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
time: 1.14 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062919
time: 1.12 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026342, -0.0040869, -0.0027780, -0.0013059, 0.0014527
1: -0.0055988, -0.0031972, -0.0057107, -0.0033440, -0.0022549, 0.0025135
2: 0.9650511, 0.9715303, 0.9656399, 0.9714647, -0.0064136, 0.0058904
3: 0.0231474, 0.0363173, 0.0221572, 0.0358347, -0.0092472, 0.0104838
4: -0.0034552, -0.0005493, -0.0034185, -0.0008060, -0.0026492, 0.0028691
5: 0.0126429, 0.0147906, 0.0127946, 0.0148667, -0.0022238, 0.0019960
6: 0.0026995, 0.0052050, 0.0029344, 0.0051870, -0.0024875, 0.0022707
7: -0.0171902, -0.0128237, -0.0170651, -0.0131395, -0.0040507, 0.0042414
8: 0.0030913, 0.0066896, 0.0031905, 0.0064255, -0.0033343, 0.0034991
9: 0.0022833, 0.0081547, 0.0025629, 0.0085209, -0.0062376, 0.0055919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A2_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 0.99 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 1.30 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040844, -0.0025704, -0.0015140, 0.0012875
1: -0.0056158, -0.0033633, -0.0056175, -0.0031321, -0.0024837, 0.0022542
2: 0.9657173, 0.9714562, 0.9647900, 0.9715592, -0.0058419, 0.0066661
3: 0.0229975, 0.0357711, 0.0229819, 0.0365314, -0.0104912, 0.0097006
4: -0.0034136, -0.0008398, -0.0034715, -0.0004355, -0.0029781, 0.0026317
5: 0.0128146, 0.0148021, 0.0125756, 0.0148033, -0.0019887, 0.0022265
6: 0.0029653, 0.0051846, 0.0025953, 0.0052130, -0.0022477, 0.0025893
7: -0.0170486, -0.0131811, -0.0172457, -0.0126837, -0.0043649, 0.0040646
8: 0.0032036, 0.0063907, 0.0030472, 0.0068067, -0.0036031, 0.0033435
9: 0.0025997, 0.0082102, 0.0021593, 0.0082159, -0.0056163, 0.0060509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
time: 1.30 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066758
time: 1.11 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040864, -0.0025727, -0.0015117, 0.0012894
1: -0.0056158, -0.0033633, -0.0056909, -0.0031344, -0.0024813, 0.0023276
2: 0.9657173, 0.9714562, 0.9647995, 0.9715582, -0.0058408, 0.0066567
3: 0.0229975, 0.0357711, 0.0223324, 0.0365237, -0.0098732, 0.0097578
4: -0.0034136, -0.0008398, -0.0034709, -0.0004396, -0.0029740, 0.0026311
5: 0.0128146, 0.0148021, 0.0125780, 0.0148533, -0.0020386, 0.0022241
6: 0.0029653, 0.0051846, 0.0025990, 0.0052127, -0.0022474, 0.0025856
7: -0.0170486, -0.0131811, -0.0172437, -0.0126887, -0.0043599, 0.0040626
8: 0.0032036, 0.0063907, 0.0030488, 0.0068025, -0.0035989, 0.0033419
9: 0.0025997, 0.0082102, 0.0021638, 0.0084561, -0.0058565, 0.0060464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
time: 1.19 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066811
time: 1.31 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040844, -0.0025704, -0.0015135, 0.0014996
1: -0.0055989, -0.0031468, -0.0056175, -0.0031321, -0.0024668, 0.0024707
2: 0.9648491, 0.9715527, 0.9647900, 0.9715592, -0.0067102, 0.0067626
3: 0.0231467, 0.0364830, 0.0229819, 0.0365314, -0.0101135, 0.0102067
4: -0.0034678, -0.0004612, -0.0034715, -0.0004355, -0.0030323, 0.0030102
5: 0.0125908, 0.0147907, 0.0125756, 0.0148033, -0.0022125, 0.0022150
6: 0.0026188, 0.0052112, 0.0025953, 0.0052130, -0.0025942, 0.0026159
7: -0.0172331, -0.0127154, -0.0172457, -0.0126837, -0.0045494, 0.0045303
8: 0.0030572, 0.0067802, 0.0030472, 0.0068067, -0.0037495, 0.0037329
9: 0.0021873, 0.0081550, 0.0021593, 0.0082159, -0.0060286, 0.0059957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0064738
time: 1.29 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067024
time: 1.20 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040864, -0.0025727, -0.0015112, 0.0015015
1: -0.0055989, -0.0031468, -0.0056909, -0.0031344, -0.0024645, 0.0025441
2: 0.9648491, 0.9715527, 0.9647995, 0.9715582, -0.0067091, 0.0067532
3: 0.0231467, 0.0364830, 0.0223324, 0.0365237, -0.0094906, 0.0102620
4: -0.0034678, -0.0004612, -0.0034709, -0.0004396, -0.0030282, 0.0030096
5: 0.0125908, 0.0147907, 0.0125780, 0.0148533, -0.0022624, 0.0022126
6: 0.0026188, 0.0052112, 0.0025990, 0.0052127, -0.0025939, 0.0026122
7: -0.0172331, -0.0127154, -0.0172437, -0.0126887, -0.0045444, 0.0045283
8: 0.0030572, 0.0067802, 0.0030488, 0.0068025, -0.0037453, 0.0037314
9: 0.0021873, 0.0081550, 0.0021638, 0.0084561, -0.0062688, 0.0059912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067051
time: 1.23 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067099
time: 1.22 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027957, -0.0040844, -0.0027180, -0.0013669, 0.0012887
1: -0.0056383, -0.0033620, -0.0056161, -0.0032827, -0.0023556, 0.0022540
2: 0.9657123, 0.9714566, 0.9653942, 0.9714921, -0.0057797, 0.0060624
3: 0.0227978, 0.0357753, 0.0229948, 0.0360360, -0.0090373, 0.0086861
4: -0.0034139, -0.0008376, -0.0034338, -0.0006989, -0.0027150, 0.0025962
5: 0.0128133, 0.0148175, 0.0127314, 0.0148024, -0.0019891, 0.0020861
6: 0.0029633, 0.0051848, 0.0028364, 0.0051945, -0.0022312, 0.0023484
7: -0.0170497, -0.0131783, -0.0171173, -0.0130078, -0.0040419, 0.0039390
8: 0.0032027, 0.0063930, 0.0031491, 0.0065357, -0.0033330, 0.0032439
9: 0.0025972, 0.0082840, 0.0024462, 0.0082112, -0.0056139, 0.0058378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
time: 1.11 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
time: 1.34 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040844, -0.0026219, -0.0014631, 0.0013113
1: -0.0056384, -0.0033390, -0.0056175, -0.0031846, -0.0024537, 0.0022785
2: 0.9656199, 0.9714670, 0.9650006, 0.9715359, -0.0059160, 0.0064664
3: 0.0227976, 0.0358510, 0.0229821, 0.0363587, -0.0091161, 0.0088180
4: -0.0034197, -0.0007973, -0.0034583, -0.0005273, -0.0028924, 0.0026610
5: 0.0127895, 0.0148175, 0.0126299, 0.0148033, -0.0020138, 0.0021876
6: 0.0029264, 0.0051876, 0.0026793, 0.0052066, -0.0022801, 0.0025083
7: -0.0170693, -0.0131288, -0.0172009, -0.0127967, -0.0042727, 0.0040721
8: 0.0031871, 0.0064344, 0.0030827, 0.0067122, -0.0035251, 0.0033517
9: 0.0025534, 0.0082841, 0.0022593, 0.0082159, -0.0056625, 0.0060248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
time: 1.32 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
time: 1.36 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040850, -0.0027732, -0.0013113, 0.0015146
1: -0.0056175, -0.0031321, -0.0056384, -0.0033390, -0.0022785, 0.0025062
2: 0.9647900, 0.9715592, 0.9656199, 0.9714670, -0.0066770, 0.0059394
3: 0.0229819, 0.0365314, 0.0227976, 0.0358510, -0.0088210, 0.0096794
4: -0.0034715, -0.0004355, -0.0034197, -0.0007973, -0.0026741, 0.0029842
5: 0.0125756, 0.0148033, 0.0127895, 0.0148175, -0.0022419, 0.0020138
6: 0.0025953, 0.0052130, 0.0029264, 0.0051876, -0.0025923, 0.0022866
7: -0.0172457, -0.0126837, -0.0170693, -0.0131288, -0.0041168, 0.0043856
8: 0.0030472, 0.0068067, 0.0031871, 0.0064344, -0.0033872, 0.0036195
9: 0.0021593, 0.0082159, 0.0025534, 0.0082841, -0.0061248, 0.0056625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.34 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.26 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040844, -0.0025704, -0.0015140, 0.0015140
1: -0.0056175, -0.0031321, -0.0056175, -0.0031321, -0.0024854, 0.0024854
2: 0.9647900, 0.9715592, 0.9647900, 0.9715592, -0.0067692, 0.0067692
3: 0.0229819, 0.0365314, 0.0229819, 0.0365314, -0.0090983, 0.0090983
4: -0.0034715, -0.0004355, -0.0034715, -0.0004355, -0.0030359, 0.0030359
5: 0.0125756, 0.0148033, 0.0125756, 0.0148033, -0.0022277, 0.0022277
6: 0.0025953, 0.0052130, 0.0025953, 0.0052130, -0.0026178, 0.0026178
7: -0.0172457, -0.0126837, -0.0172457, -0.0126837, -0.0045619, 0.0045619
8: 0.0030472, 0.0068067, 0.0030472, 0.0068067, -0.0037594, 0.0037594
9: 0.0021593, 0.0082159, 0.0021593, 0.0082159, -0.0060566, 0.0060566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
time: 1.53 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.41 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040849, -0.0029543, -0.0040864, -0.0025615, -0.0015235, 0.0011321
1: -0.0056366, -0.0035239, -0.0056911, -0.0031229, -0.0025137, 0.0021671
2: 0.9663616, 0.9713846, 0.9647532, 0.9715633, -0.0052017, 0.0066314
3: 0.0228131, 0.0352429, 0.0223310, 0.0365615, -0.0096943, 0.0091004
4: -0.0033735, -0.0011207, -0.0034737, -0.0004195, -0.0029540, 0.0023531
5: 0.0129807, 0.0148163, 0.0125662, 0.0148534, -0.0018727, 0.0022502
6: 0.0032224, 0.0051649, 0.0025806, 0.0052142, -0.0019918, 0.0025843
7: -0.0169118, -0.0135266, -0.0172535, -0.0126640, -0.0042478, 0.0037268
8: 0.0033122, 0.0061018, 0.0030411, 0.0068232, -0.0035110, 0.0030607
9: 0.0029056, 0.0082784, 0.0021419, 0.0084567, -0.0055510, 0.0061365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062919
time: 1.10 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
time: 1.12 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0028612, -0.0040864, -0.0025380, -0.0015470, 0.0012252
1: -0.0056382, -0.0034289, -0.0056911, -0.0030990, -0.0025392, 0.0022622
2: 0.9659804, 0.9714270, 0.9646574, 0.9715739, -0.0055936, 0.0067696
3: 0.0227991, 0.0355554, 0.0223306, 0.0366401, -0.0098755, 0.0091259
4: -0.0033972, -0.0009545, -0.0034797, -0.0003776, -0.0030196, 0.0025252
5: 0.0128824, 0.0148174, 0.0125414, 0.0148534, -0.0019710, 0.0022760
6: 0.0030703, 0.0051765, 0.0025423, 0.0052171, -0.0021468, 0.0026342
7: -0.0169928, -0.0133222, -0.0172739, -0.0126126, -0.0043802, 0.0039517
8: 0.0032479, 0.0062727, 0.0030249, 0.0068662, -0.0036183, 0.0032479
9: 0.0027246, 0.0082836, 0.0020963, 0.0084568, -0.0057322, 0.0061872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1_B2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062919
time: 1.13 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
time: 1.18 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040869, -0.0027780, -0.0013064, 0.0015165
1: -0.0056175, -0.0031321, -0.0057107, -0.0033440, -0.0022736, 0.0025786
2: 0.9647900, 0.9715592, 0.9656399, 0.9714647, -0.0066746, 0.0059193
3: 0.0229819, 0.0365314, 0.0221572, 0.0358347, -0.0090471, 0.0105904
4: -0.0034715, -0.0004355, -0.0034185, -0.0008060, -0.0026655, 0.0029830
5: 0.0125756, 0.0148033, 0.0127946, 0.0148667, -0.0022911, 0.0020087
6: 0.0025953, 0.0052130, 0.0029344, 0.0051870, -0.0025917, 0.0022787
7: -0.0172457, -0.0126837, -0.0170651, -0.0131395, -0.0041062, 0.0043814
8: 0.0030472, 0.0068067, 0.0031905, 0.0064255, -0.0033783, 0.0036162
9: 0.0021593, 0.0082159, 0.0025629, 0.0085209, -0.0063616, 0.0056531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0062919
time: 1.26 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
time: 1.30 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040864, -0.0025727, -0.0015117, 0.0015160
1: -0.0056175, -0.0031321, -0.0056909, -0.0031344, -0.0024831, 0.0025588
2: 0.9647900, 0.9715592, 0.9647995, 0.9715582, -0.0067681, 0.0067598
3: 0.0229819, 0.0365314, 0.0223324, 0.0365237, -0.0093404, 0.0100202
4: -0.0034715, -0.0004355, -0.0034709, -0.0004396, -0.0030319, 0.0030354
5: 0.0125756, 0.0148033, 0.0125780, 0.0148533, -0.0022776, 0.0022253
6: 0.0025953, 0.0052130, 0.0025990, 0.0052127, -0.0026175, 0.0026140
7: -0.0172457, -0.0126837, -0.0172437, -0.0126887, -0.0045569, 0.0045599
8: 0.0030472, 0.0068067, 0.0030488, 0.0068025, -0.0037552, 0.0037578
9: 0.0021593, 0.0082159, 0.0021638, 0.0084561, -0.0062968, 0.0060522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0067099
time: 1.34 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
time: 1.29 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025615, -0.0040849, -0.0029543, -0.0011321, 0.0015235
1: -0.0056911, -0.0031229, -0.0056366, -0.0035239, -0.0021671, 0.0025137
2: 0.9647532, 0.9715633, 0.9663616, 0.9713846, -0.0066314, 0.0052017
3: 0.0223310, 0.0365615, 0.0228131, 0.0352429, -0.0091004, 0.0096943
4: -0.0034737, -0.0004195, -0.0033735, -0.0011207, -0.0023531, 0.0029540
5: 0.0125662, 0.0148534, 0.0129807, 0.0148163, -0.0022502, 0.0018727
6: 0.0025806, 0.0052142, 0.0032224, 0.0051649, -0.0025843, 0.0019918
7: -0.0172535, -0.0126640, -0.0169118, -0.0135266, -0.0037268, 0.0042478
8: 0.0030411, 0.0068232, 0.0033122, 0.0061018, -0.0030607, 0.0035110
9: 0.0021419, 0.0084567, 0.0029056, 0.0082784, -0.0061365, 0.0055510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
time: 1.15 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
time: 1.38 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025380, -0.0040850, -0.0028612, -0.0012252, 0.0015470
1: -0.0056911, -0.0030990, -0.0056382, -0.0034289, -0.0022622, 0.0025392
2: 0.9646574, 0.9715739, 0.9659804, 0.9714270, -0.0067696, 0.0055936
3: 0.0223306, 0.0366401, 0.0227991, 0.0355554, -0.0091259, 0.0098755
4: -0.0034797, -0.0003776, -0.0033972, -0.0009545, -0.0025252, 0.0030196
5: 0.0125414, 0.0148534, 0.0128824, 0.0148174, -0.0022760, 0.0019710
6: 0.0025423, 0.0052171, 0.0030703, 0.0051765, -0.0026342, 0.0021468
7: -0.0172739, -0.0126126, -0.0169928, -0.0133222, -0.0039517, 0.0043802
8: 0.0030249, 0.0068662, 0.0032479, 0.0062727, -0.0032479, 0.0036183
9: 0.0020963, 0.0084568, 0.0027246, 0.0082836, -0.0061872, 0.0057322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
time: 1.16 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
time: 1.23 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040863, -0.0027268, -0.0040869, -0.0028018, -0.0012845, 0.0013602
1: -0.0056862, -0.0032917, -0.0057107, -0.0033682, -0.0023179, 0.0024190
2: 0.9654299, 0.9714881, 0.9657370, 0.9714540, -0.0060241, 0.0057511
3: 0.0223745, 0.0360067, 0.0221575, 0.0357550, -0.0089644, 0.0093125
4: -0.0034315, -0.0007145, -0.0034124, -0.0008484, -0.0025832, 0.0026979
5: 0.0127406, 0.0148500, 0.0128197, 0.0148667, -0.0021262, 0.0020303
6: 0.0028506, 0.0051934, 0.0029732, 0.0051840, -0.0023334, 0.0022202
7: -0.0171097, -0.0130269, -0.0170445, -0.0131916, -0.0039181, 0.0040175
8: 0.0031551, 0.0065196, 0.0032069, 0.0063819, -0.0032268, 0.0033128
9: 0.0024632, 0.0084406, 0.0026090, 0.0085208, -0.0060576, 0.0058315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B1_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062914
time: 1.22 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062914
time: 1.24 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0026221, -0.0040869, -0.0027780, -0.0013084, 0.0014648
1: -0.0056909, -0.0031848, -0.0057107, -0.0033440, -0.0023469, 0.0025259
2: 0.9650015, 0.9715357, 0.9656399, 0.9714647, -0.0064632, 0.0058959
3: 0.0223327, 0.0363581, 0.0221572, 0.0358347, -0.0091078, 0.0094219
4: -0.0034583, -0.0005277, -0.0034185, -0.0008060, -0.0026523, 0.0028908
5: 0.0126301, 0.0148532, 0.0127946, 0.0148667, -0.0022366, 0.0020586
6: 0.0026796, 0.0052066, 0.0029344, 0.0051870, -0.0025074, 0.0022722
7: -0.0172008, -0.0127971, -0.0170651, -0.0131395, -0.0040613, 0.0042680
8: 0.0030829, 0.0067119, 0.0031905, 0.0064255, -0.0033426, 0.0035214
9: 0.0022597, 0.0084560, 0.0025629, 0.0085209, -0.0062612, 0.0058932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 1.25 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
time: 1.19 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040844, -0.0025704, -0.0015165, 0.0013064
1: -0.0057107, -0.0033440, -0.0056175, -0.0031321, -0.0025786, 0.0022736
2: 0.9656399, 0.9714647, 0.9647900, 0.9715592, -0.0059193, 0.0066746
3: 0.0221572, 0.0358347, 0.0229819, 0.0365314, -0.0105904, 0.0090471
4: -0.0034185, -0.0008060, -0.0034715, -0.0004355, -0.0029830, 0.0026655
5: 0.0127946, 0.0148667, 0.0125756, 0.0148033, -0.0020087, 0.0022911
6: 0.0029344, 0.0051870, 0.0025953, 0.0052130, -0.0022787, 0.0025917
7: -0.0170651, -0.0131395, -0.0172457, -0.0126837, -0.0043814, 0.0041062
8: 0.0031905, 0.0064255, 0.0030472, 0.0068067, -0.0036162, 0.0033783
9: 0.0025629, 0.0085209, 0.0021593, 0.0082159, -0.0056531, 0.0063616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
time: 1.20 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
time: 1.23 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040864, -0.0025727, -0.0015142, 0.0013084
1: -0.0057107, -0.0033440, -0.0056909, -0.0031344, -0.0025763, 0.0023469
2: 0.9656399, 0.9714647, 0.9647995, 0.9715582, -0.0059183, 0.0066652
3: 0.0221572, 0.0358347, 0.0223324, 0.0365237, -0.0099722, 0.0091109
4: -0.0034185, -0.0008060, -0.0034709, -0.0004396, -0.0029789, 0.0026649
5: 0.0127946, 0.0148667, 0.0125780, 0.0148533, -0.0020586, 0.0022887
6: 0.0029344, 0.0051870, 0.0025990, 0.0052127, -0.0022784, 0.0025880
7: -0.0170651, -0.0131395, -0.0172437, -0.0126887, -0.0043764, 0.0041042
8: 0.0031905, 0.0064255, 0.0030488, 0.0068025, -0.0036120, 0.0033767
9: 0.0025629, 0.0085209, 0.0021638, 0.0084561, -0.0058933, 0.0063571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064564
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
time: 1.32 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040864, -0.0025727, -0.0015137, 0.0015137
1: -0.0056909, -0.0031344, -0.0056909, -0.0031344, -0.0025565, 0.0025565
2: 0.9647995, 0.9715582, 0.9647995, 0.9715582, -0.0067587, 0.0067587
3: 0.0223324, 0.0365237, 0.0223324, 0.0365237, -0.0093895, 0.0093895
4: -0.0034709, -0.0004396, -0.0034709, -0.0004396, -0.0030313, 0.0030313
5: 0.0125780, 0.0148533, 0.0125780, 0.0148533, -0.0022752, 0.0022752
6: 0.0025990, 0.0052127, 0.0025990, 0.0052127, -0.0026137, 0.0026137
7: -0.0172437, -0.0126887, -0.0172437, -0.0126887, -0.0045549, 0.0045549
8: 0.0030488, 0.0068025, 0.0030488, 0.0068025, -0.0037536, 0.0037536
9: 0.0021638, 0.0084561, 0.0021638, 0.0084561, -0.0062924, 0.0062924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0067079
time: 1.24 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067099
time: 1.25 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.82 seconds
IS_B1_A1_A1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_B1_A1_A1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_B1_A1_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
IS_B1_A1_A1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
IS_B1_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
IS_B1_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
IS_B1_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065425
IS_B1_A1_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
IS_B1_A1_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062919
IS_B1_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_B1_A1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0064619
IS_B1_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0066947
IS_B1_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
IS_B1_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_B1_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
IS_B1_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
IS_B1_A1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
IS_B1_A1_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
IS_B1_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062910
IS_B1_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062922
IS_B1_A1_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
IS_B1_A1_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062919
IS_B1_A1_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B1_A1_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B1_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064953
IS_B1_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066793
IS_B1_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065425
IS_B1_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067141
IS_B1_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
IS_B1_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
IS_B1_A2_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_B1_A2_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0061319
IS_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0062922
IS_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
IS_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066939
IS_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065425
IS_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0067141
IS_B1_A2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
IS_B1_A2_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
IS_B1_A2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
IS_B1_A2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
IS_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
IS_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
IS_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065425
IS_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067141
IS_B1_A2_B2_A1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0060624
IS_B1_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0064624
IS_B1_A2_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062919
IS_B1_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_B1_A2_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0060624
IS_B1_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064564
IS_B1_A2_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B1_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_B1_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_B1_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066758, upper bound: 0.0062919
IS_B1_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_B1_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066758, upper bound: 0.0062919
IS_B1_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067099
IS_B1_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0067099
IS_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0064738
IS_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0067099
IS_B2_A1_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
IS_B2_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
IS_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
IS_B2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0062922
IS_B2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066939, upper bound: 0.0062922
IS_B2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0067141
IS_B2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066940, upper bound: 0.0067141
IS_B2_A1_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062919
IS_B2_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_B2_A1_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062919
IS_B2_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
IS_B2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0062919
IS_B2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0062919
IS_B2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0067099
IS_B2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0067099
IS_B2_A1_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
IS_B2_A1_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
IS_B2_A1_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
IS_B2_A1_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
IS_B2_A1_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
IS_B2_A1_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062919
IS_B2_A1_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B2_A1_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B2_A1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
IS_B2_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066758
IS_B2_A1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
IS_B2_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066811
IS_B2_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0064738
IS_B2_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067024
IS_B2_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067051
IS_B2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067099
IS_B2_A2_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
IS_B2_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
IS_B2_A2_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
IS_B2_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
IS_B2_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_B2_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_B2_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
IS_B2_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_B2_A2_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062919
IS_B2_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_B2_A2_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062919
IS_B2_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_B2_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0062919
IS_B2_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_B2_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0067099
IS_B2_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
IS_B2_A2_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
IS_B2_A2_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0061319
IS_B2_A2_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
IS_B2_A2_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062922
IS_B2_A2_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062914
IS_B2_A2_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062914
IS_B2_A2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B2_A2_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0062919
IS_B2_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
IS_B2_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
IS_B2_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064564
IS_B2_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_B2_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0067079
IS_B2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067099

## BFS IS instance: IS_B1_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028285, -0.0040818, -0.0027775, -0.0013048, 0.0012533
1: -0.0055390, -0.0033955, -0.0055180, -0.0033434, -0.0021956, 0.0021225
2: 0.9658464, 0.9714418, 0.9656377, 0.9714649, -0.0056186, 0.0058041
3: 0.0236768, 0.0356653, 0.0238631, 0.0358364, -0.0086187, 0.0082030
4: -0.0034056, -0.0008960, -0.0034186, -0.0008051, -0.0026005, 0.0025226
5: 0.0128479, 0.0147499, 0.0127941, 0.0147356, -0.0018877, 0.0019558
6: 0.0030168, 0.0051807, 0.0029335, 0.0051870, -0.0021703, 0.0022471
7: -0.0170212, -0.0132503, -0.0170656, -0.0131384, -0.0038828, 0.0038153
8: 0.0032253, 0.0063329, 0.0031901, 0.0064265, -0.0032012, 0.0031427
9: 0.0026610, 0.0079590, 0.0025619, 0.0078901, -0.0052291, 0.0053971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060054, upper bound: 0.0058136
time: 1.12 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055882, upper bound: 0.0057678
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040819, -0.0026752, -0.0014071, 0.0012768
1: -0.0055390, -0.0033715, -0.0055218, -0.0032390, -0.0023000, 0.0021503
2: 0.9657504, 0.9714524, 0.9652189, 0.9715115, -0.0057611, 0.0062335
3: 0.0236766, 0.0357439, 0.0238290, 0.0361797, -0.0087107, 0.0083512
4: -0.0034116, -0.0008542, -0.0034447, -0.0006225, -0.0027891, 0.0025905
5: 0.0128232, 0.0147499, 0.0126862, 0.0147382, -0.0019151, 0.0020638
6: 0.0029785, 0.0051836, 0.0027664, 0.0051999, -0.0022214, 0.0024172
7: -0.0170416, -0.0131989, -0.0171545, -0.0129138, -0.0041278, 0.0039557
8: 0.0032091, 0.0063759, 0.0031195, 0.0066143, -0.0034051, 0.0032564
9: 0.0026154, 0.0079591, 0.0023630, 0.0079027, -0.0052873, 0.0055960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.19 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
time: 1.16 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026157, -0.0040818, -0.0027775, -0.0013044, 0.0014661
1: -0.0055220, -0.0031783, -0.0055180, -0.0033434, -0.0021786, 0.0023397
2: 0.9649752, 0.9715386, 0.9656377, 0.9714649, -0.0064897, 0.0059009
3: 0.0238272, 0.0363795, 0.0238631, 0.0358364, -0.0080264, 0.0084536
4: -0.0034599, -0.0005163, -0.0034186, -0.0008051, -0.0026548, 0.0029023
5: 0.0126234, 0.0147384, 0.0127941, 0.0147356, -0.0021122, 0.0019443
6: 0.0026692, 0.0052074, 0.0029335, 0.0051870, -0.0025179, 0.0022738
7: -0.0172063, -0.0127831, -0.0170656, -0.0131384, -0.0040679, 0.0042825
8: 0.0030785, 0.0067236, 0.0031901, 0.0064265, -0.0033480, 0.0035335
9: 0.0022473, 0.0079034, 0.0025619, 0.0078901, -0.0056428, 0.0053415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061170, upper bound: 0.0063503
time: 1.09 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061152, upper bound: 0.0058212
time: 1.13 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040839, -0.0026095, -0.0014727, 0.0010917
1: -0.0055329, -0.0035626, -0.0055989, -0.0031720, -0.0023609, 0.0020363
2: 0.9665166, 0.9713672, 0.9649500, 0.9715414, -0.0050249, 0.0064173
3: 0.0237312, 0.0351158, 0.0231471, 0.0364003, -0.0092414, 0.0086265
4: -0.0033638, -0.0011883, -0.0034615, -0.0005052, -0.0028586, 0.0022732
5: 0.0130206, 0.0147457, 0.0126168, 0.0147906, -0.0017700, 0.0021289
6: 0.0032842, 0.0051601, 0.0026591, 0.0052081, -0.0019239, 0.0025010
7: -0.0168788, -0.0136098, -0.0172117, -0.0127695, -0.0041094, 0.0036019
8: 0.0033383, 0.0060323, 0.0030742, 0.0067349, -0.0033967, 0.0029581
9: 0.0029792, 0.0079389, 0.0022352, 0.0081549, -0.0051756, 0.0057036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061133, upper bound: 0.0061057
time: 1.46 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0058850, upper bound: 0.0064807
time: 1.37 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040837, -0.0027483, -0.0013340, 0.0011974
1: -0.0055389, -0.0034545, -0.0055916, -0.0033136, -0.0022252, 0.0021371
2: 0.9660832, 0.9714154, 0.9655182, 0.9714783, -0.0053951, 0.0058972
3: 0.0236782, 0.0354711, 0.0232114, 0.0359343, -0.0087365, 0.0088849
4: -0.0033908, -0.0009993, -0.0034260, -0.0007530, -0.0026378, 0.0024267
5: 0.0129089, 0.0147498, 0.0127633, 0.0147857, -0.0018768, 0.0019865
6: 0.0031113, 0.0051734, 0.0028858, 0.0051907, -0.0020794, 0.0022875
7: -0.0169709, -0.0133773, -0.0170910, -0.0130743, -0.0038966, 0.0037136
8: 0.0032652, 0.0062266, 0.0031700, 0.0064800, -0.0032148, 0.0030566
9: 0.0027734, 0.0079585, 0.0025051, 0.0081311, -0.0053576, 0.0054533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0060624
time: 1.19 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0060624
time: 1.34 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040839, -0.0026342, -0.0014481, 0.0011976
1: -0.0055389, -0.0034545, -0.0055988, -0.0031972, -0.0023417, 0.0021443
2: 0.9660832, 0.9714154, 0.9650511, 0.9715303, -0.0054470, 0.0063643
3: 0.0236782, 0.0354711, 0.0231474, 0.0363173, -0.0088652, 0.0086801
4: -0.0033908, -0.0009993, -0.0034552, -0.0005493, -0.0028415, 0.0024559
5: 0.0129089, 0.0147498, 0.0126429, 0.0147906, -0.0018817, 0.0021069
6: 0.0031113, 0.0051734, 0.0026995, 0.0052050, -0.0020937, 0.0024739
7: -0.0169709, -0.0133773, -0.0171902, -0.0128237, -0.0041472, 0.0038129
8: 0.0032652, 0.0062266, 0.0030913, 0.0066896, -0.0034243, 0.0031354
9: 0.0027734, 0.0079585, 0.0022833, 0.0081547, -0.0053813, 0.0056752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0062919
time: 1.46 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0066904
time: 1.64 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040844, -0.0028212, -0.0012606, 0.0013069
1: -0.0055180, -0.0033434, -0.0056157, -0.0033881, -0.0021299, 0.0022723
2: 0.9656377, 0.9714649, 0.9658166, 0.9714451, -0.0058074, 0.0056483
3: 0.0238631, 0.0358364, 0.0229978, 0.0356897, -0.0084641, 0.0095442
4: -0.0034186, -0.0008051, -0.0034074, -0.0008831, -0.0025355, 0.0026024
5: 0.0127941, 0.0147356, 0.0128402, 0.0148021, -0.0020080, 0.0018954
6: 0.0029335, 0.0051870, 0.0030049, 0.0051816, -0.0022480, 0.0021821
7: -0.0170656, -0.0131384, -0.0170275, -0.0132344, -0.0038312, 0.0038892
8: 0.0031901, 0.0064265, 0.0032203, 0.0063462, -0.0031561, 0.0032062
9: 0.0025619, 0.0078901, 0.0026468, 0.0082101, -0.0056482, 0.0052433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063482, upper bound: 0.0062777
time: 1.27 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063219, upper bound: 0.0061057
time: 1.03 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040844, -0.0027970, -0.0012849, 0.0014092
1: -0.0055218, -0.0032390, -0.0056158, -0.0033633, -0.0021585, 0.0023767
2: 0.9652189, 0.9715115, 0.9657173, 0.9714562, -0.0062373, 0.0057942
3: 0.0238290, 0.0361797, 0.0229975, 0.0357711, -0.0086156, 0.0095914
4: -0.0034447, -0.0006225, -0.0034136, -0.0008398, -0.0026049, 0.0027911
5: 0.0126862, 0.0147382, 0.0128146, 0.0148021, -0.0021160, 0.0019236
6: 0.0027664, 0.0051999, 0.0029653, 0.0051846, -0.0024182, 0.0022346
7: -0.0171545, -0.0129138, -0.0170486, -0.0131811, -0.0039734, 0.0041349
8: 0.0031195, 0.0066143, 0.0032036, 0.0063907, -0.0032712, 0.0034107
9: 0.0023630, 0.0079027, 0.0025997, 0.0082102, -0.0058472, 0.0053030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065730, upper bound: 0.0062919
time: 1.36 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065475, upper bound: 0.0061057
time: 1.25 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040839, -0.0026095, -0.0014723, 0.0013064
1: -0.0055180, -0.0033434, -0.0055989, -0.0031720, -0.0023460, 0.0022554
2: 0.9656377, 0.9714649, 0.9649500, 0.9715414, -0.0059037, 0.0065150
3: 0.0238631, 0.0358364, 0.0231471, 0.0364003, -0.0087248, 0.0089621
4: -0.0034186, -0.0008051, -0.0034615, -0.0005052, -0.0029134, 0.0026564
5: 0.0127941, 0.0147356, 0.0126168, 0.0147906, -0.0019965, 0.0021188
6: 0.0029335, 0.0051870, 0.0026591, 0.0052081, -0.0022746, 0.0025280
7: -0.0170656, -0.0131384, -0.0172117, -0.0127695, -0.0042961, 0.0040733
8: 0.0031901, 0.0064265, 0.0030742, 0.0067349, -0.0035448, 0.0033523
9: 0.0025619, 0.0078901, 0.0022352, 0.0081549, -0.0055930, 0.0056548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_B1_A1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0064176
time: 1.27 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0061017
time: 1.33 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040839, -0.0025848, -0.0014970, 0.0014087
1: -0.0055218, -0.0032390, -0.0055989, -0.0031468, -0.0023750, 0.0023599
2: 0.9652189, 0.9715115, 0.9648491, 0.9715527, -0.0063338, 0.0066624
3: 0.0238290, 0.0361797, 0.0231467, 0.0364830, -0.0089208, 0.0089961
4: -0.0034447, -0.0006225, -0.0034678, -0.0004612, -0.0029835, 0.0028453
5: 0.0126862, 0.0147382, 0.0125908, 0.0147907, -0.0021045, 0.0021474
6: 0.0027664, 0.0051999, 0.0026188, 0.0052112, -0.0024448, 0.0025811
7: -0.0171545, -0.0129138, -0.0172331, -0.0127154, -0.0044391, 0.0043194
8: 0.0031195, 0.0066143, 0.0030572, 0.0067802, -0.0036606, 0.0035571
9: 0.0023630, 0.0079027, 0.0021873, 0.0081550, -0.0057920, 0.0057154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0064738
time: 1.42 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0067099
time: 1.57 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027483, -0.0040823, -0.0028863, -0.0011974, 0.0013340
1: -0.0055916, -0.0033136, -0.0055389, -0.0034545, -0.0021371, 0.0022252
2: 0.9655182, 0.9714783, 0.9660832, 0.9714154, -0.0058972, 0.0053951
3: 0.0232114, 0.0359343, 0.0236782, 0.0354711, -0.0088849, 0.0087365
4: -0.0034260, -0.0007530, -0.0033908, -0.0009993, -0.0024267, 0.0026378
5: 0.0127633, 0.0147857, 0.0129089, 0.0147498, -0.0019865, 0.0018768
6: 0.0028858, 0.0051907, 0.0031113, 0.0051734, -0.0022875, 0.0020794
7: -0.0170910, -0.0130743, -0.0169709, -0.0133773, -0.0037136, 0.0038966
8: 0.0031700, 0.0064800, 0.0032652, 0.0062266, -0.0030566, 0.0032148
9: 0.0025051, 0.0081311, 0.0027734, 0.0079585, -0.0054533, 0.0053576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
time: 1.19 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026342, -0.0040823, -0.0028863, -0.0011976, 0.0014481
1: -0.0055988, -0.0031972, -0.0055389, -0.0034545, -0.0021443, 0.0023417
2: 0.9650511, 0.9715303, 0.9660832, 0.9714154, -0.0063643, 0.0054470
3: 0.0231474, 0.0363173, 0.0236782, 0.0354711, -0.0086801, 0.0088652
4: -0.0034552, -0.0005493, -0.0033908, -0.0009993, -0.0024559, 0.0028415
5: 0.0126429, 0.0147906, 0.0129089, 0.0147498, -0.0021069, 0.0018817
6: 0.0026995, 0.0052050, 0.0031113, 0.0051734, -0.0024739, 0.0020937
7: -0.0171902, -0.0128237, -0.0169709, -0.0133773, -0.0038129, 0.0041472
8: 0.0030913, 0.0066896, 0.0032652, 0.0062266, -0.0031354, 0.0034243
9: 0.0022833, 0.0081547, 0.0027734, 0.0079585, -0.0056752, 0.0053813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
time: 1.12 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
time: 1.05 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028212, -0.0040818, -0.0027775, -0.0013069, 0.0012606
1: -0.0056157, -0.0033881, -0.0055180, -0.0033434, -0.0022723, 0.0021299
2: 0.9658166, 0.9714451, 0.9656377, 0.9714649, -0.0056483, 0.0058074
3: 0.0229978, 0.0356897, 0.0238631, 0.0358364, -0.0095442, 0.0084641
4: -0.0034074, -0.0008831, -0.0034186, -0.0008051, -0.0026024, 0.0025355
5: 0.0128402, 0.0148021, 0.0127941, 0.0147356, -0.0018954, 0.0020080
6: 0.0030049, 0.0051816, 0.0029335, 0.0051870, -0.0021821, 0.0022480
7: -0.0170275, -0.0132344, -0.0170656, -0.0131384, -0.0038892, 0.0038312
8: 0.0032203, 0.0063462, 0.0031901, 0.0064265, -0.0032062, 0.0031561
9: 0.0026468, 0.0082101, 0.0025619, 0.0078901, -0.0052433, 0.0056482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062777, upper bound: 0.0063482
time: 1.58 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0063219
time: 1.13 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040819, -0.0026752, -0.0014092, 0.0012849
1: -0.0056158, -0.0033633, -0.0055218, -0.0032390, -0.0023767, 0.0021585
2: 0.9657173, 0.9714562, 0.9652189, 0.9715115, -0.0057942, 0.0062373
3: 0.0229975, 0.0357711, 0.0238290, 0.0361797, -0.0095914, 0.0086156
4: -0.0034136, -0.0008398, -0.0034447, -0.0006225, -0.0027911, 0.0026049
5: 0.0128146, 0.0148021, 0.0126862, 0.0147382, -0.0019236, 0.0021160
6: 0.0029653, 0.0051846, 0.0027664, 0.0051999, -0.0022346, 0.0024182
7: -0.0170486, -0.0131811, -0.0171545, -0.0129138, -0.0041349, 0.0039734
8: 0.0032036, 0.0063907, 0.0031195, 0.0066143, -0.0034107, 0.0032712
9: 0.0025997, 0.0082102, 0.0023630, 0.0079027, -0.0053030, 0.0058472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065730
time: 1.20 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065474
time: 1.29 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026095, -0.0040818, -0.0027775, -0.0013064, 0.0014723
1: -0.0055989, -0.0031720, -0.0055180, -0.0033434, -0.0022554, 0.0023460
2: 0.9649500, 0.9715414, 0.9656377, 0.9714649, -0.0065150, 0.0059037
3: 0.0231471, 0.0364003, 0.0238631, 0.0358364, -0.0089621, 0.0087248
4: -0.0034615, -0.0005052, -0.0034186, -0.0008051, -0.0026564, 0.0029134
5: 0.0126168, 0.0147906, 0.0127941, 0.0147356, -0.0021188, 0.0019965
6: 0.0026591, 0.0052081, 0.0029335, 0.0051870, -0.0025280, 0.0022746
7: -0.0172117, -0.0127695, -0.0170656, -0.0131384, -0.0040733, 0.0042961
8: 0.0030742, 0.0067349, 0.0031901, 0.0064265, -0.0033523, 0.0035448
9: 0.0022352, 0.0081549, 0.0025619, 0.0078901, -0.0056548, 0.0055930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_A1_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064404, upper bound: 0.0058212
time: 1.15 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061075, upper bound: 0.0058212
time: 1.21 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040819, -0.0026752, -0.0014087, 0.0014970
1: -0.0055989, -0.0031468, -0.0055218, -0.0032390, -0.0023599, 0.0023750
2: 0.9648491, 0.9715527, 0.9652189, 0.9715115, -0.0066624, 0.0063338
3: 0.0231467, 0.0364830, 0.0238290, 0.0361797, -0.0089961, 0.0089208
4: -0.0034678, -0.0004612, -0.0034447, -0.0006225, -0.0028453, 0.0029835
5: 0.0125908, 0.0147907, 0.0126862, 0.0147382, -0.0021474, 0.0021045
6: 0.0026188, 0.0052112, 0.0027664, 0.0051999, -0.0025811, 0.0024448
7: -0.0172331, -0.0127154, -0.0171545, -0.0129138, -0.0043194, 0.0044391
8: 0.0030572, 0.0067802, 0.0031195, 0.0066143, -0.0035571, 0.0036606
9: 0.0021873, 0.0081550, 0.0023630, 0.0079027, -0.0057154, 0.0057920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A1_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067072
time: 1.46 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067141
time: 1.37 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028212, -0.0040837, -0.0027803, -0.0013041, 0.0012625
1: -0.0056157, -0.0033881, -0.0055914, -0.0033463, -0.0022694, 0.0022034
2: 0.9658166, 0.9714451, 0.9656492, 0.9714637, -0.0056471, 0.0057958
3: 0.0229978, 0.0356897, 0.0232130, 0.0358269, -0.0089230, 0.0085179
4: -0.0034074, -0.0008831, -0.0034179, -0.0008101, -0.0025973, 0.0025348
5: 0.0128402, 0.0148021, 0.0127971, 0.0147856, -0.0019454, 0.0020051
6: 0.0030049, 0.0051816, 0.0029381, 0.0051867, -0.0021818, 0.0022434
7: -0.0170275, -0.0132344, -0.0170631, -0.0131446, -0.0038830, 0.0038288
8: 0.0032203, 0.0063462, 0.0031921, 0.0064213, -0.0032010, 0.0031541
9: 0.0026468, 0.0082101, 0.0025674, 0.0081305, -0.0054837, 0.0056427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062653, upper bound: 0.0063063
time: 1.44 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0062835
time: 1.33 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040839, -0.0026646, -0.0014198, 0.0012870
1: -0.0056158, -0.0033633, -0.0055987, -0.0032282, -0.0023876, 0.0022354
2: 0.9657173, 0.9714562, 0.9651754, 0.9715163, -0.0057990, 0.0062808
3: 0.0229975, 0.0357711, 0.0231488, 0.0362154, -0.0090377, 0.0086778
4: -0.0034136, -0.0008398, -0.0034474, -0.0006035, -0.0028101, 0.0026076
5: 0.0128146, 0.0148021, 0.0126750, 0.0147905, -0.0019759, 0.0021272
6: 0.0029653, 0.0051846, 0.0027491, 0.0052012, -0.0022359, 0.0024355
7: -0.0170486, -0.0131811, -0.0171638, -0.0128904, -0.0041582, 0.0039827
8: 0.0032036, 0.0063907, 0.0031122, 0.0066338, -0.0034302, 0.0032785
9: 0.0025997, 0.0082102, 0.0023423, 0.0081542, -0.0055545, 0.0058678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065734
time: 1.34 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065481
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0026219, -0.0040822, -0.0029922, -0.0010922, 0.0014603
1: -0.0056175, -0.0031846, -0.0055329, -0.0035626, -0.0020549, 0.0023483
2: 0.9650006, 0.9715359, 0.9665166, 0.9713672, -0.0063666, 0.0050193
3: 0.0229821, 0.0363587, 0.0237312, 0.0351158, -0.0088098, 0.0092320
4: -0.0034583, -0.0005273, -0.0033638, -0.0011883, -0.0022701, 0.0028365
5: 0.0126299, 0.0148033, 0.0130206, 0.0147457, -0.0021159, 0.0017827
6: 0.0026793, 0.0052066, 0.0032842, 0.0051601, -0.0024808, 0.0019223
7: -0.0172009, -0.0127967, -0.0168788, -0.0136098, -0.0035912, 0.0040822
8: 0.0030827, 0.0067122, 0.0033383, 0.0060323, -0.0029495, 0.0033739
9: 0.0022593, 0.0082159, 0.0029792, 0.0079389, -0.0056795, 0.0052366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
time: 1.19 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
time: 1.32 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0026219, -0.0040823, -0.0028863, -0.0011981, 0.0014604
1: -0.0056175, -0.0031846, -0.0055389, -0.0034545, -0.0021630, 0.0023543
2: 0.9650006, 0.9715359, 0.9660832, 0.9714154, -0.0064148, 0.0054526
3: 0.0229821, 0.0363587, 0.0236782, 0.0354711, -0.0090706, 0.0091573
4: -0.0034583, -0.0005273, -0.0033908, -0.0009993, -0.0024590, 0.0028635
5: 0.0126299, 0.0148033, 0.0129089, 0.0147498, -0.0021199, 0.0018944
6: 0.0026793, 0.0052066, 0.0031113, 0.0051734, -0.0024941, 0.0020953
7: -0.0172009, -0.0127967, -0.0169709, -0.0133773, -0.0038236, 0.0041742
8: 0.0030827, 0.0067122, 0.0032652, 0.0062266, -0.0031439, 0.0034470
9: 0.0022593, 0.0082159, 0.0027734, 0.0079585, -0.0056991, 0.0054424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
time: 1.17 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
time: 1.37 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027957, -0.0040818, -0.0027775, -0.0013075, 0.0012860
1: -0.0056383, -0.0033620, -0.0055180, -0.0033434, -0.0022949, 0.0021559
2: 0.9657123, 0.9714566, 0.9656377, 0.9714649, -0.0057526, 0.0058190
3: 0.0227978, 0.0357753, 0.0238631, 0.0358364, -0.0099319, 0.0088126
4: -0.0034139, -0.0008376, -0.0034186, -0.0008051, -0.0026089, 0.0025810
5: 0.0128133, 0.0148175, 0.0127941, 0.0147356, -0.0019223, 0.0020234
6: 0.0029633, 0.0051848, 0.0029335, 0.0051870, -0.0022238, 0.0022512
7: -0.0170497, -0.0131783, -0.0170656, -0.0131384, -0.0039113, 0.0038872
8: 0.0032027, 0.0063930, 0.0031901, 0.0064265, -0.0032238, 0.0032029
9: 0.0025972, 0.0082840, 0.0025619, 0.0078901, -0.0052928, 0.0057222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062776, upper bound: 0.0063802
time: 1.37 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061120, upper bound: 0.0063618
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040819, -0.0026752, -0.0014098, 0.0013087
1: -0.0056384, -0.0033390, -0.0055218, -0.0032390, -0.0023993, 0.0021828
2: 0.9656199, 0.9714670, 0.9652189, 0.9715115, -0.0058916, 0.0062481
3: 0.0227976, 0.0358510, 0.0238290, 0.0361797, -0.0102164, 0.0089443
4: -0.0034197, -0.0007973, -0.0034447, -0.0006225, -0.0027972, 0.0026474
5: 0.0127895, 0.0148175, 0.0126862, 0.0147382, -0.0019487, 0.0021313
6: 0.0029264, 0.0051876, 0.0027664, 0.0051999, -0.0022734, 0.0024212
7: -0.0170693, -0.0131288, -0.0171545, -0.0129138, -0.0041556, 0.0040257
8: 0.0031871, 0.0064344, 0.0031195, 0.0066143, -0.0034271, 0.0033149
9: 0.0025534, 0.0082841, 0.0023630, 0.0079027, -0.0053493, 0.0059211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065895
time: 1.13 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061120, upper bound: 0.0065702
time: 1.11 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025930, -0.0040818, -0.0027775, -0.0013069, 0.0014887
1: -0.0056175, -0.0031552, -0.0055180, -0.0033434, -0.0022741, 0.0023628
2: 0.9648827, 0.9715489, 0.9656377, 0.9714649, -0.0065823, 0.0059112
3: 0.0229822, 0.0364554, 0.0238631, 0.0358364, -0.0093073, 0.0089946
4: -0.0034657, -0.0004759, -0.0034186, -0.0008051, -0.0026606, 0.0029427
5: 0.0125995, 0.0148033, 0.0127941, 0.0147356, -0.0021361, 0.0020092
6: 0.0026322, 0.0052102, 0.0029335, 0.0051870, -0.0025548, 0.0022767
7: -0.0172260, -0.0127334, -0.0170656, -0.0131384, -0.0040876, 0.0043322
8: 0.0030629, 0.0067651, 0.0031901, 0.0064265, -0.0033636, 0.0035750
9: 0.0022033, 0.0082158, 0.0025619, 0.0078901, -0.0056868, 0.0056540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064474, upper bound: 0.0058212
time: 1.22 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061152, upper bound: 0.0058212
time: 1.08 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040819, -0.0026752, -0.0014092, 0.0015114
1: -0.0056175, -0.0031321, -0.0055218, -0.0032390, -0.0023785, 0.0023897
2: 0.9647900, 0.9715592, 0.9652189, 0.9715115, -0.0067214, 0.0063403
3: 0.0229819, 0.0365314, 0.0238290, 0.0361797, -0.0095766, 0.0091726
4: -0.0034715, -0.0004355, -0.0034447, -0.0006225, -0.0028490, 0.0030092
5: 0.0125756, 0.0148033, 0.0126862, 0.0147382, -0.0021626, 0.0021172
6: 0.0025953, 0.0052130, 0.0027664, 0.0051999, -0.0026046, 0.0024466
7: -0.0172457, -0.0126837, -0.0171545, -0.0129138, -0.0043319, 0.0044708
8: 0.0030472, 0.0068067, 0.0031195, 0.0066143, -0.0035670, 0.0036871
9: 0.0021593, 0.0082159, 0.0023630, 0.0079027, -0.0057434, 0.0058529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
time: 1.15 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
time: 1.35 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0028018, -0.0040818, -0.0027775, -0.0013094, 0.0012800
1: -0.0057107, -0.0033682, -0.0055180, -0.0033434, -0.0023672, 0.0021498
2: 0.9657370, 0.9714540, 0.9656377, 0.9714649, -0.0057279, 0.0058163
3: 0.0221575, 0.0357550, 0.0238631, 0.0358364, -0.0106313, 0.0088324
4: -0.0034124, -0.0008484, -0.0034186, -0.0008051, -0.0026073, 0.0025702
5: 0.0128197, 0.0148667, 0.0127941, 0.0147356, -0.0019159, 0.0020726
6: 0.0029732, 0.0051840, 0.0029335, 0.0051870, -0.0022139, 0.0022505
7: -0.0170445, -0.0131916, -0.0170656, -0.0131384, -0.0039061, 0.0038739
8: 0.0032069, 0.0063819, 0.0031901, 0.0064265, -0.0032196, 0.0031918
9: 0.0026090, 0.0085208, 0.0025619, 0.0078901, -0.0052811, 0.0059589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062777, upper bound: 0.0063505
time: 1.36 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0063317
time: 1.20 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040819, -0.0026752, -0.0014117, 0.0013039
1: -0.0057107, -0.0033440, -0.0055218, -0.0032390, -0.0024717, 0.0021779
2: 0.9656399, 0.9714647, 0.9652189, 0.9715115, -0.0058716, 0.0062458
3: 0.0221572, 0.0358347, 0.0238290, 0.0361797, -0.0108171, 0.0089711
4: -0.0034185, -0.0008060, -0.0034447, -0.0006225, -0.0027960, 0.0026387
5: 0.0127946, 0.0148667, 0.0126862, 0.0147382, -0.0019436, 0.0021806
6: 0.0029344, 0.0051870, 0.0027664, 0.0051999, -0.0022655, 0.0024206
7: -0.0170651, -0.0131395, -0.0171545, -0.0129138, -0.0041513, 0.0040150
8: 0.0031905, 0.0064255, 0.0031195, 0.0066143, -0.0034238, 0.0033060
9: 0.0025629, 0.0085209, 0.0023630, 0.0079027, -0.0053398, 0.0061579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065754
time: 1.49 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065565
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025960, -0.0040818, -0.0027775, -0.0013089, 0.0014858
1: -0.0056909, -0.0031582, -0.0055180, -0.0033434, -0.0023474, 0.0023598
2: 0.9648947, 0.9715475, 0.9656377, 0.9714649, -0.0065702, 0.0059098
3: 0.0223328, 0.0364455, 0.0238631, 0.0358364, -0.0100156, 0.0090199
4: -0.0034649, -0.0004811, -0.0034186, -0.0008051, -0.0026599, 0.0029375
5: 0.0126026, 0.0148532, 0.0127941, 0.0147356, -0.0021330, 0.0020591
6: 0.0026370, 0.0052098, 0.0029335, 0.0051870, -0.0025500, 0.0022763
7: -0.0172234, -0.0127399, -0.0170656, -0.0131384, -0.0040850, 0.0043257
8: 0.0030649, 0.0067597, 0.0031901, 0.0064265, -0.0033616, 0.0035696
9: 0.0022090, 0.0084560, 0.0025619, 0.0078901, -0.0056811, 0.0058941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067006, upper bound: 0.0064148
time: 1.55 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0064148
time: 1.37 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040819, -0.0026752, -0.0014112, 0.0015091
1: -0.0056909, -0.0031344, -0.0055218, -0.0032390, -0.0024519, 0.0023874
2: 0.9647995, 0.9715582, 0.9652189, 0.9715115, -0.0067120, 0.0063393
3: 0.0223324, 0.0365237, 0.0238290, 0.0361797, -0.0101941, 0.0092050
4: -0.0034709, -0.0004396, -0.0034447, -0.0006225, -0.0028484, 0.0030051
5: 0.0125780, 0.0148533, 0.0126862, 0.0147382, -0.0021602, 0.0021671
6: 0.0025990, 0.0052127, 0.0027664, 0.0051999, -0.0026009, 0.0024463
7: -0.0172437, -0.0126887, -0.0171545, -0.0129138, -0.0043299, 0.0044658
8: 0.0030488, 0.0068025, 0.0031195, 0.0066143, -0.0035655, 0.0036829
9: 0.0021638, 0.0084561, 0.0023630, 0.0079027, -0.0057389, 0.0060931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0066144
time: 1.29 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066144
time: 1.15 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027957, -0.0040837, -0.0027803, -0.0013047, 0.0012880
1: -0.0056383, -0.0033620, -0.0055914, -0.0033463, -0.0022920, 0.0022294
2: 0.9657123, 0.9714566, 0.9656492, 0.9714637, -0.0057514, 0.0058074
3: 0.0227978, 0.0357753, 0.0232130, 0.0358269, -0.0100668, 0.0096041
4: -0.0034139, -0.0008376, -0.0034179, -0.0008101, -0.0026038, 0.0025803
5: 0.0128133, 0.0148175, 0.0127971, 0.0147856, -0.0019723, 0.0020204
6: 0.0029633, 0.0051848, 0.0029381, 0.0051867, -0.0022234, 0.0022466
7: -0.0170497, -0.0131783, -0.0170631, -0.0131446, -0.0039052, 0.0038848
8: 0.0032027, 0.0063930, 0.0031921, 0.0064213, -0.0032186, 0.0032010
9: 0.0025972, 0.0082840, 0.0025674, 0.0081305, -0.0055332, 0.0057167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0058149, upper bound: 0.0057198
time: 1.32 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0054953, upper bound: 0.0056646
time: 1.17 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040839, -0.0026646, -0.0014204, 0.0013108
1: -0.0056384, -0.0033390, -0.0055987, -0.0032282, -0.0024102, 0.0022597
2: 0.9656199, 0.9714670, 0.9651754, 0.9715163, -0.0058964, 0.0062916
3: 0.0227976, 0.0358510, 0.0231488, 0.0362154, -0.0103863, 0.0098400
4: -0.0034197, -0.0007973, -0.0034474, -0.0006035, -0.0028162, 0.0026501
5: 0.0127895, 0.0148175, 0.0126750, 0.0147905, -0.0020010, 0.0021426
6: 0.0029264, 0.0051876, 0.0027491, 0.0052012, -0.0022748, 0.0024385
7: -0.0170693, -0.0131288, -0.0171638, -0.0128904, -0.0041789, 0.0040349
8: 0.0031871, 0.0064344, 0.0031122, 0.0066338, -0.0034467, 0.0033222
9: 0.0025534, 0.0082841, 0.0023423, 0.0081542, -0.0056008, 0.0059418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
time: 1.22 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066910
time: 1.41 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0028018, -0.0040837, -0.0027803, -0.0013066, 0.0012820
1: -0.0057107, -0.0033682, -0.0055914, -0.0033463, -0.0023643, 0.0022232
2: 0.9657370, 0.9714540, 0.9656492, 0.9714637, -0.0057268, 0.0058047
3: 0.0221575, 0.0357550, 0.0232130, 0.0358269, -0.0102276, 0.0090997
4: -0.0034124, -0.0008484, -0.0034179, -0.0008101, -0.0026023, 0.0025695
5: 0.0128197, 0.0148667, 0.0127971, 0.0147856, -0.0019659, 0.0020696
6: 0.0029732, 0.0051840, 0.0029381, 0.0051867, -0.0022135, 0.0022459
7: -0.0170445, -0.0131916, -0.0170631, -0.0131446, -0.0038999, 0.0038715
8: 0.0032069, 0.0063819, 0.0031921, 0.0064213, -0.0032144, 0.0031898
9: 0.0026090, 0.0085208, 0.0025674, 0.0081305, -0.0055215, 0.0059535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0062425, upper bound: 0.0063075
time: 1.39 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061026, upper bound: 0.0062913
time: 1.43 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040839, -0.0026646, -0.0014223, 0.0013059
1: -0.0057107, -0.0033440, -0.0055987, -0.0032282, -0.0024825, 0.0022547
2: 0.9656399, 0.9714647, 0.9651754, 0.9715163, -0.0058764, 0.0062893
3: 0.0221572, 0.0358347, 0.0231488, 0.0362154, -0.0105349, 0.0092446
4: -0.0034185, -0.0008060, -0.0034474, -0.0006035, -0.0028149, 0.0026414
5: 0.0127946, 0.0148667, 0.0126750, 0.0147905, -0.0019959, 0.0021918
6: 0.0029344, 0.0051870, 0.0027491, 0.0052012, -0.0022669, 0.0024379
7: -0.0170651, -0.0131395, -0.0171638, -0.0128904, -0.0041747, 0.0040243
8: 0.0031905, 0.0064255, 0.0031122, 0.0066338, -0.0034433, 0.0033133
9: 0.0025629, 0.0085209, 0.0023423, 0.0081542, -0.0055913, 0.0061786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065761
time: 1.47 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065573
time: 1.14 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027528, -0.0040844, -0.0028212, -0.0012632, 0.0013315
1: -0.0056159, -0.0033183, -0.0056157, -0.0033881, -0.0022278, 0.0022975
2: 0.9655367, 0.9714762, 0.9658166, 0.9714451, -0.0059084, 0.0056596
3: 0.0229966, 0.0359192, 0.0229978, 0.0356897, -0.0096670, 0.0099389
4: -0.0034249, -0.0007610, -0.0034074, -0.0008831, -0.0025418, 0.0026464
5: 0.0127681, 0.0148022, 0.0128402, 0.0148021, -0.0020341, 0.0019620
6: 0.0028932, 0.0051901, 0.0030049, 0.0051816, -0.0022883, 0.0021852
7: -0.0170870, -0.0130842, -0.0170275, -0.0132344, -0.0038527, 0.0039433
8: 0.0031731, 0.0064717, 0.0032203, 0.0063462, -0.0031731, 0.0032514
9: 0.0025139, 0.0082105, 0.0026468, 0.0082101, -0.0056962, 0.0055637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063482, upper bound: 0.0062829
time: 1.46 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063219, upper bound: 0.0061057
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0026577, -0.0040844, -0.0027970, -0.0012875, 0.0014266
1: -0.0056173, -0.0032212, -0.0056158, -0.0033633, -0.0022540, 0.0023946
2: 0.9651476, 0.9715195, 0.9657173, 0.9714562, -0.0063086, 0.0058022
3: 0.0229840, 0.0362383, 0.0229975, 0.0357711, -0.0096958, 0.0099690
4: -0.0034492, -0.0005913, -0.0034136, -0.0008398, -0.0026094, 0.0028223
5: 0.0126677, 0.0148032, 0.0128146, 0.0148021, -0.0021344, 0.0019886
6: 0.0027379, 0.0052021, 0.0029653, 0.0051846, -0.0024467, 0.0022368
7: -0.0171697, -0.0128754, -0.0170486, -0.0131811, -0.0039886, 0.0041732
8: 0.0031075, 0.0066463, 0.0032036, 0.0063907, -0.0032832, 0.0034428
9: 0.0023291, 0.0082152, 0.0025997, 0.0082102, -0.0058811, 0.0056155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0060624
time: 1.17 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0062919
time: 1.42 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040862, -0.0027620, -0.0040844, -0.0028212, -0.0012651, 0.0013224
1: -0.0056859, -0.0033276, -0.0056157, -0.0033881, -0.0022979, 0.0022882
2: 0.9655740, 0.9714720, 0.9658166, 0.9714451, -0.0058711, 0.0056554
3: 0.0223763, 0.0358886, 0.0229978, 0.0356897, -0.0097176, 0.0093052
4: -0.0034226, -0.0007773, -0.0034074, -0.0008831, -0.0025395, 0.0026301
5: 0.0127777, 0.0148499, 0.0128402, 0.0148021, -0.0020244, 0.0020097
6: 0.0029081, 0.0051890, 0.0030049, 0.0051816, -0.0022735, 0.0021841
7: -0.0170791, -0.0131042, -0.0170275, -0.0132344, -0.0038447, 0.0039233
8: 0.0031794, 0.0064550, 0.0032203, 0.0063462, -0.0031668, 0.0032347
9: 0.0025316, 0.0084399, 0.0026468, 0.0082101, -0.0056785, 0.0057931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060201, upper bound: 0.0055312
time: 1.20 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0056409, upper bound: 0.0055091
time: 1.10 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0026585, -0.0040844, -0.0027970, -0.0012894, 0.0014259
1: -0.0056907, -0.0032220, -0.0056158, -0.0033633, -0.0023274, 0.0023938
2: 0.9651505, 0.9715191, 0.9657173, 0.9714562, -0.0063056, 0.0058018
3: 0.0223344, 0.0362358, 0.0229975, 0.0357711, -0.0097528, 0.0094006
4: -0.0034490, -0.0005927, -0.0034136, -0.0008398, -0.0026092, 0.0028210
5: 0.0126685, 0.0148531, 0.0128146, 0.0148021, -0.0021336, 0.0020385
6: 0.0027391, 0.0052020, 0.0029653, 0.0051846, -0.0024455, 0.0022367
7: -0.0171691, -0.0128771, -0.0170486, -0.0131811, -0.0039880, 0.0041716
8: 0.0031080, 0.0066450, 0.0032036, 0.0063907, -0.0032827, 0.0034414
9: 0.0023305, 0.0084554, 0.0025997, 0.0082102, -0.0058797, 0.0058557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
time: 1.33 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0062919
time: 1.41 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.10 seconds
IS_B1_A1_A1_B1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0060054, upper bound: 0.0058136
IS_B1_A1_A1_B1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0055882, upper bound: 0.0057678
IS_B1_A1_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_B1_A1_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
IS_B1_A1_A1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061170, upper bound: 0.0063503
IS_B1_A1_A1_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061152, upper bound: 0.0058212
IS_B1_A1_A1_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061133, upper bound: 0.0061057
IS_B1_A1_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0058850, upper bound: 0.0064807
IS_B1_A1_A1_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0060624
IS_B1_A1_A1_B2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0060624
IS_B1_A1_A1_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0062919
IS_B1_A1_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062887, upper bound: 0.0066904
IS_B1_A1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0063482, upper bound: 0.0062777
IS_B1_A1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0063219, upper bound: 0.0061057
IS_B1_A1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0065730, upper bound: 0.0062919
IS_B1_A1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0065475, upper bound: 0.0061057
IS_B1_A1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0064176
IS_B1_A1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0061017
IS_B1_A1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0064738
IS_B1_A1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0067099
IS_B1_A1_A2_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
IS_B1_A1_A2_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062910
IS_B1_A1_A2_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
IS_B1_A1_A2_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
IS_B1_A1_A2_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062777, upper bound: 0.0063482
IS_B1_A1_A2_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0063219
IS_B1_A1_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065730
IS_B1_A1_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065474
IS_B1_A1_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0064404, upper bound: 0.0058212
IS_B1_A1_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061075, upper bound: 0.0058212
IS_B1_A1_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067072
IS_B1_A1_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067141
IS_B1_A1_A2_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062653, upper bound: 0.0063063
IS_B1_A1_A2_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0062835
IS_B1_A1_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065734
IS_B1_A1_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065481
IS_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
IS_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0061319
IS_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
IS_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0062922
IS_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062776, upper bound: 0.0063802
IS_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061120, upper bound: 0.0063618
IS_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065895
IS_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061120, upper bound: 0.0065702
IS_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0064474, upper bound: 0.0058212
IS_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061152, upper bound: 0.0058212
IS_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
IS_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
IS_B1_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062777, upper bound: 0.0063505
IS_B1_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0063317
IS_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065754
IS_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065565
IS_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0067006, upper bound: 0.0064148
IS_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0064148
IS_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0066144
IS_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0066123, upper bound: 0.0066144
IS_B1_A2_B2_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0058149, upper bound: 0.0057198
IS_B1_A2_B2_A1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0054953, upper bound: 0.0056646
IS_B1_A2_B2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_B1_A2_B2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066910
IS_B1_A2_B2_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062425, upper bound: 0.0063075
IS_B1_A2_B2_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061026, upper bound: 0.0062913
IS_B1_A2_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0065761
IS_B1_A2_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0061057, upper bound: 0.0065573
IS_B1_A2_B2_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0063482, upper bound: 0.0062829
IS_B1_A2_B2_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0063219, upper bound: 0.0061057
IS_B1_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0060624
IS_B1_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0062919
IS_B1_A2_B2_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0060201, upper bound: 0.0055312
IS_B1_A2_B2_A2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0056409, upper bound: 0.0055091
IS_B1_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
IS_B1_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0062919
IS_B1_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067099
IS_B1_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0067099
IS_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0064738
IS_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0067024, upper bound: 0.0067099
IS_B2_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
IS_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
IS_B2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0062922
IS_B2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066939, upper bound: 0.0062922
IS_B2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0067141
IS_B2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066940, upper bound: 0.0067141
IS_B2_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_B2_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
IS_B2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0062919
IS_B2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0062919
IS_B2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0067099
IS_B2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0067099
IS_B2_A1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
IS_B2_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066758
IS_B2_A1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
IS_B2_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066811
IS_B2_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0064738
IS_B2_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067024
IS_B2_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067051
IS_B2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067099
IS_B2_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
IS_B2_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
IS_B2_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_B2_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_B2_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
IS_B2_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_B2_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_B2_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_B2_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0062919
IS_B2_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_B2_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0067099
IS_B2_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
IS_B2_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
IS_B2_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
IS_B2_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064564
IS_B2_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_B2_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0067079
IS_B2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0067099

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.11 + 597.40 = 600.51 seconds
