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
execution time: IAR + RelationalAnalysis = 1.18 + 1.90 = 3.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0023347, -0.0040851, -0.0023098, -0.0017721, 0.0017504
1: -0.0055233, -0.0028915, -0.0056444, -0.0028661, -0.0026572, 0.0027529
2: 0.9638253, 0.9716665, 0.9637231, 0.9716778, -0.0078525, 0.0079435
3: 0.0238159, 0.0373223, 0.0227438, 0.0374061, -0.0109453, 0.0112937
4: -0.0035316, -0.0000149, -0.0035380, 0.0000297, -0.0035613, 0.0035231
5: 0.0123270, 0.0147392, 0.0123006, 0.0148216, -0.0024947, 0.0024386
6: 0.0022103, 0.0052426, 0.0021695, 0.0052457, -0.0030355, 0.0030731
7: -0.0174507, -0.0121662, -0.0174724, -0.0121115, -0.0053392, 0.0053061
8: 0.0028846, 0.0072394, 0.0028674, 0.0072852, -0.0044006, 0.0043720
9: 0.0017012, 0.0079076, 0.0016526, 0.0083040, -0.0066028, 0.0062549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0040845, -0.0023108, -0.0040857, -0.0023094, -0.0017751, 0.0017749
1: -0.0056188, -0.0028671, -0.0056657, -0.0028656, -0.0027532, 0.0027986
2: 0.9637273, 0.9716773, 0.9637213, 0.9716780, -0.0079507, 0.0079560
3: 0.0229702, 0.0374027, 0.0225558, 0.0374076, -0.0111413, 0.0118661
4: -0.0035377, 0.0000279, -0.0035381, 0.0000305, -0.0035682, 0.0035660
5: 0.0123017, 0.0148042, 0.0123002, 0.0148361, -0.0025344, 0.0025041
6: 0.0021711, 0.0052456, 0.0021688, 0.0052458, -0.0030747, 0.0030768
7: -0.0174715, -0.0121136, -0.0174728, -0.0121105, -0.0053610, 0.0053591
8: 0.0028681, 0.0072834, 0.0028671, 0.0072861, -0.0044180, 0.0044163
9: 0.0016546, 0.0082203, 0.0016518, 0.0083735, -0.0067190, 0.0065685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
time: 1.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 2, lower bound: -0.0071157, upper bound: 0.0071157

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.20 seconds

## BFS IS instance: IS_A1_B2

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.24 seconds

## BFS IS instance: IS_A2_B1

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.15 seconds

## BFS IS instance: IS_A2_B2

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069117, upper bound: 0.0069084
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2

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
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040819, -0.0023347, -0.0017497, 0.0015972
1: -0.0056184, -0.0030446, -0.0055233, -0.0028915, -0.0027268, 0.0024787
2: 0.9644390, 0.9715982, 0.9638253, 0.9716665, -0.0072275, 0.0077729
3: 0.0229746, 0.0368191, 0.0238159, 0.0373223, -0.0112302, 0.0099550
4: -0.0034933, -0.0002825, -0.0035316, -0.0000149, -0.0034785, 0.0032491
5: 0.0124852, 0.0148039, 0.0123270, 0.0147392, -0.0022541, 0.0024769
6: 0.0024552, 0.0052238, 0.0022103, 0.0052426, -0.0027874, 0.0030135
7: -0.0173202, -0.0124955, -0.0174507, -0.0121662, -0.0051540, 0.0049552
8: 0.0029881, 0.0069641, 0.0028846, 0.0072394, -0.0042513, 0.0040795
9: 0.0019926, 0.0082187, 0.0017012, 0.0079076, -0.0059149, 0.0065175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040819, -0.0023644, -0.0017220, 0.0015943
1: -0.0056918, -0.0030476, -0.0055232, -0.0029219, -0.0027699, 0.0024756
2: 0.9644510, 0.9715968, 0.9639469, 0.9716529, -0.0072019, 0.0076500
3: 0.0223248, 0.0368093, 0.0238171, 0.0372227, -0.0118711, 0.0101989
4: -0.0034926, -0.0002877, -0.0035240, -0.0000678, -0.0034247, 0.0032363
5: 0.0124883, 0.0148539, 0.0123583, 0.0147391, -0.0022509, 0.0024956
6: 0.0024600, 0.0052234, 0.0022588, 0.0052389, -0.0027789, 0.0029646
7: -0.0173177, -0.0125019, -0.0174248, -0.0122315, -0.0050862, 0.0049229
8: 0.0029901, 0.0069587, 0.0029051, 0.0071849, -0.0041948, 0.0040536
9: 0.0019983, 0.0084590, 0.0017589, 0.0079071, -0.0059088, 0.0067001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1

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
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
time: 1.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.34 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0068993
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0068993, upper bound: 0.0069084
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0068993
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 2, lower bound: -0.0069084, upper bound: 0.0069084

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067797
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040819, -0.0025100, -0.0015740, 0.0015797
1: -0.0055997, -0.0030624, -0.0055228, -0.0030704, -0.0025293, 0.0024604
2: 0.9645107, 0.9715902, 0.9645426, 0.9715868, -0.0070761, 0.0070477
3: 0.0231401, 0.0367604, 0.0238203, 0.0367342, -0.0104070, 0.0097690
4: -0.0034889, -0.0003137, -0.0034869, -0.0003276, -0.0031613, 0.0031732
5: 0.0125036, 0.0147912, 0.0125119, 0.0147389, -0.0022353, 0.0022793
6: 0.0024838, 0.0052216, 0.0024965, 0.0052206, -0.0027368, 0.0027251
7: -0.0173050, -0.0125339, -0.0172982, -0.0125510, -0.0047540, 0.0047643
8: 0.0030002, 0.0069320, 0.0030055, 0.0069177, -0.0039175, 0.0039264
9: 0.0020267, 0.0081575, 0.0020418, 0.0079059, -0.0058793, 0.0061157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040839, -0.0025022, -0.0015817, 0.0015817
1: -0.0055997, -0.0030624, -0.0055997, -0.0030624, -0.0025372, 0.0025372
2: 0.9645107, 0.9715902, 0.9645107, 0.9715902, -0.0070795, 0.0070795
3: 0.0231401, 0.0367604, 0.0231401, 0.0367604, -0.0098418, 0.0098418
4: -0.0034889, -0.0003137, -0.0034889, -0.0003137, -0.0031752, 0.0031752
5: 0.0125036, 0.0147912, 0.0125036, 0.0147912, -0.0022876, 0.0022876
6: 0.0024838, 0.0052216, 0.0024838, 0.0052216, -0.0027378, 0.0027378
7: -0.0173050, -0.0125339, -0.0173050, -0.0125339, -0.0047711, 0.0047711
8: 0.0030002, 0.0069320, 0.0030002, 0.0069320, -0.0039318, 0.0039318
9: 0.0020267, 0.0081575, 0.0020267, 0.0081575, -0.0061308, 0.0061308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067797
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040844, -0.0024847, -0.0015993, 0.0015822
1: -0.0055997, -0.0030624, -0.0056184, -0.0030446, -0.0025551, 0.0025559
2: 0.9645107, 0.9715902, 0.9644390, 0.9715982, -0.0070875, 0.0071512
3: 0.0231401, 0.0367604, 0.0229746, 0.0368191, -0.0108397, 0.0108486
4: -0.0034889, -0.0003137, -0.0034933, -0.0002825, -0.0032064, 0.0031796
5: 0.0125036, 0.0147912, 0.0124852, 0.0148039, -0.0023003, 0.0023060
6: 0.0024838, 0.0052216, 0.0024552, 0.0052238, -0.0027400, 0.0027664
7: -0.0173050, -0.0125339, -0.0173202, -0.0124955, -0.0048096, 0.0047863
8: 0.0030002, 0.0069320, 0.0029881, 0.0069641, -0.0039640, 0.0039439
9: 0.0020267, 0.0081575, 0.0019926, 0.0082187, -0.0061920, 0.0061648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025022, -0.0040864, -0.0024876, -0.0015963, 0.0015842
1: -0.0055997, -0.0030624, -0.0056918, -0.0030476, -0.0025521, 0.0026293
2: 0.9645107, 0.9715902, 0.9644510, 0.9715968, -0.0070862, 0.0071392
3: 0.0231401, 0.0367604, 0.0223248, 0.0368093, -0.0102478, 0.0109164
4: -0.0034889, -0.0003137, -0.0034926, -0.0002877, -0.0032012, 0.0031789
5: 0.0125036, 0.0147912, 0.0124883, 0.0148539, -0.0023502, 0.0023029
6: 0.0024838, 0.0052216, 0.0024600, 0.0052234, -0.0027396, 0.0027616
7: -0.0173050, -0.0125339, -0.0173177, -0.0125019, -0.0048031, 0.0047838
8: 0.0030002, 0.0069320, 0.0029901, 0.0069587, -0.0039586, 0.0039419
9: 0.0020267, 0.0081575, 0.0019983, 0.0084590, -0.0064323, 0.0061591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1

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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067802
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0024847, -0.0040839, -0.0025022, -0.0015822, 0.0015993
1: -0.0056184, -0.0030446, -0.0055997, -0.0030624, -0.0025559, 0.0025551
2: 0.9644390, 0.9715982, 0.9645107, 0.9715902, -0.0071512, 0.0070875
3: 0.0229746, 0.0368191, 0.0231401, 0.0367604, -0.0108486, 0.0108397
4: -0.0034933, -0.0002825, -0.0034889, -0.0003137, -0.0031796, 0.0032064
5: 0.0124852, 0.0148039, 0.0125036, 0.0147912, -0.0023060, 0.0023003
6: 0.0024552, 0.0052238, 0.0024838, 0.0052216, -0.0027664, 0.0027400
7: -0.0173202, -0.0124955, -0.0173050, -0.0125339, -0.0047863, 0.0048096
8: 0.0029881, 0.0069641, 0.0030002, 0.0069320, -0.0039439, 0.0039640
9: 0.0019926, 0.0082187, 0.0020267, 0.0081575, -0.0061648, 0.0061920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067734
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040839, -0.0025022, -0.0015842, 0.0015963
1: -0.0056918, -0.0030476, -0.0055997, -0.0030624, -0.0026293, 0.0025521
2: 0.9644510, 0.9715968, 0.9645107, 0.9715902, -0.0071392, 0.0070862
3: 0.0223248, 0.0368093, 0.0231401, 0.0367604, -0.0109164, 0.0102478
4: -0.0034926, -0.0002877, -0.0034889, -0.0003137, -0.0031789, 0.0032012
5: 0.0124883, 0.0148539, 0.0125036, 0.0147912, -0.0023029, 0.0023502
6: 0.0024600, 0.0052234, 0.0024838, 0.0052216, -0.0027616, 0.0027396
7: -0.0173177, -0.0125019, -0.0173050, -0.0125339, -0.0047838, 0.0048031
8: 0.0029901, 0.0069587, 0.0030002, 0.0069320, -0.0039419, 0.0039586
9: 0.0019983, 0.0084590, 0.0020267, 0.0081575, -0.0061591, 0.0064323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067802
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1_B2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040844, -0.0024847, -0.0016017, 0.0015968
1: -0.0056918, -0.0030476, -0.0056184, -0.0030446, -0.0026472, 0.0025708
2: 0.9644510, 0.9715968, 0.9644390, 0.9715982, -0.0071472, 0.0071578
3: 0.0223248, 0.0368093, 0.0229746, 0.0368191, -0.0108475, 0.0101815
4: -0.0034926, -0.0002877, -0.0034933, -0.0002825, -0.0032101, 0.0032056
5: 0.0124883, 0.0148539, 0.0124852, 0.0148039, -0.0023157, 0.0023687
6: 0.0024600, 0.0052234, 0.0024552, 0.0052238, -0.0027638, 0.0027682
7: -0.0173177, -0.0125019, -0.0173202, -0.0124955, -0.0048222, 0.0048183
8: 0.0029901, 0.0069587, 0.0029881, 0.0069641, -0.0039740, 0.0039706
9: 0.0019983, 0.0084590, 0.0019926, 0.0082187, -0.0062203, 0.0064663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067734
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0024876, -0.0040864, -0.0024876, -0.0015988, 0.0015988
1: -0.0056918, -0.0030476, -0.0056918, -0.0030476, -0.0026442, 0.0026442
2: 0.9644510, 0.9715968, 0.9644510, 0.9715968, -0.0071458, 0.0071458
3: 0.0223248, 0.0368093, 0.0223248, 0.0368093, -0.0102490, 0.0102490
4: -0.0034926, -0.0002877, -0.0034926, -0.0002877, -0.0032049, 0.0032049
5: 0.0124883, 0.0148539, 0.0124883, 0.0148539, -0.0023656, 0.0023656
6: 0.0024600, 0.0052234, 0.0024600, 0.0052234, -0.0027634, 0.0027634
7: -0.0173177, -0.0125019, -0.0173177, -0.0125019, -0.0048158, 0.0048158
8: 0.0029901, 0.0069587, 0.0029901, 0.0069587, -0.0039686, 0.0039686
9: 0.0019983, 0.0084590, 0.0019983, 0.0084590, -0.0064606, 0.0064606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
time: 1.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.40 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067797
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067797
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067715
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067802
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067734
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067802
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0067977
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068119, upper bound: 0.0068059
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067734
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0067977
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 2, lower bound: -0.0068059, upper bound: 0.0068059

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040819, -0.0025620, -0.0015204, 0.0012768
1: -0.0055390, -0.0033715, -0.0055223, -0.0031235, -0.0024156, 0.0021507
2: 0.9657504, 0.9714524, 0.9647554, 0.9715630, -0.0058126, 0.0066970
3: 0.0236766, 0.0357439, 0.0238252, 0.0365597, -0.0091793, 0.0083599
4: -0.0034116, -0.0008542, -0.0034736, -0.0004204, -0.0029912, 0.0026194
5: 0.0128232, 0.0147499, 0.0125667, 0.0147385, -0.0019154, 0.0021832
6: 0.0029785, 0.0051836, 0.0025815, 0.0052141, -0.0022356, 0.0026021
7: -0.0170416, -0.0131989, -0.0172530, -0.0126652, -0.0043765, 0.0040542
8: 0.0032091, 0.0063759, 0.0030414, 0.0068222, -0.0036131, 0.0033345
9: 0.0026154, 0.0079591, 0.0021429, 0.0079041, -0.0052887, 0.0058162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040819, -0.0025100, -0.0015719, 0.0014898
1: -0.0055221, -0.0031542, -0.0055228, -0.0030704, -0.0024517, 0.0023686
2: 0.9648787, 0.9715494, 0.9645426, 0.9715868, -0.0067081, 0.0070068
3: 0.0238268, 0.0364587, 0.0238203, 0.0367342, -0.0095013, 0.0086593
4: -0.0034659, -0.0004741, -0.0034869, -0.0003276, -0.0031383, 0.0030127
5: 0.0125985, 0.0147384, 0.0125119, 0.0147389, -0.0021404, 0.0022265
6: 0.0026306, 0.0052103, 0.0024965, 0.0052206, -0.0025900, 0.0027138
7: -0.0172268, -0.0127313, -0.0172982, -0.0125510, -0.0046758, 0.0045670
8: 0.0030622, 0.0067669, 0.0030055, 0.0069177, -0.0038555, 0.0037614
9: 0.0022014, 0.0079035, 0.0020418, 0.0079059, -0.0057045, 0.0058617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040819, -0.0025620, -0.0015224, 0.0012849
1: -0.0056158, -0.0033633, -0.0055223, -0.0031235, -0.0024923, 0.0021589
2: 0.9657173, 0.9714562, 0.9647554, 0.9715630, -0.0058457, 0.0067008
3: 0.0229975, 0.0357711, 0.0238252, 0.0365597, -0.0101051, 0.0086243
4: -0.0034136, -0.0008398, -0.0034736, -0.0004204, -0.0029932, 0.0026338
5: 0.0128146, 0.0148021, 0.0125667, 0.0147385, -0.0019239, 0.0022354
6: 0.0029653, 0.0051846, 0.0025815, 0.0052141, -0.0022488, 0.0026032
7: -0.0170486, -0.0131811, -0.0172530, -0.0126652, -0.0043835, 0.0040719
8: 0.0032036, 0.0063907, 0.0030414, 0.0068222, -0.0036186, 0.0033493
9: 0.0025997, 0.0082102, 0.0021429, 0.0079041, -0.0053044, 0.0060673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064953
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066793
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040819, -0.0025100, -0.0015739, 0.0014970
1: -0.0055989, -0.0031468, -0.0055228, -0.0030704, -0.0025285, 0.0023760
2: 0.9648491, 0.9715527, 0.9645426, 0.9715868, -0.0067377, 0.0070101
3: 0.0231467, 0.0364830, 0.0238203, 0.0367342, -0.0103967, 0.0089353
4: -0.0034678, -0.0004612, -0.0034869, -0.0003276, -0.0031402, 0.0030257
5: 0.0125908, 0.0147907, 0.0125119, 0.0147389, -0.0021481, 0.0022788
6: 0.0026188, 0.0052112, 0.0024965, 0.0052206, -0.0026018, 0.0027147
7: -0.0172331, -0.0127154, -0.0172982, -0.0125510, -0.0046821, 0.0045829
8: 0.0030572, 0.0067802, 0.0030055, 0.0069177, -0.0038605, 0.0037747
9: 0.0021873, 0.0081550, 0.0020418, 0.0079059, -0.0057186, 0.0061132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0068119
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040839, -0.0025538, -0.0015306, 0.0012870
1: -0.0056158, -0.0033633, -0.0055991, -0.0031151, -0.0025007, 0.0022358
2: 0.9657173, 0.9714562, 0.9647219, 0.9715667, -0.0058494, 0.0067343
3: 0.0229975, 0.0357711, 0.0231453, 0.0365872, -0.0095118, 0.0086862
4: -0.0034136, -0.0008398, -0.0034757, -0.0004058, -0.0030078, 0.0026359
5: 0.0128146, 0.0148021, 0.0125581, 0.0147908, -0.0019762, 0.0022441
6: 0.0029653, 0.0051846, 0.0025681, 0.0052151, -0.0022498, 0.0026165
7: -0.0170486, -0.0131811, -0.0172601, -0.0126472, -0.0044015, 0.0040791
8: 0.0032036, 0.0063907, 0.0030358, 0.0068372, -0.0036337, 0.0033550
9: 0.0025997, 0.0082102, 0.0021270, 0.0081555, -0.0055559, 0.0060832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040839, -0.0025022, -0.0015817, 0.0014991
1: -0.0055989, -0.0031468, -0.0055997, -0.0030624, -0.0025365, 0.0024528
2: 0.9648491, 0.9715527, 0.9645107, 0.9715902, -0.0067412, 0.0070420
3: 0.0231467, 0.0364830, 0.0231401, 0.0367604, -0.0098323, 0.0089989
4: -0.0034678, -0.0004612, -0.0034889, -0.0003137, -0.0031541, 0.0030276
5: 0.0125908, 0.0147907, 0.0125036, 0.0147912, -0.0022004, 0.0022870
6: 0.0026188, 0.0052112, 0.0024838, 0.0052216, -0.0026028, 0.0027274
7: -0.0172331, -0.0127154, -0.0173050, -0.0125339, -0.0046992, 0.0045896
8: 0.0030572, 0.0067802, 0.0030002, 0.0069320, -0.0038748, 0.0037800
9: 0.0021873, 0.0081550, 0.0020267, 0.0081575, -0.0059701, 0.0061284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0063906
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0068119
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0063901
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0068059
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040844, -0.0025351, -0.0015493, 0.0012875
1: -0.0056158, -0.0033633, -0.0056177, -0.0030960, -0.0025197, 0.0022544
2: 0.9657173, 0.9714562, 0.9646454, 0.9715753, -0.0058579, 0.0068108
3: 0.0229975, 0.0357711, 0.0229800, 0.0366500, -0.0105708, 0.0097036
4: -0.0034136, -0.0008398, -0.0034805, -0.0003724, -0.0030412, 0.0026407
5: 0.0128146, 0.0148021, 0.0125383, 0.0148035, -0.0019889, 0.0022638
6: 0.0029653, 0.0051846, 0.0025375, 0.0052175, -0.0022522, 0.0026471
7: -0.0170486, -0.0131811, -0.0172764, -0.0126061, -0.0044425, 0.0040953
8: 0.0032036, 0.0063907, 0.0030229, 0.0068716, -0.0036680, 0.0033679
9: 0.0025997, 0.0082102, 0.0020906, 0.0082167, -0.0056170, 0.0061196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064953
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066793
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040844, -0.0024847, -0.0015992, 0.0014996
1: -0.0055989, -0.0031468, -0.0056184, -0.0030446, -0.0025543, 0.0024715
2: 0.9648491, 0.9715527, 0.9644390, 0.9715982, -0.0067492, 0.0071136
3: 0.0231467, 0.0364830, 0.0229746, 0.0368191, -0.0108294, 0.0102182
4: -0.0034678, -0.0004612, -0.0034933, -0.0002825, -0.0031853, 0.0030321
5: 0.0125908, 0.0147907, 0.0124852, 0.0148039, -0.0022131, 0.0023055
6: 0.0026188, 0.0052112, 0.0024552, 0.0052238, -0.0026050, 0.0027560
7: -0.0172331, -0.0127154, -0.0173202, -0.0124955, -0.0047377, 0.0046049
8: 0.0030572, 0.0067802, 0.0029881, 0.0069641, -0.0039069, 0.0037921
9: 0.0021873, 0.0081550, 0.0019926, 0.0082187, -0.0060313, 0.0061624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067911, upper bound: 0.0063906
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067911, upper bound: 0.0068119
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040864, -0.0025380, -0.0015464, 0.0012894
1: -0.0056158, -0.0033633, -0.0056911, -0.0030990, -0.0025167, 0.0023278
2: 0.9657173, 0.9714562, 0.9646574, 0.9715739, -0.0058566, 0.0067988
3: 0.0229975, 0.0357711, 0.0223306, 0.0366401, -0.0099495, 0.0097606
4: -0.0034136, -0.0008398, -0.0034797, -0.0003776, -0.0030360, 0.0026399
5: 0.0128146, 0.0148021, 0.0125414, 0.0148534, -0.0020388, 0.0022607
6: 0.0029653, 0.0051846, 0.0025423, 0.0052171, -0.0022518, 0.0026423
7: -0.0170486, -0.0131811, -0.0172739, -0.0126126, -0.0044361, 0.0040928
8: 0.0032036, 0.0063907, 0.0030249, 0.0068662, -0.0036626, 0.0033659
9: 0.0025997, 0.0082102, 0.0020963, 0.0084568, -0.0058571, 0.0061139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040864, -0.0024876, -0.0015963, 0.0015016
1: -0.0055989, -0.0031468, -0.0056918, -0.0030476, -0.0025513, 0.0025450
2: 0.9648491, 0.9715527, 0.9644510, 0.9715968, -0.0067478, 0.0071017
3: 0.0231467, 0.0364830, 0.0223248, 0.0368093, -0.0102383, 0.0102733
4: -0.0034678, -0.0004612, -0.0034926, -0.0002877, -0.0031801, 0.0030314
5: 0.0125908, 0.0147907, 0.0124883, 0.0148539, -0.0022630, 0.0023024
6: 0.0026188, 0.0052112, 0.0024600, 0.0052234, -0.0026046, 0.0027512
7: -0.0172331, -0.0127154, -0.0173177, -0.0125019, -0.0047312, 0.0046023
8: 0.0030572, 0.0067802, 0.0029901, 0.0069587, -0.0039015, 0.0037901
9: 0.0021873, 0.0081550, 0.0019983, 0.0084590, -0.0062716, 0.0061567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063901
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0068059
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040819, -0.0025620, -0.0015230, 0.0013087
1: -0.0056384, -0.0033390, -0.0055223, -0.0031235, -0.0025149, 0.0021832
2: 0.9656199, 0.9714670, 0.9647554, 0.9715630, -0.0059432, 0.0067116
3: 0.0227976, 0.0358510, 0.0238252, 0.0365597, -0.0104924, 0.0089530
4: -0.0034197, -0.0007973, -0.0034736, -0.0004204, -0.0029993, 0.0026763
5: 0.0127895, 0.0148175, 0.0125667, 0.0147385, -0.0019490, 0.0022508
6: 0.0029264, 0.0051876, 0.0025815, 0.0052141, -0.0022877, 0.0026061
7: -0.0170693, -0.0131288, -0.0172530, -0.0126652, -0.0044042, 0.0041242
8: 0.0031871, 0.0064344, 0.0030414, 0.0068222, -0.0036351, 0.0033930
9: 0.0025534, 0.0082841, 0.0021429, 0.0079041, -0.0053507, 0.0061412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040819, -0.0025100, -0.0015744, 0.0015115
1: -0.0056175, -0.0031321, -0.0055228, -0.0030704, -0.0025471, 0.0023907
2: 0.9647900, 0.9715592, 0.9645426, 0.9715868, -0.0067967, 0.0070167
3: 0.0229819, 0.0365314, 0.0238203, 0.0367342, -0.0105803, 0.0091870
4: -0.0034715, -0.0004355, -0.0034869, -0.0003276, -0.0031438, 0.0030514
5: 0.0125756, 0.0148033, 0.0125119, 0.0147389, -0.0021633, 0.0022915
6: 0.0025953, 0.0052130, 0.0024965, 0.0052206, -0.0026253, 0.0027165
7: -0.0172457, -0.0126837, -0.0172982, -0.0125510, -0.0046947, 0.0046145
8: 0.0030472, 0.0068067, 0.0030055, 0.0069177, -0.0038704, 0.0038011
9: 0.0021593, 0.0082159, 0.0020418, 0.0079059, -0.0057466, 0.0061741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040839, -0.0025022, -0.0015822, 0.0015135
1: -0.0056175, -0.0031321, -0.0055997, -0.0030624, -0.0025551, 0.0024676
2: 0.9647900, 0.9715592, 0.9645107, 0.9715902, -0.0068002, 0.0070485
3: 0.0229819, 0.0365314, 0.0231401, 0.0367604, -0.0108388, 0.0101243
4: -0.0034715, -0.0004355, -0.0034889, -0.0003137, -0.0031578, 0.0030534
5: 0.0125756, 0.0148033, 0.0125036, 0.0147912, -0.0022156, 0.0022997
6: 0.0025953, 0.0052130, 0.0024838, 0.0052216, -0.0026263, 0.0027292
7: -0.0172457, -0.0126837, -0.0173050, -0.0125339, -0.0047118, 0.0046213
8: 0.0030472, 0.0068067, 0.0030002, 0.0069320, -0.0038847, 0.0038065
9: 0.0021593, 0.0082159, 0.0020267, 0.0081575, -0.0059982, 0.0061893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0068059
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040819, -0.0025620, -0.0015249, 0.0013039
1: -0.0057107, -0.0033440, -0.0055223, -0.0031235, -0.0025872, 0.0021783
2: 0.9656399, 0.9714647, 0.9647554, 0.9715630, -0.0059232, 0.0067093
3: 0.0221572, 0.0358347, 0.0238252, 0.0365597, -0.0111918, 0.0089798
4: -0.0034185, -0.0008060, -0.0034736, -0.0004204, -0.0029981, 0.0026676
5: 0.0127946, 0.0148667, 0.0125667, 0.0147385, -0.0019439, 0.0023000
6: 0.0029344, 0.0051870, 0.0025815, 0.0052141, -0.0022797, 0.0026055
7: -0.0170651, -0.0131395, -0.0172530, -0.0126652, -0.0044000, 0.0041135
8: 0.0031905, 0.0064255, 0.0030414, 0.0068222, -0.0036317, 0.0033841
9: 0.0025629, 0.0085209, 0.0021429, 0.0079041, -0.0053412, 0.0063780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040819, -0.0025100, -0.0015764, 0.0015092
1: -0.0056909, -0.0031344, -0.0055228, -0.0030704, -0.0026205, 0.0023884
2: 0.9647995, 0.9715582, 0.9645426, 0.9715868, -0.0067873, 0.0070156
3: 0.0223324, 0.0365237, 0.0238203, 0.0367342, -0.0112963, 0.0092195
4: -0.0034709, -0.0004396, -0.0034869, -0.0003276, -0.0031432, 0.0030473
5: 0.0125780, 0.0148533, 0.0125119, 0.0147389, -0.0021609, 0.0023414
6: 0.0025990, 0.0052127, 0.0024965, 0.0052206, -0.0026216, 0.0027162
7: -0.0172437, -0.0126887, -0.0172982, -0.0125510, -0.0046927, 0.0046095
8: 0.0030488, 0.0068025, 0.0030055, 0.0069177, -0.0038688, 0.0037969
9: 0.0021638, 0.0084561, 0.0020418, 0.0079059, -0.0057421, 0.0064143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0068119
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040839, -0.0025022, -0.0015842, 0.0015112
1: -0.0056909, -0.0031344, -0.0055997, -0.0030624, -0.0026285, 0.0024652
2: 0.9647995, 0.9715582, 0.9645107, 0.9715902, -0.0067908, 0.0070475
3: 0.0223324, 0.0365237, 0.0231401, 0.0367604, -0.0109066, 0.0095000
4: -0.0034709, -0.0004396, -0.0034889, -0.0003137, -0.0031572, 0.0030493
5: 0.0125780, 0.0148533, 0.0125036, 0.0147912, -0.0022131, 0.0023496
6: 0.0025990, 0.0052127, 0.0024838, 0.0052216, -0.0026226, 0.0027289
7: -0.0172437, -0.0126887, -0.0173050, -0.0125339, -0.0047098, 0.0046163
8: 0.0030488, 0.0068025, 0.0030002, 0.0069320, -0.0038831, 0.0038023
9: 0.0021638, 0.0084561, 0.0020267, 0.0081575, -0.0059937, 0.0064295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0068059
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040844, -0.0025351, -0.0015518, 0.0013064
1: -0.0057107, -0.0033440, -0.0056177, -0.0030960, -0.0026147, 0.0022738
2: 0.9656399, 0.9714647, 0.9646454, 0.9715753, -0.0059354, 0.0068193
3: 0.0221572, 0.0358347, 0.0229800, 0.0366500, -0.0105607, 0.0090502
4: -0.0034185, -0.0008060, -0.0034805, -0.0003724, -0.0030460, 0.0026745
5: 0.0127946, 0.0148667, 0.0125383, 0.0148035, -0.0020089, 0.0023284
6: 0.0029344, 0.0051870, 0.0025375, 0.0052175, -0.0022831, 0.0026495
7: -0.0170651, -0.0131395, -0.0172764, -0.0126061, -0.0044590, 0.0041369
8: 0.0031905, 0.0064255, 0.0030229, 0.0068716, -0.0036811, 0.0034027
9: 0.0025629, 0.0085209, 0.0020906, 0.0082167, -0.0056538, 0.0064303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040844, -0.0024847, -0.0016017, 0.0015117
1: -0.0056909, -0.0031344, -0.0056184, -0.0030446, -0.0026463, 0.0024839
2: 0.9647995, 0.9715582, 0.9644390, 0.9715982, -0.0067987, 0.0071191
3: 0.0223324, 0.0365237, 0.0229746, 0.0368191, -0.0108368, 0.0093497
4: -0.0034709, -0.0004396, -0.0034933, -0.0002825, -0.0031884, 0.0030538
5: 0.0125780, 0.0148533, 0.0124852, 0.0148039, -0.0022259, 0.0023681
6: 0.0025990, 0.0052127, 0.0024552, 0.0052238, -0.0026248, 0.0027575
7: -0.0172437, -0.0126887, -0.0173202, -0.0124955, -0.0047482, 0.0046315
8: 0.0030488, 0.0068025, 0.0029881, 0.0069641, -0.0039153, 0.0038144
9: 0.0021638, 0.0084561, 0.0019926, 0.0082187, -0.0060549, 0.0064635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040864, -0.0025380, -0.0015489, 0.0013084
1: -0.0057107, -0.0033440, -0.0056911, -0.0030990, -0.0026117, 0.0023471
2: 0.9656399, 0.9714647, 0.9646574, 0.9715739, -0.0059341, 0.0068073
3: 0.0221572, 0.0358347, 0.0223306, 0.0366401, -0.0099443, 0.0091138
4: -0.0034185, -0.0008060, -0.0034797, -0.0003776, -0.0030408, 0.0026737
5: 0.0127946, 0.0148667, 0.0125414, 0.0148534, -0.0020588, 0.0023253
6: 0.0029344, 0.0051870, 0.0025423, 0.0052171, -0.0022827, 0.0026447
7: -0.0170651, -0.0131395, -0.0172739, -0.0126126, -0.0044525, 0.0041344
8: 0.0031905, 0.0064255, 0.0030249, 0.0068662, -0.0036757, 0.0034006
9: 0.0025629, 0.0085209, 0.0020963, 0.0084568, -0.0058939, 0.0064246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040864, -0.0024876, -0.0015987, 0.0015137
1: -0.0056909, -0.0031344, -0.0056918, -0.0030476, -0.0026433, 0.0025573
2: 0.9647995, 0.9715582, 0.9644510, 0.9715968, -0.0067974, 0.0071071
3: 0.0223324, 0.0365237, 0.0223248, 0.0368093, -0.0102391, 0.0093988
4: -0.0034709, -0.0004396, -0.0034926, -0.0002877, -0.0031832, 0.0030530
5: 0.0125780, 0.0148533, 0.0124883, 0.0148539, -0.0022758, 0.0023650
6: 0.0025990, 0.0052127, 0.0024600, 0.0052234, -0.0026244, 0.0027527
7: -0.0172437, -0.0126887, -0.0173177, -0.0125019, -0.0047418, 0.0046289
8: 0.0030488, 0.0068025, 0.0029901, 0.0069587, -0.0039099, 0.0038124
9: 0.0021638, 0.0084561, 0.0019983, 0.0084590, -0.0062952, 0.0064578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0068059
time: 1.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.77 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064953
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066793
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0068119
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065239
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066935
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0063906
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067923, upper bound: 0.0068119
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067907
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0063901
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067787, upper bound: 0.0068059
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064953
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066793
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067911, upper bound: 0.0063906
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067911, upper bound: 0.0068119
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067760
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0063901
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067878, upper bound: 0.0068059
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0068059
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0068119
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0068059
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0065243
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066940
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0063906
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067919, upper bound: 0.0068119
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0063901
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063906, upper bound: 0.0067911
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067761, upper bound: 0.0063901
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064972
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066822
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067907, upper bound: 0.0063906
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0063901
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0063901, upper bound: 0.0067789
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0063901
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 2, lower bound: -0.0067867, upper bound: 0.0068059

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028285, -0.0040818, -0.0027495, -0.0013328, 0.0012533
1: -0.0055390, -0.0033955, -0.0055181, -0.0033149, -0.0022241, 0.0021227
2: 0.9658464, 0.9714418, 0.9655231, 0.9714777, -0.0056314, 0.0059187
3: 0.0236768, 0.0356653, 0.0238616, 0.0359304, -0.0085609, 0.0082055
4: -0.0034056, -0.0008960, -0.0034257, -0.0007551, -0.0026505, 0.0025297
5: 0.0128479, 0.0147499, 0.0127646, 0.0147357, -0.0018879, 0.0019854
6: 0.0030168, 0.0051807, 0.0028878, 0.0051906, -0.0021738, 0.0022929
7: -0.0170212, -0.0132503, -0.0170899, -0.0130769, -0.0039443, 0.0038396
8: 0.0032253, 0.0063329, 0.0031708, 0.0064779, -0.0032526, 0.0031621
9: 0.0026610, 0.0079590, 0.0025074, 0.0078907, -0.0052297, 0.0054515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040819, -0.0026449, -0.0014374, 0.0012768
1: -0.0055390, -0.0033715, -0.0055220, -0.0032081, -0.0023309, 0.0021505
2: 0.9657504, 0.9714524, 0.9650951, 0.9715253, -0.0057749, 0.0063573
3: 0.0236766, 0.0357439, 0.0238274, 0.0362813, -0.0086574, 0.0083540
4: -0.0034116, -0.0008542, -0.0034524, -0.0005685, -0.0028431, 0.0025982
5: 0.0128232, 0.0147499, 0.0126542, 0.0147384, -0.0019152, 0.0020957
6: 0.0029785, 0.0051836, 0.0027170, 0.0052037, -0.0022252, 0.0024666
7: -0.0170416, -0.0131989, -0.0171809, -0.0128473, -0.0041943, 0.0039820
8: 0.0032091, 0.0063759, 0.0030987, 0.0066699, -0.0034607, 0.0032772
9: 0.0026154, 0.0079591, 0.0023042, 0.0079033, -0.0052879, 0.0056549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0025921, -0.0040823, -0.0028050, -0.0012768, 0.0014902
1: -0.0055221, -0.0031542, -0.0055390, -0.0033715, -0.0021505, 0.0023848
2: 0.9648787, 0.9715494, 0.9657504, 0.9714524, -0.0065737, 0.0057990
3: 0.0238268, 0.0364587, 0.0236766, 0.0357439, -0.0083571, 0.0092283
4: -0.0034659, -0.0004741, -0.0034116, -0.0008542, -0.0026117, 0.0029374
5: 0.0125985, 0.0147384, 0.0128232, 0.0147499, -0.0021515, 0.0019152
6: 0.0026306, 0.0052103, 0.0029785, 0.0051836, -0.0025530, 0.0022318
7: -0.0172268, -0.0127313, -0.0170416, -0.0131989, -0.0040280, 0.0043104
8: 0.0030622, 0.0067669, 0.0032091, 0.0063759, -0.0033137, 0.0035578
9: 0.0022014, 0.0079035, 0.0026154, 0.0079591, -0.0057576, 0.0052881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040839, -0.0025848, -0.0014975, 0.0012789
1: -0.0055390, -0.0033715, -0.0055989, -0.0031468, -0.0023922, 0.0022274
2: 0.9657504, 0.9714524, 0.9648491, 0.9715527, -0.0058023, 0.0066034
3: 0.0236766, 0.0357439, 0.0231467, 0.0364830, -0.0094505, 0.0092525
4: -0.0034116, -0.0008542, -0.0034678, -0.0004612, -0.0029503, 0.0026135
5: 0.0128232, 0.0147499, 0.0125908, 0.0147907, -0.0019675, 0.0021591
6: 0.0029785, 0.0051836, 0.0026188, 0.0052112, -0.0022327, 0.0025648
7: -0.0170416, -0.0131989, -0.0172331, -0.0127154, -0.0043262, 0.0040343
8: 0.0032091, 0.0063759, 0.0030572, 0.0067802, -0.0035710, 0.0033187
9: 0.0026154, 0.0079591, 0.0021873, 0.0081550, -0.0055396, 0.0057717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028212, -0.0040818, -0.0027495, -0.0013349, 0.0012606
1: -0.0056157, -0.0033881, -0.0055181, -0.0033149, -0.0023009, 0.0021301
2: 0.9658166, 0.9714451, 0.9655231, 0.9714777, -0.0056611, 0.0059220
3: 0.0229978, 0.0356897, 0.0238616, 0.0359304, -0.0094865, 0.0084666
4: -0.0034074, -0.0008831, -0.0034257, -0.0007551, -0.0026523, 0.0025427
5: 0.0128402, 0.0148021, 0.0127646, 0.0147357, -0.0018955, 0.0020376
6: 0.0030049, 0.0051816, 0.0028878, 0.0051906, -0.0021856, 0.0022938
7: -0.0170275, -0.0132344, -0.0170899, -0.0130769, -0.0039507, 0.0038556
8: 0.0032203, 0.0063462, 0.0031708, 0.0064779, -0.0032576, 0.0031754
9: 0.0026468, 0.0082101, 0.0025074, 0.0078907, -0.0052438, 0.0057027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0059736, upper bound: 0.0057816
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055301, upper bound: 0.0056970
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040819, -0.0026449, -0.0014394, 0.0012849
1: -0.0056158, -0.0033633, -0.0055220, -0.0032081, -0.0024076, 0.0021587
2: 0.9657173, 0.9714562, 0.9650951, 0.9715253, -0.0058080, 0.0063611
3: 0.0229975, 0.0357711, 0.0238274, 0.0362813, -0.0095381, 0.0086184
4: -0.0034136, -0.0008398, -0.0034524, -0.0005685, -0.0028452, 0.0026126
5: 0.0128146, 0.0148021, 0.0126542, 0.0147384, -0.0019237, 0.0021479
6: 0.0029653, 0.0051846, 0.0027170, 0.0052037, -0.0022384, 0.0024676
7: -0.0170486, -0.0131811, -0.0171809, -0.0128473, -0.0042013, 0.0039998
8: 0.0032036, 0.0063907, 0.0030987, 0.0066699, -0.0034663, 0.0032921
9: 0.0025997, 0.0082102, 0.0023042, 0.0079033, -0.0053036, 0.0059060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066793
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040823, -0.0028050, -0.0012789, 0.0014975
1: -0.0055989, -0.0031468, -0.0055390, -0.0033715, -0.0022274, 0.0023922
2: 0.9648491, 0.9715527, 0.9657504, 0.9714524, -0.0066034, 0.0058023
3: 0.0231467, 0.0364830, 0.0236766, 0.0357439, -0.0092525, 0.0094505
4: -0.0034678, -0.0004612, -0.0034116, -0.0008542, -0.0026135, 0.0029503
5: 0.0125908, 0.0147907, 0.0128232, 0.0147499, -0.0021591, 0.0019675
6: 0.0026188, 0.0052112, 0.0029785, 0.0051836, -0.0025648, 0.0022327
7: -0.0172331, -0.0127154, -0.0170416, -0.0131989, -0.0040343, 0.0043262
8: 0.0030572, 0.0067802, 0.0032091, 0.0063759, -0.0033187, 0.0035710
9: 0.0021873, 0.0081550, 0.0026154, 0.0079591, -0.0057717, 0.0055396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062910
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0067072
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0067141
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065983
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040844, -0.0027970, -0.0012870, 0.0014995
1: -0.0055989, -0.0031468, -0.0056158, -0.0033633, -0.0022356, 0.0024690
2: 0.9648491, 0.9715527, 0.9657173, 0.9714562, -0.0066071, 0.0058354
3: 0.0231467, 0.0364830, 0.0229975, 0.0357711, -0.0086835, 0.0095562
4: -0.0034678, -0.0004612, -0.0034136, -0.0008398, -0.0026280, 0.0029524
5: 0.0125908, 0.0147907, 0.0128146, 0.0148021, -0.0022113, 0.0019760
6: 0.0026188, 0.0052112, 0.0029653, 0.0051846, -0.0025658, 0.0022459
7: -0.0172331, -0.0127154, -0.0170486, -0.0131811, -0.0040520, 0.0043333
8: 0.0030572, 0.0067802, 0.0032036, 0.0063907, -0.0033335, 0.0035766
9: 0.0021873, 0.0081550, 0.0025997, 0.0082102, -0.0060228, 0.0055553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040839, -0.0025848, -0.0014991, 0.0014991
1: -0.0055989, -0.0031468, -0.0055989, -0.0031468, -0.0024521, 0.0024521
2: 0.9648491, 0.9715527, 0.9648491, 0.9715527, -0.0067036, 0.0067036
3: 0.0231467, 0.0364830, 0.0231467, 0.0364830, -0.0089895, 0.0089895
4: -0.0034678, -0.0004612, -0.0034678, -0.0004612, -0.0030065, 0.0030065
5: 0.0125908, 0.0147907, 0.0125908, 0.0147907, -0.0021998, 0.0021998
6: 0.0026188, 0.0052112, 0.0026188, 0.0052112, -0.0025924, 0.0025924
7: -0.0172331, -0.0127154, -0.0172331, -0.0127154, -0.0045177, 0.0045177
8: 0.0030572, 0.0067802, 0.0030572, 0.0067802, -0.0037230, 0.0037230
9: 0.0021873, 0.0081550, 0.0021873, 0.0081550, -0.0059677, 0.0059677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0067051
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0067099
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0062922
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066939, upper bound: 0.0062922
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

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
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0067141
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066940, upper bound: 0.0067141
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028050, -0.0040864, -0.0025727, -0.0015096, 0.0012813
1: -0.0055390, -0.0033715, -0.0056909, -0.0031344, -0.0024046, 0.0023194
2: 0.9657504, 0.9714524, 0.9647995, 0.9715582, -0.0058078, 0.0066530
3: 0.0236766, 0.0357439, 0.0223324, 0.0365237, -0.0096062, 0.0101521
4: -0.0034116, -0.0008542, -0.0034709, -0.0004396, -0.0029720, 0.0026166
5: 0.0128232, 0.0147499, 0.0125780, 0.0148533, -0.0020301, 0.0021719
6: 0.0029785, 0.0051836, 0.0025990, 0.0052127, -0.0022342, 0.0025846
7: -0.0170416, -0.0131989, -0.0172437, -0.0126887, -0.0043529, 0.0040448
8: 0.0032091, 0.0063759, 0.0030488, 0.0068025, -0.0035933, 0.0033271
9: 0.0026154, 0.0079591, 0.0021638, 0.0084561, -0.0058407, 0.0057953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0062919
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0062919
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0067099
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0067099
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028212, -0.0040844, -0.0027180, -0.0013663, 0.0012632
1: -0.0056157, -0.0033881, -0.0056161, -0.0032827, -0.0023330, 0.0022280
2: 0.9658166, 0.9714451, 0.9653942, 0.9714921, -0.0056754, 0.0060509
3: 0.0229978, 0.0356897, 0.0229948, 0.0360360, -0.0100069, 0.0096702
4: -0.0034074, -0.0008831, -0.0034338, -0.0006989, -0.0027085, 0.0025507
5: 0.0128402, 0.0148021, 0.0127314, 0.0148024, -0.0019621, 0.0020708
6: 0.0030049, 0.0051816, 0.0028364, 0.0051945, -0.0021896, 0.0023452
7: -0.0170275, -0.0132344, -0.0171173, -0.0130078, -0.0040197, 0.0038829
8: 0.0032203, 0.0063462, 0.0031491, 0.0065357, -0.0033154, 0.0031971
9: 0.0026468, 0.0082101, 0.0024462, 0.0082112, -0.0055643, 0.0057638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0059839, upper bound: 0.0057816
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055304, upper bound: 0.0056970
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0027970, -0.0040844, -0.0026219, -0.0014625, 0.0012875
1: -0.0056158, -0.0033633, -0.0056175, -0.0031846, -0.0024312, 0.0022542
2: 0.9657173, 0.9714562, 0.9650006, 0.9715359, -0.0058185, 0.0064555
3: 0.0229975, 0.0357711, 0.0229821, 0.0363587, -0.0100429, 0.0096988
4: -0.0034136, -0.0008398, -0.0034583, -0.0005273, -0.0028863, 0.0026185
5: 0.0128146, 0.0148021, 0.0126299, 0.0148033, -0.0019887, 0.0021723
6: 0.0029653, 0.0051846, 0.0026793, 0.0052066, -0.0022413, 0.0025053
7: -0.0170486, -0.0131811, -0.0172009, -0.0127967, -0.0042520, 0.0040198
8: 0.0032036, 0.0063907, 0.0030827, 0.0067122, -0.0035086, 0.0033080
9: 0.0025997, 0.0082102, 0.0022593, 0.0082159, -0.0056162, 0.0059509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066793
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040850, -0.0027732, -0.0013108, 0.0015001
1: -0.0055989, -0.0031468, -0.0056384, -0.0033390, -0.0022599, 0.0024915
2: 0.9648491, 0.9715527, 0.9656199, 0.9714670, -0.0066180, 0.0059328
3: 0.0231467, 0.0364830, 0.0227976, 0.0358510, -0.0098457, 0.0107636
4: -0.0034678, -0.0004612, -0.0034197, -0.0007973, -0.0026704, 0.0029585
5: 0.0125908, 0.0147907, 0.0127895, 0.0148175, -0.0022267, 0.0020012
6: 0.0026188, 0.0052112, 0.0029264, 0.0051876, -0.0025688, 0.0022848
7: -0.0172331, -0.0127154, -0.0170693, -0.0131288, -0.0041043, 0.0043540
8: 0.0030572, 0.0067802, 0.0031871, 0.0064344, -0.0033772, 0.0035931
9: 0.0021873, 0.0081550, 0.0025534, 0.0082841, -0.0060968, 0.0056016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064624, upper bound: 0.0062910
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066949, upper bound: 0.0062922
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064624, upper bound: 0.0067072
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066949, upper bound: 0.0067141
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065983
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0025848, -0.0040869, -0.0027780, -0.0013059, 0.0015021
1: -0.0055989, -0.0031468, -0.0057107, -0.0033440, -0.0022549, 0.0025639
2: 0.9648491, 0.9715527, 0.9656399, 0.9714647, -0.0066156, 0.0059128
3: 0.0231467, 0.0364830, 0.0221572, 0.0358347, -0.0092503, 0.0108609
4: -0.0034678, -0.0004612, -0.0034185, -0.0008060, -0.0026618, 0.0029572
5: 0.0125908, 0.0147907, 0.0127946, 0.0148667, -0.0022759, 0.0019960
6: 0.0026188, 0.0052112, 0.0029344, 0.0051870, -0.0025682, 0.0022769
7: -0.0172331, -0.0127154, -0.0170651, -0.0131395, -0.0040936, 0.0043497
8: 0.0030572, 0.0067802, 0.0031905, 0.0064255, -0.0033683, 0.0035897
9: 0.0021873, 0.0081550, 0.0025629, 0.0085209, -0.0063336, 0.0055921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062919
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0067099
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027957, -0.0040818, -0.0027495, -0.0013355, 0.0012861
1: -0.0056383, -0.0033620, -0.0055181, -0.0033149, -0.0023235, 0.0021561
2: 0.9657123, 0.9714566, 0.9655231, 0.9714777, -0.0057654, 0.0059335
3: 0.0227978, 0.0357753, 0.0238616, 0.0359304, -0.0098741, 0.0088150
4: -0.0034139, -0.0008376, -0.0034257, -0.0007551, -0.0026589, 0.0025882
5: 0.0128133, 0.0148175, 0.0127646, 0.0147357, -0.0019224, 0.0020529
6: 0.0029633, 0.0051848, 0.0028878, 0.0051906, -0.0022273, 0.0022970
7: -0.0170497, -0.0131783, -0.0170899, -0.0130769, -0.0039728, 0.0039116
8: 0.0032027, 0.0063930, 0.0031708, 0.0064779, -0.0032752, 0.0032222
9: 0.0025972, 0.0082840, 0.0025074, 0.0078907, -0.0052934, 0.0057766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065225
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065243
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040819, -0.0026449, -0.0014400, 0.0013087
1: -0.0056384, -0.0033390, -0.0055220, -0.0032081, -0.0024302, 0.0021830
2: 0.9656199, 0.9714670, 0.9650951, 0.9715253, -0.0059054, 0.0063719
3: 0.0227976, 0.0358510, 0.0238274, 0.0362813, -0.0101631, 0.0089471
4: -0.0034197, -0.0007973, -0.0034524, -0.0005685, -0.0028512, 0.0026551
5: 0.0127895, 0.0148175, 0.0126542, 0.0147384, -0.0019488, 0.0021633
6: 0.0029264, 0.0051876, 0.0027170, 0.0052037, -0.0022772, 0.0024706
7: -0.0170693, -0.0131288, -0.0171809, -0.0128473, -0.0042220, 0.0040520
8: 0.0031871, 0.0064344, 0.0030987, 0.0066699, -0.0034827, 0.0033358
9: 0.0025534, 0.0082841, 0.0023042, 0.0079033, -0.0053499, 0.0059800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066940
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0025704, -0.0040823, -0.0028050, -0.0012794, 0.0015119
1: -0.0056175, -0.0031321, -0.0055390, -0.0033715, -0.0022460, 0.0024069
2: 0.9647900, 0.9715592, 0.9657504, 0.9714524, -0.0066624, 0.0058088
3: 0.0229819, 0.0365314, 0.0236766, 0.0357439, -0.0094362, 0.0095654
4: -0.0034715, -0.0004355, -0.0034116, -0.0008542, -0.0026172, 0.0029761
5: 0.0125756, 0.0148033, 0.0128232, 0.0147499, -0.0021743, 0.0019802
6: 0.0025953, 0.0052130, 0.0029785, 0.0051836, -0.0025883, 0.0022345
7: -0.0172457, -0.0126837, -0.0170416, -0.0131989, -0.0040468, 0.0043579
8: 0.0030472, 0.0068067, 0.0032091, 0.0063759, -0.0033287, 0.0035975
9: 0.0021593, 0.0082159, 0.0026154, 0.0079591, -0.0057997, 0.0056005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0067141
time: 1.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040839, -0.0025848, -0.0015001, 0.0013108
1: -0.0056384, -0.0033390, -0.0055989, -0.0031468, -0.0024915, 0.0022599
2: 0.9656199, 0.9714670, 0.9648491, 0.9715527, -0.0059328, 0.0066180
3: 0.0227976, 0.0358510, 0.0231467, 0.0364830, -0.0107636, 0.0098457
4: -0.0034197, -0.0007973, -0.0034678, -0.0004612, -0.0029585, 0.0026704
5: 0.0127895, 0.0148175, 0.0125908, 0.0147907, -0.0020012, 0.0022267
6: 0.0029264, 0.0051876, 0.0026188, 0.0052112, -0.0022848, 0.0025688
7: -0.0170693, -0.0131288, -0.0172331, -0.0127154, -0.0043540, 0.0041043
8: 0.0031871, 0.0064344, 0.0030572, 0.0067802, -0.0035931, 0.0033772
9: 0.0025534, 0.0082841, 0.0021873, 0.0081550, -0.0056016, 0.0060968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

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
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0028018, -0.0040818, -0.0027495, -0.0013374, 0.0012800
1: -0.0057107, -0.0033682, -0.0055181, -0.0033149, -0.0023958, 0.0021499
2: 0.9657370, 0.9714540, 0.9655231, 0.9714777, -0.0057408, 0.0059308
3: 0.0221575, 0.0357550, 0.0238616, 0.0359304, -0.0105736, 0.0088349
4: -0.0034124, -0.0008484, -0.0034257, -0.0007551, -0.0026573, 0.0025774
5: 0.0128197, 0.0148667, 0.0127646, 0.0147357, -0.0019160, 0.0021022
6: 0.0029732, 0.0051840, 0.0028878, 0.0051906, -0.0022174, 0.0022962
7: -0.0170445, -0.0131916, -0.0170899, -0.0130769, -0.0039676, 0.0038983
8: 0.0032069, 0.0063819, 0.0031708, 0.0064779, -0.0032710, 0.0032111
9: 0.0026090, 0.0085208, 0.0025074, 0.0078907, -0.0052816, 0.0060134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0059736, upper bound: 0.0057835
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055301, upper bound: 0.0057100
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040819, -0.0026449, -0.0014420, 0.0013039
1: -0.0057107, -0.0033440, -0.0055220, -0.0032081, -0.0025026, 0.0021780
2: 0.9656399, 0.9714647, 0.9650951, 0.9715253, -0.0058854, 0.0063696
3: 0.0221572, 0.0358347, 0.0238274, 0.0362813, -0.0107638, 0.0089738
4: -0.0034185, -0.0008060, -0.0034524, -0.0005685, -0.0028500, 0.0026464
5: 0.0127946, 0.0148667, 0.0126542, 0.0147384, -0.0019437, 0.0022125
6: 0.0029344, 0.0051870, 0.0027170, 0.0052037, -0.0022693, 0.0024700
7: -0.0170651, -0.0131395, -0.0171809, -0.0128473, -0.0042178, 0.0040414
8: 0.0031905, 0.0064255, 0.0030987, 0.0066699, -0.0034794, 0.0033269
9: 0.0025629, 0.0085209, 0.0023042, 0.0079033, -0.0053404, 0.0062168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065980
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066822
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040823, -0.0028050, -0.0012813, 0.0015096
1: -0.0056909, -0.0031344, -0.0055390, -0.0033715, -0.0023194, 0.0024046
2: 0.9647995, 0.9715582, 0.9657504, 0.9714524, -0.0066530, 0.0058078
3: 0.0223324, 0.0365237, 0.0236766, 0.0357439, -0.0101521, 0.0096062
4: -0.0034709, -0.0004396, -0.0034116, -0.0008542, -0.0026166, 0.0029720
5: 0.0125780, 0.0148533, 0.0128232, 0.0147499, -0.0021719, 0.0020301
6: 0.0025990, 0.0052127, 0.0029785, 0.0051836, -0.0025846, 0.0022342
7: -0.0172437, -0.0126887, -0.0170416, -0.0131989, -0.0040448, 0.0043529
8: 0.0030488, 0.0068025, 0.0032091, 0.0063759, -0.0033271, 0.0035933
9: 0.0021638, 0.0084561, 0.0026154, 0.0079591, -0.0057953, 0.0058407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040839, -0.0025848, -0.0015021, 0.0013059
1: -0.0057107, -0.0033440, -0.0055989, -0.0031468, -0.0025639, 0.0022549
2: 0.9656399, 0.9714647, 0.9648491, 0.9715527, -0.0059128, 0.0066156
3: 0.0221572, 0.0358347, 0.0231467, 0.0364830, -0.0108609, 0.0092503
4: -0.0034185, -0.0008060, -0.0034678, -0.0004612, -0.0029572, 0.0026618
5: 0.0127946, 0.0148667, 0.0125908, 0.0147907, -0.0019960, 0.0022759
6: 0.0029344, 0.0051870, 0.0026188, 0.0052112, -0.0022769, 0.0025682
7: -0.0170651, -0.0131395, -0.0172331, -0.0127154, -0.0043497, 0.0040936
8: 0.0031905, 0.0064255, 0.0030572, 0.0067802, -0.0035897, 0.0033683
9: 0.0025629, 0.0085209, 0.0021873, 0.0081550, -0.0055921, 0.0063336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065997
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0067099
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065225
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065243
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066940
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

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
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

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
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040850, -0.0027732, -0.0040864, -0.0025727, -0.0015123, 0.0013132
1: -0.0056384, -0.0033390, -0.0056909, -0.0031344, -0.0025039, 0.0023519
2: 0.9656199, 0.9714670, 0.9647995, 0.9715582, -0.0059383, 0.0066676
3: 0.0227976, 0.0358510, 0.0223324, 0.0365237, -0.0098792, 0.0097121
4: -0.0034197, -0.0007973, -0.0034709, -0.0004396, -0.0029801, 0.0026735
5: 0.0127895, 0.0148175, 0.0125780, 0.0148533, -0.0020638, 0.0022395
6: 0.0029264, 0.0051876, 0.0025990, 0.0052127, -0.0022863, 0.0025886
7: -0.0170693, -0.0131288, -0.0172437, -0.0126887, -0.0043806, 0.0041148
8: 0.0031871, 0.0064344, 0.0030488, 0.0068025, -0.0036153, 0.0033856
9: 0.0025534, 0.0082841, 0.0021638, 0.0084561, -0.0059027, 0.0061203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

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
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0062919
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0067099
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0028018, -0.0040844, -0.0027180, -0.0013689, 0.0012826
1: -0.0057107, -0.0033682, -0.0056161, -0.0032827, -0.0024279, 0.0022479
2: 0.9657370, 0.9714540, 0.9653942, 0.9714921, -0.0057551, 0.0060598
3: 0.0221575, 0.0357550, 0.0229948, 0.0360360, -0.0099481, 0.0089092
4: -0.0034124, -0.0008484, -0.0034338, -0.0006989, -0.0027135, 0.0025854
5: 0.0128197, 0.0148667, 0.0127314, 0.0148024, -0.0019826, 0.0021354
6: 0.0029732, 0.0051840, 0.0028364, 0.0051945, -0.0022213, 0.0023476
7: -0.0170445, -0.0131916, -0.0171173, -0.0130078, -0.0040367, 0.0039257
8: 0.0032069, 0.0063819, 0.0031491, 0.0065357, -0.0033288, 0.0032328
9: 0.0026090, 0.0085208, 0.0024462, 0.0082112, -0.0056021, 0.0060746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0059736, upper bound: 0.0057835
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0055301, upper bound: 0.0057100
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040869, -0.0027780, -0.0040844, -0.0026219, -0.0014650, 0.0013064
1: -0.0057107, -0.0033440, -0.0056175, -0.0031846, -0.0025261, 0.0022735
2: 0.9656399, 0.9714647, 0.9650006, 0.9715359, -0.0058960, 0.0064641
3: 0.0221572, 0.0358347, 0.0229821, 0.0363587, -0.0099683, 0.0090441
4: -0.0034185, -0.0008060, -0.0034583, -0.0005273, -0.0028911, 0.0026523
5: 0.0127946, 0.0148667, 0.0126299, 0.0148033, -0.0020087, 0.0022368
6: 0.0029344, 0.0051870, 0.0026793, 0.0052066, -0.0022722, 0.0025077
7: -0.0170651, -0.0131395, -0.0172009, -0.0127967, -0.0042684, 0.0040614
8: 0.0031905, 0.0064255, 0.0030827, 0.0067122, -0.0035217, 0.0033428
9: 0.0025629, 0.0085209, 0.0022593, 0.0082159, -0.0056530, 0.0062616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065980
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066822
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040850, -0.0027732, -0.0013132, 0.0015123
1: -0.0056909, -0.0031344, -0.0056384, -0.0033390, -0.0023519, 0.0025039
2: 0.9647995, 0.9715582, 0.9656199, 0.9714670, -0.0066676, 0.0059383
3: 0.0223324, 0.0365237, 0.0227976, 0.0358510, -0.0097121, 0.0098792
4: -0.0034709, -0.0004396, -0.0034197, -0.0007973, -0.0026735, 0.0029801
5: 0.0125780, 0.0148533, 0.0127895, 0.0148175, -0.0022395, 0.0020638
6: 0.0025990, 0.0052127, 0.0029264, 0.0051876, -0.0025886, 0.0022863
7: -0.0172437, -0.0126887, -0.0170693, -0.0131288, -0.0041148, 0.0043806
8: 0.0030488, 0.0068025, 0.0031871, 0.0064344, -0.0033856, 0.0036153
9: 0.0021638, 0.0084561, 0.0025534, 0.0082841, -0.0061203, 0.0059027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
time: 1.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040844, -0.0025704, -0.0015160, 0.0015117
1: -0.0056909, -0.0031344, -0.0056175, -0.0031321, -0.0025588, 0.0024831
2: 0.9647995, 0.9715582, 0.9647900, 0.9715592, -0.0067598, 0.0067681
3: 0.0223324, 0.0365237, 0.0229819, 0.0365314, -0.0100202, 0.0093404
4: -0.0034709, -0.0004396, -0.0034715, -0.0004355, -0.0030354, 0.0030319
5: 0.0125780, 0.0148533, 0.0125756, 0.0148033, -0.0022253, 0.0022776
6: 0.0025990, 0.0052127, 0.0025953, 0.0052130, -0.0026140, 0.0026175
7: -0.0172437, -0.0126887, -0.0172457, -0.0126837, -0.0045599, 0.0045569
8: 0.0030488, 0.0068025, 0.0030472, 0.0068067, -0.0037578, 0.0037552
9: 0.0021638, 0.0084561, 0.0021593, 0.0082159, -0.0060522, 0.0062968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0067118
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0067141
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065997
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040864, -0.0025727, -0.0040869, -0.0027780, -0.0013084, 0.0015142
1: -0.0056909, -0.0031344, -0.0057107, -0.0033440, -0.0023469, 0.0025763
2: 0.9647995, 0.9715582, 0.9656399, 0.9714647, -0.0066652, 0.0059183
3: 0.0223324, 0.0365237, 0.0221572, 0.0358347, -0.0091109, 0.0099722
4: -0.0034709, -0.0004396, -0.0034185, -0.0008060, -0.0026649, 0.0029789
5: 0.0125780, 0.0148533, 0.0127946, 0.0148667, -0.0022887, 0.0020586
6: 0.0025990, 0.0052127, 0.0029344, 0.0051870, -0.0025880, 0.0022784
7: -0.0172437, -0.0126887, -0.0170651, -0.0131395, -0.0041042, 0.0043764
8: 0.0030488, 0.0068025, 0.0031905, 0.0064255, -0.0033767, 0.0036120
9: 0.0021638, 0.0084561, 0.0025629, 0.0085209, -0.0063571, 0.0058933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
time: 1.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062919
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0067078
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0067099
time: 1.38 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.23 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0059736, upper bound: 0.0057816
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0055301, upper bound: 0.0056970
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066793
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062910
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0067072
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0067141
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065983
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0067051
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0067099
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066935
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0062922
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066939, upper bound: 0.0062922
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065243, upper bound: 0.0067141
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066940, upper bound: 0.0067141
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066503
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066947
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0062919
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0062919
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064972, upper bound: 0.0067099
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066822, upper bound: 0.0067099
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0059839, upper bound: 0.0057816
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0055304, upper bound: 0.0056970
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066793
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064624, upper bound: 0.0062910
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066949, upper bound: 0.0062922
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064624, upper bound: 0.0067072
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066949, upper bound: 0.0067141
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065983
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062919
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0067099
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065225
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065243
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066940
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0067141
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0059736, upper bound: 0.0057835
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0055301, upper bound: 0.0057100
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065980
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066822
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065997
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0067099
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065225
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065243
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066940
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0062919
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0067099
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0059736, upper bound: 0.0057835
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0055301, upper bound: 0.0057100
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065980
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066822
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0067118
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0067141
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065997
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062919
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0067078
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0067099

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040818, -0.0027495, -0.0013327, 0.0010896
1: -0.0055329, -0.0035626, -0.0055181, -0.0033149, -0.0022180, 0.0019556
2: 0.9665166, 0.9713672, 0.9655231, 0.9714777, -0.0049612, 0.0058441
3: 0.0237312, 0.0351158, 0.0238616, 0.0359304, -0.0084427, 0.0076681
4: -0.0033638, -0.0011883, -0.0034257, -0.0007551, -0.0026087, 0.0022375
5: 0.0130206, 0.0147457, 0.0127646, 0.0147357, -0.0017151, 0.0019812
6: 0.0032842, 0.0051601, 0.0028878, 0.0051906, -0.0019063, 0.0022723
7: -0.0168788, -0.0136098, -0.0170899, -0.0130769, -0.0038019, 0.0034801
8: 0.0033383, 0.0060323, 0.0031708, 0.0064779, -0.0031396, 0.0028614
9: 0.0029792, 0.0079389, 0.0025074, 0.0078907, -0.0049114, 0.0054314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040818, -0.0027495, -0.0013328, 0.0011954
1: -0.0055389, -0.0034545, -0.0055181, -0.0033149, -0.0022240, 0.0020636
2: 0.9660832, 0.9714154, 0.9655231, 0.9714777, -0.0053945, 0.0058923
3: 0.0236782, 0.0354711, 0.0238616, 0.0359304, -0.0085574, 0.0080930
4: -0.0033908, -0.0009993, -0.0034257, -0.0007551, -0.0026357, 0.0024264
5: 0.0129089, 0.0147498, 0.0127646, 0.0147357, -0.0018268, 0.0019853
6: 0.0031113, 0.0051734, 0.0028878, 0.0051906, -0.0020792, 0.0022856
7: -0.0169709, -0.0133773, -0.0170899, -0.0130769, -0.0038940, 0.0037126
8: 0.0032652, 0.0062266, 0.0031708, 0.0064779, -0.0032126, 0.0030558
9: 0.0027734, 0.0079585, 0.0025074, 0.0078907, -0.0051172, 0.0054510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040819, -0.0026449, -0.0014372, 0.0010897
1: -0.0055329, -0.0035626, -0.0055220, -0.0032081, -0.0023247, 0.0019594
2: 0.9665166, 0.9713672, 0.9650951, 0.9715253, -0.0050088, 0.0062721
3: 0.0237312, 0.0351158, 0.0238274, 0.0362813, -0.0088358, 0.0077294
4: -0.0033638, -0.0011883, -0.0034524, -0.0005685, -0.0027953, 0.0022642
5: 0.0130206, 0.0147457, 0.0126542, 0.0147384, -0.0017177, 0.0020915
6: 0.0032842, 0.0051601, 0.0027170, 0.0052037, -0.0019194, 0.0024431
7: -0.0168788, -0.0136098, -0.0171809, -0.0128473, -0.0040315, 0.0035711
8: 0.0033383, 0.0060323, 0.0030987, 0.0066699, -0.0033316, 0.0029336
9: 0.0029792, 0.0079389, 0.0023042, 0.0079033, -0.0049241, 0.0056347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040819, -0.0026449, -0.0014374, 0.0011955
1: -0.0055389, -0.0034545, -0.0055220, -0.0032081, -0.0023307, 0.0020675
2: 0.9660832, 0.9714154, 0.9650951, 0.9715253, -0.0054421, 0.0063203
3: 0.0236782, 0.0354711, 0.0238274, 0.0362813, -0.0086526, 0.0078724
4: -0.0033908, -0.0009993, -0.0034524, -0.0005685, -0.0028223, 0.0024531
5: 0.0129089, 0.0147498, 0.0126542, 0.0147384, -0.0018294, 0.0020956
6: 0.0031113, 0.0051734, 0.0027170, 0.0052037, -0.0020924, 0.0024564
7: -0.0169709, -0.0133773, -0.0171809, -0.0128473, -0.0041236, 0.0038035
8: 0.0032652, 0.0062266, 0.0030987, 0.0066699, -0.0034046, 0.0031280
9: 0.0027734, 0.0079585, 0.0023042, 0.0079033, -0.0051298, 0.0056543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066899
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040823, -0.0028285, -0.0012533, 0.0013048
1: -0.0055180, -0.0033434, -0.0055390, -0.0033955, -0.0021225, 0.0021956
2: 0.9656377, 0.9714649, 0.9658464, 0.9714418, -0.0058041, 0.0056186
3: 0.0238631, 0.0358364, 0.0236768, 0.0356653, -0.0082030, 0.0086187
4: -0.0034186, -0.0008051, -0.0034056, -0.0008960, -0.0025226, 0.0026005
5: 0.0127941, 0.0147356, 0.0128479, 0.0147499, -0.0019558, 0.0018877
6: 0.0029335, 0.0051870, 0.0030168, 0.0051807, -0.0022471, 0.0021703
7: -0.0170656, -0.0131384, -0.0170212, -0.0132503, -0.0038153, 0.0038828
8: 0.0031901, 0.0064265, 0.0032253, 0.0063329, -0.0031427, 0.0032012
9: 0.0025619, 0.0078901, 0.0026610, 0.0079590, -0.0053971, 0.0052291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065223, upper bound: 0.0061319
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065223, upper bound: 0.0062922
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040823, -0.0028050, -0.0012768, 0.0014071
1: -0.0055218, -0.0032390, -0.0055390, -0.0033715, -0.0021503, 0.0023000
2: 0.9652189, 0.9715115, 0.9657504, 0.9714524, -0.0062335, 0.0057611
3: 0.0238290, 0.0361797, 0.0236766, 0.0357439, -0.0083512, 0.0087107
4: -0.0034447, -0.0006225, -0.0034116, -0.0008542, -0.0025905, 0.0027891
5: 0.0126862, 0.0147382, 0.0128232, 0.0147499, -0.0020638, 0.0019151
6: 0.0027664, 0.0051999, 0.0029785, 0.0051836, -0.0024172, 0.0022214
7: -0.0171545, -0.0129138, -0.0170416, -0.0131989, -0.0039557, 0.0041278
8: 0.0031195, 0.0066143, 0.0032091, 0.0063759, -0.0032564, 0.0034051
9: 0.0023630, 0.0079027, 0.0026154, 0.0079591, -0.0055960, 0.0052873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0061319
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0062922
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040819, -0.0026157, -0.0014661, 0.0013044
1: -0.0055180, -0.0033434, -0.0055220, -0.0031783, -0.0023397, 0.0021786
2: 0.9656377, 0.9714649, 0.9649752, 0.9715386, -0.0059009, 0.0064897
3: 0.0238631, 0.0358364, 0.0238272, 0.0363795, -0.0084536, 0.0080264
4: -0.0034186, -0.0008051, -0.0034599, -0.0005163, -0.0029023, 0.0026548
5: 0.0127941, 0.0147356, 0.0126234, 0.0147384, -0.0019443, 0.0021122
6: 0.0029335, 0.0051870, 0.0026692, 0.0052074, -0.0022738, 0.0025179
7: -0.0170656, -0.0131384, -0.0172063, -0.0127831, -0.0042825, 0.0040679
8: 0.0031901, 0.0064265, 0.0030785, 0.0067236, -0.0035335, 0.0033480
9: 0.0025619, 0.0078901, 0.0022473, 0.0079034, -0.0053415, 0.0056428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0065425
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040819, -0.0025921, -0.0014898, 0.0014067
1: -0.0055218, -0.0032390, -0.0055221, -0.0031542, -0.0023676, 0.0022830
2: 0.9652189, 0.9715115, 0.9648787, 0.9715494, -0.0063305, 0.0066328
3: 0.0238290, 0.0361797, 0.0238268, 0.0364587, -0.0086449, 0.0081172
4: -0.0034447, -0.0006225, -0.0034659, -0.0004741, -0.0029706, 0.0028434
5: 0.0126862, 0.0147382, 0.0125985, 0.0147384, -0.0020522, 0.0021398
6: 0.0027664, 0.0051999, 0.0026306, 0.0052103, -0.0024439, 0.0025692
7: -0.0171545, -0.0129138, -0.0172268, -0.0127313, -0.0044233, 0.0043131
8: 0.0031195, 0.0066143, 0.0030622, 0.0067669, -0.0036474, 0.0035521
9: 0.0023630, 0.0079027, 0.0022014, 0.0079035, -0.0055405, 0.0057013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0065425
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0067141
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0061268
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0058907
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040839, -0.0025848, -0.0014975, 0.0011976
1: -0.0055389, -0.0034545, -0.0055989, -0.0031468, -0.0023921, 0.0021444
2: 0.9660832, 0.9714154, 0.9648491, 0.9715527, -0.0054694, 0.0065663
3: 0.0236782, 0.0354711, 0.0231467, 0.0364830, -0.0094458, 0.0086831
4: -0.0033908, -0.0009993, -0.0034678, -0.0004612, -0.0029296, 0.0024685
5: 0.0129089, 0.0147498, 0.0125908, 0.0147907, -0.0018818, 0.0021590
6: 0.0031113, 0.0051734, 0.0026188, 0.0052112, -0.0020999, 0.0025546
7: -0.0169709, -0.0133773, -0.0172331, -0.0127154, -0.0042555, 0.0038558
8: 0.0032652, 0.0062266, 0.0030572, 0.0067802, -0.0035150, 0.0031694
9: 0.0027734, 0.0079585, 0.0021873, 0.0081550, -0.0053816, 0.0057711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0064619
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0066947
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057816, upper bound: 0.0059736
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0056970, upper bound: 0.0055301
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0060624
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0062919
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0064176
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0061017
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0064738
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0067099
time: 1.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040842, -0.0029917, -0.0040819, -0.0026449, -0.0014392, 0.0010902
1: -0.0056074, -0.0035621, -0.0055220, -0.0032081, -0.0023993, 0.0019599
2: 0.9665148, 0.9713675, 0.9650951, 0.9715253, -0.0050105, 0.0062724
3: 0.0230714, 0.0351174, 0.0238274, 0.0362813, -0.0097031, 0.0079208
4: -0.0033639, -0.0011874, -0.0034524, -0.0005685, -0.0027954, 0.0022650
5: 0.0130201, 0.0147965, 0.0126542, 0.0147384, -0.0017182, 0.0021422
6: 0.0032835, 0.0051602, 0.0027170, 0.0052037, -0.0019202, 0.0024432
7: -0.0168792, -0.0136087, -0.0171809, -0.0128473, -0.0040319, 0.0035721
8: 0.0033380, 0.0060331, 0.0030987, 0.0066699, -0.0033319, 0.0029345
9: 0.0029783, 0.0081829, 0.0023042, 0.0079033, -0.0049250, 0.0058787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028773, -0.0040819, -0.0026449, -0.0014394, 0.0012046
1: -0.0056156, -0.0034453, -0.0055220, -0.0032081, -0.0024074, 0.0020767
2: 0.9660460, 0.9714196, 0.9650951, 0.9715253, -0.0054793, 0.0063245
3: 0.0229991, 0.0355015, 0.0238274, 0.0362813, -0.0095334, 0.0081000
4: -0.0033931, -0.0009831, -0.0034524, -0.0005685, -0.0028247, 0.0024693
5: 0.0128994, 0.0148020, 0.0126542, 0.0147384, -0.0018390, 0.0021478
6: 0.0030965, 0.0051745, 0.0027170, 0.0052037, -0.0021072, 0.0024575
7: -0.0169788, -0.0133574, -0.0171809, -0.0128473, -0.0041315, 0.0038234
8: 0.0032590, 0.0062433, 0.0030987, 0.0066699, -0.0034109, 0.0031446
9: 0.0027558, 0.0082096, 0.0023042, 0.0079033, -0.0051475, 0.0059054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066786
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027803, -0.0040823, -0.0028285, -0.0012553, 0.0013020
1: -0.0055914, -0.0033463, -0.0055390, -0.0033955, -0.0021960, 0.0021927
2: 0.9656492, 0.9714637, 0.9658464, 0.9714418, -0.0057926, 0.0056174
3: 0.0232130, 0.0358269, 0.0236768, 0.0356653, -0.0089945, 0.0087536
4: -0.0034179, -0.0008101, -0.0034056, -0.0008960, -0.0025218, 0.0025955
5: 0.0127971, 0.0147856, 0.0128479, 0.0147499, -0.0019529, 0.0019377
6: 0.0029381, 0.0051867, 0.0030168, 0.0051807, -0.0022425, 0.0021699
7: -0.0170631, -0.0131446, -0.0170212, -0.0132503, -0.0038128, 0.0038767
8: 0.0031921, 0.0064213, 0.0032253, 0.0063329, -0.0031408, 0.0031960
9: 0.0025674, 0.0081305, 0.0026610, 0.0079590, -0.0053916, 0.0054695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064606, upper bound: 0.0061319
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064606, upper bound: 0.0062910
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026646, -0.0040823, -0.0028050, -0.0012789, 0.0014177
1: -0.0055987, -0.0032282, -0.0055390, -0.0033715, -0.0022271, 0.0023109
2: 0.9651754, 0.9715163, 0.9657504, 0.9714524, -0.0062770, 0.0057659
3: 0.0231488, 0.0362154, 0.0236766, 0.0357439, -0.0092469, 0.0088806
4: -0.0034474, -0.0006035, -0.0034116, -0.0008542, -0.0025932, 0.0028080
5: 0.0126750, 0.0147905, 0.0128232, 0.0147499, -0.0020750, 0.0019674
6: 0.0027491, 0.0052012, 0.0029785, 0.0051836, -0.0024345, 0.0022227
7: -0.0171638, -0.0128904, -0.0170416, -0.0131989, -0.0039649, 0.0041512
8: 0.0031122, 0.0066338, 0.0032091, 0.0063759, -0.0032637, 0.0034247
9: 0.0023423, 0.0081542, 0.0026154, 0.0079591, -0.0056167, 0.0055388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066503, upper bound: 0.0061319
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066503, upper bound: 0.0061319
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027803, -0.0040819, -0.0026157, -0.0014680, 0.0013016
1: -0.0055914, -0.0033463, -0.0055220, -0.0031783, -0.0024131, 0.0021757
2: 0.9656492, 0.9714637, 0.9649752, 0.9715386, -0.0058894, 0.0064885
3: 0.0232130, 0.0358269, 0.0238272, 0.0363795, -0.0093166, 0.0082180
4: -0.0034179, -0.0008101, -0.0034599, -0.0005163, -0.0029016, 0.0026498
5: 0.0127971, 0.0147856, 0.0126234, 0.0147384, -0.0019413, 0.0021622
6: 0.0029381, 0.0051867, 0.0026692, 0.0052074, -0.0022692, 0.0025175
7: -0.0170631, -0.0131446, -0.0172063, -0.0127831, -0.0042800, 0.0040618
8: 0.0031921, 0.0064213, 0.0030785, 0.0067236, -0.0035315, 0.0033428
9: 0.0025674, 0.0081305, 0.0022473, 0.0079034, -0.0053360, 0.0058832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0065425
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067072
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026646, -0.0040819, -0.0025921, -0.0014918, 0.0014173
1: -0.0055987, -0.0032282, -0.0055221, -0.0031542, -0.0024445, 0.0022939
2: 0.9651754, 0.9715163, 0.9648787, 0.9715494, -0.0063740, 0.0066376
3: 0.0231488, 0.0362154, 0.0238268, 0.0364587, -0.0095805, 0.0083448
4: -0.0034474, -0.0006035, -0.0034659, -0.0004741, -0.0029733, 0.0028624
5: 0.0126750, 0.0147905, 0.0125985, 0.0147384, -0.0020634, 0.0021920
6: 0.0027491, 0.0052012, 0.0026306, 0.0052103, -0.0024613, 0.0025706
7: -0.0171638, -0.0128904, -0.0172268, -0.0127313, -0.0044325, 0.0043364
8: 0.0031122, 0.0066338, 0.0030622, 0.0067669, -0.0036547, 0.0035716
9: 0.0023423, 0.0081542, 0.0022014, 0.0079035, -0.0055612, 0.0059528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0065425
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067141
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040842, -0.0029917, -0.0040839, -0.0026095, -0.0014747, 0.0010922
1: -0.0056074, -0.0035621, -0.0055989, -0.0031720, -0.0024355, 0.0020368
2: 0.9665148, 0.9713675, 0.9649500, 0.9715414, -0.0050266, 0.0064175
3: 0.0230714, 0.0351174, 0.0231471, 0.0364003, -0.0093470, 0.0080318
4: -0.0033639, -0.0011874, -0.0034615, -0.0005052, -0.0028587, 0.0022741
5: 0.0130201, 0.0147965, 0.0126168, 0.0147906, -0.0017705, 0.0021796
6: 0.0032835, 0.0051602, 0.0026591, 0.0052081, -0.0019247, 0.0025011
7: -0.0168792, -0.0136087, -0.0172117, -0.0127695, -0.0041098, 0.0036030
8: 0.0033380, 0.0060331, 0.0030742, 0.0067349, -0.0033970, 0.0029589
9: 0.0029783, 0.0081829, 0.0022352, 0.0081549, -0.0051766, 0.0059476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0051530, upper bound: 0.0060275
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0051530, upper bound: 0.0057966
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028773, -0.0040839, -0.0025848, -0.0014995, 0.0012067
1: -0.0056156, -0.0034453, -0.0055989, -0.0031468, -0.0024688, 0.0021536
2: 0.9660460, 0.9714196, 0.9648491, 0.9715527, -0.0055066, 0.0065705
3: 0.0229991, 0.0355015, 0.0231467, 0.0364830, -0.0095517, 0.0081976
4: -0.0033931, -0.0009831, -0.0034678, -0.0004612, -0.0029319, 0.0024846
5: 0.0128994, 0.0148020, 0.0125908, 0.0147907, -0.0018913, 0.0022112
6: 0.0030965, 0.0051745, 0.0026188, 0.0052112, -0.0021147, 0.0025557
7: -0.0169788, -0.0133574, -0.0172331, -0.0127154, -0.0042634, 0.0038757
8: 0.0032590, 0.0062433, 0.0030572, 0.0067802, -0.0035212, 0.0031861
9: 0.0027558, 0.0082096, 0.0021873, 0.0081550, -0.0053992, 0.0060222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027803, -0.0040844, -0.0028212, -0.0012625, 0.0013041
1: -0.0055914, -0.0033463, -0.0056157, -0.0033881, -0.0022034, 0.0022694
2: 0.9656492, 0.9714637, 0.9658166, 0.9714451, -0.0057958, 0.0056471
3: 0.0232130, 0.0358269, 0.0229978, 0.0356897, -0.0085179, 0.0089230
4: -0.0034179, -0.0008101, -0.0034074, -0.0008831, -0.0025348, 0.0025973
5: 0.0127971, 0.0147856, 0.0128402, 0.0148021, -0.0020051, 0.0019454
6: 0.0029381, 0.0051867, 0.0030049, 0.0051816, -0.0022434, 0.0021818
7: -0.0170631, -0.0131446, -0.0170275, -0.0132344, -0.0038288, 0.0038830
8: 0.0031921, 0.0064213, 0.0032203, 0.0063462, -0.0031541, 0.0032010
9: 0.0025674, 0.0081305, 0.0026468, 0.0082101, -0.0056427, 0.0054837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057121, upper bound: 0.0059009
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0056409, upper bound: 0.0055050
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026646, -0.0040844, -0.0027970, -0.0012870, 0.0014198
1: -0.0055987, -0.0032282, -0.0056158, -0.0033633, -0.0022354, 0.0023876
2: 0.9651754, 0.9715163, 0.9657173, 0.9714562, -0.0062808, 0.0057990
3: 0.0231488, 0.0362154, 0.0229975, 0.0357711, -0.0086778, 0.0090377
4: -0.0034474, -0.0006035, -0.0034136, -0.0008398, -0.0026076, 0.0028101
5: 0.0126750, 0.0147905, 0.0128146, 0.0148021, -0.0021272, 0.0019759
6: 0.0027491, 0.0052012, 0.0029653, 0.0051846, -0.0024355, 0.0022359
7: -0.0171638, -0.0128904, -0.0170486, -0.0131811, -0.0039827, 0.0041582
8: 0.0031122, 0.0066338, 0.0032036, 0.0063907, -0.0032785, 0.0034302
9: 0.0023423, 0.0081542, 0.0025997, 0.0082102, -0.0058678, 0.0055545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0062919
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027803, -0.0040839, -0.0026095, -0.0014742, 0.0013036
1: -0.0055914, -0.0033463, -0.0055989, -0.0031720, -0.0024195, 0.0022525
2: 0.9656492, 0.9714637, 0.9649500, 0.9715414, -0.0058922, 0.0065138
3: 0.0232130, 0.0358269, 0.0231471, 0.0364003, -0.0087831, 0.0083379
4: -0.0034179, -0.0008101, -0.0034615, -0.0005052, -0.0029127, 0.0026514
5: 0.0127971, 0.0147856, 0.0126168, 0.0147906, -0.0019936, 0.0021687
6: 0.0029381, 0.0051867, 0.0026591, 0.0052081, -0.0022700, 0.0025276
7: -0.0170631, -0.0131446, -0.0172117, -0.0127695, -0.0042936, 0.0040671
8: 0.0031921, 0.0064213, 0.0030742, 0.0067349, -0.0035429, 0.0033471
9: 0.0025674, 0.0081305, 0.0022352, 0.0081549, -0.0055875, 0.0058952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057286, upper bound: 0.0063544
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057286, upper bound: 0.0060612
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040839, -0.0026646, -0.0040839, -0.0025848, -0.0014991, 0.0014193
1: -0.0055987, -0.0032282, -0.0055989, -0.0031468, -0.0024519, 0.0023707
2: 0.9651754, 0.9715163, 0.9648491, 0.9715527, -0.0063773, 0.0066673
3: 0.0231488, 0.0362154, 0.0231467, 0.0364830, -0.0089848, 0.0084580
4: -0.0034474, -0.0006035, -0.0034678, -0.0004612, -0.0029862, 0.0028642
5: 0.0126750, 0.0147905, 0.0125908, 0.0147907, -0.0021157, 0.0021997
6: 0.0027491, 0.0052012, 0.0026188, 0.0052112, -0.0024622, 0.0025824
7: -0.0171638, -0.0128904, -0.0172331, -0.0127154, -0.0044484, 0.0043427
8: 0.0031122, 0.0066338, 0.0030572, 0.0067802, -0.0036680, 0.0035766
9: 0.0023423, 0.0081542, 0.0021873, 0.0081550, -0.0058127, 0.0059669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0064738
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067099
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040844, -0.0027180, -0.0013641, 0.0010922
1: -0.0055329, -0.0035626, -0.0056161, -0.0032827, -0.0022501, 0.0020535
2: 0.9665166, 0.9713672, 0.9653942, 0.9714921, -0.0049755, 0.0059730
3: 0.0237312, 0.0351158, 0.0229948, 0.0360360, -0.0089631, 0.0088717
4: -0.0033638, -0.0011883, -0.0034338, -0.0006989, -0.0026649, 0.0022455
5: 0.0130206, 0.0147457, 0.0127314, 0.0148024, -0.0017817, 0.0020144
6: 0.0032842, 0.0051601, 0.0028364, 0.0051945, -0.0019103, 0.0023237
7: -0.0168788, -0.0136098, -0.0171173, -0.0130078, -0.0038710, 0.0035075
8: 0.0033383, 0.0060323, 0.0031491, 0.0065357, -0.0031974, 0.0028832
9: 0.0029792, 0.0079389, 0.0024462, 0.0082112, -0.0052320, 0.0054926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040844, -0.0027180, -0.0013643, 0.0011980
1: -0.0055389, -0.0034545, -0.0056161, -0.0032827, -0.0022561, 0.0021615
2: 0.9660832, 0.9714154, 0.9653942, 0.9714921, -0.0054088, 0.0060212
3: 0.0236782, 0.0354711, 0.0229948, 0.0360360, -0.0090778, 0.0092966
4: -0.0033908, -0.0009993, -0.0034338, -0.0006989, -0.0026919, 0.0024345
5: 0.0129089, 0.0147498, 0.0127314, 0.0148024, -0.0018934, 0.0020185
6: 0.0031113, 0.0051734, 0.0028364, 0.0051945, -0.0020832, 0.0023370
7: -0.0169709, -0.0133773, -0.0171173, -0.0130078, -0.0039631, 0.0037400
8: 0.0032652, 0.0062266, 0.0031491, 0.0065357, -0.0032704, 0.0030775
9: 0.0027734, 0.0079585, 0.0024462, 0.0082112, -0.0054377, 0.0055122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040844, -0.0026219, -0.0014603, 0.0010922
1: -0.0055329, -0.0035626, -0.0056175, -0.0031846, -0.0023483, 0.0020549
2: 0.9665166, 0.9713672, 0.9650006, 0.9715359, -0.0050193, 0.0063666
3: 0.0237312, 0.0351158, 0.0229821, 0.0363587, -0.0092320, 0.0088098
4: -0.0033638, -0.0011883, -0.0034583, -0.0005273, -0.0028365, 0.0022701
5: 0.0130206, 0.0147457, 0.0126299, 0.0148033, -0.0017827, 0.0021159
6: 0.0032842, 0.0051601, 0.0026793, 0.0052066, -0.0019223, 0.0024808
7: -0.0168788, -0.0136098, -0.0172009, -0.0127967, -0.0040822, 0.0035912
8: 0.0033383, 0.0060323, 0.0030827, 0.0067122, -0.0033739, 0.0029495
9: 0.0029792, 0.0079389, 0.0022593, 0.0082159, -0.0052366, 0.0056795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040844, -0.0026219, -0.0014604, 0.0011981
1: -0.0055389, -0.0034545, -0.0056175, -0.0031846, -0.0023543, 0.0021630
2: 0.9660832, 0.9714154, 0.9650006, 0.9715359, -0.0054526, 0.0064148
3: 0.0236782, 0.0354711, 0.0229821, 0.0363587, -0.0091573, 0.0090706
4: -0.0033908, -0.0009993, -0.0034583, -0.0005273, -0.0028635, 0.0024590
5: 0.0129089, 0.0147498, 0.0126299, 0.0148033, -0.0018944, 0.0021199
6: 0.0031113, 0.0051734, 0.0026793, 0.0052066, -0.0020953, 0.0024941
7: -0.0169709, -0.0133773, -0.0172009, -0.0127967, -0.0041742, 0.0038236
8: 0.0032652, 0.0062266, 0.0030827, 0.0067122, -0.0034470, 0.0031439
9: 0.0027734, 0.0079585, 0.0022593, 0.0082159, -0.0054424, 0.0056991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066899
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040850, -0.0027957, -0.0012860, 0.0013075
1: -0.0055180, -0.0033434, -0.0056383, -0.0033620, -0.0021559, 0.0022949
2: 0.9656377, 0.9714649, 0.9657123, 0.9714566, -0.0058190, 0.0057526
3: 0.0238631, 0.0358364, 0.0227978, 0.0357753, -0.0088126, 0.0099319
4: -0.0034186, -0.0008051, -0.0034139, -0.0008376, -0.0025810, 0.0026089
5: 0.0127941, 0.0147356, 0.0128133, 0.0148175, -0.0020234, 0.0019223
6: 0.0029335, 0.0051870, 0.0029633, 0.0051848, -0.0022512, 0.0022238
7: -0.0170656, -0.0131384, -0.0170497, -0.0131783, -0.0038872, 0.0039113
8: 0.0031901, 0.0064265, 0.0032027, 0.0063930, -0.0032029, 0.0032238
9: 0.0025619, 0.0078901, 0.0025972, 0.0082840, -0.0057222, 0.0052928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065225, upper bound: 0.0061319
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065225, upper bound: 0.0062922
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040850, -0.0027732, -0.0013087, 0.0014098
1: -0.0055218, -0.0032390, -0.0056384, -0.0033390, -0.0021828, 0.0023993
2: 0.9652189, 0.9715115, 0.9656199, 0.9714670, -0.0062481, 0.0058916
3: 0.0238290, 0.0361797, 0.0227976, 0.0358510, -0.0089443, 0.0102164
4: -0.0034447, -0.0006225, -0.0034197, -0.0007973, -0.0026474, 0.0027972
5: 0.0126862, 0.0147382, 0.0127895, 0.0148175, -0.0021313, 0.0019487
6: 0.0027664, 0.0051999, 0.0029264, 0.0051876, -0.0024212, 0.0022734
7: -0.0171545, -0.0129138, -0.0170693, -0.0131288, -0.0040257, 0.0041556
8: 0.0031195, 0.0066143, 0.0031871, 0.0064344, -0.0033149, 0.0034271
9: 0.0023630, 0.0079027, 0.0025534, 0.0082841, -0.0059211, 0.0053493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0061319
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0062922
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040844, -0.0025930, -0.0014887, 0.0013069
1: -0.0055180, -0.0033434, -0.0056175, -0.0031552, -0.0023628, 0.0022741
2: 0.9656377, 0.9714649, 0.9648827, 0.9715489, -0.0059112, 0.0065823
3: 0.0238631, 0.0358364, 0.0229822, 0.0364554, -0.0089946, 0.0093073
4: -0.0034186, -0.0008051, -0.0034657, -0.0004759, -0.0029427, 0.0026606
5: 0.0127941, 0.0147356, 0.0125995, 0.0148033, -0.0020092, 0.0021361
6: 0.0029335, 0.0051870, 0.0026322, 0.0052102, -0.0022767, 0.0025548
7: -0.0170656, -0.0131384, -0.0172260, -0.0127334, -0.0043322, 0.0040876
8: 0.0031901, 0.0064265, 0.0030629, 0.0067651, -0.0035750, 0.0033636
9: 0.0025619, 0.0078901, 0.0022033, 0.0082158, -0.0056540, 0.0056868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0065425
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040844, -0.0025704, -0.0015114, 0.0014092
1: -0.0055218, -0.0032390, -0.0056175, -0.0031321, -0.0023897, 0.0023785
2: 0.9652189, 0.9715115, 0.9647900, 0.9715592, -0.0063403, 0.0067214
3: 0.0238290, 0.0361797, 0.0229819, 0.0365314, -0.0091726, 0.0095766
4: -0.0034447, -0.0006225, -0.0034715, -0.0004355, -0.0030092, 0.0028490
5: 0.0126862, 0.0147382, 0.0125756, 0.0148033, -0.0021172, 0.0021626
6: 0.0027664, 0.0051999, 0.0025953, 0.0052130, -0.0024466, 0.0026046
7: -0.0171545, -0.0129138, -0.0172457, -0.0126837, -0.0044708, 0.0043319
8: 0.0031195, 0.0066143, 0.0030472, 0.0068067, -0.0036871, 0.0035670
9: 0.0023630, 0.0079027, 0.0021593, 0.0082159, -0.0058529, 0.0057434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0065425
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0067141
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040822, -0.0029922, -0.0040864, -0.0025960, -0.0014862, 0.0010942
1: -0.0055329, -0.0035626, -0.0056909, -0.0031582, -0.0023747, 0.0021283
2: 0.9665166, 0.9713672, 0.9648947, 0.9715475, -0.0050309, 0.0064725
3: 0.0237312, 0.0351158, 0.0223328, 0.0364455, -0.0094054, 0.0095265
4: -0.0033638, -0.0011883, -0.0034649, -0.0004811, -0.0028827, 0.0022767
5: 0.0130206, 0.0147457, 0.0126026, 0.0148532, -0.0018326, 0.0021431
6: 0.0032842, 0.0051601, 0.0026370, 0.0052098, -0.0019256, 0.0025231
7: -0.0168788, -0.0136098, -0.0172234, -0.0127399, -0.0041390, 0.0036137
8: 0.0033383, 0.0060323, 0.0030649, 0.0067597, -0.0034214, 0.0029674
9: 0.0029792, 0.0079389, 0.0022090, 0.0084560, -0.0054768, 0.0057298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0061268
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0058907
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040823, -0.0028863, -0.0040864, -0.0025727, -0.0015096, 0.0012000
1: -0.0055389, -0.0034545, -0.0056909, -0.0031344, -0.0024044, 0.0022364
2: 0.9660832, 0.9714154, 0.9647995, 0.9715582, -0.0054749, 0.0066159
3: 0.0236782, 0.0354711, 0.0223324, 0.0365237, -0.0096015, 0.0097052
4: -0.0033908, -0.0009993, -0.0034709, -0.0004396, -0.0029512, 0.0024715
5: 0.0129089, 0.0147498, 0.0125780, 0.0148533, -0.0019443, 0.0021718
6: 0.0031113, 0.0051734, 0.0025990, 0.0052127, -0.0021014, 0.0025744
7: -0.0169709, -0.0133773, -0.0172437, -0.0126887, -0.0042822, 0.0038663
8: 0.0032652, 0.0062266, 0.0030488, 0.0068025, -0.0035372, 0.0031778
9: 0.0027734, 0.0079585, 0.0021638, 0.0084561, -0.0056827, 0.0057947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062915, upper bound: 0.0064619
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0062915, upper bound: 0.0064619
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040869, -0.0028018, -0.0012800, 0.0013094
1: -0.0055180, -0.0033434, -0.0057107, -0.0033682, -0.0021498, 0.0023672
2: 0.9656377, 0.9714649, 0.9657370, 0.9714540, -0.0058163, 0.0057279
3: 0.0238631, 0.0358364, 0.0221575, 0.0357550, -0.0088324, 0.0106313
4: -0.0034186, -0.0008051, -0.0034124, -0.0008484, -0.0025702, 0.0026073
5: 0.0127941, 0.0147356, 0.0128197, 0.0148667, -0.0020726, 0.0019159
6: 0.0029335, 0.0051870, 0.0029732, 0.0051840, -0.0022505, 0.0022139
7: -0.0170656, -0.0131384, -0.0170445, -0.0131916, -0.0038739, 0.0039061
8: 0.0031901, 0.0064265, 0.0032069, 0.0063819, -0.0031918, 0.0032196
9: 0.0025619, 0.0078901, 0.0026090, 0.0085208, -0.0059589, 0.0052811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057835, upper bound: 0.0059736
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0057100, upper bound: 0.0055301
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040869, -0.0027780, -0.0013039, 0.0014117
1: -0.0055218, -0.0032390, -0.0057107, -0.0033440, -0.0021779, 0.0024717
2: 0.9652189, 0.9715115, 0.9656399, 0.9714647, -0.0062458, 0.0058716
3: 0.0238290, 0.0361797, 0.0221572, 0.0358347, -0.0089711, 0.0108171
4: -0.0034447, -0.0006225, -0.0034185, -0.0008060, -0.0026387, 0.0027960
5: 0.0126862, 0.0147382, 0.0127946, 0.0148667, -0.0021806, 0.0019436
6: 0.0027664, 0.0051999, 0.0029344, 0.0051870, -0.0024206, 0.0022655
7: -0.0171545, -0.0129138, -0.0170651, -0.0131395, -0.0040150, 0.0041513
8: 0.0031195, 0.0066143, 0.0031905, 0.0064255, -0.0033060, 0.0034238
9: 0.0023630, 0.0079027, 0.0025629, 0.0085209, -0.0061579, 0.0053398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065980, upper bound: 0.0060624
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0065980, upper bound: 0.0062919
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040818, -0.0027775, -0.0040864, -0.0025960, -0.0014858, 0.0013089
1: -0.0055180, -0.0033434, -0.0056909, -0.0031582, -0.0023598, 0.0023474
2: 0.9656377, 0.9714649, 0.9648947, 0.9715475, -0.0059098, 0.0065702
3: 0.0238631, 0.0358364, 0.0223328, 0.0364455, -0.0090199, 0.0100156
4: -0.0034186, -0.0008051, -0.0034649, -0.0004811, -0.0029375, 0.0026599
5: 0.0127941, 0.0147356, 0.0126026, 0.0148532, -0.0020591, 0.0021330
6: 0.0029335, 0.0051870, 0.0026370, 0.0052098, -0.0022763, 0.0025500
7: -0.0170656, -0.0131384, -0.0172234, -0.0127399, -0.0043257, 0.0040850
8: 0.0031901, 0.0064265, 0.0030649, 0.0067597, -0.0035696, 0.0033616
9: 0.0025619, 0.0078901, 0.0022090, 0.0084560, -0.0058941, 0.0056811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0064176
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0061017
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040819, -0.0026752, -0.0040864, -0.0025727, -0.0015091, 0.0014112
1: -0.0055218, -0.0032390, -0.0056909, -0.0031344, -0.0023874, 0.0024519
2: 0.9652189, 0.9715115, 0.9647995, 0.9715582, -0.0063393, 0.0067120
3: 0.0238290, 0.0361797, 0.0223324, 0.0365237, -0.0092050, 0.0101941
4: -0.0034447, -0.0006225, -0.0034709, -0.0004396, -0.0030051, 0.0028484
5: 0.0126862, 0.0147382, 0.0125780, 0.0148533, -0.0021671, 0.0021602
6: 0.0027664, 0.0051999, 0.0025990, 0.0052127, -0.0024463, 0.0026009
7: -0.0171545, -0.0129138, -0.0172437, -0.0126887, -0.0044658, 0.0043299
8: 0.0031195, 0.0066143, 0.0030488, 0.0068025, -0.0036829, 0.0035655
9: 0.0023630, 0.0079027, 0.0021638, 0.0084561, -0.0060931, 0.0057389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067138, upper bound: 0.0064738
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0067138, upper bound: 0.0067099
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040842, -0.0029917, -0.0040844, -0.0026219, -0.0014623, 0.0010927
1: -0.0056074, -0.0035621, -0.0056175, -0.0031846, -0.0024228, 0.0020554
2: 0.9665148, 0.9713675, 0.9650006, 0.9715359, -0.0050211, 0.0063668
3: 0.0230714, 0.0351174, 0.0229821, 0.0363587, -0.0100994, 0.0090012
4: -0.0033639, -0.0011874, -0.0034583, -0.0005273, -0.0028366, 0.0022709
5: 0.0130201, 0.0147965, 0.0126299, 0.0148033, -0.0017832, 0.0021666
6: 0.0032835, 0.0051602, 0.0026793, 0.0052066, -0.0019231, 0.0024809
7: -0.0168792, -0.0136087, -0.0172009, -0.0127967, -0.0040826, 0.0035922
8: 0.0033380, 0.0060331, 0.0030827, 0.0067122, -0.0033742, 0.0029504
9: 0.0029783, 0.0081829, 0.0022593, 0.0082159, -0.0052376, 0.0059235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040844, -0.0028773, -0.0040844, -0.0026219, -0.0014625, 0.0012072
1: -0.0056156, -0.0034453, -0.0056175, -0.0031846, -0.0024310, 0.0021722
2: 0.9660460, 0.9714196, 0.9650006, 0.9715359, -0.0054898, 0.0064189
3: 0.0229991, 0.0355015, 0.0229821, 0.0363587, -0.0100382, 0.0092982
4: -0.0033931, -0.0009831, -0.0034583, -0.0005273, -0.0028658, 0.0024752
5: 0.0128994, 0.0148020, 0.0126299, 0.0148033, -0.0019040, 0.0021721
6: 0.0030965, 0.0051745, 0.0026793, 0.0052066, -0.0021101, 0.0024952
7: -0.0169788, -0.0133574, -0.0172009, -0.0127967, -0.0041821, 0.0038435
8: 0.0032590, 0.0062433, 0.0030827, 0.0067122, -0.0034532, 0.0031605
9: 0.0027558, 0.0082096, 0.0022593, 0.0082159, -0.0054601, 0.0059503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066786
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040837, -0.0027803, -0.0040850, -0.0027957, -0.0012880, 0.0013047
1: -0.0055914, -0.0033463, -0.0056383, -0.0033620, -0.0022294, 0.0022920
2: 0.9656492, 0.9714637, 0.9657123, 0.9714566, -0.0058074, 0.0057514
3: 0.0232130, 0.0358269, 0.0227978, 0.0357753, -0.0096041, 0.0100668
4: -0.0034179, -0.0008101, -0.0034139, -0.0008376, -0.0025803, 0.0026038
5: 0.0127971, 0.0147856, 0.0128133, 0.0148175, -0.0020204, 0.0019723
6: 0.0029381, 0.0051867, 0.0029633, 0.0051848, -0.0022466, 0.0022234
7: -0.0170631, -0.0131446, -0.0170497, -0.0131783, -0.0038848, 0.0039052
8: 0.0031921, 0.0064213, 0.0032027, 0.0063930, -0.0032010, 0.0032186
9: 0.0025674, 0.0081305, 0.0025972, 0.0082840, -0.0057167, 0.0055332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064609, upper bound: 0.0061319
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0064609, upper bound: 0.0062910
time: 1.63 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.62 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066899
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065223, upper bound: 0.0061319
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065223, upper bound: 0.0062922
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0061319
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0062922
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0065425
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0065425
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0067141
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0061268
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0058907
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0064619
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0062910, upper bound: 0.0066947
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0057816, upper bound: 0.0059736
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0056970, upper bound: 0.0055301
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0060624
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065968, upper bound: 0.0062919
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0064176
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0061017
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0064738
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067127, upper bound: 0.0067099
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066786
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0064606, upper bound: 0.0061319
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0064606, upper bound: 0.0062910
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066503, upper bound: 0.0061319
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066503, upper bound: 0.0061319
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0065425
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0064738, upper bound: 0.0067072
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0065425
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067141
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0051530, upper bound: 0.0060275
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0051530, upper bound: 0.0057966
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0064548
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0057121, upper bound: 0.0059009
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0056409, upper bound: 0.0055050
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0060624
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066308, upper bound: 0.0062919
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0057286, upper bound: 0.0063544
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0057286, upper bound: 0.0060612
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0064738
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067099, upper bound: 0.0067099
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065223
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0061319
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065239
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0062922
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066899
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065225, upper bound: 0.0061319
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065225, upper bound: 0.0062922
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0061319
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0066558, upper bound: 0.0062922
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0065425
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065425, upper bound: 0.0067141
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0065425
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067141, upper bound: 0.0067141
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0061268
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0053155, upper bound: 0.0058907
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0062915, upper bound: 0.0064619
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0062915, upper bound: 0.0064619
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0057835, upper bound: 0.0059736
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0057100, upper bound: 0.0055301
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065980, upper bound: 0.0060624
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0065980, upper bound: 0.0062919
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0064176
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0058212, upper bound: 0.0061017
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067138, upper bound: 0.0064738
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0067138, upper bound: 0.0067099
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062887
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065969
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0062922
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066786
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0064609, upper bound: 0.0061319
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.62
Output dim: 2, lower bound: -0.0064609, upper bound: 0.0062910
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066949, upper bound: 0.0062922
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064624, upper bound: 0.0067072
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066949, upper bound: 0.0067141
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065983
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066812
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0062919
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064564, upper bound: 0.0062919
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066920, upper bound: 0.0067099
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065225
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065243
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066940
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0067141
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0062919
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064953, upper bound: 0.0067099
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0067099
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065980
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066822
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065997
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0062919
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066913, upper bound: 0.0067099
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065225
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0065243
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066558
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066940
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0062922
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0065239, upper bound: 0.0067141
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066935, upper bound: 0.0062922
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0061319, upper bound: 0.0066502
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0062922, upper bound: 0.0066949
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0062919
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064952, upper bound: 0.0067099
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066793, upper bound: 0.0062919
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065980
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0066822
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0062915
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0062922
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064619, upper bound: 0.0067118
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066947, upper bound: 0.0067141
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0060624, upper bound: 0.0065997
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0062919, upper bound: 0.0066836
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0062919
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0062919
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0064548, upper bound: 0.0067078
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.62
Output dim: 2, lower bound: -0.0066912, upper bound: 0.0067099

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.08 + 597.60 = 600.68 seconds
