## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00079709


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0028369, -0.0016371, -0.0028369, -0.0016371, -0.0008205, 0.0008205)
1: (-0.0115095, -0.0084650, -0.0115095, -0.0084650, -0.0020821, 0.0020821)
2: (0.0278895, 0.0297783, 0.0278895, 0.0297783, -0.0012917, 0.0012917)
3: (0.0039588, 0.0074858, 0.0039588, 0.0074858, -0.0024120, 0.0024120)
4: (-0.0106001, -0.0075033, -0.0106001, -0.0075033, -0.0021178, 0.0021178)
5: (0.0097231, 0.0108961, 0.0097231, 0.0108961, -0.0008022, 0.0008022)
6: (0.0054124, 0.0098886, 0.0054124, 0.0098886, -0.0030611, 0.0030611)
7: (0.9818466, 0.9849788, 0.9818466, 0.9849788, -0.0021420, 0.0021420)
8: (-0.0060275, -0.0026693, -0.0060275, -0.0026693, -0.0022966, 0.0022966)
9: (-0.0032364, -0.0010181, -0.0032364, -0.0010181, -0.0015170, 0.0015170)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.46 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0012990, upper bound: 0.0012990

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012550, upper bound: 0.0012596
time: 0.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012629, upper bound: 0.0012630
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 7, lower bound: -0.0012550, upper bound: 0.0012596
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 7, lower bound: -0.0012629, upper bound: 0.0012630

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0028299, -0.0016857, -0.0028356, -0.0016524, -0.0007774, 0.0007676
1: -0.0114917, -0.0085884, -0.0115062, -0.0085037, -0.0019728, 0.0019480
2: 0.0279005, 0.0297018, 0.0278915, 0.0297543, -0.0012239, 0.0012085
3: 0.0041018, 0.0074652, 0.0040037, 0.0074820, -0.0022566, 0.0022854
4: -0.0105821, -0.0076288, -0.0105968, -0.0075427, -0.0020067, 0.0019814
5: 0.0097300, 0.0108486, 0.0097244, 0.0108812, -0.0007601, 0.0007505
6: 0.0055938, 0.0098625, 0.0054693, 0.0098837, -0.0028639, 0.0029005
7: 0.9819735, 0.9849605, 0.9818865, 0.9849754, -0.0020040, 0.0020296
8: -0.0058914, -0.0026889, -0.0059848, -0.0026729, -0.0021486, 0.0021761
9: -0.0032234, -0.0011080, -0.0032340, -0.0010463, -0.0014374, 0.0014193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012090, upper bound: 0.0011997
time: 0.62 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012168, upper bound: 0.0012201
time: 0.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0028354, -0.0016624, -0.0028365, -0.0016438, -0.0008035, 0.0007655
1: -0.0115059, -0.0085290, -0.0115085, -0.0084819, -0.0020391, 0.0019427
2: 0.0278917, 0.0297386, 0.0278901, 0.0297678, -0.0012651, 0.0012053
3: 0.0040331, 0.0074816, 0.0039785, 0.0074847, -0.0022505, 0.0023622
4: -0.0105965, -0.0075685, -0.0105992, -0.0075205, -0.0020741, 0.0019760
5: 0.0097245, 0.0108714, 0.0097235, 0.0108896, -0.0007856, 0.0007485
6: 0.0055066, 0.0098833, 0.0054373, 0.0098872, -0.0028562, 0.0029979
7: 0.9819126, 0.9849752, 0.9818640, 0.9849779, -0.0019986, 0.0020978
8: -0.0059569, -0.0026733, -0.0060089, -0.0026704, -0.0021428, 0.0022492
9: -0.0032337, -0.0010647, -0.0032357, -0.0010304, -0.0014857, 0.0014155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012154, upper bound: 0.0012028
time: 0.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0012232, upper bound: 0.0012233
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0012090, upper bound: 0.0011997
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0012168, upper bound: 0.0012201
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0012154, upper bound: 0.0012028
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0012232, upper bound: 0.0012233

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028291, -0.0017017, -0.0028373, -0.0017040, -0.0007300, 0.0007485
1: -0.0114898, -0.0086289, -0.0115106, -0.0086347, -0.0018525, 0.0018994
2: 0.0279017, 0.0296766, 0.0278888, 0.0296730, -0.0011493, 0.0011784
3: 0.0041488, 0.0074630, 0.0041555, 0.0074871, -0.0022004, 0.0021461
4: -0.0105801, -0.0076701, -0.0106013, -0.0076760, -0.0018844, 0.0019320
5: 0.0097307, 0.0108330, 0.0097227, 0.0108307, -0.0007137, 0.0007318
6: 0.0056535, 0.0098596, 0.0056620, 0.0098902, -0.0027926, 0.0027237
7: 0.9820153, 0.9849585, 0.9820213, 0.9849799, -0.0019541, 0.0019059
8: -0.0058467, -0.0026910, -0.0058403, -0.0026681, -0.0020951, 0.0020434
9: -0.0032220, -0.0011375, -0.0032372, -0.0011418, -0.0013498, 0.0013839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011518, upper bound: 0.0011297
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011518, upper bound: 0.0011423
time: 0.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028296, -0.0016903, -0.0028341, -0.0016770, -0.0007625, 0.0007494
1: -0.0114911, -0.0085999, -0.0115024, -0.0085661, -0.0019350, 0.0019017
2: 0.0279009, 0.0296946, 0.0278939, 0.0297155, -0.0012005, 0.0011799
3: 0.0041152, 0.0074644, 0.0040761, 0.0074776, -0.0022031, 0.0022416
4: -0.0105814, -0.0076406, -0.0105929, -0.0076062, -0.0019682, 0.0019344
5: 0.0097302, 0.0108441, 0.0097259, 0.0108571, -0.0007455, 0.0007327
6: 0.0056108, 0.0098614, 0.0055612, 0.0098781, -0.0027960, 0.0028449
7: 0.9819854, 0.9849598, 0.9819507, 0.9849714, -0.0019565, 0.0019907
8: -0.0058787, -0.0026897, -0.0059159, -0.0026771, -0.0020977, 0.0021343
9: -0.0032229, -0.0011164, -0.0032312, -0.0010918, -0.0014098, 0.0013856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011586, upper bound: 0.0011492
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011586, upper bound: 0.0011642
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028347, -0.0016777, -0.0028381, -0.0016953, -0.0007548, 0.0007581
1: -0.0115040, -0.0085681, -0.0115127, -0.0086127, -0.0019155, 0.0019239
2: 0.0278929, 0.0297144, 0.0278875, 0.0296867, -0.0011884, 0.0011936
3: 0.0040783, 0.0074794, 0.0041300, 0.0074895, -0.0022287, 0.0022190
4: -0.0105945, -0.0076082, -0.0106034, -0.0076536, -0.0019484, 0.0019569
5: 0.0097253, 0.0108564, 0.0097219, 0.0108392, -0.0007380, 0.0007412
6: 0.0055640, 0.0098805, 0.0056296, 0.0098933, -0.0028286, 0.0028163
7: 0.9819527, 0.9849732, 0.9819986, 0.9849821, -0.0019793, 0.0019707
8: -0.0059138, -0.0026754, -0.0058645, -0.0026658, -0.0021221, 0.0021129
9: -0.0032324, -0.0010932, -0.0032387, -0.0011257, -0.0013957, 0.0014018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011624, upper bound: 0.0011345
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011624, upper bound: 0.0011469
time: 0.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028352, -0.0016671, -0.0028349, -0.0016686, -0.0007851, 0.0007482
1: -0.0115052, -0.0085412, -0.0115046, -0.0085449, -0.0019924, 0.0018985
2: 0.0278921, 0.0297310, 0.0278925, 0.0297288, -0.0012361, 0.0011779
3: 0.0040471, 0.0074808, 0.0040514, 0.0074801, -0.0021994, 0.0023081
4: -0.0105958, -0.0075808, -0.0105951, -0.0075846, -0.0020266, 0.0019311
5: 0.0097248, 0.0108668, 0.0097250, 0.0108653, -0.0007676, 0.0007315
6: 0.0055244, 0.0098822, 0.0055299, 0.0098813, -0.0027913, 0.0029293
7: 0.9819250, 0.9849744, 0.9819288, 0.9849737, -0.0019532, 0.0020498
8: -0.0059435, -0.0026741, -0.0059394, -0.0026748, -0.0020941, 0.0021977
9: -0.0032332, -0.0010736, -0.0032328, -0.0010763, -0.0014517, 0.0013833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011682, upper bound: 0.0011533
time: 0.64 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011682, upper bound: 0.0011682
time: 0.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.94 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011518, upper bound: 0.0011297
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011518, upper bound: 0.0011423
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011586, upper bound: 0.0011492
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011586, upper bound: 0.0011642
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011624, upper bound: 0.0011345
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011624, upper bound: 0.0011469
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011682, upper bound: 0.0011533
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 7, lower bound: -0.0011682, upper bound: 0.0011682

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028291, -0.0017017, -0.0028366, -0.0017204, -0.0007163, 0.0007480
1: -0.0114898, -0.0086289, -0.0115088, -0.0086764, -0.0018178, 0.0018981
2: 0.0279017, 0.0296766, 0.0278899, 0.0296471, -0.0011278, 0.0011776
3: 0.0041488, 0.0074630, 0.0042038, 0.0074849, -0.0021989, 0.0021058
4: -0.0105801, -0.0076701, -0.0105994, -0.0077184, -0.0018490, 0.0019307
5: 0.0097307, 0.0108330, 0.0097234, 0.0108147, -0.0007004, 0.0007313
6: 0.0056535, 0.0098596, 0.0057233, 0.0098875, -0.0027906, 0.0026726
7: 0.9820153, 0.9849585, 0.9820641, 0.9849781, -0.0019527, 0.0018701
8: -0.0058467, -0.0026910, -0.0057943, -0.0026701, -0.0020937, 0.0020051
9: -0.0032220, -0.0011375, -0.0032358, -0.0011721, -0.0013245, 0.0013830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011297
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011297
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028282, -0.0017157, -0.0029051, -0.0017482, -0.0007161, 0.0008057
1: -0.0114876, -0.0086644, -0.0116827, -0.0087469, -0.0018172, 0.0020446
2: 0.0279031, 0.0296546, 0.0277820, 0.0296034, -0.0011274, 0.0012685
3: 0.0041898, 0.0074604, 0.0042855, 0.0076864, -0.0023685, 0.0021052
4: -0.0105778, -0.0077062, -0.0107763, -0.0077901, -0.0018484, 0.0020797
5: 0.0097316, 0.0108193, 0.0096564, 0.0107875, -0.0007001, 0.0007877
6: 0.0057056, 0.0098563, 0.0058269, 0.0101432, -0.0030060, 0.0026718
7: 0.9820517, 0.9849563, 0.9821366, 0.9851570, -0.0021034, 0.0018696
8: -0.0058076, -0.0026935, -0.0057165, -0.0024783, -0.0022552, 0.0020045
9: -0.0032204, -0.0011634, -0.0033625, -0.0012235, -0.0013241, 0.0014897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028296, -0.0016903, -0.0028333, -0.0016936, -0.0007483, 0.0007489
1: -0.0114911, -0.0085999, -0.0115005, -0.0086082, -0.0018989, 0.0019003
2: 0.0279009, 0.0296946, 0.0278951, 0.0296894, -0.0011781, 0.0011790
3: 0.0041152, 0.0074644, 0.0041248, 0.0074754, -0.0022014, 0.0021998
4: -0.0105814, -0.0076406, -0.0105910, -0.0076490, -0.0019315, 0.0019329
5: 0.0097302, 0.0108441, 0.0097266, 0.0108409, -0.0007316, 0.0007321
6: 0.0056108, 0.0098614, 0.0056230, 0.0098753, -0.0027939, 0.0027918
7: 0.9819854, 0.9849598, 0.9819940, 0.9849695, -0.0019550, 0.0019536
8: -0.0058787, -0.0026897, -0.0058695, -0.0026793, -0.0020961, 0.0020945
9: -0.0032229, -0.0011164, -0.0032298, -0.0011225, -0.0013836, 0.0013846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011428
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011393
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028287, -0.0017045, -0.0029021, -0.0017203, -0.0007487, 0.0008068
1: -0.0114888, -0.0086359, -0.0116750, -0.0086760, -0.0018999, 0.0020473
2: 0.0279023, 0.0296722, 0.0277868, 0.0296474, -0.0011787, 0.0012701
3: 0.0041569, 0.0074618, 0.0042034, 0.0076775, -0.0023717, 0.0022009
4: -0.0105791, -0.0076772, -0.0107684, -0.0077180, -0.0019325, 0.0020824
5: 0.0097311, 0.0108303, 0.0096594, 0.0108148, -0.0007320, 0.0007888
6: 0.0056638, 0.0098581, 0.0057227, 0.0101318, -0.0030099, 0.0027932
7: 0.9820225, 0.9849575, 0.9820638, 0.9851490, -0.0021062, 0.0019546
8: -0.0058389, -0.0026921, -0.0057947, -0.0024868, -0.0022582, 0.0020956
9: -0.0032213, -0.0011427, -0.0033569, -0.0011719, -0.0013843, 0.0014917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011571
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011525
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028347, -0.0016777, -0.0028374, -0.0017117, -0.0007411, 0.0007576
1: -0.0115040, -0.0085681, -0.0115109, -0.0086542, -0.0018806, 0.0019225
2: 0.0278929, 0.0297144, 0.0278886, 0.0296609, -0.0011667, 0.0011927
3: 0.0040783, 0.0074794, 0.0041781, 0.0074874, -0.0022271, 0.0021785
4: -0.0105945, -0.0076082, -0.0106015, -0.0076958, -0.0019129, 0.0019555
5: 0.0097253, 0.0108564, 0.0097226, 0.0108232, -0.0007245, 0.0007407
6: 0.0055640, 0.0098805, 0.0056907, 0.0098905, -0.0028265, 0.0027649
7: 0.9819527, 0.9849732, 0.9820413, 0.9849802, -0.0019779, 0.0019347
8: -0.0059138, -0.0026754, -0.0058188, -0.0026678, -0.0021206, 0.0020743
9: -0.0032324, -0.0010932, -0.0032373, -0.0011560, -0.0013702, 0.0014008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011479, upper bound: 0.0011345
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011479, upper bound: 0.0011344
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028338, -0.0016918, -0.0029059, -0.0017394, -0.0007410, 0.0008191
1: -0.0115018, -0.0086038, -0.0116848, -0.0087246, -0.0018805, 0.0020785
2: 0.0278943, 0.0296922, 0.0277807, 0.0296172, -0.0011667, 0.0012895
3: 0.0041196, 0.0074768, 0.0042596, 0.0076889, -0.0024078, 0.0021785
4: -0.0105923, -0.0076445, -0.0107784, -0.0077674, -0.0019128, 0.0021142
5: 0.0097261, 0.0108426, 0.0096556, 0.0107961, -0.0007245, 0.0008008
6: 0.0056165, 0.0098772, 0.0057941, 0.0101463, -0.0030558, 0.0027647
7: 0.9819894, 0.9849708, 0.9821137, 0.9851592, -0.0021383, 0.0019346
8: -0.0058744, -0.0026779, -0.0057412, -0.0024760, -0.0022926, 0.0020742
9: -0.0032307, -0.0011192, -0.0033641, -0.0012072, -0.0013701, 0.0015144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011571, upper bound: 0.0011364
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011571, upper bound: 0.0011416
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028352, -0.0016671, -0.0028341, -0.0016852, -0.0007705, 0.0007475
1: -0.0115052, -0.0085412, -0.0115025, -0.0085869, -0.0019551, 0.0018969
2: 0.0278921, 0.0297310, 0.0278938, 0.0297027, -0.0012130, 0.0011768
3: 0.0040471, 0.0074808, 0.0041001, 0.0074777, -0.0021975, 0.0022649
4: -0.0105958, -0.0075808, -0.0105930, -0.0076273, -0.0019887, 0.0019295
5: 0.0097248, 0.0108668, 0.0097258, 0.0108491, -0.0007533, 0.0007308
6: 0.0055244, 0.0098822, 0.0055917, 0.0098783, -0.0027889, 0.0028745
7: 0.9819250, 0.9849744, 0.9819721, 0.9849716, -0.0019515, 0.0020114
8: -0.0059435, -0.0026741, -0.0058930, -0.0026770, -0.0020923, 0.0021566
9: -0.0032332, -0.0010736, -0.0032313, -0.0011069, -0.0014245, 0.0013821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011474, upper bound: 0.0011479
time: 0.65 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011474, upper bound: 0.0011434
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028343, -0.0016811, -0.0029029, -0.0017114, -0.0007713, 0.0008131
1: -0.0115029, -0.0085766, -0.0116771, -0.0086535, -0.0019572, 0.0020633
2: 0.0278936, 0.0297091, 0.0277855, 0.0296614, -0.0012143, 0.0012801
3: 0.0040882, 0.0074782, 0.0041772, 0.0076799, -0.0023902, 0.0022674
4: -0.0105934, -0.0076169, -0.0107706, -0.0076950, -0.0019908, 0.0020987
5: 0.0097257, 0.0108531, 0.0096586, 0.0108235, -0.0007541, 0.0007949
6: 0.0055765, 0.0098789, 0.0056895, 0.0101349, -0.0030335, 0.0028776
7: 0.9819614, 0.9849721, 0.9820405, 0.9851512, -0.0021227, 0.0020136
8: -0.0059044, -0.0026766, -0.0058196, -0.0024845, -0.0022758, 0.0021589
9: -0.0032316, -0.0010994, -0.0033585, -0.0011554, -0.0014261, 0.0015033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011469, upper bound: 0.0011624
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011469, upper bound: 0.0011565
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.00 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011297
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011297
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011428
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011393
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011571
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011525
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011479, upper bound: 0.0011345
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011479, upper bound: 0.0011344
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011571, upper bound: 0.0011364
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011571, upper bound: 0.0011416
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011474, upper bound: 0.0011479
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011474, upper bound: 0.0011434
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011469, upper bound: 0.0011624
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0011469, upper bound: 0.0011565

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028283, -0.0017184, -0.0028366, -0.0017204, -0.0007158, 0.0007340
1: -0.0114879, -0.0086714, -0.0115088, -0.0086764, -0.0018164, 0.0018628
2: 0.0279029, 0.0296503, 0.0278899, 0.0296471, -0.0011269, 0.0011557
3: 0.0041980, 0.0074608, 0.0042038, 0.0074849, -0.0021579, 0.0021042
4: -0.0105781, -0.0077133, -0.0105994, -0.0077184, -0.0018476, 0.0018947
5: 0.0097315, 0.0108166, 0.0097234, 0.0108147, -0.0006998, 0.0007177
6: 0.0057159, 0.0098568, 0.0057233, 0.0098875, -0.0027387, 0.0026705
7: 0.9820590, 0.9849566, 0.9820641, 0.9849781, -0.0019164, 0.0018687
8: -0.0057999, -0.0026932, -0.0057943, -0.0026701, -0.0020547, 0.0020036
9: -0.0032206, -0.0011685, -0.0032358, -0.0011721, -0.0013235, 0.0013572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028961, -0.0017451, -0.0028366, -0.0017204, -0.0007874, 0.0007025
1: -0.0116598, -0.0087389, -0.0115088, -0.0086764, -0.0019981, 0.0017827
2: 0.0277962, 0.0296083, 0.0278899, 0.0296471, -0.0012396, 0.0011060
3: 0.0042762, 0.0076599, 0.0042038, 0.0074849, -0.0020652, 0.0023147
4: -0.0107530, -0.0077820, -0.0105994, -0.0077184, -0.0020324, 0.0018133
5: 0.0096652, 0.0107906, 0.0097234, 0.0108147, -0.0007698, 0.0006868
6: 0.0058152, 0.0101095, 0.0057233, 0.0098875, -0.0026210, 0.0029377
7: 0.9821285, 0.9851334, 0.9820641, 0.9849781, -0.0018340, 0.0020557
8: -0.0057253, -0.0025036, -0.0057943, -0.0026701, -0.0019664, 0.0022040
9: -0.0033459, -0.0012177, -0.0032358, -0.0011721, -0.0014559, 0.0012989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017522, -0.0029051, -0.0017482, -0.0007179, 0.0007766
1: -0.0114924, -0.0087570, -0.0116827, -0.0087469, -0.0018217, 0.0019708
2: 0.0279001, 0.0295972, 0.0277820, 0.0296034, -0.0011302, 0.0012227
3: 0.0042971, 0.0074660, 0.0042855, 0.0076864, -0.0022831, 0.0021103
4: -0.0105827, -0.0078003, -0.0107763, -0.0077901, -0.0018530, 0.0020046
5: 0.0097297, 0.0107836, 0.0096564, 0.0107875, -0.0007019, 0.0007593
6: 0.0058417, 0.0098634, 0.0058269, 0.0101432, -0.0028975, 0.0026783
7: 0.9821470, 0.9849613, 0.9821366, 0.9851570, -0.0020276, 0.0018741
8: -0.0057055, -0.0026882, -0.0057165, -0.0024783, -0.0021739, 0.0020094
9: -0.0032239, -0.0012308, -0.0033625, -0.0012235, -0.0013273, 0.0014360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011369
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017245, -0.0029051, -0.0017482, -0.0007081, 0.0007926
1: -0.0114858, -0.0086867, -0.0116827, -0.0087469, -0.0017969, 0.0020112
2: 0.0279042, 0.0296408, 0.0277820, 0.0296034, -0.0011148, 0.0012478
3: 0.0042157, 0.0074583, 0.0042855, 0.0076864, -0.0023299, 0.0020816
4: -0.0105760, -0.0077288, -0.0107763, -0.0077901, -0.0018277, 0.0020458
5: 0.0097323, 0.0108107, 0.0096564, 0.0107875, -0.0006923, 0.0007749
6: 0.0057383, 0.0098537, 0.0058269, 0.0101432, -0.0029570, 0.0026418
7: 0.9820747, 0.9849545, 0.9821366, 0.9851570, -0.0020692, 0.0018486
8: -0.0057830, -0.0026955, -0.0057165, -0.0024783, -0.0022185, 0.0019820
9: -0.0032191, -0.0011796, -0.0033625, -0.0012235, -0.0013092, 0.0014654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011369
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028310, -0.0017377, -0.0028333, -0.0016936, -0.0007336, 0.0007117
1: -0.0114946, -0.0087203, -0.0115005, -0.0086082, -0.0018616, 0.0018061
2: 0.0278987, 0.0296199, 0.0278951, 0.0296894, -0.0011549, 0.0011205
3: 0.0042546, 0.0074685, 0.0041248, 0.0074754, -0.0020923, 0.0021565
4: -0.0105849, -0.0077630, -0.0105910, -0.0076490, -0.0018935, 0.0018371
5: 0.0097289, 0.0107978, 0.0097266, 0.0108409, -0.0007172, 0.0006959
6: 0.0057878, 0.0098666, 0.0056230, 0.0098753, -0.0026554, 0.0027369
7: 0.9821093, 0.9849634, 0.9819940, 0.9849695, -0.0018581, 0.0019152
8: -0.0057459, -0.0026858, -0.0058695, -0.0026793, -0.0019922, 0.0020534
9: -0.0032255, -0.0012041, -0.0032298, -0.0011225, -0.0013564, 0.0013160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028284, -0.0017098, -0.0028333, -0.0016936, -0.0007420, 0.0007439
1: -0.0114880, -0.0086495, -0.0115005, -0.0086082, -0.0018830, 0.0018878
2: 0.0279028, 0.0296638, 0.0278951, 0.0296894, -0.0011682, 0.0011712
3: 0.0041726, 0.0074609, 0.0041248, 0.0074754, -0.0021870, 0.0021813
4: -0.0105783, -0.0076910, -0.0105910, -0.0076490, -0.0019153, 0.0019202
5: 0.0097314, 0.0108250, 0.0097266, 0.0108409, -0.0007255, 0.0007273
6: 0.0056837, 0.0098570, 0.0056230, 0.0098753, -0.0027755, 0.0027684
7: 0.9820364, 0.9849566, 0.9819940, 0.9849695, -0.0019422, 0.0019372
8: -0.0058240, -0.0026930, -0.0058695, -0.0026793, -0.0020823, 0.0020769
9: -0.0032207, -0.0011525, -0.0032298, -0.0011225, -0.0013719, 0.0013755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011394
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011393
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017522, -0.0029021, -0.0017203, -0.0007356, 0.0007689
1: -0.0114924, -0.0087570, -0.0116750, -0.0086760, -0.0018667, 0.0019512
2: 0.0279001, 0.0295972, 0.0277868, 0.0296474, -0.0011581, 0.0012105
3: 0.0042971, 0.0074660, 0.0042034, 0.0076775, -0.0022603, 0.0021625
4: -0.0105827, -0.0078003, -0.0107684, -0.0077180, -0.0018987, 0.0019847
5: 0.0097297, 0.0107836, 0.0096594, 0.0108148, -0.0007192, 0.0007517
6: 0.0058417, 0.0098634, 0.0057227, 0.0101318, -0.0028686, 0.0027445
7: 0.9821470, 0.9849613, 0.9820638, 0.9851490, -0.0020073, 0.0019204
8: -0.0057055, -0.0026882, -0.0057947, -0.0024868, -0.0021522, 0.0020590
9: -0.0032239, -0.0012308, -0.0033569, -0.0011719, -0.0013601, 0.0014216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011518
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011571
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017245, -0.0029021, -0.0017203, -0.0007424, 0.0008024
1: -0.0114858, -0.0086867, -0.0116750, -0.0086760, -0.0018839, 0.0020362
2: 0.0279042, 0.0296408, 0.0277868, 0.0296474, -0.0011688, 0.0012632
3: 0.0042157, 0.0074583, 0.0042034, 0.0076775, -0.0023588, 0.0021824
4: -0.0105760, -0.0077288, -0.0107684, -0.0077180, -0.0019162, 0.0020711
5: 0.0097323, 0.0108107, 0.0096594, 0.0108148, -0.0007258, 0.0007845
6: 0.0057383, 0.0098537, 0.0057227, 0.0101318, -0.0029936, 0.0027698
7: 0.9820747, 0.9849545, 0.9820638, 0.9851490, -0.0020948, 0.0019381
8: -0.0057830, -0.0026955, -0.0057947, -0.0024868, -0.0022460, 0.0020780
9: -0.0032191, -0.0011796, -0.0033569, -0.0011719, -0.0013726, 0.0014836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011469
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011525
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028339, -0.0016942, -0.0028374, -0.0017117, -0.0007405, 0.0007434
1: -0.0115021, -0.0086098, -0.0115109, -0.0086542, -0.0018792, 0.0018866
2: 0.0278941, 0.0296885, 0.0278886, 0.0296609, -0.0011659, 0.0011704
3: 0.0041266, 0.0074772, 0.0041781, 0.0074874, -0.0021855, 0.0021770
4: -0.0105926, -0.0076507, -0.0106015, -0.0076958, -0.0019115, 0.0019190
5: 0.0097260, 0.0108403, 0.0097226, 0.0108232, -0.0007240, 0.0007269
6: 0.0056254, 0.0098777, 0.0056907, 0.0098905, -0.0027737, 0.0027628
7: 0.9819956, 0.9849712, 0.9820413, 0.9849802, -0.0019409, 0.0019333
8: -0.0058678, -0.0026775, -0.0058188, -0.0026678, -0.0020810, 0.0020728
9: -0.0032310, -0.0011236, -0.0032373, -0.0011560, -0.0013692, 0.0013746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011251
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011303
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029027, -0.0017214, -0.0028374, -0.0017117, -0.0008118, 0.0007144
1: -0.0116765, -0.0086788, -0.0115109, -0.0086542, -0.0020600, 0.0018129
2: 0.0277859, 0.0296456, 0.0278886, 0.0296609, -0.0012780, 0.0011247
3: 0.0042066, 0.0076793, 0.0041781, 0.0074874, -0.0021002, 0.0023864
4: -0.0107700, -0.0077208, -0.0106015, -0.0076958, -0.0020953, 0.0018440
5: 0.0096588, 0.0108137, 0.0097226, 0.0108232, -0.0007937, 0.0006985
6: 0.0057268, 0.0101341, 0.0056907, 0.0098905, -0.0026654, 0.0030286
7: 0.9820666, 0.9851506, 0.9820413, 0.9849802, -0.0018651, 0.0021193
8: -0.0057917, -0.0024851, -0.0058188, -0.0026678, -0.0019997, 0.0022722
9: -0.0033580, -0.0011739, -0.0032373, -0.0011560, -0.0015009, 0.0013209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011251
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011303
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028338, -0.0016918, -0.0028975, -0.0017817, -0.0006939, 0.0008181
1: -0.0115018, -0.0086038, -0.0116633, -0.0088320, -0.0017608, 0.0020761
2: 0.0278943, 0.0296922, 0.0277940, 0.0295506, -0.0010924, 0.0012880
3: 0.0041196, 0.0074768, 0.0043840, 0.0076640, -0.0024050, 0.0020398
4: -0.0105923, -0.0076445, -0.0107566, -0.0078766, -0.0017910, 0.0021117
5: 0.0097261, 0.0108426, 0.0096639, 0.0107547, -0.0006784, 0.0007999
6: 0.0056165, 0.0098772, 0.0059520, 0.0101147, -0.0030523, 0.0025887
7: 0.9819894, 0.9849708, 0.9822242, 0.9851370, -0.0021359, 0.0018115
8: -0.0058744, -0.0026779, -0.0056227, -0.0024996, -0.0022900, 0.0019422
9: -0.0032307, -0.0011192, -0.0033484, -0.0012855, -0.0012829, 0.0015127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011364
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011364
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028338, -0.0016918, -0.0029049, -0.0017575, -0.0007069, 0.0008127
1: -0.0115018, -0.0086038, -0.0116822, -0.0087706, -0.0017938, 0.0020622
2: 0.0278943, 0.0296922, 0.0277823, 0.0295887, -0.0011129, 0.0012794
3: 0.0041196, 0.0074768, 0.0043129, 0.0076859, -0.0023890, 0.0020781
4: -0.0105923, -0.0076445, -0.0107758, -0.0078142, -0.0018246, 0.0020976
5: 0.0097261, 0.0108426, 0.0096566, 0.0107784, -0.0006911, 0.0007945
6: 0.0056165, 0.0098772, 0.0058617, 0.0101425, -0.0030319, 0.0026373
7: 0.9819894, 0.9849708, 0.9821610, 0.9851565, -0.0021216, 0.0018455
8: -0.0058744, -0.0026779, -0.0056904, -0.0024788, -0.0022747, 0.0019786
9: -0.0032307, -0.0011192, -0.0033622, -0.0012407, -0.0013070, 0.0015026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011416
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011416
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028372, -0.0017135, -0.0028341, -0.0016852, -0.0007588, 0.0007100
1: -0.0115103, -0.0086587, -0.0115025, -0.0085869, -0.0019255, 0.0018017
2: 0.0278890, 0.0296581, 0.0278938, 0.0297027, -0.0011946, 0.0011178
3: 0.0041833, 0.0074867, 0.0041001, 0.0074777, -0.0020872, 0.0022306
4: -0.0106009, -0.0077004, -0.0105930, -0.0076273, -0.0019585, 0.0018327
5: 0.0097228, 0.0108215, 0.0097258, 0.0108491, -0.0007418, 0.0006942
6: 0.0056973, 0.0098897, 0.0055917, 0.0098783, -0.0026490, 0.0028309
7: 0.9820459, 0.9849796, 0.9819721, 0.9849716, -0.0018536, 0.0019809
8: -0.0058138, -0.0026685, -0.0058930, -0.0026770, -0.0019874, 0.0021239
9: -0.0032369, -0.0011593, -0.0032313, -0.0011069, -0.0014029, 0.0013128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011479
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011479
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028339, -0.0016868, -0.0028341, -0.0016852, -0.0007640, 0.0007448
1: -0.0115021, -0.0085911, -0.0115025, -0.0085869, -0.0019388, 0.0018900
2: 0.0278941, 0.0297000, 0.0278938, 0.0297027, -0.0012028, 0.0011726
3: 0.0041050, 0.0074772, 0.0041001, 0.0074777, -0.0021895, 0.0022460
4: -0.0105926, -0.0076317, -0.0105930, -0.0076273, -0.0019720, 0.0019225
5: 0.0097260, 0.0108475, 0.0097258, 0.0108491, -0.0007470, 0.0007282
6: 0.0055979, 0.0098777, 0.0055917, 0.0098783, -0.0027788, 0.0028504
7: 0.9819764, 0.9849712, 0.9819721, 0.9849716, -0.0019444, 0.0019946
8: -0.0058884, -0.0026775, -0.0058930, -0.0026770, -0.0020848, 0.0021385
9: -0.0032310, -0.0011100, -0.0032313, -0.0011069, -0.0014126, 0.0013771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011434
time: 0.64 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011434
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028363, -0.0017275, -0.0029029, -0.0017114, -0.0007603, 0.0007770
1: -0.0115081, -0.0086944, -0.0116771, -0.0086535, -0.0019294, 0.0019717
2: 0.0278904, 0.0296360, 0.0277855, 0.0296614, -0.0011970, 0.0012232
3: 0.0042247, 0.0074842, 0.0041772, 0.0076799, -0.0022841, 0.0022352
4: -0.0105987, -0.0077367, -0.0107706, -0.0076950, -0.0019626, 0.0020055
5: 0.0097237, 0.0108077, 0.0096586, 0.0108235, -0.0007434, 0.0007596
6: 0.0057498, 0.0098865, 0.0056895, 0.0101349, -0.0028988, 0.0028367
7: 0.9820827, 0.9849773, 0.9820405, 0.9851512, -0.0020284, 0.0019850
8: -0.0057744, -0.0026709, -0.0058196, -0.0024845, -0.0021748, 0.0021282
9: -0.0032353, -0.0011853, -0.0033585, -0.0011554, -0.0014058, 0.0014366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011518
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011579
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028331, -0.0017012, -0.0029029, -0.0017114, -0.0007648, 0.0008178
1: -0.0114999, -0.0086276, -0.0116771, -0.0086535, -0.0019408, 0.0020753
2: 0.0278954, 0.0296774, 0.0277855, 0.0296614, -0.0012041, 0.0012875
3: 0.0041472, 0.0074746, 0.0041772, 0.0076799, -0.0024042, 0.0022484
4: -0.0105903, -0.0076687, -0.0107706, -0.0076950, -0.0019742, 0.0021110
5: 0.0097268, 0.0108335, 0.0096586, 0.0108235, -0.0007478, 0.0007996
6: 0.0056515, 0.0098744, 0.0056895, 0.0101349, -0.0030512, 0.0028535
7: 0.9820139, 0.9849689, 0.9820405, 0.9851512, -0.0021351, 0.0019967
8: -0.0058481, -0.0026799, -0.0058196, -0.0024845, -0.0022892, 0.0021408
9: -0.0032294, -0.0011366, -0.0033585, -0.0011554, -0.0014141, 0.0015121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011463
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011509
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.02 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011287, upper bound: 0.0011297
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011369
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011369
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011423
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011394
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011393
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011518
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011571
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011469
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011525
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011251
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011303
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011251
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011483, upper bound: 0.0011303
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011364
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011364
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011416
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011416
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011479
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011479
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011434
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011434
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011518
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011579
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011463
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 7, lower bound: -0.0011423, upper bound: 0.0011509

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028366, -0.0017204, -0.0007175, 0.0007052
1: -0.0114927, -0.0087610, -0.0115088, -0.0086764, -0.0018208, 0.0017895
2: 0.0278999, 0.0295947, 0.0278899, 0.0296471, -0.0011297, 0.0011102
3: 0.0043017, 0.0074664, 0.0042038, 0.0074849, -0.0020731, 0.0021094
4: -0.0105831, -0.0078044, -0.0105994, -0.0077184, -0.0018521, 0.0018202
5: 0.0097296, 0.0107821, 0.0097234, 0.0108147, -0.0007015, 0.0006895
6: 0.0058476, 0.0098639, 0.0057233, 0.0098875, -0.0026310, 0.0026770
7: 0.9821512, 0.9849616, 0.9820641, 0.9849781, -0.0018410, 0.0018733
8: -0.0057010, -0.0026878, -0.0057943, -0.0026701, -0.0019739, 0.0020084
9: -0.0032241, -0.0012337, -0.0032358, -0.0011721, -0.0013267, 0.0013039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011505
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011557
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028366, -0.0017204, -0.0007078, 0.0007199
1: -0.0114861, -0.0086910, -0.0115088, -0.0086764, -0.0017961, 0.0018268
2: 0.0279040, 0.0296381, 0.0278899, 0.0296471, -0.0011143, 0.0011333
3: 0.0042206, 0.0074587, 0.0042038, 0.0074849, -0.0021162, 0.0020807
4: -0.0105763, -0.0077332, -0.0105994, -0.0077184, -0.0018269, 0.0018581
5: 0.0097322, 0.0108091, 0.0097234, 0.0108147, -0.0006920, 0.0007038
6: 0.0057447, 0.0098541, 0.0057233, 0.0098875, -0.0026858, 0.0026407
7: 0.9820791, 0.9849547, 0.9820641, 0.9849781, -0.0018794, 0.0018478
8: -0.0057783, -0.0026951, -0.0057943, -0.0026701, -0.0020150, 0.0019811
9: -0.0032193, -0.0011827, -0.0032358, -0.0011721, -0.0013087, 0.0013310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011505
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011557
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028366, -0.0017204, -0.0007916, 0.0006735
1: -0.0116633, -0.0088327, -0.0115088, -0.0086764, -0.0020088, 0.0017091
2: 0.0277940, 0.0295502, 0.0278899, 0.0296471, -0.0012462, 0.0010603
3: 0.0043848, 0.0076640, 0.0042038, 0.0074849, -0.0019799, 0.0023271
4: -0.0107566, -0.0078773, -0.0105994, -0.0077184, -0.0020432, 0.0017384
5: 0.0096639, 0.0107545, 0.0097234, 0.0108147, -0.0007739, 0.0006585
6: 0.0059530, 0.0101147, 0.0057233, 0.0098875, -0.0025128, 0.0029533
7: 0.9822249, 0.9851370, 0.9820641, 0.9849781, -0.0017583, 0.0020666
8: -0.0056220, -0.0024996, -0.0057943, -0.0026701, -0.0018852, 0.0022157
9: -0.0033484, -0.0012860, -0.0032358, -0.0011721, -0.0014636, 0.0012453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011253
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011297
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028366, -0.0017204, -0.0007793, 0.0006913
1: -0.0116579, -0.0087618, -0.0115088, -0.0086764, -0.0019776, 0.0017543
2: 0.0277974, 0.0295942, 0.0278899, 0.0296471, -0.0012269, 0.0010884
3: 0.0043027, 0.0076577, 0.0042038, 0.0074849, -0.0020323, 0.0022909
4: -0.0107511, -0.0078052, -0.0105994, -0.0077184, -0.0020115, 0.0017844
5: 0.0096660, 0.0107818, 0.0097234, 0.0108147, -0.0007619, 0.0006759
6: 0.0058487, 0.0101068, 0.0057233, 0.0098875, -0.0025792, 0.0029075
7: 0.9821519, 0.9851315, 0.9820641, 0.9849781, -0.0018048, 0.0020345
8: -0.0057002, -0.0025056, -0.0057943, -0.0026701, -0.0019350, 0.0021813
9: -0.0033445, -0.0012343, -0.0032358, -0.0011721, -0.0014409, 0.0012782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011253
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011297
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017522, -0.0028975, -0.0017817, -0.0006834, 0.0007565
1: -0.0114924, -0.0087570, -0.0116633, -0.0088320, -0.0017343, 0.0019197
2: 0.0279001, 0.0295972, 0.0277940, 0.0295506, -0.0010760, 0.0011910
3: 0.0042971, 0.0074660, 0.0043840, 0.0076640, -0.0022239, 0.0020091
4: -0.0105827, -0.0078003, -0.0107566, -0.0078766, -0.0017641, 0.0019526
5: 0.0097297, 0.0107836, 0.0096639, 0.0107547, -0.0006682, 0.0007396
6: 0.0058417, 0.0098634, 0.0059520, 0.0101147, -0.0028224, 0.0025498
7: 0.9821470, 0.9849613, 0.9822242, 0.9851370, -0.0019750, 0.0017842
8: -0.0057055, -0.0026882, -0.0056227, -0.0024996, -0.0021175, 0.0019130
9: -0.0032239, -0.0012308, -0.0033484, -0.0012855, -0.0012636, 0.0013987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011369
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011256
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017522, -0.0029049, -0.0017575, -0.0007145, 0.0007640
1: -0.0114924, -0.0087570, -0.0116822, -0.0087706, -0.0018133, 0.0019386
2: 0.0279001, 0.0295972, 0.0277823, 0.0295887, -0.0011249, 0.0012027
3: 0.0042971, 0.0074660, 0.0043129, 0.0076859, -0.0022458, 0.0021006
4: -0.0105827, -0.0078003, -0.0107758, -0.0078142, -0.0018444, 0.0019719
5: 0.0097297, 0.0107836, 0.0096566, 0.0107784, -0.0006986, 0.0007469
6: 0.0058417, 0.0098634, 0.0058617, 0.0101425, -0.0028502, 0.0026659
7: 0.9821470, 0.9849613, 0.9821610, 0.9851565, -0.0019945, 0.0018655
8: -0.0057055, -0.0026882, -0.0056904, -0.0024788, -0.0021384, 0.0020001
9: -0.0032239, -0.0012308, -0.0033622, -0.0012407, -0.0013212, 0.0014125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011423
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011300
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017245, -0.0028975, -0.0017817, -0.0006737, 0.0007724
1: -0.0114858, -0.0086867, -0.0116633, -0.0088320, -0.0017095, 0.0019601
2: 0.0279042, 0.0296408, 0.0277940, 0.0295506, -0.0010606, 0.0012161
3: 0.0042157, 0.0074583, 0.0043840, 0.0076640, -0.0022707, 0.0019804
4: -0.0105760, -0.0077288, -0.0107566, -0.0078766, -0.0017389, 0.0019938
5: 0.0097323, 0.0108107, 0.0096639, 0.0107547, -0.0006586, 0.0007552
6: 0.0057383, 0.0098537, 0.0059520, 0.0101147, -0.0028818, 0.0025134
7: 0.9820747, 0.9849545, 0.9822242, 0.9851370, -0.0020166, 0.0017587
8: -0.0057830, -0.0026955, -0.0056227, -0.0024996, -0.0021621, 0.0018857
9: -0.0032191, -0.0011796, -0.0033484, -0.0012855, -0.0012456, 0.0014282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011369
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011252
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017245, -0.0029049, -0.0017575, -0.0007048, 0.0007799
1: -0.0114858, -0.0086867, -0.0116822, -0.0087706, -0.0017885, 0.0019791
2: 0.0279042, 0.0296408, 0.0277823, 0.0295887, -0.0011096, 0.0012278
3: 0.0042157, 0.0074583, 0.0043129, 0.0076859, -0.0022927, 0.0020718
4: -0.0105760, -0.0077288, -0.0107758, -0.0078142, -0.0018192, 0.0020131
5: 0.0097323, 0.0108107, 0.0096566, 0.0107784, -0.0006891, 0.0007625
6: 0.0057383, 0.0098537, 0.0058617, 0.0101425, -0.0029097, 0.0026294
7: 0.9820747, 0.9849545, 0.9821610, 0.9851565, -0.0020361, 0.0018400
8: -0.0057830, -0.0026955, -0.0056904, -0.0024788, -0.0021830, 0.0019727
9: -0.0032191, -0.0011796, -0.0033622, -0.0012407, -0.0013031, 0.0014420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011423
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011297
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028333, -0.0016936, -0.0007330, 0.0006977
1: -0.0114927, -0.0087610, -0.0115005, -0.0086082, -0.0018602, 0.0017705
2: 0.0278999, 0.0295947, 0.0278951, 0.0296894, -0.0011541, 0.0010984
3: 0.0043017, 0.0074664, 0.0041248, 0.0074754, -0.0020511, 0.0021550
4: -0.0105831, -0.0078044, -0.0105910, -0.0076490, -0.0018921, 0.0018009
5: 0.0097296, 0.0107821, 0.0097266, 0.0108409, -0.0007167, 0.0006821
6: 0.0058476, 0.0098639, 0.0056230, 0.0098753, -0.0026031, 0.0027349
7: 0.9821512, 0.9849616, 0.9819940, 0.9849695, -0.0018215, 0.0019138
8: -0.0057010, -0.0026878, -0.0058695, -0.0026793, -0.0019529, 0.0020519
9: -0.0032241, -0.0012337, -0.0032298, -0.0011225, -0.0013554, 0.0012900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011401
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028333, -0.0016936, -0.0008071, 0.0006660
1: -0.0116633, -0.0088327, -0.0115005, -0.0086082, -0.0020481, 0.0016901
2: 0.0277940, 0.0295502, 0.0278951, 0.0296894, -0.0012707, 0.0010485
3: 0.0043848, 0.0076640, 0.0041248, 0.0074754, -0.0019579, 0.0023726
4: -0.0107566, -0.0078773, -0.0105910, -0.0076490, -0.0020833, 0.0017191
5: 0.0096639, 0.0107545, 0.0097266, 0.0108409, -0.0007891, 0.0006512
6: 0.0059530, 0.0101147, 0.0056230, 0.0098753, -0.0024848, 0.0030112
7: 0.9822249, 0.9851370, 0.9819940, 0.9849695, -0.0017388, 0.0021071
8: -0.0056220, -0.0024996, -0.0058695, -0.0026793, -0.0018642, 0.0022591
9: -0.0033484, -0.0012860, -0.0032298, -0.0011225, -0.0014923, 0.0012314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011401
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028333, -0.0016936, -0.0007415, 0.0007294
1: -0.0114861, -0.0086910, -0.0115005, -0.0086082, -0.0018816, 0.0018511
2: 0.0279040, 0.0296381, 0.0278951, 0.0296894, -0.0011673, 0.0011484
3: 0.0042206, 0.0074587, 0.0041248, 0.0074754, -0.0021444, 0.0021797
4: -0.0105763, -0.0077332, -0.0105910, -0.0076490, -0.0019139, 0.0018829
5: 0.0097322, 0.0108091, 0.0097266, 0.0108409, -0.0007249, 0.0007132
6: 0.0057447, 0.0098541, 0.0056230, 0.0098753, -0.0027215, 0.0027664
7: 0.9820791, 0.9849547, 0.9819940, 0.9849695, -0.0019044, 0.0019358
8: -0.0057783, -0.0026951, -0.0058695, -0.0026793, -0.0020418, 0.0020755
9: -0.0032193, -0.0011827, -0.0032298, -0.0011225, -0.0013710, 0.0013487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011350
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011393
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028333, -0.0016936, -0.0008171, 0.0006998
1: -0.0116579, -0.0087618, -0.0115005, -0.0086082, -0.0020735, 0.0017758
2: 0.0277974, 0.0295942, 0.0278951, 0.0296894, -0.0012864, 0.0011017
3: 0.0043027, 0.0076577, 0.0041248, 0.0074754, -0.0020572, 0.0024020
4: -0.0107511, -0.0078052, -0.0105910, -0.0076490, -0.0021091, 0.0018063
5: 0.0096660, 0.0107818, 0.0097266, 0.0108409, -0.0007989, 0.0006842
6: 0.0058487, 0.0101068, 0.0056230, 0.0098753, -0.0026109, 0.0030485
7: 0.9821519, 0.9851315, 0.9819940, 0.9849695, -0.0018270, 0.0021332
8: -0.0057002, -0.0025056, -0.0058695, -0.0026793, -0.0019588, 0.0022871
9: -0.0033445, -0.0012343, -0.0032298, -0.0011225, -0.0015108, 0.0012939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011350
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011393
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017522, -0.0028953, -0.0017541, -0.0006997, 0.0007442
1: -0.0114924, -0.0087570, -0.0116579, -0.0087618, -0.0017755, 0.0018885
2: 0.0279001, 0.0295972, 0.0277974, 0.0295942, -0.0011015, 0.0011716
3: 0.0042971, 0.0074660, 0.0043027, 0.0076577, -0.0021878, 0.0020568
4: -0.0105827, -0.0078003, -0.0107511, -0.0078052, -0.0018060, 0.0019209
5: 0.0097297, 0.0107836, 0.0096660, 0.0107818, -0.0006840, 0.0007276
6: 0.0058417, 0.0098634, 0.0058487, 0.0101068, -0.0027765, 0.0026104
7: 0.9821470, 0.9849613, 0.9821519, 0.9851315, -0.0019429, 0.0018266
8: -0.0057055, -0.0026882, -0.0057002, -0.0025056, -0.0020831, 0.0019584
9: -0.0032239, -0.0012308, -0.0033445, -0.0012343, -0.0012936, 0.0013760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011518
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011401
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017522, -0.0029019, -0.0017293, -0.0007361, 0.0007556
1: -0.0114924, -0.0087570, -0.0116746, -0.0086990, -0.0018680, 0.0019175
2: 0.0279001, 0.0295972, 0.0277871, 0.0296331, -0.0011589, 0.0011896
3: 0.0042971, 0.0074660, 0.0042300, 0.0076770, -0.0022213, 0.0021640
4: -0.0105827, -0.0078003, -0.0107680, -0.0077414, -0.0019000, 0.0019504
5: 0.0097297, 0.0107836, 0.0096595, 0.0108060, -0.0007197, 0.0007388
6: 0.0058417, 0.0098634, 0.0057565, 0.0101312, -0.0028192, 0.0027463
7: 0.9821470, 0.9849613, 0.9820874, 0.9851486, -0.0019727, 0.0019218
8: -0.0057055, -0.0026882, -0.0057694, -0.0024872, -0.0021151, 0.0020604
9: -0.0032239, -0.0012308, -0.0033566, -0.0011886, -0.0013610, 0.0013971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011571
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011428
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017245, -0.0028953, -0.0017541, -0.0007080, 0.0007836
1: -0.0114858, -0.0086867, -0.0116579, -0.0087618, -0.0017968, 0.0019885
2: 0.0279042, 0.0296408, 0.0277974, 0.0295942, -0.0011147, 0.0012337
3: 0.0042157, 0.0074583, 0.0043027, 0.0076577, -0.0023036, 0.0020815
4: -0.0105760, -0.0077288, -0.0107511, -0.0078052, -0.0018276, 0.0020226
5: 0.0097323, 0.0108107, 0.0096660, 0.0107818, -0.0006923, 0.0007661
6: 0.0057383, 0.0098537, 0.0058487, 0.0101068, -0.0029235, 0.0026417
7: 0.9820747, 0.9849545, 0.9821519, 0.9851315, -0.0020458, 0.0018485
8: -0.0057830, -0.0026955, -0.0057002, -0.0025056, -0.0021934, 0.0019819
9: -0.0032191, -0.0011796, -0.0033445, -0.0012343, -0.0013092, 0.0014488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011469
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011350
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017245, -0.0029019, -0.0017293, -0.0007390, 0.0007900
1: -0.0114858, -0.0086867, -0.0116746, -0.0086990, -0.0018754, 0.0020047
2: 0.0279042, 0.0296408, 0.0277871, 0.0296331, -0.0011635, 0.0012437
3: 0.0042157, 0.0074583, 0.0042300, 0.0076770, -0.0023224, 0.0021725
4: -0.0105760, -0.0077288, -0.0107680, -0.0077414, -0.0019076, 0.0020391
5: 0.0097323, 0.0108107, 0.0096595, 0.0108060, -0.0007225, 0.0007724
6: 0.0057383, 0.0098537, 0.0057565, 0.0101312, -0.0029474, 0.0027572
7: 0.9820747, 0.9849545, 0.9820874, 0.9851486, -0.0020624, 0.0019294
8: -0.0057830, -0.0026955, -0.0057694, -0.0024872, -0.0022113, 0.0020686
9: -0.0032191, -0.0011796, -0.0033566, -0.0011886, -0.0013664, 0.0014607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011525
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011393
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028339, -0.0016942, -0.0028303, -0.0017537, -0.0006936, 0.0007452
1: -0.0115021, -0.0086098, -0.0114927, -0.0087608, -0.0017600, 0.0018911
2: 0.0278941, 0.0296885, 0.0278999, 0.0295948, -0.0010919, 0.0011732
3: 0.0041266, 0.0074772, 0.0043015, 0.0074664, -0.0021907, 0.0020389
4: -0.0105926, -0.0076507, -0.0105831, -0.0078042, -0.0017902, 0.0019235
5: 0.0097260, 0.0108403, 0.0097296, 0.0107822, -0.0006781, 0.0007286
6: 0.0056254, 0.0098777, 0.0058473, 0.0098639, -0.0027803, 0.0025876
7: 0.9819956, 0.9849712, 0.9821509, 0.9849616, -0.0019455, 0.0018107
8: -0.0058678, -0.0026775, -0.0057013, -0.0026878, -0.0020859, 0.0019413
9: -0.0032310, -0.0011236, -0.0032241, -0.0012336, -0.0012824, 0.0013779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011497
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011497
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028339, -0.0016942, -0.0028364, -0.0017300, -0.0007046, 0.0007363
1: -0.0115021, -0.0086098, -0.0115084, -0.0087008, -0.0017880, 0.0018685
2: 0.0278941, 0.0296885, 0.0278902, 0.0296320, -0.0011093, 0.0011592
3: 0.0041266, 0.0074772, 0.0042321, 0.0074845, -0.0021645, 0.0020713
4: -0.0105926, -0.0076507, -0.0105990, -0.0077432, -0.0018187, 0.0019006
5: 0.0097260, 0.0108403, 0.0097236, 0.0108053, -0.0006889, 0.0007199
6: 0.0056254, 0.0098777, 0.0057591, 0.0098870, -0.0027471, 0.0026288
7: 0.9819956, 0.9849712, 0.9820893, 0.9849777, -0.0019223, 0.0018395
8: -0.0058678, -0.0026775, -0.0057674, -0.0026705, -0.0020610, 0.0019722
9: -0.0032310, -0.0011236, -0.0032356, -0.0011899, -0.0013028, 0.0013614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011537
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011537
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029027, -0.0017214, -0.0028303, -0.0017537, -0.0007648, 0.0007155
1: -0.0116765, -0.0086788, -0.0114927, -0.0087608, -0.0019408, 0.0018156
2: 0.0277859, 0.0296456, 0.0278999, 0.0295948, -0.0012041, 0.0011264
3: 0.0042066, 0.0076793, 0.0043015, 0.0074664, -0.0021033, 0.0022483
4: -0.0107700, -0.0077208, -0.0105831, -0.0078042, -0.0019741, 0.0018467
5: 0.0096588, 0.0108137, 0.0097296, 0.0107822, -0.0007477, 0.0006995
6: 0.0057268, 0.0101341, 0.0058473, 0.0098639, -0.0026693, 0.0028534
7: 0.9820666, 0.9851506, 0.9821509, 0.9849616, -0.0018678, 0.0019966
8: -0.0057917, -0.0024851, -0.0057013, -0.0026878, -0.0020026, 0.0021407
9: -0.0033580, -0.0011739, -0.0032241, -0.0012336, -0.0014141, 0.0013228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011251
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011251
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029027, -0.0017214, -0.0028364, -0.0017300, -0.0007766, 0.0007073
1: -0.0116765, -0.0086788, -0.0115084, -0.0087008, -0.0019707, 0.0017948
2: 0.0277859, 0.0296456, 0.0278902, 0.0296320, -0.0012226, 0.0011135
3: 0.0042066, 0.0076793, 0.0042321, 0.0074845, -0.0020792, 0.0022829
4: -0.0107700, -0.0077208, -0.0105990, -0.0077432, -0.0020045, 0.0018256
5: 0.0096588, 0.0108137, 0.0097236, 0.0108053, -0.0007592, 0.0006915
6: 0.0057268, 0.0101341, 0.0057591, 0.0098870, -0.0026388, 0.0028973
7: 0.9820666, 0.9851506, 0.9820893, 0.9849777, -0.0018465, 0.0020274
8: -0.0057917, -0.0024851, -0.0057674, -0.0026705, -0.0019797, 0.0021737
9: -0.0033580, -0.0011739, -0.0032356, -0.0011899, -0.0014358, 0.0013077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011303
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011303
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028363, -0.0017275, -0.0028975, -0.0017817, -0.0006931, 0.0007876
1: -0.0115081, -0.0086944, -0.0116633, -0.0088320, -0.0017589, 0.0019987
2: 0.0278904, 0.0296360, 0.0277940, 0.0295506, -0.0010912, 0.0012400
3: 0.0042247, 0.0074842, 0.0043840, 0.0076640, -0.0023153, 0.0020376
4: -0.0105987, -0.0077367, -0.0107566, -0.0078766, -0.0017891, 0.0020330
5: 0.0097237, 0.0108077, 0.0096639, 0.0107547, -0.0006777, 0.0007700
6: 0.0057498, 0.0098865, 0.0059520, 0.0101147, -0.0029385, 0.0025860
7: 0.9820827, 0.9849773, 0.9822242, 0.9851370, -0.0020562, 0.0018095
8: -0.0057744, -0.0026709, -0.0056227, -0.0024996, -0.0022046, 0.0019401
9: -0.0032353, -0.0011853, -0.0033484, -0.0012855, -0.0012816, 0.0014562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011364
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011248
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028331, -0.0017012, -0.0028975, -0.0017817, -0.0006846, 0.0008072
1: -0.0114999, -0.0086276, -0.0116633, -0.0088320, -0.0017374, 0.0020484
2: 0.0278954, 0.0296774, 0.0277940, 0.0295506, -0.0010779, 0.0012708
3: 0.0041472, 0.0074746, 0.0043840, 0.0076640, -0.0023729, 0.0020127
4: -0.0105903, -0.0076687, -0.0107566, -0.0078766, -0.0017672, 0.0020835
5: 0.0097268, 0.0108335, 0.0096639, 0.0107547, -0.0006694, 0.0007892
6: 0.0056515, 0.0098744, 0.0059520, 0.0101147, -0.0030115, 0.0025543
7: 0.9820139, 0.9849689, 0.9822242, 0.9851370, -0.0021073, 0.0017874
8: -0.0058481, -0.0026799, -0.0056227, -0.0024996, -0.0022594, 0.0019164
9: -0.0032294, -0.0011366, -0.0033484, -0.0012855, -0.0012659, 0.0014925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011364
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011248
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028363, -0.0017275, -0.0029049, -0.0017575, -0.0007105, 0.0007846
1: -0.0115081, -0.0086944, -0.0116822, -0.0087706, -0.0018030, 0.0019910
2: 0.0278904, 0.0296360, 0.0277823, 0.0295887, -0.0011186, 0.0012352
3: 0.0042247, 0.0074842, 0.0043129, 0.0076859, -0.0023065, 0.0020887
4: -0.0105987, -0.0077367, -0.0107758, -0.0078142, -0.0018340, 0.0020252
5: 0.0097237, 0.0108077, 0.0096566, 0.0107784, -0.0006947, 0.0007671
6: 0.0057498, 0.0098865, 0.0058617, 0.0101425, -0.0029272, 0.0026508
7: 0.9820827, 0.9849773, 0.9821610, 0.9851565, -0.0020483, 0.0018549
8: -0.0057744, -0.0026709, -0.0056904, -0.0024788, -0.0021961, 0.0019888
9: -0.0032353, -0.0011853, -0.0033622, -0.0012407, -0.0013137, 0.0014507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011416
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011301
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028331, -0.0017012, -0.0029049, -0.0017575, -0.0006988, 0.0007985
1: -0.0114999, -0.0086276, -0.0116822, -0.0087706, -0.0017732, 0.0020263
2: 0.0278954, 0.0296774, 0.0277823, 0.0295887, -0.0011001, 0.0012571
3: 0.0041472, 0.0074746, 0.0043129, 0.0076859, -0.0023473, 0.0020542
4: -0.0105903, -0.0076687, -0.0107758, -0.0078142, -0.0018036, 0.0020610
5: 0.0097268, 0.0108335, 0.0096566, 0.0107784, -0.0006832, 0.0007807
6: 0.0056515, 0.0098744, 0.0058617, 0.0101425, -0.0029790, 0.0026070
7: 0.9820139, 0.9849689, 0.9821610, 0.9851565, -0.0020846, 0.0018242
8: -0.0058481, -0.0026799, -0.0056904, -0.0024788, -0.0022350, 0.0019559
9: -0.0032294, -0.0011366, -0.0033622, -0.0012407, -0.0012920, 0.0014764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011416
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011301
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028341, -0.0016852, -0.0007582, 0.0006952
1: -0.0115084, -0.0087008, -0.0115025, -0.0085869, -0.0019241, 0.0017641
2: 0.0278902, 0.0296320, 0.0278938, 0.0297027, -0.0011937, 0.0010944
3: 0.0042321, 0.0074845, 0.0041001, 0.0074777, -0.0020436, 0.0022290
4: -0.0105990, -0.0077432, -0.0105930, -0.0076273, -0.0019572, 0.0017943
5: 0.0097236, 0.0108053, 0.0097258, 0.0108491, -0.0007413, 0.0006796
6: 0.0057591, 0.0098870, 0.0055917, 0.0098783, -0.0025936, 0.0028289
7: 0.9820893, 0.9849777, 0.9819721, 0.9849716, -0.0018149, 0.0019795
8: -0.0057674, -0.0026705, -0.0058930, -0.0026770, -0.0019458, 0.0021224
9: -0.0032356, -0.0011899, -0.0032313, -0.0011069, -0.0014019, 0.0012853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011401
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011459
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028341, -0.0016852, -0.0008301, 0.0006658
1: -0.0116822, -0.0087706, -0.0115025, -0.0085869, -0.0021064, 0.0016897
2: 0.0277823, 0.0295887, 0.0278938, 0.0297027, -0.0013068, 0.0010483
3: 0.0043129, 0.0076859, 0.0041001, 0.0074777, -0.0019574, 0.0024402
4: -0.0107758, -0.0078142, -0.0105930, -0.0076273, -0.0021426, 0.0017187
5: 0.0096566, 0.0107784, 0.0097258, 0.0108491, -0.0008115, 0.0006510
6: 0.0058617, 0.0101425, 0.0055917, 0.0098783, -0.0024842, 0.0030969
7: 0.9821610, 0.9851565, 0.9819721, 0.9849716, -0.0017383, 0.0021671
8: -0.0056904, -0.0024788, -0.0058930, -0.0026770, -0.0018638, 0.0023234
9: -0.0033622, -0.0012407, -0.0032313, -0.0011069, -0.0015348, 0.0012311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011401
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011459
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028341, -0.0016852, -0.0007635, 0.0007298
1: -0.0115002, -0.0086332, -0.0115025, -0.0085869, -0.0019374, 0.0018519
2: 0.0278952, 0.0296739, 0.0278938, 0.0297027, -0.0012020, 0.0011489
3: 0.0041538, 0.0074750, 0.0041001, 0.0074777, -0.0021453, 0.0022444
4: -0.0105907, -0.0076745, -0.0105930, -0.0076273, -0.0019707, 0.0018837
5: 0.0097267, 0.0108313, 0.0097258, 0.0108491, -0.0007464, 0.0007135
6: 0.0056598, 0.0098749, 0.0055917, 0.0098783, -0.0027226, 0.0028484
7: 0.9820197, 0.9849693, 0.9819721, 0.9849716, -0.0019052, 0.0019932
8: -0.0058419, -0.0026796, -0.0058930, -0.0026770, -0.0020427, 0.0021370
9: -0.0032296, -0.0011407, -0.0032313, -0.0011069, -0.0014116, 0.0013493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011347
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011391
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028341, -0.0016852, -0.0008359, 0.0007013
1: -0.0116746, -0.0086990, -0.0115025, -0.0085869, -0.0021211, 0.0017795
2: 0.0277871, 0.0296331, 0.0278938, 0.0297027, -0.0013160, 0.0011040
3: 0.0042300, 0.0076770, 0.0041001, 0.0074777, -0.0020615, 0.0024572
4: -0.0107680, -0.0077414, -0.0105930, -0.0076273, -0.0021575, 0.0018101
5: 0.0096595, 0.0108060, 0.0097258, 0.0108491, -0.0008172, 0.0006856
6: 0.0057565, 0.0101312, 0.0055917, 0.0098783, -0.0026163, 0.0031185
7: 0.9820874, 0.9851486, 0.9819721, 0.9849716, -0.0018308, 0.0021822
8: -0.0057694, -0.0024872, -0.0058930, -0.0026770, -0.0019629, 0.0023397
9: -0.0033566, -0.0011886, -0.0032313, -0.0011069, -0.0015455, 0.0012966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011347
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011391
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028363, -0.0017275, -0.0028953, -0.0017541, -0.0007093, 0.0007753
1: -0.0115081, -0.0086944, -0.0116579, -0.0087618, -0.0018001, 0.0019675
2: 0.0278904, 0.0296360, 0.0277974, 0.0295942, -0.0011168, 0.0012206
3: 0.0042247, 0.0074842, 0.0043027, 0.0076577, -0.0022792, 0.0020853
4: -0.0105987, -0.0077367, -0.0107511, -0.0078052, -0.0018310, 0.0020013
5: 0.0097237, 0.0108077, 0.0096660, 0.0107818, -0.0006935, 0.0007580
6: 0.0057498, 0.0098865, 0.0058487, 0.0101068, -0.0028926, 0.0026465
7: 0.9820827, 0.9849773, 0.9821519, 0.9851315, -0.0020241, 0.0018519
8: -0.0057744, -0.0026709, -0.0057002, -0.0025056, -0.0021702, 0.0019855
9: -0.0032353, -0.0011853, -0.0033445, -0.0012343, -0.0013115, 0.0014335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011518
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011401
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028363, -0.0017275, -0.0029019, -0.0017293, -0.0007262, 0.0007697
1: -0.0115081, -0.0086944, -0.0116746, -0.0086990, -0.0018428, 0.0019533
2: 0.0278904, 0.0296360, 0.0277871, 0.0296331, -0.0011433, 0.0012118
3: 0.0042247, 0.0074842, 0.0042300, 0.0076770, -0.0022628, 0.0021348
4: -0.0105987, -0.0077367, -0.0107680, -0.0077414, -0.0018744, 0.0019868
5: 0.0097237, 0.0108077, 0.0096595, 0.0108060, -0.0007100, 0.0007526
6: 0.0057498, 0.0098865, 0.0057565, 0.0101312, -0.0028718, 0.0027093
7: 0.9820827, 0.9849773, 0.9820874, 0.9851486, -0.0020096, 0.0018958
8: -0.0057744, -0.0026709, -0.0057694, -0.0024872, -0.0021546, 0.0020326
9: -0.0032353, -0.0011853, -0.0033566, -0.0011886, -0.0013427, 0.0014232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011579
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011459
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028331, -0.0017012, -0.0028953, -0.0017541, -0.0007177, 0.0008137
1: -0.0114999, -0.0086276, -0.0116579, -0.0087618, -0.0018212, 0.0020649
2: 0.0278954, 0.0296774, 0.0277974, 0.0295942, -0.0011299, 0.0012811
3: 0.0041472, 0.0074746, 0.0043027, 0.0076577, -0.0023921, 0.0021098
4: -0.0105903, -0.0076687, -0.0107511, -0.0078052, -0.0018524, 0.0021004
5: 0.0097268, 0.0108335, 0.0096660, 0.0107818, -0.0007017, 0.0007956
6: 0.0056515, 0.0098744, 0.0058487, 0.0101068, -0.0030359, 0.0026775
7: 0.9820139, 0.9849689, 0.9821519, 0.9851315, -0.0021244, 0.0018736
8: -0.0058481, -0.0026799, -0.0057002, -0.0025056, -0.0022777, 0.0020088
9: -0.0032294, -0.0011366, -0.0033445, -0.0012343, -0.0013269, 0.0015045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011463
time: 0.64 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011346
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028331, -0.0017012, -0.0029019, -0.0017293, -0.0007361, 0.0008114
1: -0.0114999, -0.0086276, -0.0116746, -0.0086990, -0.0018681, 0.0020591
2: 0.0278954, 0.0296774, 0.0277871, 0.0296331, -0.0011590, 0.0012775
3: 0.0041472, 0.0074746, 0.0042300, 0.0076770, -0.0023854, 0.0021641
4: -0.0105903, -0.0076687, -0.0107680, -0.0077414, -0.0019002, 0.0020945
5: 0.0097268, 0.0108335, 0.0096595, 0.0108060, -0.0007197, 0.0007933
6: 0.0056515, 0.0098744, 0.0057565, 0.0101312, -0.0030274, 0.0027465
7: 0.9820139, 0.9849689, 0.9820874, 0.9851486, -0.0021184, 0.0019219
8: -0.0058481, -0.0026799, -0.0057694, -0.0024872, -0.0022713, 0.0020605
9: -0.0032294, -0.0011366, -0.0033566, -0.0011886, -0.0013611, 0.0015003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011509
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011389
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.97 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011505
IS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011557
IS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011505
IS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011503, upper bound: 0.0011557
IS_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011253
IS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011297
IS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011253
IS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011364, upper bound: 0.0011297
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011369
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011256
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011423
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011300
IS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011369
IS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011252
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011423
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011408, upper bound: 0.0011297
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011401
IS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
IS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011401
IS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011428
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011350
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011393
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011350
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011511, upper bound: 0.0011393
IS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011518
IS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011401
IS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011571
IS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011248, upper bound: 0.0011428
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011469
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011350
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011525
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011477, upper bound: 0.0011393
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011497
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011497
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011537
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011564, upper bound: 0.0011537
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011251
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011251
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011303
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011425, upper bound: 0.0011303
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011364
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011248
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011364
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011248
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011416
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011301
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011416
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011347, upper bound: 0.0011301
IS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011401
IS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011459
IS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011401
IS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011355, upper bound: 0.0011459
IS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011347
IS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011391
IS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011347
IS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011555, upper bound: 0.0011391
IS_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011518
IS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011401
IS_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011579
IS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011459
IS_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011463
IS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011346
IS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011509
IS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 7, lower bound: -0.0011492, upper bound: 0.0011389

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028303, -0.0017537, -0.0006831, 0.0006830
1: -0.0114927, -0.0087610, -0.0114927, -0.0087608, -0.0017335, 0.0017333
2: 0.0278999, 0.0295947, 0.0278999, 0.0295948, -0.0010755, 0.0010753
3: 0.0043017, 0.0074664, 0.0043015, 0.0074664, -0.0020079, 0.0020082
4: -0.0105831, -0.0078044, -0.0105831, -0.0078042, -0.0017633, 0.0017630
5: 0.0097296, 0.0107821, 0.0097296, 0.0107822, -0.0006679, 0.0006678
6: 0.0058476, 0.0098639, 0.0058473, 0.0098639, -0.0025483, 0.0025486
7: 0.9821512, 0.9849616, 0.9821509, 0.9849616, -0.0017832, 0.0017834
8: -0.0057010, -0.0026878, -0.0057013, -0.0026878, -0.0019118, 0.0019121
9: -0.0032241, -0.0012337, -0.0032241, -0.0012336, -0.0012631, 0.0012629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011354, upper bound: 0.0011256
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011307, upper bound: 0.0011314
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028364, -0.0017300, -0.0007138, 0.0006927
1: -0.0114927, -0.0087610, -0.0115084, -0.0087008, -0.0018113, 0.0017578
2: 0.0278999, 0.0295947, 0.0278902, 0.0296320, -0.0011237, 0.0010906
3: 0.0043017, 0.0074664, 0.0042321, 0.0074845, -0.0020364, 0.0020983
4: -0.0105831, -0.0078044, -0.0105990, -0.0077432, -0.0018424, 0.0017880
5: 0.0097296, 0.0107821, 0.0097236, 0.0108053, -0.0006978, 0.0006773
6: 0.0058476, 0.0098639, 0.0057591, 0.0098870, -0.0025844, 0.0026630
7: 0.9821512, 0.9849616, 0.9820893, 0.9849777, -0.0018084, 0.0018634
8: -0.0057010, -0.0026878, -0.0057674, -0.0026705, -0.0019389, 0.0019979
9: -0.0032241, -0.0012337, -0.0032356, -0.0011899, -0.0013197, 0.0012808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011246, upper bound: 0.0011404
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011307, upper bound: 0.0011369
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028303, -0.0017537, -0.0006734, 0.0006977
1: -0.0114861, -0.0086910, -0.0114927, -0.0087608, -0.0017088, 0.0017705
2: 0.0279040, 0.0296381, 0.0278999, 0.0295948, -0.0010601, 0.0010984
3: 0.0042206, 0.0074587, 0.0043015, 0.0074664, -0.0020511, 0.0019795
4: -0.0105763, -0.0077332, -0.0105831, -0.0078042, -0.0017381, 0.0018009
5: 0.0097322, 0.0108091, 0.0097296, 0.0107822, -0.0006583, 0.0006821
6: 0.0057447, 0.0098541, 0.0058473, 0.0098639, -0.0026031, 0.0025123
7: 0.9820791, 0.9849547, 0.9821509, 0.9849616, -0.0018215, 0.0017580
8: -0.0057783, -0.0026951, -0.0057013, -0.0026878, -0.0019529, 0.0018848
9: -0.0032193, -0.0011827, -0.0032241, -0.0012336, -0.0012450, 0.0012900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011420, upper bound: 0.0011345
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011463, upper bound: 0.0011311
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028364, -0.0017300, -0.0007040, 0.0007074
1: -0.0114861, -0.0086910, -0.0115084, -0.0087008, -0.0017865, 0.0017951
2: 0.0279040, 0.0296381, 0.0278902, 0.0296320, -0.0011084, 0.0011137
3: 0.0042206, 0.0074587, 0.0042321, 0.0074845, -0.0020795, 0.0020696
4: -0.0105763, -0.0077332, -0.0105990, -0.0077432, -0.0018172, 0.0018259
5: 0.0097322, 0.0108091, 0.0097236, 0.0108053, -0.0006883, 0.0006916
6: 0.0057447, 0.0098541, 0.0057591, 0.0098870, -0.0026392, 0.0026266
7: 0.9820791, 0.9849547, 0.9820893, 0.9849777, -0.0018468, 0.0018380
8: -0.0057783, -0.0026951, -0.0057674, -0.0026705, -0.0019801, 0.0019706
9: -0.0032193, -0.0011827, -0.0032356, -0.0011899, -0.0013017, 0.0013079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011420, upper bound: 0.0011395
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011463, upper bound: 0.0011363
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028303, -0.0017537, -0.0007572, 0.0006513
1: -0.0116633, -0.0088327, -0.0114927, -0.0087608, -0.0019214, 0.0016528
2: 0.0277940, 0.0295502, 0.0278999, 0.0295948, -0.0011921, 0.0010254
3: 0.0043848, 0.0076640, 0.0043015, 0.0074664, -0.0019147, 0.0022259
4: -0.0107566, -0.0078773, -0.0105831, -0.0078042, -0.0019544, 0.0016812
5: 0.0096639, 0.0107545, 0.0097296, 0.0107822, -0.0007403, 0.0006368
6: 0.0059530, 0.0101147, 0.0058473, 0.0098639, -0.0024301, 0.0028249
7: 0.9822249, 0.9851370, 0.9821509, 0.9849616, -0.0017004, 0.0019767
8: -0.0056220, -0.0024996, -0.0057013, -0.0026878, -0.0018231, 0.0021194
9: -0.0033484, -0.0012860, -0.0032241, -0.0012336, -0.0014000, 0.0012043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0010979
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011166, upper bound: 0.0011061
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028364, -0.0017300, -0.0007878, 0.0006610
1: -0.0116633, -0.0088327, -0.0115084, -0.0087008, -0.0019992, 0.0016774
2: 0.0277940, 0.0295502, 0.0278902, 0.0296320, -0.0012403, 0.0010407
3: 0.0043848, 0.0076640, 0.0042321, 0.0074845, -0.0019432, 0.0023160
4: -0.0107566, -0.0078773, -0.0105990, -0.0077432, -0.0020335, 0.0017062
5: 0.0096639, 0.0107545, 0.0097236, 0.0108053, -0.0007702, 0.0006463
6: 0.0059530, 0.0101147, 0.0057591, 0.0098870, -0.0024662, 0.0029393
7: 0.9822249, 0.9851370, 0.9820893, 0.9849777, -0.0017257, 0.0020568
8: -0.0056220, -0.0024996, -0.0057674, -0.0026705, -0.0018503, 0.0022052
9: -0.0033484, -0.0012860, -0.0032356, -0.0011899, -0.0014566, 0.0012222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0011016
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011166, upper bound: 0.0011101
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028303, -0.0017537, -0.0007449, 0.0006691
1: -0.0116579, -0.0087618, -0.0114927, -0.0087608, -0.0018902, 0.0016980
2: 0.0277974, 0.0295942, 0.0278999, 0.0295948, -0.0011727, 0.0010535
3: 0.0043027, 0.0076577, 0.0043015, 0.0074664, -0.0019671, 0.0021898
4: -0.0107511, -0.0078052, -0.0105831, -0.0078042, -0.0019227, 0.0017272
5: 0.0096660, 0.0107818, 0.0097296, 0.0107822, -0.0007283, 0.0006542
6: 0.0058487, 0.0101068, 0.0058473, 0.0098639, -0.0024965, 0.0027791
7: 0.9821519, 0.9851315, 0.9821509, 0.9849616, -0.0017469, 0.0019447
8: -0.0057002, -0.0025056, -0.0057013, -0.0026878, -0.0018730, 0.0020850
9: -0.0033445, -0.0012343, -0.0032241, -0.0012336, -0.0013773, 0.0012372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011180, upper bound: 0.0010975
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011321, upper bound: 0.0011057
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028364, -0.0017300, -0.0007755, 0.0006788
1: -0.0116579, -0.0087618, -0.0115084, -0.0087008, -0.0019680, 0.0017226
2: 0.0277974, 0.0295942, 0.0278902, 0.0296320, -0.0012210, 0.0010687
3: 0.0043027, 0.0076577, 0.0042321, 0.0074845, -0.0019956, 0.0022799
4: -0.0107511, -0.0078052, -0.0105990, -0.0077432, -0.0020018, 0.0017522
5: 0.0096660, 0.0107818, 0.0097236, 0.0108053, -0.0007582, 0.0006637
6: 0.0058487, 0.0101068, 0.0057591, 0.0098870, -0.0025326, 0.0028934
7: 0.9821519, 0.9851315, 0.9820893, 0.9849777, -0.0017722, 0.0020247
8: -0.0057002, -0.0025056, -0.0057674, -0.0026705, -0.0019001, 0.0021708
9: -0.0033445, -0.0012343, -0.0032356, -0.0011899, -0.0014339, 0.0012551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011180, upper bound: 0.0011013
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011321, upper bound: 0.0011100
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028975, -0.0017817, -0.0006516, 0.0007571
1: -0.0114927, -0.0087610, -0.0116633, -0.0088320, -0.0016536, 0.0019212
2: 0.0278999, 0.0295947, 0.0277940, 0.0295506, -0.0010259, 0.0011919
3: 0.0043017, 0.0074664, 0.0043840, 0.0076640, -0.0022256, 0.0019156
4: -0.0105831, -0.0078044, -0.0107566, -0.0078766, -0.0016820, 0.0019542
5: 0.0097296, 0.0107821, 0.0096639, 0.0107547, -0.0006371, 0.0007402
6: 0.0058476, 0.0098639, 0.0059520, 0.0101147, -0.0028246, 0.0024311
7: 0.9821512, 0.9849616, 0.9822242, 0.9851370, -0.0019765, 0.0017012
8: -0.0057010, -0.0026878, -0.0056227, -0.0024996, -0.0021191, 0.0018239
9: -0.0032241, -0.0012337, -0.0033484, -0.0012855, -0.0012048, 0.0013998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010973, upper bound: 0.0010944
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011059, upper bound: 0.0011171
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028975, -0.0017817, -0.0006888, 0.0006885
1: -0.0116633, -0.0088327, -0.0116633, -0.0088320, -0.0017480, 0.0017472
2: 0.0277940, 0.0295502, 0.0277940, 0.0295506, -0.0010845, 0.0010840
3: 0.0043848, 0.0076640, 0.0043840, 0.0076640, -0.0020241, 0.0020250
4: -0.0107566, -0.0078773, -0.0107566, -0.0078766, -0.0017780, 0.0017772
5: 0.0096639, 0.0107545, 0.0096639, 0.0107547, -0.0006735, 0.0006732
6: 0.0059530, 0.0101147, 0.0059520, 0.0101147, -0.0025688, 0.0025699
7: 0.9822249, 0.9851370, 0.9822242, 0.9851370, -0.0017975, 0.0017983
8: -0.0056220, -0.0024996, -0.0056227, -0.0024996, -0.0019272, 0.0019281
9: -0.0033484, -0.0012860, -0.0033484, -0.0012855, -0.0012736, 0.0012730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010973, upper bound: 0.0010877
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011059, upper bound: 0.0011059
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0029049, -0.0017575, -0.0006834, 0.0007645
1: -0.0114927, -0.0087610, -0.0116822, -0.0087706, -0.0017342, 0.0019401
2: 0.0278999, 0.0295947, 0.0277823, 0.0295887, -0.0010759, 0.0012037
3: 0.0043017, 0.0074664, 0.0043129, 0.0076859, -0.0022475, 0.0020090
4: -0.0105831, -0.0078044, -0.0107758, -0.0078142, -0.0017640, 0.0019734
5: 0.0097296, 0.0107821, 0.0096566, 0.0107784, -0.0006682, 0.0007475
6: 0.0058476, 0.0098639, 0.0058617, 0.0101425, -0.0028524, 0.0025497
7: 0.9821512, 0.9849616, 0.9821610, 0.9851565, -0.0019960, 0.0017842
8: -0.0057010, -0.0026878, -0.0056904, -0.0024788, -0.0021400, 0.0019129
9: -0.0032241, -0.0012337, -0.0033622, -0.0012407, -0.0012636, 0.0014136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010874, upper bound: 0.0011083
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011056, upper bound: 0.0011230
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0029049, -0.0017575, -0.0007199, 0.0006980
1: -0.0116633, -0.0088327, -0.0116822, -0.0087706, -0.0018269, 0.0017713
2: 0.0277940, 0.0295502, 0.0277823, 0.0295887, -0.0011334, 0.0010990
3: 0.0043848, 0.0076640, 0.0043129, 0.0076859, -0.0020520, 0.0021164
4: -0.0107566, -0.0078773, -0.0107758, -0.0078142, -0.0018583, 0.0018018
5: 0.0096639, 0.0107545, 0.0096566, 0.0107784, -0.0007039, 0.0006825
6: 0.0059530, 0.0101147, 0.0058617, 0.0101425, -0.0026043, 0.0026860
7: 0.9822249, 0.9851370, 0.9821610, 0.9851565, -0.0018223, 0.0018795
8: -0.0056220, -0.0024996, -0.0056904, -0.0024788, -0.0019538, 0.0020151
9: -0.0033484, -0.0012860, -0.0033622, -0.0012407, -0.0013311, 0.0012906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010874, upper bound: 0.0011011
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011056, upper bound: 0.0011101
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028975, -0.0017817, -0.0006419, 0.0007718
1: -0.0114861, -0.0086910, -0.0116633, -0.0088320, -0.0016288, 0.0019584
2: 0.0279040, 0.0296381, 0.0277940, 0.0295506, -0.0010105, 0.0012150
3: 0.0042206, 0.0074587, 0.0043840, 0.0076640, -0.0022688, 0.0018869
4: -0.0105763, -0.0077332, -0.0107566, -0.0078766, -0.0016568, 0.0019921
5: 0.0097322, 0.0108091, 0.0096639, 0.0107547, -0.0006275, 0.0007545
6: 0.0057447, 0.0098541, 0.0059520, 0.0101147, -0.0028794, 0.0023947
7: 0.9820791, 0.9849547, 0.9822242, 0.9851370, -0.0020148, 0.0016757
8: -0.0057783, -0.0026951, -0.0056227, -0.0024996, -0.0021602, 0.0017966
9: -0.0032193, -0.0011827, -0.0033484, -0.0012855, -0.0011868, 0.0014269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011142, upper bound: 0.0010944
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011172
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028975, -0.0017817, -0.0006791, 0.0007050
1: -0.0116579, -0.0087618, -0.0116633, -0.0088320, -0.0017232, 0.0017892
2: 0.0277974, 0.0295942, 0.0277940, 0.0295506, -0.0010691, 0.0011100
3: 0.0043027, 0.0076577, 0.0043840, 0.0076640, -0.0020726, 0.0019963
4: -0.0107511, -0.0078052, -0.0107566, -0.0078766, -0.0017528, 0.0018199
5: 0.0096660, 0.0107818, 0.0096639, 0.0107547, -0.0006639, 0.0006893
6: 0.0058487, 0.0101068, 0.0059520, 0.0101147, -0.0026305, 0.0025335
7: 0.9821519, 0.9851315, 0.9822242, 0.9851370, -0.0018407, 0.0017728
8: -0.0057002, -0.0025056, -0.0056227, -0.0024996, -0.0019735, 0.0019008
9: -0.0033445, -0.0012343, -0.0033484, -0.0012855, -0.0012556, 0.0013036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011062, upper bound: 0.0010968
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011057
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0029049, -0.0017575, -0.0006737, 0.0007792
1: -0.0114861, -0.0086910, -0.0116822, -0.0087706, -0.0017095, 0.0019774
2: 0.0279040, 0.0296381, 0.0277823, 0.0295887, -0.0010606, 0.0012268
3: 0.0042206, 0.0074587, 0.0043129, 0.0076859, -0.0022907, 0.0019804
4: -0.0105763, -0.0077332, -0.0107758, -0.0078142, -0.0017389, 0.0020113
5: 0.0097322, 0.0108091, 0.0096566, 0.0107784, -0.0006586, 0.0007618
6: 0.0057447, 0.0098541, 0.0058617, 0.0101425, -0.0029072, 0.0025134
7: 0.9820791, 0.9849547, 0.9821610, 0.9851565, -0.0020343, 0.0017587
8: -0.0057783, -0.0026951, -0.0056904, -0.0024788, -0.0021811, 0.0018856
9: -0.0032193, -0.0011827, -0.0033622, -0.0012407, -0.0012456, 0.0014407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011062, upper bound: 0.0011083
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011230
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0029049, -0.0017575, -0.0007102, 0.0007146
1: -0.0116579, -0.0087618, -0.0116822, -0.0087706, -0.0018022, 0.0018133
2: 0.0277974, 0.0295942, 0.0277823, 0.0295887, -0.0011181, 0.0011250
3: 0.0043027, 0.0076577, 0.0043129, 0.0076859, -0.0021006, 0.0020877
4: -0.0107511, -0.0078052, -0.0107758, -0.0078142, -0.0018331, 0.0018444
5: 0.0096660, 0.0107818, 0.0096566, 0.0107784, -0.0006943, 0.0006986
6: 0.0058487, 0.0101068, 0.0058617, 0.0101425, -0.0026659, 0.0026496
7: 0.9821519, 0.9851315, 0.9821610, 0.9851565, -0.0018655, 0.0018541
8: -0.0057002, -0.0025056, -0.0056904, -0.0024788, -0.0020001, 0.0019878
9: -0.0033445, -0.0012343, -0.0033622, -0.0012407, -0.0013131, 0.0013212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011062, upper bound: 0.0011009
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011100
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028276, -0.0017262, -0.0006977, 0.0006733
1: -0.0114927, -0.0087610, -0.0114861, -0.0086910, -0.0017705, 0.0017085
2: 0.0278999, 0.0295947, 0.0279040, 0.0296381, -0.0010984, 0.0010600
3: 0.0043017, 0.0074664, 0.0042206, 0.0074587, -0.0019792, 0.0020511
4: -0.0105831, -0.0078044, -0.0105763, -0.0077332, -0.0018009, 0.0017378
5: 0.0097296, 0.0107821, 0.0097322, 0.0108091, -0.0006821, 0.0006582
6: 0.0058476, 0.0098639, 0.0057447, 0.0098541, -0.0025119, 0.0026031
7: 0.9821512, 0.9849616, 0.9820791, 0.9849547, -0.0017577, 0.0018215
8: -0.0057010, -0.0026878, -0.0057783, -0.0026951, -0.0018845, 0.0019529
9: -0.0032241, -0.0012337, -0.0032193, -0.0011827, -0.0012900, 0.0012448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011340, upper bound: 0.0011420
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011303, upper bound: 0.0011461
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028332, -0.0017034, -0.0007341, 0.0006842
1: -0.0114927, -0.0087610, -0.0115002, -0.0086332, -0.0018628, 0.0017363
2: 0.0278999, 0.0295947, 0.0278952, 0.0296739, -0.0011557, 0.0010772
3: 0.0043017, 0.0074664, 0.0041538, 0.0074750, -0.0020115, 0.0021580
4: -0.0105831, -0.0078044, -0.0105907, -0.0076745, -0.0018948, 0.0017661
5: 0.0097296, 0.0107821, 0.0097267, 0.0108313, -0.0007177, 0.0006690
6: 0.0058476, 0.0098639, 0.0056598, 0.0098749, -0.0025528, 0.0027387
7: 0.9821512, 0.9849616, 0.9820197, 0.9849693, -0.0017863, 0.0019164
8: -0.0057010, -0.0026878, -0.0058419, -0.0026796, -0.0019152, 0.0020547
9: -0.0032241, -0.0012337, -0.0032296, -0.0011407, -0.0013573, 0.0012651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011241, upper bound: 0.0011521
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011303, upper bound: 0.0011500
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028276, -0.0017262, -0.0007718, 0.0006416
1: -0.0116633, -0.0088327, -0.0114861, -0.0086910, -0.0019584, 0.0016281
2: 0.0277940, 0.0295502, 0.0279040, 0.0296381, -0.0012150, 0.0010101
3: 0.0043848, 0.0076640, 0.0042206, 0.0074587, -0.0018861, 0.0022688
4: -0.0107566, -0.0078773, -0.0105763, -0.0077332, -0.0019921, 0.0016561
5: 0.0096639, 0.0107545, 0.0097322, 0.0108091, -0.0007545, 0.0006273
6: 0.0059530, 0.0101147, 0.0057447, 0.0098541, -0.0023937, 0.0028794
7: 0.9822249, 0.9851370, 0.9820791, 0.9849547, -0.0016750, 0.0020148
8: -0.0056220, -0.0024996, -0.0057783, -0.0026951, -0.0017958, 0.0021602
9: -0.0033484, -0.0012860, -0.0032193, -0.0011827, -0.0014269, 0.0011863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0011137
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011165, upper bound: 0.0011205
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028332, -0.0017034, -0.0008081, 0.0006525
1: -0.0116633, -0.0088327, -0.0115002, -0.0086332, -0.0020507, 0.0016559
2: 0.0277940, 0.0295502, 0.0278952, 0.0296739, -0.0012723, 0.0010273
3: 0.0043848, 0.0076640, 0.0041538, 0.0074750, -0.0019183, 0.0023757
4: -0.0107566, -0.0078773, -0.0105907, -0.0076745, -0.0020859, 0.0016844
5: 0.0096639, 0.0107545, 0.0097267, 0.0108313, -0.0007901, 0.0006380
6: 0.0059530, 0.0101147, 0.0056598, 0.0098749, -0.0024346, 0.0030150
7: 0.9822249, 0.9851370, 0.9820197, 0.9849693, -0.0017036, 0.0021098
8: -0.0056220, -0.0024996, -0.0058419, -0.0026796, -0.0018265, 0.0022620
9: -0.0033484, -0.0012860, -0.0032296, -0.0011407, -0.0014942, 0.0012065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0011168
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011165, upper bound: 0.0011237
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028276, -0.0017262, -0.0007074, 0.0007074
1: -0.0114861, -0.0086910, -0.0114861, -0.0086910, -0.0017952, 0.0017952
2: 0.0279040, 0.0296381, 0.0279040, 0.0296381, -0.0011137, 0.0011137
3: 0.0042206, 0.0074587, 0.0042206, 0.0074587, -0.0020796, 0.0020796
4: -0.0105763, -0.0077332, -0.0105763, -0.0077332, -0.0018260, 0.0018260
5: 0.0097322, 0.0108091, 0.0097322, 0.0108091, -0.0006916, 0.0006916
6: 0.0057447, 0.0098541, 0.0057447, 0.0098541, -0.0026393, 0.0026393
7: 0.9820791, 0.9849547, 0.9820791, 0.9849547, -0.0018469, 0.0018469
8: -0.0057783, -0.0026951, -0.0057783, -0.0026951, -0.0019801, 0.0019801
9: -0.0032193, -0.0011827, -0.0032193, -0.0011827, -0.0013080, 0.0013080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011602, upper bound: 0.0011370
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011557, upper bound: 0.0011430
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028332, -0.0017034, -0.0007377, 0.0007170
1: -0.0114861, -0.0086910, -0.0115002, -0.0086332, -0.0018719, 0.0018196
2: 0.0279040, 0.0296381, 0.0278952, 0.0296739, -0.0011614, 0.0011289
3: 0.0042206, 0.0074587, 0.0041538, 0.0074750, -0.0021079, 0.0021685
4: -0.0105763, -0.0077332, -0.0105907, -0.0076745, -0.0019041, 0.0018508
5: 0.0097322, 0.0108091, 0.0097267, 0.0108313, -0.0007212, 0.0007010
6: 0.0057447, 0.0098541, 0.0056598, 0.0098749, -0.0026752, 0.0027522
7: 0.9820791, 0.9849547, 0.9820197, 0.9849693, -0.0018720, 0.0019258
8: -0.0057783, -0.0026951, -0.0058419, -0.0026796, -0.0020071, 0.0020648
9: -0.0032193, -0.0011827, -0.0032296, -0.0011407, -0.0013639, 0.0013258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011509, upper bound: 0.0011517
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011557, upper bound: 0.0011486
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028276, -0.0017262, -0.0007830, 0.0006778
1: -0.0116579, -0.0087618, -0.0114861, -0.0086910, -0.0019871, 0.0017200
2: 0.0277974, 0.0295942, 0.0279040, 0.0296381, -0.0012328, 0.0010671
3: 0.0043027, 0.0076577, 0.0042206, 0.0074587, -0.0019925, 0.0023019
4: -0.0107511, -0.0078052, -0.0105763, -0.0077332, -0.0020212, 0.0017495
5: 0.0096660, 0.0107818, 0.0097322, 0.0108091, -0.0007656, 0.0006627
6: 0.0058487, 0.0101068, 0.0057447, 0.0098541, -0.0025287, 0.0029214
7: 0.9821519, 0.9851315, 0.9820791, 0.9849547, -0.0017695, 0.0020443
8: -0.0057002, -0.0025056, -0.0057783, -0.0026951, -0.0018972, 0.0021918
9: -0.0033445, -0.0012343, -0.0032193, -0.0011827, -0.0014478, 0.0012532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011255, upper bound: 0.0011100
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011391, upper bound: 0.0011153
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028332, -0.0017034, -0.0008133, 0.0006874
1: -0.0116579, -0.0087618, -0.0115002, -0.0086332, -0.0020638, 0.0017443
2: 0.0277974, 0.0295942, 0.0278952, 0.0296739, -0.0012804, 0.0010822
3: 0.0043027, 0.0076577, 0.0041538, 0.0074750, -0.0020207, 0.0023908
4: -0.0107511, -0.0078052, -0.0105907, -0.0076745, -0.0020992, 0.0017743
5: 0.0096660, 0.0107818, 0.0097267, 0.0108313, -0.0007951, 0.0006721
6: 0.0058487, 0.0101068, 0.0056598, 0.0098749, -0.0025646, 0.0030343
7: 0.9821519, 0.9851315, 0.9820197, 0.9849693, -0.0017946, 0.0021232
8: -0.0057002, -0.0025056, -0.0058419, -0.0026796, -0.0019241, 0.0022764
9: -0.0033445, -0.0012343, -0.0032296, -0.0011407, -0.0015037, 0.0012710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011255, upper bound: 0.0011145
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011391, upper bound: 0.0011198
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0028953, -0.0017541, -0.0006691, 0.0007448
1: -0.0114927, -0.0087610, -0.0116579, -0.0087618, -0.0016980, 0.0018900
2: 0.0278999, 0.0295947, 0.0277974, 0.0295942, -0.0010535, 0.0011726
3: 0.0043017, 0.0074664, 0.0043027, 0.0076577, -0.0021895, 0.0019671
4: -0.0105831, -0.0078044, -0.0107511, -0.0078052, -0.0017272, 0.0019224
5: 0.0097296, 0.0107821, 0.0096660, 0.0107818, -0.0006542, 0.0007282
6: 0.0058476, 0.0098639, 0.0058487, 0.0101068, -0.0027787, 0.0024965
7: 0.9821512, 0.9849616, 0.9821519, 0.9851315, -0.0019444, 0.0017469
8: -0.0057010, -0.0026878, -0.0057002, -0.0025056, -0.0020847, 0.0018730
9: -0.0032241, -0.0012337, -0.0033445, -0.0012343, -0.0012372, 0.0013771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010968, upper bound: 0.0011142
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011057, upper bound: 0.0011322
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0028953, -0.0017541, -0.0007050, 0.0006788
1: -0.0116633, -0.0088327, -0.0116579, -0.0087618, -0.0017892, 0.0017225
2: 0.0277940, 0.0295502, 0.0277974, 0.0295942, -0.0011100, 0.0010686
3: 0.0043848, 0.0076640, 0.0043027, 0.0076577, -0.0019954, 0.0020726
4: -0.0107566, -0.0078773, -0.0107511, -0.0078052, -0.0018199, 0.0017520
5: 0.0096639, 0.0107545, 0.0096660, 0.0107818, -0.0006893, 0.0006636
6: 0.0059530, 0.0101147, 0.0058487, 0.0101068, -0.0025324, 0.0026305
7: 0.9822249, 0.9851370, 0.9821519, 0.9851315, -0.0017721, 0.0018407
8: -0.0056220, -0.0024996, -0.0057002, -0.0025056, -0.0018999, 0.0019735
9: -0.0033484, -0.0012860, -0.0033445, -0.0012343, -0.0013036, 0.0012550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010968, upper bound: 0.0011062
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011057, upper bound: 0.0011210
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028303, -0.0017537, -0.0029019, -0.0017293, -0.0007050, 0.0007562
1: -0.0114927, -0.0087610, -0.0116746, -0.0086990, -0.0017891, 0.0019190
2: 0.0278999, 0.0295947, 0.0277871, 0.0296331, -0.0011100, 0.0011905
3: 0.0043017, 0.0074664, 0.0042300, 0.0076770, -0.0022230, 0.0020726
4: -0.0105831, -0.0078044, -0.0107680, -0.0077414, -0.0018198, 0.0019519
5: 0.0097296, 0.0107821, 0.0096595, 0.0108060, -0.0006893, 0.0007393
6: 0.0058476, 0.0098639, 0.0057565, 0.0101312, -0.0028213, 0.0026304
7: 0.9821512, 0.9849616, 0.9820874, 0.9851486, -0.0019742, 0.0018406
8: -0.0057010, -0.0026878, -0.0057694, -0.0024872, -0.0021167, 0.0019735
9: -0.0032241, -0.0012337, -0.0033566, -0.0011886, -0.0013036, 0.0013982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010968, upper bound: 0.0011218
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011052, upper bound: 0.0011384
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028975, -0.0017820, -0.0029019, -0.0017293, -0.0007415, 0.0006898
1: -0.0116633, -0.0088327, -0.0116746, -0.0086990, -0.0018816, 0.0017505
2: 0.0277940, 0.0295502, 0.0277871, 0.0296331, -0.0011674, 0.0010860
3: 0.0043848, 0.0076640, 0.0042300, 0.0076770, -0.0020278, 0.0021798
4: -0.0107566, -0.0078773, -0.0107680, -0.0077414, -0.0019140, 0.0017805
5: 0.0096639, 0.0107545, 0.0096595, 0.0108060, -0.0007250, 0.0006744
6: 0.0059530, 0.0101147, 0.0057565, 0.0101312, -0.0025736, 0.0027664
7: 0.9822249, 0.9851370, 0.9820874, 0.9851486, -0.0018009, 0.0019358
8: -0.0056220, -0.0024996, -0.0057694, -0.0024872, -0.0019308, 0.0020755
9: -0.0033484, -0.0012860, -0.0033566, -0.0011886, -0.0013710, 0.0012754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010873, upper bound: 0.0011167
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011052, upper bound: 0.0011237
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0028953, -0.0017541, -0.0006778, 0.0007830
1: -0.0114861, -0.0086910, -0.0116579, -0.0087618, -0.0017200, 0.0019871
2: 0.0279040, 0.0296381, 0.0277974, 0.0295942, -0.0010671, 0.0012328
3: 0.0042206, 0.0074587, 0.0043027, 0.0076577, -0.0023019, 0.0019925
4: -0.0105763, -0.0077332, -0.0107511, -0.0078052, -0.0017495, 0.0020212
5: 0.0097322, 0.0108091, 0.0096660, 0.0107818, -0.0006627, 0.0007656
6: 0.0057447, 0.0098541, 0.0058487, 0.0101068, -0.0029214, 0.0025287
7: 0.9820791, 0.9849547, 0.9821519, 0.9851315, -0.0020443, 0.0017695
8: -0.0057783, -0.0026951, -0.0057002, -0.0025056, -0.0021918, 0.0018972
9: -0.0032193, -0.0011827, -0.0033445, -0.0012343, -0.0012532, 0.0014478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011223, upper bound: 0.0011094
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011274
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0028953, -0.0017541, -0.0007138, 0.0007138
1: -0.0116579, -0.0087618, -0.0116579, -0.0087618, -0.0018114, 0.0018114
2: 0.0277974, 0.0295942, 0.0277974, 0.0295942, -0.0011238, 0.0011238
3: 0.0043027, 0.0076577, 0.0043027, 0.0076577, -0.0020985, 0.0020985
4: -0.0107511, -0.0078052, -0.0107511, -0.0078052, -0.0018425, 0.0018425
5: 0.0096660, 0.0107818, 0.0096660, 0.0107818, -0.0006979, 0.0006979
6: 0.0058487, 0.0101068, 0.0058487, 0.0101068, -0.0026632, 0.0026632
7: 0.9821519, 0.9851315, 0.9821519, 0.9851315, -0.0018636, 0.0018636
8: -0.0057002, -0.0025056, -0.0057002, -0.0025056, -0.0019981, 0.0019981
9: -0.0033445, -0.0012343, -0.0033445, -0.0012343, -0.0013198, 0.0013198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011223, upper bound: 0.0011010
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011153
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017262, -0.0029019, -0.0017293, -0.0007072, 0.0007894
1: -0.0114861, -0.0086910, -0.0116746, -0.0086990, -0.0017946, 0.0020033
2: 0.0279040, 0.0296381, 0.0277871, 0.0296331, -0.0011134, 0.0012428
3: 0.0042206, 0.0074587, 0.0042300, 0.0076770, -0.0023207, 0.0020790
4: -0.0105763, -0.0077332, -0.0107680, -0.0077414, -0.0018255, 0.0020377
5: 0.0097322, 0.0108091, 0.0096595, 0.0108060, -0.0006914, 0.0007718
6: 0.0057447, 0.0098541, 0.0057565, 0.0101312, -0.0029453, 0.0026385
7: 0.9820791, 0.9849547, 0.9820874, 0.9851486, -0.0020610, 0.0018463
8: -0.0057783, -0.0026951, -0.0057694, -0.0024872, -0.0022097, 0.0019795
9: -0.0032193, -0.0011827, -0.0033566, -0.0011886, -0.0013076, 0.0014596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011138, upper bound: 0.0011226
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011334
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017541, -0.0029019, -0.0017293, -0.0007448, 0.0007231
1: -0.0116579, -0.0087618, -0.0116746, -0.0086990, -0.0018900, 0.0018350
2: 0.0277974, 0.0295942, 0.0277871, 0.0296331, -0.0011726, 0.0011384
3: 0.0043027, 0.0076577, 0.0042300, 0.0076770, -0.0021257, 0.0021895
4: -0.0107511, -0.0078052, -0.0107680, -0.0077414, -0.0019225, 0.0018665
5: 0.0096660, 0.0107818, 0.0096595, 0.0108060, -0.0007282, 0.0007070
6: 0.0058487, 0.0101068, 0.0057565, 0.0101312, -0.0026978, 0.0027788
7: 0.9821519, 0.9851315, 0.9820874, 0.9851486, -0.0018878, 0.0019445
8: -0.0057002, -0.0025056, -0.0057694, -0.0024872, -0.0020240, 0.0020848
9: -0.0033445, -0.0012343, -0.0033566, -0.0011886, -0.0013771, 0.0013370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011138, upper bound: 0.0011144
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011198
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028303, -0.0017537, -0.0006928, 0.0007138
1: -0.0115084, -0.0087008, -0.0114927, -0.0087608, -0.0017581, 0.0018113
2: 0.0278902, 0.0296320, 0.0278999, 0.0295948, -0.0010907, 0.0011237
3: 0.0042321, 0.0074845, 0.0043015, 0.0074664, -0.0020983, 0.0020367
4: -0.0105990, -0.0077432, -0.0105831, -0.0078042, -0.0017883, 0.0018424
5: 0.0097236, 0.0108053, 0.0097296, 0.0107822, -0.0006773, 0.0006978
6: 0.0057591, 0.0098870, 0.0058473, 0.0098639, -0.0026630, 0.0025848
7: 0.9820893, 0.9849777, 0.9821509, 0.9849616, -0.0018634, 0.0018087
8: -0.0057674, -0.0026705, -0.0057013, -0.0026878, -0.0019979, 0.0019392
9: -0.0032356, -0.0011899, -0.0032241, -0.0012336, -0.0012810, 0.0013197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011241
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011369, upper bound: 0.0011303
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028303, -0.0017537, -0.0006843, 0.0007341
1: -0.0115002, -0.0086332, -0.0114927, -0.0087608, -0.0017366, 0.0018628
2: 0.0278952, 0.0296739, 0.0278999, 0.0295948, -0.0010774, 0.0011557
3: 0.0041538, 0.0074750, 0.0043015, 0.0074664, -0.0021580, 0.0020118
4: -0.0105907, -0.0076745, -0.0105831, -0.0078042, -0.0017664, 0.0018948
5: 0.0097267, 0.0108313, 0.0097296, 0.0107822, -0.0006691, 0.0007177
6: 0.0056598, 0.0098749, 0.0058473, 0.0098639, -0.0027387, 0.0025532
7: 0.9820197, 0.9849693, 0.9821509, 0.9849616, -0.0019164, 0.0017866
8: -0.0058419, -0.0026796, -0.0057013, -0.0026878, -0.0020547, 0.0019155
9: -0.0032296, -0.0011407, -0.0032241, -0.0012336, -0.0012653, 0.0013573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011241
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011369, upper bound: 0.0011303
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028364, -0.0017300, -0.0007082, 0.0007082
1: -0.0115084, -0.0087008, -0.0115084, -0.0087008, -0.0017972, 0.0017972
2: 0.0278902, 0.0296320, 0.0278902, 0.0296320, -0.0011150, 0.0011150
3: 0.0042321, 0.0074845, 0.0042321, 0.0074845, -0.0020820, 0.0020820
4: -0.0105990, -0.0077432, -0.0105990, -0.0077432, -0.0018281, 0.0018281
5: 0.0097236, 0.0108053, 0.0097236, 0.0108053, -0.0006924, 0.0006924
6: 0.0057591, 0.0098870, 0.0057591, 0.0098870, -0.0026423, 0.0026423
7: 0.9820893, 0.9849777, 0.9820893, 0.9849777, -0.0018490, 0.0018490
8: -0.0057674, -0.0026705, -0.0057674, -0.0026705, -0.0019824, 0.0019824
9: -0.0032356, -0.0011899, -0.0032356, -0.0011899, -0.0013095, 0.0013095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011429, upper bound: 0.0011274
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011405, upper bound: 0.0011342
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028364, -0.0017300, -0.0006965, 0.0007224
1: -0.0115002, -0.0086332, -0.0115084, -0.0087008, -0.0017674, 0.0018332
2: 0.0278952, 0.0296739, 0.0278902, 0.0296320, -0.0010965, 0.0011373
3: 0.0041538, 0.0074750, 0.0042321, 0.0074845, -0.0021237, 0.0020475
4: -0.0105907, -0.0076745, -0.0105990, -0.0077432, -0.0017978, 0.0018647
5: 0.0097267, 0.0108313, 0.0097236, 0.0108053, -0.0006809, 0.0007063
6: 0.0056598, 0.0098749, 0.0057591, 0.0098870, -0.0026952, 0.0025985
7: 0.9820197, 0.9849693, 0.9820893, 0.9849777, -0.0018860, 0.0018183
8: -0.0058419, -0.0026796, -0.0057674, -0.0026705, -0.0020221, 0.0019495
9: -0.0032296, -0.0011407, -0.0032356, -0.0011899, -0.0012878, 0.0013357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011351, upper bound: 0.0011371
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011405, upper bound: 0.0011342
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028303, -0.0017537, -0.0007646, 0.0006834
1: -0.0116822, -0.0087706, -0.0114927, -0.0087608, -0.0019404, 0.0017342
2: 0.0277823, 0.0295887, 0.0278999, 0.0295948, -0.0012038, 0.0010759
3: 0.0043129, 0.0076859, 0.0043015, 0.0074664, -0.0020090, 0.0022478
4: -0.0107758, -0.0078142, -0.0105831, -0.0078042, -0.0019737, 0.0017640
5: 0.0096566, 0.0107784, 0.0097296, 0.0107822, -0.0007476, 0.0006682
6: 0.0058617, 0.0101425, 0.0058473, 0.0098639, -0.0025497, 0.0028528
7: 0.9821610, 0.9851565, 0.9821509, 0.9849616, -0.0017842, 0.0019962
8: -0.0056904, -0.0024788, -0.0057013, -0.0026878, -0.0019129, 0.0021403
9: -0.0033622, -0.0012407, -0.0032241, -0.0012336, -0.0014138, 0.0012636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011122, upper bound: 0.0010874
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011054
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028303, -0.0017537, -0.0007563, 0.0007050
1: -0.0116746, -0.0086990, -0.0114927, -0.0087608, -0.0019192, 0.0017891
2: 0.0277871, 0.0296331, 0.0278999, 0.0295948, -0.0011907, 0.0011100
3: 0.0042300, 0.0076770, 0.0043015, 0.0074664, -0.0020726, 0.0022233
4: -0.0107680, -0.0077414, -0.0105831, -0.0078042, -0.0019522, 0.0018198
5: 0.0096595, 0.0108060, 0.0097296, 0.0107822, -0.0007394, 0.0006893
6: 0.0057565, 0.0101312, 0.0058473, 0.0098639, -0.0026304, 0.0028217
7: 0.9820874, 0.9851486, 0.9821509, 0.9849616, -0.0018406, 0.0019745
8: -0.0057694, -0.0024872, -0.0057013, -0.0026878, -0.0019735, 0.0021170
9: -0.0033566, -0.0011886, -0.0032241, -0.0012336, -0.0013984, 0.0013036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011087, upper bound: 0.0010976
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011053
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028364, -0.0017300, -0.0007842, 0.0006784
1: -0.0116822, -0.0087706, -0.0115084, -0.0087008, -0.0019901, 0.0017216
2: 0.0277823, 0.0295887, 0.0278902, 0.0296320, -0.0012346, 0.0010681
3: 0.0043129, 0.0076859, 0.0042321, 0.0074845, -0.0019943, 0.0023054
4: -0.0107758, -0.0078142, -0.0105990, -0.0077432, -0.0020242, 0.0017511
5: 0.0096566, 0.0107784, 0.0097236, 0.0108053, -0.0007667, 0.0006633
6: 0.0058617, 0.0101425, 0.0057591, 0.0098870, -0.0025311, 0.0029258
7: 0.9821610, 0.9851565, 0.9820893, 0.9849777, -0.0017711, 0.0020474
8: -0.0056904, -0.0024788, -0.0057674, -0.0026705, -0.0018989, 0.0021951
9: -0.0033622, -0.0012407, -0.0032356, -0.0011899, -0.0014500, 0.0012543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011116, upper bound: 0.0011023
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011108
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028364, -0.0017300, -0.0007694, 0.0006942
1: -0.0116746, -0.0086990, -0.0115084, -0.0087008, -0.0019524, 0.0017616
2: 0.0277871, 0.0296331, 0.0278902, 0.0296320, -0.0012113, 0.0010929
3: 0.0042300, 0.0076770, 0.0042321, 0.0074845, -0.0020407, 0.0022617
4: -0.0107680, -0.0077414, -0.0105990, -0.0077432, -0.0019859, 0.0017918
5: 0.0096595, 0.0108060, 0.0097236, 0.0108053, -0.0007522, 0.0006787
6: 0.0057565, 0.0101312, 0.0057591, 0.0098870, -0.0025899, 0.0028704
7: 0.9820874, 0.9851486, 0.9820893, 0.9849777, -0.0018123, 0.0020086
8: -0.0057694, -0.0024872, -0.0057674, -0.0026705, -0.0019431, 0.0021535
9: -0.0033566, -0.0011886, -0.0032356, -0.0011899, -0.0014225, 0.0012835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011116, upper bound: 0.0011023
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011108
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028975, -0.0017817, -0.0006613, 0.0007878
1: -0.0115084, -0.0087008, -0.0116633, -0.0088320, -0.0016782, 0.0019992
2: 0.0278902, 0.0296320, 0.0277940, 0.0295506, -0.0010411, 0.0012403
3: 0.0042321, 0.0074845, 0.0043840, 0.0076640, -0.0023160, 0.0019441
4: -0.0105990, -0.0077432, -0.0107566, -0.0078766, -0.0017070, 0.0020335
5: 0.0097236, 0.0108053, 0.0096639, 0.0107547, -0.0006466, 0.0007702
6: 0.0057591, 0.0098870, 0.0059520, 0.0101147, -0.0029393, 0.0024673
7: 0.9820893, 0.9849777, 0.9822242, 0.9851370, -0.0020568, 0.0017265
8: -0.0057674, -0.0026705, -0.0056227, -0.0024996, -0.0022052, 0.0018511
9: -0.0032356, -0.0011899, -0.0033484, -0.0012855, -0.0012227, 0.0014566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011011, upper bound: 0.0010937
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011101, upper bound: 0.0011166
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028975, -0.0017817, -0.0006983, 0.0007199
1: -0.0116822, -0.0087706, -0.0116633, -0.0088320, -0.0017721, 0.0018269
2: 0.0277823, 0.0295887, 0.0277940, 0.0295506, -0.0010994, 0.0011334
3: 0.0043129, 0.0076859, 0.0043840, 0.0076640, -0.0021164, 0.0020529
4: -0.0107758, -0.0078142, -0.0107566, -0.0078766, -0.0018025, 0.0018583
5: 0.0096566, 0.0107784, 0.0096639, 0.0107547, -0.0006828, 0.0007039
6: 0.0058617, 0.0101425, 0.0059520, 0.0101147, -0.0026860, 0.0026054
7: 0.9821610, 0.9851565, 0.9822242, 0.9851370, -0.0018795, 0.0018231
8: -0.0056904, -0.0024788, -0.0056227, -0.0024996, -0.0020151, 0.0019547
9: -0.0033622, -0.0012407, -0.0033484, -0.0012855, -0.0012912, 0.0013311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011011, upper bound: 0.0010874
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011101, upper bound: 0.0011056
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028975, -0.0017817, -0.0006528, 0.0008081
1: -0.0115002, -0.0086332, -0.0116633, -0.0088320, -0.0016567, 0.0020507
2: 0.0278952, 0.0296739, 0.0277940, 0.0295506, -0.0010278, 0.0012723
3: 0.0041538, 0.0074750, 0.0043840, 0.0076640, -0.0023757, 0.0019192
4: -0.0105907, -0.0076745, -0.0107566, -0.0078766, -0.0016851, 0.0020859
5: 0.0097267, 0.0108313, 0.0096639, 0.0107547, -0.0006383, 0.0007901
6: 0.0056598, 0.0098749, 0.0059520, 0.0101147, -0.0030150, 0.0024357
7: 0.9820197, 0.9849693, 0.9822242, 0.9851370, -0.0021098, 0.0017044
8: -0.0058419, -0.0026796, -0.0056227, -0.0024996, -0.0022620, 0.0018273
9: -0.0032296, -0.0011407, -0.0033484, -0.0012855, -0.0012071, 0.0014942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011167, upper bound: 0.0010937
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0011165
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028975, -0.0017817, -0.0006901, 0.0007415
1: -0.0116746, -0.0086990, -0.0116633, -0.0088320, -0.0017512, 0.0018816
2: 0.0277871, 0.0296331, 0.0277940, 0.0295506, -0.0010865, 0.0011674
3: 0.0042300, 0.0076770, 0.0043840, 0.0076640, -0.0021798, 0.0020287
4: -0.0107680, -0.0077414, -0.0107566, -0.0078766, -0.0017813, 0.0019140
5: 0.0096595, 0.0108060, 0.0096639, 0.0107547, -0.0006747, 0.0007250
6: 0.0057565, 0.0101312, 0.0059520, 0.0101147, -0.0027664, 0.0025747
7: 0.9820874, 0.9851486, 0.9822242, 0.9851370, -0.0019358, 0.0018017
8: -0.0057694, -0.0024872, -0.0056227, -0.0024996, -0.0020755, 0.0019317
9: -0.0033566, -0.0011886, -0.0033484, -0.0012855, -0.0012760, 0.0013710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011167, upper bound: 0.0010873
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0011052
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0029049, -0.0017575, -0.0006784, 0.0007842
1: -0.0115084, -0.0087008, -0.0116822, -0.0087706, -0.0017216, 0.0019901
2: 0.0278902, 0.0296320, 0.0277823, 0.0295887, -0.0010681, 0.0012346
3: 0.0042321, 0.0074845, 0.0043129, 0.0076859, -0.0023054, 0.0019943
4: -0.0105990, -0.0077432, -0.0107758, -0.0078142, -0.0017511, 0.0020242
5: 0.0097236, 0.0108053, 0.0096566, 0.0107784, -0.0006633, 0.0007667
6: 0.0057591, 0.0098870, 0.0058617, 0.0101425, -0.0029258, 0.0025311
7: 0.9820893, 0.9849777, 0.9821610, 0.9851565, -0.0020474, 0.0017711
8: -0.0057674, -0.0026705, -0.0056904, -0.0024788, -0.0021951, 0.0018989
9: -0.0032356, -0.0011899, -0.0033622, -0.0012407, -0.0012543, 0.0014500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011050, upper bound: 0.0010969
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011149, upper bound: 0.0011220
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0029049, -0.0017575, -0.0007159, 0.0007159
1: -0.0116822, -0.0087706, -0.0116822, -0.0087706, -0.0018167, 0.0018167
2: 0.0277823, 0.0295887, 0.0277823, 0.0295887, -0.0011271, 0.0011271
3: 0.0043129, 0.0076859, 0.0043129, 0.0076859, -0.0021046, 0.0021046
4: -0.0107758, -0.0078142, -0.0107758, -0.0078142, -0.0018479, 0.0018479
5: 0.0096566, 0.0107784, 0.0096566, 0.0107784, -0.0006999, 0.0006999
6: 0.0058617, 0.0101425, 0.0058617, 0.0101425, -0.0026710, 0.0026710
7: 0.9821610, 0.9851565, 0.9821610, 0.9851565, -0.0018690, 0.0018690
8: -0.0056904, -0.0024788, -0.0056904, -0.0024788, -0.0020039, 0.0020039
9: -0.0033622, -0.0012407, -0.0033622, -0.0012407, -0.0013237, 0.0013237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011050, upper bound: 0.0010918
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011149, upper bound: 0.0011111
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0029049, -0.0017575, -0.0006667, 0.0007984
1: -0.0115002, -0.0086332, -0.0116822, -0.0087706, -0.0016918, 0.0020260
2: 0.0278952, 0.0296739, 0.0277823, 0.0295887, -0.0010496, 0.0012570
3: 0.0041538, 0.0074750, 0.0043129, 0.0076859, -0.0023471, 0.0019598
4: -0.0105907, -0.0076745, -0.0107758, -0.0078142, -0.0017208, 0.0020608
5: 0.0097267, 0.0108313, 0.0096566, 0.0107784, -0.0006518, 0.0007806
6: 0.0056598, 0.0098749, 0.0058617, 0.0101425, -0.0029787, 0.0024873
7: 0.9820197, 0.9849693, 0.9821610, 0.9851565, -0.0020844, 0.0017405
8: -0.0058419, -0.0026796, -0.0056904, -0.0024788, -0.0022348, 0.0018661
9: -0.0032296, -0.0011407, -0.0033622, -0.0012407, -0.0012327, 0.0014762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011203, upper bound: 0.0010969
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011288, upper bound: 0.0011220
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0029049, -0.0017575, -0.0007037, 0.0007316
1: -0.0116746, -0.0086990, -0.0116822, -0.0087706, -0.0017858, 0.0018565
2: 0.0277871, 0.0296331, 0.0277823, 0.0295887, -0.0011079, 0.0011518
3: 0.0042300, 0.0076770, 0.0043129, 0.0076859, -0.0021506, 0.0020688
4: -0.0107680, -0.0077414, -0.0107758, -0.0078142, -0.0018165, 0.0018883
5: 0.0096595, 0.0108060, 0.0096566, 0.0107784, -0.0006880, 0.0007153
6: 0.0057565, 0.0101312, 0.0058617, 0.0101425, -0.0027294, 0.0026256
7: 0.9820874, 0.9851486, 0.9821610, 0.9851565, -0.0019099, 0.0018372
8: -0.0057694, -0.0024872, -0.0056904, -0.0024788, -0.0020477, 0.0019698
9: -0.0033566, -0.0011886, -0.0033622, -0.0012407, -0.0013012, 0.0013526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011163, upper bound: 0.0011016
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011288, upper bound: 0.0011106
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028276, -0.0017262, -0.0007074, 0.0007040
1: -0.0115084, -0.0087008, -0.0114861, -0.0086910, -0.0017951, 0.0017865
2: 0.0278902, 0.0296320, 0.0279040, 0.0296381, -0.0011137, 0.0011084
3: 0.0042321, 0.0074845, 0.0042206, 0.0074587, -0.0020696, 0.0020795
4: -0.0105990, -0.0077432, -0.0105763, -0.0077332, -0.0018259, 0.0018172
5: 0.0097236, 0.0108053, 0.0097322, 0.0108091, -0.0006916, 0.0006883
6: 0.0057591, 0.0098870, 0.0057447, 0.0098541, -0.0026266, 0.0026392
7: 0.9820893, 0.9849777, 0.9820791, 0.9849547, -0.0018380, 0.0018468
8: -0.0057674, -0.0026705, -0.0057783, -0.0026951, -0.0019706, 0.0019801
9: -0.0032356, -0.0011899, -0.0032193, -0.0011827, -0.0013079, 0.0013017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011394, upper bound: 0.0011420
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011461
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028332, -0.0017034, -0.0007224, 0.0006965
1: -0.0115084, -0.0087008, -0.0115002, -0.0086332, -0.0018332, 0.0017674
2: 0.0278902, 0.0296320, 0.0278952, 0.0296739, -0.0011373, 0.0010965
3: 0.0042321, 0.0074845, 0.0041538, 0.0074750, -0.0020475, 0.0021237
4: -0.0105990, -0.0077432, -0.0105907, -0.0076745, -0.0018647, 0.0017978
5: 0.0097236, 0.0108053, 0.0097267, 0.0108313, -0.0007063, 0.0006809
6: 0.0057591, 0.0098870, 0.0056598, 0.0098749, -0.0025985, 0.0026952
7: 0.9820893, 0.9849777, 0.9820197, 0.9849693, -0.0018183, 0.0018860
8: -0.0057674, -0.0026705, -0.0058419, -0.0026796, -0.0019495, 0.0020221
9: -0.0032356, -0.0011899, -0.0032296, -0.0011407, -0.0013357, 0.0012878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011394, upper bound: 0.0011452
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011499
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028276, -0.0017262, -0.0007792, 0.0006737
1: -0.0116822, -0.0087706, -0.0114861, -0.0086910, -0.0019774, 0.0017095
2: 0.0277823, 0.0295887, 0.0279040, 0.0296381, -0.0012268, 0.0010606
3: 0.0043129, 0.0076859, 0.0042206, 0.0074587, -0.0019804, 0.0022907
4: -0.0107758, -0.0078142, -0.0105763, -0.0077332, -0.0020113, 0.0017389
5: 0.0096566, 0.0107784, 0.0097322, 0.0108091, -0.0007618, 0.0006586
6: 0.0058617, 0.0101425, 0.0057447, 0.0098541, -0.0025134, 0.0029072
7: 0.9821610, 0.9851565, 0.9820791, 0.9849547, -0.0017587, 0.0020343
8: -0.0056904, -0.0024788, -0.0057783, -0.0026951, -0.0018856, 0.0021811
9: -0.0033622, -0.0012407, -0.0032193, -0.0011827, -0.0014407, 0.0012456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011122, upper bound: 0.0011063
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011205
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028332, -0.0017034, -0.0007984, 0.0006667
1: -0.0116822, -0.0087706, -0.0115002, -0.0086332, -0.0020260, 0.0016918
2: 0.0277823, 0.0295887, 0.0278952, 0.0296739, -0.0012570, 0.0010496
3: 0.0043129, 0.0076859, 0.0041538, 0.0074750, -0.0019598, 0.0023471
4: -0.0107758, -0.0078142, -0.0105907, -0.0076745, -0.0020608, 0.0017208
5: 0.0096566, 0.0107784, 0.0097267, 0.0108313, -0.0007806, 0.0006518
6: 0.0058617, 0.0101425, 0.0056598, 0.0098749, -0.0024873, 0.0029787
7: 0.9821610, 0.9851565, 0.9820197, 0.9849693, -0.0017405, 0.0020844
8: -0.0056904, -0.0024788, -0.0058419, -0.0026796, -0.0018661, 0.0022348
9: -0.0033622, -0.0012407, -0.0032296, -0.0011407, -0.0014762, 0.0012327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011087, upper bound: 0.0011183
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011264
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028276, -0.0017262, -0.0007170, 0.0007377
1: -0.0115002, -0.0086332, -0.0114861, -0.0086910, -0.0018196, 0.0018719
2: 0.0278952, 0.0296739, 0.0279040, 0.0296381, -0.0011289, 0.0011614
3: 0.0041538, 0.0074750, 0.0042206, 0.0074587, -0.0021685, 0.0021079
4: -0.0105907, -0.0076745, -0.0105763, -0.0077332, -0.0018508, 0.0019041
5: 0.0097267, 0.0108313, 0.0097322, 0.0108091, -0.0007010, 0.0007212
6: 0.0056598, 0.0098749, 0.0057447, 0.0098541, -0.0027522, 0.0026752
7: 0.9820197, 0.9849693, 0.9820791, 0.9849547, -0.0019258, 0.0018720
8: -0.0058419, -0.0026796, -0.0057783, -0.0026951, -0.0020648, 0.0020071
9: -0.0032296, -0.0011407, -0.0032193, -0.0011827, -0.0013258, 0.0013639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011617, upper bound: 0.0011361
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011597, upper bound: 0.0011423
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028332, -0.0017034, -0.0007340, 0.0007340
1: -0.0115002, -0.0086332, -0.0115002, -0.0086332, -0.0018627, 0.0018627
2: 0.0278952, 0.0296739, 0.0278952, 0.0296739, -0.0011556, 0.0011556
3: 0.0041538, 0.0074750, 0.0041538, 0.0074750, -0.0021578, 0.0021578
4: -0.0105907, -0.0076745, -0.0105907, -0.0076745, -0.0018947, 0.0018947
5: 0.0097267, 0.0108313, 0.0097267, 0.0108313, -0.0007176, 0.0007176
6: 0.0056598, 0.0098749, 0.0056598, 0.0098749, -0.0027386, 0.0027386
7: 0.9820197, 0.9849693, 0.9820197, 0.9849693, -0.0019163, 0.0019163
8: -0.0058419, -0.0026796, -0.0058419, -0.0026796, -0.0020546, 0.0020546
9: -0.0032296, -0.0011407, -0.0032296, -0.0011407, -0.0013572, 0.0013572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011617, upper bound: 0.0011394
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011597, upper bound: 0.0011461
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028276, -0.0017262, -0.0007894, 0.0007072
1: -0.0116746, -0.0086990, -0.0114861, -0.0086910, -0.0020033, 0.0017946
2: 0.0277871, 0.0296331, 0.0279040, 0.0296381, -0.0012428, 0.0011134
3: 0.0042300, 0.0076770, 0.0042206, 0.0074587, -0.0020790, 0.0023207
4: -0.0107680, -0.0077414, -0.0105763, -0.0077332, -0.0020377, 0.0018255
5: 0.0096595, 0.0108060, 0.0097322, 0.0108091, -0.0007718, 0.0006914
6: 0.0057565, 0.0101312, 0.0057447, 0.0098541, -0.0026385, 0.0029453
7: 0.9820874, 0.9851486, 0.9820791, 0.9849547, -0.0018463, 0.0020610
8: -0.0057694, -0.0024872, -0.0057783, -0.0026951, -0.0019795, 0.0022097
9: -0.0033566, -0.0011886, -0.0032193, -0.0011827, -0.0014596, 0.0013076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011359, upper bound: 0.0011007
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011455, upper bound: 0.0011149
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028332, -0.0017034, -0.0008111, 0.0007050
1: -0.0116746, -0.0086990, -0.0115002, -0.0086332, -0.0020582, 0.0017891
2: 0.0277871, 0.0296331, 0.0278952, 0.0296739, -0.0012769, 0.0011100
3: 0.0042300, 0.0076770, 0.0041538, 0.0074750, -0.0020726, 0.0023844
4: -0.0107680, -0.0077414, -0.0105907, -0.0076745, -0.0020936, 0.0018198
5: 0.0096595, 0.0108060, 0.0097267, 0.0108313, -0.0007930, 0.0006893
6: 0.0057565, 0.0101312, 0.0056598, 0.0098749, -0.0026303, 0.0030261
7: 0.9820874, 0.9851486, 0.9820197, 0.9849693, -0.0018406, 0.0021175
8: -0.0057694, -0.0024872, -0.0058419, -0.0026796, -0.0019734, 0.0022703
9: -0.0033566, -0.0011886, -0.0032296, -0.0011407, -0.0014997, 0.0013035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011339, upper bound: 0.0011125
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011455, upper bound: 0.0011194
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0028953, -0.0017541, -0.0006788, 0.0007755
1: -0.0115084, -0.0087008, -0.0116579, -0.0087618, -0.0017226, 0.0019680
2: 0.0278902, 0.0296320, 0.0277974, 0.0295942, -0.0010687, 0.0012210
3: 0.0042321, 0.0074845, 0.0043027, 0.0076577, -0.0022799, 0.0019956
4: -0.0105990, -0.0077432, -0.0107511, -0.0078052, -0.0017522, 0.0020018
5: 0.0097236, 0.0108053, 0.0096660, 0.0107818, -0.0006637, 0.0007582
6: 0.0057591, 0.0098870, 0.0058487, 0.0101068, -0.0028934, 0.0025326
7: 0.9820893, 0.9849777, 0.9821519, 0.9851315, -0.0020247, 0.0017722
8: -0.0057674, -0.0026705, -0.0057002, -0.0025056, -0.0021708, 0.0019001
9: -0.0032356, -0.0011899, -0.0033445, -0.0012343, -0.0012551, 0.0014339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011009, upper bound: 0.0011138
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011100, upper bound: 0.0011322
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0028953, -0.0017541, -0.0007146, 0.0007102
1: -0.0116822, -0.0087706, -0.0116579, -0.0087618, -0.0018133, 0.0018022
2: 0.0277823, 0.0295887, 0.0277974, 0.0295942, -0.0011250, 0.0011181
3: 0.0043129, 0.0076859, 0.0043027, 0.0076577, -0.0020877, 0.0021006
4: -0.0107758, -0.0078142, -0.0107511, -0.0078052, -0.0018444, 0.0018331
5: 0.0096566, 0.0107784, 0.0096660, 0.0107818, -0.0006986, 0.0006943
6: 0.0058617, 0.0101425, 0.0058487, 0.0101068, -0.0026496, 0.0026659
7: 0.9821610, 0.9851565, 0.9821519, 0.9851315, -0.0018541, 0.0018655
8: -0.0056904, -0.0024788, -0.0057002, -0.0025056, -0.0019878, 0.0020001
9: -0.0033622, -0.0012407, -0.0033445, -0.0012343, -0.0013212, 0.0013131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011009, upper bound: 0.0011062
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011100, upper bound: 0.0011210
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028364, -0.0017300, -0.0029019, -0.0017293, -0.0006942, 0.0007694
1: -0.0115084, -0.0087008, -0.0116746, -0.0086990, -0.0017616, 0.0019524
2: 0.0278902, 0.0296320, 0.0277871, 0.0296331, -0.0010929, 0.0012113
3: 0.0042321, 0.0074845, 0.0042300, 0.0076770, -0.0022617, 0.0020407
4: -0.0105990, -0.0077432, -0.0107680, -0.0077414, -0.0017918, 0.0019859
5: 0.0097236, 0.0108053, 0.0096595, 0.0108060, -0.0006787, 0.0007522
6: 0.0057591, 0.0098870, 0.0057565, 0.0101312, -0.0028704, 0.0025899
7: 0.9820893, 0.9849777, 0.9820874, 0.9851486, -0.0020086, 0.0018123
8: -0.0057674, -0.0026705, -0.0057694, -0.0024872, -0.0021535, 0.0019431
9: -0.0032356, -0.0011899, -0.0033566, -0.0011886, -0.0012835, 0.0014225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011046, upper bound: 0.0011168
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011148, upper bound: 0.0011383
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0029049, -0.0017575, -0.0029019, -0.0017293, -0.0007316, 0.0007037
1: -0.0116822, -0.0087706, -0.0116746, -0.0086990, -0.0018565, 0.0017858
2: 0.0277823, 0.0295887, 0.0277871, 0.0296331, -0.0011518, 0.0011079
3: 0.0043129, 0.0076859, 0.0042300, 0.0076770, -0.0020688, 0.0021506
4: -0.0107758, -0.0078142, -0.0107680, -0.0077414, -0.0018883, 0.0018165
5: 0.0096566, 0.0107784, 0.0096595, 0.0108060, -0.0007153, 0.0006880
6: 0.0058617, 0.0101425, 0.0057565, 0.0101312, -0.0026256, 0.0027294
7: 0.9821610, 0.9851565, 0.9820874, 0.9851486, -0.0018372, 0.0019099
8: -0.0056904, -0.0024788, -0.0057694, -0.0024872, -0.0019698, 0.0020477
9: -0.0033622, -0.0012407, -0.0033566, -0.0011886, -0.0013526, 0.0013012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011046, upper bound: 0.0011103
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011148, upper bound: 0.0011264
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0028953, -0.0017541, -0.0006874, 0.0008133
1: -0.0115002, -0.0086332, -0.0116579, -0.0087618, -0.0017443, 0.0020638
2: 0.0278952, 0.0296739, 0.0277974, 0.0295942, -0.0010822, 0.0012804
3: 0.0041538, 0.0074750, 0.0043027, 0.0076577, -0.0023908, 0.0020207
4: -0.0105907, -0.0076745, -0.0107511, -0.0078052, -0.0017743, 0.0020992
5: 0.0097267, 0.0108313, 0.0096660, 0.0107818, -0.0006721, 0.0007951
6: 0.0056598, 0.0098749, 0.0058487, 0.0101068, -0.0030343, 0.0025646
7: 0.9820197, 0.9849693, 0.9821519, 0.9851315, -0.0021232, 0.0017946
8: -0.0058419, -0.0026796, -0.0057002, -0.0025056, -0.0022764, 0.0019241
9: -0.0032296, -0.0011407, -0.0033445, -0.0012343, -0.0012710, 0.0015037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011240, upper bound: 0.0011085
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011267
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0028953, -0.0017541, -0.0007231, 0.0007448
1: -0.0116746, -0.0086990, -0.0116579, -0.0087618, -0.0018350, 0.0018900
2: 0.0277871, 0.0296331, 0.0277974, 0.0295942, -0.0011384, 0.0011726
3: 0.0042300, 0.0076770, 0.0043027, 0.0076577, -0.0021895, 0.0021257
4: -0.0107680, -0.0077414, -0.0107511, -0.0078052, -0.0018665, 0.0019225
5: 0.0096595, 0.0108060, 0.0096660, 0.0107818, -0.0007070, 0.0007282
6: 0.0057565, 0.0101312, 0.0058487, 0.0101068, -0.0027788, 0.0026978
7: 0.9820874, 0.9851486, 0.9821519, 0.9851315, -0.0019445, 0.0018878
8: -0.0057694, -0.0024872, -0.0057002, -0.0025056, -0.0020848, 0.0020240
9: -0.0033566, -0.0011886, -0.0033445, -0.0012343, -0.0013370, 0.0013771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011240, upper bound: 0.0011006
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011148
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028332, -0.0017034, -0.0029019, -0.0017293, -0.0007050, 0.0008111
1: -0.0115002, -0.0086332, -0.0116746, -0.0086990, -0.0017891, 0.0020582
2: 0.0278952, 0.0296739, 0.0277871, 0.0296331, -0.0011100, 0.0012769
3: 0.0041538, 0.0074750, 0.0042300, 0.0076770, -0.0023844, 0.0020726
4: -0.0105907, -0.0076745, -0.0107680, -0.0077414, -0.0018198, 0.0020936
5: 0.0097267, 0.0108313, 0.0096595, 0.0108060, -0.0006893, 0.0007930
6: 0.0056598, 0.0098749, 0.0057565, 0.0101312, -0.0030261, 0.0026303
7: 0.9820197, 0.9849693, 0.9820874, 0.9851486, -0.0021175, 0.0018406
8: -0.0058419, -0.0026796, -0.0057694, -0.0024872, -0.0022703, 0.0019734
9: -0.0032296, -0.0011407, -0.0033566, -0.0011886, -0.0013035, 0.0014997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011262, upper bound: 0.0011105
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011339, upper bound: 0.0011312
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0029019, -0.0017293, -0.0029019, -0.0017293, -0.0007415, 0.0007415
1: -0.0116746, -0.0086990, -0.0116746, -0.0086990, -0.0018817, 0.0018817
2: 0.0277871, 0.0296331, 0.0277871, 0.0296331, -0.0011674, 0.0011674
3: 0.0042300, 0.0076770, 0.0042300, 0.0076770, -0.0021798, 0.0021798
4: -0.0107680, -0.0077414, -0.0107680, -0.0077414, -0.0019140, 0.0019140
5: 0.0096595, 0.0108060, 0.0096595, 0.0108060, -0.0007250, 0.0007250
6: 0.0057565, 0.0101312, 0.0057565, 0.0101312, -0.0027665, 0.0027665
7: 0.9820874, 0.9851486, 0.9820874, 0.9851486, -0.0019359, 0.0019359
8: -0.0057694, -0.0024872, -0.0057694, -0.0024872, -0.0020755, 0.0020755
9: -0.0033566, -0.0011886, -0.0033566, -0.0011886, -0.0013710, 0.0013710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011262, upper bound: 0.0011031
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011339, upper bound: 0.0011193
time: 0.79 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.37 seconds
IS_A1_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011354, upper bound: 0.0011256
IS_A1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011307, upper bound: 0.0011314
IS_A1_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011246, upper bound: 0.0011404
IS_A1_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011307, upper bound: 0.0011369
IS_A1_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011420, upper bound: 0.0011345
IS_A1_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011463, upper bound: 0.0011311
IS_A1_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011420, upper bound: 0.0011395
IS_A1_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011463, upper bound: 0.0011363
IS_A1_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0010979
IS_A1_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011166, upper bound: 0.0011061
IS_A1_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0011016
IS_A1_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011166, upper bound: 0.0011101
IS_A1_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011180, upper bound: 0.0010975
IS_A1_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011321, upper bound: 0.0011057
IS_A1_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011180, upper bound: 0.0011013
IS_A1_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011321, upper bound: 0.0011100
IS_A1_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010973, upper bound: 0.0010944
IS_A1_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011059, upper bound: 0.0011171
IS_A1_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010973, upper bound: 0.0010877
IS_A1_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011059, upper bound: 0.0011059
IS_A1_B1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010874, upper bound: 0.0011083
IS_A1_B1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011056, upper bound: 0.0011230
IS_A1_B1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010874, upper bound: 0.0011011
IS_A1_B1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011056, upper bound: 0.0011101
IS_A1_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011142, upper bound: 0.0010944
IS_A1_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011172
IS_A1_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011062, upper bound: 0.0010968
IS_A1_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011057
IS_A1_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011062, upper bound: 0.0011083
IS_A1_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011230
IS_A1_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011062, upper bound: 0.0011009
IS_A1_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0011100
IS_A1_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011340, upper bound: 0.0011420
IS_A1_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011303, upper bound: 0.0011461
IS_A1_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011241, upper bound: 0.0011521
IS_A1_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011303, upper bound: 0.0011500
IS_A1_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0011137
IS_A1_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011165, upper bound: 0.0011205
IS_A1_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0011168
IS_A1_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011165, upper bound: 0.0011237
IS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011602, upper bound: 0.0011370
IS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011557, upper bound: 0.0011430
IS_A1_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011509, upper bound: 0.0011517
IS_A1_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011557, upper bound: 0.0011486
IS_A1_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011255, upper bound: 0.0011100
IS_A1_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011391, upper bound: 0.0011153
IS_A1_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011255, upper bound: 0.0011145
IS_A1_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011391, upper bound: 0.0011198
IS_A1_B2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010968, upper bound: 0.0011142
IS_A1_B2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011057, upper bound: 0.0011322
IS_A1_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010968, upper bound: 0.0011062
IS_A1_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011057, upper bound: 0.0011210
IS_A1_B2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010968, upper bound: 0.0011218
IS_A1_B2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011052, upper bound: 0.0011384
IS_A1_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0010873, upper bound: 0.0011167
IS_A1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011052, upper bound: 0.0011237
IS_A1_B2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011223, upper bound: 0.0011094
IS_A1_B2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011274
IS_A1_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011223, upper bound: 0.0011010
IS_A1_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011153
IS_A1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011138, upper bound: 0.0011226
IS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011334
IS_A1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011138, upper bound: 0.0011144
IS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011277, upper bound: 0.0011198
IS_A2_B1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011241
IS_A2_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011369, upper bound: 0.0011303
IS_A2_B1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011404, upper bound: 0.0011241
IS_A2_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011369, upper bound: 0.0011303
IS_A2_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011429, upper bound: 0.0011274
IS_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011405, upper bound: 0.0011342
IS_A2_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011351, upper bound: 0.0011371
IS_A2_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011405, upper bound: 0.0011342
IS_A2_B1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011122, upper bound: 0.0010874
IS_A2_B1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011054
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011087, upper bound: 0.0010976
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011053
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011116, upper bound: 0.0011023
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011108
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011116, upper bound: 0.0011023
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011282, upper bound: 0.0011108
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011011, upper bound: 0.0010937
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011101, upper bound: 0.0011166
IS_A2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011011, upper bound: 0.0010874
IS_A2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011101, upper bound: 0.0011056
IS_A2_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011167, upper bound: 0.0010937
IS_A2_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0011165
IS_A2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011167, upper bound: 0.0010873
IS_A2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0011052
IS_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011050, upper bound: 0.0010969
IS_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011149, upper bound: 0.0011220
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011050, upper bound: 0.0010918
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011149, upper bound: 0.0011111
IS_A2_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011203, upper bound: 0.0010969
IS_A2_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011288, upper bound: 0.0011220
IS_A2_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011163, upper bound: 0.0011016
IS_A2_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011288, upper bound: 0.0011106
IS_A2_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011394, upper bound: 0.0011420
IS_A2_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011461
IS_A2_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011394, upper bound: 0.0011452
IS_A2_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011363, upper bound: 0.0011499
IS_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011122, upper bound: 0.0011063
IS_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011205
IS_A2_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011087, upper bound: 0.0011183
IS_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011232, upper bound: 0.0011264
IS_A2_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011617, upper bound: 0.0011361
IS_A2_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011597, upper bound: 0.0011423
IS_A2_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011617, upper bound: 0.0011394
IS_A2_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011597, upper bound: 0.0011461
IS_A2_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011359, upper bound: 0.0011007
IS_A2_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011455, upper bound: 0.0011149
IS_A2_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011339, upper bound: 0.0011125
IS_A2_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011455, upper bound: 0.0011194
IS_A2_B2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011009, upper bound: 0.0011138
IS_A2_B2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011100, upper bound: 0.0011322
IS_A2_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011009, upper bound: 0.0011062
IS_A2_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011100, upper bound: 0.0011210
IS_A2_B2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011046, upper bound: 0.0011168
IS_A2_B2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011148, upper bound: 0.0011383
IS_A2_B2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011046, upper bound: 0.0011103
IS_A2_B2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011148, upper bound: 0.0011264
IS_A2_B2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011240, upper bound: 0.0011085
IS_A2_B2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011267
IS_A2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011240, upper bound: 0.0011006
IS_A2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011299, upper bound: 0.0011148
IS_A2_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011262, upper bound: 0.0011105
IS_A2_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011339, upper bound: 0.0011312
IS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011262, upper bound: 0.0011031
IS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 7, lower bound: -0.0011339, upper bound: 0.0011193

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028345, -0.0017672, -0.0028301, -0.0017581, -0.0006806, 0.0006661
1: -0.0115036, -0.0087950, -0.0114924, -0.0087719, -0.0017271, 0.0016903
2: 0.0278931, 0.0295736, 0.0279001, 0.0295879, -0.0010715, 0.0010486
3: 0.0043412, 0.0074790, 0.0043144, 0.0074660, -0.0019581, 0.0020008
4: -0.0105941, -0.0078390, -0.0105827, -0.0078155, -0.0017568, 0.0017193
5: 0.0097254, 0.0107690, 0.0097297, 0.0107779, -0.0006654, 0.0006512
6: 0.0058976, 0.0098799, 0.0058637, 0.0098634, -0.0024851, 0.0025392
7: 0.9821861, 0.9849727, 0.9821624, 0.9849612, -0.0017389, 0.0017768
8: -0.0056635, -0.0026758, -0.0056890, -0.0026882, -0.0018644, 0.0019050
9: -0.0032321, -0.0012585, -0.0032239, -0.0012417, -0.0012584, 0.0012315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011019, upper bound: 0.0011165
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011267, upper bound: 0.0011165
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028300, -0.0017655, -0.0028302, -0.0017563, -0.0006774, 0.0006729
1: -0.0114920, -0.0087908, -0.0114926, -0.0087675, -0.0017191, 0.0017075
2: 0.0279003, 0.0295762, 0.0279000, 0.0295907, -0.0010665, 0.0010593
3: 0.0043363, 0.0074655, 0.0043093, 0.0074662, -0.0019780, 0.0019915
4: -0.0105823, -0.0078347, -0.0105829, -0.0078110, -0.0017486, 0.0017368
5: 0.0097299, 0.0107706, 0.0097297, 0.0107796, -0.0006623, 0.0006578
6: 0.0058914, 0.0098628, 0.0058571, 0.0098637, -0.0025104, 0.0025275
7: 0.9821817, 0.9849609, 0.9821578, 0.9849614, -0.0017566, 0.0017686
8: -0.0056682, -0.0026886, -0.0056939, -0.0026880, -0.0018834, 0.0018962
9: -0.0032236, -0.0012555, -0.0032240, -0.0012385, -0.0012526, 0.0012441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010961, upper bound: 0.0011220
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011220, upper bound: 0.0011221
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017581, -0.0028403, -0.0017445, -0.0006950, 0.0006895
1: -0.0114924, -0.0087720, -0.0115181, -0.0087375, -0.0017636, 0.0017496
2: 0.0279001, 0.0295878, 0.0278841, 0.0296093, -0.0010942, 0.0010855
3: 0.0043145, 0.0074660, 0.0042745, 0.0074958, -0.0020269, 0.0020431
4: -0.0105827, -0.0078156, -0.0106089, -0.0077805, -0.0017939, 0.0017797
5: 0.0097297, 0.0107778, 0.0097198, 0.0107911, -0.0006795, 0.0006741
6: 0.0058638, 0.0098634, 0.0058131, 0.0099012, -0.0025723, 0.0025929
7: 0.9821625, 0.9849612, 0.9821270, 0.9849877, -0.0018000, 0.0018144
8: -0.0056888, -0.0026882, -0.0057269, -0.0026598, -0.0019299, 0.0019453
9: -0.0032239, -0.0012418, -0.0032426, -0.0012166, -0.0012850, 0.0012748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010875, upper bound: 0.0011310
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011158, upper bound: 0.0011310
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028302, -0.0017564, -0.0028361, -0.0017420, -0.0007012, 0.0006870
1: -0.0114926, -0.0087676, -0.0115076, -0.0087312, -0.0017793, 0.0017433
2: 0.0279000, 0.0295906, 0.0278907, 0.0296132, -0.0011039, 0.0010815
3: 0.0043094, 0.0074662, 0.0042672, 0.0074836, -0.0020195, 0.0020613
4: -0.0105829, -0.0078111, -0.0105982, -0.0077741, -0.0018099, 0.0017732
5: 0.0097297, 0.0107795, 0.0097239, 0.0107936, -0.0006855, 0.0006716
6: 0.0058573, 0.0098637, 0.0058038, 0.0098858, -0.0025630, 0.0026160
7: 0.9821579, 0.9849614, 0.9821205, 0.9849768, -0.0017934, 0.0018306
8: -0.0056937, -0.0026880, -0.0057339, -0.0026714, -0.0019229, 0.0019627
9: -0.0032240, -0.0012386, -0.0032350, -0.0012120, -0.0012965, 0.0012702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010959, upper bound: 0.0011271
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011215, upper bound: 0.0011272
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017303, -0.0028345, -0.0017672, -0.0006563, 0.0006952
1: -0.0114857, -0.0087016, -0.0115036, -0.0087950, -0.0016655, 0.0017643
2: 0.0279043, 0.0296315, 0.0278931, 0.0295736, -0.0010333, 0.0010946
3: 0.0042329, 0.0074582, 0.0043412, 0.0074790, -0.0020438, 0.0019294
4: -0.0105759, -0.0077440, -0.0105941, -0.0078390, -0.0016941, 0.0017946
5: 0.0097323, 0.0108050, 0.0097254, 0.0107690, -0.0006417, 0.0006797
6: 0.0057603, 0.0098535, 0.0058976, 0.0098799, -0.0025939, 0.0024486
7: 0.9820900, 0.9849543, 0.9821861, 0.9849727, -0.0018151, 0.0017134
8: -0.0057665, -0.0026956, -0.0056635, -0.0026758, -0.0019460, 0.0018371
9: -0.0032190, -0.0011905, -0.0032321, -0.0012585, -0.0012135, 0.0012855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011330, upper bound: 0.0011020
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011330, upper bound: 0.0011253
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017289, -0.0028300, -0.0017655, -0.0006628, 0.0006921
1: -0.0114859, -0.0086980, -0.0114920, -0.0087908, -0.0016819, 0.0017562
2: 0.0279041, 0.0296337, 0.0279003, 0.0295762, -0.0010434, 0.0010896
3: 0.0042288, 0.0074585, 0.0043363, 0.0074655, -0.0020345, 0.0019484
4: -0.0105761, -0.0077404, -0.0105823, -0.0078347, -0.0017108, 0.0017864
5: 0.0097322, 0.0108063, 0.0097299, 0.0107706, -0.0006480, 0.0006766
6: 0.0057550, 0.0098539, 0.0058914, 0.0098628, -0.0025821, 0.0024727
7: 0.9820864, 0.9849546, 0.9821817, 0.9849609, -0.0018068, 0.0017303
8: -0.0057705, -0.0026953, -0.0056682, -0.0026886, -0.0019372, 0.0018552
9: -0.0032192, -0.0011879, -0.0032236, -0.0012555, -0.0012254, 0.0012796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011369, upper bound: 0.0010961
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011370, upper bound: 0.0011216
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017303, -0.0028403, -0.0017445, -0.0006852, 0.0007042
1: -0.0114857, -0.0087016, -0.0115181, -0.0087375, -0.0017388, 0.0017869
2: 0.0279043, 0.0296315, 0.0278841, 0.0296093, -0.0010788, 0.0011086
3: 0.0042329, 0.0074582, 0.0042745, 0.0074958, -0.0020700, 0.0020144
4: -0.0105759, -0.0077440, -0.0106089, -0.0077805, -0.0017687, 0.0018176
5: 0.0097323, 0.0108050, 0.0097198, 0.0107911, -0.0006699, 0.0006885
6: 0.0057603, 0.0098535, 0.0058131, 0.0099012, -0.0026272, 0.0025565
7: 0.9820900, 0.9849543, 0.9821270, 0.9849877, -0.0018384, 0.0017889
8: -0.0057665, -0.0026956, -0.0057269, -0.0026598, -0.0019710, 0.0019180
9: -0.0032190, -0.0011905, -0.0032426, -0.0012166, -0.0012669, 0.0013020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011330, upper bound: 0.0011071
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011330, upper bound: 0.0011297
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017289, -0.0028361, -0.0017420, -0.0006911, 0.0007017
1: -0.0114859, -0.0086980, -0.0115076, -0.0087312, -0.0017538, 0.0017806
2: 0.0279041, 0.0296337, 0.0278907, 0.0296132, -0.0010880, 0.0011047
3: 0.0042288, 0.0074585, 0.0042672, 0.0074836, -0.0020627, 0.0020316
4: -0.0105761, -0.0077404, -0.0105982, -0.0077741, -0.0017839, 0.0018111
5: 0.0097322, 0.0108063, 0.0097239, 0.0107936, -0.0006757, 0.0006860
6: 0.0057550, 0.0098539, 0.0058038, 0.0098858, -0.0026178, 0.0025784
7: 0.9820864, 0.9849546, 0.9821205, 0.9849768, -0.0018318, 0.0018043
8: -0.0057705, -0.0026953, -0.0057339, -0.0026714, -0.0019640, 0.0019344
9: -0.0032192, -0.0011879, -0.0032350, -0.0012120, -0.0012778, 0.0012973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011052, upper bound: 0.0011263
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011369, upper bound: 0.0011264
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028973, -0.0017862, -0.0028345, -0.0017672, -0.0007402, 0.0006482
1: -0.0116629, -0.0088433, -0.0115036, -0.0087950, -0.0018784, 0.0016449
2: 0.0277943, 0.0295436, 0.0278931, 0.0295736, -0.0011654, 0.0010205
3: 0.0043971, 0.0076635, 0.0043412, 0.0074790, -0.0019056, 0.0021760
4: -0.0107562, -0.0078881, -0.0105941, -0.0078390, -0.0019106, 0.0016732
5: 0.0096640, 0.0107504, 0.0097254, 0.0107690, -0.0007237, 0.0006338
6: 0.0059686, 0.0101141, 0.0058976, 0.0098799, -0.0024184, 0.0027617
7: 0.9822357, 0.9851366, 0.9821861, 0.9849727, -0.0016923, 0.0019325
8: -0.0056103, -0.0025001, -0.0056635, -0.0026758, -0.0018144, 0.0020719
9: -0.0033482, -0.0012937, -0.0032321, -0.0012585, -0.0013686, 0.0011985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010931, upper bound: 0.0010637
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010931, upper bound: 0.0010897
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028974, -0.0017848, -0.0028300, -0.0017655, -0.0007436, 0.0006460
1: -0.0116632, -0.0088398, -0.0114920, -0.0087908, -0.0018870, 0.0016393
2: 0.0277941, 0.0295458, 0.0279003, 0.0295762, -0.0011707, 0.0010170
3: 0.0043930, 0.0076638, 0.0043363, 0.0074655, -0.0018991, 0.0021860
4: -0.0107564, -0.0078845, -0.0105823, -0.0078347, -0.0019194, 0.0016675
5: 0.0096639, 0.0107517, 0.0097299, 0.0107706, -0.0007270, 0.0006316
6: 0.0059634, 0.0101145, 0.0058914, 0.0098628, -0.0024102, 0.0027743
7: 0.9822322, 0.9851369, 0.9821817, 0.9849609, -0.0016865, 0.0019413
8: -0.0056141, -0.0024998, -0.0056682, -0.0026886, -0.0018082, 0.0020814
9: -0.0033483, -0.0012912, -0.0032236, -0.0012555, -0.0013749, 0.0011944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011085, upper bound: 0.0010714
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011085, upper bound: 0.0010966
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028973, -0.0017862, -0.0028403, -0.0017445, -0.0007691, 0.0006571
1: -0.0116629, -0.0088433, -0.0115181, -0.0087375, -0.0019517, 0.0016676
2: 0.0277943, 0.0295436, 0.0278841, 0.0296093, -0.0012109, 0.0010346
3: 0.0043971, 0.0076635, 0.0042745, 0.0074958, -0.0019318, 0.0022610
4: -0.0107562, -0.0078881, -0.0106089, -0.0077805, -0.0019853, 0.0016962
5: 0.0096640, 0.0107504, 0.0097198, 0.0107911, -0.0007520, 0.0006425
6: 0.0059686, 0.0101141, 0.0058131, 0.0099012, -0.0024517, 0.0028695
7: 0.9822357, 0.9851366, 0.9821270, 0.9849877, -0.0017156, 0.0020079
8: -0.0056103, -0.0025001, -0.0057269, -0.0026598, -0.0018394, 0.0021528
9: -0.0033482, -0.0012937, -0.0032426, -0.0012166, -0.0014221, 0.0012150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010595, upper bound: 0.0010929
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010920, upper bound: 0.0010928
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028974, -0.0017848, -0.0028361, -0.0017420, -0.0007719, 0.0006556
1: -0.0116632, -0.0088398, -0.0115076, -0.0087312, -0.0019589, 0.0016636
2: 0.0277941, 0.0295458, 0.0278907, 0.0296132, -0.0012153, 0.0010321
3: 0.0043930, 0.0076638, 0.0042672, 0.0074836, -0.0019272, 0.0022693
4: -0.0107564, -0.0078845, -0.0105982, -0.0077741, -0.0019925, 0.0016922
5: 0.0096639, 0.0107517, 0.0097239, 0.0107936, -0.0007547, 0.0006410
6: 0.0059634, 0.0101145, 0.0058038, 0.0098858, -0.0024459, 0.0028800
7: 0.9822322, 0.9851369, 0.9821205, 0.9849768, -0.0017115, 0.0020153
8: -0.0056141, -0.0024998, -0.0057339, -0.0026714, -0.0018350, 0.0021607
9: -0.0033483, -0.0012912, -0.0032350, -0.0012120, -0.0014273, 0.0012121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0011005
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011080, upper bound: 0.0011006
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0028345, -0.0017672, -0.0007279, 0.0006659
1: -0.0116575, -0.0087728, -0.0115036, -0.0087950, -0.0018473, 0.0016898
2: 0.0277977, 0.0295874, 0.0278931, 0.0295736, -0.0011461, 0.0010484
3: 0.0043154, 0.0076572, 0.0043412, 0.0074790, -0.0019576, 0.0021400
4: -0.0107507, -0.0078164, -0.0105941, -0.0078390, -0.0018790, 0.0017188
5: 0.0096661, 0.0107775, 0.0097254, 0.0107690, -0.0007117, 0.0006510
6: 0.0058649, 0.0101061, 0.0058976, 0.0098799, -0.0024844, 0.0027159
7: 0.9821633, 0.9851310, 0.9821861, 0.9849727, -0.0017385, 0.0019005
8: -0.0056880, -0.0025061, -0.0056635, -0.0026758, -0.0018639, 0.0020376
9: -0.0033442, -0.0012423, -0.0032321, -0.0012585, -0.0013459, 0.0012312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011107, upper bound: 0.0010637
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011107, upper bound: 0.0010892
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017567, -0.0028300, -0.0017655, -0.0007309, 0.0006639
1: -0.0116577, -0.0087685, -0.0114920, -0.0087908, -0.0018548, 0.0016846
2: 0.0277975, 0.0295900, 0.0279003, 0.0295762, -0.0011507, 0.0010452
3: 0.0043104, 0.0076575, 0.0043363, 0.0074655, -0.0019516, 0.0021487
4: -0.0107509, -0.0078120, -0.0105823, -0.0078347, -0.0018867, 0.0017136
5: 0.0096660, 0.0107792, 0.0097299, 0.0107706, -0.0007146, 0.0006491
6: 0.0058586, 0.0101065, 0.0058914, 0.0098628, -0.0024768, 0.0027270
7: 0.9821588, 0.9851314, 0.9821817, 0.9849609, -0.0017332, 0.0019082
8: -0.0056928, -0.0025058, -0.0056682, -0.0026886, -0.0018582, 0.0020459
9: -0.0033444, -0.0012392, -0.0032236, -0.0012555, -0.0013515, 0.0012275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0010714
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0010963
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0028403, -0.0017445, -0.0007569, 0.0006748
1: -0.0116575, -0.0087728, -0.0115181, -0.0087375, -0.0019206, 0.0017125
2: 0.0277977, 0.0295874, 0.0278841, 0.0296093, -0.0011916, 0.0010624
3: 0.0043154, 0.0076572, 0.0042745, 0.0074958, -0.0019838, 0.0022250
4: -0.0107507, -0.0078164, -0.0106089, -0.0077805, -0.0019536, 0.0017419
5: 0.0096661, 0.0107775, 0.0097198, 0.0107911, -0.0007400, 0.0006598
6: 0.0058649, 0.0101061, 0.0058131, 0.0099012, -0.0025177, 0.0028238
7: 0.9821633, 0.9851310, 0.9821270, 0.9849877, -0.0017618, 0.0019759
8: -0.0056880, -0.0025061, -0.0057269, -0.0026598, -0.0018889, 0.0021185
9: -0.0033442, -0.0012423, -0.0032426, -0.0012166, -0.0013994, 0.0012477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011105, upper bound: 0.0010699
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011105, upper bound: 0.0010926
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017567, -0.0028361, -0.0017420, -0.0007593, 0.0006734
1: -0.0116577, -0.0087685, -0.0115076, -0.0087312, -0.0019267, 0.0017090
2: 0.0277975, 0.0295900, 0.0278907, 0.0296132, -0.0011953, 0.0010603
3: 0.0043104, 0.0076575, 0.0042672, 0.0074836, -0.0019798, 0.0022320
4: -0.0107509, -0.0078120, -0.0105982, -0.0077741, -0.0019598, 0.0017383
5: 0.0096660, 0.0107792, 0.0097239, 0.0107936, -0.0007423, 0.0006584
6: 0.0058586, 0.0101065, 0.0058038, 0.0098858, -0.0025126, 0.0028327
7: 0.9821588, 0.9851314, 0.9821205, 0.9849768, -0.0017582, 0.0019822
8: -0.0056928, -0.0025058, -0.0057339, -0.0026714, -0.0018850, 0.0021252
9: -0.0033444, -0.0012392, -0.0032350, -0.0012120, -0.0014038, 0.0012452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010871, upper bound: 0.0011001
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0011002
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028345, -0.0017672, -0.0028973, -0.0017860, -0.0006485, 0.0007402
1: -0.0115036, -0.0087950, -0.0116629, -0.0088427, -0.0016455, 0.0018784
2: 0.0278931, 0.0295736, 0.0277943, 0.0295440, -0.0010209, 0.0011654
3: 0.0043412, 0.0074790, 0.0043964, 0.0076635, -0.0021760, 0.0019063
4: -0.0105941, -0.0078390, -0.0107562, -0.0078875, -0.0016738, 0.0019106
5: 0.0097254, 0.0107690, 0.0096640, 0.0107506, -0.0006340, 0.0007237
6: 0.0058976, 0.0098799, 0.0059677, 0.0101141, -0.0027617, 0.0024193
7: 0.9821861, 0.9849727, 0.9822352, 0.9851366, -0.0019325, 0.0016929
8: -0.0056635, -0.0026758, -0.0056109, -0.0025001, -0.0020719, 0.0018151
9: -0.0032321, -0.0012585, -0.0033482, -0.0012933, -0.0011990, 0.0013686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010637, upper bound: 0.0010931
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010897, upper bound: 0.0010931
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028300, -0.0017655, -0.0028974, -0.0017846, -0.0006463, 0.0007436
1: -0.0114920, -0.0087908, -0.0116632, -0.0088392, -0.0016400, 0.0018870
2: 0.0279003, 0.0295762, 0.0277941, 0.0295462, -0.0010174, 0.0011707
3: 0.0043363, 0.0074655, 0.0043923, 0.0076638, -0.0021860, 0.0018998
4: -0.0105823, -0.0078347, -0.0107564, -0.0078839, -0.0016681, 0.0019194
5: 0.0097299, 0.0107706, 0.0096639, 0.0107520, -0.0006318, 0.0007270
6: 0.0058914, 0.0098628, 0.0059626, 0.0101145, -0.0027743, 0.0024111
7: 0.9821817, 0.9849609, 0.9822316, 0.9851369, -0.0019413, 0.0016872
8: -0.0056682, -0.0026886, -0.0056148, -0.0024998, -0.0020814, 0.0018089
9: -0.0032236, -0.0012555, -0.0033483, -0.0012907, -0.0011949, 0.0013749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010714, upper bound: 0.0011085
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010967, upper bound: 0.0011085
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0029031, -0.0017953, -0.0028973, -0.0017860, -0.0006859, 0.0006713
1: -0.0116775, -0.0088664, -0.0116629, -0.0088427, -0.0017405, 0.0017036
2: 0.0277853, 0.0295293, 0.0277943, 0.0295440, -0.0010798, 0.0010569
3: 0.0044238, 0.0076804, 0.0043964, 0.0076635, -0.0019735, 0.0020163
4: -0.0107710, -0.0079116, -0.0107562, -0.0078875, -0.0017704, 0.0017328
5: 0.0096584, 0.0107415, 0.0096640, 0.0107506, -0.0006706, 0.0006564
6: 0.0060025, 0.0101355, 0.0059677, 0.0101141, -0.0025047, 0.0025589
7: 0.9822596, 0.9851516, 0.9822352, 0.9851366, -0.0017526, 0.0017906
8: -0.0055848, -0.0024840, -0.0056109, -0.0025001, -0.0018791, 0.0019198
9: -0.0033588, -0.0013105, -0.0033482, -0.0012933, -0.0012681, 0.0012413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010834, upper bound: 0.0010517
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010823, upper bound: 0.0010658
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0028972, -0.0017946, -0.0028974, -0.0017846, -0.0006832, 0.0006782
1: -0.0116625, -0.0088647, -0.0116632, -0.0088392, -0.0017338, 0.0017210
2: 0.0277945, 0.0295303, 0.0277941, 0.0295462, -0.0010756, 0.0010677
3: 0.0044220, 0.0076631, 0.0043923, 0.0076638, -0.0019937, 0.0020085
4: -0.0107558, -0.0079100, -0.0107564, -0.0078839, -0.0017635, 0.0017506
5: 0.0096642, 0.0107421, 0.0096639, 0.0107520, -0.0006680, 0.0006631
6: 0.0060002, 0.0101136, 0.0059626, 0.0101145, -0.0025303, 0.0025491
7: 0.9822579, 0.9851363, 0.9822316, 0.9851369, -0.0017706, 0.0017837
8: -0.0055866, -0.0025005, -0.0056148, -0.0024998, -0.0018983, 0.0019124
9: -0.0033479, -0.0013094, -0.0033483, -0.0012907, -0.0012633, 0.0012540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0010966
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011085, upper bound: 0.0010966
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017581, -0.0029096, -0.0017713, -0.0006630, 0.0007586
1: -0.0114924, -0.0087720, -0.0116942, -0.0088054, -0.0016826, 0.0019249
2: 0.0279001, 0.0295878, 0.0277749, 0.0295671, -0.0010439, 0.0011942
3: 0.0043145, 0.0074660, 0.0043532, 0.0076997, -0.0022299, 0.0019492
4: -0.0105827, -0.0078156, -0.0107879, -0.0078496, -0.0017115, 0.0019580
5: 0.0097297, 0.0107778, 0.0096520, 0.0107650, -0.0006483, 0.0007416
6: 0.0058638, 0.0098634, 0.0059129, 0.0101600, -0.0028301, 0.0024738
7: 0.9821625, 0.9849612, 0.9821968, 0.9851688, -0.0019804, 0.0017310
8: -0.0056888, -0.0026882, -0.0056520, -0.0024656, -0.0021233, 0.0018559
9: -0.0032239, -0.0012418, -0.0033709, -0.0012661, -0.0012260, 0.0014025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010511, upper bound: 0.0011036
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010796, upper bound: 0.0011036
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028302, -0.0017564, -0.0029046, -0.0017694, -0.0006688, 0.0007587
1: -0.0114926, -0.0087676, -0.0116814, -0.0088007, -0.0016971, 0.0019253
2: 0.0279000, 0.0295906, 0.0277828, 0.0295700, -0.0010529, 0.0011945
3: 0.0043094, 0.0074662, 0.0043477, 0.0076849, -0.0022304, 0.0019660
4: -0.0105829, -0.0078111, -0.0107750, -0.0078448, -0.0017262, 0.0019583
5: 0.0097297, 0.0107795, 0.0096569, 0.0107668, -0.0006539, 0.0007418
6: 0.0058573, 0.0098637, 0.0059060, 0.0101413, -0.0028306, 0.0024951
7: 0.9821579, 0.9849614, 0.9821920, 0.9851557, -0.0019807, 0.0017460
8: -0.0056937, -0.0026880, -0.0056572, -0.0024797, -0.0021237, 0.0018719
9: -0.0032240, -0.0012386, -0.0033616, -0.0012627, -0.0012365, 0.0014028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010713, upper bound: 0.0011145
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010966, upper bound: 0.0011145
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028973, -0.0017862, -0.0029096, -0.0017713, -0.0007011, 0.0006945
1: -0.0116629, -0.0088433, -0.0116942, -0.0088054, -0.0017792, 0.0017624
2: 0.0277943, 0.0295436, 0.0277749, 0.0295671, -0.0011038, 0.0010934
3: 0.0043971, 0.0076635, 0.0043532, 0.0076997, -0.0020417, 0.0020611
4: -0.0107562, -0.0078881, -0.0107879, -0.0078496, -0.0018097, 0.0017927
5: 0.0096640, 0.0107504, 0.0096520, 0.0107650, -0.0006855, 0.0006790
6: 0.0059686, 0.0101141, 0.0059129, 0.0101600, -0.0025911, 0.0026158
7: 0.9822357, 0.9851366, 0.9821968, 0.9851688, -0.0018131, 0.0018304
8: -0.0056103, -0.0025001, -0.0056520, -0.0024656, -0.0019440, 0.0019625
9: -0.0033482, -0.0012937, -0.0033709, -0.0012661, -0.0012963, 0.0012841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010544, upper bound: 0.0010924
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010857, upper bound: 0.0010923
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028974, -0.0017848, -0.0029046, -0.0017694, -0.0007074, 0.0006924
1: -0.0116632, -0.0088398, -0.0116814, -0.0088007, -0.0017951, 0.0017570
2: 0.0277941, 0.0295458, 0.0277828, 0.0295700, -0.0011137, 0.0010900
3: 0.0043930, 0.0076638, 0.0043477, 0.0076849, -0.0020354, 0.0020795
4: -0.0107564, -0.0078845, -0.0107750, -0.0078448, -0.0018259, 0.0017871
5: 0.0096639, 0.0107517, 0.0096569, 0.0107668, -0.0006916, 0.0006769
6: 0.0059634, 0.0101145, 0.0059060, 0.0101413, -0.0025831, 0.0026392
7: 0.9822322, 0.9851369, 0.9821920, 0.9851557, -0.0018076, 0.0018468
8: -0.0056141, -0.0024998, -0.0056572, -0.0024797, -0.0019380, 0.0019800
9: -0.0033483, -0.0012912, -0.0033616, -0.0012627, -0.0013079, 0.0012802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0011005
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011080, upper bound: 0.0011005
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028325, -0.0017394, -0.0028973, -0.0017860, -0.0006386, 0.0007549
1: -0.0114984, -0.0087245, -0.0116629, -0.0088427, -0.0016205, 0.0019158
2: 0.0278964, 0.0296173, 0.0277943, 0.0295440, -0.0010053, 0.0011886
3: 0.0042595, 0.0074729, 0.0043964, 0.0076635, -0.0022193, 0.0018772
4: -0.0105888, -0.0077673, -0.0107562, -0.0078875, -0.0016483, 0.0019487
5: 0.0097274, 0.0107961, 0.0096640, 0.0107506, -0.0006243, 0.0007381
6: 0.0057940, 0.0098722, 0.0059677, 0.0101141, -0.0028166, 0.0023824
7: 0.9821137, 0.9849674, 0.9822352, 0.9851366, -0.0019709, 0.0016671
8: -0.0057412, -0.0026816, -0.0056109, -0.0025001, -0.0021132, 0.0017874
9: -0.0032283, -0.0012072, -0.0033482, -0.0012933, -0.0011807, 0.0013959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011065, upper bound: 0.0010595
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011065, upper bound: 0.0010931
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028273, -0.0017382, -0.0028974, -0.0017846, -0.0006366, 0.0007587
1: -0.0114853, -0.0087215, -0.0116632, -0.0088392, -0.0016155, 0.0019254
2: 0.0279045, 0.0296192, 0.0277941, 0.0295462, -0.0010022, 0.0011945
3: 0.0042560, 0.0074578, 0.0043923, 0.0076638, -0.0022305, 0.0018714
4: -0.0105755, -0.0077642, -0.0107564, -0.0078839, -0.0016432, 0.0019584
5: 0.0097325, 0.0107973, 0.0096639, 0.0107520, -0.0006224, 0.0007418
6: 0.0057895, 0.0098530, 0.0059626, 0.0101145, -0.0028307, 0.0023751
7: 0.9821105, 0.9849539, 0.9822316, 0.9851369, -0.0019808, 0.0016620
8: -0.0057446, -0.0026960, -0.0056148, -0.0024998, -0.0021238, 0.0017819
9: -0.0032187, -0.0012050, -0.0033483, -0.0012907, -0.0011770, 0.0014029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011118, upper bound: 0.0010783
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011118, upper bound: 0.0011085
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0029031, -0.0017952, -0.0006616, 0.0007022
1: -0.0116575, -0.0087728, -0.0116775, -0.0088660, -0.0016790, 0.0017820
2: 0.0277977, 0.0295874, 0.0277853, 0.0295295, -0.0010417, 0.0011056
3: 0.0043154, 0.0076572, 0.0044234, 0.0076804, -0.0020644, 0.0019451
4: -0.0107507, -0.0078164, -0.0107710, -0.0079113, -0.0017078, 0.0018126
5: 0.0096661, 0.0107775, 0.0096584, 0.0107416, -0.0006469, 0.0006866
6: 0.0058649, 0.0101061, 0.0060020, 0.0101355, -0.0026200, 0.0024685
7: 0.9821633, 0.9851310, 0.9822592, 0.9851516, -0.0018333, 0.0017274
8: -0.0056880, -0.0025061, -0.0055852, -0.0024840, -0.0019656, 0.0018520
9: -0.0033442, -0.0012423, -0.0033588, -0.0013103, -0.0012234, 0.0012984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011065, upper bound: 0.0010622
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011065, upper bound: 0.0010884
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017567, -0.0028972, -0.0017945, -0.0006679, 0.0006994
1: -0.0116577, -0.0087685, -0.0116625, -0.0088644, -0.0016949, 0.0017747
2: 0.0277975, 0.0295900, 0.0277945, 0.0295305, -0.0010515, 0.0011010
3: 0.0043104, 0.0076575, 0.0044216, 0.0076631, -0.0020559, 0.0019635
4: -0.0107509, -0.0078120, -0.0107558, -0.0079096, -0.0017240, 0.0018052
5: 0.0096660, 0.0107792, 0.0096642, 0.0107422, -0.0006530, 0.0006838
6: 0.0058586, 0.0101065, 0.0059997, 0.0101136, -0.0026092, 0.0024919
7: 0.9821588, 0.9851314, 0.9822575, 0.9851363, -0.0018258, 0.0017437
8: -0.0056928, -0.0025058, -0.0055869, -0.0025005, -0.0019576, 0.0018695
9: -0.0033444, -0.0012392, -0.0033479, -0.0013091, -0.0012349, 0.0012931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0010713
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0010963
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017303, -0.0029096, -0.0017713, -0.0006533, 0.0007732
1: -0.0114857, -0.0087016, -0.0116942, -0.0088054, -0.0016578, 0.0019622
2: 0.0279043, 0.0296315, 0.0277749, 0.0295671, -0.0010285, 0.0012174
3: 0.0042329, 0.0074582, 0.0043532, 0.0076997, -0.0022731, 0.0019205
4: -0.0105759, -0.0077440, -0.0107879, -0.0078496, -0.0016863, 0.0019959
5: 0.0097323, 0.0108050, 0.0096520, 0.0107650, -0.0006387, 0.0007560
6: 0.0057603, 0.0098535, 0.0059129, 0.0101600, -0.0028849, 0.0024373
7: 0.9820900, 0.9849543, 0.9821968, 0.9851688, -0.0020187, 0.0017055
8: -0.0057665, -0.0026956, -0.0056520, -0.0024656, -0.0021644, 0.0018286
9: -0.0032190, -0.0011905, -0.0033709, -0.0012661, -0.0012079, 0.0014297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010645, upper bound: 0.0011036
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010984, upper bound: 0.0011036
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017289, -0.0029046, -0.0017694, -0.0006587, 0.0007734
1: -0.0114859, -0.0086980, -0.0116814, -0.0088007, -0.0016715, 0.0019626
2: 0.0279041, 0.0296337, 0.0277828, 0.0295700, -0.0010370, 0.0012176
3: 0.0042288, 0.0074585, 0.0043477, 0.0076849, -0.0022736, 0.0019364
4: -0.0105761, -0.0077404, -0.0107750, -0.0078448, -0.0017002, 0.0019963
5: 0.0097322, 0.0108063, 0.0096569, 0.0107668, -0.0006440, 0.0007561
6: 0.0057550, 0.0098539, 0.0059060, 0.0101413, -0.0028855, 0.0024575
7: 0.9820864, 0.9849546, 0.9821920, 0.9851557, -0.0020191, 0.0017196
8: -0.0057705, -0.0026953, -0.0056572, -0.0024797, -0.0021648, 0.0018437
9: -0.0032192, -0.0011879, -0.0033616, -0.0012627, -0.0012179, 0.0014300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010804, upper bound: 0.0011145
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011118, upper bound: 0.0011145
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0029096, -0.0017713, -0.0006913, 0.0007111
1: -0.0116575, -0.0087728, -0.0116942, -0.0088054, -0.0017542, 0.0018046
2: 0.0277977, 0.0295874, 0.0277749, 0.0295671, -0.0010883, 0.0011196
3: 0.0043154, 0.0076572, 0.0043532, 0.0076997, -0.0020905, 0.0020322
4: -0.0107507, -0.0078164, -0.0107879, -0.0078496, -0.0017844, 0.0018356
5: 0.0096661, 0.0107775, 0.0096520, 0.0107650, -0.0006759, 0.0006953
6: 0.0058649, 0.0101061, 0.0059129, 0.0101600, -0.0026532, 0.0025791
7: 0.9821633, 0.9851310, 0.9821968, 0.9851688, -0.0018566, 0.0018047
8: -0.0056880, -0.0025061, -0.0056520, -0.0024656, -0.0019905, 0.0019350
9: -0.0033442, -0.0012423, -0.0033709, -0.0012661, -0.0012782, 0.0013148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010687, upper bound: 0.0010923
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011063, upper bound: 0.0010923
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017567, -0.0029046, -0.0017694, -0.0006970, 0.0007088
1: -0.0116577, -0.0087685, -0.0116814, -0.0088007, -0.0017686, 0.0017986
2: 0.0277975, 0.0295900, 0.0277828, 0.0295700, -0.0010973, 0.0011159
3: 0.0043104, 0.0076575, 0.0043477, 0.0076849, -0.0020836, 0.0020489
4: -0.0107509, -0.0078120, -0.0107750, -0.0078448, -0.0017990, 0.0018295
5: 0.0096660, 0.0107792, 0.0096569, 0.0107668, -0.0006814, 0.0006930
6: 0.0058586, 0.0101065, 0.0059060, 0.0101413, -0.0026443, 0.0026003
7: 0.9821588, 0.9851314, 0.9821920, 0.9851557, -0.0018504, 0.0018195
8: -0.0056928, -0.0025058, -0.0056572, -0.0024797, -0.0019839, 0.0019508
9: -0.0033444, -0.0012392, -0.0033616, -0.0012627, -0.0012886, 0.0013105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010871, upper bound: 0.0011001
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011237, upper bound: 0.0011001
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028345, -0.0017672, -0.0028275, -0.0017303, -0.0006952, 0.0006563
1: -0.0115036, -0.0087950, -0.0114857, -0.0087016, -0.0017643, 0.0016655
2: 0.0278931, 0.0295736, 0.0279043, 0.0296315, -0.0010946, 0.0010333
3: 0.0043412, 0.0074790, 0.0042329, 0.0074582, -0.0019294, 0.0020438
4: -0.0105941, -0.0078390, -0.0105759, -0.0077440, -0.0017946, 0.0016941
5: 0.0097254, 0.0107690, 0.0097323, 0.0108050, -0.0006797, 0.0006417
6: 0.0058976, 0.0098799, 0.0057603, 0.0098535, -0.0024486, 0.0025939
7: 0.9821861, 0.9849727, 0.9820900, 0.9849543, -0.0017134, 0.0018151
8: -0.0056635, -0.0026758, -0.0057665, -0.0026956, -0.0018371, 0.0019460
9: -0.0032321, -0.0012585, -0.0032190, -0.0011905, -0.0012855, 0.0012135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011019, upper bound: 0.0011330
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011253, upper bound: 0.0011330
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028300, -0.0017655, -0.0028276, -0.0017289, -0.0006921, 0.0006628
1: -0.0114920, -0.0087908, -0.0114859, -0.0086980, -0.0017562, 0.0016819
2: 0.0279003, 0.0295762, 0.0279041, 0.0296337, -0.0010896, 0.0010434
3: 0.0043363, 0.0074655, 0.0042288, 0.0074585, -0.0019484, 0.0020345
4: -0.0105823, -0.0078347, -0.0105761, -0.0077404, -0.0017864, 0.0017108
5: 0.0097299, 0.0107706, 0.0097322, 0.0108063, -0.0006766, 0.0006480
6: 0.0058914, 0.0098628, 0.0057550, 0.0098539, -0.0024727, 0.0025821
7: 0.9821817, 0.9849609, 0.9820864, 0.9849546, -0.0017303, 0.0018068
8: -0.0056682, -0.0026886, -0.0057705, -0.0026953, -0.0018552, 0.0019372
9: -0.0032236, -0.0012555, -0.0032192, -0.0011879, -0.0012796, 0.0012254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010961, upper bound: 0.0011370
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011216, upper bound: 0.0011370
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028301, -0.0017581, -0.0028363, -0.0017184, -0.0007159, 0.0006819
1: -0.0114924, -0.0087720, -0.0115081, -0.0086712, -0.0018167, 0.0017303
2: 0.0279001, 0.0295878, 0.0278904, 0.0296504, -0.0011271, 0.0010735
3: 0.0043145, 0.0074660, 0.0041977, 0.0074841, -0.0020045, 0.0021046
4: -0.0105827, -0.0078156, -0.0105987, -0.0077131, -0.0018479, 0.0017600
5: 0.0097297, 0.0107778, 0.0097237, 0.0108167, -0.0006999, 0.0006667
6: 0.0058638, 0.0098634, 0.0057156, 0.0098864, -0.0025440, 0.0026710
7: 0.9821625, 0.9849612, 0.9820588, 0.9849773, -0.0017802, 0.0018690
8: -0.0056888, -0.0026882, -0.0058001, -0.0026709, -0.0019086, 0.0020039
9: -0.0032239, -0.0012418, -0.0032353, -0.0011683, -0.0013237, 0.0012607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010875, upper bound: 0.0011429
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011152, upper bound: 0.0011429
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028302, -0.0017564, -0.0028329, -0.0017146, -0.0007226, 0.0006787
1: -0.0114926, -0.0087676, -0.0114994, -0.0086617, -0.0018336, 0.0017222
2: 0.0279000, 0.0295906, 0.0278958, 0.0296562, -0.0011376, 0.0010685
3: 0.0043094, 0.0074662, 0.0041868, 0.0074740, -0.0019951, 0.0021241
4: -0.0105829, -0.0078111, -0.0105898, -0.0077035, -0.0018651, 0.0017518
5: 0.0097297, 0.0107795, 0.0097270, 0.0108203, -0.0007064, 0.0006635
6: 0.0058573, 0.0098637, 0.0057017, 0.0098736, -0.0025321, 0.0026958
7: 0.9821579, 0.9849614, 0.9820491, 0.9849683, -0.0017718, 0.0018864
8: -0.0056937, -0.0026880, -0.0058105, -0.0026805, -0.0018997, 0.0020225
9: -0.0032240, -0.0012386, -0.0032290, -0.0011614, -0.0013360, 0.0012548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010958, upper bound: 0.0011403
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011208, upper bound: 0.0011403
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028973, -0.0017862, -0.0028325, -0.0017394, -0.0007549, 0.0006383
1: -0.0116629, -0.0088433, -0.0114984, -0.0087245, -0.0019158, 0.0016198
2: 0.0277943, 0.0295436, 0.0278964, 0.0296173, -0.0011886, 0.0010050
3: 0.0043971, 0.0076635, 0.0042595, 0.0074729, -0.0018765, 0.0022193
4: -0.0107562, -0.0078881, -0.0105888, -0.0077673, -0.0019487, 0.0016477
5: 0.0096640, 0.0107504, 0.0097274, 0.0107961, -0.0007381, 0.0006241
6: 0.0059686, 0.0101141, 0.0057940, 0.0098722, -0.0023815, 0.0028166
7: 0.9822357, 0.9851366, 0.9821137, 0.9849674, -0.0016665, 0.0019709
8: -0.0056103, -0.0025001, -0.0057412, -0.0026816, -0.0017867, 0.0021132
9: -0.0033482, -0.0012937, -0.0032283, -0.0012072, -0.0013959, 0.0011802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010595, upper bound: 0.0011065
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010931, upper bound: 0.0011065
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028974, -0.0017848, -0.0028273, -0.0017382, -0.0007587, 0.0006363
1: -0.0116632, -0.0088398, -0.0114853, -0.0087215, -0.0019254, 0.0016148
2: 0.0277941, 0.0295458, 0.0279045, 0.0296192, -0.0011945, 0.0010018
3: 0.0043930, 0.0076638, 0.0042560, 0.0074578, -0.0018707, 0.0022305
4: -0.0107564, -0.0078845, -0.0105755, -0.0077642, -0.0019584, 0.0016425
5: 0.0096639, 0.0107517, 0.0097325, 0.0107973, -0.0007418, 0.0006222
6: 0.0059634, 0.0101145, 0.0057895, 0.0098530, -0.0023742, 0.0028307
7: 0.9822322, 0.9851369, 0.9821105, 0.9849539, -0.0016613, 0.0019808
8: -0.0056141, -0.0024998, -0.0057446, -0.0026960, -0.0017812, 0.0021238
9: -0.0033483, -0.0012912, -0.0032187, -0.0012050, -0.0014029, 0.0011766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0011118
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011085, upper bound: 0.0011118
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028973, -0.0017862, -0.0028363, -0.0017184, -0.0007900, 0.0006495
1: -0.0116629, -0.0088433, -0.0115081, -0.0086712, -0.0020049, 0.0016483
2: 0.0277943, 0.0295436, 0.0278904, 0.0296504, -0.0012438, 0.0010226
3: 0.0043971, 0.0076635, 0.0041977, 0.0074841, -0.0019095, 0.0023225
4: -0.0107562, -0.0078881, -0.0105987, -0.0077131, -0.0020393, 0.0016766
5: 0.0096640, 0.0107504, 0.0097237, 0.0108167, -0.0007724, 0.0006350
6: 0.0059686, 0.0101141, 0.0057156, 0.0098864, -0.0024234, 0.0029476
7: 0.9822357, 0.9851366, 0.9820588, 0.9849773, -0.0016958, 0.0020626
8: -0.0056103, -0.0025001, -0.0058001, -0.0026709, -0.0018181, 0.0022114
9: -0.0033482, -0.0012937, -0.0032353, -0.0011683, -0.0014608, 0.0012010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010595, upper bound: 0.0011085
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010920, upper bound: 0.0011085
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028974, -0.0017848, -0.0028329, -0.0017146, -0.0007933, 0.0006473
1: -0.0116632, -0.0088398, -0.0114994, -0.0086617, -0.0020131, 0.0016426
2: 0.0277941, 0.0295458, 0.0278958, 0.0296562, -0.0012489, 0.0010191
3: 0.0043930, 0.0076638, 0.0041868, 0.0074740, -0.0019029, 0.0023321
4: -0.0107564, -0.0078845, -0.0105898, -0.0077035, -0.0020477, 0.0016708
5: 0.0096639, 0.0107517, 0.0097270, 0.0108203, -0.0007756, 0.0006329
6: 0.0059634, 0.0101145, 0.0057017, 0.0098736, -0.0024150, 0.0029597
7: 0.9822322, 0.9851369, 0.9820491, 0.9849683, -0.0016899, 0.0020711
8: -0.0056141, -0.0024998, -0.0058105, -0.0026805, -0.0018119, 0.0022205
9: -0.0033483, -0.0012912, -0.0032290, -0.0011614, -0.0014668, 0.0011968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0011142
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011078, upper bound: 0.0011142
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028325, -0.0017394, -0.0028275, -0.0017303, -0.0007043, 0.0006907
1: -0.0114984, -0.0087245, -0.0114857, -0.0087016, -0.0017874, 0.0017528
2: 0.0278964, 0.0296173, 0.0279043, 0.0296315, -0.0011089, 0.0010874
3: 0.0042595, 0.0074729, 0.0042329, 0.0074582, -0.0020305, 0.0020706
4: -0.0105888, -0.0077673, -0.0105759, -0.0077440, -0.0018181, 0.0017829
5: 0.0097274, 0.0107961, 0.0097323, 0.0108050, -0.0006886, 0.0006753
6: 0.0057940, 0.0098722, 0.0057603, 0.0098535, -0.0025770, 0.0026279
7: 0.9821137, 0.9849674, 0.9820900, 0.9849543, -0.0018032, 0.0018388
8: -0.0057412, -0.0026816, -0.0057665, -0.0026956, -0.0019334, 0.0019715
9: -0.0032283, -0.0012072, -0.0032190, -0.0011905, -0.0013023, 0.0012771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011186, upper bound: 0.0011272
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011505, upper bound: 0.0011272
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028273, -0.0017382, -0.0028276, -0.0017289, -0.0007018, 0.0006972
1: -0.0114853, -0.0087215, -0.0114859, -0.0086980, -0.0017810, 0.0017693
2: 0.0279045, 0.0296192, 0.0279041, 0.0296337, -0.0011050, 0.0010977
3: 0.0042560, 0.0074578, 0.0042288, 0.0074585, -0.0020497, 0.0020632
4: -0.0105755, -0.0077642, -0.0105761, -0.0077404, -0.0018116, 0.0017997
5: 0.0097325, 0.0107973, 0.0097322, 0.0108063, -0.0006862, 0.0006817
6: 0.0057895, 0.0098530, 0.0057550, 0.0098539, -0.0026013, 0.0026185
7: 0.9821105, 0.9849539, 0.9820864, 0.9849546, -0.0018203, 0.0018323
8: -0.0057446, -0.0026960, -0.0057705, -0.0026953, -0.0019516, 0.0019645
9: -0.0032187, -0.0012050, -0.0032192, -0.0011879, -0.0012977, 0.0012892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011140, upper bound: 0.0011326
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011456, upper bound: 0.0011326
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017303, -0.0028363, -0.0017184, -0.0007189, 0.0007131
1: -0.0114857, -0.0087016, -0.0115081, -0.0086712, -0.0018244, 0.0018095
2: 0.0279043, 0.0296315, 0.0278904, 0.0296504, -0.0011319, 0.0011226
3: 0.0042329, 0.0074582, 0.0041977, 0.0074841, -0.0020962, 0.0021135
4: -0.0105759, -0.0077440, -0.0105987, -0.0077131, -0.0018558, 0.0018406
5: 0.0097323, 0.0108050, 0.0097237, 0.0108167, -0.0007029, 0.0006972
6: 0.0057603, 0.0098535, 0.0057156, 0.0098864, -0.0026604, 0.0026823
7: 0.9820900, 0.9849543, 0.9820588, 0.9849773, -0.0018616, 0.0018770
8: -0.0057665, -0.0026956, -0.0058001, -0.0026709, -0.0019959, 0.0020124
9: -0.0032190, -0.0011905, -0.0032353, -0.0011683, -0.0013293, 0.0013184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011070, upper bound: 0.0011410
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011413, upper bound: 0.0011410
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017289, -0.0028329, -0.0017146, -0.0007251, 0.0007113
1: -0.0114859, -0.0086980, -0.0114994, -0.0086617, -0.0018400, 0.0018051
2: 0.0279041, 0.0296337, 0.0278958, 0.0296562, -0.0011415, 0.0011199
3: 0.0042288, 0.0074585, 0.0041868, 0.0074740, -0.0020912, 0.0021315
4: -0.0105761, -0.0077404, -0.0105898, -0.0077035, -0.0018716, 0.0018361
5: 0.0097322, 0.0108063, 0.0097270, 0.0108203, -0.0007089, 0.0006955
6: 0.0057550, 0.0098539, 0.0057017, 0.0098736, -0.0026540, 0.0027052
7: 0.9820864, 0.9849546, 0.9820491, 0.9849683, -0.0018571, 0.0018929
8: -0.0057705, -0.0026953, -0.0058105, -0.0026805, -0.0019911, 0.0020295
9: -0.0032192, -0.0011879, -0.0032290, -0.0011614, -0.0013406, 0.0013152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011141, upper bound: 0.0011377
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011455, upper bound: 0.0011377
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0028325, -0.0017394, -0.0007664, 0.0006738
1: -0.0116575, -0.0087728, -0.0114984, -0.0087245, -0.0019448, 0.0017098
2: 0.0277977, 0.0295874, 0.0278964, 0.0296173, -0.0012066, 0.0010608
3: 0.0043154, 0.0076572, 0.0042595, 0.0074729, -0.0019808, 0.0022530
4: -0.0107507, -0.0078164, -0.0105888, -0.0077673, -0.0019782, 0.0017392
5: 0.0096661, 0.0107775, 0.0097274, 0.0107961, -0.0007493, 0.0006588
6: 0.0058649, 0.0101061, 0.0057940, 0.0098722, -0.0025139, 0.0028594
7: 0.9821633, 0.9851310, 0.9821137, 0.9849674, -0.0017591, 0.0020008
8: -0.0056880, -0.0025061, -0.0057412, -0.0026816, -0.0018860, 0.0021452
9: -0.0033442, -0.0012423, -0.0032283, -0.0012072, -0.0014170, 0.0012458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011172, upper bound: 0.0010765
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011171, upper bound: 0.0011007
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017567, -0.0028273, -0.0017382, -0.0007700, 0.0006725
1: -0.0116577, -0.0087685, -0.0114853, -0.0087215, -0.0019541, 0.0017064
2: 0.0277975, 0.0295900, 0.0279045, 0.0296192, -0.0012123, 0.0010587
3: 0.0043104, 0.0076575, 0.0042560, 0.0074578, -0.0019768, 0.0022637
4: -0.0107509, -0.0078120, -0.0105755, -0.0077642, -0.0019876, 0.0017357
5: 0.0096660, 0.0107792, 0.0097325, 0.0107973, -0.0007529, 0.0006575
6: 0.0058586, 0.0101065, 0.0057895, 0.0098530, -0.0025089, 0.0028729
7: 0.9821588, 0.9851314, 0.9821105, 0.9849539, -0.0017556, 0.0020103
8: -0.0056928, -0.0025058, -0.0057446, -0.0026960, -0.0018823, 0.0021554
9: -0.0033444, -0.0012392, -0.0032187, -0.0012050, -0.0014238, 0.0012433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0010804
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011047
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0028363, -0.0017184, -0.0007946, 0.0006825
1: -0.0116575, -0.0087728, -0.0115081, -0.0086712, -0.0020165, 0.0017320
2: 0.0277977, 0.0295874, 0.0278904, 0.0296504, -0.0012510, 0.0010745
3: 0.0043154, 0.0076572, 0.0041977, 0.0074841, -0.0020064, 0.0023360
4: -0.0107507, -0.0078164, -0.0105987, -0.0077131, -0.0020511, 0.0017617
5: 0.0096661, 0.0107775, 0.0097237, 0.0108167, -0.0007769, 0.0006673
6: 0.0058649, 0.0101061, 0.0057156, 0.0098864, -0.0025464, 0.0029647
7: 0.9821633, 0.9851310, 0.9820588, 0.9849773, -0.0017818, 0.0020746
8: -0.0056880, -0.0025061, -0.0058001, -0.0026709, -0.0019104, 0.0022243
9: -0.0033442, -0.0012423, -0.0032353, -0.0011683, -0.0014692, 0.0012619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010797, upper bound: 0.0011048
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011172, upper bound: 0.0011048
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0028953, -0.0017567, -0.0028329, -0.0017146, -0.0007979, 0.0006820
1: -0.0116577, -0.0087685, -0.0114994, -0.0086617, -0.0020247, 0.0017306
2: 0.0277975, 0.0295900, 0.0278958, 0.0296562, -0.0012561, 0.0010736
3: 0.0043104, 0.0076575, 0.0041868, 0.0074740, -0.0020048, 0.0023455
4: -0.0107509, -0.0078120, -0.0105898, -0.0077035, -0.0020595, 0.0017603
5: 0.0096660, 0.0107792, 0.0097270, 0.0108203, -0.0007801, 0.0006667
6: 0.0058586, 0.0101065, 0.0057017, 0.0098736, -0.0025443, 0.0029768
7: 0.9821588, 0.9851314, 0.9820491, 0.9849683, -0.0017804, 0.0020830
8: -0.0056928, -0.0025058, -0.0058105, -0.0026805, -0.0019089, 0.0022333
9: -0.0033444, -0.0012392, -0.0032290, -0.0011614, -0.0014752, 0.0012609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010933, upper bound: 0.0011088
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011088
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028345, -0.0017672, -0.0028952, -0.0017584, -0.0006659, 0.0007279
1: -0.0115036, -0.0087950, -0.0116575, -0.0087728, -0.0016898, 0.0018473
2: 0.0278931, 0.0295736, 0.0277977, 0.0295874, -0.0010484, 0.0011461
3: 0.0043412, 0.0074790, 0.0043154, 0.0076572, -0.0021400, 0.0019576
4: -0.0105941, -0.0078390, -0.0107507, -0.0078164, -0.0017188, 0.0018790
5: 0.0097254, 0.0107690, 0.0096661, 0.0107775, -0.0006510, 0.0007117
6: 0.0058976, 0.0098799, 0.0058649, 0.0101061, -0.0027159, 0.0024844
7: 0.9821861, 0.9849727, 0.9821633, 0.9851310, -0.0019005, 0.0017385
8: -0.0056635, -0.0026758, -0.0056880, -0.0025061, -0.0020376, 0.0018639
9: -0.0032321, -0.0012585, -0.0033442, -0.0012423, -0.0012312, 0.0013459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010637, upper bound: 0.0011107
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010892, upper bound: 0.0011107
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028300, -0.0017655, -0.0028953, -0.0017567, -0.0006639, 0.0007309
1: -0.0114920, -0.0087908, -0.0116577, -0.0087685, -0.0016846, 0.0018548
2: 0.0279003, 0.0295762, 0.0277975, 0.0295900, -0.0010452, 0.0011507
3: 0.0043363, 0.0074655, 0.0043104, 0.0076575, -0.0021487, 0.0019516
4: -0.0105823, -0.0078347, -0.0107509, -0.0078120, -0.0017136, 0.0018867
5: 0.0097299, 0.0107706, 0.0096660, 0.0107792, -0.0006491, 0.0007146
6: 0.0058914, 0.0098628, 0.0058586, 0.0101065, -0.0027270, 0.0024768
7: 0.9821817, 0.9849609, 0.9821588, 0.9851314, -0.0019082, 0.0017332
8: -0.0056682, -0.0026886, -0.0056928, -0.0025058, -0.0020459, 0.0018582
9: -0.0032236, -0.0012555, -0.0033444, -0.0012392, -0.0012275, 0.0013515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010714, upper bound: 0.0011237
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010963, upper bound: 0.0011237
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0029031, -0.0017953, -0.0028952, -0.0017584, -0.0007022, 0.0006615
1: -0.0116775, -0.0088664, -0.0116575, -0.0087728, -0.0017820, 0.0016787
2: 0.0277853, 0.0295293, 0.0277977, 0.0295874, -0.0011056, 0.0010414
3: 0.0044238, 0.0076804, 0.0043154, 0.0076572, -0.0019446, 0.0020644
4: -0.0107710, -0.0079116, -0.0107507, -0.0078164, -0.0018126, 0.0017075
5: 0.0096584, 0.0107415, 0.0096661, 0.0107775, -0.0006866, 0.0006467
6: 0.0060025, 0.0101355, 0.0058649, 0.0101061, -0.0024680, 0.0026200
7: 0.9822596, 0.9851516, 0.9821633, 0.9851310, -0.0017270, 0.0018333
8: -0.0055848, -0.0024840, -0.0056880, -0.0025061, -0.0018516, 0.0019656
9: -0.0033588, -0.0013105, -0.0033442, -0.0012423, -0.0012984, 0.0012231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010663, upper bound: 0.0010984
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010958, upper bound: 0.0010984
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0028972, -0.0017946, -0.0028953, -0.0017567, -0.0006994, 0.0006678
1: -0.0116625, -0.0088647, -0.0116577, -0.0087685, -0.0017747, 0.0016946
2: 0.0277945, 0.0295303, 0.0277975, 0.0295900, -0.0011010, 0.0010513
3: 0.0044220, 0.0076631, 0.0043104, 0.0076575, -0.0019631, 0.0020559
4: -0.0107558, -0.0079100, -0.0107509, -0.0078120, -0.0018052, 0.0017237
5: 0.0096642, 0.0107421, 0.0096660, 0.0107792, -0.0006838, 0.0006529
6: 0.0060002, 0.0101136, 0.0058586, 0.0101065, -0.0024914, 0.0026092
7: 0.9822579, 0.9851363, 0.9821588, 0.9851314, -0.0017434, 0.0018258
8: -0.0055866, -0.0025005, -0.0056928, -0.0025058, -0.0018691, 0.0019576
9: -0.0033479, -0.0013094, -0.0033444, -0.0012392, -0.0012931, 0.0012347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0011117
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011085, upper bound: 0.0011117
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028345, -0.0017672, -0.0029017, -0.0017340, -0.0007010, 0.0007394
1: -0.0115036, -0.0087950, -0.0116741, -0.0087108, -0.0017788, 0.0018763
2: 0.0278931, 0.0295736, 0.0277874, 0.0296258, -0.0011036, 0.0011641
3: 0.0043412, 0.0074790, 0.0042436, 0.0076765, -0.0021736, 0.0020607
4: -0.0105941, -0.0078390, -0.0107676, -0.0077534, -0.0018094, 0.0019085
5: 0.0097254, 0.0107690, 0.0096597, 0.0108014, -0.0006853, 0.0007229
6: 0.0058976, 0.0098799, 0.0057739, 0.0101306, -0.0027586, 0.0026153
7: 0.9821861, 0.9849727, 0.9820995, 0.9851481, -0.0019304, 0.0018300
8: -0.0056635, -0.0026758, -0.0057564, -0.0024878, -0.0020696, 0.0019621
9: -0.0032321, -0.0012585, -0.0033563, -0.0011972, -0.0012961, 0.0013671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010636, upper bound: 0.0011182
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010892, upper bound: 0.0011182
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028300, -0.0017655, -0.0029018, -0.0017319, -0.0006999, 0.0007406
1: -0.0114920, -0.0087908, -0.0116744, -0.0087056, -0.0017762, 0.0018795
2: 0.0279003, 0.0295762, 0.0277872, 0.0296290, -0.0011019, 0.0011660
3: 0.0043363, 0.0074655, 0.0042376, 0.0076768, -0.0021773, 0.0020576
4: -0.0105823, -0.0078347, -0.0107678, -0.0077481, -0.0018067, 0.0019117
5: 0.0097299, 0.0107706, 0.0096596, 0.0108034, -0.0006843, 0.0007241
6: 0.0058914, 0.0098628, 0.0057662, 0.0101310, -0.0027633, 0.0026114
7: 0.9821817, 0.9849609, 0.9820942, 0.9851484, -0.0019336, 0.0018273
8: -0.0056682, -0.0026886, -0.0057621, -0.0024874, -0.0020731, 0.0019592
9: -0.0032236, -0.0012555, -0.0033565, -0.0011934, -0.0012941, 0.0013694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010713, upper bound: 0.0011299
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010959, upper bound: 0.0011299
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028973, -0.0017862, -0.0029065, -0.0017439, -0.0007234, 0.0006865
1: -0.0116629, -0.0088433, -0.0116861, -0.0087360, -0.0018358, 0.0017420
2: 0.0277943, 0.0295436, 0.0277799, 0.0296102, -0.0011390, 0.0010808
3: 0.0043971, 0.0076635, 0.0042728, 0.0076904, -0.0020181, 0.0021267
4: -0.0107562, -0.0078881, -0.0107798, -0.0077790, -0.0018674, 0.0017719
5: 0.0096640, 0.0107504, 0.0096551, 0.0107917, -0.0007073, 0.0006712
6: 0.0059686, 0.0101141, 0.0058109, 0.0101482, -0.0025612, 0.0026991
7: 0.9822357, 0.9851366, 0.9821255, 0.9851605, -0.0017922, 0.0018887
8: -0.0056103, -0.0025001, -0.0057285, -0.0024745, -0.0019215, 0.0020250
9: -0.0033482, -0.0012937, -0.0033651, -0.0012156, -0.0013376, 0.0012693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010544, upper bound: 0.0011085
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010857, upper bound: 0.0011085
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028974, -0.0017848, -0.0029016, -0.0017411, -0.0007302, 0.0006842
1: -0.0116632, -0.0088398, -0.0116738, -0.0087289, -0.0018531, 0.0017364
2: 0.0277941, 0.0295458, 0.0277876, 0.0296146, -0.0011497, 0.0010772
3: 0.0043930, 0.0076638, 0.0042646, 0.0076761, -0.0020115, 0.0021467
4: -0.0107564, -0.0078845, -0.0107672, -0.0077717, -0.0018849, 0.0017662
5: 0.0096639, 0.0107517, 0.0096599, 0.0107945, -0.0007139, 0.0006690
6: 0.0059634, 0.0101145, 0.0058004, 0.0101300, -0.0025528, 0.0027245
7: 0.9822322, 0.9851369, 0.9821181, 0.9851477, -0.0017863, 0.0019064
8: -0.0056141, -0.0024998, -0.0057364, -0.0024881, -0.0019152, 0.0020440
9: -0.0033483, -0.0012912, -0.0033560, -0.0012104, -0.0013502, 0.0012651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010783, upper bound: 0.0011142
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011078, upper bound: 0.0011142
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0028325, -0.0017394, -0.0028952, -0.0017584, -0.0006738, 0.0007664
1: -0.0114984, -0.0087245, -0.0116575, -0.0087728, -0.0017098, 0.0019448
2: 0.0278964, 0.0296173, 0.0277977, 0.0295874, -0.0010608, 0.0012066
3: 0.0042595, 0.0074729, 0.0043154, 0.0076572, -0.0022530, 0.0019808
4: -0.0105888, -0.0077673, -0.0107507, -0.0078164, -0.0017392, 0.0019782
5: 0.0097274, 0.0107961, 0.0096661, 0.0107775, -0.0006588, 0.0007493
6: 0.0057940, 0.0098722, 0.0058649, 0.0101061, -0.0028594, 0.0025139
7: 0.9821137, 0.9849674, 0.9821633, 0.9851310, -0.0020008, 0.0017591
8: -0.0057412, -0.0026816, -0.0056880, -0.0025061, -0.0021452, 0.0018860
9: -0.0032283, -0.0012072, -0.0033442, -0.0012423, -0.0012458, 0.0014170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010822, upper bound: 0.0011046
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011140, upper bound: 0.0011046
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0028273, -0.0017382, -0.0028953, -0.0017567, -0.0006725, 0.0007700
1: -0.0114853, -0.0087215, -0.0116577, -0.0087685, -0.0017064, 0.0019541
2: 0.0279045, 0.0296192, 0.0277975, 0.0295900, -0.0010587, 0.0012123
3: 0.0042560, 0.0074578, 0.0043104, 0.0076575, -0.0022637, 0.0019768
4: -0.0105755, -0.0077642, -0.0107509, -0.0078120, -0.0017357, 0.0019876
5: 0.0097325, 0.0107973, 0.0096660, 0.0107792, -0.0006575, 0.0007529
6: 0.0057895, 0.0098530, 0.0058586, 0.0101065, -0.0028729, 0.0025089
7: 0.9821105, 0.9849539, 0.9821588, 0.9851314, -0.0020103, 0.0017556
8: -0.0057446, -0.0026960, -0.0056928, -0.0025058, -0.0021554, 0.0018823
9: -0.0032187, -0.0012050, -0.0033444, -0.0012392, -0.0012433, 0.0014238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010869, upper bound: 0.0011177
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011177, upper bound: 0.0011177
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0029005, -0.0017678, -0.0028952, -0.0017584, -0.0007103, 0.0006965
1: -0.0116711, -0.0087967, -0.0116575, -0.0087728, -0.0018025, 0.0017676
2: 0.0277892, 0.0295725, 0.0277977, 0.0295874, -0.0011183, 0.0010966
3: 0.0043431, 0.0076730, 0.0043154, 0.0076572, -0.0020476, 0.0020881
4: -0.0107645, -0.0078407, -0.0107507, -0.0078164, -0.0018334, 0.0017979
5: 0.0096609, 0.0107683, 0.0096661, 0.0107775, -0.0006945, 0.0006810
6: 0.0059001, 0.0101261, 0.0058649, 0.0101061, -0.0025987, 0.0026501
7: 0.9821879, 0.9851450, 0.9821633, 0.9851310, -0.0018185, 0.0018544
8: -0.0056616, -0.0024911, -0.0056880, -0.0025061, -0.0019497, 0.0019882
9: -0.0033541, -0.0012598, -0.0033442, -0.0012423, -0.0013133, 0.0012879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010841, upper bound: 0.0010918
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011210, upper bound: 0.0010918
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0028950, -0.0017658, -0.0028953, -0.0017567, -0.0007083, 0.0007038
1: -0.0116571, -0.0087917, -0.0116577, -0.0087685, -0.0017974, 0.0017859
2: 0.0277979, 0.0295756, 0.0277975, 0.0295900, -0.0011151, 0.0011080
3: 0.0043373, 0.0076568, 0.0043104, 0.0076575, -0.0020689, 0.0020822
4: -0.0107503, -0.0078356, -0.0107509, -0.0078120, -0.0018282, 0.0018166
5: 0.0096663, 0.0107703, 0.0096660, 0.0107792, -0.0006925, 0.0006881
6: 0.0058927, 0.0101056, 0.0058586, 0.0101065, -0.0026257, 0.0026426
7: 0.9821827, 0.9851307, 0.9821588, 0.9851314, -0.0018373, 0.0018491
8: -0.0056672, -0.0025065, -0.0056928, -0.0025058, -0.0019699, 0.0019826
9: -0.0033439, -0.0012561, -0.0033444, -0.0012392, -0.0013096, 0.0013012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010934, upper bound: 0.0011048
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011297, upper bound: 0.0011047
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028275, -0.0017303, -0.0029065, -0.0017439, -0.0006868, 0.0007828
1: -0.0114857, -0.0087016, -0.0116861, -0.0087360, -0.0017429, 0.0019865
2: 0.0279043, 0.0296315, 0.0277799, 0.0296102, -0.0010813, 0.0012324
3: 0.0042329, 0.0074582, 0.0042728, 0.0076904, -0.0023012, 0.0020191
4: -0.0105759, -0.0077440, -0.0107798, -0.0077790, -0.0017728, 0.0020206
5: 0.0097323, 0.0108050, 0.0096551, 0.0107917, -0.0006715, 0.0007653
6: 0.0057603, 0.0098535, 0.0058109, 0.0101482, -0.0029206, 0.0025625
7: 0.9820900, 0.9849543, 0.9821255, 0.9851605, -0.0020437, 0.0017931
8: -0.0057665, -0.0026956, -0.0057285, -0.0024745, -0.0021911, 0.0019225
9: -0.0032190, -0.0011905, -0.0033651, -0.0012156, -0.0012699, 0.0014474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010723, upper bound: 0.0011151
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011054, upper bound: 0.0011151
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028276, -0.0017289, -0.0029016, -0.0017411, -0.0006921, 0.0007836
1: -0.0114859, -0.0086980, -0.0116738, -0.0087289, -0.0017563, 0.0019885
2: 0.0279041, 0.0296337, 0.0277876, 0.0296146, -0.0010896, 0.0012337
3: 0.0042288, 0.0074585, 0.0042646, 0.0076761, -0.0023036, 0.0020346
4: -0.0105761, -0.0077404, -0.0107672, -0.0077717, -0.0017865, 0.0020226
5: 0.0097322, 0.0108063, 0.0096599, 0.0107945, -0.0006767, 0.0007661
6: 0.0057550, 0.0098539, 0.0058004, 0.0101300, -0.0029235, 0.0025822
7: 0.9820864, 0.9849546, 0.9821181, 0.9851477, -0.0020457, 0.0018069
8: -0.0057705, -0.0026953, -0.0057364, -0.0024881, -0.0021934, 0.0019373
9: -0.0032192, -0.0011879, -0.0033560, -0.0012104, -0.0012797, 0.0014488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 81

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010869, upper bound: 0.0011236
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011177, upper bound: 0.0011236
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028952, -0.0017584, -0.0029065, -0.0017439, -0.0007257, 0.0007188
1: -0.0116575, -0.0087728, -0.0116861, -0.0087360, -0.0018416, 0.0018242
2: 0.0277977, 0.0295874, 0.0277799, 0.0296102, -0.0011425, 0.0011317
3: 0.0043154, 0.0076572, 0.0042728, 0.0076904, -0.0021132, 0.0021334
4: -0.0107507, -0.0078164, -0.0107798, -0.0077790, -0.0018732, 0.0018555
5: 0.0096661, 0.0107775, 0.0096551, 0.0107917, -0.0007095, 0.0007028
6: 0.0058649, 0.0101061, 0.0058109, 0.0101482, -0.0026819, 0.0027075
7: 0.9821633, 0.9851310, 0.9821255, 0.9851605, -0.0018767, 0.0018946
8: -0.0056880, -0.0025061, -0.0057285, -0.0024745, -0.0020121, 0.0020313
9: -0.0033442, -0.0012423, -0.0033651, -0.0012156, -0.0013418, 0.0013291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.05 + 597.37 = 600.41 seconds
