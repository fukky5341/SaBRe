## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058113


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0007723, 0.0011158, 0.0007723, 0.0011158, -0.0003434, 0.0003434)
1: (0.9934251, 0.9942437, 0.9934251, 0.9942437, -0.0008186, 0.0008186)
2: (-0.0086082, -0.0051781, -0.0086082, -0.0051781, -0.0031634, 0.0031634)
3: (0.0036543, 0.0041496, 0.0036543, 0.0041496, -0.0004953, 0.0004953)
4: (0.0025095, 0.0052205, 0.0025095, 0.0052205, -0.0027109, 0.0027109)
5: (0.0051925, 0.0064771, 0.0051925, 0.0064771, -0.0012846, 0.0012846)
6: (-0.0021091, -0.0009185, -0.0021091, -0.0009185, -0.0011906, 0.0011906)
7: (-0.0082924, -0.0075232, -0.0082924, -0.0075232, -0.0007692, 0.0007692)
8: (0.0050551, 0.0095618, 0.0050551, 0.0095618, -0.0044337, 0.0044337)
9: (-0.0036861, -0.0031783, -0.0036861, -0.0031783, -0.0005077, 0.0005077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 1.55 = 3.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0007088, upper bound: 0.0007087

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007026, upper bound: 0.0007075
time: 0.69 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007088, upper bound: 0.0007087
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 1, lower bound: -0.0007026, upper bound: 0.0007075
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 1, lower bound: -0.0007088, upper bound: 0.0007087

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: 0.0007729, 0.0011157, 0.0007406, 0.0011166, -0.0003437, 0.0003752
1: 0.9934254, 0.9942425, 0.9934148, 0.9943110, -0.0008856, 0.0008277
2: -0.0086062, -0.0052274, -0.0087180, -0.0053372, -0.0030021, 0.0032172
3: 0.0036550, 0.0041494, 0.0036146, 0.0041568, -0.0005018, 0.0005349
4: 0.0025485, 0.0052189, 0.0026352, 0.0053072, -0.0027587, 0.0025837
5: 0.0051938, 0.0064664, 0.0051172, 0.0064427, -0.0012489, 0.0013492
6: -0.0021084, -0.0009356, -0.0021472, -0.0009048, -0.0012036, 0.0012116
7: -0.0082831, -0.0075237, -0.0082623, -0.0074909, -0.0007921, 0.0007386
8: 0.0051198, 0.0095592, 0.0052640, 0.0097060, -0.0045130, 0.0042220
9: -0.0036856, -0.0031792, -0.0036844, -0.0031323, -0.0005532, 0.0005053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006916, upper bound: 0.0006885
time: 0.79 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006878
time: 0.74 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: 0.0007724, 0.0011158, 0.0007728, 0.0011157, -0.0003434, 0.0003429
1: 0.9934251, 0.9942437, 0.9934254, 0.9942426, -0.0008175, 0.0008183
2: -0.0086082, -0.0051798, -0.0086064, -0.0052548, -0.0030856, 0.0031598
3: 0.0036543, 0.0041496, 0.0036550, 0.0041495, -0.0004951, 0.0004946
4: 0.0025109, 0.0052204, 0.0025701, 0.0052190, -0.0027081, 0.0026503
5: 0.0051925, 0.0064767, 0.0051937, 0.0064605, -0.0012680, 0.0012830
6: -0.0021091, -0.0009191, -0.0021084, -0.0009451, -0.0011640, 0.0011894
7: -0.0082921, -0.0075232, -0.0082779, -0.0075237, -0.0007684, 0.0007547
8: 0.0050573, 0.0095617, 0.0051557, 0.0095593, -0.0044291, 0.0043332
9: -0.0036860, -0.0031783, -0.0036853, -0.0031791, -0.0005069, 0.0005069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007075, upper bound: 0.0007026
time: 0.71 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007075, upper bound: 0.0007088
time: 0.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.22 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0006916, upper bound: 0.0006885
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006878
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0007075, upper bound: 0.0007026
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0007075, upper bound: 0.0007088

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: 0.0007729, 0.0011157, 0.0007513, 0.0011163, -0.0003434, 0.0003645
1: 0.9934254, 0.9942425, 0.9934182, 0.9942882, -0.0008628, 0.0008243
2: -0.0086062, -0.0052274, -0.0086809, -0.0053395, -0.0029997, 0.0031798
3: 0.0036550, 0.0041494, 0.0036280, 0.0041544, -0.0004994, 0.0005215
4: 0.0025485, 0.0052189, 0.0026370, 0.0052780, -0.0027294, 0.0025819
5: 0.0051938, 0.0064664, 0.0051426, 0.0064422, -0.0012484, 0.0013238
6: -0.0021084, -0.0009356, -0.0021343, -0.0009283, -0.0011801, 0.0011987
7: -0.0082831, -0.0075237, -0.0082619, -0.0075018, -0.0007812, 0.0007381
8: 0.0051198, 0.0095592, 0.0052671, 0.0096573, -0.0044643, 0.0042190
9: -0.0036856, -0.0031792, -0.0036844, -0.0031479, -0.0005377, 0.0005052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006912, upper bound: 0.0006815
time: 0.63 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006912, upper bound: 0.0006886
time: 0.75 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: 0.0007853, 0.0011154, 0.0007851, 0.0011154, -0.0003301, 0.0003303
1: 0.9934295, 0.9942161, 0.9934294, 0.9942166, -0.0007871, 0.0007867
2: -0.0085632, -0.0052299, -0.0085639, -0.0052390, -0.0030676, 0.0030823
3: 0.0036706, 0.0041466, 0.0036704, 0.0041466, -0.0004760, 0.0004762
4: 0.0025504, 0.0051849, 0.0025576, 0.0051855, -0.0026350, 0.0026273
5: 0.0052234, 0.0064659, 0.0052229, 0.0064639, -0.0012406, 0.0012430
6: -0.0020935, -0.0009364, -0.0020937, -0.0009396, -0.0011539, 0.0011573
7: -0.0082826, -0.0075364, -0.0082809, -0.0075362, -0.0007464, 0.0007445
8: 0.0051230, 0.0095027, 0.0051349, 0.0095036, -0.0043129, 0.0042985
9: -0.0036855, -0.0031972, -0.0036854, -0.0031969, -0.0004886, 0.0004883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006878
time: 0.74 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006879
time: 0.69 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007406, 0.0011166, 0.0007728, 0.0011157, -0.0003752, 0.0003437
1: 0.9934148, 0.9943110, 0.9934254, 0.9942426, -0.0008278, 0.0008856
2: -0.0087180, -0.0053372, -0.0086064, -0.0052548, -0.0031911, 0.0030022
3: 0.0036146, 0.0041568, 0.0036550, 0.0041495, -0.0005349, 0.0005019
4: 0.0026352, 0.0053072, 0.0025701, 0.0052190, -0.0025838, 0.0027371
5: 0.0051172, 0.0064427, 0.0051937, 0.0064605, -0.0013433, 0.0012490
6: -0.0021472, -0.0009048, -0.0021084, -0.0009451, -0.0012021, 0.0012036
7: -0.0082623, -0.0074909, -0.0082779, -0.0075237, -0.0007386, 0.0007870
8: 0.0052640, 0.0097060, 0.0051557, 0.0095593, -0.0042222, 0.0044772
9: -0.0036844, -0.0031323, -0.0036853, -0.0031791, -0.0005053, 0.0005529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006916
time: 0.80 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
time: 0.71 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007728, 0.0011157, 0.0007728, 0.0011157, -0.0003429, 0.0003429
1: 0.9934254, 0.9942426, 0.9934254, 0.9942426, -0.0008172, 0.0008172
2: -0.0086064, -0.0052548, -0.0086064, -0.0052548, -0.0030838, 0.0030838
3: 0.0036550, 0.0041495, 0.0036550, 0.0041495, -0.0004945, 0.0004945
4: 0.0025701, 0.0052190, 0.0025701, 0.0052190, -0.0026489, 0.0026489
5: 0.0051937, 0.0064605, 0.0051937, 0.0064605, -0.0012668, 0.0012668
6: -0.0021084, -0.0009451, -0.0021084, -0.0009451, -0.0011634, 0.0011634
7: -0.0082779, -0.0075237, -0.0082779, -0.0075237, -0.0007542, 0.0007542
8: 0.0051557, 0.0095593, 0.0051557, 0.0095593, -0.0043309, 0.0043309
9: -0.0036853, -0.0031791, -0.0036853, -0.0031791, -0.0005062, 0.0005062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006984
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
time: 0.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.45 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006912, upper bound: 0.0006815
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006912, upper bound: 0.0006886
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006878
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006879
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006916
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006984
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007406, 0.0011166, 0.0007513, 0.0011163, -0.0003758, 0.0003653
1: 0.9934148, 0.9943110, 0.9934182, 0.9942882, -0.0008734, 0.0008928
2: -0.0087180, -0.0053372, -0.0086809, -0.0053395, -0.0031048, 0.0030698
3: 0.0036146, 0.0041568, 0.0036280, 0.0041544, -0.0005398, 0.0005289
4: 0.0026352, 0.0053072, 0.0026370, 0.0052780, -0.0026427, 0.0026702
5: 0.0051172, 0.0064427, 0.0051426, 0.0064422, -0.0013250, 0.0013001
6: -0.0021472, -0.0009048, -0.0021343, -0.0009283, -0.0012189, 0.0012295
7: -0.0082623, -0.0074909, -0.0082619, -0.0075018, -0.0007605, 0.0007709
8: 0.0052640, 0.0097060, 0.0052671, 0.0096573, -0.0043200, 0.0043656
9: -0.0036844, -0.0031323, -0.0036844, -0.0031479, -0.0005366, 0.0005521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006913, upper bound: 0.0006815
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A1_A2

### Relational analysis result of IS_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006913, upper bound: 0.0006815
time: 0.70 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007728, 0.0011157, 0.0007513, 0.0011163, -0.0003435, 0.0003645
1: 0.9934254, 0.9942426, 0.9934182, 0.9942882, -0.0008628, 0.0008244
2: -0.0086064, -0.0052548, -0.0086809, -0.0053395, -0.0029999, 0.0031537
3: 0.0036550, 0.0041495, 0.0036280, 0.0041544, -0.0004994, 0.0005215
4: 0.0025701, 0.0052190, 0.0026370, 0.0052780, -0.0027078, 0.0025820
5: 0.0051937, 0.0064605, 0.0051426, 0.0064422, -0.0012485, 0.0013179
6: -0.0021084, -0.0009451, -0.0021343, -0.0009283, -0.0011801, 0.0011892
7: -0.0082779, -0.0075237, -0.0082619, -0.0075018, -0.0007761, 0.0007382
8: 0.0051557, 0.0095593, 0.0052671, 0.0096573, -0.0044285, 0.0042191
9: -0.0036853, -0.0031791, -0.0036844, -0.0031479, -0.0005374, 0.0005053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006903, upper bound: 0.0006863
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006910, upper bound: 0.0006883
time: 0.62 seconds

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007829, 0.0011155, 0.0007851, 0.0011154, -0.0003325, 0.0003303
1: 0.9934286, 0.9942212, 0.9934294, 0.9942166, -0.0007879, 0.0007918
2: -0.0085716, -0.0052299, -0.0085639, -0.0052390, -0.0030675, 0.0030708
3: 0.0036676, 0.0041472, 0.0036704, 0.0041466, -0.0004791, 0.0004768
4: 0.0025505, 0.0051915, 0.0025576, 0.0051855, -0.0026350, 0.0026339
5: 0.0052176, 0.0064659, 0.0052229, 0.0064639, -0.0012463, 0.0012430
6: -0.0020964, -0.0009365, -0.0020937, -0.0009396, -0.0011568, 0.0011572
7: -0.0082826, -0.0075339, -0.0082809, -0.0075362, -0.0007464, 0.0007470
8: 0.0051231, 0.0095136, 0.0051349, 0.0095036, -0.0043084, 0.0043074
9: -0.0036855, -0.0031937, -0.0036854, -0.0031969, -0.0004886, 0.0004918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006791, upper bound: 0.0006875
time: 0.73 seconds

## Relational analysis of IS_B1_B2_A1_B2

### Relational analysis result of IS_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006877
time: 0.76 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0007851, 0.0011154, -0.0003022, 0.0003296
1: 0.9934386, 0.9941570, 0.9934294, 0.9942166, -0.0007780, 0.0007277
2: -0.0084668, -0.0051286, -0.0085639, -0.0052390, -0.0029684, 0.0031714
3: 0.0037055, 0.0041402, 0.0036704, 0.0041466, -0.0004411, 0.0004698
4: 0.0024703, 0.0051087, 0.0025576, 0.0051855, -0.0027151, 0.0025511
5: 0.0052895, 0.0064878, 0.0052229, 0.0064639, -0.0011744, 0.0012649
6: -0.0020600, -0.0009013, -0.0020937, -0.0009396, -0.0011204, 0.0011924
7: -0.0083018, -0.0075647, -0.0082809, -0.0075362, -0.0007656, 0.0007162
8: 0.0049899, 0.0093759, 0.0051349, 0.0095036, -0.0044426, 0.0041714
9: -0.0036866, -0.0032376, -0.0036854, -0.0031969, -0.0004897, 0.0004478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
time: 0.80 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006878
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007513, 0.0011163, 0.0007728, 0.0011157, -0.0003645, 0.0003435
1: 0.9934182, 0.9942882, 0.9934254, 0.9942426, -0.0008244, 0.0008628
2: -0.0086809, -0.0053395, -0.0086064, -0.0052548, -0.0031537, 0.0029998
3: 0.0036280, 0.0041544, 0.0036550, 0.0041495, -0.0005215, 0.0004994
4: 0.0026370, 0.0052780, 0.0025701, 0.0052190, -0.0025820, 0.0027078
5: 0.0051426, 0.0064422, 0.0051937, 0.0064605, -0.0013179, 0.0012485
6: -0.0021343, -0.0009283, -0.0021084, -0.0009451, -0.0011892, 0.0011801
7: -0.0082619, -0.0075018, -0.0082779, -0.0075237, -0.0007382, 0.0007761
8: 0.0052671, 0.0096573, 0.0051557, 0.0095593, -0.0042191, 0.0044285
9: -0.0036844, -0.0031479, -0.0036853, -0.0031791, -0.0005053, 0.0005374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006860, upper bound: 0.0006906
time: 0.69 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006884, upper bound: 0.0006914
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007851, 0.0011154, 0.0007853, 0.0011154, -0.0003303, 0.0003301
1: 0.9934294, 0.9942166, 0.9934294, 0.9942162, -0.0007868, 0.0007871
2: -0.0085639, -0.0052390, -0.0085634, -0.0052572, -0.0030562, 0.0030677
3: 0.0036704, 0.0041466, 0.0036706, 0.0041466, -0.0004762, 0.0004761
4: 0.0025576, 0.0051855, 0.0025720, 0.0051850, -0.0026275, 0.0026134
5: 0.0052229, 0.0064639, 0.0052232, 0.0064600, -0.0012371, 0.0012407
6: -0.0020937, -0.0009396, -0.0020935, -0.0009459, -0.0011478, 0.0011539
7: -0.0082809, -0.0075362, -0.0082774, -0.0075363, -0.0007446, 0.0007413
8: 0.0051349, 0.0095036, 0.0051589, 0.0095028, -0.0042987, 0.0042770
9: -0.0036854, -0.0031969, -0.0036852, -0.0031971, -0.0004883, 0.0004884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006791
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006877, upper bound: 0.0006815
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007829, 0.0011155, 0.0007728, 0.0011157, -0.0003329, 0.0003426
1: 0.9934287, 0.9942213, 0.9934254, 0.9942426, -0.0008139, 0.0007959
2: -0.0085717, -0.0052573, -0.0086064, -0.0052548, -0.0030448, 0.0030813
3: 0.0036676, 0.0041472, 0.0036550, 0.0041495, -0.0004819, 0.0004922
4: 0.0025721, 0.0051916, 0.0025701, 0.0052190, -0.0026469, 0.0026215
5: 0.0052175, 0.0064600, 0.0051937, 0.0064605, -0.0012430, 0.0012662
6: -0.0020964, -0.0009459, -0.0021084, -0.0009451, -0.0011513, 0.0011625
7: -0.0082774, -0.0075339, -0.0082779, -0.0075237, -0.0007537, 0.0007440
8: 0.0051591, 0.0095138, 0.0051557, 0.0095593, -0.0043276, 0.0042854
9: -0.0036852, -0.0031936, -0.0036853, -0.0031791, -0.0005061, 0.0004916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.65 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0007853, 0.0011154, -0.0003022, 0.0003294
1: 0.9934386, 0.9941571, 0.9934294, 0.9942162, -0.0007776, 0.0007277
2: -0.0084668, -0.0051481, -0.0085634, -0.0052572, -0.0029607, 0.0031580
3: 0.0037055, 0.0041402, 0.0036706, 0.0041466, -0.0004411, 0.0004696
4: 0.0024858, 0.0051087, 0.0025720, 0.0051850, -0.0026992, 0.0025367
5: 0.0052895, 0.0064836, 0.0052232, 0.0064600, -0.0011705, 0.0012603
6: -0.0020600, -0.0009080, -0.0020935, -0.0009459, -0.0011141, 0.0011855
7: -0.0082981, -0.0075647, -0.0082774, -0.0075363, -0.0007617, 0.0007128
8: 0.0050156, 0.0093760, 0.0051589, 0.0095028, -0.0044182, 0.0041505
9: -0.0036864, -0.0032376, -0.0036852, -0.0031971, -0.0004892, 0.0004477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.68 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.12 seconds
IS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006913, upper bound: 0.0006815
IS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006913, upper bound: 0.0006815
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006903, upper bound: 0.0006863
IS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006910, upper bound: 0.0006883
IS_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006791, upper bound: 0.0006875
IS_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006877
IS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
IS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006878
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006860, upper bound: 0.0006906
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006884, upper bound: 0.0006914
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006791
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006877, upper bound: 0.0006815
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894

## BFS IS instance: IS_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007513, 0.0011163, 0.0007513, 0.0011163, -0.0003651, 0.0003651
1: 0.9934182, 0.9942882, 0.9934182, 0.9942882, -0.0008700, 0.0008700
2: -0.0086809, -0.0053395, -0.0086809, -0.0053395, -0.0030674, 0.0030674
3: 0.0036280, 0.0041544, 0.0036280, 0.0041544, -0.0005264, 0.0005264
4: 0.0026370, 0.0052780, 0.0026370, 0.0052780, -0.0026409, 0.0026409
5: 0.0051426, 0.0064422, 0.0051426, 0.0064422, -0.0012996, 0.0012996
6: -0.0021343, -0.0009283, -0.0021343, -0.0009283, -0.0012060, 0.0012060
7: -0.0082619, -0.0075018, -0.0082619, -0.0075018, -0.0007600, 0.0007600
8: 0.0052671, 0.0096573, 0.0052671, 0.0096573, -0.0043169, 0.0043169
9: -0.0036844, -0.0031479, -0.0036844, -0.0031479, -0.0005365, 0.0005365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006903, upper bound: 0.0006791
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006910, upper bound: 0.0006815
time: 1.15 seconds

## BFS IS instance: IS_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007851, 0.0011154, 0.0007513, 0.0011163, -0.0003312, 0.0003642
1: 0.9934294, 0.9942166, 0.9934182, 0.9942882, -0.0008588, 0.0007983
2: -0.0085639, -0.0052390, -0.0086809, -0.0053395, -0.0029610, 0.0031760
3: 0.0036704, 0.0041466, 0.0036280, 0.0041544, -0.0004840, 0.0005187
4: 0.0025576, 0.0051855, 0.0026370, 0.0052780, -0.0027204, 0.0025484
5: 0.0052229, 0.0064639, 0.0051426, 0.0064422, -0.0012193, 0.0013213
6: -0.0020937, -0.0009396, -0.0021343, -0.0009283, -0.0011654, 0.0011948
7: -0.0082809, -0.0075362, -0.0082619, -0.0075018, -0.0007791, 0.0007257
8: 0.0051349, 0.0095036, 0.0052671, 0.0096573, -0.0044508, 0.0041643
9: -0.0036854, -0.0031969, -0.0036844, -0.0031479, -0.0005376, 0.0004875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_A2_A1

### Relational analysis result of IS_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006903, upper bound: 0.0006791
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2

### Relational analysis result of IS_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006910, upper bound: 0.0006815
time: 0.64 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007704, 0.0011158, 0.0007513, 0.0011163, -0.0003459, 0.0003645
1: 0.9934245, 0.9942477, 0.9934182, 0.9942881, -0.0008637, 0.0008295
2: -0.0086146, -0.0053143, -0.0086807, -0.0053555, -0.0029925, 0.0030948
3: 0.0036520, 0.0041500, 0.0036280, 0.0041544, -0.0005024, 0.0005220
4: 0.0026172, 0.0052256, 0.0026497, 0.0052778, -0.0026606, 0.0025759
5: 0.0051881, 0.0064477, 0.0051427, 0.0064388, -0.0012507, 0.0013049
6: -0.0021113, -0.0009657, -0.0021342, -0.0009284, -0.0011829, 0.0011685
7: -0.0082666, -0.0075213, -0.0082589, -0.0075019, -0.0007647, 0.0007376
8: 0.0052340, 0.0095702, 0.0052880, 0.0096570, -0.0043504, 0.0042098
9: -0.0036847, -0.0031756, -0.0036842, -0.0031479, -0.0005367, 0.0005086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006840, upper bound: 0.0006818
time: 0.71 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006827
time: 0.70 seconds

## BFS IS instance: IS_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007731, 0.0011157, 0.0007513, 0.0011163, -0.0003432, 0.0003644
1: 0.9934254, 0.9942420, 0.9934182, 0.9942882, -0.0008628, 0.0008238
2: -0.0086054, -0.0052995, -0.0086807, -0.0053491, -0.0029892, 0.0031070
3: 0.0036553, 0.0041494, 0.0036281, 0.0041544, -0.0004990, 0.0005213
4: 0.0026054, 0.0052182, 0.0026447, 0.0052778, -0.0026723, 0.0025736
5: 0.0051944, 0.0064509, 0.0051427, 0.0064401, -0.0012457, 0.0013081
6: -0.0021081, -0.0009606, -0.0021343, -0.0009284, -0.0011797, 0.0011737
7: -0.0082695, -0.0075240, -0.0082601, -0.0075019, -0.0007676, 0.0007361
8: 0.0052145, 0.0095580, 0.0052797, 0.0096570, -0.0043692, 0.0042052
9: -0.0036848, -0.0031795, -0.0036843, -0.0031479, -0.0005369, 0.0005048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006842
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006882, upper bound: 0.0006852
time: 0.74 seconds

## BFS IS instance: IS_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0007822, 0.0011155, -0.0003325, 0.0003333
1: 0.9934287, 0.9942211, 0.9934284, 0.9942229, -0.0007942, 0.0007927
2: -0.0085713, -0.0052460, -0.0085742, -0.0052968, -0.0030092, 0.0030679
3: 0.0036677, 0.0041471, 0.0036666, 0.0041473, -0.0004797, 0.0004805
4: 0.0025632, 0.0051913, 0.0026033, 0.0051936, -0.0026304, 0.0025880
5: 0.0052178, 0.0064624, 0.0052158, 0.0064514, -0.0012337, 0.0012466
6: -0.0020963, -0.0009420, -0.0020973, -0.0009597, -0.0011366, 0.0011552
7: -0.0082796, -0.0075340, -0.0082700, -0.0075331, -0.0007464, 0.0007360
8: 0.0051442, 0.0095133, 0.0052109, 0.0095171, -0.0043022, 0.0042306
9: -0.0036854, -0.0031938, -0.0036848, -0.0031926, -0.0004928, 0.0004911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_B2_A1_B1_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006877
time: 0.74 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006898
time: 0.69 seconds

## BFS IS instance: IS_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0007854, 0.0011154, -0.0003324, 0.0003301
1: 0.9934287, 0.9942211, 0.9934294, 0.9942161, -0.0007874, 0.0007917
2: -0.0085714, -0.0052396, -0.0085631, -0.0052859, -0.0030200, 0.0030603
3: 0.0036677, 0.0041471, 0.0036706, 0.0041466, -0.0004789, 0.0004765
4: 0.0025581, 0.0051914, 0.0025947, 0.0051848, -0.0026267, 0.0025966
5: 0.0052178, 0.0064638, 0.0052234, 0.0064538, -0.0012360, 0.0012404
6: -0.0020963, -0.0009398, -0.0020934, -0.0009559, -0.0011404, 0.0011536
7: -0.0082808, -0.0075340, -0.0082720, -0.0075364, -0.0007444, 0.0007380
8: 0.0051358, 0.0095134, 0.0051967, 0.0095025, -0.0042947, 0.0042453
9: -0.0036854, -0.0031938, -0.0036850, -0.0031972, -0.0004882, 0.0004912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_B2_A1_B2_A1

### Relational analysis result of IS_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006885
time: 0.68 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2

### Relational analysis result of IS_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006905
time: 0.70 seconds

## BFS IS instance: IS_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007851, 0.0011154, 0.0007851, 0.0011154, -0.0003303, 0.0003303
1: 0.9934294, 0.9942166, 0.9934294, 0.9942166, -0.0007872, 0.0007872
2: -0.0085639, -0.0052390, -0.0085639, -0.0052390, -0.0030609, 0.0030609
3: 0.0036704, 0.0041466, 0.0036704, 0.0041466, -0.0004763, 0.0004763
4: 0.0025576, 0.0051855, 0.0025576, 0.0051855, -0.0026279, 0.0026279
5: 0.0052229, 0.0064639, 0.0052229, 0.0064639, -0.0012411, 0.0012411
6: -0.0020937, -0.0009396, -0.0020937, -0.0009396, -0.0011541, 0.0011541
7: -0.0082809, -0.0075362, -0.0082809, -0.0075362, -0.0007447, 0.0007447
8: 0.0051349, 0.0095036, 0.0051349, 0.0095036, -0.0042975, 0.0042975
9: -0.0036854, -0.0031969, -0.0036854, -0.0031969, -0.0004885, 0.0004885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A2_A1_B1

### Relational analysis result of IS_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006791, upper bound: 0.0006815
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0007851, 0.0011154, -0.0003022, 0.0003296
1: 0.9934386, 0.9941571, 0.9934294, 0.9942166, -0.0007780, 0.0007277
2: -0.0084668, -0.0051481, -0.0085639, -0.0052390, -0.0029684, 0.0031550
3: 0.0037055, 0.0041402, 0.0036704, 0.0041466, -0.0004411, 0.0004698
4: 0.0024858, 0.0051087, 0.0025576, 0.0051855, -0.0026996, 0.0025511
5: 0.0052895, 0.0064836, 0.0052229, 0.0064639, -0.0011744, 0.0012607
6: -0.0020600, -0.0009080, -0.0020937, -0.0009396, -0.0011204, 0.0011857
7: -0.0082981, -0.0075647, -0.0082809, -0.0075362, -0.0007619, 0.0007162
8: 0.0050156, 0.0093760, 0.0051349, 0.0095036, -0.0044178, 0.0041714
9: -0.0036864, -0.0032376, -0.0036854, -0.0031969, -0.0004895, 0.0004478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006850
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006877
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007513, 0.0011163, 0.0007704, 0.0011158, -0.0003645, 0.0003459
1: 0.9934182, 0.9942881, 0.9934245, 0.9942477, -0.0008295, 0.0008637
2: -0.0086807, -0.0053555, -0.0086146, -0.0053143, -0.0030948, 0.0029925
3: 0.0036280, 0.0041544, 0.0036520, 0.0041500, -0.0005220, 0.0005024
4: 0.0026497, 0.0052778, 0.0026172, 0.0052256, -0.0025759, 0.0026606
5: 0.0051427, 0.0064388, 0.0051881, 0.0064477, -0.0013049, 0.0012507
6: -0.0021342, -0.0009284, -0.0021113, -0.0009657, -0.0011685, 0.0011829
7: -0.0082589, -0.0075019, -0.0082666, -0.0075213, -0.0007376, 0.0007647
8: 0.0052880, 0.0096570, 0.0052340, 0.0095702, -0.0042098, 0.0043504
9: -0.0036842, -0.0031479, -0.0036847, -0.0031756, -0.0005086, 0.0005367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006819, upper bound: 0.0006840
time: 0.61 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006826, upper bound: 0.0006874
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007513, 0.0011163, 0.0007731, 0.0011157, -0.0003644, 0.0003432
1: 0.9934182, 0.9942882, 0.9934254, 0.9942420, -0.0008238, 0.0008628
2: -0.0086807, -0.0053491, -0.0086054, -0.0052995, -0.0031070, 0.0029892
3: 0.0036281, 0.0041544, 0.0036553, 0.0041494, -0.0005213, 0.0004990
4: 0.0026447, 0.0052778, 0.0026054, 0.0052182, -0.0025736, 0.0026723
5: 0.0051427, 0.0064401, 0.0051944, 0.0064509, -0.0013081, 0.0012457
6: -0.0021343, -0.0009284, -0.0021081, -0.0009606, -0.0011737, 0.0011797
7: -0.0082601, -0.0075019, -0.0082695, -0.0075240, -0.0007361, 0.0007676
8: 0.0052797, 0.0096570, 0.0052145, 0.0095580, -0.0042052, 0.0043692
9: -0.0036843, -0.0031479, -0.0036848, -0.0031795, -0.0005048, 0.0005369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006842, upper bound: 0.0006850
time: 0.61 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006882
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007822, 0.0011155, 0.0007854, 0.0011154, -0.0003333, 0.0003301
1: 0.9934284, 0.9942229, 0.9934295, 0.9942160, -0.0007876, 0.0007935
2: -0.0085742, -0.0052968, -0.0085631, -0.0052734, -0.0030533, 0.0030095
3: 0.0036666, 0.0041473, 0.0036706, 0.0041466, -0.0004800, 0.0004767
4: 0.0026033, 0.0051936, 0.0025848, 0.0051849, -0.0025815, 0.0026088
5: 0.0052158, 0.0064514, 0.0052234, 0.0064565, -0.0012407, 0.0012280
6: -0.0020973, -0.0009597, -0.0020934, -0.0009515, -0.0011458, 0.0011338
7: -0.0082700, -0.0075331, -0.0082744, -0.0075364, -0.0007336, 0.0007412
8: 0.0052109, 0.0095171, 0.0051801, 0.0095025, -0.0042220, 0.0042701
9: -0.0036848, -0.0031926, -0.0036851, -0.0031972, -0.0004876, 0.0004925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006791
time: 0.60 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006791
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007854, 0.0011154, 0.0007854, 0.0011154, -0.0003300, 0.0003301
1: 0.9934294, 0.9942161, 0.9934295, 0.9942160, -0.0007866, 0.0007867
2: -0.0085631, -0.0052859, -0.0085631, -0.0052668, -0.0030456, 0.0030205
3: 0.0036706, 0.0041466, 0.0036706, 0.0041466, -0.0004759, 0.0004760
4: 0.0025947, 0.0051848, 0.0025796, 0.0051849, -0.0025901, 0.0026052
5: 0.0052234, 0.0064538, 0.0052234, 0.0064579, -0.0012345, 0.0012304
6: -0.0020934, -0.0009559, -0.0020934, -0.0009492, -0.0011442, 0.0011376
7: -0.0082720, -0.0075364, -0.0082756, -0.0075364, -0.0007356, 0.0007392
8: 0.0051967, 0.0095025, 0.0051716, 0.0095026, -0.0042368, 0.0042634
9: -0.0036850, -0.0031972, -0.0036851, -0.0031972, -0.0004877, 0.0004879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006828, upper bound: 0.0006782
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006842, upper bound: 0.0006782
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007829, 0.0011155, 0.0007829, 0.0011155, -0.0003326, 0.0003326
1: 0.9934287, 0.9942213, 0.9934287, 0.9942213, -0.0007926, 0.0007926
2: -0.0085717, -0.0052573, -0.0085717, -0.0052573, -0.0030423, 0.0030423
3: 0.0036676, 0.0041472, 0.0036676, 0.0041472, -0.0004796, 0.0004796
4: 0.0025721, 0.0051916, 0.0025721, 0.0051916, -0.0026195, 0.0026195
5: 0.0052175, 0.0064600, 0.0052175, 0.0064600, -0.0012425, 0.0012425
6: -0.0020964, -0.0009459, -0.0020964, -0.0009459, -0.0011505, 0.0011505
7: -0.0082774, -0.0075339, -0.0082774, -0.0075339, -0.0007436, 0.0007436
8: 0.0051591, 0.0095138, 0.0051591, 0.0095138, -0.0042821, 0.0042821
9: -0.0036852, -0.0031936, -0.0036852, -0.0031936, -0.0004916, 0.0004916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006967
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006980
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007829, 0.0011155, 0.0008133, 0.0011147, -0.0003318, 0.0003022
1: 0.9934287, 0.9942213, 0.9934386, 0.9941571, -0.0007284, 0.0007827
2: -0.0085717, -0.0052573, -0.0084668, -0.0051481, -0.0031597, 0.0029506
3: 0.0036676, 0.0041472, 0.0037055, 0.0041402, -0.0004727, 0.0004416
4: 0.0025721, 0.0051916, 0.0024858, 0.0051087, -0.0025366, 0.0027058
5: 0.0052175, 0.0064600, 0.0052895, 0.0064836, -0.0012661, 0.0011705
6: -0.0020964, -0.0009459, -0.0020600, -0.0009080, -0.0011883, 0.0011141
7: -0.0082774, -0.0075339, -0.0082981, -0.0075647, -0.0007128, 0.0007642
8: 0.0051591, 0.0095138, 0.0050156, 0.0093760, -0.0041468, 0.0044277
9: -0.0036852, -0.0031936, -0.0036864, -0.0032376, -0.0004477, 0.0004927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006974
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006980
time: 0.63 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0007829, 0.0011155, -0.0003022, 0.0003318
1: 0.9934386, 0.9941571, 0.9934287, 0.9942213, -0.0007827, 0.0007284
2: -0.0084668, -0.0051481, -0.0085717, -0.0052573, -0.0029506, 0.0031597
3: 0.0037055, 0.0041402, 0.0036676, 0.0041472, -0.0004416, 0.0004727
4: 0.0024858, 0.0051087, 0.0025721, 0.0051916, -0.0027058, 0.0025366
5: 0.0052895, 0.0064836, 0.0052175, 0.0064600, -0.0011705, 0.0012661
6: -0.0020600, -0.0009080, -0.0020964, -0.0009459, -0.0011141, 0.0011883
7: -0.0082981, -0.0075647, -0.0082774, -0.0075339, -0.0007642, 0.0007128
8: 0.0050156, 0.0093760, 0.0051591, 0.0095138, -0.0044277, 0.0041468
9: -0.0036864, -0.0032376, -0.0036852, -0.0031936, -0.0004927, 0.0004477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006879
time: 0.65 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0008133, 0.0011147, -0.0003014, 0.0003014
1: 0.9934386, 0.9941571, 0.9934386, 0.9941571, -0.0007185, 0.0007185
2: -0.0084668, -0.0051481, -0.0084668, -0.0051481, -0.0030584, 0.0030584
3: 0.0037055, 0.0041402, 0.0037055, 0.0041402, -0.0004347, 0.0004347
4: 0.0024858, 0.0051087, 0.0024858, 0.0051087, -0.0026229, 0.0026229
5: 0.0052895, 0.0064836, 0.0052895, 0.0064836, -0.0011941, 0.0011941
6: -0.0020600, -0.0009080, -0.0020600, -0.0009080, -0.0011519, 0.0011519
7: -0.0082981, -0.0075647, -0.0082981, -0.0075647, -0.0007334, 0.0007334
8: 0.0050156, 0.0093760, 0.0050156, 0.0093760, -0.0042908, 0.0042908
9: -0.0036864, -0.0032376, -0.0036864, -0.0032376, -0.0004488, 0.0004488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006894
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.15 seconds
IS_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006903, upper bound: 0.0006791
IS_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006910, upper bound: 0.0006815
IS_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006903, upper bound: 0.0006791
IS_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006910, upper bound: 0.0006815
IS_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006840, upper bound: 0.0006818
IS_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006827
IS_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006842
IS_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006882, upper bound: 0.0006852
IS_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006877
IS_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006898
IS_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006885
IS_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006905
IS_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006791, upper bound: 0.0006815
IS_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006815
IS_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006850
IS_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006877
IS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006819, upper bound: 0.0006840
IS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006826, upper bound: 0.0006874
IS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006842, upper bound: 0.0006850
IS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006882
IS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006791
IS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006791
IS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006828, upper bound: 0.0006782
IS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006842, upper bound: 0.0006782
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006967
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006980
IS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006974
IS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006980
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006879
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894
IS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006894
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006894

## BFS IS instance: IS_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007499, 0.0011163, 0.0007513, 0.0011163, -0.0003664, 0.0003650
1: 0.9934179, 0.9942911, 0.9934182, 0.9942881, -0.0008703, 0.0008729
2: -0.0086855, -0.0053988, -0.0086807, -0.0053555, -0.0030555, 0.0030077
3: 0.0036263, 0.0041547, 0.0036280, 0.0041544, -0.0005281, 0.0005267
4: 0.0026839, 0.0052816, 0.0026497, 0.0052778, -0.0025939, 0.0026319
5: 0.0051394, 0.0064294, 0.0051427, 0.0064388, -0.0012993, 0.0012866
6: -0.0021359, -0.0009254, -0.0021342, -0.0009284, -0.0012075, 0.0012089
7: -0.0082507, -0.0075005, -0.0082589, -0.0075019, -0.0007488, 0.0007584
8: 0.0053450, 0.0096634, 0.0052880, 0.0096570, -0.0042387, 0.0043019
9: -0.0036838, -0.0031459, -0.0036842, -0.0031479, -0.0005359, 0.0005383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006963, upper bound: 0.0006900
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006960
time: 0.74 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007515, 0.0011163, 0.0007513, 0.0011163, -0.0003648, 0.0003650
1: 0.9934184, 0.9942878, 0.9934182, 0.9942882, -0.0008698, 0.0008696
2: -0.0086801, -0.0053846, -0.0086807, -0.0053491, -0.0030570, 0.0030207
3: 0.0036283, 0.0041543, 0.0036281, 0.0041544, -0.0005261, 0.0005263
4: 0.0026727, 0.0052773, 0.0026447, 0.0052778, -0.0026051, 0.0026326
5: 0.0051432, 0.0064325, 0.0051427, 0.0064401, -0.0012970, 0.0012897
6: -0.0021340, -0.0009288, -0.0021343, -0.0009284, -0.0012056, 0.0012054
7: -0.0082534, -0.0075021, -0.0082601, -0.0075019, -0.0007515, 0.0007580
8: 0.0053263, 0.0096562, 0.0052797, 0.0096570, -0.0042571, 0.0043032
9: -0.0036839, -0.0031482, -0.0036843, -0.0031479, -0.0005360, 0.0005361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006963, upper bound: 0.0006915
time: 0.75 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006964
time: 0.65 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007822, 0.0011155, 0.0007513, 0.0011163, -0.0003341, 0.0003642
1: 0.9934284, 0.9942229, 0.9934182, 0.9942881, -0.0008597, 0.0008047
2: -0.0085742, -0.0052968, -0.0086807, -0.0053555, -0.0029581, 0.0031177
3: 0.0036666, 0.0041473, 0.0036280, 0.0041544, -0.0004877, 0.0005193
4: 0.0026033, 0.0051936, 0.0026497, 0.0052778, -0.0026745, 0.0025439
5: 0.0052158, 0.0064514, 0.0051427, 0.0064388, -0.0012230, 0.0013087
6: -0.0020973, -0.0009597, -0.0021342, -0.0009284, -0.0011688, 0.0011746
7: -0.0082700, -0.0075331, -0.0082589, -0.0075019, -0.0007681, 0.0007257
8: 0.0052109, 0.0095171, 0.0052880, 0.0096570, -0.0043741, 0.0041581
9: -0.0036848, -0.0031926, -0.0036842, -0.0031479, -0.0005369, 0.0004917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006838, upper bound: 0.0006763
time: 0.70 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006872, upper bound: 0.0006763
time: 0.66 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007854, 0.0011154, 0.0007513, 0.0011163, -0.0003309, 0.0003641
1: 0.9934294, 0.9942161, 0.9934182, 0.9942882, -0.0008588, 0.0007979
2: -0.0085631, -0.0052859, -0.0086807, -0.0053491, -0.0029505, 0.0031283
3: 0.0036706, 0.0041466, 0.0036281, 0.0041544, -0.0004837, 0.0005185
4: 0.0025947, 0.0051848, 0.0026447, 0.0052778, -0.0026830, 0.0025402
5: 0.0052234, 0.0064538, 0.0051427, 0.0064401, -0.0012167, 0.0013111
6: -0.0020934, -0.0009559, -0.0021343, -0.0009284, -0.0011650, 0.0011784
7: -0.0082720, -0.0075364, -0.0082601, -0.0075019, -0.0007701, 0.0007237
8: 0.0051967, 0.0095025, 0.0052797, 0.0096570, -0.0043888, 0.0041507
9: -0.0036850, -0.0031972, -0.0036843, -0.0031479, -0.0005370, 0.0004871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006848, upper bound: 0.0006782
time: 0.67 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006878, upper bound: 0.0006782
time: 0.60 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007754, 0.0011157, 0.0007687, 0.0011159, -0.0003404, 0.0003469
1: 0.9934261, 0.9942371, 0.9934240, 0.9942514, -0.0008252, 0.0008131
2: -0.0085975, -0.0053187, -0.0086206, -0.0053494, -0.0029784, 0.0030306
3: 0.0036582, 0.0041489, 0.0036498, 0.0041504, -0.0004922, 0.0004990
4: 0.0026206, 0.0052120, 0.0026449, 0.0052303, -0.0026096, 0.0025671
5: 0.0051998, 0.0064467, 0.0051840, 0.0064401, -0.0012402, 0.0012627
6: -0.0021054, -0.0009673, -0.0021134, -0.0009666, -0.0011388, 0.0011461
7: -0.0082658, -0.0075263, -0.0082600, -0.0075195, -0.0007463, 0.0007337
8: 0.0052397, 0.0095477, 0.0052801, 0.0095780, -0.0042658, 0.0041949
9: -0.0036846, -0.0031828, -0.0036843, -0.0031731, -0.0005115, 0.0005015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006730, upper bound: 0.0006697
time: 0.71 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006837, upper bound: 0.0006819
time: 0.65 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007727, 0.0011158, 0.0007641, 0.0011160, -0.0003433, 0.0003516
1: 0.9934253, 0.9942430, 0.9934224, 0.9942610, -0.0008357, 0.0008206
2: -0.0086070, -0.0053157, -0.0086365, -0.0053636, -0.0029773, 0.0030478
3: 0.0036547, 0.0041495, 0.0036441, 0.0041514, -0.0004967, 0.0005054
4: 0.0026183, 0.0052195, 0.0026561, 0.0052428, -0.0026246, 0.0025634
5: 0.0051933, 0.0064473, 0.0051731, 0.0064370, -0.0012437, 0.0012743
6: -0.0021087, -0.0009662, -0.0021189, -0.0009565, -0.0011522, 0.0011527
7: -0.0082664, -0.0075235, -0.0082573, -0.0075149, -0.0007515, 0.0007338
8: 0.0052358, 0.0095602, 0.0052987, 0.0095989, -0.0042902, 0.0041892
9: -0.0036846, -0.0031788, -0.0036842, -0.0031665, -0.0005182, 0.0005053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006827
time: 0.71 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006827
time: 0.72 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007781, 0.0011156, 0.0007687, 0.0011159, -0.0003378, 0.0003469
1: 0.9934270, 0.9942315, 0.9934240, 0.9942513, -0.0008243, 0.0008075
2: -0.0085883, -0.0053041, -0.0086206, -0.0053440, -0.0029749, 0.0030430
3: 0.0036615, 0.0041483, 0.0036498, 0.0041504, -0.0004889, 0.0004984
4: 0.0026091, 0.0052047, 0.0026406, 0.0052303, -0.0026212, 0.0025641
5: 0.0052061, 0.0064499, 0.0051840, 0.0064412, -0.0012351, 0.0012659
6: -0.0021022, -0.0009622, -0.0021134, -0.0009665, -0.0011356, 0.0011512
7: -0.0082686, -0.0075290, -0.0082610, -0.0075195, -0.0007491, 0.0007320
8: 0.0052205, 0.0095356, 0.0052730, 0.0095781, -0.0042844, 0.0041893
9: -0.0036848, -0.0031867, -0.0036844, -0.0031731, -0.0005116, 0.0004977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006768, upper bound: 0.0006738
time: 0.60 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006848, upper bound: 0.0006842
time: 0.65 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007754, 0.0011157, 0.0007641, 0.0011160, -0.0003406, 0.0003516
1: 0.9934261, 0.9942372, 0.9934224, 0.9942611, -0.0008349, 0.0008148
2: -0.0085977, -0.0053010, -0.0086365, -0.0053572, -0.0029738, 0.0030599
3: 0.0036581, 0.0041489, 0.0036441, 0.0041515, -0.0004933, 0.0005048
4: 0.0026066, 0.0052122, 0.0026511, 0.0052428, -0.0026362, 0.0025611
5: 0.0051997, 0.0064505, 0.0051731, 0.0064384, -0.0012387, 0.0012775
6: -0.0021054, -0.0009611, -0.0021189, -0.0009565, -0.0011490, 0.0011578
7: -0.0082692, -0.0075263, -0.0082585, -0.0075148, -0.0007543, 0.0007323
8: 0.0052164, 0.0095480, 0.0052904, 0.0095990, -0.0043089, 0.0041848
9: -0.0036848, -0.0031827, -0.0036842, -0.0031665, -0.0005183, 0.0005015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006882, upper bound: 0.0006852
time: 0.81 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006882, upper bound: 0.0006852
time: 0.73 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008007, 0.0011150, 0.0007866, 0.0011154, -0.0003147, 0.0003284
1: 0.9934345, 0.9941837, 0.9934298, 0.9942135, -0.0007790, 0.0007539
2: -0.0085104, -0.0052421, -0.0085588, -0.0053007, -0.0029443, 0.0030539
3: 0.0036898, 0.0041431, 0.0036722, 0.0041463, -0.0004565, 0.0004709
4: 0.0025601, 0.0051431, 0.0026064, 0.0051815, -0.0026214, 0.0025367
5: 0.0052596, 0.0064633, 0.0052263, 0.0064506, -0.0011910, 0.0012369
6: -0.0020751, -0.0009407, -0.0020920, -0.0009610, -0.0011141, 0.0011513
7: -0.0082803, -0.0075519, -0.0082692, -0.0075376, -0.0007426, 0.0007173
8: 0.0051391, 0.0094332, 0.0052161, 0.0094969, -0.0042868, 0.0041454
9: -0.0036854, -0.0032193, -0.0036848, -0.0031990, -0.0004864, 0.0004655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006744
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006873
time: 0.69 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007956, 0.0011152, 0.0007847, 0.0011154, -0.0003199, 0.0003305
1: 0.9934328, 0.9941945, 0.9934292, 0.9942175, -0.0007848, 0.0007653
2: -0.0085280, -0.0052540, -0.0085656, -0.0052982, -0.0029658, 0.0030515
3: 0.0036834, 0.0041443, 0.0036698, 0.0041468, -0.0004634, 0.0004745
4: 0.0025695, 0.0051570, 0.0026044, 0.0051868, -0.0026173, 0.0025526
5: 0.0052475, 0.0064607, 0.0052217, 0.0064511, -0.0012036, 0.0012390
6: -0.0020812, -0.0009448, -0.0020943, -0.0009601, -0.0011211, 0.0011495
7: -0.0082781, -0.0075467, -0.0082697, -0.0075357, -0.0007424, 0.0007230
8: 0.0051547, 0.0094563, 0.0052128, 0.0095058, -0.0042807, 0.0041715
9: -0.0036853, -0.0032120, -0.0036848, -0.0031962, -0.0004891, 0.0004729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A1_B1_A2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006872
time: 0.74 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006898
time: 0.71 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008006, 0.0011150, 0.0007901, 0.0011153, -0.0003147, 0.0003249
1: 0.9934345, 0.9941837, 0.9934310, 0.9942060, -0.0007715, 0.0007527
2: -0.0085104, -0.0052360, -0.0085467, -0.0052900, -0.0029544, 0.0030458
3: 0.0036897, 0.0041431, 0.0036766, 0.0041455, -0.0004558, 0.0004665
4: 0.0025553, 0.0051432, 0.0025979, 0.0051719, -0.0026166, 0.0025453
5: 0.0052596, 0.0064646, 0.0052347, 0.0064529, -0.0011933, 0.0012299
6: -0.0020751, -0.0009386, -0.0020877, -0.0009573, -0.0011178, 0.0011492
7: -0.0082814, -0.0075519, -0.0082712, -0.0075412, -0.0007402, 0.0007194
8: 0.0051311, 0.0094332, 0.0052020, 0.0094810, -0.0042772, 0.0041599
9: -0.0036855, -0.0032193, -0.0036849, -0.0032041, -0.0004814, 0.0004656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006848
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006885
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007956, 0.0011152, 0.0007878, 0.0011154, -0.0003198, 0.0003274
1: 0.9934328, 0.9941946, 0.9934303, 0.9942110, -0.0007782, 0.0007643
2: -0.0085280, -0.0052476, -0.0085548, -0.0052873, -0.0029762, 0.0030443
3: 0.0036834, 0.0041443, 0.0036736, 0.0041460, -0.0004627, 0.0004706
4: 0.0025644, 0.0051571, 0.0025958, 0.0051783, -0.0026139, 0.0025612
5: 0.0052475, 0.0064621, 0.0052291, 0.0064535, -0.0012060, 0.0012330
6: -0.0020812, -0.0009426, -0.0020905, -0.0009564, -0.0011249, 0.0011480
7: -0.0082793, -0.0075467, -0.0082717, -0.0075388, -0.0007404, 0.0007250
8: 0.0051463, 0.0094564, 0.0051985, 0.0094916, -0.0042740, 0.0041860
9: -0.0036853, -0.0032120, -0.0036849, -0.0032007, -0.0004846, 0.0004730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006878
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006905
time: 0.68 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007852, 0.0011154, 0.0007822, 0.0011155, -0.0003303, 0.0003333
1: 0.9934294, 0.9942164, 0.9934284, 0.9942229, -0.0007936, 0.0007880
2: -0.0085637, -0.0052544, -0.0085742, -0.0052968, -0.0030024, 0.0030584
3: 0.0036704, 0.0041466, 0.0036666, 0.0041473, -0.0004769, 0.0004800
4: 0.0025698, 0.0051853, 0.0026033, 0.0051936, -0.0026238, 0.0025820
5: 0.0052230, 0.0064606, 0.0052158, 0.0064514, -0.0012284, 0.0012448
6: -0.0020936, -0.0009449, -0.0020973, -0.0009597, -0.0011340, 0.0011523
7: -0.0082780, -0.0075362, -0.0082700, -0.0075331, -0.0007448, 0.0007337
8: 0.0051552, 0.0095033, 0.0052109, 0.0095171, -0.0042914, 0.0042207
9: -0.0036853, -0.0031970, -0.0036848, -0.0031926, -0.0004927, 0.0004878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006713, upper bound: 0.0006782
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006782
time: 0.62 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007852, 0.0011154, 0.0007854, 0.0011154, -0.0003302, 0.0003300
1: 0.9934294, 0.9942163, 0.9934294, 0.9942161, -0.0007868, 0.0007869
2: -0.0085637, -0.0052490, -0.0085631, -0.0052859, -0.0030140, 0.0030502
3: 0.0036704, 0.0041466, 0.0036706, 0.0041466, -0.0004762, 0.0004760
4: 0.0025655, 0.0051853, 0.0025947, 0.0051848, -0.0026193, 0.0025906
5: 0.0052230, 0.0064618, 0.0052234, 0.0064538, -0.0012308, 0.0012384
6: -0.0020936, -0.0009431, -0.0020934, -0.0009559, -0.0011378, 0.0011504
7: -0.0082790, -0.0075362, -0.0082720, -0.0075364, -0.0007426, 0.0007358
8: 0.0051481, 0.0095034, 0.0051967, 0.0095025, -0.0042833, 0.0042355
9: -0.0036853, -0.0031970, -0.0036850, -0.0031972, -0.0004881, 0.0004880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006782
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006782
time: 0.68 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0008092, 0.0011148, 0.0007852, 0.0011154, -0.0003063, 0.0003296
1: 0.9934373, 0.9941657, 0.9934294, 0.9942164, -0.0007791, 0.0007363
2: -0.0084809, -0.0052019, -0.0085637, -0.0052544, -0.0029699, 0.0031007
3: 0.0037004, 0.0041412, 0.0036704, 0.0041466, -0.0004462, 0.0004707
4: 0.0025283, 0.0051199, 0.0025698, 0.0051853, -0.0026570, 0.0025501
5: 0.0052798, 0.0064719, 0.0052230, 0.0064606, -0.0011808, 0.0012489
6: -0.0020649, -0.0009267, -0.0020936, -0.0009449, -0.0011200, 0.0011669
7: -0.0082879, -0.0075605, -0.0082780, -0.0075362, -0.0007517, 0.0007175
8: 0.0050863, 0.0093945, 0.0051552, 0.0095033, -0.0043466, 0.0041705
9: -0.0036858, -0.0032317, -0.0036853, -0.0031970, -0.0004888, 0.0004536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006809
time: 0.71 seconds

## Relational analysis of IS_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_B1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006816
time: 0.68 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008135, 0.0011147, 0.0007852, 0.0011154, -0.0003019, 0.0003295
1: 0.9934387, 0.9941565, 0.9934294, 0.9942163, -0.0007777, 0.0007271
2: -0.0084658, -0.0051983, -0.0085637, -0.0052490, -0.0029574, 0.0031058
3: 0.0037059, 0.0041401, 0.0036704, 0.0041466, -0.0004407, 0.0004697
4: 0.0025254, 0.0051079, 0.0025655, 0.0051853, -0.0026599, 0.0025424
5: 0.0052902, 0.0064727, 0.0052230, 0.0064618, -0.0011716, 0.0012497
6: -0.0020597, -0.0009255, -0.0020936, -0.0009431, -0.0011166, 0.0011682
7: -0.0082886, -0.0075650, -0.0082790, -0.0075362, -0.0007524, 0.0007140
8: 0.0050815, 0.0093747, 0.0051481, 0.0095034, -0.0043518, 0.0041570
9: -0.0036859, -0.0032380, -0.0036853, -0.0031970, -0.0004889, 0.0004473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006835
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006843
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007687, 0.0011159, 0.0007754, 0.0011157, -0.0003469, 0.0003404
1: 0.9934240, 0.9942514, 0.9934261, 0.9942371, -0.0008131, 0.0008252
2: -0.0086206, -0.0053494, -0.0085975, -0.0053187, -0.0030306, 0.0029784
3: 0.0036498, 0.0041504, 0.0036582, 0.0041489, -0.0004990, 0.0004922
4: 0.0026449, 0.0052303, 0.0026206, 0.0052120, -0.0025671, 0.0026096
5: 0.0051840, 0.0064401, 0.0051998, 0.0064467, -0.0012627, 0.0012402
6: -0.0021134, -0.0009666, -0.0021054, -0.0009673, -0.0011461, 0.0011388
7: -0.0082600, -0.0075195, -0.0082658, -0.0075263, -0.0007337, 0.0007463
8: 0.0052801, 0.0095780, 0.0052397, 0.0095477, -0.0041949, 0.0042658
9: -0.0036843, -0.0031731, -0.0036846, -0.0031828, -0.0005015, 0.0005115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006730
time: 0.59 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006818, upper bound: 0.0006837
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0007727, 0.0011158, -0.0003516, 0.0003433
1: 0.9934224, 0.9942610, 0.9934253, 0.9942430, -0.0008206, 0.0008357
2: -0.0086365, -0.0053636, -0.0086070, -0.0053157, -0.0030478, 0.0029773
3: 0.0036441, 0.0041514, 0.0036547, 0.0041495, -0.0005054, 0.0004967
4: 0.0026561, 0.0052428, 0.0026183, 0.0052195, -0.0025634, 0.0026246
5: 0.0051731, 0.0064370, 0.0051933, 0.0064473, -0.0012743, 0.0012437
6: -0.0021189, -0.0009565, -0.0021087, -0.0009662, -0.0011527, 0.0011522
7: -0.0082573, -0.0075149, -0.0082664, -0.0075235, -0.0007338, 0.0007515
8: 0.0052987, 0.0095989, 0.0052358, 0.0095602, -0.0041892, 0.0042902
9: -0.0036842, -0.0031665, -0.0036846, -0.0031788, -0.0005053, 0.0005182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006826, upper bound: 0.0006874
time: 0.63 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006826, upper bound: 0.0006874
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007687, 0.0011159, 0.0007781, 0.0011156, -0.0003469, 0.0003378
1: 0.9934240, 0.9942513, 0.9934270, 0.9942315, -0.0008075, 0.0008243
2: -0.0086206, -0.0053440, -0.0085883, -0.0053041, -0.0030430, 0.0029749
3: 0.0036498, 0.0041504, 0.0036615, 0.0041483, -0.0004984, 0.0004889
4: 0.0026406, 0.0052303, 0.0026091, 0.0052047, -0.0025641, 0.0026212
5: 0.0051840, 0.0064412, 0.0052061, 0.0064499, -0.0012659, 0.0012351
6: -0.0021134, -0.0009665, -0.0021022, -0.0009622, -0.0011512, 0.0011356
7: -0.0082610, -0.0075195, -0.0082686, -0.0075290, -0.0007320, 0.0007491
8: 0.0052730, 0.0095781, 0.0052205, 0.0095356, -0.0041893, 0.0042844
9: -0.0036844, -0.0031731, -0.0036848, -0.0031867, -0.0004977, 0.0005116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
time: 0.59 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006842, upper bound: 0.0006848
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0007754, 0.0011157, -0.0003516, 0.0003406
1: 0.9934224, 0.9942611, 0.9934261, 0.9942372, -0.0008148, 0.0008349
2: -0.0086365, -0.0053572, -0.0085977, -0.0053010, -0.0030599, 0.0029738
3: 0.0036441, 0.0041515, 0.0036581, 0.0041489, -0.0005048, 0.0004933
4: 0.0026511, 0.0052428, 0.0026066, 0.0052122, -0.0025611, 0.0026362
5: 0.0051731, 0.0064384, 0.0051997, 0.0064505, -0.0012775, 0.0012387
6: -0.0021189, -0.0009565, -0.0021054, -0.0009611, -0.0011578, 0.0011490
7: -0.0082585, -0.0075148, -0.0082692, -0.0075263, -0.0007323, 0.0007543
8: 0.0052904, 0.0095990, 0.0052164, 0.0095480, -0.0041848, 0.0043089
9: -0.0036842, -0.0031665, -0.0036848, -0.0031827, -0.0005015, 0.0005183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006882
time: 0.79 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006881
time: 0.81 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007822, 0.0011155, 0.0007830, 0.0011155, -0.0003333, 0.0003325
1: 0.9934284, 0.9942229, 0.9934286, 0.9942212, -0.0007928, 0.0007943
2: -0.0085742, -0.0052968, -0.0085715, -0.0052735, -0.0030420, 0.0030093
3: 0.0036666, 0.0041473, 0.0036676, 0.0041471, -0.0004805, 0.0004797
4: 0.0026033, 0.0051936, 0.0025849, 0.0051914, -0.0025881, 0.0026087
5: 0.0052158, 0.0064514, 0.0052177, 0.0064565, -0.0012407, 0.0012338
6: -0.0020973, -0.0009597, -0.0020963, -0.0009516, -0.0011457, 0.0011367
7: -0.0082700, -0.0075331, -0.0082744, -0.0075340, -0.0007360, 0.0007412
8: 0.0052109, 0.0095171, 0.0051803, 0.0095135, -0.0042308, 0.0042662
9: -0.0036848, -0.0031926, -0.0036851, -0.0031937, -0.0004911, 0.0004925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_A1_B1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006827, upper bound: 0.0006763
time: 0.85 seconds

## Relational analysis of IS_B2_A1_A2_A1_B1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006763
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007822, 0.0011155, 0.0008133, 0.0011147, -0.0003325, 0.0003022
1: 0.9934284, 0.9942229, 0.9934386, 0.9941568, -0.0007284, 0.0007843
2: -0.0085742, -0.0052968, -0.0084665, -0.0051627, -0.0031533, 0.0029099
3: 0.0036666, 0.0041473, 0.0037056, 0.0041402, -0.0004736, 0.0004417
4: 0.0026033, 0.0051936, 0.0024973, 0.0051085, -0.0025052, 0.0026963
5: 0.0052158, 0.0064514, 0.0052897, 0.0064804, -0.0012646, 0.0011618
6: -0.0020973, -0.0009597, -0.0020599, -0.0009131, -0.0011842, 0.0011002
7: -0.0082700, -0.0075331, -0.0082953, -0.0075647, -0.0007052, 0.0007622
8: 0.0052109, 0.0095171, 0.0050348, 0.0093757, -0.0040946, 0.0044130
9: -0.0036848, -0.0031926, -0.0036862, -0.0032377, -0.0004472, 0.0004936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_A1_A2_A1_B2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006851, upper bound: 0.0006791
time: 0.62 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006851, upper bound: 0.0006791
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007901, 0.0011153, 0.0008033, 0.0011149, -0.0003248, 0.0003120
1: 0.9934310, 0.9942060, 0.9934354, 0.9941781, -0.0007470, 0.0007706
2: -0.0085467, -0.0052900, -0.0085011, -0.0052645, -0.0030316, 0.0029532
3: 0.0036766, 0.0041455, 0.0036931, 0.0041425, -0.0004659, 0.0004524
4: 0.0025979, 0.0051719, 0.0025778, 0.0051358, -0.0025379, 0.0025941
5: 0.0052347, 0.0064529, 0.0052660, 0.0064584, -0.0012238, 0.0011869
6: -0.0020877, -0.0009573, -0.0020719, -0.0009484, -0.0011393, 0.0011146
7: -0.0082712, -0.0075412, -0.0082761, -0.0075546, -0.0007166, 0.0007349
8: 0.0052020, 0.0094810, 0.0051685, 0.0094211, -0.0041486, 0.0042443
9: -0.0036849, -0.0032041, -0.0036852, -0.0032232, -0.0004617, 0.0004811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006718
time: 0.69 seconds

## Relational analysis of IS_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006827, upper bound: 0.0006782
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007878, 0.0011154, 0.0007974, 0.0011151, -0.0003273, 0.0003180
1: 0.9934303, 0.9942110, 0.9934334, 0.9941908, -0.0007605, 0.0007776
2: -0.0085548, -0.0052873, -0.0085216, -0.0052750, -0.0030295, 0.0029755
3: 0.0036736, 0.0041460, 0.0036856, 0.0041438, -0.0004702, 0.0004604
4: 0.0025958, 0.0051783, 0.0025861, 0.0051520, -0.0025562, 0.0025921
5: 0.0052291, 0.0064535, 0.0052519, 0.0064561, -0.0012270, 0.0012016
6: -0.0020905, -0.0009564, -0.0020790, -0.0009521, -0.0011384, 0.0011227
7: -0.0082717, -0.0075388, -0.0082741, -0.0075486, -0.0007232, 0.0007352
8: 0.0051985, 0.0094916, 0.0051824, 0.0094480, -0.0041800, 0.0042419
9: -0.0036849, -0.0032007, -0.0036851, -0.0032146, -0.0004703, 0.0004844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006720
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006782
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007816, 0.0011155, 0.0007830, 0.0011155, -0.0003339, 0.0003326
1: 0.9934282, 0.9942240, 0.9934286, 0.9942212, -0.0007930, 0.0007954
2: -0.0085762, -0.0053169, -0.0085715, -0.0052735, -0.0030329, 0.0029834
3: 0.0036659, 0.0041475, 0.0036676, 0.0041471, -0.0004812, 0.0004798
4: 0.0026192, 0.0051952, 0.0025849, 0.0051914, -0.0025722, 0.0026103
5: 0.0052145, 0.0064471, 0.0052177, 0.0064565, -0.0012420, 0.0012294
6: -0.0020980, -0.0009666, -0.0020963, -0.0009516, -0.0011464, 0.0011297
7: -0.0082662, -0.0075326, -0.0082744, -0.0075340, -0.0007322, 0.0007418
8: 0.0052374, 0.0095197, 0.0051803, 0.0095135, -0.0042037, 0.0042674
9: -0.0036846, -0.0031917, -0.0036851, -0.0031937, -0.0004909, 0.0004933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007010
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007832, 0.0011155, 0.0007829, 0.0011155, -0.0003323, 0.0003325
1: 0.9934287, 0.9942207, 0.9934286, 0.9942212, -0.0007924, 0.0007920
2: -0.0085707, -0.0053020, -0.0085715, -0.0052669, -0.0030317, 0.0029961
3: 0.0036679, 0.0041471, 0.0036676, 0.0041471, -0.0004792, 0.0004795
4: 0.0026074, 0.0051908, 0.0025797, 0.0051915, -0.0025841, 0.0026111
5: 0.0052182, 0.0064503, 0.0052177, 0.0064579, -0.0012397, 0.0012326
6: -0.0020961, -0.0009614, -0.0020963, -0.0009493, -0.0011468, 0.0011349
7: -0.0082690, -0.0075342, -0.0082756, -0.0075339, -0.0007350, 0.0007414
8: 0.0052177, 0.0095125, 0.0051717, 0.0095135, -0.0042228, 0.0042682
9: -0.0036848, -0.0031940, -0.0036851, -0.0031937, -0.0004911, 0.0004911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007012
time: 0.71 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0008092, 0.0011148, -0.0003318, 0.0003063
1: 0.9934286, 0.9942212, 0.9934373, 0.9941657, -0.0007370, 0.0007839
2: -0.0085715, -0.0052735, -0.0084809, -0.0052019, -0.0031072, 0.0029516
3: 0.0036676, 0.0041471, 0.0037004, 0.0041412, -0.0004735, 0.0004467
4: 0.0025849, 0.0051914, 0.0025283, 0.0051199, -0.0025350, 0.0026631
5: 0.0052177, 0.0064565, 0.0052798, 0.0064719, -0.0012543, 0.0011767
6: -0.0020963, -0.0009516, -0.0020649, -0.0009267, -0.0011696, 0.0011133
7: -0.0082744, -0.0075340, -0.0082879, -0.0075605, -0.0007139, 0.0007539
8: 0.0051803, 0.0095135, 0.0050863, 0.0093945, -0.0041458, 0.0043568
9: -0.0036851, -0.0031937, -0.0036858, -0.0032317, -0.0004534, 0.0004921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006929
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006942
time: 0.90 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0007829, 0.0011155, 0.0008135, 0.0011147, -0.0003317, 0.0003019
1: 0.9934286, 0.9942212, 0.9934387, 0.9941565, -0.0007278, 0.0007825
2: -0.0085715, -0.0052669, -0.0084658, -0.0051983, -0.0031096, 0.0029399
3: 0.0036676, 0.0041471, 0.0037059, 0.0041401, -0.0004726, 0.0004413
4: 0.0025797, 0.0051915, 0.0025254, 0.0051079, -0.0025282, 0.0026660
5: 0.0052177, 0.0064579, 0.0052902, 0.0064727, -0.0012551, 0.0011677
6: -0.0020963, -0.0009493, -0.0020597, -0.0009255, -0.0011709, 0.0011104
7: -0.0082756, -0.0075339, -0.0082886, -0.0075650, -0.0007107, 0.0007547
8: 0.0051717, 0.0095135, 0.0050815, 0.0093747, -0.0041329, 0.0043615
9: -0.0036851, -0.0031937, -0.0036859, -0.0032380, -0.0004471, 0.0004921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006937
time: 0.68 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006948
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008092, 0.0011148, 0.0007830, 0.0011155, -0.0003063, 0.0003318
1: 0.9934373, 0.9941657, 0.9934286, 0.9942212, -0.0007839, 0.0007370
2: -0.0084809, -0.0052019, -0.0085715, -0.0052735, -0.0029516, 0.0031072
3: 0.0037004, 0.0041412, 0.0036676, 0.0041471, -0.0004467, 0.0004735
4: 0.0025283, 0.0051199, 0.0025849, 0.0051914, -0.0026631, 0.0025350
5: 0.0052798, 0.0064719, 0.0052177, 0.0064565, -0.0011767, 0.0012543
6: -0.0020649, -0.0009267, -0.0020963, -0.0009516, -0.0011133, 0.0011696
7: -0.0082879, -0.0075605, -0.0082744, -0.0075340, -0.0007539, 0.0007139
8: 0.0050863, 0.0093945, 0.0051803, 0.0095135, -0.0043568, 0.0041458
9: -0.0036858, -0.0032317, -0.0036851, -0.0031937, -0.0004921, 0.0004534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006929, upper bound: 0.0006841
time: 0.71 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006942, upper bound: 0.0006841
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008135, 0.0011147, 0.0007829, 0.0011155, -0.0003019, 0.0003317
1: 0.9934387, 0.9941565, 0.9934286, 0.9942212, -0.0007825, 0.0007278
2: -0.0084658, -0.0051983, -0.0085715, -0.0052669, -0.0029399, 0.0031096
3: 0.0037059, 0.0041401, 0.0036676, 0.0041471, -0.0004413, 0.0004726
4: 0.0025254, 0.0051079, 0.0025797, 0.0051915, -0.0026660, 0.0025282
5: 0.0052902, 0.0064727, 0.0052177, 0.0064579, -0.0011677, 0.0012551
6: -0.0020597, -0.0009255, -0.0020963, -0.0009493, -0.0011104, 0.0011709
7: -0.0082886, -0.0075650, -0.0082756, -0.0075339, -0.0007547, 0.0007107
8: 0.0050815, 0.0093747, 0.0051717, 0.0095135, -0.0043615, 0.0041329
9: -0.0036859, -0.0032380, -0.0036851, -0.0031937, -0.0004921, 0.0004471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006937, upper bound: 0.0006855
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006948, upper bound: 0.0006855
time: 0.63 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0008092, 0.0011148, -0.0003015, 0.0003055
1: 0.9934386, 0.9941568, 0.9934373, 0.9941657, -0.0007271, 0.0007195
2: -0.0084665, -0.0051627, -0.0084809, -0.0052019, -0.0030043, 0.0030610
3: 0.0037056, 0.0041402, 0.0037004, 0.0041412, -0.0004355, 0.0004398
4: 0.0024973, 0.0051085, 0.0025283, 0.0051199, -0.0026225, 0.0025802
5: 0.0052897, 0.0064804, 0.0052798, 0.0064719, -0.0011823, 0.0012006
6: -0.0020599, -0.0009131, -0.0020649, -0.0009267, -0.0011332, 0.0011518
7: -0.0082953, -0.0075647, -0.0082879, -0.0075605, -0.0007348, 0.0007232
8: 0.0050348, 0.0093757, 0.0050863, 0.0093945, -0.0042911, 0.0042195
9: -0.0036862, -0.0032377, -0.0036858, -0.0032317, -0.0004545, 0.0004481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006839, upper bound: 0.0006855
time: 0.68 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006856
time: 0.64 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0008133, 0.0011147, 0.0008135, 0.0011147, -0.0003013, 0.0003012
1: 0.9934386, 0.9941569, 0.9934387, 0.9941565, -0.0007179, 0.0007182
2: -0.0084666, -0.0051588, -0.0084658, -0.0051983, -0.0030092, 0.0030466
3: 0.0037056, 0.0041402, 0.0037059, 0.0041401, -0.0004345, 0.0004343
4: 0.0024943, 0.0051086, 0.0025254, 0.0051079, -0.0026137, 0.0025831
5: 0.0052896, 0.0064813, 0.0052902, 0.0064727, -0.0011831, 0.0011911
6: -0.0020599, -0.0009118, -0.0020597, -0.0009255, -0.0011345, 0.0011479
7: -0.0082960, -0.0075647, -0.0082886, -0.0075650, -0.0007311, 0.0007239
8: 0.0050297, 0.0093757, 0.0050815, 0.0093747, -0.0042753, 0.0042248
9: -0.0036863, -0.0032377, -0.0036859, -0.0032380, -0.0004483, 0.0004482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006856
time: 0.65 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006856
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.16 seconds
IS_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006963, upper bound: 0.0006900
IS_B1_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006960
IS_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006963, upper bound: 0.0006915
IS_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006964
IS_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006838, upper bound: 0.0006763
IS_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006872, upper bound: 0.0006763
IS_B1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006848, upper bound: 0.0006782
IS_B1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006878, upper bound: 0.0006782
IS_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006730, upper bound: 0.0006697
IS_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006837, upper bound: 0.0006819
IS_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006827
IS_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006874, upper bound: 0.0006827
IS_B1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006768, upper bound: 0.0006738
IS_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006848, upper bound: 0.0006842
IS_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006882, upper bound: 0.0006852
IS_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006882, upper bound: 0.0006852
IS_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006744
IS_B1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006873
IS_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006872
IS_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006898
IS_B1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006848
IS_B1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006885
IS_B1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006878
IS_B1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006905
IS_B1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006713, upper bound: 0.0006782
IS_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006782
IS_B1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006782
IS_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006782
IS_B1_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006809
IS_B1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006816
IS_B1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006835
IS_B1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006843
IS_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006730
IS_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006818, upper bound: 0.0006837
IS_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006826, upper bound: 0.0006874
IS_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006826, upper bound: 0.0006874
IS_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
IS_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006842, upper bound: 0.0006848
IS_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006882
IS_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006881
IS_B2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006827, upper bound: 0.0006763
IS_B2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006763
IS_B2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006851, upper bound: 0.0006791
IS_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006851, upper bound: 0.0006791
IS_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006718
IS_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006827, upper bound: 0.0006782
IS_B2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006720
IS_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006782
IS_B2_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007010
IS_B2_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
IS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007012
IS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
IS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006929
IS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006942
IS_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006937
IS_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006948
IS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006929, upper bound: 0.0006841
IS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006942, upper bound: 0.0006841
IS_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006937, upper bound: 0.0006855
IS_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006948, upper bound: 0.0006855
IS_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006839, upper bound: 0.0006855
IS_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006856
IS_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006856
IS_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006856

## BFS IS instance: IS_B1_B1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007670, 0.0011159, 0.0007563, 0.0011162, -0.0003492, 0.0003596
1: 0.9934235, 0.9942550, 0.9934199, 0.9942776, -0.0008541, 0.0008351
2: -0.0086267, -0.0053914, -0.0086635, -0.0053597, -0.0029933, 0.0029969
3: 0.0036476, 0.0041508, 0.0036343, 0.0041532, -0.0005056, 0.0005165
4: 0.0026781, 0.0052350, 0.0026531, 0.0052642, -0.0025861, 0.0025820
5: 0.0051798, 0.0064310, 0.0051546, 0.0064378, -0.0012580, 0.0012764
6: -0.0021155, -0.0009627, -0.0021283, -0.0009394, -0.0011761, 0.0011655
7: -0.0082521, -0.0075178, -0.0082581, -0.0075069, -0.0007451, 0.0007403
8: 0.0053352, 0.0095860, 0.0052936, 0.0096344, -0.0042255, 0.0042192
9: -0.0036839, -0.0031706, -0.0036842, -0.0031552, -0.0005287, 0.0005136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A1_A1_A1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006895
time: 0.74 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006962, upper bound: 0.0006900
time: 0.71 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007629, 0.0011160, 0.0007536, 0.0011162, -0.0003534, 0.0003624
1: 0.9934221, 0.9942636, 0.9934190, 0.9942832, -0.0008611, 0.0008447
2: -0.0086408, -0.0054069, -0.0086727, -0.0053569, -0.0030088, 0.0029919
3: 0.0036425, 0.0041517, 0.0036309, 0.0041538, -0.0005113, 0.0005208
4: 0.0026903, 0.0052462, 0.0026508, 0.0052715, -0.0025811, 0.0025954
5: 0.0051701, 0.0064276, 0.0051482, 0.0064385, -0.0012683, 0.0012794
6: -0.0021204, -0.0009538, -0.0021315, -0.0009335, -0.0011869, 0.0011777
7: -0.0082491, -0.0075136, -0.0082586, -0.0075042, -0.0007449, 0.0007450
8: 0.0053556, 0.0096045, 0.0052899, 0.0096465, -0.0042176, 0.0042411
9: -0.0036837, -0.0031647, -0.0036842, -0.0031513, -0.0005324, 0.0005195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A1_A1_A2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006943, upper bound: 0.0006837
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006960
time: 0.63 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007689, 0.0011158, 0.0007563, 0.0011162, -0.0003473, 0.0003595
1: 0.9934241, 0.9942509, 0.9934199, 0.9942776, -0.0008535, 0.0008309
2: -0.0086199, -0.0053802, -0.0086635, -0.0053534, -0.0029930, 0.0030076
3: 0.0036501, 0.0041504, 0.0036343, 0.0041532, -0.0005032, 0.0005161
4: 0.0026692, 0.0052297, 0.0026480, 0.0052642, -0.0025949, 0.0025817
5: 0.0051844, 0.0064334, 0.0051545, 0.0064392, -0.0012548, 0.0012789
6: -0.0021131, -0.0009670, -0.0021283, -0.0009393, -0.0011738, 0.0011613
7: -0.0082542, -0.0075197, -0.0082593, -0.0075069, -0.0007473, 0.0007395
8: 0.0053206, 0.0095772, 0.0052853, 0.0096344, -0.0042400, 0.0042188
9: -0.0036840, -0.0031734, -0.0036843, -0.0031552, -0.0005288, 0.0005108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006911
time: 0.65 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006915
time: 0.69 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007643, 0.0011160, 0.0007537, 0.0011162, -0.0003519, 0.0003623
1: 0.9934225, 0.9942606, 0.9934190, 0.9942833, -0.0008608, 0.0008416
2: -0.0086358, -0.0053927, -0.0086728, -0.0053505, -0.0030105, 0.0030051
3: 0.0036443, 0.0041514, 0.0036309, 0.0041538, -0.0005095, 0.0005205
4: 0.0026791, 0.0052423, 0.0026458, 0.0052715, -0.0025924, 0.0025965
5: 0.0051735, 0.0064307, 0.0051482, 0.0064398, -0.0012663, 0.0012825
6: -0.0021187, -0.0009569, -0.0021315, -0.0009335, -0.0011852, 0.0011746
7: -0.0082518, -0.0075150, -0.0082598, -0.0075042, -0.0007476, 0.0007448
8: 0.0053370, 0.0095981, 0.0052815, 0.0096465, -0.0042360, 0.0042432
9: -0.0036839, -0.0031668, -0.0036843, -0.0031513, -0.0005326, 0.0005175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006943, upper bound: 0.0006845
time: 0.74 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006964
time: 0.63 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007866, 0.0011154, 0.0007687, 0.0011159, -0.0003292, 0.0003466
1: 0.9934298, 0.9942135, 0.9934240, 0.9942514, -0.0008215, 0.0007895
2: -0.0085588, -0.0053007, -0.0086206, -0.0053494, -0.0029465, 0.0030540
3: 0.0036722, 0.0041463, 0.0036498, 0.0041504, -0.0004782, 0.0004965
4: 0.0026064, 0.0051815, 0.0026449, 0.0052303, -0.0026238, 0.0025366
5: 0.0052263, 0.0064506, 0.0051840, 0.0064401, -0.0012137, 0.0012666
6: -0.0020920, -0.0009610, -0.0021134, -0.0009666, -0.0011254, 0.0011524
7: -0.0082692, -0.0075376, -0.0082600, -0.0075195, -0.0007497, 0.0007224
8: 0.0052161, 0.0094969, 0.0052801, 0.0095780, -0.0042901, 0.0041459
9: -0.0036848, -0.0031990, -0.0036843, -0.0031731, -0.0005117, 0.0004853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006559
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006836, upper bound: 0.0006763
time: 0.73 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007847, 0.0011154, 0.0007641, 0.0011160, -0.0003313, 0.0003513
1: 0.9934292, 0.9942175, 0.9934224, 0.9942610, -0.0008318, 0.0007951
2: -0.0085656, -0.0052982, -0.0086365, -0.0053636, -0.0029417, 0.0030713
3: 0.0036698, 0.0041468, 0.0036441, 0.0041514, -0.0004817, 0.0005027
4: 0.0026044, 0.0051868, 0.0026561, 0.0052428, -0.0026384, 0.0025307
5: 0.0052217, 0.0064511, 0.0051731, 0.0064370, -0.0012153, 0.0012780
6: -0.0020943, -0.0009601, -0.0021189, -0.0009565, -0.0011378, 0.0011587
7: -0.0082697, -0.0075357, -0.0082573, -0.0075149, -0.0007548, 0.0007217
8: 0.0052128, 0.0095058, 0.0052987, 0.0095989, -0.0043140, 0.0041364
9: -0.0036848, -0.0031962, -0.0036842, -0.0031665, -0.0005184, 0.0004880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006744, upper bound: 0.0006560
time: 0.59 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006869, upper bound: 0.0006763
time: 0.64 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007901, 0.0011153, 0.0007687, 0.0011159, -0.0003257, 0.0003466
1: 0.9934310, 0.9942060, 0.9934240, 0.9942513, -0.0008203, 0.0007820
2: -0.0085467, -0.0052900, -0.0086206, -0.0053440, -0.0029373, 0.0030648
3: 0.0036766, 0.0041455, 0.0036498, 0.0041504, -0.0004738, 0.0004957
4: 0.0025979, 0.0051719, 0.0026406, 0.0052303, -0.0026324, 0.0025313
5: 0.0052347, 0.0064529, 0.0051840, 0.0064412, -0.0012066, 0.0012689
6: -0.0020877, -0.0009573, -0.0021134, -0.0009665, -0.0011212, 0.0011561
7: -0.0082712, -0.0075412, -0.0082610, -0.0075195, -0.0007517, 0.0007198
8: 0.0052020, 0.0094810, 0.0052730, 0.0095781, -0.0043048, 0.0041352
9: -0.0036849, -0.0032041, -0.0036844, -0.0031731, -0.0005118, 0.0004803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006744, upper bound: 0.0006589
time: 0.59 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006782
time: 0.63 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007878, 0.0011154, 0.0007641, 0.0011160, -0.0003282, 0.0003512
1: 0.9934303, 0.9942110, 0.9934224, 0.9942611, -0.0008308, 0.0007885
2: -0.0085548, -0.0052873, -0.0086365, -0.0053572, -0.0029345, 0.0030818
3: 0.0036736, 0.0041460, 0.0036441, 0.0041515, -0.0004778, 0.0005020
4: 0.0025958, 0.0051783, 0.0026511, 0.0052428, -0.0026470, 0.0025272
5: 0.0052291, 0.0064535, 0.0051731, 0.0064384, -0.0012093, 0.0012804
6: -0.0020905, -0.0009564, -0.0021189, -0.0009565, -0.0011341, 0.0011625
7: -0.0082717, -0.0075388, -0.0082585, -0.0075148, -0.0007569, 0.0007197
8: 0.0051985, 0.0094916, 0.0052904, 0.0095990, -0.0043287, 0.0041298
9: -0.0036849, -0.0032007, -0.0036842, -0.0031665, -0.0005185, 0.0004835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006765, upper bound: 0.0006589
time: 0.59 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006876, upper bound: 0.0006782
time: 0.91 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007700, 0.0011158, 0.0007689, 0.0011158, -0.0003459, 0.0003469
1: 0.9934244, 0.9942486, 0.9934240, 0.9942510, -0.0008265, 0.0008246
2: -0.0086163, -0.0053818, -0.0086201, -0.0053685, -0.0029872, 0.0029721
3: 0.0036514, 0.0041501, 0.0036500, 0.0041504, -0.0004990, 0.0005001
4: 0.0026705, 0.0052269, 0.0026600, 0.0052299, -0.0025594, 0.0025669
5: 0.0051869, 0.0064331, 0.0051843, 0.0064359, -0.0012490, 0.0012488
6: -0.0021119, -0.0009693, -0.0021132, -0.0009669, -0.0011450, 0.0011439
7: -0.0082539, -0.0075208, -0.0082564, -0.0075197, -0.0007342, 0.0007356
8: 0.0053226, 0.0095724, 0.0053052, 0.0095774, -0.0041821, 0.0041959
9: -0.0036840, -0.0031749, -0.0036841, -0.0031733, -0.0005106, 0.0005092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006730, upper bound: 0.0006697
time: 0.79 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006730, upper bound: 0.0006697
time: 0.71 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007758, 0.0011157, 0.0007688, 0.0011159, -0.0003400, 0.0003468
1: 0.9934263, 0.9942364, 0.9934240, 0.9942511, -0.0008248, 0.0008124
2: -0.0085961, -0.0053589, -0.0086203, -0.0053564, -0.0029700, 0.0029904
3: 0.0036587, 0.0041488, 0.0036499, 0.0041504, -0.0004917, 0.0004989
4: 0.0026524, 0.0052109, 0.0026504, 0.0052300, -0.0025777, 0.0025605
5: 0.0052008, 0.0064380, 0.0051842, 0.0064386, -0.0012378, 0.0012538
6: -0.0021049, -0.0009812, -0.0021133, -0.0009667, -0.0011381, 0.0011321
7: -0.0082582, -0.0075267, -0.0082587, -0.0075196, -0.0007386, 0.0007320
8: 0.0052925, 0.0095459, 0.0052893, 0.0095777, -0.0042127, 0.0041839
9: -0.0036842, -0.0031834, -0.0036842, -0.0031733, -0.0005110, 0.0005008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006769, upper bound: 0.0006787
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006813, upper bound: 0.0006790
time: 0.68 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007838, 0.0011155, 0.0007641, 0.0011160, -0.0003322, 0.0003513
1: 0.9934289, 0.9942195, 0.9934224, 0.9942610, -0.0008321, 0.0007970
2: -0.0085686, -0.0053183, -0.0086365, -0.0053636, -0.0029351, 0.0030452
3: 0.0036687, 0.0041470, 0.0036441, 0.0041514, -0.0004828, 0.0005029
4: 0.0026203, 0.0051892, 0.0026561, 0.0052428, -0.0026225, 0.0025331
5: 0.0052196, 0.0064468, 0.0051731, 0.0064370, -0.0012174, 0.0012737
6: -0.0020953, -0.0009671, -0.0021189, -0.0009565, -0.0011389, 0.0011518
7: -0.0082659, -0.0075348, -0.0082573, -0.0075149, -0.0007510, 0.0007226
8: 0.0052392, 0.0095097, 0.0052987, 0.0095989, -0.0042868, 0.0041388
9: -0.0036846, -0.0031949, -0.0036842, -0.0031665, -0.0005182, 0.0004892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006760, upper bound: 0.0006720
time: 0.70 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006871, upper bound: 0.0006825
time: 0.67 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008114, 0.0011147, 0.0007641, 0.0011160, -0.0003045, 0.0003506
1: 0.9934381, 0.9941609, 0.9934224, 0.9942610, -0.0008230, 0.0007384
2: -0.0084731, -0.0052033, -0.0086365, -0.0053636, -0.0028539, 0.0031702
3: 0.0037033, 0.0041406, 0.0036441, 0.0041514, -0.0004482, 0.0004966
4: 0.0025294, 0.0051137, 0.0026561, 0.0052428, -0.0027134, 0.0024576
5: 0.0052852, 0.0064716, 0.0051731, 0.0064370, -0.0011518, 0.0012986
6: -0.0020622, -0.0009272, -0.0021189, -0.0009565, -0.0011057, 0.0011917
7: -0.0082876, -0.0075628, -0.0082573, -0.0075149, -0.0007728, 0.0006945
8: 0.0050881, 0.0093842, 0.0052987, 0.0095989, -0.0044401, 0.0040171
9: -0.0036858, -0.0032350, -0.0036842, -0.0031665, -0.0005193, 0.0004492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006760, upper bound: 0.0006720
time: 0.90 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006871, upper bound: 0.0006825
time: 0.73 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007726, 0.0011158, 0.0007689, 0.0011159, -0.0003433, 0.0003469
1: 0.9934253, 0.9942432, 0.9934240, 0.9942509, -0.0008256, 0.0008192
2: -0.0086073, -0.0053646, -0.0086201, -0.0053627, -0.0029817, 0.0029891
3: 0.0036546, 0.0041495, 0.0036500, 0.0041504, -0.0004957, 0.0004995
4: 0.0026569, 0.0052198, 0.0026555, 0.0052299, -0.0025730, 0.0025643
5: 0.0051931, 0.0064368, 0.0051843, 0.0064372, -0.0012441, 0.0012525
6: -0.0021088, -0.0009750, -0.0021132, -0.0009669, -0.0011419, 0.0011382
7: -0.0082571, -0.0075234, -0.0082575, -0.0075197, -0.0007375, 0.0007341
8: 0.0053000, 0.0095606, 0.0052976, 0.0095774, -0.0042043, 0.0041912
9: -0.0036841, -0.0031787, -0.0036842, -0.0031733, -0.0005108, 0.0005055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006738
time: 0.60 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006738
time: 0.65 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007785, 0.0011156, 0.0007688, 0.0011159, -0.0003373, 0.0003468
1: 0.9934272, 0.9942306, 0.9934239, 0.9942511, -0.0008239, 0.0008067
2: -0.0085868, -0.0053446, -0.0086203, -0.0053512, -0.0029662, 0.0030016
3: 0.0036620, 0.0041482, 0.0036499, 0.0041504, -0.0004883, 0.0004983
4: 0.0026411, 0.0052036, 0.0026463, 0.0052301, -0.0025890, 0.0025573
5: 0.0052071, 0.0064411, 0.0051842, 0.0064397, -0.0012325, 0.0012569
6: -0.0021017, -0.0009762, -0.0021133, -0.0009667, -0.0011349, 0.0011370
7: -0.0082609, -0.0075294, -0.0082597, -0.0075196, -0.0007413, 0.0007302
8: 0.0052737, 0.0095337, 0.0052825, 0.0095777, -0.0042307, 0.0041779
9: -0.0036844, -0.0031873, -0.0036843, -0.0031732, -0.0005111, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006774, upper bound: 0.0006808
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006823, upper bound: 0.0006813
time: 0.70 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007854, 0.0011154, 0.0007641, 0.0011160, -0.0003306, 0.0003513
1: 0.9934295, 0.9942161, 0.9934224, 0.9942611, -0.0008316, 0.0007937
2: -0.0085631, -0.0053034, -0.0086365, -0.0053572, -0.0029333, 0.0030574
3: 0.0036707, 0.0041466, 0.0036441, 0.0041515, -0.0004808, 0.0005025
4: 0.0026086, 0.0051848, 0.0026511, 0.0052428, -0.0026343, 0.0025337
5: 0.0052234, 0.0064500, 0.0051731, 0.0064384, -0.0012150, 0.0012769
6: -0.0020934, -0.0009620, -0.0021189, -0.0009565, -0.0011370, 0.0011569
7: -0.0082687, -0.0075364, -0.0082585, -0.0075148, -0.0007539, 0.0007221
8: 0.0052197, 0.0095025, 0.0052904, 0.0095990, -0.0043057, 0.0041390
9: -0.0036848, -0.0031972, -0.0036842, -0.0031665, -0.0005183, 0.0004870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006789, upper bound: 0.0006756
time: 0.69 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006851
time: 0.81 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008158, 0.0011146, 0.0007641, 0.0011160, -0.0003002, 0.0003505
1: 0.9934393, 0.9941516, 0.9934224, 0.9942611, -0.0008218, 0.0007291
2: -0.0084580, -0.0051997, -0.0086365, -0.0053572, -0.0028423, 0.0031725
3: 0.0037088, 0.0041396, 0.0036441, 0.0041515, -0.0004427, 0.0004956
4: 0.0025266, 0.0051017, 0.0026511, 0.0052428, -0.0027163, 0.0024506
5: 0.0052955, 0.0064724, 0.0051731, 0.0064384, -0.0011428, 0.0012994
6: -0.0020569, -0.0009259, -0.0021189, -0.0009565, -0.0011005, 0.0011930
7: -0.0082883, -0.0075673, -0.0082585, -0.0075148, -0.0007735, 0.0006913
8: 0.0050833, 0.0093643, 0.0052904, 0.0095990, -0.0044451, 0.0040040
9: -0.0036858, -0.0032413, -0.0036842, -0.0031665, -0.0005194, 0.0004429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006789, upper bound: 0.0006756
time: 0.73 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006851
time: 0.80 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0008008, 0.0011150, 0.0007809, 0.0011155, -0.0003147, 0.0003341
1: 0.9934345, 0.9941834, 0.9934280, 0.9942254, -0.0007910, 0.0007554
2: -0.0085099, -0.0052605, -0.0085785, -0.0053542, -0.0028907, 0.0030572
3: 0.0036899, 0.0041431, 0.0036650, 0.0041476, -0.0004577, 0.0004780
4: 0.0025746, 0.0051428, 0.0026487, 0.0051970, -0.0026224, 0.0024941
5: 0.0052599, 0.0064593, 0.0052128, 0.0064390, -0.0011791, 0.0012465
6: -0.0020750, -0.0009471, -0.0020988, -0.0009796, -0.0010954, 0.0011517
7: -0.0082768, -0.0075520, -0.0082591, -0.0075319, -0.0007449, 0.0007071
8: 0.0051633, 0.0094326, 0.0052864, 0.0095228, -0.0042879, 0.0040750
9: -0.0036852, -0.0032195, -0.0036843, -0.0031908, -0.0004945, 0.0004647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006716
time: 0.65 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006744
time: 0.64 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0008007, 0.0011150, 0.0007870, 0.0011154, -0.0003146, 0.0003280
1: 0.9934345, 0.9941835, 0.9934300, 0.9942125, -0.0007780, 0.0007536
2: -0.0085101, -0.0052494, -0.0085573, -0.0053427, -0.0029032, 0.0030450
3: 0.0036899, 0.0041431, 0.0036727, 0.0041462, -0.0004563, 0.0004703
4: 0.0025659, 0.0051429, 0.0026396, 0.0051803, -0.0026144, 0.0025034
5: 0.0052598, 0.0064617, 0.0052274, 0.0064415, -0.0011817, 0.0012343
6: -0.0020750, -0.0009432, -0.0020914, -0.0009756, -0.0010994, 0.0011482
7: -0.0082789, -0.0075520, -0.0082613, -0.0075381, -0.0007408, 0.0007093
8: 0.0051487, 0.0094328, 0.0052712, 0.0094949, -0.0042753, 0.0040892
9: -0.0036853, -0.0032194, -0.0036844, -0.0031997, -0.0004857, 0.0004649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006833
time: 0.77 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006847
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0007847, 0.0011154, -0.0003513, 0.0003313
1: 0.9934224, 0.9942610, 0.9934292, 0.9942175, -0.0007951, 0.0008318
2: -0.0086365, -0.0053636, -0.0085656, -0.0052982, -0.0030713, 0.0029417
3: 0.0036441, 0.0041514, 0.0036698, 0.0041468, -0.0005027, 0.0004817
4: 0.0026561, 0.0052428, 0.0026044, 0.0051868, -0.0025307, 0.0026384
5: 0.0051731, 0.0064370, 0.0052217, 0.0064511, -0.0012780, 0.0012153
6: -0.0021189, -0.0009565, -0.0020943, -0.0009601, -0.0011587, 0.0011378
7: -0.0082573, -0.0075149, -0.0082697, -0.0075357, -0.0007217, 0.0007548
8: 0.0052987, 0.0095989, 0.0052128, 0.0095058, -0.0041364, 0.0043140
9: -0.0036842, -0.0031665, -0.0036848, -0.0031962, -0.0004880, 0.0005184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006744
time: 0.65 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006869
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007955, 0.0011152, 0.0007847, 0.0011154, -0.0003199, 0.0003305
1: 0.9934328, 0.9941946, 0.9934292, 0.9942175, -0.0007847, 0.0007653
2: -0.0085281, -0.0052817, -0.0085656, -0.0052982, -0.0029659, 0.0030255
3: 0.0036833, 0.0041443, 0.0036698, 0.0041468, -0.0004634, 0.0004745
4: 0.0025913, 0.0051571, 0.0026044, 0.0051868, -0.0025955, 0.0025527
5: 0.0052475, 0.0064547, 0.0052217, 0.0064511, -0.0012037, 0.0012330
6: -0.0020813, -0.0009544, -0.0020943, -0.0009601, -0.0011211, 0.0011399
7: -0.0082728, -0.0075467, -0.0082697, -0.0075357, -0.0007372, 0.0007230
8: 0.0051910, 0.0094565, 0.0052128, 0.0095058, -0.0042444, 0.0041716
9: -0.0036850, -0.0032119, -0.0036848, -0.0031962, -0.0004888, 0.0004729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B1_A2_A2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006745, upper bound: 0.0006821
time: 0.65 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006894
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007687, 0.0011159, 0.0007901, 0.0011153, -0.0003466, 0.0003257
1: 0.9934240, 0.9942513, 0.9934310, 0.9942060, -0.0007820, 0.0008203
2: -0.0086206, -0.0053440, -0.0085467, -0.0052900, -0.0030648, 0.0029373
3: 0.0036498, 0.0041504, 0.0036766, 0.0041455, -0.0004957, 0.0004738
4: 0.0026406, 0.0052303, 0.0025979, 0.0051719, -0.0025313, 0.0026324
5: 0.0051840, 0.0064412, 0.0052347, 0.0064529, -0.0012689, 0.0012066
6: -0.0021134, -0.0009665, -0.0020877, -0.0009573, -0.0011561, 0.0011212
7: -0.0082610, -0.0075195, -0.0082712, -0.0075412, -0.0007198, 0.0007517
8: 0.0052730, 0.0095781, 0.0052020, 0.0094810, -0.0041352, 0.0043048
9: -0.0036844, -0.0031731, -0.0036849, -0.0032041, -0.0004803, 0.0005118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006589, upper bound: 0.0006744
time: 0.68 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006845
time: 0.71 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0008006, 0.0011150, 0.0007901, 0.0011153, -0.0003147, 0.0003249
1: 0.9934344, 0.9941838, 0.9934310, 0.9942060, -0.0007716, 0.0007527
2: -0.0085105, -0.0052648, -0.0085467, -0.0052900, -0.0029545, 0.0030199
3: 0.0036897, 0.0041431, 0.0036766, 0.0041455, -0.0004558, 0.0004665
4: 0.0025780, 0.0051432, 0.0025979, 0.0051719, -0.0025939, 0.0025453
5: 0.0052595, 0.0064584, 0.0052347, 0.0064529, -0.0011934, 0.0012237
6: -0.0020751, -0.0009485, -0.0020877, -0.0009573, -0.0011179, 0.0011392
7: -0.0082760, -0.0075518, -0.0082712, -0.0075412, -0.0007348, 0.0007194
8: 0.0051689, 0.0094333, 0.0052020, 0.0094810, -0.0042394, 0.0041600
9: -0.0036852, -0.0032193, -0.0036849, -0.0032041, -0.0004811, 0.0004656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006753, upper bound: 0.0006766
time: 0.67 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A2_A2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006881
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0007878, 0.0011154, -0.0003512, 0.0003282
1: 0.9934224, 0.9942611, 0.9934303, 0.9942110, -0.0007885, 0.0008308
2: -0.0086365, -0.0053572, -0.0085548, -0.0052873, -0.0030818, 0.0029345
3: 0.0036441, 0.0041515, 0.0036736, 0.0041460, -0.0005020, 0.0004778
4: 0.0026511, 0.0052428, 0.0025958, 0.0051783, -0.0025272, 0.0026470
5: 0.0051731, 0.0064384, 0.0052291, 0.0064535, -0.0012804, 0.0012093
6: -0.0021189, -0.0009565, -0.0020905, -0.0009564, -0.0011625, 0.0011341
7: -0.0082585, -0.0075148, -0.0082717, -0.0075388, -0.0007197, 0.0007569
8: 0.0052904, 0.0095990, 0.0051985, 0.0094916, -0.0041298, 0.0043287
9: -0.0036842, -0.0031665, -0.0036849, -0.0032007, -0.0004835, 0.0005185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006589, upper bound: 0.0006765
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A1_B2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006876
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007955, 0.0011152, 0.0007878, 0.0011154, -0.0003198, 0.0003274
1: 0.9934328, 0.9941947, 0.9934303, 0.9942110, -0.0007781, 0.0007644
2: -0.0085281, -0.0052751, -0.0085548, -0.0052873, -0.0029763, 0.0030179
3: 0.0036833, 0.0041443, 0.0036736, 0.0041460, -0.0004627, 0.0004706
4: 0.0025862, 0.0051572, 0.0025958, 0.0051783, -0.0025921, 0.0025613
5: 0.0052475, 0.0064561, 0.0052291, 0.0064535, -0.0012060, 0.0012270
6: -0.0020813, -0.0009521, -0.0020905, -0.0009564, -0.0011249, 0.0011384
7: -0.0082741, -0.0075467, -0.0082717, -0.0075388, -0.0007352, 0.0007251
8: 0.0051825, 0.0094565, 0.0051985, 0.0094916, -0.0042379, 0.0041862
9: -0.0036851, -0.0032119, -0.0036849, -0.0032007, -0.0004844, 0.0004730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006777
time: 0.67 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006902
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0007899, 0.0011153, 0.0007976, 0.0011151, -0.0003252, 0.0003177
1: 0.9934309, 0.9942064, 0.9934334, 0.9941903, -0.0007594, 0.0007730
2: -0.0085473, -0.0052583, -0.0085210, -0.0052847, -0.0029934, 0.0029991
3: 0.0036764, 0.0041455, 0.0036859, 0.0041438, -0.0004674, 0.0004596
4: 0.0025729, 0.0051723, 0.0025937, 0.0051515, -0.0025786, 0.0025786
5: 0.0052343, 0.0064598, 0.0052523, 0.0064540, -0.0012198, 0.0012074
6: -0.0020879, -0.0009463, -0.0020788, -0.0009555, -0.0011325, 0.0011325
7: -0.0082772, -0.0075410, -0.0082722, -0.0075487, -0.0007285, 0.0007312
8: 0.0051604, 0.0094817, 0.0051951, 0.0094472, -0.0042160, 0.0042143
9: -0.0036852, -0.0032039, -0.0036850, -0.0032149, -0.0004704, 0.0004811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006650, upper bound: 0.0006589
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006713, upper bound: 0.0006782
time: 0.68 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0007876, 0.0011154, 0.0007961, 0.0011151, -0.0003275, 0.0003193
1: 0.9934301, 0.9942113, 0.9934329, 0.9941933, -0.0007631, 0.0007783
2: -0.0085554, -0.0052558, -0.0085262, -0.0053050, -0.0029863, 0.0030084
3: 0.0036734, 0.0041461, 0.0036840, 0.0041441, -0.0004707, 0.0004620
4: 0.0025709, 0.0051787, 0.0026098, 0.0051556, -0.0025847, 0.0025690
5: 0.0052287, 0.0064603, 0.0052488, 0.0064497, -0.0012210, 0.0012115
6: -0.0020907, -0.0009454, -0.0020806, -0.0009625, -0.0011282, 0.0011352
7: -0.0082777, -0.0075387, -0.0082684, -0.0075472, -0.0007305, 0.0007297
8: 0.0051570, 0.0094924, 0.0052217, 0.0094539, -0.0042264, 0.0041993
9: -0.0036853, -0.0032005, -0.0036848, -0.0032127, -0.0004726, 0.0004843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006689
time: 0.65 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006782
time: 0.70 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0007899, 0.0011153, 0.0008018, 0.0011150, -0.0003251, 0.0003135
1: 0.9934309, 0.9942063, 0.9934349, 0.9941812, -0.0007504, 0.0007714
2: -0.0085474, -0.0052530, -0.0085063, -0.0052777, -0.0030014, 0.0029887
3: 0.0036763, 0.0041455, 0.0036912, 0.0041428, -0.0004665, 0.0004543
4: 0.0025687, 0.0051724, 0.0025882, 0.0051399, -0.0025713, 0.0025842
5: 0.0052342, 0.0064609, 0.0052624, 0.0064556, -0.0012214, 0.0011985
6: -0.0020880, -0.0009444, -0.0020737, -0.0009530, -0.0011349, 0.0011293
7: -0.0082782, -0.0075410, -0.0082736, -0.0075531, -0.0007252, 0.0007326
8: 0.0051534, 0.0094818, 0.0051859, 0.0094279, -0.0042030, 0.0042231
9: -0.0036853, -0.0032038, -0.0036850, -0.0032210, -0.0004643, 0.0004812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006683, upper bound: 0.0006589
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006782
time: 0.70 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0007876, 0.0011154, 0.0007985, 0.0011151, -0.0003275, 0.0003169
1: 0.9934301, 0.9942113, 0.9934338, 0.9941882, -0.0007581, 0.0007775
2: -0.0085555, -0.0052503, -0.0085178, -0.0052941, -0.0029978, 0.0030021
3: 0.0036734, 0.0041461, 0.0036871, 0.0041436, -0.0004702, 0.0004590
4: 0.0025666, 0.0051788, 0.0026012, 0.0051490, -0.0025824, 0.0025776
5: 0.0052287, 0.0064615, 0.0052545, 0.0064520, -0.0012234, 0.0012070
6: -0.0020908, -0.0009435, -0.0020777, -0.0009587, -0.0011321, 0.0011342
7: -0.0082787, -0.0075386, -0.0082705, -0.0075497, -0.0007291, 0.0007318
8: 0.0051499, 0.0094924, 0.0052074, 0.0094430, -0.0042218, 0.0042141
9: -0.0036853, -0.0032004, -0.0036849, -0.0032162, -0.0004691, 0.0004844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006589, upper bound: 0.0006700
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006782
time: 0.74 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0008140, 0.0011147, 0.0008016, 0.0011150, -0.0003010, 0.0003130
1: 0.9934388, 0.9941555, 0.9934348, 0.9941816, -0.0007427, 0.0007207
2: -0.0084643, -0.0052059, -0.0085069, -0.0052441, -0.0029595, 0.0030393
3: 0.0037064, 0.0041401, 0.0036910, 0.0041429, -0.0004364, 0.0004491
4: 0.0025314, 0.0051068, 0.0025617, 0.0051404, -0.0026090, 0.0025451
5: 0.0052912, 0.0064711, 0.0052620, 0.0064628, -0.0011716, 0.0012091
6: -0.0020591, -0.0009281, -0.0020739, -0.0009414, -0.0011178, 0.0011458
7: -0.0082871, -0.0075654, -0.0082799, -0.0075529, -0.0007343, 0.0007146
8: 0.0050915, 0.0093727, 0.0051417, 0.0094287, -0.0042664, 0.0041611
9: -0.0036858, -0.0032386, -0.0036854, -0.0032208, -0.0004650, 0.0004468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A2_A1_B1_A1

### Relational analysis result of IS_B1_B2_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006677, upper bound: 0.0006679
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A2_A2_A1_B1_A2

### Relational analysis result of IS_B1_B2_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006807
time: 0.71 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0008114, 0.0011147, 0.0007984, 0.0011151, -0.0003036, 0.0003164
1: 0.9934381, 0.9941609, 0.9934337, 0.9941887, -0.0007506, 0.0007272
2: -0.0084731, -0.0052033, -0.0085183, -0.0052625, -0.0029538, 0.0030529
3: 0.0037033, 0.0041406, 0.0036869, 0.0041436, -0.0004404, 0.0004538
4: 0.0025294, 0.0051137, 0.0025762, 0.0051494, -0.0026200, 0.0025375
5: 0.0052852, 0.0064716, 0.0052541, 0.0064589, -0.0011737, 0.0012175
6: -0.0020622, -0.0009272, -0.0020779, -0.0009478, -0.0011144, 0.0011507
7: -0.0082876, -0.0075628, -0.0082764, -0.0075496, -0.0007381, 0.0007136
8: 0.0050881, 0.0093842, 0.0051659, 0.0094436, -0.0042852, 0.0041496
9: -0.0036858, -0.0032350, -0.0036852, -0.0032160, -0.0004698, 0.0004502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A2_A1_B2_A1

### Relational analysis result of IS_B1_B2_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006700, upper bound: 0.0006702
time: 0.64 seconds

## Relational analysis of IS_B1_B2_A2_A2_A1_B2_A2

### Relational analysis result of IS_B1_B2_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006814
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008186, 0.0011146, 0.0008016, 0.0011150, -0.0002964, 0.0003129
1: 0.9934404, 0.9941456, 0.9934348, 0.9941816, -0.0007412, 0.0007108
2: -0.0084484, -0.0052022, -0.0085070, -0.0052399, -0.0029457, 0.0030447
3: 0.0037122, 0.0041390, 0.0036910, 0.0041429, -0.0004307, 0.0004480
4: 0.0025286, 0.0050941, 0.0025584, 0.0051405, -0.0026119, 0.0025358
5: 0.0053022, 0.0064719, 0.0052619, 0.0064637, -0.0011616, 0.0012099
6: -0.0020536, -0.0009268, -0.0020739, -0.0009399, -0.0011137, 0.0011471
7: -0.0082878, -0.0075701, -0.0082807, -0.0075529, -0.0007350, 0.0007107
8: 0.0050867, 0.0093517, 0.0051362, 0.0094288, -0.0042717, 0.0041447
9: -0.0036858, -0.0032453, -0.0036854, -0.0032208, -0.0004651, 0.0004401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A2_A2_B1_A1

### Relational analysis result of IS_B1_B2_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006708, upper bound: 0.0006721
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A2_A2_A2_B1_A2

### Relational analysis result of IS_B1_B2_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006835
time: 0.71 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008158, 0.0011146, 0.0007983, 0.0011151, -0.0002993, 0.0003163
1: 0.9934393, 0.9941516, 0.9934337, 0.9941886, -0.0007493, 0.0007179
2: -0.0084580, -0.0051997, -0.0085184, -0.0052571, -0.0029412, 0.0030576
3: 0.0037088, 0.0041396, 0.0036869, 0.0041436, -0.0004349, 0.0004528
4: 0.0025266, 0.0051017, 0.0025719, 0.0051495, -0.0026229, 0.0025298
5: 0.0052955, 0.0064724, 0.0052541, 0.0064600, -0.0011645, 0.0012183
6: -0.0020569, -0.0009259, -0.0020779, -0.0009459, -0.0011110, 0.0011519
7: -0.0082883, -0.0075673, -0.0082775, -0.0075495, -0.0007388, 0.0007102
8: 0.0050833, 0.0093643, 0.0051588, 0.0094437, -0.0042904, 0.0041360
9: -0.0036858, -0.0032413, -0.0036852, -0.0032160, -0.0004699, 0.0004439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A2_A2_A2_B2_A1

### Relational analysis result of IS_B1_B2_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006720, upper bound: 0.0006738
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A2_A2_A2_B2_A2

### Relational analysis result of IS_B1_B2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006841
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007689, 0.0011158, 0.0007700, 0.0011158, -0.0003469, 0.0003459
1: 0.9934240, 0.9942510, 0.9934244, 0.9942486, -0.0008246, 0.0008265
2: -0.0086201, -0.0053685, -0.0086163, -0.0053818, -0.0029721, 0.0029872
3: 0.0036500, 0.0041504, 0.0036514, 0.0041501, -0.0005001, 0.0004990
4: 0.0026600, 0.0052299, 0.0026705, 0.0052269, -0.0025669, 0.0025594
5: 0.0051843, 0.0064359, 0.0051869, 0.0064331, -0.0012488, 0.0012490
6: -0.0021132, -0.0009669, -0.0021119, -0.0009693, -0.0011439, 0.0011450
7: -0.0082564, -0.0075197, -0.0082539, -0.0075208, -0.0007356, 0.0007342
8: 0.0053052, 0.0095774, 0.0053226, 0.0095724, -0.0041959, 0.0041821
9: -0.0036841, -0.0031733, -0.0036840, -0.0031749, -0.0005092, 0.0005106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006731
time: 0.61 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006731
time: 0.61 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007688, 0.0011159, 0.0007758, 0.0011157, -0.0003468, 0.0003400
1: 0.9934240, 0.9942511, 0.9934263, 0.9942364, -0.0008124, 0.0008248
2: -0.0086203, -0.0053564, -0.0085961, -0.0053589, -0.0029904, 0.0029700
3: 0.0036499, 0.0041504, 0.0036587, 0.0041488, -0.0004989, 0.0004917
4: 0.0026504, 0.0052300, 0.0026524, 0.0052109, -0.0025605, 0.0025777
5: 0.0051842, 0.0064386, 0.0052008, 0.0064380, -0.0012538, 0.0012378
6: -0.0021133, -0.0009667, -0.0021049, -0.0009812, -0.0011321, 0.0011381
7: -0.0082587, -0.0075196, -0.0082582, -0.0075267, -0.0007320, 0.0007386
8: 0.0052893, 0.0095777, 0.0052925, 0.0095459, -0.0041839, 0.0042127
9: -0.0036842, -0.0031733, -0.0036842, -0.0031834, -0.0005008, 0.0005110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006787, upper bound: 0.0006770
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006790, upper bound: 0.0006813
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0007838, 0.0011155, -0.0003513, 0.0003322
1: 0.9934224, 0.9942610, 0.9934289, 0.9942195, -0.0007970, 0.0008321
2: -0.0086365, -0.0053636, -0.0085686, -0.0053183, -0.0030452, 0.0029351
3: 0.0036441, 0.0041514, 0.0036687, 0.0041470, -0.0005029, 0.0004828
4: 0.0026561, 0.0052428, 0.0026203, 0.0051892, -0.0025331, 0.0026225
5: 0.0051731, 0.0064370, 0.0052196, 0.0064468, -0.0012737, 0.0012174
6: -0.0021189, -0.0009565, -0.0020953, -0.0009671, -0.0011518, 0.0011389
7: -0.0082573, -0.0075149, -0.0082659, -0.0075348, -0.0007226, 0.0007510
8: 0.0052987, 0.0095989, 0.0052392, 0.0095097, -0.0041388, 0.0042868
9: -0.0036842, -0.0031665, -0.0036846, -0.0031949, -0.0004892, 0.0005182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006718, upper bound: 0.0006764
time: 0.60 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006871
time: 0.75 seconds

## BFS IS instance: IS_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0008114, 0.0011147, -0.0003506, 0.0003045
1: 0.9934224, 0.9942610, 0.9934381, 0.9941609, -0.0007384, 0.0008230
2: -0.0086365, -0.0053636, -0.0084731, -0.0052033, -0.0031702, 0.0028539
3: 0.0036441, 0.0041514, 0.0037033, 0.0041406, -0.0004966, 0.0004482
4: 0.0026561, 0.0052428, 0.0025294, 0.0051137, -0.0024576, 0.0027134
5: 0.0051731, 0.0064370, 0.0052852, 0.0064716, -0.0012986, 0.0011518
6: -0.0021189, -0.0009565, -0.0020622, -0.0009272, -0.0011917, 0.0011057
7: -0.0082573, -0.0075149, -0.0082876, -0.0075628, -0.0006945, 0.0007728
8: 0.0052987, 0.0095989, 0.0050881, 0.0093842, -0.0040171, 0.0044401
9: -0.0036842, -0.0031665, -0.0036858, -0.0032350, -0.0004492, 0.0005193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006718, upper bound: 0.0006764
time: 0.60 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006871
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007689, 0.0011159, 0.0007726, 0.0011158, -0.0003469, 0.0003433
1: 0.9934240, 0.9942509, 0.9934253, 0.9942432, -0.0008192, 0.0008256
2: -0.0086201, -0.0053627, -0.0086073, -0.0053646, -0.0029891, 0.0029817
3: 0.0036500, 0.0041504, 0.0036546, 0.0041495, -0.0004995, 0.0004957
4: 0.0026555, 0.0052299, 0.0026569, 0.0052198, -0.0025643, 0.0025730
5: 0.0051843, 0.0064372, 0.0051931, 0.0064368, -0.0012525, 0.0012441
6: -0.0021132, -0.0009669, -0.0021088, -0.0009750, -0.0011382, 0.0011419
7: -0.0082575, -0.0075197, -0.0082571, -0.0075234, -0.0007341, 0.0007375
8: 0.0052976, 0.0095774, 0.0053000, 0.0095606, -0.0041912, 0.0042043
9: -0.0036842, -0.0031733, -0.0036841, -0.0031787, -0.0005055, 0.0005108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
time: 0.62 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007688, 0.0011159, 0.0007785, 0.0011156, -0.0003468, 0.0003373
1: 0.9934239, 0.9942511, 0.9934272, 0.9942306, -0.0008067, 0.0008239
2: -0.0086203, -0.0053512, -0.0085868, -0.0053446, -0.0030016, 0.0029662
3: 0.0036499, 0.0041504, 0.0036620, 0.0041482, -0.0004983, 0.0004883
4: 0.0026463, 0.0052301, 0.0026411, 0.0052036, -0.0025573, 0.0025890
5: 0.0051842, 0.0064397, 0.0052071, 0.0064411, -0.0012569, 0.0012325
6: -0.0021133, -0.0009667, -0.0021017, -0.0009762, -0.0011370, 0.0011349
7: -0.0082597, -0.0075196, -0.0082609, -0.0075294, -0.0007302, 0.0007413
8: 0.0052825, 0.0095777, 0.0052737, 0.0095337, -0.0041779, 0.0042307
9: -0.0036843, -0.0031732, -0.0036844, -0.0031873, -0.0004970, 0.0005111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006808, upper bound: 0.0006775
time: 0.67 seconds

## Relational analysis of IS_B2_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006825
time: 0.62 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0007854, 0.0011154, -0.0003513, 0.0003306
1: 0.9934224, 0.9942611, 0.9934295, 0.9942161, -0.0007937, 0.0008316
2: -0.0086365, -0.0053572, -0.0085631, -0.0053034, -0.0030574, 0.0029333
3: 0.0036441, 0.0041515, 0.0036707, 0.0041466, -0.0005025, 0.0004808
4: 0.0026511, 0.0052428, 0.0026086, 0.0051848, -0.0025337, 0.0026343
5: 0.0051731, 0.0064384, 0.0052234, 0.0064500, -0.0012769, 0.0012150
6: -0.0021189, -0.0009565, -0.0020934, -0.0009620, -0.0011569, 0.0011370
7: -0.0082585, -0.0075148, -0.0082687, -0.0075364, -0.0007221, 0.0007539
8: 0.0052904, 0.0095990, 0.0052197, 0.0095025, -0.0041390, 0.0043057
9: -0.0036842, -0.0031665, -0.0036848, -0.0031972, -0.0004870, 0.0005183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006789
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006879
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007641, 0.0011160, 0.0008158, 0.0011146, -0.0003505, 0.0003002
1: 0.9934224, 0.9942611, 0.9934393, 0.9941516, -0.0007291, 0.0008218
2: -0.0086365, -0.0053572, -0.0084580, -0.0051997, -0.0031725, 0.0028423
3: 0.0036441, 0.0041515, 0.0037088, 0.0041396, -0.0004956, 0.0004427
4: 0.0026511, 0.0052428, 0.0025266, 0.0051017, -0.0024506, 0.0027163
5: 0.0051731, 0.0064384, 0.0052955, 0.0064724, -0.0012994, 0.0011428
6: -0.0021189, -0.0009565, -0.0020569, -0.0009259, -0.0011930, 0.0011005
7: -0.0082585, -0.0075148, -0.0082883, -0.0075673, -0.0006913, 0.0007735
8: 0.0052904, 0.0095990, 0.0050833, 0.0093643, -0.0040040, 0.0044451
9: -0.0036842, -0.0031665, -0.0036858, -0.0032413, -0.0004429, 0.0005194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006789
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006879
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0007866, 0.0011154, 0.0008006, 0.0011150, -0.0003284, 0.0003148
1: 0.9934298, 0.9942135, 0.9934344, 0.9941837, -0.0007539, 0.0007790
2: -0.0085588, -0.0053007, -0.0085104, -0.0052714, -0.0030282, 0.0029443
3: 0.0036722, 0.0041463, 0.0036897, 0.0041431, -0.0004709, 0.0004566
4: 0.0026064, 0.0051815, 0.0025832, 0.0051432, -0.0025368, 0.0025982
5: 0.0052263, 0.0064506, 0.0052596, 0.0064569, -0.0012306, 0.0011910
6: -0.0020920, -0.0009610, -0.0020751, -0.0009508, -0.0011411, 0.0011141
7: -0.0082692, -0.0075376, -0.0082748, -0.0075518, -0.0007174, 0.0007371
8: 0.0052161, 0.0094969, 0.0051776, 0.0094333, -0.0041455, 0.0042485
9: -0.0036848, -0.0031990, -0.0036851, -0.0032193, -0.0004655, 0.0004861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006747
time: 0.68 seconds

## Relational analysis of IS_B2_A1_A2_A1_B1_B1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006763
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0007847, 0.0011154, 0.0007955, 0.0011152, -0.0003305, 0.0003199
1: 0.9934292, 0.9942175, 0.9934328, 0.9941946, -0.0007653, 0.0007847
2: -0.0085656, -0.0052982, -0.0085281, -0.0052817, -0.0030255, 0.0029659
3: 0.0036698, 0.0041468, 0.0036833, 0.0041443, -0.0004745, 0.0004634
4: 0.0026044, 0.0051868, 0.0025913, 0.0051571, -0.0025527, 0.0025955
5: 0.0052217, 0.0064511, 0.0052475, 0.0064547, -0.0012330, 0.0012037
6: -0.0020943, -0.0009601, -0.0020813, -0.0009544, -0.0011399, 0.0011211
7: -0.0082697, -0.0075357, -0.0082728, -0.0075467, -0.0007230, 0.0007372
8: 0.0052128, 0.0095058, 0.0051910, 0.0094565, -0.0041716, 0.0042444
9: -0.0036848, -0.0031962, -0.0036850, -0.0032119, -0.0004729, 0.0004888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006821, upper bound: 0.0006753
time: 0.67 seconds

## Relational analysis of IS_B2_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006895, upper bound: 0.0006763
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0007822, 0.0011155, 0.0008092, 0.0011148, -0.0003326, 0.0003063
1: 0.9934284, 0.9942229, 0.9934373, 0.9941657, -0.0007372, 0.0007856
2: -0.0085742, -0.0052968, -0.0084809, -0.0052019, -0.0031138, 0.0029271
3: 0.0036666, 0.0041473, 0.0037004, 0.0041412, -0.0004745, 0.0004469
4: 0.0026033, 0.0051936, 0.0025283, 0.0051199, -0.0025165, 0.0026653
5: 0.0052158, 0.0064514, 0.0052798, 0.0064719, -0.0012562, 0.0011716
6: -0.0020973, -0.0009597, -0.0020649, -0.0009267, -0.0011706, 0.0011052
7: -0.0082700, -0.0075331, -0.0082879, -0.0075605, -0.0007094, 0.0007548
8: 0.0052109, 0.0095171, 0.0050863, 0.0093945, -0.0041144, 0.0043612
9: -0.0036848, -0.0031926, -0.0036858, -0.0032317, -0.0004532, 0.0004932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006807, upper bound: 0.0006713
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006763
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0007822, 0.0011155, 0.0008135, 0.0011147, -0.0003325, 0.0003020
1: 0.9934284, 0.9942229, 0.9934387, 0.9941565, -0.0007281, 0.0007843
2: -0.0085742, -0.0052968, -0.0084658, -0.0051983, -0.0031186, 0.0029091
3: 0.0036666, 0.0041473, 0.0037059, 0.0041401, -0.0004735, 0.0004414
4: 0.0026033, 0.0051936, 0.0025254, 0.0051079, -0.0025046, 0.0026682
5: 0.0052158, 0.0064514, 0.0052902, 0.0064727, -0.0012569, 0.0011613
6: -0.0020973, -0.0009597, -0.0020597, -0.0009255, -0.0011718, 0.0011000
7: -0.0082700, -0.0075331, -0.0082886, -0.0075650, -0.0007050, 0.0007554
8: 0.0052109, 0.0095171, 0.0050815, 0.0093747, -0.0040936, 0.0043663
9: -0.0036848, -0.0031926, -0.0036859, -0.0032380, -0.0004468, 0.0004933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006807, upper bound: 0.0006713
time: 0.61 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006763
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0007903, 0.0011153, 0.0007974, 0.0011151, -0.0003248, 0.0003179
1: 0.9934310, 0.9942057, 0.9934333, 0.9941906, -0.0007595, 0.0007724
2: -0.0085462, -0.0053055, -0.0085215, -0.0053254, -0.0029720, 0.0029675
3: 0.0036767, 0.0041455, 0.0036857, 0.0041438, -0.0004671, 0.0004598
4: 0.0026102, 0.0051715, 0.0026259, 0.0051519, -0.0025417, 0.0025456
5: 0.0052350, 0.0064495, 0.0052520, 0.0064453, -0.0012103, 0.0011976
6: -0.0020876, -0.0009627, -0.0020790, -0.0009696, -0.0011180, 0.0011163
7: -0.0082683, -0.0075414, -0.0082645, -0.0075486, -0.0007197, 0.0007232
8: 0.0052225, 0.0094803, 0.0052485, 0.0094478, -0.0041560, 0.0041635
9: -0.0036848, -0.0032043, -0.0036846, -0.0032147, -0.0004701, 0.0004802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006718
time: 0.83 seconds

## Relational analysis of IS_B2_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006716
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0007902, 0.0011153, 0.0008038, 0.0011149, -0.0003247, 0.0003115
1: 0.9934310, 0.9942058, 0.9934354, 0.9941772, -0.0007461, 0.0007704
2: -0.0085465, -0.0052978, -0.0084996, -0.0053064, -0.0029900, 0.0029435
3: 0.0036767, 0.0041455, 0.0036936, 0.0041424, -0.0004657, 0.0004519
4: 0.0026041, 0.0051717, 0.0026109, 0.0051347, -0.0025306, 0.0025608
5: 0.0052349, 0.0064512, 0.0052670, 0.0064494, -0.0012145, 0.0011842
6: -0.0020876, -0.0009600, -0.0020714, -0.0009630, -0.0011246, 0.0011114
7: -0.0082698, -0.0075413, -0.0082681, -0.0075550, -0.0007147, 0.0007268
8: 0.0052123, 0.0094806, 0.0052236, 0.0094191, -0.0041361, 0.0041892
9: -0.0036848, -0.0032042, -0.0036847, -0.0032238, -0.0004610, 0.0004805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A1_A2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006786, upper bound: 0.0006698
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006799, upper bound: 0.0006757
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0007879, 0.0011154, 0.0007910, 0.0011153, -0.0003274, 0.0003243
1: 0.9934303, 0.9942107, 0.9934313, 0.9942041, -0.0007738, 0.0007794
2: -0.0085543, -0.0053029, -0.0085437, -0.0053355, -0.0029743, 0.0029861
3: 0.0036738, 0.0041460, 0.0036777, 0.0041453, -0.0004715, 0.0004683
4: 0.0026082, 0.0051779, 0.0026339, 0.0051695, -0.0025613, 0.0025440
5: 0.0052295, 0.0064501, 0.0052368, 0.0064431, -0.0012136, 0.0012133
6: -0.0020904, -0.0009618, -0.0020867, -0.0009731, -0.0011173, 0.0011249
7: -0.0082688, -0.0075390, -0.0082626, -0.0075421, -0.0007267, 0.0007237
8: 0.0052190, 0.0094910, 0.0052618, 0.0094770, -0.0041886, 0.0041617
9: -0.0036848, -0.0032009, -0.0036844, -0.0032054, -0.0004794, 0.0004835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_B2_A1_A2_A2_B2_B1_B1

### Relational analysis result of IS_B2_A1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006720
time: 0.60 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2_B1_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006720
time: 0.60 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0007879, 0.0011154, 0.0007978, 0.0011151, -0.0003272, 0.0003175
1: 0.9934303, 0.9942108, 0.9934335, 0.9941897, -0.0007594, 0.0007774
2: -0.0085546, -0.0052952, -0.0085201, -0.0053154, -0.0029897, 0.0029658
3: 0.0036737, 0.0041460, 0.0036862, 0.0041437, -0.0004700, 0.0004598
4: 0.0026021, 0.0051781, 0.0026180, 0.0051508, -0.0025488, 0.0025601
5: 0.0052293, 0.0064518, 0.0052529, 0.0064474, -0.0012181, 0.0011989
6: -0.0020904, -0.0009591, -0.0020785, -0.0009661, -0.0011243, 0.0011194
7: -0.0082703, -0.0075389, -0.0082664, -0.0075490, -0.0007212, 0.0007275
8: 0.0052088, 0.0094913, 0.0052354, 0.0094460, -0.0041674, 0.0041888
9: -0.0036849, -0.0032008, -0.0036847, -0.0032153, -0.0004696, 0.0004838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A1_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006801, upper bound: 0.0006698
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006757
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007992, 0.0011151, 0.0007880, 0.0011153, -0.0003161, 0.0003270
1: 0.9934340, 0.9941866, 0.9934304, 0.9942104, -0.0007764, 0.0007563
2: -0.0085153, -0.0053147, -0.0085539, -0.0052779, -0.0029678, 0.0029666
3: 0.0036880, 0.0041434, 0.0036739, 0.0041460, -0.0004580, 0.0004695
4: 0.0026175, 0.0051470, 0.0025884, 0.0051776, -0.0025601, 0.0025587
5: 0.0052562, 0.0064476, 0.0052297, 0.0064555, -0.0011993, 0.0012179
6: -0.0020768, -0.0009659, -0.0020902, -0.0009531, -0.0011237, 0.0011244
7: -0.0082666, -0.0075504, -0.0082735, -0.0075391, -0.0007275, 0.0007231
8: 0.0052345, 0.0094397, 0.0051861, 0.0094905, -0.0041828, 0.0041814
9: -0.0036847, -0.0032173, -0.0036850, -0.0032011, -0.0004836, 0.0004678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007009, upper bound: 0.0007010
time: 0.75 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007010
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007940, 0.0011152, 0.0007851, 0.0011154, -0.0003214, 0.0003300
1: 0.9934323, 0.9941976, 0.9934294, 0.9942166, -0.0007843, 0.0007682
2: -0.0085332, -0.0053249, -0.0085639, -0.0052749, -0.0029874, 0.0029677
3: 0.0036815, 0.0041446, 0.0036704, 0.0041466, -0.0004652, 0.0004742
4: 0.0026255, 0.0051612, 0.0025860, 0.0051854, -0.0025599, 0.0025752
5: 0.0052439, 0.0064454, 0.0052229, 0.0064562, -0.0012122, 0.0012225
6: -0.0020830, -0.0009694, -0.0020937, -0.0009521, -0.0011310, 0.0011243
7: -0.0082646, -0.0075452, -0.0082741, -0.0075362, -0.0007285, 0.0007289
8: 0.0052479, 0.0094632, 0.0051822, 0.0095035, -0.0041832, 0.0042085
9: -0.0036846, -0.0032097, -0.0036851, -0.0031969, -0.0004877, 0.0004753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007002
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0008008, 0.0011150, 0.0007880, 0.0011153, -0.0003145, 0.0003270
1: 0.9934346, 0.9941832, 0.9934304, 0.9942104, -0.0007758, 0.0007529
2: -0.0085097, -0.0053001, -0.0085540, -0.0052714, -0.0029667, 0.0029799
3: 0.0036900, 0.0041431, 0.0036740, 0.0041460, -0.0004560, 0.0004691
4: 0.0026059, 0.0051426, 0.0025832, 0.0051776, -0.0025717, 0.0025594
5: 0.0052601, 0.0064507, 0.0052297, 0.0064569, -0.0011969, 0.0012210
6: -0.0020749, -0.0009608, -0.0020903, -0.0009508, -0.0011241, 0.0011295
7: -0.0082693, -0.0075521, -0.0082748, -0.0075391, -0.0007303, 0.0007227
8: 0.0052153, 0.0094323, 0.0051775, 0.0094905, -0.0042018, 0.0041823
9: -0.0036848, -0.0032196, -0.0036851, -0.0032010, -0.0004838, 0.0004655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007027, upper bound: 0.0007000
time: 0.69 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007012
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007957, 0.0011151, 0.0007851, 0.0011154, -0.0003197, 0.0003300
1: 0.9934329, 0.9941942, 0.9934294, 0.9942166, -0.0007837, 0.0007648
2: -0.0085274, -0.0053104, -0.0085639, -0.0052683, -0.0029867, 0.0029801
3: 0.0036836, 0.0041442, 0.0036704, 0.0041466, -0.0004630, 0.0004739
4: 0.0026141, 0.0051566, 0.0025808, 0.0051854, -0.0025714, 0.0025757
5: 0.0052480, 0.0064485, 0.0052229, 0.0064576, -0.0012096, 0.0012256
6: -0.0020810, -0.0009644, -0.0020937, -0.0009498, -0.0011312, 0.0011293
7: -0.0082674, -0.0075469, -0.0082753, -0.0075362, -0.0007312, 0.0007285
8: 0.0052288, 0.0094555, 0.0051736, 0.0095035, -0.0042018, 0.0042089
9: -0.0036847, -0.0032122, -0.0036851, -0.0031969, -0.0004878, 0.0004729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007009
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008006, 0.0011150, 0.0008140, 0.0011147, -0.0003140, 0.0003010
1: 0.9934344, 0.9941837, 0.9934388, 0.9941555, -0.0007210, 0.0007449
2: -0.0085104, -0.0052714, -0.0084643, -0.0052059, -0.0030426, 0.0029364
3: 0.0036897, 0.0041431, 0.0037064, 0.0041401, -0.0004503, 0.0004367
4: 0.0025832, 0.0051432, 0.0025314, 0.0051068, -0.0025235, 0.0026118
5: 0.0052596, 0.0064569, 0.0052912, 0.0064711, -0.0012115, 0.0011657
6: -0.0020751, -0.0009508, -0.0020591, -0.0009281, -0.0011470, 0.0011083
7: -0.0082748, -0.0075518, -0.0082871, -0.0075654, -0.0007094, 0.0007353
8: 0.0051776, 0.0094333, 0.0050915, 0.0093727, -0.0041266, 0.0042715
9: -0.0036851, -0.0032193, -0.0036858, -0.0032386, -0.0004465, 0.0004665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006876
time: 0.74 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006926
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007955, 0.0011152, 0.0008114, 0.0011147, -0.0003192, 0.0003037
1: 0.9934328, 0.9941946, 0.9934381, 0.9941609, -0.0007281, 0.0007565
2: -0.0085281, -0.0052817, -0.0084731, -0.0052033, -0.0030624, 0.0029354
3: 0.0036833, 0.0041443, 0.0037033, 0.0041406, -0.0004573, 0.0004410
4: 0.0025913, 0.0051571, 0.0025294, 0.0051137, -0.0025223, 0.0026277
5: 0.0052475, 0.0064547, 0.0052852, 0.0064716, -0.0012242, 0.0011695
6: -0.0020813, -0.0009544, -0.0020622, -0.0009272, -0.0011540, 0.0011078
7: -0.0082728, -0.0075467, -0.0082876, -0.0075628, -0.0007100, 0.0007409
8: 0.0051910, 0.0094565, 0.0050881, 0.0093842, -0.0041247, 0.0042975
9: -0.0036850, -0.0032119, -0.0036858, -0.0032350, -0.0004500, 0.0004739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006883
time: 0.74 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006938
time: 0.76 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008006, 0.0011150, 0.0008186, 0.0011146, -0.0003139, 0.0002964
1: 0.9934344, 0.9941838, 0.9934404, 0.9941456, -0.0007112, 0.0007434
2: -0.0085105, -0.0052648, -0.0084484, -0.0052022, -0.0030449, 0.0029244
3: 0.0036897, 0.0041431, 0.0037122, 0.0041390, -0.0004493, 0.0004309
4: 0.0025780, 0.0051432, 0.0025286, 0.0050941, -0.0025161, 0.0026147
5: 0.0052595, 0.0064584, 0.0053022, 0.0064719, -0.0012123, 0.0011562
6: -0.0020751, -0.0009485, -0.0020536, -0.0009268, -0.0011483, 0.0011051
7: -0.0082760, -0.0075518, -0.0082878, -0.0075701, -0.0007060, 0.0007360
8: 0.0051689, 0.0094333, 0.0050867, 0.0093517, -0.0041124, 0.0042762
9: -0.0036852, -0.0032193, -0.0036858, -0.0032453, -0.0004399, 0.0004665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006886
time: 0.70 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006934
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007955, 0.0011152, 0.0008158, 0.0011146, -0.0003191, 0.0002993
1: 0.9934328, 0.9941947, 0.9934393, 0.9941516, -0.0007188, 0.0007554
2: -0.0085281, -0.0052751, -0.0084580, -0.0051997, -0.0030647, 0.0029236
3: 0.0036833, 0.0041443, 0.0037088, 0.0041396, -0.0004563, 0.0004355
4: 0.0025862, 0.0051572, 0.0025266, 0.0051017, -0.0025155, 0.0026306
5: 0.0052475, 0.0064561, 0.0052955, 0.0064724, -0.0012250, 0.0011606
6: -0.0020813, -0.0009521, -0.0020569, -0.0009259, -0.0011553, 0.0011048
7: -0.0082741, -0.0075467, -0.0082883, -0.0075673, -0.0007068, 0.0007417
8: 0.0051825, 0.0094565, 0.0050833, 0.0093643, -0.0041118, 0.0043023
9: -0.0036851, -0.0032119, -0.0036858, -0.0032413, -0.0004438, 0.0004739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006893
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006944
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0008140, 0.0011147, 0.0008006, 0.0011150, -0.0003010, 0.0003140
1: 0.9934388, 0.9941555, 0.9934344, 0.9941837, -0.0007449, 0.0007210
2: -0.0084643, -0.0052059, -0.0085104, -0.0052714, -0.0029364, 0.0030426
3: 0.0037064, 0.0041401, 0.0036897, 0.0041431, -0.0004367, 0.0004503
4: 0.0025314, 0.0051068, 0.0025832, 0.0051432, -0.0026118, 0.0025235
5: 0.0052912, 0.0064711, 0.0052596, 0.0064569, -0.0011657, 0.0012115
6: -0.0020591, -0.0009281, -0.0020751, -0.0009508, -0.0011083, 0.0011470
7: -0.0082871, -0.0075654, -0.0082748, -0.0075518, -0.0007353, 0.0007094
8: 0.0050915, 0.0093727, 0.0051776, 0.0094333, -0.0042715, 0.0041266
9: -0.0036858, -0.0032386, -0.0036851, -0.0032193, -0.0004665, 0.0004465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006876, upper bound: 0.0006752
time: 0.75 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006925, upper bound: 0.0006840
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0008114, 0.0011147, 0.0007955, 0.0011152, -0.0003037, 0.0003192
1: 0.9934381, 0.9941609, 0.9934328, 0.9941946, -0.0007565, 0.0007281
2: -0.0084731, -0.0052033, -0.0085281, -0.0052817, -0.0029354, 0.0030624
3: 0.0037033, 0.0041406, 0.0036833, 0.0041443, -0.0004410, 0.0004573
4: 0.0025294, 0.0051137, 0.0025913, 0.0051571, -0.0026277, 0.0025223
5: 0.0052852, 0.0064716, 0.0052475, 0.0064547, -0.0011695, 0.0012242
6: -0.0020622, -0.0009272, -0.0020813, -0.0009544, -0.0011078, 0.0011540
7: -0.0082876, -0.0075628, -0.0082728, -0.0075467, -0.0007409, 0.0007100
8: 0.0050881, 0.0093842, 0.0051910, 0.0094565, -0.0042975, 0.0041247
9: -0.0036858, -0.0032350, -0.0036850, -0.0032119, -0.0004739, 0.0004500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006884, upper bound: 0.0006752
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006938, upper bound: 0.0006840
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008186, 0.0011146, 0.0008006, 0.0011150, -0.0002964, 0.0003139
1: 0.9934404, 0.9941456, 0.9934344, 0.9941838, -0.0007434, 0.0007112
2: -0.0084484, -0.0052022, -0.0085105, -0.0052648, -0.0029244, 0.0030449
3: 0.0037122, 0.0041390, 0.0036897, 0.0041431, -0.0004309, 0.0004493
4: 0.0025286, 0.0050941, 0.0025780, 0.0051432, -0.0026147, 0.0025161
5: 0.0053022, 0.0064719, 0.0052595, 0.0064584, -0.0011562, 0.0012123
6: -0.0020536, -0.0009268, -0.0020751, -0.0009485, -0.0011051, 0.0011483
7: -0.0082878, -0.0075701, -0.0082760, -0.0075518, -0.0007360, 0.0007060
8: 0.0050867, 0.0093517, 0.0051689, 0.0094333, -0.0042762, 0.0041124
9: -0.0036858, -0.0032453, -0.0036852, -0.0032193, -0.0004665, 0.0004399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006766
time: 0.66 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006934, upper bound: 0.0006856
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008158, 0.0011146, 0.0007955, 0.0011152, -0.0002993, 0.0003191
1: 0.9934393, 0.9941516, 0.9934328, 0.9941947, -0.0007554, 0.0007188
2: -0.0084580, -0.0051997, -0.0085281, -0.0052751, -0.0029236, 0.0030647
3: 0.0037088, 0.0041396, 0.0036833, 0.0041443, -0.0004355, 0.0004563
4: 0.0025266, 0.0051017, 0.0025862, 0.0051572, -0.0026306, 0.0025155
5: 0.0052955, 0.0064724, 0.0052475, 0.0064561, -0.0011606, 0.0012250
6: -0.0020569, -0.0009259, -0.0020813, -0.0009521, -0.0011048, 0.0011553
7: -0.0082883, -0.0075673, -0.0082741, -0.0075467, -0.0007417, 0.0007068
8: 0.0050833, 0.0093643, 0.0051825, 0.0094565, -0.0043023, 0.0041118
9: -0.0036858, -0.0032413, -0.0036851, -0.0032119, -0.0004739, 0.0004438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006766
time: 0.61 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006944, upper bound: 0.0006856
time: 0.64 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0008184, 0.0011146, 0.0008261, 0.0011144, -0.0002960, 0.0002884
1: 0.9934403, 0.9941462, 0.9934428, 0.9941297, -0.0006894, 0.0007034
2: -0.0084491, -0.0051667, -0.0084224, -0.0052004, -0.0029896, 0.0029988
3: 0.0037120, 0.0041390, 0.0037216, 0.0041373, -0.0004253, 0.0004174
4: 0.0025005, 0.0050947, 0.0025271, 0.0050736, -0.0025731, 0.0025676
5: 0.0053016, 0.0064796, 0.0053200, 0.0064723, -0.0011706, 0.0011596
6: -0.0020538, -0.0009145, -0.0020446, -0.0009262, -0.0011276, 0.0011301
7: -0.0082946, -0.0075698, -0.0082882, -0.0075777, -0.0007169, 0.0007183
8: 0.0050400, 0.0093527, 0.0050843, 0.0093176, -0.0042090, 0.0041988
9: -0.0036862, -0.0032450, -0.0036858, -0.0032562, -0.0004300, 0.0004408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006766
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006839, upper bound: 0.0006856
time: 0.73 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0008156, 0.0011146, 0.0008221, 0.0011145, -0.0002989, 0.0002925
1: 0.9934393, 0.9941520, 0.9934415, 0.9941384, -0.0006991, 0.0007105
2: -0.0084587, -0.0051641, -0.0084363, -0.0052101, -0.0029880, 0.0030141
3: 0.0037085, 0.0041397, 0.0037166, 0.0041382, -0.0004297, 0.0004231
4: 0.0024985, 0.0051023, 0.0025348, 0.0050846, -0.0025862, 0.0025675
5: 0.0052951, 0.0064801, 0.0053104, 0.0064702, -0.0011751, 0.0011697
6: -0.0020572, -0.0009136, -0.0020494, -0.0009295, -0.0011277, 0.0011358
7: -0.0082950, -0.0075670, -0.0082864, -0.0075736, -0.0007215, 0.0007193
8: 0.0050366, 0.0093653, 0.0050970, 0.0093359, -0.0042304, 0.0041985
9: -0.0036862, -0.0032410, -0.0036857, -0.0032504, -0.0004358, 0.0004447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006828
time: 0.71 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006856
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0008184, 0.0011146, 0.0008318, 0.0011142, -0.0002958, 0.0002827
1: 0.9934402, 0.9941460, 0.9934447, 0.9941177, -0.0006775, 0.0007014
2: -0.0084491, -0.0051628, -0.0084027, -0.0051972, -0.0029942, 0.0029815
3: 0.0037119, 0.0041390, 0.0037287, 0.0041360, -0.0004241, 0.0004103
4: 0.0024974, 0.0050948, 0.0025246, 0.0050580, -0.0025606, 0.0025702
5: 0.0053016, 0.0064804, 0.0053335, 0.0064730, -0.0011714, 0.0011469
6: -0.0020539, -0.0009132, -0.0020377, -0.0009251, -0.0011288, 0.0011246
7: -0.0082953, -0.0075698, -0.0082888, -0.0075835, -0.0007118, 0.0007190
8: 0.0050349, 0.0093528, 0.0050800, 0.0092917, -0.0041872, 0.0042034
9: -0.0036862, -0.0032450, -0.0036859, -0.0032645, -0.0004218, 0.0004409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006841
time: 0.75 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006855
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0008156, 0.0011146, 0.0008263, 0.0011143, -0.0002987, 0.0002884
1: 0.9934393, 0.9941521, 0.9934428, 0.9941294, -0.0006901, 0.0007093
2: -0.0084587, -0.0051603, -0.0084218, -0.0052065, -0.0029928, 0.0029994
3: 0.0037085, 0.0041397, 0.0037218, 0.0041372, -0.0004288, 0.0004179
4: 0.0024954, 0.0051023, 0.0025320, 0.0050732, -0.0025778, 0.0025704
5: 0.0052950, 0.0064809, 0.0053204, 0.0064709, -0.0011759, 0.0011606
6: -0.0020572, -0.0009123, -0.0020444, -0.0009283, -0.0011289, 0.0011321
7: -0.0082958, -0.0075670, -0.0082870, -0.0075778, -0.0007179, 0.0007200
8: 0.0050316, 0.0093654, 0.0050923, 0.0093169, -0.0042152, 0.0042036
9: -0.0036862, -0.0032410, -0.0036858, -0.0032564, -0.0004298, 0.0004448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006842
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006855
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.29 seconds
IS_B1_B1_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006895
IS_B1_B1_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006962, upper bound: 0.0006900
IS_B1_B1_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006943, upper bound: 0.0006837
IS_B1_B1_A1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006960
IS_B1_B1_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006911
IS_B1_B1_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006915
IS_B1_B1_A1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006943, upper bound: 0.0006845
IS_B1_B1_A1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006964, upper bound: 0.0006964
IS_B1_B1_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006559
IS_B1_B1_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006836, upper bound: 0.0006763
IS_B1_B1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006744, upper bound: 0.0006560
IS_B1_B1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006869, upper bound: 0.0006763
IS_B1_B1_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006744, upper bound: 0.0006589
IS_B1_B1_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006782
IS_B1_B1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006765, upper bound: 0.0006589
IS_B1_B1_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006876, upper bound: 0.0006782
IS_B1_B1_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006730, upper bound: 0.0006697
IS_B1_B1_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006730, upper bound: 0.0006697
IS_B1_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006769, upper bound: 0.0006787
IS_B1_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006813, upper bound: 0.0006790
IS_B1_B1_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006760, upper bound: 0.0006720
IS_B1_B1_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006871, upper bound: 0.0006825
IS_B1_B1_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006760, upper bound: 0.0006720
IS_B1_B1_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006871, upper bound: 0.0006825
IS_B1_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006738
IS_B1_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006738
IS_B1_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006774, upper bound: 0.0006808
IS_B1_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006823, upper bound: 0.0006813
IS_B1_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006789, upper bound: 0.0006756
IS_B1_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006851
IS_B1_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006789, upper bound: 0.0006756
IS_B1_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006879, upper bound: 0.0006851
IS_B1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006716
IS_B1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006744
IS_B1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006833
IS_B1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006847
IS_B1_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006744
IS_B1_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006869
IS_B1_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006745, upper bound: 0.0006821
IS_B1_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006894
IS_B1_B2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006589, upper bound: 0.0006744
IS_B1_B2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006845
IS_B1_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006753, upper bound: 0.0006766
IS_B1_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006881
IS_B1_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006589, upper bound: 0.0006765
IS_B1_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006876
IS_B1_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006777
IS_B1_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006902
IS_B1_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006650, upper bound: 0.0006589
IS_B1_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006713, upper bound: 0.0006782
IS_B1_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006560, upper bound: 0.0006689
IS_B1_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006782
IS_B1_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006683, upper bound: 0.0006589
IS_B1_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006782
IS_B1_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006589, upper bound: 0.0006700
IS_B1_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006782
IS_B1_B2_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006677, upper bound: 0.0006679
IS_B1_B2_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006807
IS_B1_B2_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006700, upper bound: 0.0006702
IS_B1_B2_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006814
IS_B1_B2_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006708, upper bound: 0.0006721
IS_B1_B2_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006835
IS_B1_B2_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006720, upper bound: 0.0006738
IS_B1_B2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006841
IS_B2_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006731
IS_B2_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006731
IS_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006787, upper bound: 0.0006770
IS_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006790, upper bound: 0.0006813
IS_B2_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006718, upper bound: 0.0006764
IS_B2_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006871
IS_B2_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006718, upper bound: 0.0006764
IS_B2_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006871
IS_B2_A1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
IS_B2_A1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
IS_B2_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006808, upper bound: 0.0006775
IS_B2_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006825
IS_B2_A1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006789
IS_B2_A1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006879
IS_B2_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006789
IS_B2_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006879
IS_B2_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006747
IS_B2_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006763
IS_B2_A1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006821, upper bound: 0.0006753
IS_B2_A1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006895, upper bound: 0.0006763
IS_B2_A1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006807, upper bound: 0.0006713
IS_B2_A1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006763
IS_B2_A1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006807, upper bound: 0.0006713
IS_B2_A1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006763
IS_B2_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006718
IS_B2_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006716
IS_B2_A1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006786, upper bound: 0.0006698
IS_B2_A1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006799, upper bound: 0.0006757
IS_B2_A1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006720
IS_B2_A1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006720
IS_B2_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006801, upper bound: 0.0006698
IS_B2_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006757
IS_B2_A2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007009, upper bound: 0.0007010
IS_B2_A2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007010
IS_B2_A2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007002
IS_B2_A2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
IS_B2_A2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007027, upper bound: 0.0007000
IS_B2_A2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007012
IS_B2_A2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007009
IS_B2_A2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
IS_B2_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006876
IS_B2_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006926
IS_B2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006883
IS_B2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006938
IS_B2_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006886
IS_B2_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006934
IS_B2_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006893
IS_B2_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006944
IS_B2_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006876, upper bound: 0.0006752
IS_B2_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006925, upper bound: 0.0006840
IS_B2_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006884, upper bound: 0.0006752
IS_B2_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006938, upper bound: 0.0006840
IS_B2_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006766
IS_B2_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006934, upper bound: 0.0006856
IS_B2_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006766
IS_B2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006944, upper bound: 0.0006856
IS_B2_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006766
IS_B2_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006839, upper bound: 0.0006856
IS_B2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006828
IS_B2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006856
IS_B2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006841
IS_B2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006855
IS_B2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006842
IS_B2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006855

## BFS IS instance: IS_B1_B1_A1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007671, 0.0011159, 0.0007514, 0.0011163, -0.0003492, 0.0003645
1: 0.9934235, 0.9942548, 0.9934182, 0.9942880, -0.0008646, 0.0008366
2: -0.0086262, -0.0054102, -0.0086806, -0.0054205, -0.0029315, 0.0029944
3: 0.0036478, 0.0041508, 0.0036281, 0.0041544, -0.0005066, 0.0005227
4: 0.0026929, 0.0052347, 0.0027011, 0.0052777, -0.0025848, 0.0025336
5: 0.0051801, 0.0064269, 0.0051428, 0.0064247, -0.0012446, 0.0012841
6: -0.0021153, -0.0009630, -0.0021342, -0.0009285, -0.0011869, 0.0011712
7: -0.0082485, -0.0075179, -0.0082466, -0.0075019, -0.0007466, 0.0007287
8: 0.0053599, 0.0095854, 0.0053735, 0.0096569, -0.0042233, 0.0041387
9: -0.0036837, -0.0031708, -0.0036836, -0.0031480, -0.0005357, 0.0005128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006868
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006820, upper bound: 0.0006868
time: 0.71 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007671, 0.0011159, 0.0007568, 0.0011162, -0.0003491, 0.0003591
1: 0.9934235, 0.9942549, 0.9934201, 0.9942766, -0.0008532, 0.0008349
2: -0.0086264, -0.0053985, -0.0086619, -0.0054007, -0.0029522, 0.0029882
3: 0.0036477, 0.0041508, 0.0036349, 0.0041531, -0.0005054, 0.0005159
4: 0.0026837, 0.0052348, 0.0026854, 0.0052629, -0.0025792, 0.0025494
5: 0.0051800, 0.0064295, 0.0051557, 0.0064290, -0.0012490, 0.0012738
6: -0.0021154, -0.0009629, -0.0021277, -0.0009404, -0.0011750, 0.0011648
7: -0.0082507, -0.0075178, -0.0082503, -0.0075074, -0.0007433, 0.0007325
8: 0.0053446, 0.0095856, 0.0053475, 0.0096322, -0.0042140, 0.0041650
9: -0.0036838, -0.0031707, -0.0036838, -0.0031558, -0.0005280, 0.0005131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006927, upper bound: 0.0006853
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006938, upper bound: 0.0006874
time: 0.70 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007577, 0.0011161, 0.0007538, 0.0011162, -0.0003585, 0.0003624
1: 0.9934204, 0.9942747, 0.9934191, 0.9942830, -0.0008626, 0.0008556
2: -0.0086586, -0.0054670, -0.0086722, -0.0053749, -0.0030080, 0.0029309
3: 0.0036360, 0.0041529, 0.0036311, 0.0041538, -0.0005178, 0.0005218
4: 0.0027378, 0.0052603, 0.0026651, 0.0052711, -0.0025332, 0.0025953
5: 0.0051579, 0.0064147, 0.0051486, 0.0064346, -0.0012767, 0.0012661
6: -0.0021266, -0.0009424, -0.0021313, -0.0009338, -0.0011928, 0.0011889
7: -0.0082378, -0.0075084, -0.0082552, -0.0075044, -0.0007334, 0.0007468
8: 0.0054346, 0.0096280, 0.0053136, 0.0096459, -0.0041379, 0.0042409
9: -0.0036831, -0.0031572, -0.0036840, -0.0031515, -0.0005316, 0.0005268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A1_A2_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006914, upper bound: 0.0006806
time: 0.74 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A2_A1_A2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006918, upper bound: 0.0006812
time: 0.79 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007633, 0.0011160, 0.0007537, 0.0011162, -0.0003529, 0.0003623
1: 0.9934222, 0.9942628, 0.9934191, 0.9942831, -0.0008609, 0.0008437
2: -0.0086391, -0.0054486, -0.0086724, -0.0053642, -0.0029998, 0.0029512
3: 0.0036431, 0.0041516, 0.0036310, 0.0041538, -0.0005107, 0.0005206
4: 0.0027233, 0.0052449, 0.0026566, 0.0052713, -0.0025480, 0.0025883
5: 0.0051712, 0.0064186, 0.0051484, 0.0064369, -0.0012656, 0.0012702
6: -0.0021198, -0.0009548, -0.0021314, -0.0009337, -0.0011862, 0.0011766
7: -0.0082413, -0.0075141, -0.0082572, -0.0075043, -0.0007370, 0.0007431
8: 0.0054104, 0.0096024, 0.0052996, 0.0096462, -0.0041628, 0.0042293
9: -0.0036833, -0.0031654, -0.0036842, -0.0031514, -0.0005319, 0.0005188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006935
time: 0.75 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006935
time: 0.66 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007691, 0.0011158, 0.0007514, 0.0011163, -0.0003472, 0.0003645
1: 0.9934241, 0.9942507, 0.9934182, 0.9942881, -0.0008640, 0.0008324
2: -0.0086195, -0.0053980, -0.0086807, -0.0054144, -0.0029308, 0.0030059
3: 0.0036502, 0.0041503, 0.0036281, 0.0041544, -0.0005042, 0.0005223
4: 0.0026833, 0.0052294, 0.0026963, 0.0052777, -0.0025945, 0.0025331
5: 0.0051848, 0.0064296, 0.0051428, 0.0064260, -0.0012413, 0.0012868
6: -0.0021130, -0.0009673, -0.0021342, -0.0009285, -0.0011845, 0.0011669
7: -0.0082508, -0.0075199, -0.0082477, -0.0075019, -0.0007489, 0.0007279
8: 0.0053439, 0.0095766, 0.0053655, 0.0096570, -0.0042389, 0.0041379
9: -0.0036838, -0.0031736, -0.0036836, -0.0031480, -0.0005358, 0.0005100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006881
time: 0.80 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006820, upper bound: 0.0006885
time: 0.74 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007690, 0.0011158, 0.0007568, 0.0011162, -0.0003472, 0.0003591
1: 0.9934241, 0.9942507, 0.9934201, 0.9942766, -0.0008525, 0.0008307
2: -0.0086197, -0.0053877, -0.0086619, -0.0053943, -0.0029525, 0.0029986
3: 0.0036501, 0.0041503, 0.0036349, 0.0041531, -0.0005030, 0.0005155
4: 0.0026751, 0.0052295, 0.0026804, 0.0052629, -0.0025878, 0.0025492
5: 0.0051846, 0.0064318, 0.0051557, 0.0064304, -0.0012458, 0.0012761
6: -0.0021131, -0.0009671, -0.0021277, -0.0009404, -0.0011727, 0.0011606
7: -0.0082528, -0.0075198, -0.0082515, -0.0075074, -0.0007454, 0.0007317
8: 0.0053304, 0.0095768, 0.0053391, 0.0096323, -0.0042281, 0.0041647
9: -0.0036839, -0.0031735, -0.0036838, -0.0031558, -0.0005281, 0.0005103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006928, upper bound: 0.0006853
time: 0.70 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006940, upper bound: 0.0006889
time: 0.69 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007595, 0.0011161, 0.0007538, 0.0011162, -0.0003567, 0.0003623
1: 0.9934210, 0.9942707, 0.9934191, 0.9942830, -0.0008620, 0.0008517
2: -0.0086524, -0.0054534, -0.0086722, -0.0053689, -0.0030086, 0.0029434
3: 0.0036383, 0.0041525, 0.0036311, 0.0041538, -0.0005155, 0.0005214
4: 0.0027270, 0.0052554, 0.0026603, 0.0052711, -0.0025441, 0.0025951
5: 0.0051622, 0.0064176, 0.0051486, 0.0064359, -0.0012737, 0.0012690
6: -0.0021244, -0.0009464, -0.0021313, -0.0009338, -0.0011906, 0.0011849
7: -0.0082404, -0.0075102, -0.0082563, -0.0075044, -0.0007360, 0.0007461
8: 0.0054166, 0.0096198, 0.0053057, 0.0096459, -0.0041555, 0.0042407
9: -0.0036832, -0.0031598, -0.0036841, -0.0031515, -0.0005317, 0.0005243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006914, upper bound: 0.0006810
time: 0.69 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_A1_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006920, upper bound: 0.0006820
time: 0.69 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007648, 0.0011160, 0.0007537, 0.0011162, -0.0003515, 0.0003622
1: 0.9934227, 0.9942597, 0.9934191, 0.9942831, -0.0008604, 0.0008406
2: -0.0086343, -0.0054337, -0.0086725, -0.0053578, -0.0030016, 0.0029644
3: 0.0036449, 0.0041513, 0.0036311, 0.0041538, -0.0005090, 0.0005202
4: 0.0027116, 0.0052411, 0.0026515, 0.0052713, -0.0025597, 0.0025895
5: 0.0051746, 0.0064218, 0.0051484, 0.0064382, -0.0012636, 0.0012734
6: -0.0021181, -0.0009579, -0.0021314, -0.0009337, -0.0011845, 0.0011735
7: -0.0082441, -0.0075155, -0.0082584, -0.0075043, -0.0007398, 0.0007429
8: 0.0053909, 0.0095960, 0.0052912, 0.0096462, -0.0041819, 0.0042315
9: -0.0036834, -0.0031674, -0.0036842, -0.0031514, -0.0005320, 0.0005168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006941
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006941
time: 0.64 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007809, 0.0011155, 0.0007689, 0.0011158, -0.0003349, 0.0003467
1: 0.9934280, 0.9942254, 0.9934240, 0.9942510, -0.0008230, 0.0008014
2: -0.0085785, -0.0053542, -0.0086201, -0.0053685, -0.0029488, 0.0030006
3: 0.0036650, 0.0041476, 0.0036500, 0.0041504, -0.0004853, 0.0004976
4: 0.0026487, 0.0051970, 0.0026600, 0.0052299, -0.0025812, 0.0025370
5: 0.0052128, 0.0064390, 0.0051843, 0.0064359, -0.0012231, 0.0012547
6: -0.0020988, -0.0009796, -0.0021132, -0.0009669, -0.0011319, 0.0011336
7: -0.0082591, -0.0075319, -0.0082564, -0.0075197, -0.0007394, 0.0007245
8: 0.0052864, 0.0095228, 0.0053052, 0.0095774, -0.0042197, 0.0041459
9: -0.0036843, -0.0031908, -0.0036841, -0.0031733, -0.0005109, 0.0004933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006659, upper bound: 0.0006533
time: 0.60 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006692, upper bound: 0.0006532
time: 0.65 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007870, 0.0011154, 0.0007688, 0.0011159, -0.0003288, 0.0003466
1: 0.9934300, 0.9942125, 0.9934240, 0.9942511, -0.0008211, 0.0007885
2: -0.0085573, -0.0053427, -0.0086203, -0.0053564, -0.0029379, 0.0030132
3: 0.0036727, 0.0041462, 0.0036499, 0.0041504, -0.0004776, 0.0004963
4: 0.0026396, 0.0051803, 0.0026504, 0.0052300, -0.0025905, 0.0025298
5: 0.0052274, 0.0064415, 0.0051842, 0.0064386, -0.0012112, 0.0012573
6: -0.0020914, -0.0009756, -0.0021133, -0.0009667, -0.0011247, 0.0011377
7: -0.0082613, -0.0075381, -0.0082587, -0.0075196, -0.0007417, 0.0007206
8: 0.0052712, 0.0094949, 0.0052893, 0.0095777, -0.0042339, 0.0041347
9: -0.0036844, -0.0031997, -0.0036842, -0.0031733, -0.0005111, 0.0004846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006767, upper bound: 0.0006738
time: 0.76 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006739
time: 0.72 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007778, 0.0011156, 0.0007643, 0.0011160, -0.0003382, 0.0003513
1: 0.9934269, 0.9942322, 0.9934225, 0.9942607, -0.0008338, 0.0008097
2: -0.0085894, -0.0053512, -0.0086360, -0.0053816, -0.0029460, 0.0030183
3: 0.0036611, 0.0041483, 0.0036443, 0.0041514, -0.0004903, 0.0005041
4: 0.0026463, 0.0052056, 0.0026703, 0.0052424, -0.0025961, 0.0025353
5: 0.0052054, 0.0064397, 0.0051734, 0.0064331, -0.0012277, 0.0012662
6: -0.0021026, -0.0009785, -0.0021187, -0.0009568, -0.0011458, 0.0011402
7: -0.0082597, -0.0075287, -0.0082539, -0.0075150, -0.0007446, 0.0007252
8: 0.0052824, 0.0095371, 0.0053224, 0.0095982, -0.0042441, 0.0041432
9: -0.0036843, -0.0031862, -0.0036840, -0.0031667, -0.0005176, 0.0004978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006683, upper bound: 0.0006532
time: 0.71 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006719, upper bound: 0.0006532
time: 0.65 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007851, 0.0011154, 0.0007642, 0.0011160, -0.0003309, 0.0003512
1: 0.9934294, 0.9942166, 0.9934224, 0.9942609, -0.0008315, 0.0007942
2: -0.0085641, -0.0053402, -0.0086362, -0.0053709, -0.0029327, 0.0030305
3: 0.0036703, 0.0041467, 0.0036442, 0.0041514, -0.0004811, 0.0005025
4: 0.0026376, 0.0051856, 0.0026619, 0.0052426, -0.0026050, 0.0025237
5: 0.0052228, 0.0064421, 0.0051733, 0.0064354, -0.0012127, 0.0012688
6: -0.0020938, -0.0009747, -0.0021188, -0.0009567, -0.0011371, 0.0011441
7: -0.0082618, -0.0075361, -0.0082559, -0.0075150, -0.0007468, 0.0007198
8: 0.0052679, 0.0095038, 0.0053084, 0.0095985, -0.0042580, 0.0041246
9: -0.0036844, -0.0031968, -0.0036841, -0.0031666, -0.0005178, 0.0004873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006791, upper bound: 0.0006739
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006739
time: 0.69 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007829, 0.0011155, 0.0007689, 0.0011159, -0.0003330, 0.0003466
1: 0.9934286, 0.9942215, 0.9934240, 0.9942509, -0.0008222, 0.0007975
2: -0.0085719, -0.0053419, -0.0086201, -0.0053627, -0.0029447, 0.0030131
3: 0.0036675, 0.0041472, 0.0036500, 0.0041504, -0.0004829, 0.0004972
4: 0.0026389, 0.0051918, 0.0026555, 0.0052299, -0.0025910, 0.0025363
5: 0.0052174, 0.0064417, 0.0051843, 0.0064372, -0.0012198, 0.0012574
6: -0.0020965, -0.0009753, -0.0021132, -0.0009669, -0.0011296, 0.0011379
7: -0.0082614, -0.0075338, -0.0082575, -0.0075197, -0.0007418, 0.0007236
8: 0.0052702, 0.0095140, 0.0052976, 0.0095774, -0.0042363, 0.0041452
9: -0.0036844, -0.0031936, -0.0036842, -0.0031733, -0.0005110, 0.0004906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006690, upper bound: 0.0006564
time: 0.63 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006719, upper bound: 0.0006564
time: 0.59 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007906, 0.0011153, 0.0007688, 0.0011159, -0.0003253, 0.0003465
1: 0.9934312, 0.9942051, 0.9934239, 0.9942511, -0.0008199, 0.0007812
2: -0.0085452, -0.0053337, -0.0086203, -0.0053512, -0.0029286, 0.0030221
3: 0.0036771, 0.0041454, 0.0036499, 0.0041504, -0.0004732, 0.0004955
4: 0.0026325, 0.0051707, 0.0026463, 0.0052301, -0.0025976, 0.0025244
5: 0.0052357, 0.0064435, 0.0051842, 0.0064397, -0.0012040, 0.0012593
6: -0.0020872, -0.0009725, -0.0021133, -0.0009667, -0.0011205, 0.0011408
7: -0.0082630, -0.0075416, -0.0082597, -0.0075196, -0.0007434, 0.0007180
8: 0.0052595, 0.0094790, 0.0052825, 0.0095777, -0.0042459, 0.0041237
9: -0.0036845, -0.0032047, -0.0036843, -0.0031732, -0.0005112, 0.0004796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006772, upper bound: 0.0006756
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006822, upper bound: 0.0006757
time: 0.72 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007798, 0.0011156, 0.0007643, 0.0011160, -0.0003362, 0.0003513
1: 0.9934276, 0.9942279, 0.9934225, 0.9942608, -0.0008332, 0.0008054
2: -0.0085823, -0.0053388, -0.0086360, -0.0053756, -0.0029431, 0.0030307
3: 0.0036637, 0.0041479, 0.0036442, 0.0041514, -0.0004877, 0.0005036
4: 0.0026365, 0.0052000, 0.0026656, 0.0052424, -0.0026060, 0.0025344
5: 0.0052102, 0.0064424, 0.0051734, 0.0064344, -0.0012242, 0.0012690
6: -0.0021001, -0.0009742, -0.0021187, -0.0009568, -0.0011433, 0.0011445
7: -0.0082620, -0.0075308, -0.0082551, -0.0075150, -0.0007470, 0.0007243
8: 0.0052661, 0.0095278, 0.0053145, 0.0095983, -0.0042608, 0.0041423
9: -0.0036844, -0.0031892, -0.0036840, -0.0031667, -0.0005177, 0.0004949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006702, upper bound: 0.0006564
time: 0.61 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006564
time: 0.61 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007882, 0.0011153, 0.0007642, 0.0011160, -0.0003278, 0.0003511
1: 0.9934304, 0.9942100, 0.9934224, 0.9942608, -0.0008304, 0.0007876
2: -0.0085533, -0.0053310, -0.0086362, -0.0053646, -0.0029256, 0.0030390
3: 0.0036742, 0.0041459, 0.0036441, 0.0041514, -0.0004773, 0.0005018
4: 0.0026304, 0.0051771, 0.0026569, 0.0052426, -0.0026122, 0.0025203
5: 0.0052301, 0.0064440, 0.0051733, 0.0064368, -0.0012067, 0.0012708
6: -0.0020900, -0.0009715, -0.0021188, -0.0009566, -0.0011334, 0.0011473
7: -0.0082635, -0.0075392, -0.0082571, -0.0075149, -0.0007485, 0.0007179
8: 0.0052559, 0.0094897, 0.0053000, 0.0095986, -0.0042700, 0.0041181
9: -0.0036845, -0.0032013, -0.0036841, -0.0031666, -0.0005179, 0.0004828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006794, upper bound: 0.0006757
time: 0.67 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006757
time: 0.68 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007820, 0.0011155, 0.0007689, 0.0011158, -0.0003338, 0.0003466
1: 0.9934284, 0.9942232, 0.9934240, 0.9942510, -0.0008225, 0.0007992
2: -0.0085746, -0.0053837, -0.0086201, -0.0053685, -0.0029422, 0.0029701
3: 0.0036665, 0.0041474, 0.0036500, 0.0041504, -0.0004839, 0.0004973
4: 0.0026720, 0.0051939, 0.0026600, 0.0052299, -0.0025578, 0.0025339
5: 0.0052155, 0.0064326, 0.0051843, 0.0064359, -0.0012204, 0.0012483
6: -0.0020974, -0.0009898, -0.0021132, -0.0009669, -0.0011305, 0.0011234
7: -0.0082535, -0.0075330, -0.0082564, -0.0075197, -0.0007338, 0.0007234
8: 0.0053252, 0.0095177, 0.0053052, 0.0095774, -0.0041795, 0.0041397
9: -0.0036840, -0.0031924, -0.0036841, -0.0031733, -0.0005106, 0.0004917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006695, upper bound: 0.0006663
time: 0.63 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006706, upper bound: 0.0006668
time: 0.71 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0008071, 0.0011149, 0.0007689, 0.0011158, -0.0003087, 0.0003460
1: 0.9934365, 0.9941701, 0.9934240, 0.9942510, -0.0008144, 0.0007461
2: -0.0084881, -0.0052573, -0.0086201, -0.0053685, -0.0028636, 0.0031006
3: 0.0036978, 0.0041416, 0.0036500, 0.0041504, -0.0004526, 0.0004916
4: 0.0025721, 0.0051256, 0.0026600, 0.0052299, -0.0026578, 0.0024656
5: 0.0052749, 0.0064600, 0.0051843, 0.0064359, -0.0011611, 0.0012756
6: -0.0020674, -0.0009460, -0.0021132, -0.0009669, -0.0011005, 0.0011673
7: -0.0082774, -0.0075584, -0.0082564, -0.0075197, -0.0007578, 0.0006980
8: 0.0051591, 0.0094040, 0.0053052, 0.0095774, -0.0043478, 0.0040282
9: -0.0036852, -0.0032286, -0.0036841, -0.0031733, -0.0005119, 0.0004555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006677, upper bound: 0.0006662
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006706, upper bound: 0.0006668
time: 0.64 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007790, 0.0011156, 0.0007802, 0.0011156, -0.0003365, 0.0003354
1: 0.9934273, 0.9942296, 0.9934278, 0.9942271, -0.0007997, 0.0008018
2: -0.0085851, -0.0053618, -0.0085811, -0.0053561, -0.0029581, 0.0029480
3: 0.0036627, 0.0041480, 0.0036641, 0.0041478, -0.0004851, 0.0004839
4: 0.0026546, 0.0052022, 0.0026502, 0.0051990, -0.0025444, 0.0025520
5: 0.0052083, 0.0064374, 0.0052111, 0.0064386, -0.0012303, 0.0012263
6: -0.0021010, -0.0009822, -0.0020997, -0.0009803, -0.0011208, 0.0011175
7: -0.0082577, -0.0075300, -0.0082587, -0.0075311, -0.0007266, 0.0007288
8: 0.0052963, 0.0095314, 0.0052889, 0.0095261, -0.0041574, 0.0041694
9: -0.0036842, -0.0031880, -0.0036842, -0.0031897, -0.0004945, 0.0004962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006770, upper bound: 0.0006787
time: 0.79 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006770, upper bound: 0.0006787
time: 0.85 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007788, 0.0011156, 0.0007804, 0.0011155, -0.0003368, 0.0003352
1: 0.9934273, 0.9942302, 0.9934278, 0.9942266, -0.0007993, 0.0008023
2: -0.0085860, -0.0053614, -0.0085803, -0.0053662, -0.0029505, 0.0029482
3: 0.0036624, 0.0041481, 0.0036644, 0.0041477, -0.0004854, 0.0004837
4: 0.0026544, 0.0052029, 0.0026582, 0.0051984, -0.0025440, 0.0025447
5: 0.0052077, 0.0064375, 0.0052117, 0.0064364, -0.0012287, 0.0012258
6: -0.0021014, -0.0009821, -0.0020994, -0.0009838, -0.0011176, 0.0011173
7: -0.0082577, -0.0075297, -0.0082568, -0.0075314, -0.0007264, 0.0007272
8: 0.0052958, 0.0095326, 0.0053022, 0.0095250, -0.0041568, 0.0041577
9: -0.0036842, -0.0031876, -0.0036841, -0.0031900, -0.0004941, 0.0004965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006813, upper bound: 0.0006790
time: 0.72 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006813, upper bound: 0.0006790
time: 0.73 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007791, 0.0011156, 0.0007643, 0.0011160, -0.0003368, 0.0003513
1: 0.9934275, 0.9942292, 0.9934225, 0.9942607, -0.0008333, 0.0008067
2: -0.0085847, -0.0053803, -0.0086360, -0.0053816, -0.0029415, 0.0029879
3: 0.0036628, 0.0041480, 0.0036443, 0.0041514, -0.0004886, 0.0005038
4: 0.0026694, 0.0052019, 0.0026703, 0.0052424, -0.0025731, 0.0025316
5: 0.0052086, 0.0064334, 0.0051734, 0.0064331, -0.0012245, 0.0012599
6: -0.0021009, -0.0009887, -0.0021187, -0.0009568, -0.0011441, 0.0011301
7: -0.0082542, -0.0075301, -0.0082539, -0.0075150, -0.0007391, 0.0007239
8: 0.0053207, 0.0095309, 0.0053224, 0.0095982, -0.0042045, 0.0041374
9: -0.0036840, -0.0031882, -0.0036840, -0.0031667, -0.0005173, 0.0004958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006922, upper bound: 0.0006919
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006925, upper bound: 0.0006924
time: 0.76 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007842, 0.0011154, 0.0007642, 0.0011160, -0.0003318, 0.0003512
1: 0.9934291, 0.9942185, 0.9934224, 0.9942609, -0.0008318, 0.0007961
2: -0.0085672, -0.0053584, -0.0086362, -0.0053709, -0.0029262, 0.0030050
3: 0.0036692, 0.0041469, 0.0036442, 0.0041514, -0.0004823, 0.0005027
4: 0.0026520, 0.0051881, 0.0026619, 0.0052426, -0.0025906, 0.0025262
5: 0.0052206, 0.0064381, 0.0051733, 0.0064354, -0.0012148, 0.0012648
6: -0.0020948, -0.0009810, -0.0021188, -0.0009567, -0.0011382, 0.0011378
7: -0.0082583, -0.0075352, -0.0082559, -0.0075150, -0.0007434, 0.0007207
8: 0.0052919, 0.0095079, 0.0053084, 0.0095985, -0.0042338, 0.0041273
9: -0.0036842, -0.0031955, -0.0036841, -0.0031666, -0.0005176, 0.0004886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006972
time: 0.75 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006986
time: 0.75 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0008039, 0.0011149, 0.0007643, 0.0011160, -0.0003120, 0.0003507
1: 0.9934356, 0.9941767, 0.9934225, 0.9942607, -0.0008252, 0.0007542
2: -0.0084990, -0.0052543, -0.0086360, -0.0053816, -0.0028606, 0.0031185
3: 0.0036938, 0.0041423, 0.0036443, 0.0041514, -0.0004576, 0.0004981
4: 0.0025697, 0.0051342, 0.0026703, 0.0052424, -0.0026727, 0.0024638
5: 0.0052674, 0.0064606, 0.0051734, 0.0064331, -0.0011657, 0.0012872
6: -0.0020712, -0.0009449, -0.0021187, -0.0009568, -0.0011144, 0.0011738
7: -0.0082780, -0.0075552, -0.0082539, -0.0075150, -0.0007630, 0.0006987
8: 0.0051552, 0.0094183, 0.0053224, 0.0095982, -0.0043723, 0.0040256
9: -0.0036853, -0.0032241, -0.0036840, -0.0031667, -0.0005186, 0.0004599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006714, upper bound: 0.0006684
time: 0.82 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006692
time: 0.66 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008119, 0.0011147, 0.0007642, 0.0011160, -0.0003041, 0.0003505
1: 0.9934381, 0.9941599, 0.9934224, 0.9942609, -0.0008228, 0.0007375
2: -0.0084716, -0.0052474, -0.0086362, -0.0053709, -0.0028452, 0.0031275
3: 0.0037038, 0.0041405, 0.0036442, 0.0041514, -0.0004476, 0.0004964
4: 0.0025643, 0.0051125, 0.0026619, 0.0052426, -0.0026783, 0.0024506
5: 0.0052862, 0.0064621, 0.0051733, 0.0064354, -0.0011492, 0.0012888
6: -0.0020617, -0.0009425, -0.0021188, -0.0009567, -0.0011050, 0.0011763
7: -0.0082793, -0.0075632, -0.0082559, -0.0075150, -0.0007643, 0.0006927
8: 0.0051461, 0.0093823, 0.0053084, 0.0095985, -0.0043822, 0.0040056
9: -0.0036853, -0.0032356, -0.0036841, -0.0031666, -0.0005187, 0.0004485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A1_B2_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006792, upper bound: 0.0006793
time: 0.76 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006848, upper bound: 0.0006796
time: 0.78 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007839, 0.0011155, 0.0007689, 0.0011159, -0.0003319, 0.0003466
1: 0.9934291, 0.9942191, 0.9934240, 0.9942509, -0.0008218, 0.0007951
2: -0.0085681, -0.0053665, -0.0086201, -0.0053627, -0.0029397, 0.0029854
3: 0.0036689, 0.0041469, 0.0036500, 0.0041504, -0.0004815, 0.0004969
4: 0.0026584, 0.0051888, 0.0026555, 0.0052299, -0.0025715, 0.0025333
5: 0.0052200, 0.0064364, 0.0051843, 0.0064372, -0.0012172, 0.0012521
6: -0.0020951, -0.0009838, -0.0021132, -0.0009669, -0.0011283, 0.0011294
7: -0.0082568, -0.0075350, -0.0082575, -0.0075197, -0.0007371, 0.0007225
8: 0.0053025, 0.0095091, 0.0052976, 0.0095774, -0.0042016, 0.0041378
9: -0.0036841, -0.0031951, -0.0036842, -0.0031733, -0.0005108, 0.0004890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B1_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006734, upper bound: 0.0006709
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006742, upper bound: 0.0006709
time: 0.73 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0008112, 0.0011147, 0.0007689, 0.0011159, -0.0003047, 0.0003459
1: 0.9934379, 0.9941614, 0.9934240, 0.9942509, -0.0008130, 0.0007374
2: -0.0084740, -0.0052531, -0.0086201, -0.0053627, -0.0028514, 0.0031071
3: 0.0037030, 0.0041407, 0.0036500, 0.0041504, -0.0004474, 0.0004907
4: 0.0025688, 0.0051144, 0.0026555, 0.0052299, -0.0026611, 0.0024589
5: 0.0052846, 0.0064609, 0.0051843, 0.0064372, -0.0011526, 0.0012766
6: -0.0020625, -0.0009445, -0.0021132, -0.0009669, -0.0010956, 0.0011687
7: -0.0082782, -0.0075626, -0.0082575, -0.0075197, -0.0007585, 0.0006949
8: 0.0051536, 0.0093854, 0.0052976, 0.0095774, -0.0043537, 0.0040175
9: -0.0036853, -0.0032346, -0.0036842, -0.0031733, -0.0005120, 0.0004496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B1_A1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006707, upper bound: 0.0006708
time: 0.68 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006742, upper bound: 0.0006710
time: 0.67 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007818, 0.0011155, 0.0007802, 0.0011156, -0.0003338, 0.0003354
1: 0.9934283, 0.9942237, 0.9934278, 0.9942271, -0.0007988, 0.0007960
2: -0.0085756, -0.0053474, -0.0085811, -0.0053508, -0.0029551, 0.0029595
3: 0.0036661, 0.0041474, 0.0036641, 0.0041478, -0.0004817, 0.0004833
4: 0.0026433, 0.0051947, 0.0026460, 0.0051991, -0.0025558, 0.0025487
5: 0.0052149, 0.0064405, 0.0052111, 0.0064398, -0.0012249, 0.0012294
6: -0.0020978, -0.0009772, -0.0020997, -0.0009784, -0.0011193, 0.0011225
7: -0.0082604, -0.0075328, -0.0082597, -0.0075311, -0.0007293, 0.0007270
8: 0.0052774, 0.0095189, 0.0052819, 0.0095262, -0.0041755, 0.0041635
9: -0.0036843, -0.0031920, -0.0036843, -0.0031897, -0.0004946, 0.0004923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006774, upper bound: 0.0006809
time: 0.78 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006774, upper bound: 0.0006808
time: 0.67 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007813, 0.0011155, 0.0007804, 0.0011155, -0.0003342, 0.0003351
1: 0.9934281, 0.9942247, 0.9934278, 0.9942266, -0.0007985, 0.0007969
2: -0.0085772, -0.0053471, -0.0085803, -0.0053610, -0.0029462, 0.0029591
3: 0.0036655, 0.0041475, 0.0036644, 0.0041477, -0.0004822, 0.0004831
4: 0.0026430, 0.0051960, 0.0026541, 0.0051984, -0.0025554, 0.0025419
5: 0.0052137, 0.0064406, 0.0052116, 0.0064376, -0.0012238, 0.0012289
6: -0.0020983, -0.0009771, -0.0020994, -0.0009819, -0.0011164, 0.0011223
7: -0.0082605, -0.0075323, -0.0082578, -0.0075314, -0.0007291, 0.0007255
8: 0.0052770, 0.0095210, 0.0052953, 0.0095250, -0.0041748, 0.0041523
9: -0.0036843, -0.0031913, -0.0036842, -0.0031900, -0.0004943, 0.0004929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006813
time: 0.77 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006812
time: 0.82 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0007811, 0.0011155, 0.0007643, 0.0011160, -0.0003349, 0.0003513
1: 0.9934280, 0.9942250, 0.9934225, 0.9942608, -0.0008329, 0.0008025
2: -0.0085779, -0.0053629, -0.0086360, -0.0053756, -0.0029383, 0.0030030
3: 0.0036653, 0.0041476, 0.0036442, 0.0041514, -0.0004861, 0.0005033
4: 0.0026556, 0.0051965, 0.0026656, 0.0052424, -0.0025869, 0.0025309
5: 0.0052133, 0.0064371, 0.0051734, 0.0064344, -0.0012211, 0.0012637
6: -0.0020985, -0.0009826, -0.0021187, -0.0009568, -0.0011418, 0.0011361
7: -0.0082575, -0.0075321, -0.0082551, -0.0075150, -0.0007424, 0.0007230
8: 0.0052978, 0.0095219, 0.0053145, 0.0095983, -0.0042267, 0.0041356
9: -0.0036842, -0.0031911, -0.0036840, -0.0031667, -0.0005175, 0.0004930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006922, upper bound: 0.0006924
time: 0.67 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006928, upper bound: 0.0006928
time: 0.62 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0007858, 0.0011154, 0.0007642, 0.0011160, -0.0003302, 0.0003512
1: 0.9934295, 0.9942152, 0.9934224, 0.9942608, -0.0008313, 0.0007928
2: -0.0085616, -0.0053440, -0.0086362, -0.0053646, -0.0029245, 0.0030162
3: 0.0036712, 0.0041465, 0.0036441, 0.0041514, -0.0004803, 0.0005023
4: 0.0026406, 0.0051837, 0.0026569, 0.0052426, -0.0026020, 0.0025268
5: 0.0052244, 0.0064412, 0.0051733, 0.0064368, -0.0012124, 0.0012680
6: -0.0020929, -0.0009760, -0.0021188, -0.0009566, -0.0011363, 0.0011428
7: -0.0082610, -0.0075368, -0.0082571, -0.0075149, -0.0007461, 0.0007203
8: 0.0052730, 0.0095006, 0.0053000, 0.0095986, -0.0042520, 0.0041275
9: -0.0036844, -0.0031979, -0.0036841, -0.0031666, -0.0005178, 0.0004863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006972
time: 0.76 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006987
time: 0.72 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0008083, 0.0011148, 0.0007643, 0.0011160, -0.0003077, 0.0003505
1: 0.9934370, 0.9941676, 0.9934225, 0.9942608, -0.0008239, 0.0007451
2: -0.0084840, -0.0052501, -0.0086360, -0.0053756, -0.0028487, 0.0031250
3: 0.0036993, 0.0041414, 0.0036442, 0.0041514, -0.0004521, 0.0004971
4: 0.0025664, 0.0051223, 0.0026656, 0.0052424, -0.0026760, 0.0024567
5: 0.0052777, 0.0064615, 0.0051734, 0.0064344, -0.0011567, 0.0012881
6: -0.0020660, -0.0009434, -0.0021187, -0.0009568, -0.0011092, 0.0011753
7: -0.0082788, -0.0075596, -0.0082551, -0.0075150, -0.0007638, 0.0006954
8: 0.0051496, 0.0093985, 0.0053145, 0.0095983, -0.0043781, 0.0040139
9: -0.0036853, -0.0032304, -0.0036840, -0.0031667, -0.0005186, 0.0004536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006729, upper bound: 0.0006724
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006765, upper bound: 0.0006727
time: 0.73 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008162, 0.0011146, 0.0007642, 0.0011160, -0.0002998, 0.0003504
1: 0.9934396, 0.9941507, 0.9934224, 0.9942608, -0.0008213, 0.0007283
2: -0.0084566, -0.0052433, -0.0086362, -0.0053646, -0.0028336, 0.0031293
3: 0.0037092, 0.0041395, 0.0036441, 0.0041514, -0.0004422, 0.0004954
4: 0.0025610, 0.0051006, 0.0026569, 0.0052426, -0.0026816, 0.0024438
5: 0.0052965, 0.0064630, 0.0051733, 0.0064368, -0.0011403, 0.0012898
6: -0.0020564, -0.0009411, -0.0021188, -0.0009566, -0.0010998, 0.0011777
7: -0.0082801, -0.0075677, -0.0082571, -0.0075149, -0.0007651, 0.0006895
8: 0.0051406, 0.0093625, 0.0053000, 0.0095986, -0.0043872, 0.0039926
9: -0.0036854, -0.0032419, -0.0036841, -0.0031666, -0.0005188, 0.0004423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006797, upper bound: 0.0006817
time: 0.74 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006856, upper bound: 0.0006822
time: 0.64 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007689, 0.0011158, 0.0007809, 0.0011155, -0.0003467, 0.0003349
1: 0.9934240, 0.9942510, 0.9934280, 0.9942254, -0.0008014, 0.0008230
2: -0.0086201, -0.0053685, -0.0085785, -0.0053542, -0.0030006, 0.0029488
3: 0.0036500, 0.0041504, 0.0036650, 0.0041476, -0.0004976, 0.0004853
4: 0.0026600, 0.0052299, 0.0026487, 0.0051970, -0.0025370, 0.0025812
5: 0.0051843, 0.0064359, 0.0052128, 0.0064390, -0.0012547, 0.0012231
6: -0.0021132, -0.0009669, -0.0020988, -0.0009796, -0.0011336, 0.0011319
7: -0.0082564, -0.0075197, -0.0082591, -0.0075319, -0.0007245, 0.0007394
8: 0.0053052, 0.0095774, 0.0052864, 0.0095228, -0.0041459, 0.0042197
9: -0.0036841, -0.0031733, -0.0036843, -0.0031908, -0.0004933, 0.0005109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006533, upper bound: 0.0006659
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006532, upper bound: 0.0006692
time: 0.61 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008008, 0.0011150, 0.0007809, 0.0011155, -0.0003148, 0.0003341
1: 0.9934344, 0.9941835, 0.9934280, 0.9942254, -0.0007910, 0.0007555
2: -0.0085100, -0.0052894, -0.0085785, -0.0053542, -0.0028908, 0.0030326
3: 0.0036899, 0.0041431, 0.0036650, 0.0041476, -0.0004577, 0.0004780
4: 0.0025974, 0.0051428, 0.0026487, 0.0051970, -0.0025996, 0.0024942
5: 0.0052599, 0.0064530, 0.0052128, 0.0064390, -0.0011792, 0.0012402
6: -0.0020750, -0.0009571, -0.0020988, -0.0009796, -0.0010954, 0.0011417
7: -0.0082714, -0.0075520, -0.0082591, -0.0075319, -0.0007395, 0.0007071
8: 0.0052012, 0.0094327, 0.0052864, 0.0095228, -0.0042500, 0.0040751
9: -0.0036849, -0.0032195, -0.0036843, -0.0031908, -0.0004942, 0.0004648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006725
time: 0.66 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006744
time: 0.68 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008123, 0.0011147, 0.0007901, 0.0011153, -0.0003029, 0.0003246
1: 0.9934383, 0.9941589, 0.9934310, 0.9942060, -0.0007676, 0.0007279
2: -0.0084699, -0.0052502, -0.0085466, -0.0053455, -0.0028620, 0.0030335
3: 0.0037044, 0.0041404, 0.0036766, 0.0041455, -0.0004411, 0.0004638
4: 0.0025665, 0.0051112, 0.0026418, 0.0051718, -0.0026054, 0.0024694
5: 0.0052874, 0.0064615, 0.0052347, 0.0064409, -0.0011535, 0.0012268
6: -0.0020611, -0.0009435, -0.0020877, -0.0009766, -0.0010845, 0.0011442
7: -0.0082788, -0.0075637, -0.0082608, -0.0075412, -0.0007375, 0.0006970
8: 0.0051497, 0.0093801, 0.0052749, 0.0094809, -0.0042603, 0.0040333
9: -0.0036853, -0.0032363, -0.0036843, -0.0032041, -0.0004812, 0.0004481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006769
time: 0.83 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006769
time: 0.81 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008120, 0.0011147, 0.0007901, 0.0011153, -0.0003033, 0.0003247
1: 0.9934382, 0.9941598, 0.9934310, 0.9942062, -0.0007680, 0.0007288
2: -0.0084712, -0.0052592, -0.0085470, -0.0053452, -0.0028652, 0.0030254
3: 0.0037039, 0.0041405, 0.0036765, 0.0041455, -0.0004416, 0.0004640
4: 0.0025736, 0.0051122, 0.0026416, 0.0051721, -0.0025985, 0.0024706
5: 0.0052865, 0.0064596, 0.0052345, 0.0064410, -0.0011545, 0.0012251
6: -0.0020615, -0.0009466, -0.0020878, -0.0009765, -0.0010851, 0.0011412
7: -0.0082771, -0.0075634, -0.0082608, -0.0075411, -0.0007359, 0.0006974
8: 0.0051616, 0.0093818, 0.0052746, 0.0094813, -0.0042488, 0.0040351
9: -0.0036852, -0.0032357, -0.0036843, -0.0032040, -0.0004812, 0.0004486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_B2_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006812
time: 0.71 seconds

## Relational analysis of IS_B1_B2_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006847
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007643, 0.0011160, 0.0007778, 0.0011156, -0.0003513, 0.0003382
1: 0.9934225, 0.9942607, 0.9934269, 0.9942322, -0.0008097, 0.0008338
2: -0.0086360, -0.0053816, -0.0085894, -0.0053512, -0.0030183, 0.0029460
3: 0.0036443, 0.0041514, 0.0036611, 0.0041483, -0.0005041, 0.0004903
4: 0.0026703, 0.0052424, 0.0026463, 0.0052056, -0.0025353, 0.0025961
5: 0.0051734, 0.0064331, 0.0052054, 0.0064397, -0.0012662, 0.0012277
6: -0.0021187, -0.0009568, -0.0021026, -0.0009785, -0.0011402, 0.0011458
7: -0.0082539, -0.0075150, -0.0082597, -0.0075287, -0.0007252, 0.0007446
8: 0.0053224, 0.0095982, 0.0052824, 0.0095371, -0.0041432, 0.0042441
9: -0.0036840, -0.0031667, -0.0036843, -0.0031862, -0.0004978, 0.0005176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006532, upper bound: 0.0006683
time: 0.68 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006533, upper bound: 0.0006719
time: 0.61 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007642, 0.0011160, 0.0007851, 0.0011154, -0.0003512, 0.0003309
1: 0.9934224, 0.9942609, 0.9934294, 0.9942166, -0.0007942, 0.0008315
2: -0.0086362, -0.0053709, -0.0085641, -0.0053402, -0.0030305, 0.0029327
3: 0.0036442, 0.0041514, 0.0036703, 0.0041467, -0.0005025, 0.0004811
4: 0.0026619, 0.0052426, 0.0026376, 0.0051856, -0.0025237, 0.0026050
5: 0.0051733, 0.0064354, 0.0052228, 0.0064421, -0.0012688, 0.0012127
6: -0.0021188, -0.0009567, -0.0020938, -0.0009747, -0.0011441, 0.0011371
7: -0.0082559, -0.0075150, -0.0082618, -0.0075361, -0.0007198, 0.0007468
8: 0.0053084, 0.0095985, 0.0052679, 0.0095038, -0.0041246, 0.0042580
9: -0.0036841, -0.0031666, -0.0036844, -0.0031968, -0.0004873, 0.0005178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006791
time: 0.74 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006845
time: 0.69 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007908, 0.0011153, 0.0007848, 0.0011154, -0.0003247, 0.0003305
1: 0.9934313, 0.9942047, 0.9934293, 0.9942173, -0.0007861, 0.0007754
2: -0.0085445, -0.0053427, -0.0085651, -0.0053146, -0.0029718, 0.0029693
3: 0.0036774, 0.0041454, 0.0036699, 0.0041467, -0.0004694, 0.0004755
4: 0.0026396, 0.0051701, 0.0026174, 0.0051864, -0.0025468, 0.0025527
5: 0.0052362, 0.0064415, 0.0052220, 0.0064476, -0.0012114, 0.0012195
6: -0.0020870, -0.0009756, -0.0020941, -0.0009658, -0.0011211, 0.0011185
7: -0.0082613, -0.0075418, -0.0082666, -0.0075358, -0.0007255, 0.0007248
8: 0.0052713, 0.0094781, 0.0052343, 0.0095052, -0.0041632, 0.0041735
9: -0.0036844, -0.0032050, -0.0036847, -0.0031964, -0.0004880, 0.0004796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B1_A2_A2_A1_A1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006726, upper bound: 0.0006793
time: 0.79 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A2_A1_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006726, upper bound: 0.0006795
time: 0.65 seconds

## BFS IS instance: IS_B1_B2_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007960, 0.0011151, 0.0007847, 0.0011154, -0.0003195, 0.0003304
1: 0.9934329, 0.9941937, 0.9934292, 0.9942175, -0.0007845, 0.0007645
2: -0.0085265, -0.0053220, -0.0085653, -0.0053057, -0.0029566, 0.0029857
3: 0.0036839, 0.0041442, 0.0036699, 0.0041467, -0.0004628, 0.0004743
4: 0.0026233, 0.0051559, 0.0026104, 0.0051866, -0.0025633, 0.0025455
5: 0.0052485, 0.0064460, 0.0052219, 0.0064495, -0.0012010, 0.0012241
6: -0.0020807, -0.0009684, -0.0020942, -0.0009628, -0.0011179, 0.0011258
7: -0.0082652, -0.0075471, -0.0082683, -0.0075357, -0.0007295, 0.0007211
8: 0.0052441, 0.0094544, 0.0052227, 0.0095054, -0.0041913, 0.0041593
9: -0.0036846, -0.0032126, -0.0036848, -0.0031963, -0.0004883, 0.0004722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006884
time: 0.70 seconds

## Relational analysis of IS_B1_B2_A1_B1_A2_A2_A2_A2

### Relational analysis result of IS_B1_B2_A1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006895
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007689, 0.0011159, 0.0007829, 0.0011155, -0.0003466, 0.0003330
1: 0.9934240, 0.9942509, 0.9934286, 0.9942215, -0.0007975, 0.0008222
2: -0.0086201, -0.0053627, -0.0085719, -0.0053419, -0.0030131, 0.0029447
3: 0.0036500, 0.0041504, 0.0036675, 0.0041472, -0.0004972, 0.0004829
4: 0.0026555, 0.0052299, 0.0026389, 0.0051918, -0.0025363, 0.0025910
5: 0.0051843, 0.0064372, 0.0052174, 0.0064417, -0.0012574, 0.0012198
6: -0.0021132, -0.0009669, -0.0020965, -0.0009753, -0.0011379, 0.0011296
7: -0.0082575, -0.0075197, -0.0082614, -0.0075338, -0.0007236, 0.0007418
8: 0.0052976, 0.0095774, 0.0052702, 0.0095140, -0.0041452, 0.0042363
9: -0.0036842, -0.0031733, -0.0036844, -0.0031936, -0.0004906, 0.0005110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006690
time: 0.64 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006719
time: 0.68 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007688, 0.0011159, 0.0007906, 0.0011153, -0.0003465, 0.0003253
1: 0.9934239, 0.9942511, 0.9934312, 0.9942051, -0.0007812, 0.0008199
2: -0.0086203, -0.0053512, -0.0085452, -0.0053337, -0.0030221, 0.0029286
3: 0.0036499, 0.0041504, 0.0036771, 0.0041454, -0.0004955, 0.0004732
4: 0.0026463, 0.0052301, 0.0026325, 0.0051707, -0.0025244, 0.0025976
5: 0.0051842, 0.0064397, 0.0052357, 0.0064435, -0.0012593, 0.0012040
6: -0.0021133, -0.0009667, -0.0020872, -0.0009725, -0.0011408, 0.0011205
7: -0.0082597, -0.0075196, -0.0082630, -0.0075416, -0.0007180, 0.0007434
8: 0.0052825, 0.0095777, 0.0052595, 0.0094790, -0.0041237, 0.0042459
9: -0.0036843, -0.0031732, -0.0036845, -0.0032047, -0.0004796, 0.0005112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006773
time: 0.67 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006822
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007965, 0.0011151, 0.0007903, 0.0011153, -0.0003188, 0.0003249
1: 0.9934331, 0.9941925, 0.9934310, 0.9942057, -0.0007727, 0.0007614
2: -0.0085246, -0.0053254, -0.0085462, -0.0053055, -0.0029631, 0.0029609
3: 0.0036846, 0.0041440, 0.0036767, 0.0041455, -0.0004609, 0.0004673
4: 0.0026259, 0.0051544, 0.0026102, 0.0051715, -0.0025455, 0.0025442
5: 0.0052498, 0.0064453, 0.0052350, 0.0064495, -0.0011997, 0.0012103
6: -0.0020800, -0.0009696, -0.0020876, -0.0009627, -0.0011173, 0.0011180
7: -0.0082645, -0.0075477, -0.0082683, -0.0075414, -0.0007232, 0.0007206
8: 0.0052485, 0.0094519, 0.0052225, 0.0094803, -0.0041590, 0.0041582
9: -0.0036845, -0.0032134, -0.0036848, -0.0032043, -0.0004802, 0.0004714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A1_A2_A1_A1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006786
time: 0.64 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A2_A1_A2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006797
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008011, 0.0011150, 0.0007902, 0.0011153, -0.0003142, 0.0003248
1: 0.9934345, 0.9941828, 0.9934310, 0.9942058, -0.0007713, 0.0007518
2: -0.0085089, -0.0053067, -0.0085465, -0.0052978, -0.0029448, 0.0029797
3: 0.0036902, 0.0041430, 0.0036767, 0.0041455, -0.0004552, 0.0004663
4: 0.0026111, 0.0051420, 0.0026041, 0.0051717, -0.0025605, 0.0025379
5: 0.0052606, 0.0064493, 0.0052349, 0.0064512, -0.0011906, 0.0012144
6: -0.0020746, -0.0009631, -0.0020876, -0.0009600, -0.0011146, 0.0011246
7: -0.0082681, -0.0075523, -0.0082698, -0.0075413, -0.0007268, 0.0007174
8: 0.0052239, 0.0094313, 0.0052123, 0.0094806, -0.0041847, 0.0041474
9: -0.0036847, -0.0032199, -0.0036848, -0.0032042, -0.0004805, 0.0004649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006698, upper bound: 0.0006828
time: 0.71 seconds

## Relational analysis of IS_B1_B2_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_B1_B2_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006856
time: 0.82 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007643, 0.0011160, 0.0007798, 0.0011156, -0.0003513, 0.0003362
1: 0.9934225, 0.9942608, 0.9934276, 0.9942279, -0.0008054, 0.0008332
2: -0.0086360, -0.0053756, -0.0085823, -0.0053388, -0.0030307, 0.0029431
3: 0.0036442, 0.0041514, 0.0036637, 0.0041479, -0.0005036, 0.0004877
4: 0.0026656, 0.0052424, 0.0026365, 0.0052000, -0.0025344, 0.0026060
5: 0.0051734, 0.0064344, 0.0052102, 0.0064424, -0.0012690, 0.0012242
6: -0.0021187, -0.0009568, -0.0021001, -0.0009742, -0.0011445, 0.0011433
7: -0.0082551, -0.0075150, -0.0082620, -0.0075308, -0.0007243, 0.0007470
8: 0.0053145, 0.0095983, 0.0052661, 0.0095278, -0.0041423, 0.0042608
9: -0.0036840, -0.0031667, -0.0036844, -0.0031892, -0.0004949, 0.0005177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006702
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006739
time: 0.65 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007642, 0.0011160, 0.0007882, 0.0011153, -0.0003511, 0.0003278
1: 0.9934224, 0.9942608, 0.9934304, 0.9942100, -0.0007876, 0.0008304
2: -0.0086362, -0.0053646, -0.0085533, -0.0053310, -0.0030390, 0.0029256
3: 0.0036441, 0.0041514, 0.0036742, 0.0041459, -0.0005018, 0.0004773
4: 0.0026569, 0.0052426, 0.0026304, 0.0051771, -0.0025203, 0.0026122
5: 0.0051733, 0.0064368, 0.0052301, 0.0064440, -0.0012708, 0.0012067
6: -0.0021188, -0.0009566, -0.0020900, -0.0009715, -0.0011473, 0.0011334
7: -0.0082571, -0.0075149, -0.0082635, -0.0075392, -0.0007179, 0.0007485
8: 0.0053000, 0.0095986, 0.0052559, 0.0094897, -0.0041181, 0.0042700
9: -0.0036841, -0.0031666, -0.0036845, -0.0032013, -0.0004828, 0.0005179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006795
time: 0.73 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006852
time: 0.76 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0007908, 0.0011153, 0.0007879, 0.0011154, -0.0003246, 0.0003274
1: 0.9934313, 0.9942047, 0.9934303, 0.9942107, -0.0007794, 0.0007744
2: -0.0085446, -0.0053354, -0.0085543, -0.0053029, -0.0029835, 0.0029628
3: 0.0036774, 0.0041454, 0.0036738, 0.0041460, -0.0004686, 0.0004715
4: 0.0026338, 0.0051702, 0.0026082, 0.0051779, -0.0025441, 0.0025620
5: 0.0052361, 0.0064431, 0.0052295, 0.0064501, -0.0012140, 0.0012136
6: -0.0020870, -0.0009730, -0.0020904, -0.0009618, -0.0011252, 0.0011173
7: -0.0082627, -0.0075418, -0.0082688, -0.0075390, -0.0007237, 0.0007270
8: 0.0052616, 0.0094781, 0.0052190, 0.0094910, -0.0041580, 0.0041886
9: -0.0036844, -0.0032050, -0.0036848, -0.0032009, -0.0004836, 0.0004798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A2_A2_A1_A1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006740, upper bound: 0.0006799
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A2_A1_A2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006741, upper bound: 0.0006805
time: 0.62 seconds

## BFS IS instance: IS_B1_B2_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0007960, 0.0011151, 0.0007879, 0.0011154, -0.0003194, 0.0003273
1: 0.9934329, 0.9941936, 0.9934303, 0.9942108, -0.0007779, 0.0007632
2: -0.0085265, -0.0053155, -0.0085546, -0.0052952, -0.0029665, 0.0029791
3: 0.0036839, 0.0041442, 0.0036737, 0.0041460, -0.0004621, 0.0004704
4: 0.0026181, 0.0051559, 0.0026021, 0.0051781, -0.0025600, 0.0025539
5: 0.0052485, 0.0064474, 0.0052293, 0.0064518, -0.0012033, 0.0012181
6: -0.0020807, -0.0009661, -0.0020904, -0.0009591, -0.0011216, 0.0011243
7: -0.0082664, -0.0075471, -0.0082703, -0.0075389, -0.0007275, 0.0007231
8: 0.0052355, 0.0094544, 0.0052088, 0.0094913, -0.0041852, 0.0041735
9: -0.0036847, -0.0032126, -0.0036849, -0.0032008, -0.0004838, 0.0004723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006853
time: 0.70 seconds

## Relational analysis of IS_B1_B2_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_B1_B2_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006876
time: 0.72 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007827, 0.0011155, 0.0007977, 0.0011151, -0.0003324, 0.0003178
1: 0.9934285, 0.9942218, 0.9934336, 0.9941899, -0.0007614, 0.0007882
2: -0.0085725, -0.0053110, -0.0085205, -0.0053021, -0.0030040, 0.0029463
3: 0.0036672, 0.0041472, 0.0036861, 0.0041438, -0.0004765, 0.0004612
4: 0.0026146, 0.0051923, 0.0026075, 0.0051512, -0.0025366, 0.0025847
5: 0.0052170, 0.0064484, 0.0052526, 0.0064503, -0.0012333, 0.0011957
6: -0.0020967, -0.0009646, -0.0020786, -0.0009615, -0.0011352, 0.0011141
7: -0.0082673, -0.0075336, -0.0082689, -0.0075489, -0.0007184, 0.0007353
8: 0.0052297, 0.0095149, 0.0052180, 0.0094466, -0.0041462, 0.0042253
9: -0.0036847, -0.0031933, -0.0036848, -0.0032151, -0.0004696, 0.0004915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006615, upper bound: 0.0006518
time: 0.67 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B1_A1_A2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006624, upper bound: 0.0006564
time: 0.62 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007904, 0.0011153, 0.0007976, 0.0011151, -0.0003247, 0.0003176
1: 0.9934311, 0.9942054, 0.9934334, 0.9941901, -0.0007590, 0.0007719
2: -0.0085458, -0.0053022, -0.0085207, -0.0052921, -0.0029845, 0.0029550
3: 0.0036769, 0.0041454, 0.0036860, 0.0041438, -0.0004669, 0.0004594
4: 0.0026075, 0.0051711, 0.0025996, 0.0051513, -0.0025438, 0.0025716
5: 0.0052353, 0.0064503, 0.0052525, 0.0064525, -0.0012172, 0.0011978
6: -0.0020874, -0.0009615, -0.0020787, -0.0009580, -0.0011294, 0.0011172
7: -0.0082689, -0.0075415, -0.0082709, -0.0075488, -0.0007201, 0.0007294
8: 0.0052180, 0.0094797, 0.0052047, 0.0094468, -0.0041574, 0.0042026
9: -0.0036848, -0.0032045, -0.0036849, -0.0032150, -0.0004698, 0.0004804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006666, upper bound: 0.0006751
time: 0.76 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006684, upper bound: 0.0006757
time: 0.64 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0007878, 0.0011154, 0.0007879, 0.0011154, -0.0003276, 0.0003274
1: 0.9934303, 0.9942111, 0.9934303, 0.9942107, -0.0007804, 0.0007808
2: -0.0085549, -0.0052716, -0.0085543, -0.0053573, -0.0029335, 0.0030166
3: 0.0036736, 0.0041460, 0.0036739, 0.0041460, -0.0004724, 0.0004722
4: 0.0025834, 0.0051783, 0.0026512, 0.0051778, -0.0025944, 0.0025272
5: 0.0052291, 0.0064569, 0.0052295, 0.0064384, -0.0012093, 0.0012274
6: -0.0020906, -0.0009509, -0.0020904, -0.0009807, -0.0011099, 0.0011394
7: -0.0082747, -0.0075388, -0.0082585, -0.0075390, -0.0007357, 0.0007197
8: 0.0051779, 0.0094917, 0.0052905, 0.0094909, -0.0042419, 0.0041301
9: -0.0036851, -0.0032007, -0.0036842, -0.0032009, -0.0004842, 0.0004835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006679
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006688
time: 0.65 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0007877, 0.0011154, 0.0007966, 0.0011151, -0.0003274, 0.0003188
1: 0.9934302, 0.9942111, 0.9934331, 0.9941924, -0.0007622, 0.0007780
2: -0.0085551, -0.0052636, -0.0085245, -0.0053470, -0.0029456, 0.0029986
3: 0.0036735, 0.0041461, 0.0036846, 0.0041440, -0.0004705, 0.0004614
4: 0.0025771, 0.0051785, 0.0026430, 0.0051543, -0.0025772, 0.0025355
5: 0.0052289, 0.0064586, 0.0052499, 0.0064406, -0.0012117, 0.0012087
6: -0.0020906, -0.0009481, -0.0020800, -0.0009771, -0.0011136, 0.0011319
7: -0.0082762, -0.0075388, -0.0082605, -0.0075477, -0.0007285, 0.0007217
8: 0.0051674, 0.0094920, 0.0052770, 0.0094518, -0.0042137, 0.0041440
9: -0.0036852, -0.0032006, -0.0036843, -0.0032134, -0.0004718, 0.0004838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006697
time: 0.63 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006757
time: 0.65 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007826, 0.0011155, 0.0008020, 0.0011150, -0.0003323, 0.0003135
1: 0.9934285, 0.9942219, 0.9934349, 0.9941810, -0.0007525, 0.0007870
2: -0.0085725, -0.0053054, -0.0085058, -0.0052943, -0.0030118, 0.0029363
3: 0.0036672, 0.0041472, 0.0036914, 0.0041428, -0.0004756, 0.0004558
4: 0.0026101, 0.0051923, 0.0026014, 0.0051396, -0.0025295, 0.0025909
5: 0.0052170, 0.0064496, 0.0052627, 0.0064520, -0.0012350, 0.0011869
6: -0.0020967, -0.0009626, -0.0020735, -0.0009588, -0.0011379, 0.0011109
7: -0.0082683, -0.0075336, -0.0082704, -0.0075532, -0.0007151, 0.0007368
8: 0.0052222, 0.0095149, 0.0052077, 0.0094273, -0.0041336, 0.0042352
9: -0.0036848, -0.0031933, -0.0036849, -0.0032212, -0.0004635, 0.0004916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006650, upper bound: 0.0006518
time: 0.64 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006659, upper bound: 0.0006564
time: 0.61 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007904, 0.0011153, 0.0008019, 0.0011150, -0.0003246, 0.0003134
1: 0.9934311, 0.9942054, 0.9934348, 0.9941811, -0.0007500, 0.0007705
2: -0.0085459, -0.0052969, -0.0085061, -0.0052855, -0.0029919, 0.0029446
3: 0.0036769, 0.0041454, 0.0036913, 0.0041428, -0.0004659, 0.0004541
4: 0.0026034, 0.0051712, 0.0025944, 0.0051398, -0.0025363, 0.0025768
5: 0.0052352, 0.0064514, 0.0052626, 0.0064539, -0.0012186, 0.0011888
6: -0.0020874, -0.0009597, -0.0020736, -0.0009557, -0.0011317, 0.0011139
7: -0.0082699, -0.0075415, -0.0082721, -0.0075531, -0.0007168, 0.0007306
8: 0.0052111, 0.0094798, 0.0051961, 0.0094276, -0.0041448, 0.0042111
9: -0.0036848, -0.0032045, -0.0036850, -0.0032211, -0.0004637, 0.0004805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006682, upper bound: 0.0006754
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006732, upper bound: 0.0006757
time: 0.65 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0007877, 0.0011154, 0.0007900, 0.0011153, -0.0003276, 0.0003254
1: 0.9934302, 0.9942110, 0.9934310, 0.9942062, -0.0007761, 0.0007800
2: -0.0085550, -0.0052661, -0.0085472, -0.0053450, -0.0029460, 0.0030160
3: 0.0036736, 0.0041461, 0.0036764, 0.0041455, -0.0004719, 0.0004696
4: 0.0025791, 0.0051784, 0.0026414, 0.0051722, -0.0025932, 0.0025370
5: 0.0052290, 0.0064581, 0.0052344, 0.0064410, -0.0012120, 0.0012237
6: -0.0020906, -0.0009490, -0.0020879, -0.0009764, -0.0011142, 0.0011389
7: -0.0082758, -0.0075388, -0.0082608, -0.0075411, -0.0007347, 0.0007221
8: 0.0051706, 0.0094918, 0.0052743, 0.0094815, -0.0042399, 0.0041465
9: -0.0036852, -0.0032006, -0.0036843, -0.0032039, -0.0004812, 0.0004837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006518, upper bound: 0.0006669
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006675
time: 0.62 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0007877, 0.0011154, 0.0007990, 0.0011151, -0.0003274, 0.0003164
1: 0.9934303, 0.9942111, 0.9934339, 0.9941872, -0.0007570, 0.0007772
2: -0.0085552, -0.0052582, -0.0085162, -0.0053379, -0.0029557, 0.0029923
3: 0.0036735, 0.0041461, 0.0036876, 0.0041435, -0.0004700, 0.0004584
4: 0.0025728, 0.0051786, 0.0026358, 0.0051478, -0.0025749, 0.0025428
5: 0.0052288, 0.0064598, 0.0052556, 0.0064426, -0.0012137, 0.0012042
6: -0.0020907, -0.0009463, -0.0020771, -0.0009739, -0.0011167, 0.0011309
7: -0.0082773, -0.0075387, -0.0082622, -0.0075501, -0.0007271, 0.0007235
8: 0.0051603, 0.0094921, 0.0052649, 0.0094409, -0.0042091, 0.0041563
9: -0.0036852, -0.0032005, -0.0036844, -0.0032169, -0.0004683, 0.0004839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_B2_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006756, upper bound: 0.0006698
time: 0.71 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006757
time: 0.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.37 seconds
IS_B1_B1_A1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006868
IS_B1_B1_A1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006820, upper bound: 0.0006868
IS_B1_B1_A1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006927, upper bound: 0.0006853
IS_B1_B1_A1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006938, upper bound: 0.0006874
IS_B1_B1_A1_A1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006914, upper bound: 0.0006806
IS_B1_B1_A1_A1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006918, upper bound: 0.0006812
IS_B1_B1_A1_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006935
IS_B1_B1_A1_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006935
IS_B1_B1_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006881
IS_B1_B1_A1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006820, upper bound: 0.0006885
IS_B1_B1_A1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006928, upper bound: 0.0006853
IS_B1_B1_A1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006940, upper bound: 0.0006889
IS_B1_B1_A1_A1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006914, upper bound: 0.0006810
IS_B1_B1_A1_A1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006920, upper bound: 0.0006820
IS_B1_B1_A1_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006941
IS_B1_B1_A1_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006941
IS_B1_B1_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006659, upper bound: 0.0006533
IS_B1_B1_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006692, upper bound: 0.0006532
IS_B1_B1_A1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006767, upper bound: 0.0006738
IS_B1_B1_A1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006739
IS_B1_B1_A1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006683, upper bound: 0.0006532
IS_B1_B1_A1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006719, upper bound: 0.0006532
IS_B1_B1_A1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006791, upper bound: 0.0006739
IS_B1_B1_A1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006845, upper bound: 0.0006739
IS_B1_B1_A1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006690, upper bound: 0.0006564
IS_B1_B1_A1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006719, upper bound: 0.0006564
IS_B1_B1_A1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006772, upper bound: 0.0006756
IS_B1_B1_A1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006822, upper bound: 0.0006757
IS_B1_B1_A1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006702, upper bound: 0.0006564
IS_B1_B1_A1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006564
IS_B1_B1_A1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006794, upper bound: 0.0006757
IS_B1_B1_A1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006757
IS_B1_B1_A2_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006695, upper bound: 0.0006663
IS_B1_B1_A2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006706, upper bound: 0.0006668
IS_B1_B1_A2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006677, upper bound: 0.0006662
IS_B1_B1_A2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006706, upper bound: 0.0006668
IS_B1_B1_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006770, upper bound: 0.0006787
IS_B1_B1_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006770, upper bound: 0.0006787
IS_B1_B1_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006813, upper bound: 0.0006790
IS_B1_B1_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006813, upper bound: 0.0006790
IS_B1_B1_A2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006922, upper bound: 0.0006919
IS_B1_B1_A2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006925, upper bound: 0.0006924
IS_B1_B1_A2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006972
IS_B1_B1_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006986
IS_B1_B1_A2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006714, upper bound: 0.0006684
IS_B1_B1_A2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006692
IS_B1_B1_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006792, upper bound: 0.0006793
IS_B1_B1_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006848, upper bound: 0.0006796
IS_B1_B1_A2_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006734, upper bound: 0.0006709
IS_B1_B1_A2_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006742, upper bound: 0.0006709
IS_B1_B1_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006707, upper bound: 0.0006708
IS_B1_B1_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006742, upper bound: 0.0006710
IS_B1_B1_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006774, upper bound: 0.0006809
IS_B1_B1_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006774, upper bound: 0.0006808
IS_B1_B1_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006813
IS_B1_B1_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006812
IS_B1_B1_A2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006922, upper bound: 0.0006924
IS_B1_B1_A2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006928, upper bound: 0.0006928
IS_B1_B1_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006972
IS_B1_B1_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006941, upper bound: 0.0006987
IS_B1_B1_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006729, upper bound: 0.0006724
IS_B1_B1_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006765, upper bound: 0.0006727
IS_B1_B1_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006797, upper bound: 0.0006817
IS_B1_B1_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006856, upper bound: 0.0006822
IS_B1_B2_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006533, upper bound: 0.0006659
IS_B1_B2_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006532, upper bound: 0.0006692
IS_B1_B2_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006725
IS_B1_B2_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006744
IS_B1_B2_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006769
IS_B1_B2_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006769
IS_B1_B2_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006812
IS_B1_B2_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006847
IS_B1_B2_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006532, upper bound: 0.0006683
IS_B1_B2_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006533, upper bound: 0.0006719
IS_B1_B2_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006791
IS_B1_B2_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006845
IS_B1_B2_A1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006726, upper bound: 0.0006793
IS_B1_B2_A1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006726, upper bound: 0.0006795
IS_B1_B2_A1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006884
IS_B1_B2_A1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006895
IS_B1_B2_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006690
IS_B1_B2_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006719
IS_B1_B2_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006773
IS_B1_B2_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006822
IS_B1_B2_A1_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006786
IS_B1_B2_A1_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006797
IS_B1_B2_A1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006698, upper bound: 0.0006828
IS_B1_B2_A1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006856
IS_B1_B2_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006702
IS_B1_B2_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006739
IS_B1_B2_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006795
IS_B1_B2_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006852
IS_B1_B2_A1_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006740, upper bound: 0.0006799
IS_B1_B2_A1_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006741, upper bound: 0.0006805
IS_B1_B2_A1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006853
IS_B1_B2_A1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006876
IS_B1_B2_A2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006615, upper bound: 0.0006518
IS_B1_B2_A2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006624, upper bound: 0.0006564
IS_B1_B2_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006666, upper bound: 0.0006751
IS_B1_B2_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006684, upper bound: 0.0006757
IS_B1_B2_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006679
IS_B1_B2_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006559, upper bound: 0.0006688
IS_B1_B2_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006697
IS_B1_B2_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006739, upper bound: 0.0006757
IS_B1_B2_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006650, upper bound: 0.0006518
IS_B1_B2_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006659, upper bound: 0.0006564
IS_B1_B2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006682, upper bound: 0.0006754
IS_B1_B2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006732, upper bound: 0.0006757
IS_B1_B2_A2_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006518, upper bound: 0.0006669
IS_B1_B2_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006564, upper bound: 0.0006675
IS_B1_B2_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006756, upper bound: 0.0006698
IS_B1_B2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 1, lower bound: -0.0006757, upper bound: 0.0006757
IS_B1_B2_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006677, upper bound: 0.0006679
IS_B1_B2_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006807
IS_B1_B2_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006700, upper bound: 0.0006702
IS_B1_B2_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006814
IS_B1_B2_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006708, upper bound: 0.0006721
IS_B1_B2_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006835
IS_B1_B2_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006720, upper bound: 0.0006738
IS_B1_B2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006782, upper bound: 0.0006841
IS_B2_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006731
IS_B2_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006697, upper bound: 0.0006731
IS_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006787, upper bound: 0.0006770
IS_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006790, upper bound: 0.0006813
IS_B2_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006718, upper bound: 0.0006764
IS_B2_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006871
IS_B2_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006718, upper bound: 0.0006764
IS_B2_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006825, upper bound: 0.0006871
IS_B2_A1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
IS_B2_A1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006738, upper bound: 0.0006768
IS_B2_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006808, upper bound: 0.0006775
IS_B2_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006825
IS_B2_A1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006789
IS_B2_A1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006879
IS_B2_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006755, upper bound: 0.0006789
IS_B2_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006850, upper bound: 0.0006879
IS_B2_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006810, upper bound: 0.0006747
IS_B2_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006763
IS_B2_A1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006821, upper bound: 0.0006753
IS_B2_A1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006895, upper bound: 0.0006763
IS_B2_A1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006807, upper bound: 0.0006713
IS_B2_A1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006763
IS_B2_A1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006807, upper bound: 0.0006713
IS_B2_A1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006763
IS_B2_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006718
IS_B2_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006724, upper bound: 0.0006716
IS_B2_A1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006786, upper bound: 0.0006698
IS_B2_A1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006799, upper bound: 0.0006757
IS_B2_A1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006720
IS_B2_A1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006737, upper bound: 0.0006720
IS_B2_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006801, upper bound: 0.0006698
IS_B2_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006812, upper bound: 0.0006757
IS_B2_A2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007009, upper bound: 0.0007010
IS_B2_A2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007010
IS_B2_A2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007002
IS_B2_A2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
IS_B2_A2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007027, upper bound: 0.0007000
IS_B2_A2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007029, upper bound: 0.0007012
IS_B2_A2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007009
IS_B2_A2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0007030, upper bound: 0.0007030
IS_B2_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006876
IS_B2_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006926
IS_B2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006883
IS_B2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006938
IS_B2_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006886
IS_B2_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006934
IS_B2_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006893
IS_B2_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006944
IS_B2_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006876, upper bound: 0.0006752
IS_B2_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006925, upper bound: 0.0006840
IS_B2_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006884, upper bound: 0.0006752
IS_B2_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006938, upper bound: 0.0006840
IS_B2_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006766
IS_B2_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006934, upper bound: 0.0006856
IS_B2_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006894, upper bound: 0.0006766
IS_B2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006944, upper bound: 0.0006856
IS_B2_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006815, upper bound: 0.0006766
IS_B2_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006839, upper bound: 0.0006856
IS_B2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006752, upper bound: 0.0006828
IS_B2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006841, upper bound: 0.0006856
IS_B2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006763, upper bound: 0.0006841
IS_B2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006852, upper bound: 0.0006855
IS_B2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006766, upper bound: 0.0006842
IS_B2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 1, lower bound: -0.0006855, upper bound: 0.0006855

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.30 + 597.53 = 600.83 seconds
