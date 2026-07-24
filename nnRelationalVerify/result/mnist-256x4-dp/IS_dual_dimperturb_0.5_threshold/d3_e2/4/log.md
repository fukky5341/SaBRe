## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0051876


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042366, -0.0037133, -0.0042366, -0.0037133, -0.0004426, 0.0004426)
1: (0.0003087, 0.0032058, 0.0003087, 0.0032058, -0.0024509, 0.0024509)
2: (0.0078040, 0.0142764, 0.0078040, 0.0142764, -0.0054756, 0.0054756)
3: (0.0013183, 0.0040457, 0.0013183, 0.0040457, -0.0023074, 0.0023074)
4: (1.0018647, 1.0124462, 1.0018647, 1.0124462, -0.0089519, 0.0089519)
5: (0.0026041, 0.0046627, 0.0026041, 0.0046627, -0.0017415, 0.0017415)
6: (-0.0118107, -0.0091319, -0.0118107, -0.0091319, -0.0022663, 0.0022663)
7: (-0.0103099, -0.0099682, -0.0103099, -0.0099682, -0.0002891, 0.0002891)
8: (-0.0045447, -0.0026938, -0.0045447, -0.0026938, -0.0015658, 0.0015658)
9: (-0.0046850, 0.0045810, -0.0046850, 0.0045810, -0.0078390, 0.0078390)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 1.65 = 3.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0061640, upper bound: 0.0061640

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 129

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058802, upper bound: 0.0055627
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058802, upper bound: 0.0058802
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -0.0058802, upper bound: 0.0055627
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -0.0058802, upper bound: 0.0058802

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041976, -0.0037260, -0.0042315, -0.0037135, -0.0003937, 0.0004249
1: 0.0003789, 0.0029902, 0.0003098, 0.0031778, -0.0023526, 0.0021800
2: 0.0082856, 0.0141196, 0.0078667, 0.0142740, -0.0048703, 0.0052560
3: 0.0013843, 0.0038428, 0.0013193, 0.0040193, -0.0022149, 0.0020523
4: 1.0021211, 1.0116588, 1.0018686, 1.0123438, -0.0085929, 0.0079623
5: 0.0026540, 0.0045095, 0.0026049, 0.0046427, -0.0016717, 0.0015490
6: -0.0116114, -0.0091968, -0.0117848, -0.0091329, -0.0020158, 0.0021754
7: -0.0102845, -0.0099765, -0.0103066, -0.0099683, -0.0002571, 0.0002775
8: -0.0044999, -0.0028316, -0.0045440, -0.0027118, -0.0015030, 0.0013927
9: -0.0039954, 0.0043565, -0.0045952, 0.0045776, -0.0069724, 0.0075246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055499, upper bound: 0.0048760
time: 0.78 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055044, upper bound: 0.0051851
time: 0.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0042198, -0.0037145, -0.0042363, -0.0037134, -0.0004131, 0.0004412
1: 0.0003154, 0.0031131, 0.0003088, 0.0032042, -0.0024430, 0.0022872
2: 0.0080112, 0.0142616, 0.0078076, 0.0142761, -0.0051098, 0.0054579
3: 0.0013245, 0.0039584, 0.0013183, 0.0040442, -0.0023000, 0.0021533
4: 1.0018888, 1.0121075, 1.0018649, 1.0124403, -0.0089230, 0.0083539
5: 0.0026088, 0.0045968, 0.0026042, 0.0046615, -0.0017359, 0.0016252
6: -0.0117250, -0.0091380, -0.0118093, -0.0091320, -0.0021149, 0.0022590
7: -0.0102990, -0.0099690, -0.0103097, -0.0099682, -0.0002698, 0.0002882
8: -0.0045405, -0.0027531, -0.0045447, -0.0026949, -0.0015608, 0.0014612
9: -0.0043884, 0.0045599, -0.0046798, 0.0045807, -0.0073153, 0.0078136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055627, upper bound: 0.0058802
time: 0.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055627, upper bound: 0.0058802
time: 0.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.43 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 4, lower bound: -0.0055499, upper bound: 0.0048760
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 4, lower bound: -0.0055044, upper bound: 0.0051851
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 4, lower bound: -0.0055627, upper bound: 0.0058802
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 4, lower bound: -0.0055627, upper bound: 0.0058802

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041762, -0.0037275, -0.0042315, -0.0037135, -0.0003720, 0.0004239
1: 0.0003869, 0.0028715, 0.0003098, 0.0031778, -0.0023471, 0.0020600
2: 0.0085509, 0.0141017, 0.0078667, 0.0142740, -0.0046023, 0.0052437
3: 0.0013919, 0.0037310, 0.0013193, 0.0040193, -0.0022097, 0.0019394
4: 1.0021502, 1.0112250, 1.0018686, 1.0123438, -0.0085728, 0.0075242
5: 0.0026597, 0.0044251, 0.0026049, 0.0046427, -0.0016677, 0.0014638
6: -0.0115016, -0.0092042, -0.0117848, -0.0091329, -0.0019049, 0.0021703
7: -0.0102705, -0.0099774, -0.0103066, -0.0099683, -0.0002430, 0.0002768
8: -0.0044948, -0.0029074, -0.0045440, -0.0027118, -0.0014995, 0.0013161
9: -0.0036156, 0.0043309, -0.0045952, 0.0045776, -0.0065888, 0.0075070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052276, upper bound: 0.0048760
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052276, upper bound: 0.0048760
time: 1.08 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041672, -0.0036583, -0.0042259, -0.0037141, -0.0003892, 0.0004947
1: 0.0000040, 0.0028216, 0.0003129, 0.0031467, -0.0027392, 0.0021547
2: 0.0086623, 0.0149571, 0.0079360, 0.0142670, -0.0048139, 0.0061197
3: 0.0010314, 0.0036840, 0.0013222, 0.0039901, -0.0025789, 0.0020286
4: 1.0007516, 1.0110428, 1.0018799, 1.0122303, -0.0100050, 0.0078701
5: 0.0023876, 0.0043897, 0.0026071, 0.0046207, -0.0019464, 0.0015311
6: -0.0114555, -0.0088501, -0.0117561, -0.0091357, -0.0019924, 0.0025329
7: -0.0102646, -0.0099323, -0.0103030, -0.0099687, -0.0002542, 0.0003231
8: -0.0047394, -0.0029393, -0.0045420, -0.0027316, -0.0017500, 0.0013766
9: -0.0034561, 0.0055556, -0.0044959, 0.0045677, -0.0068917, 0.0087612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051947, upper bound: 0.0051851
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051947, upper bound: 0.0051851
time: 0.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042198, -0.0037145, -0.0041976, -0.0037260, -0.0004136, 0.0003929
1: 0.0003154, 0.0031131, 0.0003789, 0.0029902, -0.0021757, 0.0022900
2: 0.0080112, 0.0142616, 0.0082856, 0.0141196, -0.0051161, 0.0048608
3: 0.0013245, 0.0039584, 0.0013843, 0.0038428, -0.0020484, 0.0021559
4: 1.0018888, 1.0121075, 1.0021211, 1.0116588, -0.0079468, 0.0083641
5: 0.0026088, 0.0045968, 0.0026540, 0.0045095, -0.0015460, 0.0016272
6: -0.0117250, -0.0091380, -0.0116114, -0.0091968, -0.0021175, 0.0020119
7: -0.0102990, -0.0099690, -0.0102845, -0.0099765, -0.0002701, 0.0002566
8: -0.0045405, -0.0027531, -0.0044999, -0.0028316, -0.0013900, 0.0014630
9: -0.0043884, 0.0045599, -0.0039954, 0.0043565, -0.0073243, 0.0069589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052464, upper bound: 0.0052276
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051851, upper bound: 0.0055044
time: 0.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042198, -0.0037145, -0.0042198, -0.0037145, -0.0004120, 0.0004120
1: 0.0003154, 0.0031131, 0.0003154, 0.0031131, -0.0022811, 0.0022811
2: 0.0080112, 0.0142616, 0.0080112, 0.0142616, -0.0050963, 0.0050963
3: 0.0013245, 0.0039584, 0.0013245, 0.0039584, -0.0021476, 0.0021476
4: 1.0018888, 1.0121075, 1.0018888, 1.0121075, -0.0083318, 0.0083318
5: 0.0026088, 0.0045968, 0.0026088, 0.0045968, -0.0016209, 0.0016209
6: -0.0117250, -0.0091380, -0.0117250, -0.0091380, -0.0021093, 0.0021093
7: -0.0102990, -0.0099690, -0.0102990, -0.0099690, -0.0002691, 0.0002691
8: -0.0045405, -0.0027531, -0.0045405, -0.0027531, -0.0014574, 0.0014574
9: -0.0043884, 0.0045599, -0.0043884, 0.0045599, -0.0072959, 0.0072959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052464, upper bound: 0.0052276
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051851, upper bound: 0.0055044
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.48 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0052276, upper bound: 0.0048760
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0052276, upper bound: 0.0048760
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0051947, upper bound: 0.0051851
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0051947, upper bound: 0.0051851
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0052464, upper bound: 0.0052276
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0051851, upper bound: 0.0055044
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0052464, upper bound: 0.0052276
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 4, lower bound: -0.0051851, upper bound: 0.0055044

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041762, -0.0037275, -0.0042105, -0.0037150, -0.0003710, 0.0004036
1: 0.0003869, 0.0028715, 0.0003177, 0.0030613, -0.0022346, 0.0020542
2: 0.0085509, 0.0141017, 0.0081269, 0.0142563, -0.0045892, 0.0049925
3: 0.0013919, 0.0037310, 0.0013267, 0.0039097, -0.0021038, 0.0019339
4: 1.0021502, 1.0112250, 1.0018973, 1.0119183, -0.0081621, 0.0075028
5: 0.0026597, 0.0044251, 0.0026105, 0.0045600, -0.0015878, 0.0014596
6: -0.0115016, -0.0092042, -0.0116771, -0.0091402, -0.0018995, 0.0020663
7: -0.0102705, -0.0099774, -0.0102929, -0.0099693, -0.0002423, 0.0002636
8: -0.0044948, -0.0029074, -0.0045390, -0.0027862, -0.0014277, 0.0013124
9: -0.0036156, 0.0043309, -0.0042227, 0.0045524, -0.0065700, 0.0071473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041762, -0.0037275, -0.0042002, -0.0036504, -0.0004389, 0.0004010
1: 0.0003869, 0.0028715, -0.0000395, 0.0030043, -0.0022203, 0.0024299
2: 0.0085509, 0.0141017, 0.0082543, 0.0150545, -0.0054287, 0.0049605
3: 0.0013919, 0.0037310, 0.0009904, 0.0038560, -0.0020904, 0.0022877
4: 1.0021502, 1.0112250, 1.0005925, 1.0117099, -0.0081098, 0.0088752
5: 0.0026597, 0.0044251, 0.0023567, 0.0045195, -0.0015777, 0.0017266
6: -0.0115016, -0.0092042, -0.0116244, -0.0088098, -0.0022469, 0.0020531
7: -0.0102705, -0.0099774, -0.0102862, -0.0099271, -0.0002866, 0.0002619
8: -0.0044948, -0.0029074, -0.0047672, -0.0028226, -0.0014185, 0.0015524
9: -0.0036156, 0.0043309, -0.0040403, 0.0056950, -0.0077718, 0.0071016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
time: 1.06 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041672, -0.0036583, -0.0041922, -0.0037266, -0.0003749, 0.0004506
1: 0.0000040, 0.0028216, 0.0003820, 0.0029599, -0.0024950, 0.0020756
2: 0.0086623, 0.0149571, 0.0083533, 0.0141126, -0.0046370, 0.0055741
3: 0.0010314, 0.0036840, 0.0013873, 0.0038142, -0.0023489, 0.0019540
4: 1.0007516, 1.0110428, 1.0021324, 1.0115480, -0.0091129, 0.0075810
5: 0.0023876, 0.0043897, 0.0026562, 0.0044880, -0.0017728, 0.0014748
6: -0.0114555, -0.0088501, -0.0115834, -0.0091996, -0.0019192, 0.0023071
7: -0.0102646, -0.0099323, -0.0102809, -0.0099769, -0.0002448, 0.0002943
8: -0.0047394, -0.0029393, -0.0044979, -0.0028509, -0.0015940, 0.0013260
9: -0.0034561, 0.0055556, -0.0038985, 0.0043466, -0.0066385, 0.0079800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041672, -0.0036583, -0.0042140, -0.0037151, -0.0003884, 0.0004816
1: 0.0000040, 0.0028216, 0.0003186, 0.0030811, -0.0026665, 0.0021505
2: 0.0086623, 0.0149571, 0.0080826, 0.0142545, -0.0048044, 0.0059574
3: 0.0010314, 0.0036840, 0.0013275, 0.0039283, -0.0025104, 0.0020246
4: 1.0007516, 1.0110428, 1.0019004, 1.0119907, -0.0097396, 0.0078547
5: 0.0023876, 0.0043897, 0.0026111, 0.0045741, -0.0018947, 0.0015281
6: -0.0114555, -0.0088501, -0.0116954, -0.0091409, -0.0019885, 0.0024657
7: -0.0102646, -0.0099323, -0.0102952, -0.0099694, -0.0002537, 0.0003145
8: -0.0047394, -0.0029393, -0.0045384, -0.0027735, -0.0017036, 0.0013739
9: -0.0034561, 0.0055556, -0.0042861, 0.0045497, -0.0068782, 0.0085287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041993, -0.0037160, -0.0041976, -0.0037260, -0.0003944, 0.0003919
1: 0.0003234, 0.0029993, 0.0003789, 0.0029902, -0.0021699, 0.0021838
2: 0.0082653, 0.0142437, 0.0082856, 0.0141196, -0.0048787, 0.0048478
3: 0.0013320, 0.0038513, 0.0013843, 0.0038428, -0.0020429, 0.0020559
4: 1.0019180, 1.0116919, 1.0021211, 1.0116588, -0.0079255, 0.0079762
5: 0.0026145, 0.0045160, 0.0026540, 0.0045095, -0.0015418, 0.0015517
6: -0.0116198, -0.0091454, -0.0116114, -0.0091968, -0.0020193, 0.0020065
7: -0.0102856, -0.0099699, -0.0102845, -0.0099765, -0.0002576, 0.0002559
8: -0.0045354, -0.0028258, -0.0044999, -0.0028316, -0.0013863, 0.0013952
9: -0.0040245, 0.0045343, -0.0039954, 0.0043565, -0.0069845, 0.0069402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0052276
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0052276
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041870, -0.0036508, -0.0041922, -0.0037266, -0.0004087, 0.0004569
1: -0.0000373, 0.0029312, 0.0003820, 0.0029599, -0.0025297, 0.0022632
2: 0.0084175, 0.0150496, 0.0083533, 0.0141126, -0.0050562, 0.0056515
3: 0.0009924, 0.0037872, 0.0013873, 0.0038142, -0.0023816, 0.0021307
4: 1.0006006, 1.0114433, 1.0021324, 1.0115480, -0.0092396, 0.0082663
5: 0.0023582, 0.0044676, 0.0026562, 0.0044880, -0.0017975, 0.0016081
6: -0.0115568, -0.0088119, -0.0115834, -0.0091996, -0.0020927, 0.0023391
7: -0.0102775, -0.0099274, -0.0102809, -0.0099769, -0.0002669, 0.0002984
8: -0.0047658, -0.0028693, -0.0044979, -0.0028509, -0.0016161, 0.0014459
9: -0.0038066, 0.0056880, -0.0038985, 0.0043466, -0.0072386, 0.0080909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0055044
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0055044
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041993, -0.0037160, -0.0042198, -0.0037145, -0.0003905, 0.0004110
1: 0.0003234, 0.0029993, 0.0003154, 0.0031131, -0.0022757, 0.0021622
2: 0.0082653, 0.0142437, 0.0080112, 0.0142616, -0.0048306, 0.0050841
3: 0.0013320, 0.0038513, 0.0013245, 0.0039584, -0.0021425, 0.0020356
4: 1.0019180, 1.0116919, 1.0018888, 1.0121075, -0.0083120, 0.0078975
5: 0.0026145, 0.0045160, 0.0026088, 0.0045968, -0.0016170, 0.0015364
6: -0.0116198, -0.0091454, -0.0117250, -0.0091380, -0.0019994, 0.0021043
7: -0.0102856, -0.0099699, -0.0102990, -0.0099690, -0.0002550, 0.0002684
8: -0.0045354, -0.0028258, -0.0045405, -0.0027531, -0.0014539, 0.0013814
9: -0.0040245, 0.0045343, -0.0043884, 0.0045599, -0.0069156, 0.0072786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0052276
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0052276
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041870, -0.0036508, -0.0042140, -0.0037151, -0.0004034, 0.0004830
1: -0.0000373, 0.0029312, 0.0003186, 0.0030811, -0.0026742, 0.0022337
2: 0.0084175, 0.0150496, 0.0080826, 0.0142545, -0.0049903, 0.0059745
3: 0.0009924, 0.0037872, 0.0013275, 0.0039283, -0.0025177, 0.0021029
4: 1.0006006, 1.0114433, 1.0019004, 1.0119907, -0.0097677, 0.0081586
5: 0.0023582, 0.0044676, 0.0026111, 0.0045741, -0.0019002, 0.0015872
6: -0.0115568, -0.0088119, -0.0116954, -0.0091409, -0.0020655, 0.0024728
7: -0.0102775, -0.0099274, -0.0102952, -0.0099694, -0.0002635, 0.0003154
8: -0.0047658, -0.0028693, -0.0045384, -0.0027735, -0.0017085, 0.0014271
9: -0.0038066, 0.0056880, -0.0042861, 0.0045497, -0.0071442, 0.0085533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0055044
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0055044
time: 1.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.75 seconds
IS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
IS_A1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
IS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
IS_A1_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0050694, upper bound: 0.0048760
IS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
IS_A1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048996, upper bound: 0.0051851
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0052276
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0052276
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0055044
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048760, upper bound: 0.0055044
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0052276
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0052276
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0055044
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -0.0048873, upper bound: 0.0055044

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041993, -0.0037160, -0.0041762, -0.0037275, -0.0003934, 0.0003702
1: 0.0003234, 0.0029993, 0.0003869, 0.0028715, -0.0020499, 0.0021782
2: 0.0082653, 0.0142437, 0.0085509, 0.0141017, -0.0048664, 0.0045798
3: 0.0013320, 0.0038513, 0.0013919, 0.0037310, -0.0019299, 0.0020507
4: 1.0019180, 1.0116919, 1.0021502, 1.0112250, -0.0074874, 0.0079560
5: 0.0026145, 0.0045160, 0.0026597, 0.0044251, -0.0014566, 0.0015478
6: -0.0116198, -0.0091454, -0.0115016, -0.0092042, -0.0020142, 0.0018956
7: -0.0102856, -0.0099699, -0.0102705, -0.0099774, -0.0002569, 0.0002418
8: -0.0045354, -0.0028258, -0.0044948, -0.0029074, -0.0013097, 0.0013916
9: -0.0040245, 0.0045343, -0.0036156, 0.0043309, -0.0069669, 0.0065566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0037857, upper bound: 0.0043515
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0032716, upper bound: 0.0031526
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041993, -0.0037160, -0.0041672, -0.0036583, -0.0004664, 0.0003703
1: 0.0003234, 0.0029993, 0.0000040, 0.0028216, -0.0020501, 0.0025826
2: 0.0082653, 0.0142437, 0.0086623, 0.0149571, -0.0057699, 0.0045801
3: 0.0013320, 0.0038513, 0.0010314, 0.0036840, -0.0019301, 0.0024315
4: 1.0019180, 1.0116919, 1.0007516, 1.0110428, -0.0074879, 0.0094331
5: 0.0026145, 0.0045160, 0.0023876, 0.0043897, -0.0014567, 0.0018351
6: -0.0116198, -0.0091454, -0.0114555, -0.0088501, -0.0023881, 0.0018957
7: -0.0102856, -0.0099699, -0.0102646, -0.0099323, -0.0003046, 0.0002418
8: -0.0045354, -0.0028258, -0.0047394, -0.0029393, -0.0013098, 0.0016500
9: -0.0040245, 0.0045343, -0.0034561, 0.0055556, -0.0082604, 0.0065570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041906, upper bound: 0.0034097
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0032716, upper bound: 0.0031526
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041870, -0.0036508, -0.0041762, -0.0037275, -0.0003869, 0.0004384
1: -0.0000373, 0.0029312, 0.0003869, 0.0028715, -0.0024277, 0.0021420
2: 0.0084175, 0.0150496, 0.0085509, 0.0141017, -0.0047855, 0.0054237
3: 0.0009924, 0.0037872, 0.0013919, 0.0037310, -0.0022856, 0.0020166
4: 1.0006006, 1.0114433, 1.0021502, 1.0112250, -0.0088671, 0.0078238
5: 0.0023582, 0.0044676, 0.0026597, 0.0044251, -0.0017250, 0.0015220
6: -0.0115568, -0.0088119, -0.0115016, -0.0092042, -0.0019807, 0.0022448
7: -0.0102775, -0.0099274, -0.0102705, -0.0099774, -0.0002527, 0.0002863
8: -0.0047658, -0.0028693, -0.0044948, -0.0029074, -0.0015510, 0.0013685
9: -0.0038066, 0.0056880, -0.0036156, 0.0043309, -0.0068511, 0.0077647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0030624, upper bound: 0.0042235
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0025509, upper bound: 0.0029338
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041870, -0.0036508, -0.0041672, -0.0036583, -0.0004102, 0.0003894
1: -0.0000373, 0.0029312, 0.0000040, 0.0028216, -0.0021560, 0.0022713
2: 0.0084175, 0.0150496, 0.0086623, 0.0149571, -0.0050743, 0.0048167
3: 0.0009924, 0.0037872, 0.0010314, 0.0036840, -0.0020298, 0.0021383
4: 1.0006006, 1.0114433, 1.0007516, 1.0110428, -0.0078747, 0.0082959
5: 0.0023582, 0.0044676, 0.0023876, 0.0043897, -0.0015319, 0.0016139
6: -0.0115568, -0.0088119, -0.0114555, -0.0088501, -0.0021002, 0.0019936
7: -0.0102775, -0.0099274, -0.0102646, -0.0099323, -0.0002679, 0.0002543
8: -0.0047658, -0.0028693, -0.0047394, -0.0029393, -0.0013774, 0.0014511
9: -0.0038066, 0.0056880, -0.0034561, 0.0055556, -0.0072645, 0.0068957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0030624, upper bound: 0.0042235
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0025509, upper bound: 0.0029338
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041993, -0.0037160, -0.0041993, -0.0037160, -0.0003895, 0.0003895
1: 0.0003234, 0.0029993, 0.0003234, 0.0029993, -0.0021568, 0.0021568
2: 0.0082653, 0.0142437, 0.0082653, 0.0142437, -0.0048185, 0.0048185
3: 0.0013320, 0.0038513, 0.0013320, 0.0038513, -0.0020305, 0.0020305
4: 1.0019180, 1.0116919, 1.0019180, 1.0116919, -0.0078777, 0.0078777
5: 0.0026145, 0.0045160, 0.0026145, 0.0045160, -0.0015325, 0.0015325
6: -0.0116198, -0.0091454, -0.0116198, -0.0091454, -0.0019943, 0.0019943
7: -0.0102856, -0.0099699, -0.0102856, -0.0099699, -0.0002544, 0.0002544
8: -0.0045354, -0.0028258, -0.0045354, -0.0028258, -0.0013779, 0.0013779
9: -0.0040245, 0.0045343, -0.0040245, 0.0045343, -0.0068983, 0.0068983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043406, upper bound: 0.0034677
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0034614, upper bound: 0.0032283
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041993, -0.0037160, -0.0041870, -0.0036508, -0.0004647, 0.0003890
1: 0.0003234, 0.0029993, -0.0000373, 0.0029312, -0.0021540, 0.0025728
2: 0.0082653, 0.0142437, 0.0084175, 0.0150496, -0.0057480, 0.0048123
3: 0.0013320, 0.0038513, 0.0009924, 0.0037872, -0.0020279, 0.0024222
4: 1.0019180, 1.0116919, 1.0006006, 1.0114433, -0.0078675, 0.0093973
5: 0.0026145, 0.0045160, 0.0023582, 0.0044676, -0.0015305, 0.0018281
6: -0.0116198, -0.0091454, -0.0115568, -0.0088119, -0.0023791, 0.0019918
7: -0.0102856, -0.0099699, -0.0102775, -0.0099274, -0.0003035, 0.0002541
8: -0.0045354, -0.0028258, -0.0047658, -0.0028693, -0.0013761, 0.0016437
9: -0.0040245, 0.0045343, -0.0038066, 0.0056880, -0.0082290, 0.0068893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043406, upper bound: 0.0034677
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0034614, upper bound: 0.0032283
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041870, -0.0036508, -0.0041993, -0.0037160, -0.0003890, 0.0004647
1: -0.0000373, 0.0029312, 0.0003234, 0.0029993, -0.0025728, 0.0021540
2: 0.0084175, 0.0150496, 0.0082653, 0.0142437, -0.0048123, 0.0057480
3: 0.0009924, 0.0037872, 0.0013320, 0.0038513, -0.0024222, 0.0020279
4: 1.0006006, 1.0114433, 1.0019180, 1.0116919, -0.0093973, 0.0078675
5: 0.0023582, 0.0044676, 0.0026145, 0.0045160, -0.0018281, 0.0015305
6: -0.0115568, -0.0088119, -0.0116198, -0.0091454, -0.0019918, 0.0023791
7: -0.0102775, -0.0099274, -0.0102856, -0.0099699, -0.0002541, 0.0003035
8: -0.0047658, -0.0028693, -0.0045354, -0.0028258, -0.0016437, 0.0013761
9: -0.0038066, 0.0056880, -0.0040245, 0.0045343, -0.0068893, 0.0082290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0032156, upper bound: 0.0042235
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0028355, upper bound: 0.0030161
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041870, -0.0036508, -0.0041870, -0.0036508, -0.0004066, 0.0004066
1: -0.0000373, 0.0029312, -0.0000373, 0.0029312, -0.0022513, 0.0022513
2: 0.0084175, 0.0150496, 0.0084175, 0.0150496, -0.0050296, 0.0050296
3: 0.0009924, 0.0037872, 0.0009924, 0.0037872, -0.0021195, 0.0021195
4: 1.0006006, 1.0114433, 1.0006006, 1.0114433, -0.0082227, 0.0082227
5: 0.0023582, 0.0044676, 0.0023582, 0.0044676, -0.0015997, 0.0015997
6: -0.0115568, -0.0088119, -0.0115568, -0.0088119, -0.0020817, 0.0020817
7: -0.0102775, -0.0099274, -0.0102775, -0.0099274, -0.0002655, 0.0002655
8: -0.0047658, -0.0028693, -0.0047658, -0.0028693, -0.0014383, 0.0014383
9: -0.0038066, 0.0056880, -0.0038066, 0.0056880, -0.0072004, 0.0072004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0039454, upper bound: 0.0034240
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0028355, upper bound: 0.0030161
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.11 seconds
IS_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0037857, upper bound: 0.0043515
IS_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0032716, upper bound: 0.0031526
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0041906, upper bound: 0.0034097
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0032716, upper bound: 0.0031526
IS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0030624, upper bound: 0.0042235
IS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0025509, upper bound: 0.0029338
IS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0030624, upper bound: 0.0042235
IS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0025509, upper bound: 0.0029338
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0043406, upper bound: 0.0034677
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0034614, upper bound: 0.0032283
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0043406, upper bound: 0.0034677
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0034614, upper bound: 0.0032283
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0032156, upper bound: 0.0042235
IS_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0028355, upper bound: 0.0030161
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0039454, upper bound: 0.0034240
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 4, lower bound: -0.0028355, upper bound: 0.0030161

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.30 + 74.38 = 77.68 seconds
