## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00014742


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0060577, 0.0063437, 0.0060577, 0.0063437, -0.0001147, 0.0001147)
1: (-0.0001848, 0.0003692, -0.0001848, 0.0003692, -0.0002221, 0.0002221)
2: (0.0139502, 0.0184181, 0.0139502, 0.0184181, -0.0017917, 0.0017917)
3: (-0.0041576, -0.0037585, -0.0041576, -0.0037585, -0.0001600, 0.0001600)
4: (0.0011643, 0.0031005, 0.0011643, 0.0031005, -0.0007764, 0.0007764)
5: (-0.0009826, -0.0006935, -0.0009826, -0.0006935, -0.0001159, 0.0001159)
6: (0.9915251, 0.9920551, 0.9915251, 0.9920551, -0.0002126, 0.0002126)
7: (-0.0112752, -0.0077704, -0.0112752, -0.0077704, -0.0014055, 0.0014055)
8: (-0.0025441, -0.0014461, -0.0025441, -0.0014461, -0.0004403, 0.0004403)
9: (-0.0044430, -0.0022515, -0.0044430, -0.0022515, -0.0008788, 0.0008788)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.42 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0001780, upper bound: 0.0001780

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001602, upper bound: 0.0001718
time: 0.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001718, upper bound: 0.0001718
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 6, lower bound: -0.0001602, upper bound: 0.0001718
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 6, lower bound: -0.0001718, upper bound: 0.0001718

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0060669, 0.0063436, 0.0060581, 0.0063437, -0.0001054, 0.0001142
1: -0.0001669, 0.0003691, -0.0001839, 0.0003692, -0.0002042, 0.0002212
2: 0.0139504, 0.0182742, 0.0139502, 0.0184114, -0.0017840, 0.0016471
3: -0.0041447, -0.0037586, -0.0041570, -0.0037585, -0.0001471, 0.0001593
4: 0.0012267, 0.0031004, 0.0011672, 0.0031005, -0.0007137, 0.0007731
5: -0.0009826, -0.0007029, -0.0009826, -0.0006940, -0.0001154, 0.0001065
6: 0.9915423, 0.9920552, 0.9915258, 0.9920552, -0.0001954, 0.0002117
7: -0.0111623, -0.0077707, -0.0112699, -0.0077705, -0.0012920, 0.0013994
8: -0.0025087, -0.0014461, -0.0025425, -0.0014461, -0.0004048, 0.0004384
9: -0.0044428, -0.0023221, -0.0044430, -0.0022548, -0.0008750, 0.0008079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001603, upper bound: 0.0001602
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001603, upper bound: 0.0001718
time: 0.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0060658, 0.0063702, 0.0060613, 0.0063436, -0.0001120, 0.0001447
1: -0.0001690, 0.0004206, -0.0001778, 0.0003691, -0.0002170, 0.0002804
2: 0.0135353, 0.0182908, 0.0139505, 0.0183618, -0.0022613, 0.0017503
3: -0.0041462, -0.0037215, -0.0041526, -0.0037586, -0.0001563, 0.0002020
4: 0.0012195, 0.0032803, 0.0011887, 0.0031003, -0.0007585, 0.0009799
5: -0.0010094, -0.0007018, -0.0009826, -0.0006972, -0.0001463, 0.0001132
6: 0.9915402, 0.9921044, 0.9915318, 0.9920551, -0.0002077, 0.0002683
7: -0.0111754, -0.0074450, -0.0112310, -0.0077707, -0.0013730, 0.0017738
8: -0.0025128, -0.0013441, -0.0025303, -0.0014462, -0.0004301, 0.0005557
9: -0.0046464, -0.0023139, -0.0044428, -0.0022791, -0.0011091, 0.0008585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001718, upper bound: 0.0001602
time: 0.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001718, upper bound: 0.0001718
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 6, lower bound: -0.0001603, upper bound: 0.0001602
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 6, lower bound: -0.0001603, upper bound: 0.0001718
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 6, lower bound: -0.0001718, upper bound: 0.0001602
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 6, lower bound: -0.0001718, upper bound: 0.0001718

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060669, 0.0063436, 0.0060669, 0.0063436, -0.0001054, 0.0001054
1: -0.0001669, 0.0003691, -0.0001669, 0.0003691, -0.0002041, 0.0002041
2: 0.0139504, 0.0182742, 0.0139504, 0.0182742, -0.0016466, 0.0016466
3: -0.0041447, -0.0037586, -0.0041447, -0.0037586, -0.0001471, 0.0001471
4: 0.0012267, 0.0031004, 0.0012267, 0.0031004, -0.0007136, 0.0007136
5: -0.0009826, -0.0007029, -0.0009826, -0.0007029, -0.0001065, 0.0001065
6: 0.9915423, 0.9920552, 0.9915423, 0.9920552, -0.0001954, 0.0001954
7: -0.0111623, -0.0077707, -0.0111623, -0.0077707, -0.0012916, 0.0012916
8: -0.0025087, -0.0014461, -0.0025087, -0.0014461, -0.0004047, 0.0004047
9: -0.0044428, -0.0023221, -0.0044428, -0.0023221, -0.0008077, 0.0008077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001496, upper bound: 0.0001526
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001488, upper bound: 0.0001522
time: 0.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060669, 0.0063436, 0.0060658, 0.0063702, -0.0001380, 0.0001109
1: -0.0001669, 0.0003691, -0.0001690, 0.0004206, -0.0002674, 0.0002148
2: 0.0139504, 0.0182742, 0.0135353, 0.0182908, -0.0017327, 0.0021566
3: -0.0041447, -0.0037586, -0.0041462, -0.0037215, -0.0001926, 0.0001548
4: 0.0012267, 0.0031004, 0.0012195, 0.0032803, -0.0009345, 0.0007508
5: -0.0009826, -0.0007029, -0.0010094, -0.0007018, -0.0001121, 0.0001395
6: 0.9915423, 0.9920552, 0.9915402, 0.9921044, -0.0002559, 0.0002056
7: -0.0111623, -0.0077707, -0.0111754, -0.0074450, -0.0016917, 0.0013592
8: -0.0025087, -0.0014461, -0.0025128, -0.0013441, -0.0005300, 0.0004258
9: -0.0044428, -0.0023221, -0.0046464, -0.0023139, -0.0008499, 0.0010578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001496, upper bound: 0.0001557
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001488, upper bound: 0.0001543
time: 0.51 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0060658, 0.0063702, 0.0060669, 0.0063436, -0.0001109, 0.0001380
1: -0.0001690, 0.0004206, -0.0001669, 0.0003691, -0.0002148, 0.0002674
2: 0.0135353, 0.0182908, 0.0139504, 0.0182742, -0.0021566, 0.0017327
3: -0.0041462, -0.0037215, -0.0041447, -0.0037586, -0.0001548, 0.0001926
4: 0.0012195, 0.0032803, 0.0012267, 0.0031004, -0.0007508, 0.0009345
5: -0.0010094, -0.0007018, -0.0009826, -0.0007029, -0.0001395, 0.0001121
6: 0.9915402, 0.9921044, 0.9915423, 0.9920552, -0.0002056, 0.0002559
7: -0.0111754, -0.0074450, -0.0111623, -0.0077707, -0.0013592, 0.0016917
8: -0.0025128, -0.0013441, -0.0025087, -0.0014461, -0.0004258, 0.0005300
9: -0.0046464, -0.0023139, -0.0044428, -0.0023221, -0.0010578, 0.0008499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001554, upper bound: 0.0001496
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001488
time: 0.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0060658, 0.0063702, 0.0060658, 0.0063702, -0.0001128, 0.0001128
1: -0.0001690, 0.0004206, -0.0001690, 0.0004206, -0.0002185, 0.0002185
2: 0.0135353, 0.0182908, 0.0135353, 0.0182908, -0.0017622, 0.0017622
3: -0.0041462, -0.0037215, -0.0041462, -0.0037215, -0.0001574, 0.0001574
4: 0.0012195, 0.0032803, 0.0012195, 0.0032803, -0.0007636, 0.0007636
5: -0.0010094, -0.0007018, -0.0010094, -0.0007018, -0.0001140, 0.0001140
6: 0.9915402, 0.9921044, 0.9915402, 0.9921044, -0.0002091, 0.0002091
7: -0.0111754, -0.0074450, -0.0111754, -0.0074450, -0.0013823, 0.0013823
8: -0.0025128, -0.0013441, -0.0025128, -0.0013441, -0.0004331, 0.0004331
9: -0.0046464, -0.0023139, -0.0046464, -0.0023139, -0.0008643, 0.0008643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001547, upper bound: 0.0001491
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001488
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001496, upper bound: 0.0001526
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001488, upper bound: 0.0001522
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001496, upper bound: 0.0001557
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001488, upper bound: 0.0001543
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001554, upper bound: 0.0001496
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001488
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001547, upper bound: 0.0001491
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 6, lower bound: -0.0001540, upper bound: 0.0001488

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0060696, 0.0063436, 0.0060669, 0.0063436, -0.0001017, 0.0001053
1: -0.0001617, 0.0003691, -0.0001669, 0.0003691, -0.0001970, 0.0002040
2: 0.0139506, 0.0182323, 0.0139504, 0.0182742, -0.0016455, 0.0015891
3: -0.0041410, -0.0037586, -0.0041447, -0.0037586, -0.0001419, 0.0001470
4: 0.0012449, 0.0031003, 0.0012267, 0.0031004, -0.0006886, 0.0007131
5: -0.0009825, -0.0007056, -0.0009826, -0.0007029, -0.0001064, 0.0001028
6: 0.9915472, 0.9920551, 0.9915423, 0.9920552, -0.0001885, 0.0001952
7: -0.0111294, -0.0077708, -0.0111623, -0.0077707, -0.0012465, 0.0012908
8: -0.0024984, -0.0014462, -0.0025087, -0.0014461, -0.0003905, 0.0004044
9: -0.0044428, -0.0023426, -0.0044428, -0.0023221, -0.0008071, 0.0007794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001448
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001510
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0060779, 0.0063575, 0.0060706, 0.0063436, -0.0001036, 0.0001255
1: -0.0001455, 0.0003959, -0.0001597, 0.0003691, -0.0002006, 0.0002430
2: 0.0137346, 0.0181017, 0.0139507, 0.0182160, -0.0019600, 0.0016180
3: -0.0041293, -0.0037393, -0.0041395, -0.0037586, -0.0001445, 0.0001751
4: 0.0013014, 0.0031939, 0.0012519, 0.0031003, -0.0007011, 0.0008494
5: -0.0009965, -0.0007140, -0.0009825, -0.0007066, -0.0001268, 0.0001047
6: 0.9915627, 0.9920807, 0.9915491, 0.9920551, -0.0001920, 0.0002325
7: -0.0110270, -0.0076013, -0.0111166, -0.0077708, -0.0012692, 0.0015375
8: -0.0024663, -0.0013931, -0.0024944, -0.0014462, -0.0003976, 0.0004817
9: -0.0045487, -0.0024067, -0.0044427, -0.0023506, -0.0009614, 0.0007936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001508, upper bound: 0.0001448
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001506, upper bound: 0.0001506
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0060696, 0.0063436, 0.0060658, 0.0063702, -0.0001344, 0.0001108
1: -0.0001617, 0.0003691, -0.0001690, 0.0004206, -0.0002602, 0.0002147
2: 0.0139506, 0.0182323, 0.0135353, 0.0182908, -0.0017316, 0.0020990
3: -0.0041410, -0.0037586, -0.0041462, -0.0037215, -0.0001875, 0.0001547
4: 0.0012449, 0.0031003, 0.0012195, 0.0032803, -0.0009096, 0.0007504
5: -0.0009825, -0.0007056, -0.0010094, -0.0007018, -0.0001120, 0.0001358
6: 0.9915472, 0.9920551, 0.9915402, 0.9921044, -0.0002490, 0.0002054
7: -0.0111294, -0.0077708, -0.0111754, -0.0074450, -0.0016465, 0.0013583
8: -0.0024984, -0.0014462, -0.0025128, -0.0013441, -0.0005158, 0.0004255
9: -0.0044428, -0.0023426, -0.0046464, -0.0023139, -0.0008493, 0.0010296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001356, upper bound: 0.0001481
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001382, upper bound: 0.0001449
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0060779, 0.0063575, 0.0060694, 0.0063702, -0.0001362, 0.0001298
1: -0.0001455, 0.0003959, -0.0001621, 0.0004206, -0.0002639, 0.0002515
2: 0.0137346, 0.0181017, 0.0135355, 0.0182355, -0.0020282, 0.0021282
3: -0.0041293, -0.0037393, -0.0041413, -0.0037215, -0.0001901, 0.0001812
4: 0.0013014, 0.0031939, 0.0012435, 0.0032801, -0.0009222, 0.0008789
5: -0.0009965, -0.0007140, -0.0010094, -0.0007054, -0.0001312, 0.0001377
6: 0.9915627, 0.9920807, 0.9915468, 0.9921044, -0.0002525, 0.0002406
7: -0.0110270, -0.0076013, -0.0111320, -0.0074452, -0.0016694, 0.0015910
8: -0.0024663, -0.0013931, -0.0024992, -0.0013442, -0.0005230, 0.0004984
9: -0.0045487, -0.0024067, -0.0046463, -0.0023410, -0.0009948, 0.0010439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001303, upper bound: 0.0001462
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001426
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060658, 0.0063702, 0.0060696, 0.0063436, -0.0001108, 0.0001344
1: -0.0001690, 0.0004206, -0.0001617, 0.0003691, -0.0002147, 0.0002602
2: 0.0135353, 0.0182908, 0.0139506, 0.0182323, -0.0020990, 0.0017316
3: -0.0041462, -0.0037215, -0.0041410, -0.0037586, -0.0001547, 0.0001875
4: 0.0012195, 0.0032803, 0.0012449, 0.0031003, -0.0007504, 0.0009096
5: -0.0010094, -0.0007018, -0.0009825, -0.0007056, -0.0001358, 0.0001120
6: 0.9915402, 0.9921044, 0.9915472, 0.9920551, -0.0002054, 0.0002490
7: -0.0111754, -0.0074450, -0.0111294, -0.0077708, -0.0013583, 0.0016465
8: -0.0025128, -0.0013441, -0.0024984, -0.0014462, -0.0004255, 0.0005158
9: -0.0046464, -0.0023139, -0.0044428, -0.0023426, -0.0010296, 0.0008493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001481, upper bound: 0.0001356
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001449, upper bound: 0.0001382
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060694, 0.0063702, 0.0060779, 0.0063575, -0.0001298, 0.0001362
1: -0.0001621, 0.0004206, -0.0001455, 0.0003959, -0.0002515, 0.0002639
2: 0.0135355, 0.0182355, 0.0137346, 0.0181017, -0.0021282, 0.0020282
3: -0.0041413, -0.0037215, -0.0041293, -0.0037393, -0.0001812, 0.0001901
4: 0.0012435, 0.0032801, 0.0013014, 0.0031939, -0.0008789, 0.0009222
5: -0.0010094, -0.0007054, -0.0009965, -0.0007140, -0.0001377, 0.0001312
6: 0.9915468, 0.9921044, 0.9915627, 0.9920807, -0.0002406, 0.0002525
7: -0.0111320, -0.0074452, -0.0110270, -0.0076013, -0.0015910, 0.0016694
8: -0.0024992, -0.0013442, -0.0024663, -0.0013931, -0.0004984, 0.0005230
9: -0.0046463, -0.0023410, -0.0045487, -0.0024067, -0.0010439, 0.0009948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001462, upper bound: 0.0001303
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001426, upper bound: 0.0001338
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0060688, 0.0063702, 0.0060658, 0.0063702, -0.0001091, 0.0001127
1: -0.0001632, 0.0004206, -0.0001690, 0.0004206, -0.0002112, 0.0002183
2: 0.0135354, 0.0182440, 0.0135353, 0.0182908, -0.0017610, 0.0017039
3: -0.0041420, -0.0037215, -0.0041462, -0.0037215, -0.0001522, 0.0001573
4: 0.0012398, 0.0032802, 0.0012195, 0.0032803, -0.0007384, 0.0007631
5: -0.0010094, -0.0007048, -0.0010094, -0.0007018, -0.0001139, 0.0001102
6: 0.9915457, 0.9921044, 0.9915402, 0.9921044, -0.0002022, 0.0002089
7: -0.0111386, -0.0074451, -0.0111754, -0.0074450, -0.0013366, 0.0013814
8: -0.0025013, -0.0013442, -0.0025128, -0.0013441, -0.0004187, 0.0004328
9: -0.0046464, -0.0023369, -0.0046464, -0.0023139, -0.0008638, 0.0008357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001452, upper bound: 0.0001303
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001413, upper bound: 0.0001357
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0060769, 0.0063841, 0.0060694, 0.0063702, -0.0001111, 0.0001326
1: -0.0001475, 0.0004475, -0.0001621, 0.0004206, -0.0002151, 0.0002568
2: 0.0133183, 0.0181173, 0.0135355, 0.0182355, -0.0020713, 0.0017354
3: -0.0041307, -0.0037021, -0.0041413, -0.0037215, -0.0001550, 0.0001850
4: 0.0012947, 0.0033743, 0.0012435, 0.0032801, -0.0007520, 0.0008976
5: -0.0010234, -0.0007130, -0.0010094, -0.0007054, -0.0001340, 0.0001123
6: 0.9915608, 0.9921302, 0.9915468, 0.9921044, -0.0002059, 0.0002457
7: -0.0110392, -0.0072748, -0.0111320, -0.0074452, -0.0013612, 0.0016247
8: -0.0024702, -0.0012908, -0.0024992, -0.0013442, -0.0004265, 0.0005090
9: -0.0047529, -0.0023990, -0.0046463, -0.0023410, -0.0010159, 0.0008512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001429, upper bound: 0.0001299
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001332, upper bound: 0.0001330
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.77 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001448
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001510
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001508, upper bound: 0.0001448
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001506, upper bound: 0.0001506
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001356, upper bound: 0.0001481
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001382, upper bound: 0.0001449
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001303, upper bound: 0.0001462
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001338, upper bound: 0.0001426
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001481, upper bound: 0.0001356
IS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001449, upper bound: 0.0001382
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001462, upper bound: 0.0001303
IS_A2_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001426, upper bound: 0.0001338
IS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001452, upper bound: 0.0001303
IS_A2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001413, upper bound: 0.0001357
IS_A2_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001429, upper bound: 0.0001299
IS_A2_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001332, upper bound: 0.0001330

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0060697, 0.0063404, 0.0060669, 0.0063426, -0.0001007, 0.0001030
1: -0.0001614, 0.0003629, -0.0001668, 0.0003672, -0.0001951, 0.0001995
2: 0.0140007, 0.0182297, 0.0139663, 0.0182733, -0.0016094, 0.0015733
3: -0.0041408, -0.0037630, -0.0041447, -0.0037600, -0.0001405, 0.0001437
4: 0.0012460, 0.0030786, 0.0012271, 0.0030935, -0.0006818, 0.0006974
5: -0.0009793, -0.0007057, -0.0009815, -0.0007029, -0.0001041, 0.0001018
6: 0.9915475, 0.9920492, 0.9915423, 0.9920533, -0.0001867, 0.0001909
7: -0.0111274, -0.0078101, -0.0111616, -0.0077831, -0.0012341, 0.0012624
8: -0.0024978, -0.0014585, -0.0025085, -0.0014500, -0.0003866, 0.0003955
9: -0.0044182, -0.0023439, -0.0044351, -0.0023225, -0.0007894, 0.0007717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001448
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001448
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0060663, 0.0063410, 0.0060670, 0.0063426, -0.0001047, 0.0001042
1: -0.0001679, 0.0003641, -0.0001668, 0.0003671, -0.0002029, 0.0002018
2: 0.0139911, 0.0182825, 0.0139669, 0.0182730, -0.0016280, 0.0016362
3: -0.0041455, -0.0037622, -0.0041446, -0.0037600, -0.0001461, 0.0001454
4: 0.0012231, 0.0030827, 0.0012272, 0.0030932, -0.0007090, 0.0007055
5: -0.0009799, -0.0007023, -0.0009815, -0.0007029, -0.0001053, 0.0001058
6: 0.9915412, 0.9920503, 0.9915424, 0.9920532, -0.0001941, 0.0001932
7: -0.0111688, -0.0078026, -0.0111613, -0.0077836, -0.0012835, 0.0012771
8: -0.0025108, -0.0014561, -0.0025084, -0.0014502, -0.0004021, 0.0004001
9: -0.0044229, -0.0023180, -0.0044347, -0.0023227, -0.0007985, 0.0008025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of IS_A1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001487
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001478
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0060781, 0.0063539, 0.0060707, 0.0063426, -0.0001026, 0.0001226
1: -0.0001452, 0.0003891, -0.0001596, 0.0003671, -0.0001986, 0.0002374
2: 0.0137895, 0.0180990, 0.0139665, 0.0182152, -0.0019148, 0.0016023
3: -0.0041291, -0.0037442, -0.0041395, -0.0037600, -0.0001431, 0.0001710
4: 0.0013026, 0.0031701, 0.0012523, 0.0030934, -0.0006943, 0.0008297
5: -0.0009930, -0.0007142, -0.0009815, -0.0007067, -0.0001239, 0.0001036
6: 0.9915630, 0.9920742, 0.9915492, 0.9920533, -0.0001901, 0.0002272
7: -0.0110249, -0.0076444, -0.0111160, -0.0077833, -0.0012569, 0.0015020
8: -0.0024657, -0.0014066, -0.0024942, -0.0014501, -0.0003938, 0.0004706
9: -0.0045218, -0.0024080, -0.0044349, -0.0023510, -0.0009392, 0.0007859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of IS_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001469, upper bound: 0.0001424
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001475, upper bound: 0.0001415
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0060740, 0.0063554, 0.0060707, 0.0063426, -0.0001065, 0.0001235
1: -0.0001532, 0.0003919, -0.0001595, 0.0003671, -0.0002063, 0.0002393
2: 0.0137671, 0.0181636, 0.0139671, 0.0182148, -0.0019299, 0.0016638
3: -0.0041348, -0.0037422, -0.0041394, -0.0037600, -0.0001486, 0.0001724
4: 0.0012746, 0.0031798, 0.0012524, 0.0030931, -0.0007210, 0.0008363
5: -0.0009944, -0.0007100, -0.0009815, -0.0007067, -0.0001248, 0.0001076
6: 0.9915553, 0.9920769, 0.9915493, 0.9920532, -0.0001974, 0.0002290
7: -0.0110755, -0.0076268, -0.0111157, -0.0077838, -0.0013051, 0.0015139
8: -0.0024815, -0.0014011, -0.0024941, -0.0014503, -0.0004089, 0.0004743
9: -0.0045328, -0.0023763, -0.0044346, -0.0023512, -0.0009466, 0.0008161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001469, upper bound: 0.0001485
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001475, upper bound: 0.0001475
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060696, 0.0063426, 0.0060660, 0.0063668, -0.0001313, 0.0001098
1: -0.0001616, 0.0003672, -0.0001686, 0.0004140, -0.0002544, 0.0002126
2: 0.0139664, 0.0182315, 0.0135887, 0.0182879, -0.0017148, 0.0020519
3: -0.0041409, -0.0037600, -0.0041460, -0.0037262, -0.0001833, 0.0001532
4: 0.0012452, 0.0030934, 0.0012208, 0.0032571, -0.0008892, 0.0007431
5: -0.0009815, -0.0007056, -0.0010060, -0.0007020, -0.0001109, 0.0001327
6: 0.9915473, 0.9920533, 0.9915406, 0.9920982, -0.0002434, 0.0002035
7: -0.0111288, -0.0077832, -0.0111731, -0.0074869, -0.0016096, 0.0013451
8: -0.0024982, -0.0014501, -0.0025121, -0.0013572, -0.0005043, 0.0004214
9: -0.0044350, -0.0023430, -0.0046203, -0.0023153, -0.0008411, 0.0010064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 160

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001332, upper bound: 0.0001446
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001328, upper bound: 0.0001448
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0060660, 0.0063668, 0.0060696, 0.0063426, -0.0001098, 0.0001313
1: -0.0001686, 0.0004140, -0.0001616, 0.0003672, -0.0002126, 0.0002544
2: 0.0135887, 0.0182879, 0.0139664, 0.0182315, -0.0020519, 0.0017148
3: -0.0041460, -0.0037262, -0.0041409, -0.0037600, -0.0001532, 0.0001833
4: 0.0012208, 0.0032571, 0.0012452, 0.0030934, -0.0007431, 0.0008892
5: -0.0010060, -0.0007020, -0.0009815, -0.0007056, -0.0001327, 0.0001109
6: 0.9915406, 0.9920982, 0.9915473, 0.9920533, -0.0002035, 0.0002434
7: -0.0111731, -0.0074869, -0.0111288, -0.0077832, -0.0013451, 0.0016096
8: -0.0025121, -0.0013572, -0.0024982, -0.0014501, -0.0004214, 0.0005043
9: -0.0046203, -0.0023153, -0.0044350, -0.0023430, -0.0010064, 0.0008411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001446, upper bound: 0.0001332
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001448, upper bound: 0.0001328
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.70 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001448
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001501, upper bound: 0.0001448
IS_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001487
IS_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001478
IS_A1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001469, upper bound: 0.0001424
IS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001475, upper bound: 0.0001415
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001469, upper bound: 0.0001485
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001475, upper bound: 0.0001475
IS_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001332, upper bound: 0.0001446
IS_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001328, upper bound: 0.0001448
IS_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001446, upper bound: 0.0001332
IS_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001448, upper bound: 0.0001328

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060697, 0.0063404, 0.0060696, 0.0063426, -0.0001006, 0.0000993
1: -0.0001614, 0.0003629, -0.0001616, 0.0003672, -0.0001949, 0.0001924
2: 0.0140007, 0.0182297, 0.0139664, 0.0182315, -0.0015518, 0.0015721
3: -0.0041408, -0.0037630, -0.0041409, -0.0037600, -0.0001404, 0.0001386
4: 0.0012460, 0.0030786, 0.0012452, 0.0030934, -0.0006813, 0.0006724
5: -0.0009793, -0.0007057, -0.0009815, -0.0007056, -0.0001004, 0.0001017
6: 0.9915475, 0.9920492, 0.9915473, 0.9920533, -0.0001865, 0.0001841
7: -0.0111274, -0.0078101, -0.0111288, -0.0077832, -0.0012332, 0.0012172
8: -0.0024978, -0.0014585, -0.0024982, -0.0014501, -0.0003864, 0.0003814
9: -0.0044182, -0.0023439, -0.0044350, -0.0023430, -0.0007611, 0.0007711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 160

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001424
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001415
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060697, 0.0063404, 0.0060780, 0.0063563, -0.0001225, 0.0000996
1: -0.0001614, 0.0003629, -0.0001454, 0.0003937, -0.0002373, 0.0001930
2: 0.0140007, 0.0182297, 0.0137524, 0.0181009, -0.0015567, 0.0019141
3: -0.0041408, -0.0037630, -0.0041292, -0.0037409, -0.0001710, 0.0001390
4: 0.0012460, 0.0030786, 0.0013018, 0.0031862, -0.0008295, 0.0006746
5: -0.0009793, -0.0007057, -0.0009954, -0.0007141, -0.0001007, 0.0001238
6: 0.9915475, 0.9920492, 0.9915628, 0.9920787, -0.0002271, 0.0001847
7: -0.0111274, -0.0078101, -0.0110263, -0.0076153, -0.0015015, 0.0012211
8: -0.0024978, -0.0014585, -0.0024661, -0.0013975, -0.0004704, 0.0003826
9: -0.0044182, -0.0023439, -0.0045399, -0.0024071, -0.0007635, 0.0009388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 160

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001480, upper bound: 0.0001409
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001415
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0060665, 0.0063388, 0.0060670, 0.0063418, -0.0001036, 0.0001010
1: -0.0001677, 0.0003597, -0.0001667, 0.0003656, -0.0002007, 0.0001956
2: 0.0140263, 0.0182803, 0.0139789, 0.0182722, -0.0015773, 0.0016187
3: -0.0041453, -0.0037653, -0.0041446, -0.0037611, -0.0001446, 0.0001409
4: 0.0012240, 0.0030675, 0.0012276, 0.0030880, -0.0007014, 0.0006835
5: -0.0009777, -0.0007025, -0.0009807, -0.0007030, -0.0001020, 0.0001047
6: 0.9915414, 0.9920462, 0.9915424, 0.9920518, -0.0001920, 0.0001871
7: -0.0111671, -0.0078302, -0.0111608, -0.0077930, -0.0012697, 0.0012373
8: -0.0025102, -0.0014648, -0.0025082, -0.0014531, -0.0003978, 0.0003876
9: -0.0044056, -0.0023190, -0.0044289, -0.0023230, -0.0007736, 0.0007939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001484
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001487
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0060667, 0.0063377, 0.0060670, 0.0063412, -0.0001039, 0.0001007
1: -0.0001672, 0.0003576, -0.0001667, 0.0003645, -0.0002012, 0.0001950
2: 0.0140433, 0.0182769, 0.0139880, 0.0182721, -0.0015732, 0.0016229
3: -0.0041450, -0.0037668, -0.0041445, -0.0037619, -0.0001450, 0.0001405
4: 0.0012255, 0.0030601, 0.0012276, 0.0030841, -0.0007033, 0.0006817
5: -0.0009765, -0.0007027, -0.0009801, -0.0007030, -0.0001018, 0.0001050
6: 0.9915419, 0.9920442, 0.9915425, 0.9920507, -0.0001925, 0.0001866
7: -0.0111644, -0.0078435, -0.0111607, -0.0078002, -0.0012730, 0.0012340
8: -0.0025094, -0.0014690, -0.0025082, -0.0014554, -0.0003988, 0.0003866
9: -0.0043973, -0.0023207, -0.0044244, -0.0023231, -0.0007716, 0.0007960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001474
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001478
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0060783, 0.0063507, 0.0060707, 0.0063413, -0.0001038, 0.0001195
1: -0.0001448, 0.0003828, -0.0001595, 0.0003645, -0.0002011, 0.0002314
2: 0.0138406, 0.0180960, 0.0139877, 0.0182143, -0.0018664, 0.0016222
3: -0.0041288, -0.0037487, -0.0041394, -0.0037619, -0.0001449, 0.0001667
4: 0.0013039, 0.0031479, 0.0012526, 0.0030842, -0.0007030, 0.0008088
5: -0.0009897, -0.0007144, -0.0009801, -0.0007067, -0.0001207, 0.0001049
6: 0.9915633, 0.9920682, 0.9915493, 0.9920508, -0.0001925, 0.0002214
7: -0.0110225, -0.0076845, -0.0111153, -0.0077999, -0.0012725, 0.0014641
8: -0.0024649, -0.0014192, -0.0024940, -0.0014553, -0.0003987, 0.0004587
9: -0.0044967, -0.0024095, -0.0044246, -0.0023514, -0.0009155, 0.0007957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001450, upper bound: 0.0001344
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001455, upper bound: 0.0001396
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0060741, 0.0063530, 0.0060707, 0.0063418, -0.0001054, 0.0001208
1: -0.0001529, 0.0003873, -0.0001595, 0.0003656, -0.0002041, 0.0002340
2: 0.0138042, 0.0181615, 0.0139791, 0.0182140, -0.0018876, 0.0016461
3: -0.0041347, -0.0037455, -0.0041394, -0.0037611, -0.0001470, 0.0001686
4: 0.0012755, 0.0031637, 0.0012528, 0.0030879, -0.0007133, 0.0008180
5: -0.0009920, -0.0007102, -0.0009807, -0.0007068, -0.0001221, 0.0001065
6: 0.9915556, 0.9920725, 0.9915493, 0.9920518, -0.0001953, 0.0002240
7: -0.0110739, -0.0076560, -0.0111151, -0.0077932, -0.0012913, 0.0014807
8: -0.0024810, -0.0014102, -0.0024939, -0.0014532, -0.0004045, 0.0004639
9: -0.0045145, -0.0023773, -0.0044288, -0.0023516, -0.0009258, 0.0008074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001443, upper bound: 0.0001421
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001448, upper bound: 0.0001465
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0060741, 0.0063521, 0.0060707, 0.0063412, -0.0001057, 0.0001206
1: -0.0001529, 0.0003855, -0.0001594, 0.0003645, -0.0002048, 0.0002336
2: 0.0138182, 0.0181612, 0.0139883, 0.0182139, -0.0018838, 0.0016517
3: -0.0041346, -0.0037467, -0.0041393, -0.0037619, -0.0001475, 0.0001683
4: 0.0012757, 0.0031576, 0.0012528, 0.0030840, -0.0007158, 0.0008163
5: -0.0009911, -0.0007102, -0.0009801, -0.0007068, -0.0001219, 0.0001068
6: 0.9915556, 0.9920708, 0.9915493, 0.9920506, -0.0001960, 0.0002235
7: -0.0110736, -0.0076670, -0.0111150, -0.0078004, -0.0012957, 0.0014777
8: -0.0024810, -0.0014137, -0.0024939, -0.0014555, -0.0004059, 0.0004630
9: -0.0045077, -0.0023775, -0.0044243, -0.0023516, -0.0009240, 0.0008102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001448, upper bound: 0.0001411
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001455, upper bound: 0.0001455
time: 0.55 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001424
IS_A1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001415
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001480, upper bound: 0.0001409
IS_A1_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001415
IS_A1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001484
IS_A1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001464, upper bound: 0.0001487
IS_A1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001474
IS_A1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001468, upper bound: 0.0001478
IS_A1_B1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001450, upper bound: 0.0001344
IS_A1_B1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001455, upper bound: 0.0001396
IS_A1_B1_A2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001443, upper bound: 0.0001421
IS_A1_B1_A2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001448, upper bound: 0.0001465
IS_A1_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001448, upper bound: 0.0001411
IS_A1_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 6, lower bound: -0.0001455, upper bound: 0.0001455

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060698, 0.0063397, 0.0060781, 0.0063539, -0.0001199, 0.0000983
1: -0.0001613, 0.0003614, -0.0001452, 0.0003891, -0.0002322, 0.0001903
2: 0.0140126, 0.0182290, 0.0137894, 0.0180987, -0.0015351, 0.0018725
3: -0.0041407, -0.0037641, -0.0041291, -0.0037442, -0.0001672, 0.0001371
4: 0.0012463, 0.0030734, 0.0013027, 0.0031702, -0.0008114, 0.0006652
5: -0.0009785, -0.0007058, -0.0009930, -0.0007142, -0.0000993, 0.0001211
6: 0.9915476, 0.9920478, 0.9915630, 0.9920743, -0.0002222, 0.0001821
7: -0.0111268, -0.0078195, -0.0110247, -0.0076443, -0.0014689, 0.0012042
8: -0.0024976, -0.0014614, -0.0024656, -0.0014066, -0.0004602, 0.0003773
9: -0.0044123, -0.0023443, -0.0045218, -0.0024081, -0.0007530, 0.0009185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001433, upper bound: 0.0001387
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001458, upper bound: 0.0001389
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060665, 0.0063388, 0.0060697, 0.0063418, -0.0001035, 0.0000973
1: -0.0001677, 0.0003597, -0.0001615, 0.0003656, -0.0002006, 0.0001884
2: 0.0140263, 0.0182803, 0.0139790, 0.0182304, -0.0015197, 0.0016176
3: -0.0041453, -0.0037653, -0.0041408, -0.0037611, -0.0001445, 0.0001357
4: 0.0012240, 0.0030675, 0.0012457, 0.0030880, -0.0007010, 0.0006585
5: -0.0009777, -0.0007025, -0.0009807, -0.0007057, -0.0000983, 0.0001046
6: 0.9915414, 0.9920462, 0.9915474, 0.9920517, -0.0001919, 0.0001803
7: -0.0111671, -0.0078302, -0.0111279, -0.0077931, -0.0012689, 0.0011921
8: -0.0025102, -0.0014648, -0.0024980, -0.0014532, -0.0003975, 0.0003735
9: -0.0044056, -0.0023190, -0.0044288, -0.0023435, -0.0007454, 0.0007934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001440, upper bound: 0.0001434
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001443, upper bound: 0.0001463
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060665, 0.0063388, 0.0060780, 0.0063558, -0.0001255, 0.0000976
1: -0.0001677, 0.0003597, -0.0001453, 0.0003927, -0.0002431, 0.0001890
2: 0.0140263, 0.0182803, 0.0137606, 0.0180997, -0.0015245, 0.0019604
3: -0.0041453, -0.0037653, -0.0041291, -0.0037416, -0.0001751, 0.0001362
4: 0.0012240, 0.0030675, 0.0013023, 0.0031826, -0.0008495, 0.0006606
5: -0.0009777, -0.0007025, -0.0009948, -0.0007141, -0.0000986, 0.0001268
6: 0.9915414, 0.9920462, 0.9915629, 0.9920776, -0.0002326, 0.0001809
7: -0.0111671, -0.0078302, -0.0110254, -0.0076218, -0.0015378, 0.0011958
8: -0.0025102, -0.0014648, -0.0024658, -0.0013995, -0.0004818, 0.0003746
9: -0.0044056, -0.0023190, -0.0045359, -0.0024076, -0.0007477, 0.0009616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001403, upper bound: 0.0001465
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001443, upper bound: 0.0001467
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0060667, 0.0063377, 0.0060697, 0.0063412, -0.0001038, 0.0000970
1: -0.0001672, 0.0003576, -0.0001615, 0.0003645, -0.0002011, 0.0001879
2: 0.0140433, 0.0182769, 0.0139882, 0.0182303, -0.0015156, 0.0016217
3: -0.0041450, -0.0037668, -0.0041408, -0.0037619, -0.0001448, 0.0001354
4: 0.0012255, 0.0030601, 0.0012457, 0.0030840, -0.0007028, 0.0006568
5: -0.0009765, -0.0007027, -0.0009801, -0.0007057, -0.0000980, 0.0001049
6: 0.9915419, 0.9920442, 0.9915473, 0.9920506, -0.0001924, 0.0001798
7: -0.0111644, -0.0078435, -0.0111279, -0.0078003, -0.0012721, 0.0011888
8: -0.0025094, -0.0014690, -0.0024979, -0.0014554, -0.0003985, 0.0003725
9: -0.0043973, -0.0023207, -0.0044243, -0.0023436, -0.0007434, 0.0007954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001444, upper bound: 0.0001422
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001447, upper bound: 0.0001454
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0060667, 0.0063377, 0.0060781, 0.0063554, -0.0001253, 0.0000973
1: -0.0001672, 0.0003576, -0.0001453, 0.0003919, -0.0002427, 0.0001885
2: 0.0140433, 0.0182769, 0.0137669, 0.0180996, -0.0015204, 0.0019580
3: -0.0041450, -0.0037668, -0.0041291, -0.0037422, -0.0001749, 0.0001358
4: 0.0012255, 0.0030601, 0.0013023, 0.0031799, -0.0008485, 0.0006588
5: -0.0009765, -0.0007027, -0.0009944, -0.0007142, -0.0000984, 0.0001267
6: 0.9915419, 0.9920442, 0.9915629, 0.9920769, -0.0002323, 0.0001804
7: -0.0111644, -0.0078435, -0.0110254, -0.0076267, -0.0015359, 0.0011926
8: -0.0025094, -0.0014690, -0.0024658, -0.0014010, -0.0004812, 0.0003736
9: -0.0043973, -0.0023207, -0.0045328, -0.0024077, -0.0007457, 0.0009604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001420, upper bound: 0.0001454
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001447, upper bound: 0.0001457
time: 0.60 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001433, upper bound: 0.0001387
IS_A1_B1_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001458, upper bound: 0.0001389
IS_A1_B1_A1_A2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001440, upper bound: 0.0001434
IS_A1_B1_A1_A2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001443, upper bound: 0.0001463
IS_A1_B1_A1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001403, upper bound: 0.0001465
IS_A1_B1_A1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001443, upper bound: 0.0001467
IS_A1_B1_A1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001444, upper bound: 0.0001422
IS_A1_B1_A1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001447, upper bound: 0.0001454
IS_A1_B1_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001420, upper bound: 0.0001454
IS_A1_B1_A1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 6, lower bound: -0.0001447, upper bound: 0.0001457

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.95 + 88.47 = 91.42 seconds
