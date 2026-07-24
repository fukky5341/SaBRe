## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0001407


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0083470, -0.0071343, -0.0083470, -0.0071343, -0.0007379, 0.0007379)
1: (-0.0052920, -0.0049501, -0.0052920, -0.0049501, -0.0002081, 0.0002081)
2: (-0.0004856, 0.0020371, -0.0004856, 0.0020371, -0.0015351, 0.0015351)
3: (0.0015630, 0.0018969, 0.0015630, 0.0018969, -0.0002031, 0.0002031)
4: (0.0045694, 0.0064548, 0.0045694, 0.0064548, -0.0011472, 0.0011472)
5: (0.9967757, 0.9972996, 0.9967757, 0.9972996, -0.0003187, 0.0003187)
6: (0.0049570, 0.0054325, 0.0049570, 0.0054325, -0.0002893, 0.0002893)
7: (-0.0048828, -0.0031085, -0.0048828, -0.0031085, -0.0010797, 0.0010797)
8: (-0.0067735, -0.0053926, -0.0067735, -0.0053926, -0.0008403, 0.0008403)
9: (-0.0035445, -0.0034253, -0.0035445, -0.0034253, -0.0000725, 0.0000725)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.43 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0001939, upper bound: 0.0001940

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001802, upper bound: 0.0001809
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0001828
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 5, lower bound: -0.0001802, upper bound: 0.0001809
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0001828

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0084170, -0.0072217, -0.0083470, -0.0071585, -0.0007092, 0.0006124
1: -0.0053117, -0.0049747, -0.0052920, -0.0049569, -0.0002000, 0.0001727
2: -0.0006313, 0.0018553, -0.0004856, 0.0019866, -0.0014754, 0.0012740
3: 0.0015438, 0.0018728, 0.0015630, 0.0018902, -0.0001952, 0.0001686
4: 0.0047054, 0.0065637, 0.0046072, 0.0064547, -0.0009521, 0.0011026
5: 0.9968135, 0.9973298, 0.9967862, 0.9972996, -0.0002645, 0.0003063
6: 0.0049913, 0.0054599, 0.0049665, 0.0054325, -0.0002401, 0.0002781
7: -0.0047549, -0.0030060, -0.0048473, -0.0031085, -0.0008960, 0.0010377
8: -0.0068533, -0.0054921, -0.0067735, -0.0054202, -0.0008076, 0.0006974
9: -0.0035359, -0.0034185, -0.0035421, -0.0034254, -0.0000602, 0.0000697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001696, upper bound: 0.0001760
time: 0.61 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001749, upper bound: 0.0001762
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0083470, -0.0071522, -0.0083470, -0.0071343, -0.0007379, 0.0006177
1: -0.0052920, -0.0049551, -0.0052920, -0.0049501, -0.0002080, 0.0001741
2: -0.0004856, 0.0019997, -0.0004856, 0.0020371, -0.0015350, 0.0012848
3: 0.0015630, 0.0018919, 0.0015630, 0.0018969, -0.0002031, 0.0001700
4: 0.0045974, 0.0064547, 0.0045694, 0.0064548, -0.0009602, 0.0011472
5: 0.9967836, 0.9972996, 0.9967757, 0.9972996, -0.0002668, 0.0003187
6: 0.0049641, 0.0054325, 0.0049570, 0.0054325, -0.0002422, 0.0002893
7: -0.0048565, -0.0031085, -0.0048828, -0.0031085, -0.0009037, 0.0010796
8: -0.0067735, -0.0054131, -0.0067735, -0.0053926, -0.0008403, 0.0007033
9: -0.0035427, -0.0034253, -0.0035445, -0.0034253, -0.0000607, 0.0000725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001809, upper bound: 0.0001797
time: 0.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001809, upper bound: 0.0001827
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.66 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001696, upper bound: 0.0001760
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001749, upper bound: 0.0001762
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001809, upper bound: 0.0001797
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001809, upper bound: 0.0001827

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0084070, -0.0072219, -0.0083120, -0.0071651, -0.0006796, 0.0005721
1: -0.0053089, -0.0049748, -0.0052821, -0.0049588, -0.0001916, 0.0001613
2: -0.0006105, 0.0018549, -0.0004129, 0.0019730, -0.0014137, 0.0011901
3: 0.0015465, 0.0018728, 0.0015727, 0.0018884, -0.0001871, 0.0001575
4: 0.0047056, 0.0065481, 0.0046174, 0.0064004, -0.0008894, 0.0010565
5: 0.9968136, 0.9973254, 0.9967890, 0.9972844, -0.0002471, 0.0002935
6: 0.0049914, 0.0054560, 0.0049691, 0.0054188, -0.0002243, 0.0002664
7: -0.0047546, -0.0030206, -0.0048377, -0.0031596, -0.0008370, 0.0009943
8: -0.0068419, -0.0054924, -0.0067337, -0.0054277, -0.0007739, 0.0006514
9: -0.0035359, -0.0034194, -0.0035415, -0.0034288, -0.0000562, 0.0000668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001631, upper bound: 0.0001688
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.ADV_EXAMPLE
time: 0.57 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.93 + 9.56 = 12.49 seconds
