## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026658


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9881800, 0.9887558, 0.9881800, 0.9887558, -0.0004357, 0.0004357)
1: (-0.0042092, -0.0040657, -0.0042092, -0.0040657, -0.0001086, 0.0001086)
2: (0.0114922, 0.0122524, 0.0114922, 0.0122524, -0.0005753, 0.0005753)
3: (-0.0068499, -0.0065039, -0.0068499, -0.0065039, -0.0002619, 0.0002619)
4: (0.0027522, 0.0028993, 0.0027522, 0.0028993, -0.0001114, 0.0001113)
5: (0.0134136, 0.0143698, 0.0134136, 0.0143698, -0.0007236, 0.0007236)
6: (-0.0021064, -0.0018637, -0.0021064, -0.0018637, -0.0001837, 0.0001837)
7: (-0.0085874, -0.0079595, -0.0085874, -0.0079595, -0.0004752, 0.0004752)
8: (-0.0040802, -0.0037500, -0.0040802, -0.0037500, -0.0002499, 0.0002499)
9: (0.0024844, 0.0028673, 0.0024844, 0.0028673, -0.0002898, 0.0002898)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.29 = 2.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0003004, upper bound: 0.0003004

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002765, upper bound: 0.0002556
time: 0.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002765, upper bound: 0.0002765
time: 0.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 0, lower bound: -0.0002765, upper bound: 0.0002556
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 0, lower bound: -0.0002765, upper bound: 0.0002765

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9881793, 0.9886873, 0.9881802, 0.9887446, -0.0003734, 0.0003483
1: -0.0042093, -0.0040828, -0.0042092, -0.0040685, -0.0000930, 0.0000868
2: 0.0115825, 0.0122533, 0.0115069, 0.0122523, -0.0004599, 0.0004930
3: -0.0068503, -0.0065450, -0.0068498, -0.0065106, -0.0002244, 0.0002093
4: 0.0027697, 0.0028995, 0.0027550, 0.0028993, -0.0000890, 0.0000954
5: 0.0135273, 0.0143709, 0.0134321, 0.0143696, -0.0005784, 0.0006201
6: -0.0021067, -0.0018925, -0.0021063, -0.0018684, -0.0001574, 0.0001468
7: -0.0085882, -0.0080342, -0.0085874, -0.0079717, -0.0004072, 0.0003798
8: -0.0040806, -0.0037892, -0.0040802, -0.0037564, -0.0002141, 0.0001998
9: 0.0025300, 0.0028678, 0.0024919, 0.0028673, -0.0002316, 0.0002483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002556
time: 0.48 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002556
time: 0.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9881803, 0.9887382, 0.9881800, 0.9887558, -0.0004337, 0.0003266
1: -0.0042091, -0.0040701, -0.0042092, -0.0040657, -0.0001081, 0.0000814
2: 0.0115154, 0.0122521, 0.0114922, 0.0122524, -0.0004312, 0.0005727
3: -0.0068497, -0.0065144, -0.0068499, -0.0065039, -0.0002607, 0.0001963
4: 0.0027567, 0.0028993, 0.0027522, 0.0028993, -0.0000835, 0.0001108
5: 0.0134428, 0.0143694, 0.0134136, 0.0143698, -0.0005424, 0.0007203
6: -0.0021063, -0.0018711, -0.0021064, -0.0018637, -0.0001828, 0.0001377
7: -0.0085872, -0.0079787, -0.0085874, -0.0079595, -0.0004730, 0.0003562
8: -0.0040801, -0.0037601, -0.0040802, -0.0037500, -0.0002487, 0.0001873
9: 0.0024961, 0.0028672, 0.0024844, 0.0028673, -0.0002172, 0.0002884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002765
time: 0.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002765
time: 0.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.25 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002556
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002556
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002765
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0002555, upper bound: 0.0002765

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9881803, 0.9887382, 0.9881793, 0.9886873, -0.0003464, 0.0003853
1: -0.0042091, -0.0040701, -0.0042093, -0.0040828, -0.0000863, 0.0000960
2: 0.0115154, 0.0122521, 0.0115825, 0.0122533, -0.0005087, 0.0004574
3: -0.0068497, -0.0065144, -0.0068503, -0.0065450, -0.0002082, 0.0002316
4: 0.0027567, 0.0028993, 0.0027697, 0.0028995, -0.0000985, 0.0000885
5: 0.0134428, 0.0143694, 0.0135273, 0.0143709, -0.0006399, 0.0005753
6: -0.0021063, -0.0018711, -0.0021067, -0.0018925, -0.0001460, 0.0001624
7: -0.0085872, -0.0079787, -0.0085882, -0.0080342, -0.0003778, 0.0004202
8: -0.0040801, -0.0037601, -0.0040806, -0.0037892, -0.0001987, 0.0002210
9: 0.0024961, 0.0028672, 0.0025300, 0.0028678, -0.0002562, 0.0002304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002059, upper bound: 0.0002445
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001957, upper bound: 0.0002199
time: 0.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9881803, 0.9887382, 0.9881803, 0.9887382, -0.0003252, 0.0003252
1: -0.0042091, -0.0040701, -0.0042091, -0.0040701, -0.0000810, 0.0000810
2: 0.0115154, 0.0122521, 0.0115154, 0.0122521, -0.0004295, 0.0004295
3: -0.0068497, -0.0065144, -0.0068497, -0.0065144, -0.0001955, 0.0001955
4: 0.0027567, 0.0028993, 0.0027567, 0.0028993, -0.0000831, 0.0000831
5: 0.0134428, 0.0143694, 0.0134428, 0.0143694, -0.0005402, 0.0005402
6: -0.0021063, -0.0018711, -0.0021063, -0.0018711, -0.0001371, 0.0001371
7: -0.0085872, -0.0079787, -0.0085872, -0.0079787, -0.0003547, 0.0003547
8: -0.0040801, -0.0037601, -0.0040801, -0.0037601, -0.0001865, 0.0001865
9: 0.0024961, 0.0028672, 0.0024961, 0.0028672, -0.0002163, 0.0002163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002059, upper bound: 0.0002445
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0001957, upper bound: 0.0002206
time: 0.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.26 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.26
Output dim: 0, lower bound: -0.0002059, upper bound: 0.0002445
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.26
Output dim: 0, lower bound: -0.0001957, upper bound: 0.0002199
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.26
Output dim: 0, lower bound: -0.0002059, upper bound: 0.0002445
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.26
Output dim: 0, lower bound: -0.0001957, upper bound: 0.0002206

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.57 + 10.05 = 12.62 seconds
