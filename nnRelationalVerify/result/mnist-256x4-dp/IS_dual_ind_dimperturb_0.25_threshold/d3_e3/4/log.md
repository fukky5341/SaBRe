## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085666


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9937598, 0.9964234, 0.9937598, 0.9964234, -0.0020074, 0.0020074)
1: (-0.0028189, -0.0021551, -0.0028189, -0.0021551, -0.0005002, 0.0005002)
2: (0.0013671, 0.0048845, 0.0013671, 0.0048845, -0.0026507, 0.0026508)
3: (-0.0034963, -0.0018954, -0.0034963, -0.0018954, -0.0012065, 0.0012065)
4: (0.0007925, 0.0014733, 0.0007925, 0.0014733, -0.0005130, 0.0005130)
5: (0.0006789, 0.0051029, 0.0006789, 0.0051029, -0.0033339, 0.0033339)
6: (0.0002457, 0.0013685, 0.0002457, 0.0013685, -0.0008462, 0.0008462)
7: (-0.0025020, 0.0004031, -0.0025020, 0.0004031, -0.0021894, 0.0021894)
8: (-0.0008799, 0.0006479, -0.0008799, 0.0006479, -0.0011514, 0.0011514)
9: (-0.0026151, -0.0008435, -0.0026151, -0.0008435, -0.0013351, 0.0013351)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.70 + 1.51 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012179, upper bound: 0.0012180

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011455, upper bound: 0.0010714
time: 0.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011494, upper bound: 0.0011495
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.45 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 0, lower bound: -0.0011455, upper bound: 0.0010714
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 0, lower bound: -0.0011494, upper bound: 0.0011495

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9937513, 0.9960531, 0.9937657, 0.9963052, -0.0018369, 0.0016064
1: -0.0028210, -0.0022474, -0.0028174, -0.0021846, -0.0004577, 0.0004003
2: 0.0018562, 0.0048956, 0.0015233, 0.0048766, -0.0021212, 0.0024256
3: -0.0035014, -0.0021180, -0.0034927, -0.0019664, -0.0011040, 0.0009655
4: 0.0008872, 0.0014754, 0.0008227, 0.0014717, -0.0004106, 0.0004695
5: 0.0012941, 0.0051168, 0.0008753, 0.0050929, -0.0026680, 0.0030508
6: 0.0002421, 0.0012124, 0.0002482, 0.0013187, -0.0007743, 0.0006772
7: -0.0025111, -0.0000008, -0.0024955, 0.0002742, -0.0020034, 0.0017520
8: -0.0008847, 0.0004354, -0.0008765, 0.0005800, -0.0010536, 0.0009214
9: -0.0023687, -0.0008380, -0.0025364, -0.0008475, -0.0010684, 0.0012217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010714
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010714
time: 0.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9937649, 0.9963568, 0.9937599, 0.9964205, -0.0019876, 0.0017168
1: -0.0028176, -0.0021717, -0.0028188, -0.0021559, -0.0004953, 0.0004278
2: 0.0014551, 0.0048777, 0.0013709, 0.0048842, -0.0022670, 0.0026247
3: -0.0034932, -0.0019354, -0.0034962, -0.0018971, -0.0011946, 0.0010319
4: 0.0008095, 0.0014720, 0.0007932, 0.0014732, -0.0004388, 0.0005080
5: 0.0007896, 0.0050943, 0.0006837, 0.0051025, -0.0028513, 0.0033011
6: 0.0002478, 0.0013404, 0.0002458, 0.0013673, -0.0008379, 0.0007237
7: -0.0024964, 0.0003305, -0.0025018, 0.0004000, -0.0021678, 0.0018724
8: -0.0008770, 0.0006096, -0.0008798, 0.0006462, -0.0011400, 0.0009847
9: -0.0025708, -0.0008470, -0.0026132, -0.0008437, -0.0011418, 0.0013219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0011455
time: 0.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0011494
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010714
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0010714
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0011455
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -0.0010714, upper bound: 0.0011494

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937513, 0.9960531, 0.9937513, 0.9960531, -0.0015865, 0.0015865
1: -0.0028210, -0.0022474, -0.0028210, -0.0022474, -0.0003953, 0.0003953
2: 0.0018562, 0.0048956, 0.0018562, 0.0048956, -0.0020950, 0.0020950
3: -0.0035014, -0.0021180, -0.0035014, -0.0021180, -0.0009536, 0.0009536
4: 0.0008872, 0.0014754, 0.0008872, 0.0014754, -0.0004055, 0.0004055
5: 0.0012941, 0.0051168, 0.0012941, 0.0051168, -0.0026350, 0.0026350
6: 0.0002421, 0.0012124, 0.0002421, 0.0012124, -0.0006688, 0.0006688
7: -0.0025111, -0.0000008, -0.0025111, -0.0000008, -0.0017303, 0.0017303
8: -0.0008847, 0.0004354, -0.0008847, 0.0004354, -0.0009100, 0.0009100
9: -0.0023687, -0.0008380, -0.0023687, -0.0008380, -0.0010552, 0.0010552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010214, upper bound: 0.0009644
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010254, upper bound: 0.0010238
time: 0.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937513, 0.9960531, 0.9937649, 0.9963568, -0.0019471, 0.0015960
1: -0.0028210, -0.0022474, -0.0028176, -0.0021717, -0.0004852, 0.0003977
2: 0.0018562, 0.0048956, 0.0014551, 0.0048777, -0.0021076, 0.0025711
3: -0.0035014, -0.0021180, -0.0034932, -0.0019354, -0.0011702, 0.0009593
4: 0.0008872, 0.0014754, 0.0008095, 0.0014720, -0.0004079, 0.0004976
5: 0.0012941, 0.0051168, 0.0007896, 0.0050943, -0.0026507, 0.0032337
6: 0.0002421, 0.0012124, 0.0002478, 0.0013404, -0.0008208, 0.0006728
7: -0.0025111, -0.0000008, -0.0024964, 0.0003305, -0.0021236, 0.0017407
8: -0.0008847, 0.0004354, -0.0008770, 0.0006096, -0.0011168, 0.0009154
9: -0.0023687, -0.0008380, -0.0025708, -0.0008470, -0.0010615, 0.0012949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010214, upper bound: 0.0009644
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010254, upper bound: 0.0010238
time: 0.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937649, 0.9963568, 0.9937513, 0.9960531, -0.0015960, 0.0019471
1: -0.0028176, -0.0021717, -0.0028210, -0.0022474, -0.0003977, 0.0004852
2: 0.0014551, 0.0048777, 0.0018562, 0.0048956, -0.0025711, 0.0021076
3: -0.0034932, -0.0019354, -0.0035014, -0.0021180, -0.0009593, 0.0011702
4: 0.0008095, 0.0014720, 0.0008872, 0.0014754, -0.0004976, 0.0004079
5: 0.0007896, 0.0050943, 0.0012941, 0.0051168, -0.0032337, 0.0026508
6: 0.0002478, 0.0013404, 0.0002421, 0.0012124, -0.0006728, 0.0008208
7: -0.0024964, 0.0003305, -0.0025111, -0.0000008, -0.0017407, 0.0021236
8: -0.0008770, 0.0006096, -0.0008847, 0.0004354, -0.0009154, 0.0011168
9: -0.0025708, -0.0008470, -0.0023687, -0.0008380, -0.0012949, 0.0010615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010197, upper bound: 0.0010268
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010238, upper bound: 0.0010985
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937649, 0.9963568, 0.9937649, 0.9963568, -0.0017035, 0.0017035
1: -0.0028176, -0.0021717, -0.0028176, -0.0021717, -0.0004245, 0.0004245
2: 0.0014551, 0.0048777, 0.0014551, 0.0048777, -0.0022494, 0.0022494
3: -0.0034932, -0.0019354, -0.0034932, -0.0019354, -0.0010239, 0.0010239
4: 0.0008095, 0.0014720, 0.0008095, 0.0014720, -0.0004354, 0.0004354
5: 0.0007896, 0.0050943, 0.0007896, 0.0050943, -0.0028292, 0.0028292
6: 0.0002478, 0.0013404, 0.0002478, 0.0013404, -0.0007181, 0.0007181
7: -0.0024964, 0.0003305, -0.0024964, 0.0003305, -0.0018579, 0.0018579
8: -0.0008770, 0.0006096, -0.0008770, 0.0006096, -0.0009771, 0.0009771
9: -0.0025708, -0.0008470, -0.0025708, -0.0008470, -0.0011329, 0.0011329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010197, upper bound: 0.0010287
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010238, upper bound: 0.0011019
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010214, upper bound: 0.0009644
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010254, upper bound: 0.0010238
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010214, upper bound: 0.0009644
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010254, upper bound: 0.0010238
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010197, upper bound: 0.0010268
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010238, upper bound: 0.0010985
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010197, upper bound: 0.0010287
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -0.0010238, upper bound: 0.0011019

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937435, 0.9958508, 0.9937530, 0.9959894, -0.0014665, 0.0013471
1: -0.0028229, -0.0022978, -0.0028205, -0.0022633, -0.0003654, 0.0003357
2: 0.0021232, 0.0049060, 0.0019403, 0.0048933, -0.0017789, 0.0019365
3: -0.0035061, -0.0022395, -0.0035003, -0.0021562, -0.0008814, 0.0008097
4: 0.0009388, 0.0014774, 0.0009034, 0.0014750, -0.0003443, 0.0003748
5: 0.0016299, 0.0051300, 0.0013998, 0.0051139, -0.0022374, 0.0024356
6: 0.0002388, 0.0011271, 0.0002429, 0.0011856, -0.0006182, 0.0005679
7: -0.0025198, -0.0002214, -0.0025093, -0.0000702, -0.0015994, 0.0014693
8: -0.0008893, 0.0003194, -0.0008837, 0.0003989, -0.0008411, 0.0007727
9: -0.0022343, -0.0008327, -0.0023264, -0.0008391, -0.0008959, 0.0009753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0009649
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0009649
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9960003, 0.9937516, 0.9960466, -0.0015822, 0.0013658
1: -0.0028205, -0.0022606, -0.0028209, -0.0022490, -0.0003942, 0.0003403
2: 0.0019258, 0.0048929, 0.0018647, 0.0048953, -0.0018035, 0.0020893
3: -0.0035002, -0.0021497, -0.0035012, -0.0021219, -0.0009509, 0.0008209
4: 0.0009006, 0.0014749, 0.0008888, 0.0014754, -0.0003491, 0.0004044
5: 0.0013816, 0.0051135, 0.0013048, 0.0051164, -0.0022683, 0.0026277
6: 0.0002430, 0.0011902, 0.0002422, 0.0012097, -0.0006669, 0.0005757
7: -0.0025090, -0.0000583, -0.0025109, -0.0000079, -0.0017256, 0.0014896
8: -0.0008836, 0.0004052, -0.0008846, 0.0004317, -0.0009075, 0.0007833
9: -0.0023337, -0.0008393, -0.0023644, -0.0008381, -0.0009083, 0.0010523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0010214
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0010254
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937435, 0.9958508, 0.9937670, 0.9962959, -0.0018420, 0.0013557
1: -0.0028229, -0.0022978, -0.0028171, -0.0021869, -0.0004590, 0.0003378
2: 0.0021232, 0.0049060, 0.0015354, 0.0048750, -0.0017902, 0.0024324
3: -0.0035061, -0.0022395, -0.0034920, -0.0019720, -0.0011071, 0.0008148
4: 0.0009388, 0.0014774, 0.0008251, 0.0014714, -0.0003465, 0.0004708
5: 0.0016299, 0.0051300, 0.0008906, 0.0050909, -0.0022517, 0.0030593
6: 0.0002388, 0.0011271, 0.0002487, 0.0013148, -0.0007765, 0.0005715
7: -0.0025198, -0.0002214, -0.0024941, 0.0002641, -0.0020090, 0.0014786
8: -0.0008893, 0.0003194, -0.0008758, 0.0005747, -0.0010565, 0.0007776
9: -0.0022343, -0.0008327, -0.0025303, -0.0008483, -0.0009017, 0.0012251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0009644
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0009644
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9960003, 0.9937652, 0.9963498, -0.0019401, 0.0014249
1: -0.0028205, -0.0022606, -0.0028175, -0.0021735, -0.0004834, 0.0003550
2: 0.0019258, 0.0048929, 0.0014644, 0.0048773, -0.0018816, 0.0025619
3: -0.0035002, -0.0021497, -0.0034931, -0.0019397, -0.0011661, 0.0008564
4: 0.0009006, 0.0014749, 0.0008113, 0.0014719, -0.0003642, 0.0004958
5: 0.0013816, 0.0051135, 0.0008013, 0.0050939, -0.0023665, 0.0032222
6: 0.0002430, 0.0011902, 0.0002480, 0.0013375, -0.0008178, 0.0006006
7: -0.0025090, -0.0000583, -0.0024961, 0.0003228, -0.0021160, 0.0015541
8: -0.0008836, 0.0004052, -0.0008768, 0.0006056, -0.0011128, 0.0008173
9: -0.0023337, -0.0008393, -0.0025661, -0.0008471, -0.0009477, 0.0012903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0010198
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0010237
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9961596, 0.9937530, 0.9959894, -0.0015315, 0.0017527
1: -0.0028204, -0.0022209, -0.0028205, -0.0022633, -0.0003816, 0.0004367
2: 0.0017156, 0.0048928, 0.0019403, 0.0048933, -0.0023145, 0.0020223
3: -0.0035001, -0.0020540, -0.0035003, -0.0021562, -0.0009204, 0.0010535
4: 0.0008599, 0.0014749, 0.0009034, 0.0014750, -0.0004480, 0.0003914
5: 0.0011172, 0.0051134, 0.0013998, 0.0051139, -0.0029110, 0.0025435
6: 0.0002430, 0.0012573, 0.0002429, 0.0011856, -0.0006456, 0.0007388
7: -0.0025089, 0.0001153, -0.0025093, -0.0000702, -0.0016703, 0.0019116
8: -0.0008835, 0.0004965, -0.0008837, 0.0003989, -0.0008784, 0.0010053
9: -0.0024396, -0.0008393, -0.0023264, -0.0008391, -0.0011657, 0.0010185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010267
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010268
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937674, 0.9962958, 0.9937516, 0.9960466, -0.0015905, 0.0017639
1: -0.0028170, -0.0021869, -0.0028209, -0.0022490, -0.0003963, 0.0004395
2: 0.0015356, 0.0048744, 0.0018647, 0.0048953, -0.0023292, 0.0021002
3: -0.0034917, -0.0019721, -0.0035012, -0.0021219, -0.0009559, 0.0010602
4: 0.0008251, 0.0014713, 0.0008888, 0.0014754, -0.0004508, 0.0004065
5: 0.0008909, 0.0050902, 0.0013048, 0.0051164, -0.0029296, 0.0026415
6: 0.0002489, 0.0013147, 0.0002422, 0.0012097, -0.0006704, 0.0007436
7: -0.0024937, 0.0002639, -0.0025109, -0.0000079, -0.0017346, 0.0019238
8: -0.0008755, 0.0005747, -0.0008846, 0.0004317, -0.0009122, 0.0010117
9: -0.0025302, -0.0008486, -0.0023644, -0.0008381, -0.0011731, 0.0010578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010964
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010985
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9961596, 0.9937670, 0.9962959, -0.0015867, 0.0014665
1: -0.0028204, -0.0022209, -0.0028171, -0.0021869, -0.0003954, 0.0003654
2: 0.0017156, 0.0048928, 0.0015354, 0.0048750, -0.0019365, 0.0020953
3: -0.0035001, -0.0020540, -0.0034920, -0.0019720, -0.0009537, 0.0008814
4: 0.0008599, 0.0014749, 0.0008251, 0.0014714, -0.0003748, 0.0004055
5: 0.0011172, 0.0051134, 0.0008906, 0.0050909, -0.0024356, 0.0026353
6: 0.0002430, 0.0012573, 0.0002487, 0.0013148, -0.0006689, 0.0006182
7: -0.0025089, 0.0001153, -0.0024941, 0.0002641, -0.0017306, 0.0015994
8: -0.0008835, 0.0004965, -0.0008758, 0.0005747, -0.0009101, 0.0008411
9: -0.0024396, -0.0008393, -0.0025303, -0.0008483, -0.0009753, 0.0010553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0010287
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0010287
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937674, 0.9962958, 0.9937652, 0.9963498, -0.0016990, 0.0014848
1: -0.0028170, -0.0021869, -0.0028175, -0.0021735, -0.0004234, 0.0003700
2: 0.0015356, 0.0048744, 0.0014644, 0.0048773, -0.0019607, 0.0022436
3: -0.0034917, -0.0019721, -0.0034931, -0.0019397, -0.0010212, 0.0008924
4: 0.0008251, 0.0014713, 0.0008113, 0.0014719, -0.0003795, 0.0004342
5: 0.0008909, 0.0050902, 0.0008013, 0.0050939, -0.0024660, 0.0028218
6: 0.0002489, 0.0013147, 0.0002480, 0.0013375, -0.0007162, 0.0006259
7: -0.0024937, 0.0002639, -0.0024961, 0.0003228, -0.0018531, 0.0016194
8: -0.0008755, 0.0005747, -0.0008768, 0.0006056, -0.0009745, 0.0008516
9: -0.0025302, -0.0008486, -0.0025661, -0.0008471, -0.0009875, 0.0011300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0010984
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0011019
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0009649
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0009649
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0010214
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009649, upper bound: 0.0010254
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0009644
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0009644
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0010198
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0010267, upper bound: 0.0010237
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010267
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010268
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010964
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009644, upper bound: 0.0010985
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0010287
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0010287
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0010984
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -0.0009725, upper bound: 0.0011019

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937435, 0.9958508, 0.9937435, 0.9958508, -0.0013085, 0.0013085
1: -0.0028229, -0.0022978, -0.0028229, -0.0022978, -0.0003261, 0.0003261
2: 0.0021232, 0.0049060, 0.0021232, 0.0049060, -0.0017279, 0.0017279
3: -0.0035061, -0.0022395, -0.0035061, -0.0022395, -0.0007865, 0.0007865
4: 0.0009388, 0.0014774, 0.0009388, 0.0014774, -0.0003344, 0.0003344
5: 0.0016299, 0.0051300, 0.0016299, 0.0051300, -0.0021733, 0.0021733
6: 0.0002388, 0.0011271, 0.0002388, 0.0011271, -0.0005516, 0.0005516
7: -0.0025198, -0.0002214, -0.0025198, -0.0002214, -0.0014271, 0.0014271
8: -0.0008893, 0.0003194, -0.0008893, 0.0003194, -0.0007505, 0.0007505
9: -0.0022343, -0.0008327, -0.0022343, -0.0008327, -0.0008703, 0.0008703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0008975
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009412, upper bound: 0.0009313
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937435, 0.9958508, 0.9937534, 0.9960003, -0.0015247, 0.0013470
1: -0.0028229, -0.0022978, -0.0028205, -0.0022606, -0.0003799, 0.0003356
2: 0.0021232, 0.0049060, 0.0019258, 0.0048929, -0.0017787, 0.0020134
3: -0.0035061, -0.0022395, -0.0035002, -0.0021497, -0.0009164, 0.0008096
4: 0.0009388, 0.0014774, 0.0009006, 0.0014749, -0.0003443, 0.0003897
5: 0.0016299, 0.0051300, 0.0013816, 0.0051135, -0.0022372, 0.0025323
6: 0.0002388, 0.0011271, 0.0002430, 0.0011902, -0.0006427, 0.0005678
7: -0.0025198, -0.0002214, -0.0025090, -0.0000583, -0.0016629, 0.0014691
8: -0.0008893, 0.0003194, -0.0008836, 0.0004052, -0.0008745, 0.0007726
9: -0.0022343, -0.0008327, -0.0023337, -0.0008393, -0.0008959, 0.0010140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0008975
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009412, upper bound: 0.0009312
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9960003, 0.9937435, 0.9958508, -0.0013470, 0.0015247
1: -0.0028205, -0.0022606, -0.0028229, -0.0022978, -0.0003356, 0.0003799
2: 0.0019258, 0.0048929, 0.0021232, 0.0049060, -0.0020134, 0.0017787
3: -0.0035002, -0.0021497, -0.0035061, -0.0022395, -0.0008096, 0.0009164
4: 0.0009006, 0.0014749, 0.0009388, 0.0014774, -0.0003897, 0.0003443
5: 0.0013816, 0.0051135, 0.0016299, 0.0051300, -0.0025323, 0.0022372
6: 0.0002430, 0.0011902, 0.0002388, 0.0011271, -0.0005678, 0.0006427
7: -0.0025090, -0.0000583, -0.0025198, -0.0002214, -0.0014691, 0.0016629
8: -0.0008836, 0.0004052, -0.0008893, 0.0003194, -0.0007726, 0.0008745
9: -0.0023337, -0.0008393, -0.0022343, -0.0008327, -0.0010140, 0.0008959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0009614
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009313, upper bound: 0.0009871
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9960003, 0.9937534, 0.9960003, -0.0013648, 0.0013648
1: -0.0028205, -0.0022606, -0.0028205, -0.0022606, -0.0003401, 0.0003401
2: 0.0019258, 0.0048929, 0.0019258, 0.0048929, -0.0018022, 0.0018022
3: -0.0035002, -0.0021497, -0.0035002, -0.0021497, -0.0008203, 0.0008203
4: 0.0009006, 0.0014749, 0.0009006, 0.0014749, -0.0003488, 0.0003488
5: 0.0013816, 0.0051135, 0.0013816, 0.0051135, -0.0022667, 0.0022667
6: 0.0002430, 0.0011902, 0.0002430, 0.0011902, -0.0005753, 0.0005753
7: -0.0025090, -0.0000583, -0.0025090, -0.0000583, -0.0014885, 0.0014885
8: -0.0008836, 0.0004052, -0.0008836, 0.0004052, -0.0007828, 0.0007828
9: -0.0023337, -0.0008393, -0.0023337, -0.0008393, -0.0009077, 0.0009077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0009725
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009313, upper bound: 0.0009906
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937435, 0.9958508, 0.9937534, 0.9961596, -0.0017141, 0.0013735
1: -0.0028229, -0.0022978, -0.0028204, -0.0022209, -0.0004271, 0.0003422
2: 0.0021232, 0.0049060, 0.0017156, 0.0048928, -0.0018137, 0.0022635
3: -0.0035061, -0.0022395, -0.0035001, -0.0020540, -0.0010302, 0.0008255
4: 0.0009388, 0.0014774, 0.0008599, 0.0014749, -0.0003510, 0.0004381
5: 0.0016299, 0.0051300, 0.0011172, 0.0051134, -0.0022811, 0.0028469
6: 0.0002388, 0.0011271, 0.0002430, 0.0012573, -0.0007226, 0.0005790
7: -0.0025198, -0.0002214, -0.0025089, 0.0001153, -0.0018695, 0.0014980
8: -0.0008893, 0.0003194, -0.0008835, 0.0004965, -0.0009832, 0.0007878
9: -0.0022343, -0.0008327, -0.0024396, -0.0008393, -0.0009135, 0.0011400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010030, upper bound: 0.0008974
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0009304
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937435, 0.9958508, 0.9937674, 0.9962958, -0.0018610, 0.0013553
1: -0.0028229, -0.0022978, -0.0028170, -0.0021869, -0.0004637, 0.0003377
2: 0.0021232, 0.0049060, 0.0015356, 0.0048744, -0.0017897, 0.0024574
3: -0.0035061, -0.0022395, -0.0034917, -0.0019721, -0.0011185, 0.0008146
4: 0.0009388, 0.0014774, 0.0008251, 0.0014713, -0.0003464, 0.0004756
5: 0.0016299, 0.0051300, 0.0008909, 0.0050902, -0.0022509, 0.0030907
6: 0.0002388, 0.0011271, 0.0002489, 0.0013147, -0.0007845, 0.0005713
7: -0.0025198, -0.0002214, -0.0024937, 0.0002639, -0.0020296, 0.0014782
8: -0.0008893, 0.0003194, -0.0008755, 0.0005747, -0.0010674, 0.0007773
9: -0.0022343, -0.0008327, -0.0025302, -0.0008486, -0.0009014, 0.0012377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010030, upper bound: 0.0008974
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0009304
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9960003, 0.9937534, 0.9961596, -0.0017526, 0.0015897
1: -0.0028205, -0.0022606, -0.0028204, -0.0022209, -0.0004367, 0.0003961
2: 0.0019258, 0.0048929, 0.0017156, 0.0048928, -0.0020991, 0.0023143
3: -0.0035002, -0.0021497, -0.0035001, -0.0020540, -0.0010534, 0.0009554
4: 0.0009006, 0.0014749, 0.0008599, 0.0014749, -0.0004063, 0.0004479
5: 0.0013816, 0.0051135, 0.0011172, 0.0051134, -0.0026401, 0.0029108
6: 0.0002430, 0.0011902, 0.0002430, 0.0012573, -0.0007388, 0.0006701
7: -0.0025090, -0.0000583, -0.0025089, 0.0001153, -0.0019115, 0.0017337
8: -0.0008836, 0.0004052, -0.0008835, 0.0004965, -0.0010052, 0.0009118
9: -0.0023337, -0.0008393, -0.0024396, -0.0008393, -0.0010572, 0.0011656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009819, upper bound: 0.0009606
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009948, upper bound: 0.0009848
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9960003, 0.9937674, 0.9962958, -0.0017629, 0.0014227
1: -0.0028205, -0.0022606, -0.0028170, -0.0021869, -0.0004393, 0.0003545
2: 0.0019258, 0.0048929, 0.0015356, 0.0048744, -0.0018787, 0.0023280
3: -0.0035002, -0.0021497, -0.0034917, -0.0019721, -0.0010596, 0.0008551
4: 0.0009006, 0.0014749, 0.0008251, 0.0014713, -0.0003636, 0.0004506
5: 0.0013816, 0.0051135, 0.0008909, 0.0050902, -0.0023629, 0.0029280
6: 0.0002430, 0.0011902, 0.0002489, 0.0013147, -0.0007431, 0.0005997
7: -0.0025090, -0.0000583, -0.0024937, 0.0002639, -0.0019227, 0.0015517
8: -0.0008836, 0.0004052, -0.0008755, 0.0005747, -0.0010112, 0.0008160
9: -0.0023337, -0.0008393, -0.0025302, -0.0008486, -0.0009462, 0.0011725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009819, upper bound: 0.0009724
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009948, upper bound: 0.0009886
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9961596, 0.9937435, 0.9958508, -0.0013735, 0.0017141
1: -0.0028204, -0.0022209, -0.0028229, -0.0022978, -0.0003422, 0.0004271
2: 0.0017156, 0.0048928, 0.0021232, 0.0049060, -0.0022635, 0.0018137
3: -0.0035001, -0.0020540, -0.0035061, -0.0022395, -0.0008255, 0.0010302
4: 0.0008599, 0.0014749, 0.0009388, 0.0014774, -0.0004381, 0.0003510
5: 0.0011172, 0.0051134, 0.0016299, 0.0051300, -0.0028469, 0.0022811
6: 0.0002430, 0.0012573, 0.0002388, 0.0011271, -0.0005790, 0.0007226
7: -0.0025089, 0.0001153, -0.0025198, -0.0002214, -0.0014980, 0.0018695
8: -0.0008835, 0.0004965, -0.0008893, 0.0003194, -0.0007878, 0.0009832
9: -0.0024396, -0.0008393, -0.0022343, -0.0008327, -0.0011400, 0.0009135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009076, upper bound: 0.0009843
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009404, upper bound: 0.0009948
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9961596, 0.9937534, 0.9960003, -0.0015897, 0.0017526
1: -0.0028204, -0.0022209, -0.0028205, -0.0022606, -0.0003961, 0.0004367
2: 0.0017156, 0.0048928, 0.0019258, 0.0048929, -0.0023143, 0.0020991
3: -0.0035001, -0.0020540, -0.0035002, -0.0021497, -0.0009554, 0.0010534
4: 0.0008599, 0.0014749, 0.0009006, 0.0014749, -0.0004479, 0.0004063
5: 0.0011172, 0.0051134, 0.0013816, 0.0051135, -0.0029108, 0.0026401
6: 0.0002430, 0.0012573, 0.0002430, 0.0011902, -0.0006701, 0.0007388
7: -0.0025089, 0.0001153, -0.0025090, -0.0000583, -0.0017337, 0.0019115
8: -0.0008835, 0.0004965, -0.0008836, 0.0004052, -0.0009118, 0.0010052
9: -0.0024396, -0.0008393, -0.0023337, -0.0008393, -0.0011656, 0.0010572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009076, upper bound: 0.0009843
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009404, upper bound: 0.0009948
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937674, 0.9962958, 0.9937435, 0.9958508, -0.0013553, 0.0018610
1: -0.0028170, -0.0021869, -0.0028229, -0.0022978, -0.0003377, 0.0004637
2: 0.0015356, 0.0048744, 0.0021232, 0.0049060, -0.0024574, 0.0017897
3: -0.0034917, -0.0019721, -0.0035061, -0.0022395, -0.0008146, 0.0011185
4: 0.0008251, 0.0014713, 0.0009388, 0.0014774, -0.0004756, 0.0003464
5: 0.0008909, 0.0050902, 0.0016299, 0.0051300, -0.0030907, 0.0022509
6: 0.0002489, 0.0013147, 0.0002388, 0.0011271, -0.0005713, 0.0007845
7: -0.0024937, 0.0002639, -0.0025198, -0.0002214, -0.0014782, 0.0020296
8: -0.0008755, 0.0005747, -0.0008893, 0.0003194, -0.0007773, 0.0010674
9: -0.0025302, -0.0008486, -0.0022343, -0.0008327, -0.0012377, 0.0009014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0010548
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009304, upper bound: 0.0010632
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937674, 0.9962958, 0.9937534, 0.9960003, -0.0014227, 0.0017629
1: -0.0028170, -0.0021869, -0.0028205, -0.0022606, -0.0003545, 0.0004393
2: 0.0015356, 0.0048744, 0.0019258, 0.0048929, -0.0023280, 0.0018787
3: -0.0034917, -0.0019721, -0.0035002, -0.0021497, -0.0008551, 0.0010596
4: 0.0008251, 0.0014713, 0.0009006, 0.0014749, -0.0004506, 0.0003636
5: 0.0008909, 0.0050902, 0.0013816, 0.0051135, -0.0029280, 0.0023629
6: 0.0002489, 0.0013147, 0.0002430, 0.0011902, -0.0005997, 0.0007431
7: -0.0024937, 0.0002639, -0.0025090, -0.0000583, -0.0015517, 0.0019227
8: -0.0008755, 0.0005747, -0.0008836, 0.0004052, -0.0008160, 0.0010112
9: -0.0025302, -0.0008486, -0.0023337, -0.0008393, -0.0011725, 0.0009462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0010584
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009304, upper bound: 0.0010653
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9961596, 0.9937534, 0.9961596, -0.0014310, 0.0014310
1: -0.0028204, -0.0022209, -0.0028204, -0.0022209, -0.0003566, 0.0003566
2: 0.0017156, 0.0048928, 0.0017156, 0.0048928, -0.0018896, 0.0018896
3: -0.0035001, -0.0020540, -0.0035001, -0.0020540, -0.0008601, 0.0008601
4: 0.0008599, 0.0014749, 0.0008599, 0.0014749, -0.0003657, 0.0003657
5: 0.0011172, 0.0051134, 0.0011172, 0.0051134, -0.0023766, 0.0023766
6: 0.0002430, 0.0012573, 0.0002430, 0.0012573, -0.0006032, 0.0006032
7: -0.0025089, 0.0001153, -0.0025089, 0.0001153, -0.0015607, 0.0015607
8: -0.0008835, 0.0004965, -0.0008835, 0.0004965, -0.0008207, 0.0008207
9: -0.0024396, -0.0008393, -0.0024396, -0.0008393, -0.0009517, 0.0009517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009435, upper bound: 0.0009902
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009496, upper bound: 0.0009965
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937534, 0.9961596, 0.9937674, 0.9962958, -0.0016439, 0.0014663
1: -0.0028204, -0.0022209, -0.0028170, -0.0021869, -0.0004096, 0.0003654
2: 0.0017156, 0.0048928, 0.0015356, 0.0048744, -0.0019362, 0.0021708
3: -0.0035001, -0.0020540, -0.0034917, -0.0019721, -0.0009880, 0.0008813
4: 0.0008599, 0.0014749, 0.0008251, 0.0014713, -0.0003748, 0.0004201
5: 0.0011172, 0.0051134, 0.0008909, 0.0050902, -0.0024353, 0.0027302
6: 0.0002430, 0.0012573, 0.0002489, 0.0013147, -0.0006930, 0.0006181
7: -0.0025089, 0.0001153, -0.0024937, 0.0002639, -0.0017929, 0.0015992
8: -0.0008835, 0.0004965, -0.0008755, 0.0005747, -0.0009429, 0.0008410
9: -0.0024396, -0.0008393, -0.0025302, -0.0008486, -0.0009752, 0.0010933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009435, upper bound: 0.0009902
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009496, upper bound: 0.0009965
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937674, 0.9962958, 0.9937534, 0.9961596, -0.0014663, 0.0016439
1: -0.0028170, -0.0021869, -0.0028204, -0.0022209, -0.0003654, 0.0004096
2: 0.0015356, 0.0048744, 0.0017156, 0.0048928, -0.0021708, 0.0019362
3: -0.0034917, -0.0019721, -0.0035001, -0.0020540, -0.0008813, 0.0009880
4: 0.0008251, 0.0014713, 0.0008599, 0.0014749, -0.0004201, 0.0003748
5: 0.0008909, 0.0050902, 0.0011172, 0.0051134, -0.0027302, 0.0024353
6: 0.0002489, 0.0013147, 0.0002430, 0.0012573, -0.0006181, 0.0006930
7: -0.0024937, 0.0002639, -0.0025089, 0.0001153, -0.0015992, 0.0017929
8: -0.0008755, 0.0005747, -0.0008835, 0.0004965, -0.0008410, 0.0009429
9: -0.0025302, -0.0008486, -0.0024396, -0.0008393, -0.0010933, 0.0009752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009320, upper bound: 0.0010594
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009384, upper bound: 0.0010652
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937674, 0.9962958, 0.9937674, 0.9962958, -0.0014838, 0.0014838
1: -0.0028170, -0.0021869, -0.0028170, -0.0021869, -0.0003697, 0.0003697
2: 0.0015356, 0.0048744, 0.0015356, 0.0048744, -0.0019594, 0.0019594
3: -0.0034917, -0.0019721, -0.0034917, -0.0019721, -0.0008918, 0.0008918
4: 0.0008251, 0.0014713, 0.0008251, 0.0014713, -0.0003792, 0.0003792
5: 0.0008909, 0.0050902, 0.0008909, 0.0050902, -0.0024644, 0.0024644
6: 0.0002489, 0.0013147, 0.0002489, 0.0013147, -0.0006255, 0.0006255
7: -0.0024937, 0.0002639, -0.0024937, 0.0002639, -0.0016183, 0.0016183
8: -0.0008755, 0.0005747, -0.0008755, 0.0005747, -0.0008511, 0.0008511
9: -0.0025302, -0.0008486, -0.0025302, -0.0008486, -0.0009869, 0.0009869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009320, upper bound: 0.0010646
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009384, upper bound: 0.0010686
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.46 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0008975
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009412, upper bound: 0.0009313
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0008975
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009412, upper bound: 0.0009312
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0009614
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009313, upper bound: 0.0009871
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0009725
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009313, upper bound: 0.0009906
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0010030, upper bound: 0.0008974
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0009304
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0010030, upper bound: 0.0008974
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0009304
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009819, upper bound: 0.0009606
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009948, upper bound: 0.0009848
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009819, upper bound: 0.0009724
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009948, upper bound: 0.0009886
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009076, upper bound: 0.0009843
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009404, upper bound: 0.0009948
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009076, upper bound: 0.0009843
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009404, upper bound: 0.0009948
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0010548
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009304, upper bound: 0.0010632
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009031, upper bound: 0.0010584
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009304, upper bound: 0.0010653
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009435, upper bound: 0.0009902
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009496, upper bound: 0.0009965
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009435, upper bound: 0.0009902
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009496, upper bound: 0.0009965
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009320, upper bound: 0.0010594
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009384, upper bound: 0.0010652
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009320, upper bound: 0.0010646
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -0.0009384, upper bound: 0.0010686

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9937764, 0.9958490, -0.0012010, 0.0012963
1: -0.0027994, -0.0022899, -0.0028147, -0.0022983, -0.0002993, 0.0003230
2: 0.0020812, 0.0047811, 0.0021258, 0.0048625, -0.0017117, 0.0015859
3: -0.0034493, -0.0022204, -0.0034863, -0.0022407, -0.0007218, 0.0007791
4: 0.0009307, 0.0014533, 0.0009393, 0.0014690, -0.0003313, 0.0003069
5: 0.0015771, 0.0049728, 0.0016331, 0.0050752, -0.0021529, 0.0019946
6: 0.0002787, 0.0011406, 0.0002527, 0.0011263, -0.0005063, 0.0005464
7: -0.0024166, -0.0001867, -0.0024838, -0.0002235, -0.0013098, 0.0014138
8: -0.0008350, 0.0003377, -0.0008704, 0.0003183, -0.0006888, 0.0007435
9: -0.0022554, -0.0008956, -0.0022330, -0.0008546, -0.0008621, 0.0007987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009004
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009004
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9937465, 0.9958507, -0.0012563, 0.0013026
1: -0.0028108, -0.0022985, -0.0028222, -0.0022979, -0.0003130, 0.0003246
2: 0.0021269, 0.0048415, 0.0021235, 0.0049020, -0.0017200, 0.0016589
3: -0.0034768, -0.0022412, -0.0035043, -0.0022396, -0.0007551, 0.0007829
4: 0.0009395, 0.0014650, 0.0009389, 0.0014767, -0.0003329, 0.0003211
5: 0.0016345, 0.0050488, 0.0016302, 0.0051248, -0.0021633, 0.0020864
6: 0.0002594, 0.0011260, 0.0002401, 0.0011271, -0.0005296, 0.0005491
7: -0.0024665, -0.0002244, -0.0025164, -0.0002216, -0.0013701, 0.0014206
8: -0.0008613, 0.0003179, -0.0008875, 0.0003193, -0.0007205, 0.0007471
9: -0.0022324, -0.0008652, -0.0022341, -0.0008347, -0.0008663, 0.0008355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009076
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009412
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9937857, 0.9959982, -0.0014170, 0.0013343
1: -0.0027994, -0.0022899, -0.0028124, -0.0022611, -0.0003531, 0.0003325
2: 0.0020812, 0.0047811, 0.0019285, 0.0048501, -0.0017619, 0.0018711
3: -0.0034493, -0.0022204, -0.0034807, -0.0021509, -0.0008517, 0.0008020
4: 0.0009307, 0.0014533, 0.0009011, 0.0014666, -0.0003410, 0.0003622
5: 0.0015771, 0.0049728, 0.0013850, 0.0050597, -0.0022160, 0.0023534
6: 0.0002787, 0.0011406, 0.0002566, 0.0011893, -0.0005973, 0.0005625
7: -0.0024166, -0.0001867, -0.0024736, -0.0000605, -0.0015454, 0.0014552
8: -0.0008350, 0.0003377, -0.0008650, 0.0004040, -0.0008127, 0.0007653
9: -0.0022554, -0.0008956, -0.0023323, -0.0008608, -0.0008874, 0.0009424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0008975
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0008975
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9937564, 0.9960002, -0.0014697, 0.0013412
1: -0.0028108, -0.0022985, -0.0028197, -0.0022606, -0.0003662, 0.0003342
2: 0.0021269, 0.0048415, 0.0019260, 0.0048889, -0.0017710, 0.0019407
3: -0.0034768, -0.0022412, -0.0034983, -0.0021498, -0.0008833, 0.0008061
4: 0.0009395, 0.0014650, 0.0009007, 0.0014741, -0.0003428, 0.0003756
5: 0.0016345, 0.0050488, 0.0013819, 0.0051084, -0.0022275, 0.0024409
6: 0.0002594, 0.0011260, 0.0002443, 0.0011901, -0.0006195, 0.0005654
7: -0.0024665, -0.0002244, -0.0025056, -0.0000585, -0.0016029, 0.0014627
8: -0.0008613, 0.0003179, -0.0008818, 0.0004051, -0.0008429, 0.0007692
9: -0.0022324, -0.0008652, -0.0023336, -0.0008413, -0.0008920, 0.0009774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0009031
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0009313
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9937764, 0.9958490, -0.0012363, 0.0015088
1: -0.0027977, -0.0022522, -0.0028147, -0.0022983, -0.0003081, 0.0003760
2: 0.0018816, 0.0047721, 0.0021258, 0.0048625, -0.0019924, 0.0016326
3: -0.0034452, -0.0021295, -0.0034863, -0.0022407, -0.0007431, 0.0009069
4: 0.0008921, 0.0014515, 0.0009393, 0.0014690, -0.0003856, 0.0003160
5: 0.0013260, 0.0049615, 0.0016331, 0.0050752, -0.0025059, 0.0020534
6: 0.0002815, 0.0012043, 0.0002527, 0.0011263, -0.0005212, 0.0006360
7: -0.0024092, -0.0000218, -0.0024838, -0.0002235, -0.0013484, 0.0016456
8: -0.0008311, 0.0004244, -0.0008704, 0.0003183, -0.0007091, 0.0008654
9: -0.0023560, -0.0009001, -0.0022330, -0.0008546, -0.0010035, 0.0008223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009614
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009614
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9937465, 0.9958507, -0.0012934, 0.0015187
1: -0.0028086, -0.0022613, -0.0028222, -0.0022979, -0.0003223, 0.0003784
2: 0.0019295, 0.0048300, 0.0021235, 0.0049020, -0.0020054, 0.0017079
3: -0.0034715, -0.0021513, -0.0035043, -0.0022396, -0.0007774, 0.0009128
4: 0.0009013, 0.0014627, 0.0009389, 0.0014767, -0.0003881, 0.0003306
5: 0.0013862, 0.0050343, 0.0016302, 0.0051248, -0.0025223, 0.0021481
6: 0.0002631, 0.0011890, 0.0002401, 0.0011271, -0.0005452, 0.0006402
7: -0.0024570, -0.0000613, -0.0025164, -0.0002216, -0.0014106, 0.0016563
8: -0.0008563, 0.0004036, -0.0008875, 0.0003193, -0.0007418, 0.0008711
9: -0.0023318, -0.0008710, -0.0022341, -0.0008347, -0.0010100, 0.0008602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009659
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009871
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9937857, 0.9959982, -0.0012559, 0.0013525
1: -0.0027977, -0.0022522, -0.0028124, -0.0022611, -0.0003129, 0.0003370
2: 0.0018816, 0.0047721, 0.0019285, 0.0048501, -0.0017859, 0.0016584
3: -0.0034452, -0.0021295, -0.0034807, -0.0021509, -0.0007549, 0.0008129
4: 0.0008921, 0.0014515, 0.0009011, 0.0014666, -0.0003457, 0.0003210
5: 0.0013260, 0.0049615, 0.0013850, 0.0050597, -0.0022462, 0.0020859
6: 0.0002815, 0.0012043, 0.0002566, 0.0011893, -0.0005294, 0.0005701
7: -0.0024092, -0.0000218, -0.0024736, -0.0000605, -0.0013698, 0.0014751
8: -0.0008311, 0.0004244, -0.0008650, 0.0004040, -0.0007203, 0.0007757
9: -0.0023560, -0.0009001, -0.0023323, -0.0008608, -0.0008995, 0.0008353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009725
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009724
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9937564, 0.9960002, -0.0013111, 0.0013591
1: -0.0028086, -0.0022613, -0.0028197, -0.0022606, -0.0003267, 0.0003386
2: 0.0019295, 0.0048300, 0.0019260, 0.0048889, -0.0017947, 0.0017313
3: -0.0034715, -0.0021513, -0.0034983, -0.0021498, -0.0007880, 0.0008168
4: 0.0009013, 0.0014627, 0.0009007, 0.0014741, -0.0003474, 0.0003351
5: 0.0013862, 0.0050343, 0.0013819, 0.0051084, -0.0022572, 0.0021775
6: 0.0002631, 0.0011890, 0.0002443, 0.0011901, -0.0005527, 0.0005729
7: -0.0024570, -0.0000613, -0.0025056, -0.0000585, -0.0014299, 0.0014823
8: -0.0008563, 0.0004036, -0.0008818, 0.0004051, -0.0007520, 0.0007795
9: -0.0023318, -0.0008710, -0.0023336, -0.0008413, -0.0009039, 0.0008720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009781
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009906
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9937866, 0.9961576, -0.0016066, 0.0013658
1: -0.0027994, -0.0022899, -0.0028122, -0.0022214, -0.0004003, 0.0003403
2: 0.0020812, 0.0047811, 0.0017182, 0.0048489, -0.0018036, 0.0021215
3: -0.0034493, -0.0022204, -0.0034801, -0.0020552, -0.0009656, 0.0008209
4: 0.0009307, 0.0014533, 0.0008604, 0.0014664, -0.0003491, 0.0004106
5: 0.0015771, 0.0049728, 0.0011205, 0.0050581, -0.0022684, 0.0026683
6: 0.0002787, 0.0011406, 0.0002570, 0.0012564, -0.0006773, 0.0005758
7: -0.0024166, -0.0001867, -0.0024726, 0.0001132, -0.0017523, 0.0014897
8: -0.0008350, 0.0003377, -0.0008645, 0.0004954, -0.0009215, 0.0007834
9: -0.0022554, -0.0008956, -0.0024383, -0.0008615, -0.0009084, 0.0010685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010029, upper bound: 0.0009004
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010029, upper bound: 0.0009004
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9937562, 0.9961593, -0.0016645, 0.0013677
1: -0.0028108, -0.0022985, -0.0028197, -0.0022209, -0.0004148, 0.0003408
2: 0.0021269, 0.0048415, 0.0017158, 0.0048892, -0.0018060, 0.0021980
3: -0.0034768, -0.0022412, -0.0034985, -0.0020541, -0.0010004, 0.0008220
4: 0.0009395, 0.0014650, 0.0008600, 0.0014742, -0.0003496, 0.0004254
5: 0.0016345, 0.0050488, 0.0011175, 0.0051088, -0.0022715, 0.0027645
6: 0.0002594, 0.0011260, 0.0002442, 0.0012572, -0.0007017, 0.0005765
7: -0.0024665, -0.0002244, -0.0025059, 0.0001151, -0.0018154, 0.0014917
8: -0.0008613, 0.0003179, -0.0008820, 0.0004964, -0.0009547, 0.0007844
9: -0.0022324, -0.0008652, -0.0024394, -0.0008412, -0.0009096, 0.0011070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0009076
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0009404
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938007, 0.9962938, -0.0017534, 0.0013472
1: -0.0027994, -0.0022899, -0.0028086, -0.0021874, -0.0004369, 0.0003357
2: 0.0020812, 0.0047811, 0.0015383, 0.0048304, -0.0017789, 0.0023153
3: -0.0034493, -0.0022204, -0.0034717, -0.0019733, -0.0010538, 0.0008097
4: 0.0009307, 0.0014533, 0.0008256, 0.0014628, -0.0003443, 0.0004481
5: 0.0015771, 0.0049728, 0.0008942, 0.0050348, -0.0022374, 0.0029121
6: 0.0002787, 0.0011406, 0.0002630, 0.0013139, -0.0007391, 0.0005679
7: -0.0024166, -0.0001867, -0.0024573, 0.0002618, -0.0019123, 0.0014693
8: -0.0008350, 0.0003377, -0.0008564, 0.0005735, -0.0010057, 0.0007727
9: -0.0022554, -0.0008956, -0.0025289, -0.0008708, -0.0008959, 0.0011661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0008974
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0008974
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9937702, 0.9962957, -0.0018085, 0.0013495
1: -0.0028108, -0.0022985, -0.0028163, -0.0021870, -0.0004506, 0.0003363
2: 0.0021269, 0.0048415, 0.0015359, 0.0048707, -0.0017820, 0.0023880
3: -0.0034768, -0.0022412, -0.0034900, -0.0019722, -0.0010869, 0.0008111
4: 0.0009395, 0.0014650, 0.0008252, 0.0014706, -0.0003449, 0.0004622
5: 0.0016345, 0.0050488, 0.0008912, 0.0050854, -0.0022413, 0.0030035
6: 0.0002594, 0.0011260, 0.0002501, 0.0013146, -0.0007623, 0.0005689
7: -0.0024665, -0.0002244, -0.0024906, 0.0002638, -0.0019724, 0.0014718
8: -0.0008613, 0.0003179, -0.0008739, 0.0005746, -0.0010373, 0.0007740
9: -0.0022324, -0.0008652, -0.0025301, -0.0008505, -0.0008975, 0.0012027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0009031
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0009304
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9937866, 0.9961576, -0.0016420, 0.0015784
1: -0.0027977, -0.0022522, -0.0028122, -0.0022214, -0.0004091, 0.0003933
2: 0.0018816, 0.0047721, 0.0017182, 0.0048489, -0.0020843, 0.0021682
3: -0.0034452, -0.0021295, -0.0034801, -0.0020552, -0.0009869, 0.0009487
4: 0.0008921, 0.0014515, 0.0008604, 0.0014664, -0.0004034, 0.0004197
5: 0.0013260, 0.0049615, 0.0011205, 0.0050581, -0.0026215, 0.0027271
6: 0.0002815, 0.0012043, 0.0002570, 0.0012564, -0.0006922, 0.0006654
7: -0.0024092, -0.0000218, -0.0024726, 0.0001132, -0.0017908, 0.0017215
8: -0.0008311, 0.0004244, -0.0008645, 0.0004954, -0.0009418, 0.0009053
9: -0.0023560, -0.0009001, -0.0024383, -0.0008615, -0.0010498, 0.0010920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009818, upper bound: 0.0009606
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009818, upper bound: 0.0009606
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9937562, 0.9961593, -0.0017017, 0.0015838
1: -0.0028086, -0.0022613, -0.0028197, -0.0022209, -0.0004240, 0.0003946
2: 0.0019295, 0.0048300, 0.0017158, 0.0048892, -0.0020914, 0.0022470
3: -0.0034715, -0.0021513, -0.0034985, -0.0020541, -0.0010227, 0.0009519
4: 0.0009013, 0.0014627, 0.0008600, 0.0014742, -0.0004048, 0.0004349
5: 0.0013862, 0.0050343, 0.0011175, 0.0051088, -0.0026304, 0.0028262
6: 0.0002631, 0.0011890, 0.0002442, 0.0012572, -0.0007173, 0.0006676
7: -0.0024570, -0.0000613, -0.0025059, 0.0001151, -0.0018559, 0.0017274
8: -0.0008563, 0.0004036, -0.0008820, 0.0004964, -0.0009760, 0.0009084
9: -0.0023318, -0.0008710, -0.0024394, -0.0008412, -0.0010533, 0.0011317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009843, upper bound: 0.0009653
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009843, upper bound: 0.0009848
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938007, 0.9962938, -0.0016542, 0.0014169
1: -0.0027977, -0.0022522, -0.0028086, -0.0021874, -0.0004122, 0.0003531
2: 0.0018816, 0.0047721, 0.0015383, 0.0048304, -0.0018710, 0.0021843
3: -0.0034452, -0.0021295, -0.0034717, -0.0019733, -0.0009942, 0.0008516
4: 0.0008921, 0.0014515, 0.0008256, 0.0014628, -0.0003621, 0.0004228
5: 0.0013260, 0.0049615, 0.0008942, 0.0050348, -0.0023533, 0.0027473
6: 0.0002815, 0.0012043, 0.0002630, 0.0013139, -0.0006973, 0.0005973
7: -0.0024092, -0.0000218, -0.0024573, 0.0002618, -0.0018041, 0.0015454
8: -0.0008311, 0.0004244, -0.0008564, 0.0005735, -0.0009488, 0.0008127
9: -0.0023560, -0.0009001, -0.0025289, -0.0008708, -0.0009423, 0.0011001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009913, upper bound: 0.0009724
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009913, upper bound: 0.0009724
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9937702, 0.9962957, -0.0017096, 0.0014169
1: -0.0028086, -0.0022613, -0.0028163, -0.0021870, -0.0004260, 0.0003531
2: 0.0019295, 0.0048300, 0.0015359, 0.0048707, -0.0018710, 0.0022576
3: -0.0034715, -0.0021513, -0.0034900, -0.0019722, -0.0010275, 0.0008516
4: 0.0009013, 0.0014627, 0.0008252, 0.0014706, -0.0003621, 0.0004369
5: 0.0013862, 0.0050343, 0.0008912, 0.0050854, -0.0023533, 0.0028394
6: 0.0002631, 0.0011890, 0.0002501, 0.0013146, -0.0007207, 0.0005973
7: -0.0024570, -0.0000613, -0.0024906, 0.0002638, -0.0018646, 0.0015454
8: -0.0008563, 0.0004036, -0.0008739, 0.0005746, -0.0009806, 0.0008127
9: -0.0023318, -0.0008710, -0.0025301, -0.0008505, -0.0009424, 0.0011370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0009769
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0009886
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9937764, 0.9958490, -0.0012721, 0.0017051
1: -0.0027951, -0.0022148, -0.0028147, -0.0022983, -0.0003170, 0.0004249
2: 0.0016835, 0.0047584, 0.0021258, 0.0048625, -0.0022516, 0.0016798
3: -0.0034389, -0.0020394, -0.0034863, -0.0022407, -0.0007646, 0.0010248
4: 0.0008537, 0.0014489, 0.0009393, 0.0014690, -0.0004358, 0.0003251
5: 0.0010769, 0.0049442, 0.0016331, 0.0050752, -0.0028319, 0.0021128
6: 0.0002859, 0.0012675, 0.0002527, 0.0011263, -0.0005363, 0.0007188
7: -0.0023978, 0.0001418, -0.0024838, -0.0002235, -0.0013875, 0.0018597
8: -0.0008251, 0.0005104, -0.0008704, 0.0003183, -0.0007296, 0.0009780
9: -0.0024557, -0.0009071, -0.0022330, -0.0008546, -0.0011340, 0.0008461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010029
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010052
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9937465, 0.9958507, -0.0013261, 0.0017084
1: -0.0028095, -0.0022216, -0.0028222, -0.0022979, -0.0003304, 0.0004257
2: 0.0017192, 0.0048349, 0.0021235, 0.0049020, -0.0022560, 0.0017511
3: -0.0034738, -0.0020556, -0.0035043, -0.0022396, -0.0007970, 0.0010268
4: 0.0008606, 0.0014637, 0.0009389, 0.0014767, -0.0004366, 0.0003389
5: 0.0011218, 0.0050405, 0.0016302, 0.0051248, -0.0028374, 0.0022025
6: 0.0002615, 0.0012561, 0.0002401, 0.0011271, -0.0005590, 0.0007202
7: -0.0024611, 0.0001123, -0.0025164, -0.0002216, -0.0014463, 0.0018633
8: -0.0008584, 0.0004949, -0.0008875, 0.0003193, -0.0007606, 0.0009799
9: -0.0024377, -0.0008685, -0.0022341, -0.0008347, -0.0011362, 0.0008820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010030
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010211
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9937857, 0.9959982, -0.0014881, 0.0017431
1: -0.0027951, -0.0022148, -0.0028124, -0.0022611, -0.0003708, 0.0004343
2: 0.0016835, 0.0047584, 0.0019285, 0.0048501, -0.0023018, 0.0019651
3: -0.0034389, -0.0020394, -0.0034807, -0.0021509, -0.0008944, 0.0010477
4: 0.0008537, 0.0014489, 0.0009011, 0.0014666, -0.0004455, 0.0003803
5: 0.0010769, 0.0049442, 0.0013850, 0.0050597, -0.0028951, 0.0024716
6: 0.0002859, 0.0012675, 0.0002566, 0.0011893, -0.0006273, 0.0007348
7: -0.0023978, 0.0001418, -0.0024736, -0.0000605, -0.0016230, 0.0019012
8: -0.0008251, 0.0005104, -0.0008650, 0.0004040, -0.0008535, 0.0009998
9: -0.0024557, -0.0009071, -0.0023323, -0.0008608, -0.0011593, 0.0009897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009818
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009843
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9937564, 0.9960002, -0.0015396, 0.0017470
1: -0.0028095, -0.0022216, -0.0028197, -0.0022606, -0.0003836, 0.0004353
2: 0.0017192, 0.0048349, 0.0019260, 0.0048889, -0.0023069, 0.0020330
3: -0.0034738, -0.0020556, -0.0034983, -0.0021498, -0.0009253, 0.0010500
4: 0.0008606, 0.0014637, 0.0009007, 0.0014741, -0.0004465, 0.0003935
5: 0.0011218, 0.0050405, 0.0013819, 0.0051084, -0.0029015, 0.0025569
6: 0.0002615, 0.0012561, 0.0002443, 0.0011901, -0.0006490, 0.0007364
7: -0.0024611, 0.0001123, -0.0025056, -0.0000585, -0.0016791, 0.0019054
8: -0.0008584, 0.0004949, -0.0008818, 0.0004051, -0.0008830, 0.0010020
9: -0.0024377, -0.0008685, -0.0023336, -0.0008413, -0.0011619, 0.0010239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009818
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009948
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9937764, 0.9958490, -0.0012506, 0.0018485
1: -0.0027919, -0.0021805, -0.0028147, -0.0022983, -0.0003116, 0.0004606
2: 0.0015012, 0.0047415, 0.0021258, 0.0048625, -0.0024409, 0.0016514
3: -0.0034313, -0.0019564, -0.0034863, -0.0022407, -0.0007516, 0.0011110
4: 0.0008185, 0.0014456, 0.0009393, 0.0014690, -0.0004724, 0.0003196
5: 0.0008476, 0.0049230, 0.0016331, 0.0050752, -0.0030701, 0.0020770
6: 0.0002913, 0.0013257, 0.0002527, 0.0011263, -0.0005272, 0.0007792
7: -0.0023839, 0.0002924, -0.0024838, -0.0002235, -0.0013640, 0.0020161
8: -0.0008178, 0.0005896, -0.0008704, 0.0003183, -0.0007173, 0.0010602
9: -0.0025475, -0.0009156, -0.0022330, -0.0008546, -0.0012294, 0.0008317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010548
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010548
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9937465, 0.9958507, -0.0013070, 0.0018552
1: -0.0028058, -0.0021876, -0.0028222, -0.0022979, -0.0003257, 0.0004623
2: 0.0015393, 0.0048152, 0.0021235, 0.0049020, -0.0024497, 0.0017258
3: -0.0034648, -0.0019738, -0.0035043, -0.0022396, -0.0007855, 0.0011150
4: 0.0008258, 0.0014599, 0.0009389, 0.0014767, -0.0004741, 0.0003340
5: 0.0008955, 0.0050157, 0.0016302, 0.0051248, -0.0030811, 0.0021706
6: 0.0002678, 0.0013135, 0.0002401, 0.0011271, -0.0005509, 0.0007820
7: -0.0024447, 0.0002609, -0.0025164, -0.0002216, -0.0014254, 0.0020233
8: -0.0008498, 0.0005731, -0.0008875, 0.0003193, -0.0007496, 0.0010640
9: -0.0025283, -0.0008785, -0.0022341, -0.0008347, -0.0012338, 0.0008692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010553
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010632
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9937857, 0.9959982, -0.0013241, 0.0017518
1: -0.0027919, -0.0021805, -0.0028124, -0.0022611, -0.0003299, 0.0004365
2: 0.0015012, 0.0047415, 0.0019285, 0.0048501, -0.0023133, 0.0017485
3: -0.0034313, -0.0019564, -0.0034807, -0.0021509, -0.0007958, 0.0010529
4: 0.0008185, 0.0014456, 0.0009011, 0.0014666, -0.0004477, 0.0003384
5: 0.0008476, 0.0049230, 0.0013850, 0.0050597, -0.0029095, 0.0021992
6: 0.0002913, 0.0013257, 0.0002566, 0.0011893, -0.0005582, 0.0007385
7: -0.0023839, 0.0002924, -0.0024736, -0.0000605, -0.0014442, 0.0019106
8: -0.0008178, 0.0005896, -0.0008650, 0.0004040, -0.0007595, 0.0010048
9: -0.0025475, -0.0009156, -0.0023323, -0.0008608, -0.0011651, 0.0008806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010584
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010583
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9937564, 0.9960002, -0.0013769, 0.0017573
1: -0.0028058, -0.0021876, -0.0028197, -0.0022606, -0.0003431, 0.0004379
2: 0.0015393, 0.0048152, 0.0019260, 0.0048889, -0.0023205, 0.0018182
3: -0.0034648, -0.0019738, -0.0034983, -0.0021498, -0.0008276, 0.0010562
4: 0.0008258, 0.0014599, 0.0009007, 0.0014741, -0.0004491, 0.0003519
5: 0.0008955, 0.0050157, 0.0013819, 0.0051084, -0.0029186, 0.0022869
6: 0.0002678, 0.0013135, 0.0002443, 0.0011901, -0.0005804, 0.0007408
7: -0.0024447, 0.0002609, -0.0025056, -0.0000585, -0.0015017, 0.0019166
8: -0.0008498, 0.0005731, -0.0008818, 0.0004051, -0.0007898, 0.0010079
9: -0.0025283, -0.0008785, -0.0023336, -0.0008413, -0.0011687, 0.0009158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010590
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010653
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9937866, 0.9961576, -0.0013116, 0.0014180
1: -0.0027951, -0.0022148, -0.0028122, -0.0022214, -0.0003268, 0.0003533
2: 0.0016835, 0.0047584, 0.0017182, 0.0048489, -0.0018724, 0.0017320
3: -0.0034389, -0.0020394, -0.0034801, -0.0020552, -0.0007883, 0.0008522
4: 0.0008537, 0.0014489, 0.0008604, 0.0014664, -0.0003624, 0.0003352
5: 0.0010769, 0.0049442, 0.0011205, 0.0050581, -0.0023550, 0.0021784
6: 0.0002859, 0.0012675, 0.0002570, 0.0012564, -0.0005529, 0.0005977
7: -0.0023978, 0.0001418, -0.0024726, 0.0001132, -0.0014305, 0.0015465
8: -0.0008251, 0.0005104, -0.0008645, 0.0004954, -0.0007523, 0.0008133
9: -0.0024557, -0.0009071, -0.0024383, -0.0008615, -0.0009431, 0.0008723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010136
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010143
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9937562, 0.9961593, -0.0013801, 0.0014251
1: -0.0028095, -0.0022216, -0.0028197, -0.0022209, -0.0003439, 0.0003551
2: 0.0017192, 0.0048349, 0.0017158, 0.0048892, -0.0018818, 0.0018224
3: -0.0034738, -0.0020556, -0.0034985, -0.0020541, -0.0008295, 0.0008565
4: 0.0008606, 0.0014637, 0.0008600, 0.0014742, -0.0003642, 0.0003527
5: 0.0011218, 0.0050405, 0.0011175, 0.0051088, -0.0023668, 0.0022920
6: 0.0002615, 0.0012561, 0.0002442, 0.0012572, -0.0005817, 0.0006007
7: -0.0024611, 0.0001123, -0.0025059, 0.0001151, -0.0015051, 0.0015543
8: -0.0008584, 0.0004949, -0.0008820, 0.0004964, -0.0007915, 0.0008174
9: -0.0024377, -0.0008685, -0.0024394, -0.0008412, -0.0009478, 0.0009178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010136
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010228
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938007, 0.9962938, -0.0015244, 0.0014532
1: -0.0027951, -0.0022148, -0.0028086, -0.0021874, -0.0003798, 0.0003621
2: 0.0016835, 0.0047584, 0.0015383, 0.0048304, -0.0019189, 0.0020130
3: -0.0034389, -0.0020394, -0.0034717, -0.0019733, -0.0009162, 0.0008734
4: 0.0008537, 0.0014489, 0.0008256, 0.0014628, -0.0003714, 0.0003896
5: 0.0010769, 0.0049442, 0.0008942, 0.0050348, -0.0024135, 0.0025318
6: 0.0002859, 0.0012675, 0.0002630, 0.0013139, -0.0006426, 0.0006126
7: -0.0023978, 0.0001418, -0.0024573, 0.0002618, -0.0016626, 0.0015849
8: -0.0008251, 0.0005104, -0.0008564, 0.0005735, -0.0008743, 0.0008335
9: -0.0024557, -0.0009071, -0.0025289, -0.0008708, -0.0009665, 0.0010139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009891
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009902
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9937702, 0.9962957, -0.0015924, 0.0014605
1: -0.0028095, -0.0022216, -0.0028163, -0.0021870, -0.0003968, 0.0003639
2: 0.0017192, 0.0048349, 0.0015359, 0.0048707, -0.0019285, 0.0021028
3: -0.0034738, -0.0020556, -0.0034900, -0.0019722, -0.0009571, 0.0008778
4: 0.0008606, 0.0014637, 0.0008252, 0.0014706, -0.0003733, 0.0004070
5: 0.0011218, 0.0050405, 0.0008912, 0.0050854, -0.0024256, 0.0026448
6: 0.0002615, 0.0012561, 0.0002501, 0.0013146, -0.0006713, 0.0006156
7: -0.0024611, 0.0001123, -0.0024906, 0.0002638, -0.0017368, 0.0015928
8: -0.0008584, 0.0004949, -0.0008739, 0.0005746, -0.0009134, 0.0008377
9: -0.0024377, -0.0008685, -0.0025301, -0.0008505, -0.0009713, 0.0010591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009891
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009965
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9937866, 0.9961576, -0.0013463, 0.0016295
1: -0.0027919, -0.0021805, -0.0028122, -0.0022214, -0.0003355, 0.0004060
2: 0.0015012, 0.0047415, 0.0017182, 0.0048489, -0.0021517, 0.0017778
3: -0.0034313, -0.0019564, -0.0034801, -0.0020552, -0.0008092, 0.0009794
4: 0.0008185, 0.0014456, 0.0008604, 0.0014664, -0.0004165, 0.0003441
5: 0.0008476, 0.0049230, 0.0011205, 0.0050581, -0.0027063, 0.0022360
6: 0.0002913, 0.0013257, 0.0002570, 0.0012564, -0.0005675, 0.0006869
7: -0.0023839, 0.0002924, -0.0024726, 0.0001132, -0.0014683, 0.0017772
8: -0.0008178, 0.0005896, -0.0008645, 0.0004954, -0.0007722, 0.0009346
9: -0.0025475, -0.0009156, -0.0024383, -0.0008615, -0.0010837, 0.0008954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009277, upper bound: 0.0010594
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009277, upper bound: 0.0010594
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9937562, 0.9961593, -0.0014152, 0.0016379
1: -0.0028058, -0.0021876, -0.0028197, -0.0022209, -0.0003526, 0.0004081
2: 0.0015393, 0.0048152, 0.0017158, 0.0048892, -0.0021628, 0.0018687
3: -0.0034648, -0.0019738, -0.0034985, -0.0020541, -0.0008506, 0.0009844
4: 0.0008258, 0.0014599, 0.0008600, 0.0014742, -0.0004186, 0.0003617
5: 0.0008955, 0.0050157, 0.0011175, 0.0051088, -0.0027203, 0.0023504
6: 0.0002678, 0.0013135, 0.0002442, 0.0012572, -0.0005965, 0.0006904
7: -0.0024447, 0.0002609, -0.0025059, 0.0001151, -0.0015434, 0.0017864
8: -0.0008498, 0.0005731, -0.0008820, 0.0004964, -0.0008117, 0.0009394
9: -0.0025283, -0.0008785, -0.0024394, -0.0008412, -0.0010893, 0.0009412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009278, upper bound: 0.0010598
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009278, upper bound: 0.0010652
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938007, 0.9962938, -0.0013649, 0.0014715
1: -0.0027919, -0.0021805, -0.0028086, -0.0021874, -0.0003401, 0.0003667
2: 0.0015012, 0.0047415, 0.0015383, 0.0048304, -0.0019431, 0.0018023
3: -0.0034313, -0.0019564, -0.0034717, -0.0019733, -0.0008203, 0.0008844
4: 0.0008185, 0.0014456, 0.0008256, 0.0014628, -0.0003761, 0.0003488
5: 0.0008476, 0.0049230, 0.0008942, 0.0050348, -0.0024439, 0.0022668
6: 0.0002913, 0.0013257, 0.0002630, 0.0013139, -0.0005753, 0.0006203
7: -0.0023839, 0.0002924, -0.0024573, 0.0002618, -0.0014886, 0.0016049
8: -0.0008178, 0.0005896, -0.0008564, 0.0005735, -0.0007828, 0.0008440
9: -0.0025475, -0.0009156, -0.0025289, -0.0008708, -0.0009786, 0.0009077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010646
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010646
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9937702, 0.9962957, -0.0014330, 0.0014780
1: -0.0028058, -0.0021876, -0.0028163, -0.0021870, -0.0003571, 0.0003683
2: 0.0015393, 0.0048152, 0.0015359, 0.0048707, -0.0019517, 0.0018923
3: -0.0034648, -0.0019738, -0.0034900, -0.0019722, -0.0008613, 0.0008883
4: 0.0008258, 0.0014599, 0.0008252, 0.0014706, -0.0003777, 0.0003662
5: 0.0008955, 0.0050157, 0.0008912, 0.0050854, -0.0024547, 0.0023800
6: 0.0002678, 0.0013135, 0.0002501, 0.0013146, -0.0006041, 0.0006230
7: -0.0024447, 0.0002609, -0.0024906, 0.0002638, -0.0015629, 0.0016119
8: -0.0008498, 0.0005731, -0.0008739, 0.0005746, -0.0008219, 0.0008477
9: -0.0025283, -0.0008785, -0.0025301, -0.0008505, -0.0009830, 0.0009531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010649
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010686
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009004
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009004
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009076
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009412
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0008975
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0008975
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0009031
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009614, upper bound: 0.0009313
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009614
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009614
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009659
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008975, upper bound: 0.0009871
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009725
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009724
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009781
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009210, upper bound: 0.0009906
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010029, upper bound: 0.0009004
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010029, upper bound: 0.0009004
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0009076
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0009404
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0008974
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0008974
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0009031
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0010548, upper bound: 0.0009304
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009818, upper bound: 0.0009606
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009818, upper bound: 0.0009606
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009843, upper bound: 0.0009653
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009843, upper bound: 0.0009848
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009913, upper bound: 0.0009724
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009913, upper bound: 0.0009724
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0009769
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0009886
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010029
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010052
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010030
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0010211
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009818
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009843
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009818
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009606, upper bound: 0.0009948
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010548
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010548
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010553
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0008974, upper bound: 0.0010632
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010584
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010583
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010590
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009209, upper bound: 0.0010653
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010136
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010143
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010136
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009375, upper bound: 0.0010228
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009891
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009902
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009891
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009815, upper bound: 0.0009965
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009277, upper bound: 0.0010594
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009277, upper bound: 0.0010594
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009278, upper bound: 0.0010598
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009278, upper bound: 0.0010652
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010646
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010646
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010649
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 0, lower bound: -0.0009389, upper bound: 0.0010686

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938380, 0.9958827, -0.0012293, 0.0012293
1: -0.0027994, -0.0022899, -0.0027994, -0.0022899, -0.0003063, 0.0003063
2: 0.0020812, 0.0047811, 0.0020812, 0.0047811, -0.0016233, 0.0016233
3: -0.0034493, -0.0022204, -0.0034493, -0.0022204, -0.0007388, 0.0007388
4: 0.0009307, 0.0014533, 0.0009307, 0.0014533, -0.0003142, 0.0003142
5: 0.0015771, 0.0049728, 0.0015771, 0.0049728, -0.0020417, 0.0020417
6: 0.0002787, 0.0011406, 0.0002787, 0.0011406, -0.0005182, 0.0005182
7: -0.0024166, -0.0001867, -0.0024166, -0.0001867, -0.0013407, 0.0013407
8: -0.0008350, 0.0003377, -0.0008350, 0.0003377, -0.0007051, 0.0007051
9: -0.0022554, -0.0008956, -0.0022554, -0.0008956, -0.0008176, 0.0008176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0008534
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0008738
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9937923, 0.9958481, -0.0012000, 0.0012899
1: -0.0027994, -0.0022899, -0.0028108, -0.0022985, -0.0002990, 0.0003214
2: 0.0020812, 0.0047811, 0.0021269, 0.0048415, -0.0017033, 0.0015846
3: -0.0034493, -0.0022204, -0.0034768, -0.0022412, -0.0007212, 0.0007753
4: 0.0009307, 0.0014533, 0.0009395, 0.0014650, -0.0003297, 0.0003067
5: 0.0015771, 0.0049728, 0.0016345, 0.0050488, -0.0021423, 0.0019930
6: 0.0002787, 0.0011406, 0.0002594, 0.0011260, -0.0005059, 0.0005437
7: -0.0024166, -0.0001867, -0.0024665, -0.0002244, -0.0013088, 0.0014068
8: -0.0008350, 0.0003377, -0.0008613, 0.0003179, -0.0006883, 0.0007398
9: -0.0022554, -0.0008956, -0.0022324, -0.0008652, -0.0008579, 0.0007981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0008535
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0008738
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9938380, 0.9958827, -0.0012899, 0.0012000
1: -0.0028108, -0.0022985, -0.0027994, -0.0022899, -0.0003214, 0.0002990
2: 0.0021269, 0.0048415, 0.0020812, 0.0047811, -0.0015846, 0.0017033
3: -0.0034768, -0.0022412, -0.0034493, -0.0022204, -0.0007753, 0.0007212
4: 0.0009395, 0.0014650, 0.0009307, 0.0014533, -0.0003067, 0.0003297
5: 0.0016345, 0.0050488, 0.0015771, 0.0049728, -0.0019930, 0.0021423
6: 0.0002594, 0.0011260, 0.0002787, 0.0011406, -0.0005437, 0.0005059
7: -0.0024665, -0.0002244, -0.0024166, -0.0001867, -0.0014068, 0.0013088
8: -0.0008613, 0.0003179, -0.0008350, 0.0003377, -0.0007398, 0.0006883
9: -0.0022324, -0.0008652, -0.0022554, -0.0008956, -0.0007981, 0.0008579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0008587
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0008810
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9937923, 0.9958481, -0.0012537, 0.0012537
1: -0.0028108, -0.0022985, -0.0028108, -0.0022985, -0.0003124, 0.0003124
2: 0.0021269, 0.0048415, 0.0021269, 0.0048415, -0.0016554, 0.0016554
3: -0.0034768, -0.0022412, -0.0034768, -0.0022412, -0.0007535, 0.0007535
4: 0.0009395, 0.0014650, 0.0009395, 0.0014650, -0.0003204, 0.0003204
5: 0.0016345, 0.0050488, 0.0016345, 0.0050488, -0.0020821, 0.0020821
6: 0.0002594, 0.0011260, 0.0002594, 0.0011260, -0.0005285, 0.0005285
7: -0.0024665, -0.0002244, -0.0024665, -0.0002244, -0.0013673, 0.0013673
8: -0.0008613, 0.0003179, -0.0008613, 0.0003179, -0.0007190, 0.0007190
9: -0.0022324, -0.0008652, -0.0022324, -0.0008652, -0.0008338, 0.0008338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0008940
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009151
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938447, 0.9960338, -0.0014419, 0.0012647
1: -0.0027994, -0.0022899, -0.0027977, -0.0022522, -0.0003593, 0.0003151
2: 0.0020812, 0.0047811, 0.0018816, 0.0047721, -0.0016700, 0.0019040
3: -0.0034493, -0.0022204, -0.0034452, -0.0021295, -0.0008666, 0.0007601
4: 0.0009307, 0.0014533, 0.0008921, 0.0014515, -0.0003232, 0.0003685
5: 0.0015771, 0.0049728, 0.0013260, 0.0049615, -0.0021004, 0.0023947
6: 0.0002787, 0.0011406, 0.0002815, 0.0012043, -0.0006078, 0.0005331
7: -0.0024166, -0.0001867, -0.0024092, -0.0000218, -0.0015726, 0.0013793
8: -0.0008350, 0.0003377, -0.0008311, 0.0004244, -0.0008270, 0.0007254
9: -0.0022554, -0.0008956, -0.0023560, -0.0009001, -0.0008411, 0.0009589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009253, upper bound: 0.0008505
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009367, upper bound: 0.0008711
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938010, 0.9959976, -0.0014161, 0.0013290
1: -0.0027994, -0.0022899, -0.0028086, -0.0022613, -0.0003529, 0.0003312
2: 0.0020812, 0.0047811, 0.0019295, 0.0048300, -0.0017549, 0.0018700
3: -0.0034493, -0.0022204, -0.0034715, -0.0021513, -0.0008511, 0.0007988
4: 0.0009307, 0.0014533, 0.0009013, 0.0014627, -0.0003397, 0.0003619
5: 0.0015771, 0.0049728, 0.0013862, 0.0050343, -0.0022072, 0.0023520
6: 0.0002787, 0.0011406, 0.0002631, 0.0011890, -0.0005970, 0.0005602
7: -0.0024166, -0.0001867, -0.0024570, -0.0000613, -0.0015445, 0.0014495
8: -0.0008350, 0.0003377, -0.0008563, 0.0004036, -0.0008122, 0.0007623
9: -0.0022554, -0.0008956, -0.0023318, -0.0008710, -0.0008839, 0.0009418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009253, upper bound: 0.0008506
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009367, upper bound: 0.0008712
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9938447, 0.9960338, -0.0015025, 0.0012354
1: -0.0028108, -0.0022985, -0.0027977, -0.0022522, -0.0003744, 0.0003078
2: 0.0021269, 0.0048415, 0.0018816, 0.0047721, -0.0016313, 0.0019840
3: -0.0034768, -0.0022412, -0.0034452, -0.0021295, -0.0009030, 0.0007425
4: 0.0009395, 0.0014650, 0.0008921, 0.0014515, -0.0003157, 0.0003840
5: 0.0016345, 0.0050488, 0.0013260, 0.0049615, -0.0020517, 0.0024954
6: 0.0002594, 0.0011260, 0.0002815, 0.0012043, -0.0006333, 0.0005208
7: -0.0024665, -0.0002244, -0.0024092, -0.0000218, -0.0016387, 0.0013474
8: -0.0008613, 0.0003179, -0.0008311, 0.0004244, -0.0008618, 0.0007086
9: -0.0022324, -0.0008652, -0.0023560, -0.0009001, -0.0008216, 0.0009993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009220, upper bound: 0.0008537
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009348, upper bound: 0.0008772
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9938010, 0.9959976, -0.0014666, 0.0012908
1: -0.0028108, -0.0022985, -0.0028086, -0.0022613, -0.0003654, 0.0003216
2: 0.0021269, 0.0048415, 0.0019295, 0.0048300, -0.0017045, 0.0019366
3: -0.0034768, -0.0022412, -0.0034715, -0.0021513, -0.0008815, 0.0007758
4: 0.0009395, 0.0014650, 0.0009013, 0.0014627, -0.0003299, 0.0003748
5: 0.0016345, 0.0050488, 0.0013862, 0.0050343, -0.0021438, 0.0024358
6: 0.0002594, 0.0011260, 0.0002631, 0.0011890, -0.0006182, 0.0005441
7: -0.0024665, -0.0002244, -0.0024570, -0.0000613, -0.0015995, 0.0014078
8: -0.0008613, 0.0003179, -0.0008563, 0.0004036, -0.0008412, 0.0007403
9: -0.0022324, -0.0008652, -0.0023318, -0.0008710, -0.0008585, 0.0009754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009220, upper bound: 0.0008886
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009348, upper bound: 0.0009062
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938380, 0.9958827, -0.0012647, 0.0014419
1: -0.0027977, -0.0022522, -0.0027994, -0.0022899, -0.0003151, 0.0003593
2: 0.0018816, 0.0047721, 0.0020812, 0.0047811, -0.0019040, 0.0016700
3: -0.0034452, -0.0021295, -0.0034493, -0.0022204, -0.0007601, 0.0008666
4: 0.0008921, 0.0014515, 0.0009307, 0.0014533, -0.0003685, 0.0003232
5: 0.0013260, 0.0049615, 0.0015771, 0.0049728, -0.0023947, 0.0021004
6: 0.0002815, 0.0012043, 0.0002787, 0.0011406, -0.0005331, 0.0006078
7: -0.0024092, -0.0000218, -0.0024166, -0.0001867, -0.0013793, 0.0015726
8: -0.0008311, 0.0004244, -0.0008350, 0.0003377, -0.0007254, 0.0008270
9: -0.0023560, -0.0009001, -0.0022554, -0.0008956, -0.0009589, 0.0008411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0009147
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0009348
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9937923, 0.9958481, -0.0012354, 0.0015025
1: -0.0027977, -0.0022522, -0.0028108, -0.0022985, -0.0003078, 0.0003744
2: 0.0018816, 0.0047721, 0.0021269, 0.0048415, -0.0019840, 0.0016313
3: -0.0034452, -0.0021295, -0.0034768, -0.0022412, -0.0007425, 0.0009030
4: 0.0008921, 0.0014515, 0.0009395, 0.0014650, -0.0003840, 0.0003157
5: 0.0013260, 0.0049615, 0.0016345, 0.0050488, -0.0024954, 0.0020517
6: 0.0002815, 0.0012043, 0.0002594, 0.0011260, -0.0005208, 0.0006333
7: -0.0024092, -0.0000218, -0.0024665, -0.0002244, -0.0013474, 0.0016387
8: -0.0008311, 0.0004244, -0.0008613, 0.0003179, -0.0007086, 0.0008618
9: -0.0023560, -0.0009001, -0.0022324, -0.0008652, -0.0009993, 0.0008216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0009154
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0009348
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9938380, 0.9958827, -0.0013290, 0.0014161
1: -0.0028086, -0.0022613, -0.0027994, -0.0022899, -0.0003312, 0.0003529
2: 0.0019295, 0.0048300, 0.0020812, 0.0047811, -0.0018700, 0.0017549
3: -0.0034715, -0.0021513, -0.0034493, -0.0022204, -0.0007988, 0.0008511
4: 0.0009013, 0.0014627, 0.0009307, 0.0014533, -0.0003619, 0.0003397
5: 0.0013862, 0.0050343, 0.0015771, 0.0049728, -0.0023520, 0.0022072
6: 0.0002631, 0.0011890, 0.0002787, 0.0011406, -0.0005602, 0.0005970
7: -0.0024570, -0.0000613, -0.0024166, -0.0001867, -0.0014495, 0.0015445
8: -0.0008563, 0.0004036, -0.0008350, 0.0003377, -0.0007623, 0.0008122
9: -0.0023318, -0.0008710, -0.0022554, -0.0008956, -0.0009418, 0.0008839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0009167
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008711, upper bound: 0.0009387
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9937923, 0.9958481, -0.0012908, 0.0014666
1: -0.0028086, -0.0022613, -0.0028108, -0.0022985, -0.0003216, 0.0003654
2: 0.0019295, 0.0048300, 0.0021269, 0.0048415, -0.0019366, 0.0017045
3: -0.0034715, -0.0021513, -0.0034768, -0.0022412, -0.0007758, 0.0008815
4: 0.0009013, 0.0014627, 0.0009395, 0.0014650, -0.0003748, 0.0003299
5: 0.0013862, 0.0050343, 0.0016345, 0.0050488, -0.0024358, 0.0021438
6: 0.0002631, 0.0011890, 0.0002594, 0.0011260, -0.0005441, 0.0006182
7: -0.0024570, -0.0000613, -0.0024665, -0.0002244, -0.0014078, 0.0015995
8: -0.0008563, 0.0004036, -0.0008613, 0.0003179, -0.0007403, 0.0008412
9: -0.0023318, -0.0008710, -0.0022324, -0.0008652, -0.0009754, 0.0008585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0009445
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008711, upper bound: 0.0009613
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938447, 0.9960338, -0.0012852, 0.0012852
1: -0.0027977, -0.0022522, -0.0027977, -0.0022522, -0.0003202, 0.0003202
2: 0.0018816, 0.0047721, 0.0018816, 0.0047721, -0.0016971, 0.0016971
3: -0.0034452, -0.0021295, -0.0034452, -0.0021295, -0.0007725, 0.0007725
4: 0.0008921, 0.0014515, 0.0008921, 0.0014515, -0.0003285, 0.0003285
5: 0.0013260, 0.0049615, 0.0013260, 0.0049615, -0.0021346, 0.0021346
6: 0.0002815, 0.0012043, 0.0002815, 0.0012043, -0.0005418, 0.0005418
7: -0.0024092, -0.0000218, -0.0024092, -0.0000218, -0.0014017, 0.0014017
8: -0.0008311, 0.0004244, -0.0008311, 0.0004244, -0.0007372, 0.0007372
9: -0.0023560, -0.0009001, -0.0023560, -0.0009001, -0.0008548, 0.0008548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008774, upper bound: 0.0009211
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0009470
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938010, 0.9959976, -0.0012552, 0.0013480
1: -0.0027977, -0.0022522, -0.0028086, -0.0022613, -0.0003128, 0.0003359
2: 0.0018816, 0.0047721, 0.0019295, 0.0048300, -0.0017801, 0.0016574
3: -0.0034452, -0.0021295, -0.0034715, -0.0021513, -0.0007544, 0.0008102
4: 0.0008921, 0.0014515, 0.0009013, 0.0014627, -0.0003445, 0.0003208
5: 0.0013260, 0.0049615, 0.0013862, 0.0050343, -0.0022389, 0.0020846
6: 0.0002815, 0.0012043, 0.0002631, 0.0011890, -0.0005291, 0.0005682
7: -0.0024092, -0.0000218, -0.0024570, -0.0000613, -0.0013689, 0.0014702
8: -0.0008311, 0.0004244, -0.0008563, 0.0004036, -0.0007199, 0.0007732
9: -0.0023560, -0.0009001, -0.0023318, -0.0008710, -0.0008965, 0.0008348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008774, upper bound: 0.0009216
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0009471
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9938447, 0.9960338, -0.0013480, 0.0012552
1: -0.0028086, -0.0022613, -0.0027977, -0.0022522, -0.0003359, 0.0003128
2: 0.0019295, 0.0048300, 0.0018816, 0.0047721, -0.0016574, 0.0017801
3: -0.0034715, -0.0021513, -0.0034452, -0.0021295, -0.0008102, 0.0007544
4: 0.0009013, 0.0014627, 0.0008921, 0.0014515, -0.0003208, 0.0003445
5: 0.0013862, 0.0050343, 0.0013260, 0.0049615, -0.0020846, 0.0022389
6: 0.0002631, 0.0011890, 0.0002815, 0.0012043, -0.0005682, 0.0005291
7: -0.0024570, -0.0000613, -0.0024092, -0.0000218, -0.0014702, 0.0013689
8: -0.0008563, 0.0004036, -0.0008311, 0.0004244, -0.0007732, 0.0007199
9: -0.0023318, -0.0008710, -0.0023560, -0.0009001, -0.0008348, 0.0008965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008747, upper bound: 0.0009234
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008956, upper bound: 0.0009517
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9938010, 0.9959976, -0.0013084, 0.0013084
1: -0.0028086, -0.0022613, -0.0028086, -0.0022613, -0.0003260, 0.0003260
2: 0.0019295, 0.0048300, 0.0019295, 0.0048300, -0.0017277, 0.0017277
3: -0.0034715, -0.0021513, -0.0034715, -0.0021513, -0.0007864, 0.0007864
4: 0.0009013, 0.0014627, 0.0009013, 0.0014627, -0.0003344, 0.0003344
5: 0.0013862, 0.0050343, 0.0013862, 0.0050343, -0.0021730, 0.0021730
6: 0.0002631, 0.0011890, 0.0002631, 0.0011890, -0.0005515, 0.0005515
7: -0.0024570, -0.0000613, -0.0024570, -0.0000613, -0.0014270, 0.0014270
8: -0.0008563, 0.0004036, -0.0008563, 0.0004036, -0.0007504, 0.0007504
9: -0.0023318, -0.0008710, -0.0023318, -0.0008710, -0.0008702, 0.0008702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008747, upper bound: 0.0009478
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008956, upper bound: 0.0009653
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938553, 0.9961838, -0.0016381, 0.0013005
1: -0.0027994, -0.0022899, -0.0027951, -0.0022148, -0.0004082, 0.0003240
2: 0.0020812, 0.0047811, 0.0016835, 0.0047584, -0.0017172, 0.0021631
3: -0.0034493, -0.0022204, -0.0034389, -0.0020394, -0.0009846, 0.0007816
4: 0.0009307, 0.0014533, 0.0008537, 0.0014489, -0.0003324, 0.0004187
5: 0.0015771, 0.0049728, 0.0010769, 0.0049442, -0.0021598, 0.0027207
6: 0.0002787, 0.0011406, 0.0002859, 0.0012675, -0.0006905, 0.0005482
7: -0.0024166, -0.0001867, -0.0023978, 0.0001418, -0.0017866, 0.0014183
8: -0.0008350, 0.0003377, -0.0008251, 0.0005104, -0.0009396, 0.0007459
9: -0.0022554, -0.0008956, -0.0024557, -0.0009071, -0.0008649, 0.0010895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009627, upper bound: 0.0008534
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009752, upper bound: 0.0008738
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9937973, 0.9961568, -0.0016059, 0.0013566
1: -0.0027994, -0.0022899, -0.0028095, -0.0022216, -0.0004001, 0.0003380
2: 0.0020812, 0.0047811, 0.0017192, 0.0048349, -0.0017913, 0.0021205
3: -0.0034493, -0.0022204, -0.0034738, -0.0020556, -0.0009652, 0.0008153
4: 0.0009307, 0.0014533, 0.0008606, 0.0014637, -0.0003467, 0.0004104
5: 0.0015771, 0.0049728, 0.0011218, 0.0050405, -0.0022530, 0.0026671
6: 0.0002787, 0.0011406, 0.0002615, 0.0012561, -0.0006769, 0.0005718
7: -0.0024166, -0.0001867, -0.0024611, 0.0001123, -0.0017514, 0.0014795
8: -0.0008350, 0.0003377, -0.0008584, 0.0004949, -0.0009211, 0.0007781
9: -0.0022554, -0.0008956, -0.0024377, -0.0008685, -0.0009022, 0.0010680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009627, upper bound: 0.0008535
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009752, upper bound: 0.0008738
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9938553, 0.9961838, -0.0016988, 0.0012712
1: -0.0028108, -0.0022985, -0.0027951, -0.0022148, -0.0004233, 0.0003167
2: 0.0021269, 0.0048415, 0.0016835, 0.0047584, -0.0016786, 0.0022432
3: -0.0034768, -0.0022412, -0.0034389, -0.0020394, -0.0010210, 0.0007640
4: 0.0009395, 0.0014650, 0.0008537, 0.0014489, -0.0003249, 0.0004342
5: 0.0016345, 0.0050488, 0.0010769, 0.0049442, -0.0021112, 0.0028213
6: 0.0002594, 0.0011260, 0.0002859, 0.0012675, -0.0007161, 0.0005358
7: -0.0024665, -0.0002244, -0.0023978, 0.0001418, -0.0018527, 0.0013864
8: -0.0008613, 0.0003179, -0.0008251, 0.0005104, -0.0009743, 0.0007291
9: -0.0022324, -0.0008652, -0.0024557, -0.0009071, -0.0008454, 0.0011298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009656, upper bound: 0.0008587
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009775, upper bound: 0.0008810
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9937973, 0.9961568, -0.0016621, 0.0013235
1: -0.0028108, -0.0022985, -0.0028095, -0.0022216, -0.0004141, 0.0003298
2: 0.0021269, 0.0048415, 0.0017192, 0.0048349, -0.0017477, 0.0021948
3: -0.0034768, -0.0022412, -0.0034738, -0.0020556, -0.0009990, 0.0007955
4: 0.0009395, 0.0014650, 0.0008606, 0.0014637, -0.0003383, 0.0004248
5: 0.0016345, 0.0050488, 0.0011218, 0.0050405, -0.0021982, 0.0027604
6: 0.0002594, 0.0011260, 0.0002615, 0.0012561, -0.0007006, 0.0005579
7: -0.0024665, -0.0002244, -0.0024611, 0.0001123, -0.0018127, 0.0014435
8: -0.0008613, 0.0003179, -0.0008584, 0.0004949, -0.0009533, 0.0007591
9: -0.0022324, -0.0008652, -0.0024377, -0.0008685, -0.0008802, 0.0011054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009656, upper bound: 0.0008940
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009775, upper bound: 0.0009145
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938680, 0.9963219, -0.0017815, 0.0012789
1: -0.0027994, -0.0022899, -0.0027919, -0.0021805, -0.0004439, 0.0003187
2: 0.0020812, 0.0047811, 0.0015012, 0.0047415, -0.0016888, 0.0023525
3: -0.0034493, -0.0022204, -0.0034313, -0.0019564, -0.0010708, 0.0007687
4: 0.0009307, 0.0014533, 0.0008185, 0.0014456, -0.0003269, 0.0004553
5: 0.0015771, 0.0049728, 0.0008476, 0.0049230, -0.0021240, 0.0029588
6: 0.0002787, 0.0011406, 0.0002913, 0.0013257, -0.0007510, 0.0005391
7: -0.0024166, -0.0001867, -0.0023839, 0.0002924, -0.0019430, 0.0013948
8: -0.0008350, 0.0003377, -0.0008178, 0.0005896, -0.0010218, 0.0007335
9: -0.0022554, -0.0008956, -0.0025475, -0.0009156, -0.0008506, 0.0011848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0008505
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010293, upper bound: 0.0008709
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938380, 0.9958827, 0.9938123, 0.9962930, -0.0017526, 0.0013380
1: -0.0027994, -0.0022899, -0.0028058, -0.0021876, -0.0004367, 0.0003334
2: 0.0020812, 0.0047811, 0.0015393, 0.0048152, -0.0017668, 0.0023143
3: -0.0034493, -0.0022204, -0.0034648, -0.0019738, -0.0010534, 0.0008042
4: 0.0009307, 0.0014533, 0.0008258, 0.0014599, -0.0003420, 0.0004479
5: 0.0015771, 0.0049728, 0.0008955, 0.0050157, -0.0022222, 0.0029108
6: 0.0002787, 0.0011406, 0.0002678, 0.0013135, -0.0007388, 0.0005640
7: -0.0024166, -0.0001867, -0.0024447, 0.0002609, -0.0019115, 0.0014593
8: -0.0008350, 0.0003377, -0.0008498, 0.0005731, -0.0010052, 0.0007674
9: -0.0022554, -0.0008956, -0.0025283, -0.0008785, -0.0008899, 0.0011656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0008506
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010293, upper bound: 0.0008710
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9938680, 0.9963219, -0.0018421, 0.0012496
1: -0.0028108, -0.0022985, -0.0027919, -0.0021805, -0.0004590, 0.0003114
2: 0.0021269, 0.0048415, 0.0015012, 0.0047415, -0.0016501, 0.0024325
3: -0.0034768, -0.0022412, -0.0034313, -0.0019564, -0.0011072, 0.0007511
4: 0.0009395, 0.0014650, 0.0008185, 0.0014456, -0.0003194, 0.0004708
5: 0.0016345, 0.0050488, 0.0008476, 0.0049230, -0.0020754, 0.0030595
6: 0.0002594, 0.0011260, 0.0002913, 0.0013257, -0.0007765, 0.0005268
7: -0.0024665, -0.0002244, -0.0023839, 0.0002924, -0.0020091, 0.0013629
8: -0.0008613, 0.0003179, -0.0008178, 0.0005896, -0.0010566, 0.0007167
9: -0.0022324, -0.0008652, -0.0025475, -0.0009156, -0.0008311, 0.0012252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010195, upper bound: 0.0008537
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010284, upper bound: 0.0008772
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937923, 0.9958481, 0.9938123, 0.9962930, -0.0018057, 0.0013044
1: -0.0028108, -0.0022985, -0.0028058, -0.0021876, -0.0004499, 0.0003250
2: 0.0021269, 0.0048415, 0.0015393, 0.0048152, -0.0017224, 0.0023844
3: -0.0034768, -0.0022412, -0.0034648, -0.0019738, -0.0010853, 0.0007840
4: 0.0009395, 0.0014650, 0.0008258, 0.0014599, -0.0003334, 0.0004615
5: 0.0016345, 0.0050488, 0.0008955, 0.0050157, -0.0021663, 0.0029989
6: 0.0002594, 0.0011260, 0.0002678, 0.0013135, -0.0007612, 0.0005498
7: -0.0024665, -0.0002244, -0.0024447, 0.0002609, -0.0019693, 0.0014226
8: -0.0008613, 0.0003179, -0.0008498, 0.0005731, -0.0010357, 0.0007481
9: -0.0022324, -0.0008652, -0.0025283, -0.0008785, -0.0008675, 0.0012009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010195, upper bound: 0.0008886
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010284, upper bound: 0.0009053
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938553, 0.9961838, -0.0016735, 0.0015130
1: -0.0027977, -0.0022522, -0.0027951, -0.0022148, -0.0004170, 0.0003770
2: 0.0018816, 0.0047721, 0.0016835, 0.0047584, -0.0019979, 0.0022098
3: -0.0034452, -0.0021295, -0.0034389, -0.0020394, -0.0010058, 0.0009094
4: 0.0008921, 0.0014515, 0.0008537, 0.0014489, -0.0003867, 0.0004277
5: 0.0013260, 0.0049615, 0.0010769, 0.0049442, -0.0025129, 0.0027794
6: 0.0002815, 0.0012043, 0.0002859, 0.0012675, -0.0007054, 0.0006378
7: -0.0024092, -0.0000218, -0.0023978, 0.0001418, -0.0018252, 0.0016502
8: -0.0008311, 0.0004244, -0.0008251, 0.0005104, -0.0009599, 0.0008678
9: -0.0023560, -0.0009001, -0.0024557, -0.0009071, -0.0010063, 0.0011130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009449, upper bound: 0.0009147
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009543, upper bound: 0.0009343
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9937973, 0.9961568, -0.0016412, 0.0015691
1: -0.0027977, -0.0022522, -0.0028095, -0.0022216, -0.0004090, 0.0003910
2: 0.0018816, 0.0047721, 0.0017192, 0.0048349, -0.0020720, 0.0021672
3: -0.0034452, -0.0021295, -0.0034738, -0.0020556, -0.0009864, 0.0009431
4: 0.0008921, 0.0014515, 0.0008606, 0.0014637, -0.0004010, 0.0004195
5: 0.0013260, 0.0049615, 0.0011218, 0.0050405, -0.0026061, 0.0027258
6: 0.0002815, 0.0012043, 0.0002615, 0.0012561, -0.0006918, 0.0006614
7: -0.0024092, -0.0000218, -0.0024611, 0.0001123, -0.0017900, 0.0017114
8: -0.0008311, 0.0004244, -0.0008584, 0.0004949, -0.0009413, 0.0009000
9: -0.0023560, -0.0009001, -0.0024377, -0.0008685, -0.0010436, 0.0010915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009449, upper bound: 0.0009152
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009543, upper bound: 0.0009344
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9938553, 0.9961838, -0.0017378, 0.0014873
1: -0.0028086, -0.0022613, -0.0027951, -0.0022148, -0.0004330, 0.0003706
2: 0.0019295, 0.0048300, 0.0016835, 0.0047584, -0.0019639, 0.0022948
3: -0.0034715, -0.0021513, -0.0034389, -0.0020394, -0.0010445, 0.0008939
4: 0.0009013, 0.0014627, 0.0008537, 0.0014489, -0.0003801, 0.0004442
5: 0.0013862, 0.0050343, 0.0010769, 0.0049442, -0.0024701, 0.0028863
6: 0.0002631, 0.0011890, 0.0002859, 0.0012675, -0.0007326, 0.0006269
7: -0.0024570, -0.0000613, -0.0023978, 0.0001418, -0.0018954, 0.0016221
8: -0.0008563, 0.0004036, -0.0008251, 0.0005104, -0.0009968, 0.0008530
9: -0.0023318, -0.0008710, -0.0024557, -0.0009071, -0.0009891, 0.0011558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009473, upper bound: 0.0009167
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009572, upper bound: 0.0009385
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9937973, 0.9961568, -0.0016992, 0.0015365
1: -0.0028086, -0.0022613, -0.0028095, -0.0022216, -0.0004234, 0.0003829
2: 0.0019295, 0.0048300, 0.0017192, 0.0048349, -0.0020289, 0.0022438
3: -0.0034715, -0.0021513, -0.0034738, -0.0020556, -0.0010213, 0.0009235
4: 0.0009013, 0.0014627, 0.0008606, 0.0014637, -0.0003927, 0.0004343
5: 0.0013862, 0.0050343, 0.0011218, 0.0050405, -0.0025518, 0.0028221
6: 0.0002631, 0.0011890, 0.0002615, 0.0012561, -0.0007163, 0.0006477
7: -0.0024570, -0.0000613, -0.0024611, 0.0001123, -0.0018532, 0.0016758
8: -0.0008563, 0.0004036, -0.0008584, 0.0004949, -0.0009746, 0.0008813
9: -0.0023318, -0.0008710, -0.0024377, -0.0008685, -0.0010219, 0.0011301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009473, upper bound: 0.0009437
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009572, upper bound: 0.0009591
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938680, 0.9963219, -0.0016846, 0.0013534
1: -0.0027977, -0.0022522, -0.0027919, -0.0021805, -0.0004198, 0.0003372
2: 0.0018816, 0.0047721, 0.0015012, 0.0047415, -0.0017872, 0.0022245
3: -0.0034452, -0.0021295, -0.0034313, -0.0019564, -0.0010125, 0.0008135
4: 0.0008921, 0.0014515, 0.0008185, 0.0014456, -0.0003459, 0.0004305
5: 0.0013260, 0.0049615, 0.0008476, 0.0049230, -0.0022478, 0.0027978
6: 0.0002815, 0.0012043, 0.0002913, 0.0013257, -0.0007101, 0.0005705
7: -0.0024092, -0.0000218, -0.0023839, 0.0002924, -0.0018373, 0.0014761
8: -0.0008311, 0.0004244, -0.0008178, 0.0005896, -0.0009662, 0.0007763
9: -0.0023560, -0.0009001, -0.0025475, -0.0009156, -0.0009001, 0.0011204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009549, upper bound: 0.0009209
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009652, upper bound: 0.0009470
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938447, 0.9960338, 0.9938123, 0.9962930, -0.0016534, 0.0014065
1: -0.0027977, -0.0022522, -0.0028058, -0.0021876, -0.0004120, 0.0003505
2: 0.0018816, 0.0047721, 0.0015393, 0.0048152, -0.0018573, 0.0021833
3: -0.0034452, -0.0021295, -0.0034648, -0.0019738, -0.0009937, 0.0008454
4: 0.0008921, 0.0014515, 0.0008258, 0.0014599, -0.0003595, 0.0004226
5: 0.0013260, 0.0049615, 0.0008955, 0.0050157, -0.0023360, 0.0027460
6: 0.0002815, 0.0012043, 0.0002678, 0.0013135, -0.0006970, 0.0005929
7: -0.0024092, -0.0000218, -0.0024447, 0.0002609, -0.0018033, 0.0015340
8: -0.0008311, 0.0004244, -0.0008498, 0.0005731, -0.0009483, 0.0008067
9: -0.0023560, -0.0009001, -0.0025283, -0.0008785, -0.0009354, 0.0010996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009549, upper bound: 0.0009215
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009652, upper bound: 0.0009470
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9938680, 0.9963219, -0.0017474, 0.0013234
1: -0.0028086, -0.0022613, -0.0027919, -0.0021805, -0.0004354, 0.0003297
2: 0.0019295, 0.0048300, 0.0015012, 0.0047415, -0.0017475, 0.0023074
3: -0.0034715, -0.0021513, -0.0034313, -0.0019564, -0.0010502, 0.0007954
4: 0.0009013, 0.0014627, 0.0008185, 0.0014456, -0.0003382, 0.0004466
5: 0.0013862, 0.0050343, 0.0008476, 0.0049230, -0.0021979, 0.0029021
6: 0.0002631, 0.0011890, 0.0002913, 0.0013257, -0.0007366, 0.0005578
7: -0.0024570, -0.0000613, -0.0023839, 0.0002924, -0.0019058, 0.0014433
8: -0.0008563, 0.0004036, -0.0008178, 0.0005896, -0.0010022, 0.0007590
9: -0.0023318, -0.0008710, -0.0025475, -0.0009156, -0.0008801, 0.0011621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009574, upper bound: 0.0009234
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009670, upper bound: 0.0009508
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938010, 0.9959976, 0.9938123, 0.9962930, -0.0017072, 0.0013742
1: -0.0028086, -0.0022613, -0.0028058, -0.0021876, -0.0004254, 0.0003424
2: 0.0019295, 0.0048300, 0.0015393, 0.0048152, -0.0018147, 0.0022543
3: -0.0034715, -0.0021513, -0.0034648, -0.0019738, -0.0010261, 0.0008260
4: 0.0009013, 0.0014627, 0.0008258, 0.0014599, -0.0003512, 0.0004363
5: 0.0013862, 0.0050343, 0.0008955, 0.0050157, -0.0022824, 0.0028353
6: 0.0002631, 0.0011890, 0.0002678, 0.0013135, -0.0007196, 0.0005793
7: -0.0024570, -0.0000613, -0.0024447, 0.0002609, -0.0018619, 0.0014988
8: -0.0008563, 0.0004036, -0.0008498, 0.0005731, -0.0009792, 0.0007882
9: -0.0023318, -0.0008710, -0.0025283, -0.0008785, -0.0009140, 0.0011354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009574, upper bound: 0.0009469
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009670, upper bound: 0.0009631
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938380, 0.9958827, -0.0013005, 0.0016381
1: -0.0027951, -0.0022148, -0.0027994, -0.0022899, -0.0003240, 0.0004082
2: 0.0016835, 0.0047584, 0.0020812, 0.0047811, -0.0021632, 0.0017172
3: -0.0034389, -0.0020394, -0.0034493, -0.0022204, -0.0007816, 0.0009846
4: 0.0008537, 0.0014489, 0.0009307, 0.0014533, -0.0004187, 0.0003324
5: 0.0010769, 0.0049442, 0.0015771, 0.0049728, -0.0027207, 0.0021598
6: 0.0002859, 0.0012675, 0.0002787, 0.0011406, -0.0005482, 0.0006905
7: -0.0023978, 0.0001418, -0.0024166, -0.0001867, -0.0014183, 0.0017866
8: -0.0008251, 0.0005104, -0.0008350, 0.0003377, -0.0007459, 0.0009396
9: -0.0024557, -0.0009071, -0.0022554, -0.0008956, -0.0010895, 0.0008649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0009409
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0009752
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9937923, 0.9958481, -0.0012712, 0.0016988
1: -0.0027951, -0.0022148, -0.0028108, -0.0022985, -0.0003167, 0.0004233
2: 0.0016835, 0.0047584, 0.0021269, 0.0048415, -0.0022432, 0.0016786
3: -0.0034389, -0.0020394, -0.0034768, -0.0022412, -0.0007640, 0.0010210
4: 0.0008537, 0.0014489, 0.0009395, 0.0014650, -0.0004342, 0.0003249
5: 0.0010769, 0.0049442, 0.0016345, 0.0050488, -0.0028213, 0.0021112
6: 0.0002859, 0.0012675, 0.0002594, 0.0011260, -0.0005358, 0.0007161
7: -0.0023978, 0.0001418, -0.0024665, -0.0002244, -0.0013864, 0.0018527
8: -0.0008251, 0.0005104, -0.0008613, 0.0003179, -0.0007291, 0.0009743
9: -0.0024557, -0.0009071, -0.0022324, -0.0008652, -0.0011298, 0.0008454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0009448
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0009775
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9938380, 0.9958827, -0.0013566, 0.0016059
1: -0.0028095, -0.0022216, -0.0027994, -0.0022899, -0.0003380, 0.0004001
2: 0.0017192, 0.0048349, 0.0020812, 0.0047811, -0.0021205, 0.0017913
3: -0.0034738, -0.0020556, -0.0034493, -0.0022204, -0.0008153, 0.0009652
4: 0.0008606, 0.0014637, 0.0009307, 0.0014533, -0.0004104, 0.0003467
5: 0.0011218, 0.0050405, 0.0015771, 0.0049728, -0.0026671, 0.0022530
6: 0.0002615, 0.0012561, 0.0002787, 0.0011406, -0.0005718, 0.0006769
7: -0.0024611, 0.0001123, -0.0024166, -0.0001867, -0.0014795, 0.0017514
8: -0.0008584, 0.0004949, -0.0008350, 0.0003377, -0.0007781, 0.0009211
9: -0.0024377, -0.0008685, -0.0022554, -0.0008956, -0.0010680, 0.0009022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0009410
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009753
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9937923, 0.9958481, -0.0013235, 0.0016621
1: -0.0028095, -0.0022216, -0.0028108, -0.0022985, -0.0003298, 0.0004141
2: 0.0017192, 0.0048349, 0.0021269, 0.0048415, -0.0021948, 0.0017477
3: -0.0034738, -0.0020556, -0.0034768, -0.0022412, -0.0007955, 0.0009990
4: 0.0008606, 0.0014637, 0.0009395, 0.0014650, -0.0004248, 0.0003383
5: 0.0011218, 0.0050405, 0.0016345, 0.0050488, -0.0027604, 0.0021982
6: 0.0002615, 0.0012561, 0.0002594, 0.0011260, -0.0005579, 0.0007006
7: -0.0024611, 0.0001123, -0.0024665, -0.0002244, -0.0014435, 0.0018127
8: -0.0008584, 0.0004949, -0.0008613, 0.0003179, -0.0007591, 0.0009533
9: -0.0024377, -0.0008685, -0.0022324, -0.0008652, -0.0011054, 0.0008802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0009704
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009930
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938447, 0.9960338, -0.0015130, 0.0016735
1: -0.0027951, -0.0022148, -0.0027977, -0.0022522, -0.0003770, 0.0004170
2: 0.0016835, 0.0047584, 0.0018816, 0.0047721, -0.0022098, 0.0019979
3: -0.0034389, -0.0020394, -0.0034452, -0.0021295, -0.0009094, 0.0010058
4: 0.0008537, 0.0014489, 0.0008921, 0.0014515, -0.0004277, 0.0003867
5: 0.0010769, 0.0049442, 0.0013260, 0.0049615, -0.0027794, 0.0025129
6: 0.0002859, 0.0012675, 0.0002815, 0.0012043, -0.0006378, 0.0007054
7: -0.0023978, 0.0001418, -0.0024092, -0.0000218, -0.0016502, 0.0018252
8: -0.0008251, 0.0005104, -0.0008311, 0.0004244, -0.0008678, 0.0009599
9: -0.0024557, -0.0009071, -0.0023560, -0.0009001, -0.0011130, 0.0010063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009296
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009363, upper bound: 0.0009543
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938010, 0.9959976, -0.0014873, 0.0017378
1: -0.0027951, -0.0022148, -0.0028086, -0.0022613, -0.0003706, 0.0004330
2: 0.0016835, 0.0047584, 0.0019295, 0.0048300, -0.0022948, 0.0019639
3: -0.0034389, -0.0020394, -0.0034715, -0.0021513, -0.0008939, 0.0010445
4: 0.0008537, 0.0014489, 0.0009013, 0.0014627, -0.0004442, 0.0003801
5: 0.0010769, 0.0049442, 0.0013862, 0.0050343, -0.0028863, 0.0024701
6: 0.0002859, 0.0012675, 0.0002631, 0.0011890, -0.0006269, 0.0007326
7: -0.0023978, 0.0001418, -0.0024570, -0.0000613, -0.0016221, 0.0018954
8: -0.0008251, 0.0005104, -0.0008563, 0.0004036, -0.0008530, 0.0009968
9: -0.0024557, -0.0009071, -0.0023318, -0.0008710, -0.0011558, 0.0009891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009353
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009363, upper bound: 0.0009572
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9938447, 0.9960338, -0.0015691, 0.0016412
1: -0.0028095, -0.0022216, -0.0027977, -0.0022522, -0.0003910, 0.0004090
2: 0.0017192, 0.0048349, 0.0018816, 0.0047721, -0.0021672, 0.0020720
3: -0.0034738, -0.0020556, -0.0034452, -0.0021295, -0.0009431, 0.0009864
4: 0.0008606, 0.0014637, 0.0008921, 0.0014515, -0.0004195, 0.0004010
5: 0.0011218, 0.0050405, 0.0013260, 0.0049615, -0.0027258, 0.0026061
6: 0.0002615, 0.0012561, 0.0002815, 0.0012043, -0.0006614, 0.0006918
7: -0.0024611, 0.0001123, -0.0024092, -0.0000218, -0.0017114, 0.0017900
8: -0.0008584, 0.0004949, -0.0008311, 0.0004244, -0.0009000, 0.0009413
9: -0.0024377, -0.0008685, -0.0023560, -0.0009001, -0.0010915, 0.0010436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009213, upper bound: 0.0009296
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009344, upper bound: 0.0009543
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9938010, 0.9959976, -0.0015365, 0.0016992
1: -0.0028095, -0.0022216, -0.0028086, -0.0022613, -0.0003829, 0.0004234
2: 0.0017192, 0.0048349, 0.0019295, 0.0048300, -0.0022438, 0.0020289
3: -0.0034738, -0.0020556, -0.0034715, -0.0021513, -0.0009235, 0.0010213
4: 0.0008606, 0.0014637, 0.0009013, 0.0014627, -0.0004343, 0.0003927
5: 0.0011218, 0.0050405, 0.0013862, 0.0050343, -0.0028221, 0.0025518
6: 0.0002615, 0.0012561, 0.0002631, 0.0011890, -0.0006477, 0.0007163
7: -0.0024611, 0.0001123, -0.0024570, -0.0000613, -0.0016758, 0.0018532
8: -0.0008584, 0.0004949, -0.0008563, 0.0004036, -0.0008813, 0.0009746
9: -0.0024377, -0.0008685, -0.0023318, -0.0008710, -0.0011301, 0.0010219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009213, upper bound: 0.0009530
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009344, upper bound: 0.0009675
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938380, 0.9958827, -0.0012789, 0.0017815
1: -0.0027919, -0.0021805, -0.0027994, -0.0022899, -0.0003187, 0.0004439
2: 0.0015012, 0.0047415, 0.0020812, 0.0047811, -0.0023525, 0.0016888
3: -0.0034313, -0.0019564, -0.0034493, -0.0022204, -0.0007687, 0.0010708
4: 0.0008185, 0.0014456, 0.0009307, 0.0014533, -0.0004553, 0.0003269
5: 0.0008476, 0.0049230, 0.0015771, 0.0049728, -0.0029588, 0.0021241
6: 0.0002913, 0.0013257, 0.0002787, 0.0011406, -0.0005391, 0.0007510
7: -0.0023839, 0.0002924, -0.0024166, -0.0001867, -0.0013948, 0.0019430
8: -0.0008178, 0.0005896, -0.0008350, 0.0003377, -0.0007335, 0.0010218
9: -0.0025475, -0.0009156, -0.0022554, -0.0008956, -0.0011848, 0.0008506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0010033
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0010284
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9937923, 0.9958481, -0.0012496, 0.0018421
1: -0.0027919, -0.0021805, -0.0028108, -0.0022985, -0.0003114, 0.0004590
2: 0.0015012, 0.0047415, 0.0021269, 0.0048415, -0.0024325, 0.0016501
3: -0.0034313, -0.0019564, -0.0034768, -0.0022412, -0.0007511, 0.0011072
4: 0.0008185, 0.0014456, 0.0009395, 0.0014650, -0.0004708, 0.0003194
5: 0.0008476, 0.0049230, 0.0016345, 0.0050488, -0.0030595, 0.0020754
6: 0.0002913, 0.0013257, 0.0002594, 0.0011260, -0.0005268, 0.0007765
7: -0.0023839, 0.0002924, -0.0024665, -0.0002244, -0.0013629, 0.0020091
8: -0.0008178, 0.0005896, -0.0008613, 0.0003179, -0.0007167, 0.0010566
9: -0.0025475, -0.0009156, -0.0022324, -0.0008652, -0.0012252, 0.0008311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0010052
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0010284
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9938380, 0.9958827, -0.0013380, 0.0017526
1: -0.0028058, -0.0021876, -0.0027994, -0.0022899, -0.0003334, 0.0004367
2: 0.0015393, 0.0048152, 0.0020812, 0.0047811, -0.0023143, 0.0017668
3: -0.0034648, -0.0019738, -0.0034493, -0.0022204, -0.0008042, 0.0010534
4: 0.0008258, 0.0014599, 0.0009307, 0.0014533, -0.0004479, 0.0003420
5: 0.0008955, 0.0050157, 0.0015771, 0.0049728, -0.0029108, 0.0022222
6: 0.0002678, 0.0013135, 0.0002787, 0.0011406, -0.0005640, 0.0007388
7: -0.0024447, 0.0002609, -0.0024166, -0.0001867, -0.0014593, 0.0019115
8: -0.0008498, 0.0005731, -0.0008350, 0.0003377, -0.0007674, 0.0010052
9: -0.0025283, -0.0008785, -0.0022554, -0.0008956, -0.0011656, 0.0008899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0010033
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008710, upper bound: 0.0010286
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9937923, 0.9958481, -0.0013044, 0.0018057
1: -0.0028058, -0.0021876, -0.0028108, -0.0022985, -0.0003250, 0.0004499
2: 0.0015393, 0.0048152, 0.0021269, 0.0048415, -0.0023844, 0.0017224
3: -0.0034648, -0.0019738, -0.0034768, -0.0022412, -0.0007840, 0.0010853
4: 0.0008258, 0.0014599, 0.0009395, 0.0014650, -0.0004615, 0.0003334
5: 0.0008955, 0.0050157, 0.0016345, 0.0050488, -0.0029989, 0.0021663
6: 0.0002678, 0.0013135, 0.0002594, 0.0011260, -0.0005498, 0.0007612
7: -0.0024447, 0.0002609, -0.0024665, -0.0002244, -0.0014226, 0.0019693
8: -0.0008498, 0.0005731, -0.0008613, 0.0003179, -0.0007481, 0.0010357
9: -0.0025283, -0.0008785, -0.0022324, -0.0008652, -0.0012009, 0.0008675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0010175
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008710, upper bound: 0.0010370
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938447, 0.9960338, -0.0013534, 0.0016846
1: -0.0027919, -0.0021805, -0.0027977, -0.0022522, -0.0003372, 0.0004198
2: 0.0015012, 0.0047415, 0.0018816, 0.0047721, -0.0022245, 0.0017872
3: -0.0034313, -0.0019564, -0.0034452, -0.0021295, -0.0008135, 0.0010125
4: 0.0008185, 0.0014456, 0.0008921, 0.0014515, -0.0004305, 0.0003459
5: 0.0008476, 0.0049230, 0.0013260, 0.0049615, -0.0027978, 0.0022478
6: 0.0002913, 0.0013257, 0.0002815, 0.0012043, -0.0005705, 0.0007101
7: -0.0023839, 0.0002924, -0.0024092, -0.0000218, -0.0014761, 0.0018373
8: -0.0008178, 0.0005896, -0.0008311, 0.0004244, -0.0007763, 0.0009662
9: -0.0025475, -0.0009156, -0.0023560, -0.0009001, -0.0011204, 0.0009001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0010072
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0010324
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938010, 0.9959976, -0.0013234, 0.0017474
1: -0.0027919, -0.0021805, -0.0028086, -0.0022613, -0.0003297, 0.0004354
2: 0.0015012, 0.0047415, 0.0019295, 0.0048300, -0.0023074, 0.0017475
3: -0.0034313, -0.0019564, -0.0034715, -0.0021513, -0.0007954, 0.0010502
4: 0.0008185, 0.0014456, 0.0009013, 0.0014627, -0.0004466, 0.0003382
5: 0.0008476, 0.0049230, 0.0013862, 0.0050343, -0.0029021, 0.0021979
6: 0.0002913, 0.0013257, 0.0002631, 0.0011890, -0.0005578, 0.0007366
7: -0.0023839, 0.0002924, -0.0024570, -0.0000613, -0.0014433, 0.0019058
8: -0.0008178, 0.0005896, -0.0008563, 0.0004036, -0.0007590, 0.0010022
9: -0.0025475, -0.0009156, -0.0023318, -0.0008710, -0.0011621, 0.0008801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0010087
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0010324
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9938447, 0.9960338, -0.0014065, 0.0016534
1: -0.0028058, -0.0021876, -0.0027977, -0.0022522, -0.0003505, 0.0004120
2: 0.0015393, 0.0048152, 0.0018816, 0.0047721, -0.0021833, 0.0018573
3: -0.0034648, -0.0019738, -0.0034452, -0.0021295, -0.0008454, 0.0009937
4: 0.0008258, 0.0014599, 0.0008921, 0.0014515, -0.0004226, 0.0003595
5: 0.0008955, 0.0050157, 0.0013260, 0.0049615, -0.0027460, 0.0023360
6: 0.0002678, 0.0013135, 0.0002815, 0.0012043, -0.0005929, 0.0006970
7: -0.0024447, 0.0002609, -0.0024092, -0.0000218, -0.0015340, 0.0018033
8: -0.0008498, 0.0005731, -0.0008311, 0.0004244, -0.0008067, 0.0009483
9: -0.0025283, -0.0008785, -0.0023560, -0.0009001, -0.0010996, 0.0009354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008742, upper bound: 0.0010072
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008955, upper bound: 0.0010327
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9938010, 0.9959976, -0.0013742, 0.0017072
1: -0.0028058, -0.0021876, -0.0028086, -0.0022613, -0.0003424, 0.0004254
2: 0.0015393, 0.0048152, 0.0019295, 0.0048300, -0.0022543, 0.0018147
3: -0.0034648, -0.0019738, -0.0034715, -0.0021513, -0.0008260, 0.0010261
4: 0.0008258, 0.0014599, 0.0009013, 0.0014627, -0.0004363, 0.0003512
5: 0.0008955, 0.0050157, 0.0013862, 0.0050343, -0.0028353, 0.0022824
6: 0.0002678, 0.0013135, 0.0002631, 0.0011890, -0.0005793, 0.0007196
7: -0.0024447, 0.0002609, -0.0024570, -0.0000613, -0.0014988, 0.0018619
8: -0.0008498, 0.0005731, -0.0008563, 0.0004036, -0.0007882, 0.0009792
9: -0.0025283, -0.0008785, -0.0023318, -0.0008710, -0.0011354, 0.0009140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008742, upper bound: 0.0010188
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008955, upper bound: 0.0010393
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938553, 0.9961838, -0.0013396, 0.0013396
1: -0.0027951, -0.0022148, -0.0027951, -0.0022148, -0.0003338, 0.0003338
2: 0.0016835, 0.0047584, 0.0016835, 0.0047584, -0.0017689, 0.0017689
3: -0.0034389, -0.0020394, -0.0034389, -0.0020394, -0.0008051, 0.0008051
4: 0.0008537, 0.0014489, 0.0008537, 0.0014489, -0.0003424, 0.0003424
5: 0.0010769, 0.0049442, 0.0010769, 0.0049442, -0.0022248, 0.0022248
6: 0.0002859, 0.0012675, 0.0002859, 0.0012675, -0.0005647, 0.0005647
7: -0.0023978, 0.0001418, -0.0023978, 0.0001418, -0.0014610, 0.0014610
8: -0.0008251, 0.0005104, -0.0008251, 0.0005104, -0.0007683, 0.0007683
9: -0.0024557, -0.0009071, -0.0024557, -0.0009071, -0.0008909, 0.0008909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009006, upper bound: 0.0009582
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009183, upper bound: 0.0009862
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9937973, 0.9961568, -0.0013108, 0.0014109
1: -0.0027951, -0.0022148, -0.0028095, -0.0022216, -0.0003266, 0.0003516
2: 0.0016835, 0.0047584, 0.0017192, 0.0048349, -0.0018631, 0.0017309
3: -0.0034389, -0.0020394, -0.0034738, -0.0020556, -0.0007878, 0.0008480
4: 0.0008537, 0.0014489, 0.0008606, 0.0014637, -0.0003606, 0.0003350
5: 0.0010769, 0.0049442, 0.0011218, 0.0050405, -0.0023433, 0.0021770
6: 0.0002859, 0.0012675, 0.0002615, 0.0012561, -0.0005525, 0.0005947
7: -0.0023978, 0.0001418, -0.0024611, 0.0001123, -0.0014296, 0.0015388
8: -0.0008251, 0.0005104, -0.0008584, 0.0004949, -0.0007518, 0.0008092
9: -0.0024557, -0.0009071, -0.0024377, -0.0008685, -0.0009383, 0.0008718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009006, upper bound: 0.0009608
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009183, upper bound: 0.0009872
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9938553, 0.9961838, -0.0014109, 0.0013108
1: -0.0028095, -0.0022216, -0.0027951, -0.0022148, -0.0003516, 0.0003266
2: 0.0017192, 0.0048349, 0.0016835, 0.0047584, -0.0017309, 0.0018631
3: -0.0034738, -0.0020556, -0.0034389, -0.0020394, -0.0008480, 0.0007878
4: 0.0008606, 0.0014637, 0.0008537, 0.0014489, -0.0003350, 0.0003606
5: 0.0011218, 0.0050405, 0.0010769, 0.0049442, -0.0021770, 0.0023433
6: 0.0002615, 0.0012561, 0.0002859, 0.0012675, -0.0005947, 0.0005525
7: -0.0024611, 0.0001123, -0.0023978, 0.0001418, -0.0015388, 0.0014296
8: -0.0008584, 0.0004949, -0.0008251, 0.0005104, -0.0008092, 0.0007518
9: -0.0024377, -0.0008685, -0.0024557, -0.0009071, -0.0008718, 0.0009383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008962, upper bound: 0.0009584
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009116, upper bound: 0.0009862
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9937973, 0.9961568, -0.0013776, 0.0013776
1: -0.0028095, -0.0022216, -0.0028095, -0.0022216, -0.0003432, 0.0003432
2: 0.0017192, 0.0048349, 0.0017192, 0.0048349, -0.0018190, 0.0018190
3: -0.0034738, -0.0020556, -0.0034738, -0.0020556, -0.0008279, 0.0008279
4: 0.0008606, 0.0014637, 0.0008606, 0.0014637, -0.0003521, 0.0003521
5: 0.0011218, 0.0050405, 0.0011218, 0.0050405, -0.0022879, 0.0022879
6: 0.0002615, 0.0012561, 0.0002615, 0.0012561, -0.0005807, 0.0005807
7: -0.0024611, 0.0001123, -0.0024611, 0.0001123, -0.0015024, 0.0015024
8: -0.0008584, 0.0004949, -0.0008584, 0.0004949, -0.0007901, 0.0007901
9: -0.0024377, -0.0008685, -0.0024377, -0.0008685, -0.0009162, 0.0009162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008962, upper bound: 0.0009747
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009116, upper bound: 0.0009952
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938680, 0.9963219, -0.0015511, 0.0013743
1: -0.0027951, -0.0022148, -0.0027919, -0.0021805, -0.0003865, 0.0003424
2: 0.0016835, 0.0047584, 0.0015012, 0.0047415, -0.0018147, 0.0020482
3: -0.0034389, -0.0020394, -0.0034313, -0.0019564, -0.0009322, 0.0008260
4: 0.0008537, 0.0014489, 0.0008185, 0.0014456, -0.0003512, 0.0003964
5: 0.0010769, 0.0049442, 0.0008476, 0.0049230, -0.0022824, 0.0025761
6: 0.0002859, 0.0012675, 0.0002913, 0.0013257, -0.0006538, 0.0005793
7: -0.0023978, 0.0001418, -0.0023839, 0.0002924, -0.0016917, 0.0014988
8: -0.0008251, 0.0005104, -0.0008178, 0.0005896, -0.0008896, 0.0007882
9: -0.0024557, -0.0009071, -0.0025475, -0.0009156, -0.0009140, 0.0010316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009528, upper bound: 0.0009433
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009619, upper bound: 0.0009621
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938553, 0.9961838, 0.9938123, 0.9962930, -0.0015236, 0.0014463
1: -0.0027951, -0.0022148, -0.0028058, -0.0021876, -0.0003796, 0.0003604
2: 0.0016835, 0.0047584, 0.0015393, 0.0048152, -0.0019098, 0.0020119
3: -0.0034389, -0.0020394, -0.0034648, -0.0019738, -0.0009157, 0.0008692
4: 0.0008537, 0.0014489, 0.0008258, 0.0014599, -0.0003696, 0.0003894
5: 0.0010769, 0.0049442, 0.0008955, 0.0050157, -0.0024020, 0.0025304
6: 0.0002859, 0.0012675, 0.0002678, 0.0013135, -0.0006423, 0.0006096
7: -0.0023978, 0.0001418, -0.0024447, 0.0002609, -0.0016617, 0.0015773
8: -0.0008251, 0.0005104, -0.0008498, 0.0005731, -0.0008739, 0.0008295
9: -0.0024557, -0.0009071, -0.0025283, -0.0008785, -0.0009619, 0.0010133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009528, upper bound: 0.0009460
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009619, upper bound: 0.0009636
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9938680, 0.9963219, -0.0016224, 0.0013455
1: -0.0028095, -0.0022216, -0.0027919, -0.0021805, -0.0004043, 0.0003353
2: 0.0017192, 0.0048349, 0.0015012, 0.0047415, -0.0017767, 0.0021423
3: -0.0034738, -0.0020556, -0.0034313, -0.0019564, -0.0009751, 0.0008087
4: 0.0008606, 0.0014637, 0.0008185, 0.0014456, -0.0003439, 0.0004146
5: 0.0011218, 0.0050405, 0.0008476, 0.0049230, -0.0022346, 0.0026945
6: 0.0002615, 0.0012561, 0.0002913, 0.0013257, -0.0006839, 0.0005672
7: -0.0024611, 0.0001123, -0.0023839, 0.0002924, -0.0017694, 0.0014674
8: -0.0008584, 0.0004949, -0.0008178, 0.0005896, -0.0009305, 0.0007717
9: -0.0024377, -0.0008685, -0.0025475, -0.0009156, -0.0008948, 0.0010790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009465, upper bound: 0.0009433
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009564, upper bound: 0.0009621
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937973, 0.9961568, 0.9938123, 0.9962930, -0.0015894, 0.0014127
1: -0.0028095, -0.0022216, -0.0028058, -0.0021876, -0.0003960, 0.0003520
2: 0.0017192, 0.0048349, 0.0015393, 0.0048152, -0.0018654, 0.0020988
3: -0.0034738, -0.0020556, -0.0034648, -0.0019738, -0.0009553, 0.0008491
4: 0.0008606, 0.0014637, 0.0008258, 0.0014599, -0.0003610, 0.0004062
5: 0.0011218, 0.0050405, 0.0008955, 0.0050157, -0.0023462, 0.0026397
6: 0.0002615, 0.0012561, 0.0002678, 0.0013135, -0.0006700, 0.0005955
7: -0.0024611, 0.0001123, -0.0024447, 0.0002609, -0.0017334, 0.0015407
8: -0.0008584, 0.0004949, -0.0008498, 0.0005731, -0.0009116, 0.0008102
9: -0.0024377, -0.0008685, -0.0025283, -0.0008785, -0.0009395, 0.0010570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009465, upper bound: 0.0009565
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009564, upper bound: 0.0009693
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938553, 0.9961838, -0.0013743, 0.0015511
1: -0.0027919, -0.0021805, -0.0027951, -0.0022148, -0.0003424, 0.0003865
2: 0.0015012, 0.0047415, 0.0016835, 0.0047584, -0.0020482, 0.0018147
3: -0.0034313, -0.0019564, -0.0034389, -0.0020394, -0.0008260, 0.0009322
4: 0.0008185, 0.0014456, 0.0008537, 0.0014489, -0.0003964, 0.0003512
5: 0.0008476, 0.0049230, 0.0010769, 0.0049442, -0.0025761, 0.0022824
6: 0.0002913, 0.0013257, 0.0002859, 0.0012675, -0.0005793, 0.0006538
7: -0.0023839, 0.0002924, -0.0023978, 0.0001418, -0.0014988, 0.0016917
8: -0.0008178, 0.0005896, -0.0008251, 0.0005104, -0.0007882, 0.0008896
9: -0.0025475, -0.0009156, -0.0024557, -0.0009071, -0.0010316, 0.0009140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008925, upper bound: 0.0010115
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010334
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9937973, 0.9961568, -0.0013455, 0.0016224
1: -0.0027919, -0.0021805, -0.0028095, -0.0022216, -0.0003353, 0.0004043
2: 0.0015012, 0.0047415, 0.0017192, 0.0048349, -0.0021423, 0.0017767
3: -0.0034313, -0.0019564, -0.0034738, -0.0020556, -0.0008087, 0.0009751
4: 0.0008185, 0.0014456, 0.0008606, 0.0014637, -0.0004146, 0.0003439
5: 0.0008476, 0.0049230, 0.0011218, 0.0050405, -0.0026945, 0.0022346
6: 0.0002913, 0.0013257, 0.0002615, 0.0012561, -0.0005672, 0.0006839
7: -0.0023839, 0.0002924, -0.0024611, 0.0001123, -0.0014674, 0.0017694
8: -0.0008178, 0.0005896, -0.0008584, 0.0004949, -0.0007717, 0.0009305
9: -0.0025475, -0.0009156, -0.0024377, -0.0008685, -0.0010790, 0.0008948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008925, upper bound: 0.0010115
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010334
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9938553, 0.9961838, -0.0014463, 0.0015236
1: -0.0028058, -0.0021876, -0.0027951, -0.0022148, -0.0003604, 0.0003796
2: 0.0015393, 0.0048152, 0.0016835, 0.0047584, -0.0020119, 0.0019098
3: -0.0034648, -0.0019738, -0.0034389, -0.0020394, -0.0008692, 0.0009157
4: 0.0008258, 0.0014599, 0.0008537, 0.0014489, -0.0003894, 0.0003696
5: 0.0008955, 0.0050157, 0.0010769, 0.0049442, -0.0025304, 0.0024020
6: 0.0002678, 0.0013135, 0.0002859, 0.0012675, -0.0006096, 0.0006423
7: -0.0024447, 0.0002609, -0.0023978, 0.0001418, -0.0015773, 0.0016617
8: -0.0008498, 0.0005731, -0.0008251, 0.0005104, -0.0008295, 0.0008739
9: -0.0025283, -0.0008785, -0.0024557, -0.0009071, -0.0010133, 0.0009619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008882, upper bound: 0.0010115
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009034, upper bound: 0.0010336
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9937973, 0.9961568, -0.0014127, 0.0015894
1: -0.0028058, -0.0021876, -0.0028095, -0.0022216, -0.0003520, 0.0003960
2: 0.0015393, 0.0048152, 0.0017192, 0.0048349, -0.0020988, 0.0018654
3: -0.0034648, -0.0019738, -0.0034738, -0.0020556, -0.0008491, 0.0009553
4: 0.0008258, 0.0014599, 0.0008606, 0.0014637, -0.0004062, 0.0003610
5: 0.0008955, 0.0050157, 0.0011218, 0.0050405, -0.0026397, 0.0023462
6: 0.0002678, 0.0013135, 0.0002615, 0.0012561, -0.0005955, 0.0006700
7: -0.0024447, 0.0002609, -0.0024611, 0.0001123, -0.0015407, 0.0017334
8: -0.0008498, 0.0005731, -0.0008584, 0.0004949, -0.0008102, 0.0009116
9: -0.0025283, -0.0008785, -0.0024377, -0.0008685, -0.0010570, 0.0009395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008882, upper bound: 0.0010196
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009034, upper bound: 0.0010391
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938680, 0.9963219, -0.0013941, 0.0013941
1: -0.0027919, -0.0021805, -0.0027919, -0.0021805, -0.0003474, 0.0003474
2: 0.0015012, 0.0047415, 0.0015012, 0.0047415, -0.0018410, 0.0018410
3: -0.0034313, -0.0019564, -0.0034313, -0.0019564, -0.0008379, 0.0008379
4: 0.0008185, 0.0014456, 0.0008185, 0.0014456, -0.0003563, 0.0003563
5: 0.0008476, 0.0049230, 0.0008476, 0.0049230, -0.0023154, 0.0023154
6: 0.0002913, 0.0013257, 0.0002913, 0.0013257, -0.0005877, 0.0005877
7: -0.0023839, 0.0002924, -0.0023839, 0.0002924, -0.0015205, 0.0015205
8: -0.0008178, 0.0005896, -0.0008178, 0.0005896, -0.0007996, 0.0007996
9: -0.0025475, -0.0009156, -0.0025475, -0.0009156, -0.0009272, 0.0009272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0010161
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009194, upper bound: 0.0010387
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938680, 0.9963219, 0.9938123, 0.9962930, -0.0013641, 0.0014654
1: -0.0027919, -0.0021805, -0.0028058, -0.0021876, -0.0003399, 0.0003651
2: 0.0015012, 0.0047415, 0.0015393, 0.0048152, -0.0019350, 0.0018013
3: -0.0034313, -0.0019564, -0.0034648, -0.0019738, -0.0008199, 0.0008807
4: 0.0008185, 0.0014456, 0.0008258, 0.0014599, -0.0003745, 0.0003486
5: 0.0008476, 0.0049230, 0.0008955, 0.0050157, -0.0024338, 0.0022655
6: 0.0002913, 0.0013257, 0.0002678, 0.0013135, -0.0005750, 0.0006177
7: -0.0023839, 0.0002924, -0.0024447, 0.0002609, -0.0014877, 0.0015982
8: -0.0008178, 0.0005896, -0.0008498, 0.0005731, -0.0007824, 0.0008405
9: -0.0025475, -0.0009156, -0.0025283, -0.0008785, -0.0009746, 0.0009072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0010169
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009194, upper bound: 0.0010388
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9938680, 0.9963219, -0.0014654, 0.0013641
1: -0.0028058, -0.0021876, -0.0027919, -0.0021805, -0.0003651, 0.0003399
2: 0.0015393, 0.0048152, 0.0015012, 0.0047415, -0.0018013, 0.0019350
3: -0.0034648, -0.0019738, -0.0034313, -0.0019564, -0.0008807, 0.0008199
4: 0.0008258, 0.0014599, 0.0008185, 0.0014456, -0.0003486, 0.0003745
5: 0.0008955, 0.0050157, 0.0008476, 0.0049230, -0.0022655, 0.0024338
6: 0.0002678, 0.0013135, 0.0002913, 0.0013257, -0.0006177, 0.0005750
7: -0.0024447, 0.0002609, -0.0023839, 0.0002924, -0.0015982, 0.0014877
8: -0.0008498, 0.0005731, -0.0008178, 0.0005896, -0.0008405, 0.0007824
9: -0.0025283, -0.0008785, -0.0025475, -0.0009156, -0.0009072, 0.0009746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009029, upper bound: 0.0010161
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009143, upper bound: 0.0010389
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938123, 0.9962930, 0.9938123, 0.9962930, -0.0014304, 0.0014304
1: -0.0028058, -0.0021876, -0.0028058, -0.0021876, -0.0003564, 0.0003564
2: 0.0015393, 0.0048152, 0.0015393, 0.0048152, -0.0018888, 0.0018888
3: -0.0034648, -0.0019738, -0.0034648, -0.0019738, -0.0008597, 0.0008597
4: 0.0008258, 0.0014599, 0.0008258, 0.0014599, -0.0003656, 0.0003656
5: 0.0008955, 0.0050157, 0.0008955, 0.0050157, -0.0023756, 0.0023756
6: 0.0002678, 0.0013135, 0.0002678, 0.0013135, -0.0006029, 0.0006029
7: -0.0024447, 0.0002609, -0.0024447, 0.0002609, -0.0015600, 0.0015600
8: -0.0008498, 0.0005731, -0.0008498, 0.0005731, -0.0008204, 0.0008204
9: -0.0025283, -0.0008785, -0.0025283, -0.0008785, -0.0009513, 0.0009513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009029, upper bound: 0.0010220
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009143, upper bound: 0.0010426
time: 0.69 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.25 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0008534
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0008738
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0008535
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0008738
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0008587
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0008810
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0008940
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009151
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009253, upper bound: 0.0008505
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009367, upper bound: 0.0008711
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009253, upper bound: 0.0008506
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009367, upper bound: 0.0008712
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009220, upper bound: 0.0008537
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009348, upper bound: 0.0008772
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009220, upper bound: 0.0008886
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009348, upper bound: 0.0009062
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0009147
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0009348
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0009154
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0009348
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0009167
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008711, upper bound: 0.0009387
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0009445
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008711, upper bound: 0.0009613
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008774, upper bound: 0.0009211
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0009470
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008774, upper bound: 0.0009216
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0009471
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008747, upper bound: 0.0009234
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008956, upper bound: 0.0009517
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008747, upper bound: 0.0009478
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008956, upper bound: 0.0009653
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009627, upper bound: 0.0008534
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009752, upper bound: 0.0008738
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009627, upper bound: 0.0008535
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009752, upper bound: 0.0008738
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009656, upper bound: 0.0008587
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009775, upper bound: 0.0008810
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009656, upper bound: 0.0008940
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009775, upper bound: 0.0009145
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0008505
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010293, upper bound: 0.0008709
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0008506
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010293, upper bound: 0.0008710
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010195, upper bound: 0.0008537
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010284, upper bound: 0.0008772
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010195, upper bound: 0.0008886
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0010284, upper bound: 0.0009053
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009449, upper bound: 0.0009147
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009543, upper bound: 0.0009343
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009449, upper bound: 0.0009152
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009543, upper bound: 0.0009344
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009473, upper bound: 0.0009167
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009572, upper bound: 0.0009385
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009473, upper bound: 0.0009437
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009572, upper bound: 0.0009591
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009549, upper bound: 0.0009209
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009652, upper bound: 0.0009470
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009549, upper bound: 0.0009215
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009652, upper bound: 0.0009470
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009574, upper bound: 0.0009234
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009670, upper bound: 0.0009508
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009574, upper bound: 0.0009469
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009670, upper bound: 0.0009631
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0009409
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0009752
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0009448
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0009775
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0009410
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009753
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0009704
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009930
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009296
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009363, upper bound: 0.0009543
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009353
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009363, upper bound: 0.0009572
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009213, upper bound: 0.0009296
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009344, upper bound: 0.0009543
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009213, upper bound: 0.0009530
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009344, upper bound: 0.0009675
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0010033
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0010284
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0010052
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0010284
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0010033
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008710, upper bound: 0.0010286
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0010175
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008710, upper bound: 0.0010370
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0010072
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0010324
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0010087
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0010324
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008742, upper bound: 0.0010072
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008955, upper bound: 0.0010327
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008742, upper bound: 0.0010188
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008955, upper bound: 0.0010393
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009006, upper bound: 0.0009582
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009183, upper bound: 0.0009862
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009006, upper bound: 0.0009608
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009183, upper bound: 0.0009872
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008962, upper bound: 0.0009584
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009116, upper bound: 0.0009862
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008962, upper bound: 0.0009747
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009116, upper bound: 0.0009952
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009528, upper bound: 0.0009433
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009619, upper bound: 0.0009621
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009528, upper bound: 0.0009460
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009619, upper bound: 0.0009636
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009465, upper bound: 0.0009433
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009564, upper bound: 0.0009621
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009465, upper bound: 0.0009565
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009564, upper bound: 0.0009693
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008925, upper bound: 0.0010115
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010334
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008925, upper bound: 0.0010115
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010334
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008882, upper bound: 0.0010115
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009034, upper bound: 0.0010336
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0008882, upper bound: 0.0010196
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009034, upper bound: 0.0010391
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0010161
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009194, upper bound: 0.0010387
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0010169
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009194, upper bound: 0.0010388
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009029, upper bound: 0.0010161
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009143, upper bound: 0.0010389
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009029, upper bound: 0.0010220
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -0.0009143, upper bound: 0.0010426

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938554, 0.9958768, -0.0011608, 0.0012248
1: -0.0027854, -0.0022862, -0.0027950, -0.0022913, -0.0002892, 0.0003052
2: 0.0020615, 0.0047071, 0.0020888, 0.0047582, -0.0016173, 0.0015328
3: -0.0034156, -0.0022114, -0.0034388, -0.0022239, -0.0006977, 0.0007361
4: 0.0009269, 0.0014389, 0.0009322, 0.0014488, -0.0003130, 0.0002967
5: 0.0015523, 0.0048797, 0.0015866, 0.0049440, -0.0020341, 0.0019279
6: 0.0003023, 0.0011468, 0.0002860, 0.0011381, -0.0004893, 0.0005163
7: -0.0023555, -0.0001704, -0.0023977, -0.0001929, -0.0012660, 0.0013358
8: -0.0008029, 0.0003462, -0.0008251, 0.0003344, -0.0006658, 0.0007025
9: -0.0022653, -0.0009329, -0.0022516, -0.0009072, -0.0008145, 0.0007720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006758, upper bound: 0.0007326
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008456, upper bound: 0.0008364
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9938400, 0.9958824, -0.0011983, 0.0012245
1: -0.0027914, -0.0022909, -0.0027988, -0.0022900, -0.0002986, 0.0003051
2: 0.0020865, 0.0047389, 0.0020816, 0.0047784, -0.0016170, 0.0015824
3: -0.0034301, -0.0022228, -0.0034480, -0.0022206, -0.0007202, 0.0007360
4: 0.0009317, 0.0014451, 0.0009308, 0.0014527, -0.0003130, 0.0003063
5: 0.0015837, 0.0049198, 0.0015775, 0.0049694, -0.0020337, 0.0019902
6: 0.0002921, 0.0011389, 0.0002795, 0.0011404, -0.0005051, 0.0005162
7: -0.0023818, -0.0001910, -0.0024144, -0.0001870, -0.0013069, 0.0013355
8: -0.0008167, 0.0003354, -0.0008338, 0.0003375, -0.0006873, 0.0007023
9: -0.0022528, -0.0009169, -0.0022552, -0.0008970, -0.0008144, 0.0007970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0008636
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0008768
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938125, 0.9958426, -0.0011315, 0.0012794
1: -0.0027854, -0.0022862, -0.0028057, -0.0022999, -0.0002819, 0.0003188
2: 0.0020615, 0.0047071, 0.0021340, 0.0048148, -0.0016895, 0.0014942
3: -0.0034156, -0.0022114, -0.0034646, -0.0022444, -0.0006801, 0.0007690
4: 0.0009269, 0.0014389, 0.0009409, 0.0014598, -0.0003270, 0.0002892
5: 0.0015523, 0.0048797, 0.0016435, 0.0050152, -0.0021249, 0.0018793
6: 0.0003023, 0.0011468, 0.0002679, 0.0011237, -0.0004770, 0.0005393
7: -0.0023555, -0.0001704, -0.0024445, -0.0002303, -0.0012341, 0.0013954
8: -0.0008029, 0.0003462, -0.0008497, 0.0003147, -0.0006490, 0.0007338
9: -0.0022653, -0.0009329, -0.0022288, -0.0008786, -0.0008509, 0.0007525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006875, upper bound: 0.0007365
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008492, upper bound: 0.0008350
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9937946, 0.9958479, -0.0011681, 0.0012850
1: -0.0027914, -0.0022909, -0.0028102, -0.0022986, -0.0002911, 0.0003202
2: 0.0020865, 0.0047389, 0.0021272, 0.0048385, -0.0016969, 0.0015424
3: -0.0034301, -0.0022228, -0.0034754, -0.0022413, -0.0007021, 0.0007723
4: 0.0009317, 0.0014451, 0.0009396, 0.0014644, -0.0003284, 0.0002985
5: 0.0015837, 0.0049198, 0.0016349, 0.0050450, -0.0021342, 0.0019400
6: 0.0002921, 0.0011389, 0.0002604, 0.0011259, -0.0004924, 0.0005417
7: -0.0023818, -0.0001910, -0.0024640, -0.0002247, -0.0012740, 0.0014015
8: -0.0008167, 0.0003354, -0.0008599, 0.0003177, -0.0006700, 0.0007370
9: -0.0022528, -0.0009169, -0.0022322, -0.0008667, -0.0008546, 0.0007769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008587, upper bound: 0.0008606
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008587, upper bound: 0.0008738
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938554, 0.9958768, -0.0012088, 0.0011931
1: -0.0027962, -0.0022948, -0.0027950, -0.0022913, -0.0003012, 0.0002973
2: 0.0021073, 0.0047642, 0.0020888, 0.0047582, -0.0015755, 0.0015962
3: -0.0034416, -0.0022323, -0.0034388, -0.0022239, -0.0007265, 0.0007171
4: 0.0009358, 0.0014500, 0.0009322, 0.0014488, -0.0003049, 0.0003089
5: 0.0016099, 0.0049516, 0.0015866, 0.0049440, -0.0019816, 0.0020076
6: 0.0002841, 0.0011322, 0.0002860, 0.0011381, -0.0005095, 0.0005029
7: -0.0024026, -0.0002082, -0.0023977, -0.0001929, -0.0013183, 0.0013013
8: -0.0008277, 0.0003264, -0.0008251, 0.0003344, -0.0006933, 0.0006843
9: -0.0022423, -0.0009041, -0.0022516, -0.0009072, -0.0007935, 0.0008039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006810, upper bound: 0.0007461
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008428, upper bound: 0.0008405
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9938400, 0.9958824, -0.0012554, 0.0011951
1: -0.0028020, -0.0022995, -0.0027988, -0.0022900, -0.0003128, 0.0002978
2: 0.0021321, 0.0047952, 0.0020816, 0.0047784, -0.0015782, 0.0016577
3: -0.0034557, -0.0022436, -0.0034480, -0.0022206, -0.0007545, 0.0007183
4: 0.0009406, 0.0014560, 0.0009308, 0.0014527, -0.0003055, 0.0003208
5: 0.0016411, 0.0049906, 0.0015775, 0.0049694, -0.0019849, 0.0020850
6: 0.0002742, 0.0011243, 0.0002795, 0.0011404, -0.0005292, 0.0005038
7: -0.0024283, -0.0002287, -0.0024144, -0.0001870, -0.0013692, 0.0013035
8: -0.0008412, 0.0003156, -0.0008338, 0.0003375, -0.0007200, 0.0006855
9: -0.0022298, -0.0008885, -0.0022552, -0.0008970, -0.0007949, 0.0008349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008535, upper bound: 0.0008669
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008535, upper bound: 0.0008810
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938125, 0.9958426, -0.0011777, 0.0012445
1: -0.0027962, -0.0022948, -0.0028057, -0.0022999, -0.0002935, 0.0003101
2: 0.0021073, 0.0047642, 0.0021340, 0.0048148, -0.0016434, 0.0015552
3: -0.0034416, -0.0022323, -0.0034646, -0.0022444, -0.0007078, 0.0007480
4: 0.0009358, 0.0014500, 0.0009409, 0.0014598, -0.0003181, 0.0003010
5: 0.0016099, 0.0049516, 0.0016435, 0.0050152, -0.0020670, 0.0019560
6: 0.0002841, 0.0011322, 0.0002679, 0.0011237, -0.0004965, 0.0005246
7: -0.0024026, -0.0002082, -0.0024445, -0.0002303, -0.0012845, 0.0013573
8: -0.0008277, 0.0003264, -0.0008497, 0.0003147, -0.0006755, 0.0007138
9: -0.0022423, -0.0009041, -0.0022288, -0.0008786, -0.0008277, 0.0007833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007610, upper bound: 0.0008044
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008836, upper bound: 0.0008773
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9937946, 0.9958479, -0.0012194, 0.0012486
1: -0.0028020, -0.0022995, -0.0028102, -0.0022986, -0.0003038, 0.0003111
2: 0.0021321, 0.0047952, 0.0021272, 0.0048385, -0.0016487, 0.0016102
3: -0.0034557, -0.0022436, -0.0034754, -0.0022413, -0.0007329, 0.0007504
4: 0.0009406, 0.0014560, 0.0009396, 0.0014644, -0.0003191, 0.0003117
5: 0.0016411, 0.0049906, 0.0016349, 0.0050450, -0.0020736, 0.0020252
6: 0.0002742, 0.0011243, 0.0002604, 0.0011259, -0.0005140, 0.0005263
7: -0.0024283, -0.0002287, -0.0024640, -0.0002247, -0.0013299, 0.0013617
8: -0.0008412, 0.0003156, -0.0008599, 0.0003177, -0.0006994, 0.0007161
9: -0.0022298, -0.0008885, -0.0022322, -0.0008667, -0.0008304, 0.0008110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008923, upper bound: 0.0009025
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008923, upper bound: 0.0009151
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938613, 0.9960279, -0.0013720, 0.0012600
1: -0.0027854, -0.0022862, -0.0027936, -0.0022537, -0.0003419, 0.0003140
2: 0.0020615, 0.0047071, 0.0018894, 0.0047504, -0.0016638, 0.0018117
3: -0.0034156, -0.0022114, -0.0034353, -0.0021331, -0.0008246, 0.0007573
4: 0.0009269, 0.0014389, 0.0008936, 0.0014473, -0.0003220, 0.0003506
5: 0.0015523, 0.0048797, 0.0013358, 0.0049342, -0.0020926, 0.0022786
6: 0.0003023, 0.0011468, 0.0002885, 0.0012018, -0.0005783, 0.0005311
7: -0.0023555, -0.0001704, -0.0023912, -0.0000282, -0.0014963, 0.0013742
8: -0.0008029, 0.0003462, -0.0008217, 0.0004210, -0.0007869, 0.0007227
9: -0.0022653, -0.0009329, -0.0023520, -0.0009111, -0.0008380, 0.0009125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007936, upper bound: 0.0007326
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0008333
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9938470, 0.9960336, -0.0014104, 0.0012599
1: -0.0027914, -0.0022909, -0.0027971, -0.0022523, -0.0003514, 0.0003139
2: 0.0020865, 0.0047389, 0.0018820, 0.0047693, -0.0016637, 0.0018625
3: -0.0034301, -0.0022228, -0.0034439, -0.0021297, -0.0008477, 0.0007572
4: 0.0009317, 0.0014451, 0.0008921, 0.0014510, -0.0003220, 0.0003605
5: 0.0015837, 0.0049198, 0.0013265, 0.0049580, -0.0020925, 0.0023425
6: 0.0002921, 0.0011389, 0.0002824, 0.0012042, -0.0005945, 0.0005311
7: -0.0023818, -0.0001910, -0.0024069, -0.0000221, -0.0015383, 0.0013741
8: -0.0008167, 0.0003354, -0.0008299, 0.0004242, -0.0008090, 0.0007226
9: -0.0022528, -0.0009169, -0.0023558, -0.0009015, -0.0008379, 0.0009380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009153, upper bound: 0.0008597
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009153, upper bound: 0.0008732
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938198, 0.9959918, -0.0013472, 0.0013218
1: -0.0027854, -0.0022862, -0.0028039, -0.0022627, -0.0003357, 0.0003294
2: 0.0020615, 0.0047071, 0.0019370, 0.0048052, -0.0017454, 0.0017789
3: -0.0034156, -0.0022114, -0.0034602, -0.0021548, -0.0008097, 0.0007944
4: 0.0009269, 0.0014389, 0.0009028, 0.0014579, -0.0003378, 0.0003443
5: 0.0015523, 0.0048797, 0.0013957, 0.0050031, -0.0021953, 0.0022374
6: 0.0003023, 0.0011468, 0.0002710, 0.0011866, -0.0005679, 0.0005572
7: -0.0023555, -0.0001704, -0.0024365, -0.0000676, -0.0014693, 0.0014416
8: -0.0008029, 0.0003462, -0.0008455, 0.0004003, -0.0007727, 0.0007581
9: -0.0022653, -0.0009329, -0.0023280, -0.0008835, -0.0008791, 0.0008960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007965, upper bound: 0.0007365
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009085, upper bound: 0.0008323
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9938033, 0.9959973, -0.0013836, 0.0013240
1: -0.0027914, -0.0022909, -0.0028080, -0.0022613, -0.0003447, 0.0003299
2: 0.0020865, 0.0047389, 0.0019298, 0.0048270, -0.0017483, 0.0018270
3: -0.0034301, -0.0022228, -0.0034701, -0.0021515, -0.0008316, 0.0007958
4: 0.0009317, 0.0014451, 0.0009014, 0.0014621, -0.0003384, 0.0003536
5: 0.0015837, 0.0049198, 0.0013867, 0.0050305, -0.0021989, 0.0022978
6: 0.0002921, 0.0011389, 0.0002640, 0.0011889, -0.0005832, 0.0005581
7: -0.0023818, -0.0001910, -0.0024545, -0.0000616, -0.0015090, 0.0014440
8: -0.0008167, 0.0003354, -0.0008549, 0.0004034, -0.0007935, 0.0007594
9: -0.0022528, -0.0009169, -0.0023317, -0.0008725, -0.0008805, 0.0009202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009167, upper bound: 0.0008569
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009167, upper bound: 0.0008712
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938613, 0.9960279, -0.0014200, 0.0012283
1: -0.0027962, -0.0022948, -0.0027936, -0.0022537, -0.0003538, 0.0003061
2: 0.0021073, 0.0047642, 0.0018894, 0.0047504, -0.0016220, 0.0018750
3: -0.0034416, -0.0022323, -0.0034353, -0.0021331, -0.0008534, 0.0007383
4: 0.0009358, 0.0014500, 0.0008936, 0.0014473, -0.0003139, 0.0003629
5: 0.0016099, 0.0049516, 0.0013358, 0.0049342, -0.0020401, 0.0023583
6: 0.0002841, 0.0011322, 0.0002885, 0.0012018, -0.0005986, 0.0005178
7: -0.0024026, -0.0002082, -0.0023912, -0.0000282, -0.0015487, 0.0013397
8: -0.0008277, 0.0003264, -0.0008217, 0.0004210, -0.0008144, 0.0007045
9: -0.0022423, -0.0009041, -0.0023520, -0.0009111, -0.0008169, 0.0009444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008041, upper bound: 0.0007461
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009035, upper bound: 0.0008357
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9938470, 0.9960336, -0.0014675, 0.0012305
1: -0.0028020, -0.0022995, -0.0027971, -0.0022523, -0.0003657, 0.0003066
2: 0.0021321, 0.0047952, 0.0018820, 0.0047693, -0.0016249, 0.0019378
3: -0.0034557, -0.0022436, -0.0034439, -0.0021297, -0.0008820, 0.0007396
4: 0.0009406, 0.0014560, 0.0008921, 0.0014510, -0.0003145, 0.0003751
5: 0.0016411, 0.0049906, 0.0013265, 0.0049580, -0.0020437, 0.0024372
6: 0.0002742, 0.0011243, 0.0002824, 0.0012042, -0.0006186, 0.0005187
7: -0.0024283, -0.0002287, -0.0024069, -0.0000221, -0.0016005, 0.0013421
8: -0.0008412, 0.0003156, -0.0008299, 0.0004242, -0.0008417, 0.0007058
9: -0.0022298, -0.0008885, -0.0023558, -0.0009015, -0.0008184, 0.0009760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009154, upper bound: 0.0008625
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009154, upper bound: 0.0008772
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938198, 0.9959918, -0.0013894, 0.0012833
1: -0.0027962, -0.0022948, -0.0028039, -0.0022627, -0.0003462, 0.0003198
2: 0.0021073, 0.0047642, 0.0019370, 0.0048052, -0.0016946, 0.0018347
3: -0.0034416, -0.0022323, -0.0034602, -0.0021548, -0.0008351, 0.0007713
4: 0.0009358, 0.0014500, 0.0009028, 0.0014579, -0.0003280, 0.0003551
5: 0.0016099, 0.0049516, 0.0013957, 0.0050031, -0.0021313, 0.0023076
6: 0.0002841, 0.0011322, 0.0002710, 0.0011866, -0.0005857, 0.0005410
7: -0.0024026, -0.0002082, -0.0024365, -0.0000676, -0.0015153, 0.0013996
8: -0.0008277, 0.0003264, -0.0008455, 0.0004003, -0.0007969, 0.0007360
9: -0.0022423, -0.0009041, -0.0023280, -0.0008835, -0.0008535, 0.0009241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008555, upper bound: 0.0008040
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009334, upper bound: 0.0008717
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9938033, 0.9959973, -0.0014317, 0.0012857
1: -0.0028020, -0.0022995, -0.0028080, -0.0022613, -0.0003567, 0.0003204
2: 0.0021321, 0.0047952, 0.0019298, 0.0048270, -0.0016977, 0.0018905
3: -0.0034557, -0.0022436, -0.0034701, -0.0021515, -0.0008605, 0.0007727
4: 0.0009406, 0.0014560, 0.0009014, 0.0014621, -0.0003286, 0.0003659
5: 0.0016411, 0.0049906, 0.0013867, 0.0050305, -0.0021353, 0.0023777
6: 0.0002742, 0.0011243, 0.0002640, 0.0011889, -0.0006035, 0.0005420
7: -0.0024283, -0.0002287, -0.0024545, -0.0000616, -0.0015614, 0.0014022
8: -0.0008412, 0.0003156, -0.0008549, 0.0004034, -0.0008211, 0.0007374
9: -0.0022298, -0.0008885, -0.0023317, -0.0008725, -0.0008550, 0.0009521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009444, upper bound: 0.0008948
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009444, upper bound: 0.0009062
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938554, 0.9958768, -0.0011938, 0.0014312
1: -0.0027845, -0.0022479, -0.0027950, -0.0022913, -0.0002975, 0.0003566
2: 0.0018588, 0.0047025, 0.0020888, 0.0047582, -0.0018899, 0.0015764
3: -0.0034135, -0.0021192, -0.0034388, -0.0022239, -0.0007175, 0.0008602
4: 0.0008876, 0.0014381, 0.0009322, 0.0014488, -0.0003658, 0.0003051
5: 0.0012973, 0.0048740, 0.0015866, 0.0049440, -0.0023770, 0.0019827
6: 0.0003038, 0.0012116, 0.0002860, 0.0011381, -0.0005032, 0.0006033
7: -0.0023517, -0.0000029, -0.0023977, -0.0001929, -0.0013020, 0.0015609
8: -0.0008009, 0.0004343, -0.0008251, 0.0003344, -0.0006847, 0.0008209
9: -0.0023675, -0.0009352, -0.0022516, -0.0009072, -0.0009518, 0.0007940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006758, upper bound: 0.0008211
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008418, upper bound: 0.0008973
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9938400, 0.9958824, -0.0012342, 0.0014363
1: -0.0027895, -0.0022532, -0.0027988, -0.0022900, -0.0003075, 0.0003579
2: 0.0018870, 0.0047289, 0.0020816, 0.0047784, -0.0018966, 0.0016298
3: -0.0034255, -0.0021320, -0.0034480, -0.0022206, -0.0007418, 0.0008632
4: 0.0008931, 0.0014432, 0.0009308, 0.0014527, -0.0003671, 0.0003154
5: 0.0013328, 0.0049072, 0.0015775, 0.0049694, -0.0023854, 0.0020499
6: 0.0002953, 0.0012025, 0.0002795, 0.0011404, -0.0005203, 0.0006054
7: -0.0023735, -0.0000263, -0.0024144, -0.0001870, -0.0013461, 0.0015665
8: -0.0008124, 0.0004220, -0.0008338, 0.0003375, -0.0007079, 0.0008238
9: -0.0023532, -0.0009219, -0.0022552, -0.0008970, -0.0009552, 0.0008209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008517, upper bound: 0.0009253
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008517, upper bound: 0.0009367
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938125, 0.9958426, -0.0011645, 0.0014859
1: -0.0027845, -0.0022479, -0.0028057, -0.0022999, -0.0002902, 0.0003702
2: 0.0018588, 0.0047025, 0.0021340, 0.0048148, -0.0019621, 0.0015377
3: -0.0034135, -0.0021192, -0.0034646, -0.0022444, -0.0006999, 0.0008930
4: 0.0008876, 0.0014381, 0.0009409, 0.0014598, -0.0003798, 0.0002976
5: 0.0012973, 0.0048740, 0.0016435, 0.0050152, -0.0024677, 0.0019340
6: 0.0003038, 0.0012116, 0.0002679, 0.0011237, -0.0004909, 0.0006263
7: -0.0023517, -0.0000029, -0.0024445, -0.0002303, -0.0012701, 0.0016205
8: -0.0008009, 0.0004343, -0.0008497, 0.0003147, -0.0006679, 0.0008522
9: -0.0023675, -0.0009352, -0.0022288, -0.0008786, -0.0009882, 0.0007745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006875, upper bound: 0.0008271
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008449, upper bound: 0.0008971
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9937946, 0.9958479, -0.0012040, 0.0014968
1: -0.0027895, -0.0022532, -0.0028102, -0.0022986, -0.0003000, 0.0003730
2: 0.0018870, 0.0047289, 0.0021272, 0.0048385, -0.0019765, 0.0015899
3: -0.0034255, -0.0021320, -0.0034754, -0.0022413, -0.0007237, 0.0008996
4: 0.0008931, 0.0014432, 0.0009396, 0.0014644, -0.0003825, 0.0003077
5: 0.0013328, 0.0049072, 0.0016349, 0.0050450, -0.0024859, 0.0019997
6: 0.0002953, 0.0012025, 0.0002604, 0.0011259, -0.0005075, 0.0006309
7: -0.0023735, -0.0000263, -0.0024640, -0.0002247, -0.0013132, 0.0016324
8: -0.0008124, 0.0004220, -0.0008599, 0.0003177, -0.0006906, 0.0008585
9: -0.0023532, -0.0009219, -0.0022322, -0.0008667, -0.0009955, 0.0008008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008537, upper bound: 0.0009220
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008537, upper bound: 0.0009348
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938569, 0.9960149, 0.9938554, 0.9958768, -0.0012532, 0.0014044
1: -0.0027947, -0.0022569, -0.0027950, -0.0022913, -0.0003123, 0.0003499
2: 0.0019066, 0.0047562, 0.0020888, 0.0047582, -0.0018545, 0.0016548
3: -0.0034379, -0.0021409, -0.0034388, -0.0022239, -0.0007532, 0.0008441
4: 0.0008969, 0.0014484, 0.0009322, 0.0014488, -0.0003589, 0.0003203
5: 0.0013575, 0.0049415, 0.0015866, 0.0049440, -0.0023325, 0.0020813
6: 0.0002866, 0.0011963, 0.0002860, 0.0011381, -0.0005283, 0.0005920
7: -0.0023960, -0.0000425, -0.0023977, -0.0001929, -0.0013668, 0.0015317
8: -0.0008242, 0.0004135, -0.0008251, 0.0003344, -0.0007188, 0.0008055
9: -0.0023433, -0.0009081, -0.0022516, -0.0009072, -0.0009340, 0.0008334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006810, upper bound: 0.0008265
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008390, upper bound: 0.0008989
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938343, 0.9959934, 0.9938400, 0.9958824, -0.0012964, 0.0014110
1: -0.0028003, -0.0022623, -0.0027988, -0.0022900, -0.0003230, 0.0003516
2: 0.0019350, 0.0047860, 0.0020816, 0.0047784, -0.0018632, 0.0017119
3: -0.0034515, -0.0021539, -0.0034480, -0.0022206, -0.0007792, 0.0008480
4: 0.0009024, 0.0014542, 0.0009308, 0.0014527, -0.0003606, 0.0003313
5: 0.0013932, 0.0049789, 0.0015775, 0.0049694, -0.0023434, 0.0021531
6: 0.0002771, 0.0011872, 0.0002795, 0.0011404, -0.0005465, 0.0005948
7: -0.0024206, -0.0000659, -0.0024144, -0.0001870, -0.0014139, 0.0015389
8: -0.0008371, 0.0004012, -0.0008338, 0.0003375, -0.0007436, 0.0008093
9: -0.0023291, -0.0008932, -0.0022552, -0.0008970, -0.0009384, 0.0008622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008506, upper bound: 0.0009266
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008506, upper bound: 0.0009387
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938569, 0.9960149, 0.9938125, 0.9958426, -0.0012138, 0.0014519
1: -0.0027947, -0.0022569, -0.0028057, -0.0022999, -0.0003025, 0.0003618
2: 0.0019066, 0.0047562, 0.0021340, 0.0048148, -0.0019172, 0.0016029
3: -0.0034379, -0.0021409, -0.0034646, -0.0022444, -0.0007296, 0.0008726
4: 0.0008969, 0.0014484, 0.0009409, 0.0014598, -0.0003711, 0.0003102
5: 0.0013575, 0.0049415, 0.0016435, 0.0050152, -0.0024113, 0.0020160
6: 0.0002866, 0.0011963, 0.0002679, 0.0011237, -0.0005117, 0.0006120
7: -0.0023960, -0.0000425, -0.0024445, -0.0002303, -0.0013239, 0.0015835
8: -0.0008242, 0.0004135, -0.0008497, 0.0003147, -0.0006962, 0.0008327
9: -0.0023433, -0.0009081, -0.0022288, -0.0008786, -0.0009656, 0.0008073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007606, upper bound: 0.0008799
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008766, upper bound: 0.0009267
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938343, 0.9959934, 0.9937946, 0.9958479, -0.0012582, 0.0014607
1: -0.0028003, -0.0022623, -0.0028102, -0.0022986, -0.0003135, 0.0003640
2: 0.0019350, 0.0047860, 0.0021272, 0.0048385, -0.0019289, 0.0016615
3: -0.0034515, -0.0021539, -0.0034754, -0.0022413, -0.0007562, 0.0008779
4: 0.0009024, 0.0014542, 0.0009396, 0.0014644, -0.0003733, 0.0003216
5: 0.0013932, 0.0049789, 0.0016349, 0.0050450, -0.0024260, 0.0020897
6: 0.0002771, 0.0011872, 0.0002604, 0.0011259, -0.0005304, 0.0006157
7: -0.0024206, -0.0000659, -0.0024640, -0.0002247, -0.0013723, 0.0015931
8: -0.0008371, 0.0004012, -0.0008599, 0.0003177, -0.0007217, 0.0008378
9: -0.0023291, -0.0008932, -0.0022322, -0.0008667, -0.0009715, 0.0008368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008879, upper bound: 0.0009521
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008879, upper bound: 0.0009613
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938613, 0.9960279, -0.0012168, 0.0012774
1: -0.0027845, -0.0022479, -0.0027936, -0.0022537, -0.0003032, 0.0003183
2: 0.0018588, 0.0047025, 0.0018894, 0.0047504, -0.0016868, 0.0016067
3: -0.0034135, -0.0021192, -0.0034353, -0.0021331, -0.0007313, 0.0007678
4: 0.0008876, 0.0014381, 0.0008936, 0.0014473, -0.0003265, 0.0003110
5: 0.0012973, 0.0048740, 0.0013358, 0.0049342, -0.0021216, 0.0020209
6: 0.0003038, 0.0012116, 0.0002885, 0.0012018, -0.0005129, 0.0005385
7: -0.0023517, -0.0000029, -0.0023912, -0.0000282, -0.0013271, 0.0013932
8: -0.0008009, 0.0004343, -0.0008217, 0.0004210, -0.0006979, 0.0007327
9: -0.0023675, -0.0009352, -0.0023520, -0.0009111, -0.0008496, 0.0008092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007720, upper bound: 0.0008574
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008591, upper bound: 0.0009034
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9938470, 0.9960336, -0.0012540, 0.0012801
1: -0.0027895, -0.0022532, -0.0027971, -0.0022523, -0.0003125, 0.0003190
2: 0.0018870, 0.0047289, 0.0018820, 0.0047693, -0.0016904, 0.0016558
3: -0.0034255, -0.0021320, -0.0034439, -0.0021297, -0.0007537, 0.0007694
4: 0.0008931, 0.0014432, 0.0008921, 0.0014510, -0.0003272, 0.0003205
5: 0.0013328, 0.0049072, 0.0013265, 0.0049580, -0.0021261, 0.0020826
6: 0.0002953, 0.0012025, 0.0002824, 0.0012042, -0.0005286, 0.0005396
7: -0.0023735, -0.0000263, -0.0024069, -0.0000221, -0.0013676, 0.0013962
8: -0.0008124, 0.0004220, -0.0008299, 0.0004242, -0.0007192, 0.0007342
9: -0.0023532, -0.0009219, -0.0023558, -0.0009015, -0.0008514, 0.0008340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008707, upper bound: 0.0009325
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008707, upper bound: 0.0009503
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938198, 0.9959918, -0.0011872, 0.0013361
1: -0.0027845, -0.0022479, -0.0028039, -0.0022627, -0.0002958, 0.0003329
2: 0.0018588, 0.0047025, 0.0019370, 0.0048052, -0.0017643, 0.0015676
3: -0.0034135, -0.0021192, -0.0034602, -0.0021548, -0.0007135, 0.0008030
4: 0.0008876, 0.0014381, 0.0009028, 0.0014579, -0.0003415, 0.0003034
5: 0.0012973, 0.0048740, 0.0013957, 0.0050031, -0.0022190, 0.0019717
6: 0.0003038, 0.0012116, 0.0002710, 0.0011866, -0.0005004, 0.0005632
7: -0.0023517, -0.0000029, -0.0024365, -0.0000676, -0.0012948, 0.0014572
8: -0.0008009, 0.0004343, -0.0008455, 0.0004003, -0.0006809, 0.0007663
9: -0.0023675, -0.0009352, -0.0023280, -0.0008835, -0.0008886, 0.0007895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007813, upper bound: 0.0008589
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008605, upper bound: 0.0009027
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9938033, 0.9959973, -0.0012223, 0.0013428
1: -0.0027895, -0.0022532, -0.0028080, -0.0022613, -0.0003046, 0.0003346
2: 0.0018870, 0.0047289, 0.0019298, 0.0048270, -0.0017731, 0.0016140
3: -0.0034255, -0.0021320, -0.0034701, -0.0021515, -0.0007346, 0.0008070
4: 0.0008931, 0.0014432, 0.0009014, 0.0014621, -0.0003432, 0.0003124
5: 0.0013328, 0.0049072, 0.0013867, 0.0050305, -0.0022301, 0.0020300
6: 0.0002953, 0.0012025, 0.0002640, 0.0011889, -0.0005152, 0.0005660
7: -0.0023735, -0.0000263, -0.0024545, -0.0000616, -0.0013330, 0.0014645
8: -0.0008124, 0.0004220, -0.0008549, 0.0004034, -0.0007010, 0.0007702
9: -0.0023532, -0.0009219, -0.0023317, -0.0008725, -0.0008930, 0.0008129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008719, upper bound: 0.0009280
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008719, upper bound: 0.0009471
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938569, 0.9960149, 0.9938613, 0.9960279, -0.0012686, 0.0012483
1: -0.0027947, -0.0022569, -0.0027936, -0.0022537, -0.0003161, 0.0003111
2: 0.0019066, 0.0047562, 0.0018894, 0.0047504, -0.0016484, 0.0016752
3: -0.0034379, -0.0021409, -0.0034353, -0.0021331, -0.0007625, 0.0007503
4: 0.0008969, 0.0014484, 0.0008936, 0.0014473, -0.0003190, 0.0003242
5: 0.0013575, 0.0049415, 0.0013358, 0.0049342, -0.0020733, 0.0021070
6: 0.0002866, 0.0011963, 0.0002885, 0.0012018, -0.0005348, 0.0005262
7: -0.0023960, -0.0000425, -0.0023912, -0.0000282, -0.0013836, 0.0013615
8: -0.0008242, 0.0004135, -0.0008217, 0.0004210, -0.0007276, 0.0007160
9: -0.0023433, -0.0009081, -0.0023520, -0.0009111, -0.0008302, 0.0008437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007764, upper bound: 0.0008622
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008563, upper bound: 0.0009047
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938343, 0.9959934, 0.9938470, 0.9960336, -0.0013144, 0.0012502
1: -0.0028003, -0.0022623, -0.0027971, -0.0022523, -0.0003275, 0.0003115
2: 0.0019350, 0.0047860, 0.0018820, 0.0047693, -0.0016509, 0.0017357
3: -0.0034515, -0.0021539, -0.0034439, -0.0021297, -0.0007900, 0.0007514
4: 0.0009024, 0.0014542, 0.0008921, 0.0014510, -0.0003195, 0.0003359
5: 0.0013932, 0.0049789, 0.0013265, 0.0049580, -0.0020764, 0.0021830
6: 0.0002771, 0.0011872, 0.0002824, 0.0012042, -0.0005541, 0.0005270
7: -0.0024206, -0.0000659, -0.0024069, -0.0000221, -0.0014336, 0.0013635
8: -0.0008371, 0.0004012, -0.0008299, 0.0004242, -0.0007539, 0.0007171
9: -0.0023291, -0.0008932, -0.0023558, -0.0009015, -0.0008315, 0.0008742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008696, upper bound: 0.0009338
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008696, upper bound: 0.0009518
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938569, 0.9960149, 0.9938198, 0.9959918, -0.0012331, 0.0012986
1: -0.0027947, -0.0022569, -0.0028039, -0.0022627, -0.0003073, 0.0003236
2: 0.0019066, 0.0047562, 0.0019370, 0.0048052, -0.0017148, 0.0016283
3: -0.0034379, -0.0021409, -0.0034602, -0.0021548, -0.0007411, 0.0007805
4: 0.0008969, 0.0014484, 0.0009028, 0.0014579, -0.0003319, 0.0003152
5: 0.0013575, 0.0049415, 0.0013957, 0.0050031, -0.0021568, 0.0020480
6: 0.0002866, 0.0011963, 0.0002710, 0.0011866, -0.0005198, 0.0005474
7: -0.0023960, -0.0000425, -0.0024365, -0.0000676, -0.0013449, 0.0014164
8: -0.0008242, 0.0004135, -0.0008455, 0.0004003, -0.0007073, 0.0007448
9: -0.0023433, -0.0009081, -0.0023280, -0.0008835, -0.0008637, 0.0008201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008227, upper bound: 0.0008965
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008852, upper bound: 0.0009290
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938343, 0.9959934, 0.9938033, 0.9959973, -0.0012759, 0.0013029
1: -0.0028003, -0.0022623, -0.0028080, -0.0022613, -0.0003179, 0.0003246
2: 0.0019350, 0.0047860, 0.0019298, 0.0048270, -0.0017205, 0.0016848
3: -0.0034515, -0.0021539, -0.0034701, -0.0021515, -0.0007668, 0.0007831
4: 0.0009024, 0.0014542, 0.0009014, 0.0014621, -0.0003330, 0.0003261
5: 0.0013932, 0.0049789, 0.0013867, 0.0050305, -0.0021639, 0.0021190
6: 0.0002771, 0.0011872, 0.0002640, 0.0011889, -0.0005378, 0.0005492
7: -0.0024206, -0.0000659, -0.0024545, -0.0000616, -0.0013915, 0.0014210
8: -0.0008371, 0.0004012, -0.0008549, 0.0004034, -0.0007318, 0.0007473
9: -0.0023291, -0.0008932, -0.0023317, -0.0008725, -0.0008665, 0.0008486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008992, upper bound: 0.0009555
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008992, upper bound: 0.0009653
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938709, 0.9961782, -0.0015696, 0.0013008
1: -0.0027854, -0.0022862, -0.0027912, -0.0022163, -0.0003911, 0.0003241
2: 0.0020615, 0.0047071, 0.0016910, 0.0047377, -0.0017177, 0.0020727
3: -0.0034156, -0.0022114, -0.0034295, -0.0020428, -0.0009434, 0.0007818
4: 0.0009269, 0.0014389, 0.0008552, 0.0014449, -0.0003325, 0.0004012
5: 0.0015523, 0.0048797, 0.0010863, 0.0049182, -0.0021604, 0.0026069
6: 0.0003023, 0.0011468, 0.0002925, 0.0012651, -0.0006617, 0.0005483
7: -0.0023555, -0.0001704, -0.0023807, 0.0001356, -0.0017119, 0.0014187
8: -0.0008029, 0.0003462, -0.0008162, 0.0005072, -0.0009003, 0.0007461
9: -0.0022653, -0.0009329, -0.0024520, -0.0009175, -0.0008651, 0.0010439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007504, upper bound: 0.0007326
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009467, upper bound: 0.0008364
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9938574, 0.9961836, -0.0016093, 0.0012954
1: -0.0027914, -0.0022909, -0.0027945, -0.0022149, -0.0004010, 0.0003228
2: 0.0020865, 0.0047389, 0.0016839, 0.0047556, -0.0017105, 0.0021251
3: -0.0034301, -0.0022228, -0.0034376, -0.0020396, -0.0009672, 0.0007786
4: 0.0009317, 0.0014451, 0.0008538, 0.0014483, -0.0003311, 0.0004113
5: 0.0015837, 0.0049198, 0.0010774, 0.0049407, -0.0021514, 0.0026728
6: 0.0002921, 0.0011389, 0.0002868, 0.0012674, -0.0006784, 0.0005461
7: -0.0023818, -0.0001910, -0.0023955, 0.0001415, -0.0017552, 0.0014128
8: -0.0008167, 0.0003354, -0.0008239, 0.0005103, -0.0009230, 0.0007430
9: -0.0022528, -0.0009169, -0.0024555, -0.0009085, -0.0008615, 0.0010703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009409, upper bound: 0.0008636
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009409, upper bound: 0.0008768
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938157, 0.9961513, -0.0015374, 0.0013544
1: -0.0027854, -0.0022862, -0.0028049, -0.0022229, -0.0003831, 0.0003375
2: 0.0020615, 0.0047071, 0.0017264, 0.0048107, -0.0017885, 0.0020301
3: -0.0034156, -0.0022114, -0.0034627, -0.0020589, -0.0009240, 0.0008141
4: 0.0009269, 0.0014389, 0.0008620, 0.0014590, -0.0003462, 0.0003929
5: 0.0015523, 0.0048797, 0.0011308, 0.0050100, -0.0022495, 0.0025533
6: 0.0003023, 0.0011468, 0.0002692, 0.0012538, -0.0006481, 0.0005709
7: -0.0023555, -0.0001704, -0.0024410, 0.0001064, -0.0016767, 0.0014772
8: -0.0008029, 0.0003462, -0.0008479, 0.0004918, -0.0008818, 0.0007768
9: -0.0022653, -0.0009329, -0.0024341, -0.0008807, -0.0009008, 0.0010225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007577, upper bound: 0.0007363
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009468, upper bound: 0.0008350
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9937996, 0.9961565, -0.0015781, 0.0013513
1: -0.0027914, -0.0022909, -0.0028089, -0.0022217, -0.0003932, 0.0003367
2: 0.0020865, 0.0047389, 0.0017196, 0.0048318, -0.0017843, 0.0020839
3: -0.0034301, -0.0022228, -0.0034724, -0.0020558, -0.0009485, 0.0008122
4: 0.0009317, 0.0014451, 0.0008607, 0.0014631, -0.0003454, 0.0004033
5: 0.0015837, 0.0049198, 0.0011223, 0.0050366, -0.0022442, 0.0026209
6: 0.0002921, 0.0011389, 0.0002625, 0.0012560, -0.0006652, 0.0005696
7: -0.0023818, -0.0001910, -0.0024585, 0.0001120, -0.0017211, 0.0014738
8: -0.0008167, 0.0003354, -0.0008570, 0.0004948, -0.0009051, 0.0007750
9: -0.0022528, -0.0009169, -0.0024375, -0.0008701, -0.0008987, 0.0010495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009410, upper bound: 0.0008606
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009410, upper bound: 0.0008738
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938709, 0.9961782, -0.0016176, 0.0012692
1: -0.0027962, -0.0022948, -0.0027912, -0.0022163, -0.0004031, 0.0003162
2: 0.0021073, 0.0047642, 0.0016910, 0.0047377, -0.0016759, 0.0021361
3: -0.0034416, -0.0022323, -0.0034295, -0.0020428, -0.0009722, 0.0007628
4: 0.0009358, 0.0014500, 0.0008552, 0.0014449, -0.0003244, 0.0004134
5: 0.0016099, 0.0049516, 0.0010863, 0.0049182, -0.0021079, 0.0026866
6: 0.0002841, 0.0011322, 0.0002925, 0.0012651, -0.0006819, 0.0005350
7: -0.0024026, -0.0002082, -0.0023807, 0.0001356, -0.0017642, 0.0013842
8: -0.0008277, 0.0003264, -0.0008162, 0.0005072, -0.0009278, 0.0007279
9: -0.0022423, -0.0009041, -0.0024520, -0.0009175, -0.0008441, 0.0010758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007607, upper bound: 0.0007461
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009492, upper bound: 0.0008405
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9938574, 0.9961836, -0.0016664, 0.0012660
1: -0.0028020, -0.0022995, -0.0027945, -0.0022149, -0.0004152, 0.0003155
2: 0.0021321, 0.0047952, 0.0016839, 0.0047556, -0.0016717, 0.0022004
3: -0.0034557, -0.0022436, -0.0034376, -0.0020396, -0.0010015, 0.0007609
4: 0.0009406, 0.0014560, 0.0008538, 0.0014483, -0.0003236, 0.0004259
5: 0.0016411, 0.0049906, 0.0010774, 0.0049407, -0.0021026, 0.0027676
6: 0.0002742, 0.0011243, 0.0002868, 0.0012674, -0.0007024, 0.0005337
7: -0.0024283, -0.0002287, -0.0023955, 0.0001415, -0.0018174, 0.0013807
8: -0.0008412, 0.0003156, -0.0008239, 0.0005103, -0.0009558, 0.0007261
9: -0.0022298, -0.0008885, -0.0024555, -0.0009085, -0.0008420, 0.0011083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009448, upper bound: 0.0008669
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009448, upper bound: 0.0008810
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938157, 0.9961513, -0.0015862, 0.0013217
1: -0.0027962, -0.0022948, -0.0028049, -0.0022229, -0.0003952, 0.0003293
2: 0.0021073, 0.0047642, 0.0017264, 0.0048107, -0.0017453, 0.0020946
3: -0.0034416, -0.0022323, -0.0034627, -0.0020589, -0.0009534, 0.0007944
4: 0.0009358, 0.0014500, 0.0008620, 0.0014590, -0.0003378, 0.0004054
5: 0.0016099, 0.0049516, 0.0011308, 0.0050100, -0.0021951, 0.0026344
6: 0.0002841, 0.0011322, 0.0002692, 0.0012538, -0.0006686, 0.0005571
7: -0.0024026, -0.0002082, -0.0024410, 0.0001064, -0.0017300, 0.0014415
8: -0.0008277, 0.0003264, -0.0008479, 0.0004918, -0.0009098, 0.0007581
9: -0.0022423, -0.0009041, -0.0024341, -0.0008807, -0.0008790, 0.0010549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008374, upper bound: 0.0008040
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009714, upper bound: 0.0008773
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9937996, 0.9961565, -0.0016298, 0.0013182
1: -0.0028020, -0.0022995, -0.0028089, -0.0022217, -0.0004061, 0.0003284
2: 0.0021321, 0.0047952, 0.0017196, 0.0048318, -0.0017406, 0.0021521
3: -0.0034557, -0.0022436, -0.0034724, -0.0020558, -0.0009795, 0.0007922
4: 0.0009406, 0.0014560, 0.0008607, 0.0014631, -0.0003369, 0.0004165
5: 0.0016411, 0.0049906, 0.0011223, 0.0050366, -0.0021892, 0.0027068
6: 0.0002742, 0.0011243, 0.0002625, 0.0012560, -0.0006870, 0.0005556
7: -0.0024283, -0.0002287, -0.0024585, 0.0001120, -0.0017775, 0.0014376
8: -0.0008412, 0.0003156, -0.0008570, 0.0004948, -0.0009348, 0.0007560
9: -0.0022298, -0.0008885, -0.0024375, -0.0008701, -0.0008767, 0.0010839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009707, upper bound: 0.0009025
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009707, upper bound: 0.0009146
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938834, 0.9963160, -0.0017121, 0.0012786
1: -0.0027854, -0.0022862, -0.0027880, -0.0021819, -0.0004266, 0.0003186
2: 0.0020615, 0.0047071, 0.0015090, 0.0047211, -0.0016884, 0.0022609
3: -0.0034156, -0.0022114, -0.0034220, -0.0019600, -0.0010291, 0.0007685
4: 0.0009269, 0.0014389, 0.0008200, 0.0014416, -0.0003268, 0.0004376
5: 0.0015523, 0.0048797, 0.0008574, 0.0048974, -0.0021236, 0.0028436
6: 0.0003023, 0.0011468, 0.0002978, 0.0013232, -0.0007217, 0.0005390
7: -0.0023555, -0.0001704, -0.0023671, 0.0002859, -0.0018673, 0.0013945
8: -0.0008029, 0.0003462, -0.0008090, 0.0005862, -0.0009820, 0.0007334
9: -0.0022653, -0.0009329, -0.0025436, -0.0009258, -0.0008504, 0.0011387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008650, upper bound: 0.0007326
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010039, upper bound: 0.0008333
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9938701, 0.9963216, -0.0017516, 0.0012740
1: -0.0027914, -0.0022909, -0.0027914, -0.0021805, -0.0004365, 0.0003174
2: 0.0020865, 0.0047389, 0.0015016, 0.0047388, -0.0016822, 0.0023130
3: -0.0034301, -0.0022228, -0.0034300, -0.0019566, -0.0010528, 0.0007657
4: 0.0009317, 0.0014451, 0.0008185, 0.0014451, -0.0003256, 0.0004477
5: 0.0015837, 0.0049198, 0.0008481, 0.0049196, -0.0021158, 0.0029091
6: 0.0002921, 0.0011389, 0.0002922, 0.0013256, -0.0007384, 0.0005370
7: -0.0023818, -0.0001910, -0.0023817, 0.0002921, -0.0019104, 0.0013894
8: -0.0008167, 0.0003354, -0.0008166, 0.0005894, -0.0010047, 0.0007307
9: -0.0022528, -0.0009169, -0.0025473, -0.0009169, -0.0008473, 0.0011649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008597
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008732
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938942, 0.9958975, 0.9938318, 0.9962875, -0.0016839, 0.0013332
1: -0.0027854, -0.0022862, -0.0028009, -0.0021890, -0.0004196, 0.0003322
2: 0.0020615, 0.0047071, 0.0015466, 0.0047894, -0.0017605, 0.0022236
3: -0.0034156, -0.0022114, -0.0034530, -0.0019771, -0.0010121, 0.0008013
4: 0.0009269, 0.0014389, 0.0008272, 0.0014549, -0.0003407, 0.0004304
5: 0.0015523, 0.0048797, 0.0009047, 0.0049832, -0.0022142, 0.0027967
6: 0.0003023, 0.0011468, 0.0002760, 0.0013112, -0.0007098, 0.0005620
7: -0.0023555, -0.0001704, -0.0024235, 0.0002549, -0.0018365, 0.0014540
8: -0.0008029, 0.0003462, -0.0008386, 0.0005699, -0.0009658, 0.0007647
9: -0.0022653, -0.0009329, -0.0025247, -0.0008914, -0.0008867, 0.0011199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008661, upper bound: 0.0007363
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010039, upper bound: 0.0008323
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938700, 0.9958786, 0.9938145, 0.9962928, -0.0017230, 0.0013329
1: -0.0027914, -0.0022909, -0.0028052, -0.0021877, -0.0004293, 0.0003321
2: 0.0020865, 0.0047389, 0.0015397, 0.0048122, -0.0017600, 0.0022752
3: -0.0034301, -0.0022228, -0.0034634, -0.0019739, -0.0010356, 0.0008011
4: 0.0009317, 0.0014451, 0.0008259, 0.0014593, -0.0003406, 0.0004404
5: 0.0015837, 0.0049198, 0.0008960, 0.0050119, -0.0022137, 0.0028617
6: 0.0002921, 0.0011389, 0.0002687, 0.0013134, -0.0007263, 0.0005618
7: -0.0023818, -0.0001910, -0.0024423, 0.0002606, -0.0018792, 0.0014537
8: -0.0008167, 0.0003354, -0.0008485, 0.0005729, -0.0009883, 0.0007645
9: -0.0022528, -0.0009169, -0.0025282, -0.0008799, -0.0008864, 0.0011459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008568
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008710
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938834, 0.9963160, -0.0017601, 0.0012470
1: -0.0027962, -0.0022948, -0.0027880, -0.0021819, -0.0004386, 0.0003107
2: 0.0021073, 0.0047642, 0.0015090, 0.0047211, -0.0016466, 0.0023242
3: -0.0034416, -0.0022323, -0.0034220, -0.0019600, -0.0010579, 0.0007495
4: 0.0009358, 0.0014500, 0.0008200, 0.0014416, -0.0003187, 0.0004498
5: 0.0016099, 0.0049516, 0.0008574, 0.0048974, -0.0020710, 0.0029233
6: 0.0002841, 0.0011322, 0.0002978, 0.0013232, -0.0007420, 0.0005256
7: -0.0024026, -0.0002082, -0.0023671, 0.0002859, -0.0019197, 0.0013600
8: -0.0008277, 0.0003264, -0.0008090, 0.0005862, -0.0010095, 0.0007152
9: -0.0022423, -0.0009041, -0.0025436, -0.0009258, -0.0008293, 0.0011706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008755, upper bound: 0.0007461
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010022, upper bound: 0.0008357
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9938701, 0.9963216, -0.0018087, 0.0012446
1: -0.0028020, -0.0022995, -0.0027914, -0.0021805, -0.0004507, 0.0003101
2: 0.0021321, 0.0047952, 0.0015016, 0.0047388, -0.0016434, 0.0023883
3: -0.0034557, -0.0022436, -0.0034300, -0.0019566, -0.0010871, 0.0007480
4: 0.0009406, 0.0014560, 0.0008185, 0.0014451, -0.0003181, 0.0004623
5: 0.0016411, 0.0049906, 0.0008481, 0.0049196, -0.0020670, 0.0030039
6: 0.0002742, 0.0011243, 0.0002922, 0.0013256, -0.0007624, 0.0005246
7: -0.0024283, -0.0002287, -0.0023817, 0.0002921, -0.0019726, 0.0013574
8: -0.0008412, 0.0003156, -0.0008166, 0.0005894, -0.0010374, 0.0007138
9: -0.0022298, -0.0008885, -0.0025473, -0.0009169, -0.0008277, 0.0012029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0008625
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0008772
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938508, 0.9958628, 0.9938318, 0.9962875, -0.0017291, 0.0012998
1: -0.0027962, -0.0022948, -0.0028009, -0.0021890, -0.0004308, 0.0003239
2: 0.0021073, 0.0047642, 0.0015466, 0.0047894, -0.0017164, 0.0022832
3: -0.0034416, -0.0022323, -0.0034530, -0.0019771, -0.0010392, 0.0007812
4: 0.0009358, 0.0014500, 0.0008272, 0.0014549, -0.0003322, 0.0004419
5: 0.0016099, 0.0049516, 0.0009047, 0.0049832, -0.0021588, 0.0028717
6: 0.0002841, 0.0011322, 0.0002760, 0.0013112, -0.0007289, 0.0005479
7: -0.0024026, -0.0002082, -0.0024235, 0.0002549, -0.0018858, 0.0014176
8: -0.0008277, 0.0003264, -0.0008386, 0.0005699, -0.0009917, 0.0007455
9: -0.0022423, -0.0009041, -0.0025247, -0.0008914, -0.0008645, 0.0011500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009279, upper bound: 0.0008038
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010170, upper bound: 0.0008717
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938273, 0.9958441, 0.9938145, 0.9962928, -0.0017721, 0.0012991
1: -0.0028020, -0.0022995, -0.0028052, -0.0021877, -0.0004416, 0.0003237
2: 0.0021321, 0.0047952, 0.0015397, 0.0048122, -0.0017155, 0.0023401
3: -0.0034557, -0.0022436, -0.0034634, -0.0019739, -0.0010651, 0.0007808
4: 0.0009406, 0.0014560, 0.0008259, 0.0014593, -0.0003320, 0.0004529
5: 0.0016411, 0.0049906, 0.0008960, 0.0050119, -0.0021577, 0.0029432
6: 0.0002742, 0.0011243, 0.0002687, 0.0013134, -0.0007470, 0.0005476
7: -0.0024283, -0.0002287, -0.0024423, 0.0002606, -0.0019327, 0.0014169
8: -0.0008412, 0.0003156, -0.0008485, 0.0005729, -0.0010164, 0.0007451
9: -0.0022298, -0.0008885, -0.0025282, -0.0008799, -0.0008640, 0.0011786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010177, upper bound: 0.0008943
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010177, upper bound: 0.0009054
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938709, 0.9961782, -0.0016026, 0.0015073
1: -0.0027845, -0.0022479, -0.0027912, -0.0022163, -0.0003993, 0.0003756
2: 0.0018588, 0.0047025, 0.0016910, 0.0047377, -0.0019903, 0.0021163
3: -0.0034135, -0.0021192, -0.0034295, -0.0020428, -0.0009632, 0.0009059
4: 0.0008876, 0.0014381, 0.0008552, 0.0014449, -0.0003852, 0.0004096
5: 0.0012973, 0.0048740, 0.0010863, 0.0049182, -0.0025033, 0.0026617
6: 0.0003038, 0.0012116, 0.0002925, 0.0012651, -0.0006756, 0.0006354
7: -0.0023517, -0.0000029, -0.0023807, 0.0001356, -0.0017479, 0.0016439
8: -0.0008009, 0.0004343, -0.0008162, 0.0005072, -0.0009192, 0.0008645
9: -0.0023675, -0.0009352, -0.0024520, -0.0009175, -0.0010024, 0.0010659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007504, upper bound: 0.0008211
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009279, upper bound: 0.0008973
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9938574, 0.9961836, -0.0016452, 0.0015071
1: -0.0027895, -0.0022532, -0.0027945, -0.0022149, -0.0004100, 0.0003755
2: 0.0018870, 0.0047289, 0.0016839, 0.0047556, -0.0019901, 0.0021725
3: -0.0034255, -0.0021320, -0.0034376, -0.0020396, -0.0009888, 0.0009058
4: 0.0008931, 0.0014432, 0.0008538, 0.0014483, -0.0003852, 0.0004205
5: 0.0013328, 0.0049072, 0.0010774, 0.0049407, -0.0025031, 0.0027325
6: 0.0002953, 0.0012025, 0.0002868, 0.0012674, -0.0006935, 0.0006353
7: -0.0023735, -0.0000263, -0.0023955, 0.0001415, -0.0017944, 0.0016437
8: -0.0008124, 0.0004220, -0.0008239, 0.0005103, -0.0009436, 0.0008644
9: -0.0023532, -0.0009219, -0.0024555, -0.0009085, -0.0010023, 0.0010942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009248
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009363
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938157, 0.9961513, -0.0015703, 0.0015609
1: -0.0027845, -0.0022479, -0.0028049, -0.0022229, -0.0003913, 0.0003889
2: 0.0018588, 0.0047025, 0.0017264, 0.0048107, -0.0020611, 0.0020736
3: -0.0034135, -0.0021192, -0.0034627, -0.0020589, -0.0009438, 0.0009381
4: 0.0008876, 0.0014381, 0.0008620, 0.0014590, -0.0003989, 0.0004013
5: 0.0012973, 0.0048740, 0.0011308, 0.0050100, -0.0025924, 0.0026081
6: 0.0003038, 0.0012116, 0.0002692, 0.0012538, -0.0006620, 0.0006580
7: -0.0023517, -0.0000029, -0.0024410, 0.0001064, -0.0017127, 0.0017024
8: -0.0008009, 0.0004343, -0.0008479, 0.0004918, -0.0009007, 0.0008953
9: -0.0023675, -0.0009352, -0.0024341, -0.0008807, -0.0010381, 0.0010444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007577, upper bound: 0.0008268
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009279, upper bound: 0.0008970
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9937996, 0.9961565, -0.0016140, 0.0015630
1: -0.0027895, -0.0022532, -0.0028089, -0.0022217, -0.0004022, 0.0003895
2: 0.0018870, 0.0047289, 0.0017196, 0.0048318, -0.0020640, 0.0021313
3: -0.0034255, -0.0021320, -0.0034724, -0.0020558, -0.0009701, 0.0009394
4: 0.0008931, 0.0014432, 0.0008607, 0.0014631, -0.0003995, 0.0004125
5: 0.0013328, 0.0049072, 0.0011223, 0.0050366, -0.0025959, 0.0026806
6: 0.0002953, 0.0012025, 0.0002625, 0.0012560, -0.0006804, 0.0006589
7: -0.0023735, -0.0000263, -0.0024585, 0.0001120, -0.0017603, 0.0017047
8: -0.0008124, 0.0004220, -0.0008570, 0.0004948, -0.0009257, 0.0008965
9: -0.0023532, -0.0009219, -0.0024375, -0.0008701, -0.0010395, 0.0010734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009213
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009344
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938569, 0.9960149, 0.9938709, 0.9961782, -0.0016620, 0.0014805
1: -0.0027947, -0.0022569, -0.0027912, -0.0022163, -0.0004141, 0.0003689
2: 0.0019066, 0.0047562, 0.0016910, 0.0047377, -0.0019550, 0.0021947
3: -0.0034379, -0.0021409, -0.0034295, -0.0020428, -0.0009989, 0.0008898
4: 0.0008969, 0.0014484, 0.0008552, 0.0014449, -0.0003784, 0.0004248
5: 0.0013575, 0.0049415, 0.0010863, 0.0049182, -0.0024589, 0.0027603
6: 0.0002866, 0.0011963, 0.0002925, 0.0012651, -0.0007006, 0.0006241
7: -0.0023960, -0.0000425, -0.0023807, 0.0001356, -0.0018127, 0.0016147
8: -0.0008242, 0.0004135, -0.0008162, 0.0005072, -0.0009533, 0.0008492
9: -0.0023433, -0.0009081, -0.0024520, -0.0009175, -0.0009846, 0.0011053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007607, upper bound: 0.0008265
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009301, upper bound: 0.0008989
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938343, 0.9959934, 0.9938574, 0.9961836, -0.0017074, 0.0014818
1: -0.0028003, -0.0022623, -0.0027945, -0.0022149, -0.0004254, 0.0003692
2: 0.0019350, 0.0047860, 0.0016839, 0.0047556, -0.0019567, 0.0022546
3: -0.0034515, -0.0021539, -0.0034376, -0.0020396, -0.0010262, 0.0008906
4: 0.0009024, 0.0014542, 0.0008538, 0.0014483, -0.0003787, 0.0004364
5: 0.0013932, 0.0049789, 0.0010774, 0.0049407, -0.0024611, 0.0028357
6: 0.0002771, 0.0011872, 0.0002868, 0.0012674, -0.0007197, 0.0006246
7: -0.0024206, -0.0000659, -0.0023955, 0.0001415, -0.0018621, 0.0016161
8: -0.0008371, 0.0004012, -0.0008239, 0.0005103, -0.0009793, 0.0008499
9: -0.0023291, -0.0008932, -0.0024555, -0.0009085, -0.0009855, 0.0011355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009353, upper bound: 0.0009264
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009353, upper bound: 0.0009385
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938569, 0.9960149, 0.9938157, 0.9961513, -0.0016223, 0.0015290
1: -0.0027947, -0.0022569, -0.0028049, -0.0022229, -0.0004042, 0.0003810
2: 0.0019066, 0.0047562, 0.0017264, 0.0048107, -0.0020190, 0.0021423
3: -0.0034379, -0.0021409, -0.0034627, -0.0020589, -0.0009751, 0.0009190
4: 0.0008969, 0.0014484, 0.0008620, 0.0014590, -0.0003908, 0.0004146
5: 0.0013575, 0.0049415, 0.0011308, 0.0050100, -0.0025394, 0.0026944
6: 0.0002866, 0.0011963, 0.0002692, 0.0012538, -0.0006839, 0.0006445
7: -0.0023960, -0.0000425, -0.0024410, 0.0001064, -0.0017694, 0.0016676
8: -0.0008242, 0.0004135, -0.0008479, 0.0004918, -0.0009305, 0.0008770
9: -0.0023433, -0.0009081, -0.0024341, -0.0008807, -0.0010169, 0.0010790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008364, upper bound: 0.0008790
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009468, upper bound: 0.0009262
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938343, 0.9959934, 0.9937996, 0.9961565, -0.0016686, 0.0015303
1: -0.0028003, -0.0022623, -0.0028089, -0.0022217, -0.0004158, 0.0003813
2: 0.0019350, 0.0047860, 0.0017196, 0.0048318, -0.0020208, 0.0022034
3: -0.0034515, -0.0021539, -0.0034724, -0.0020558, -0.0010029, 0.0009198
4: 0.0009024, 0.0014542, 0.0008607, 0.0014631, -0.0003911, 0.0004265
5: 0.0013932, 0.0049789, 0.0011223, 0.0050366, -0.0025416, 0.0027713
6: 0.0002771, 0.0011872, 0.0002625, 0.0012560, -0.0007034, 0.0006451
7: -0.0024206, -0.0000659, -0.0024585, 0.0001120, -0.0018198, 0.0016690
8: -0.0008371, 0.0004012, -0.0008570, 0.0004948, -0.0009570, 0.0008777
9: -0.0023291, -0.0008932, -0.0024375, -0.0008701, -0.0010178, 0.0011097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009535, upper bound: 0.0009498
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009535, upper bound: 0.0009591
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938976, 0.9960512, 0.9938834, 0.9963160, -0.0016163, 0.0013510
1: -0.0027845, -0.0022479, -0.0027880, -0.0021819, -0.0004027, 0.0003366
2: 0.0018588, 0.0047025, 0.0015090, 0.0047211, -0.0017840, 0.0021343
3: -0.0034135, -0.0021192, -0.0034220, -0.0019600, -0.0009715, 0.0008120
4: 0.0008876, 0.0014381, 0.0008200, 0.0014416, -0.0003453, 0.0004131
5: 0.0012973, 0.0048740, 0.0008574, 0.0048974, -0.0022438, 0.0026844
6: 0.0003038, 0.0012116, 0.0002978, 0.0013232, -0.0006813, 0.0005695
7: -0.0023517, -0.0000029, -0.0023671, 0.0002859, -0.0017628, 0.0014735
8: -0.0008009, 0.0004343, -0.0008090, 0.0005862, -0.0009270, 0.0007749
9: -0.0023675, -0.0009352, -0.0025436, -0.0009258, -0.0008985, 0.0010750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008436, upper bound: 0.0008574
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009370, upper bound: 0.0009034
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938775, 0.9960297, 0.9938701, 0.9963216, -0.0016562, 0.0013481
1: -0.0027895, -0.0022532, -0.0027914, -0.0021805, -0.0004127, 0.0003359
2: 0.0018870, 0.0047289, 0.0015016, 0.0047388, -0.0017801, 0.0021870
3: -0.0034255, -0.0021320, -0.0034300, -0.0019566, -0.0009954, 0.0008102
4: 0.0008931, 0.0014432, 0.0008185, 0.0014451, -0.0003445, 0.0004233
5: 0.0013328, 0.0049072, 0.0008481, 0.0049196, -0.0022389, 0.0027507
6: 0.0002953, 0.0012025, 0.0002922, 0.0013256, -0.0006982, 0.0005683
7: -0.0023735, -0.0000263, -0.0023817, 0.0002921, -0.0018063, 0.0014702
8: -0.0008124, 0.0004220, -0.0008166, 0.0005894, -0.0009499, 0.0007732
9: -0.0023532, -0.0009219, -0.0025473, -0.0009169, -0.0008965, 0.0011015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009437, upper bound: 0.0009321
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009437, upper bound: 0.0009501
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.23 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0006758, upper bound: 0.0007326
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008456, upper bound: 0.0008364
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0008636
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0008768
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0006875, upper bound: 0.0007365
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008492, upper bound: 0.0008350
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008587, upper bound: 0.0008606
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008587, upper bound: 0.0008738
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0006810, upper bound: 0.0007461
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008428, upper bound: 0.0008405
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008535, upper bound: 0.0008669
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008535, upper bound: 0.0008810
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007610, upper bound: 0.0008044
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008836, upper bound: 0.0008773
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008923, upper bound: 0.0009025
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008923, upper bound: 0.0009151
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007936, upper bound: 0.0007326
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0008333
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009153, upper bound: 0.0008597
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009153, upper bound: 0.0008732
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007965, upper bound: 0.0007365
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009085, upper bound: 0.0008323
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009167, upper bound: 0.0008569
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009167, upper bound: 0.0008712
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008041, upper bound: 0.0007461
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009035, upper bound: 0.0008357
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009154, upper bound: 0.0008625
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009154, upper bound: 0.0008772
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008555, upper bound: 0.0008040
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009334, upper bound: 0.0008717
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009444, upper bound: 0.0008948
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009444, upper bound: 0.0009062
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0006758, upper bound: 0.0008211
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008418, upper bound: 0.0008973
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008517, upper bound: 0.0009253
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008517, upper bound: 0.0009367
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0006875, upper bound: 0.0008271
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008449, upper bound: 0.0008971
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008537, upper bound: 0.0009220
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008537, upper bound: 0.0009348
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0006810, upper bound: 0.0008265
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008390, upper bound: 0.0008989
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008506, upper bound: 0.0009266
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008506, upper bound: 0.0009387
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007606, upper bound: 0.0008799
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008766, upper bound: 0.0009267
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008879, upper bound: 0.0009521
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008879, upper bound: 0.0009613
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007720, upper bound: 0.0008574
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008591, upper bound: 0.0009034
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008707, upper bound: 0.0009325
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008707, upper bound: 0.0009503
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007813, upper bound: 0.0008589
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008605, upper bound: 0.0009027
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008719, upper bound: 0.0009280
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008719, upper bound: 0.0009471
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007764, upper bound: 0.0008622
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008563, upper bound: 0.0009047
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008696, upper bound: 0.0009338
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008696, upper bound: 0.0009518
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008227, upper bound: 0.0008965
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008852, upper bound: 0.0009290
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008992, upper bound: 0.0009555
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008992, upper bound: 0.0009653
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007504, upper bound: 0.0007326
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009467, upper bound: 0.0008364
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009409, upper bound: 0.0008636
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009409, upper bound: 0.0008768
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007577, upper bound: 0.0007363
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009468, upper bound: 0.0008350
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009410, upper bound: 0.0008606
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009410, upper bound: 0.0008738
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007607, upper bound: 0.0007461
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009492, upper bound: 0.0008405
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009448, upper bound: 0.0008669
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009448, upper bound: 0.0008810
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008374, upper bound: 0.0008040
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009714, upper bound: 0.0008773
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009707, upper bound: 0.0009025
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009707, upper bound: 0.0009146
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008650, upper bound: 0.0007326
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010039, upper bound: 0.0008333
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008597
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008732
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008661, upper bound: 0.0007363
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010039, upper bound: 0.0008323
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008568
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010033, upper bound: 0.0008710
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008755, upper bound: 0.0007461
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010022, upper bound: 0.0008357
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0008625
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010052, upper bound: 0.0008772
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009279, upper bound: 0.0008038
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010170, upper bound: 0.0008717
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010177, upper bound: 0.0008943
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0010177, upper bound: 0.0009054
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007504, upper bound: 0.0008211
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009279, upper bound: 0.0008973
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009248
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009363
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007577, upper bound: 0.0008268
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009279, upper bound: 0.0008970
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009213
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009296, upper bound: 0.0009344
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0007607, upper bound: 0.0008265
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009301, upper bound: 0.0008989
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009353, upper bound: 0.0009264
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009353, upper bound: 0.0009385
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008364, upper bound: 0.0008790
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009468, upper bound: 0.0009262
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009535, upper bound: 0.0009498
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009535, upper bound: 0.0009591
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0008436, upper bound: 0.0008574
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009370, upper bound: 0.0009034
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009437, upper bound: 0.0009321
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 0, lower bound: -0.0009437, upper bound: 0.0009501
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009549, upper bound: 0.0009215
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009652, upper bound: 0.0009470
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009574, upper bound: 0.0009234
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009670, upper bound: 0.0009508
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009574, upper bound: 0.0009469
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009670, upper bound: 0.0009631
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0009409
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0009752
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008636, upper bound: 0.0009448
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0009775
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0009410
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009753
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008606, upper bound: 0.0009704
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008738, upper bound: 0.0009930
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009296
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009363, upper bound: 0.0009543
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009353
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009363, upper bound: 0.0009572
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009213, upper bound: 0.0009296
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009344, upper bound: 0.0009543
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009213, upper bound: 0.0009530
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009344, upper bound: 0.0009675
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0010033
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0010284
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008597, upper bound: 0.0010052
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008732, upper bound: 0.0010284
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0010033
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008710, upper bound: 0.0010286
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0010175
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008710, upper bound: 0.0010370
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0010072
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0010324
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008768, upper bound: 0.0010087
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0010324
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008742, upper bound: 0.0010072
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008955, upper bound: 0.0010327
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008742, upper bound: 0.0010188
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008955, upper bound: 0.0010393
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009006, upper bound: 0.0009582
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009183, upper bound: 0.0009862
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009006, upper bound: 0.0009608
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009183, upper bound: 0.0009872
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008962, upper bound: 0.0009584
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009116, upper bound: 0.0009862
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008962, upper bound: 0.0009747
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009116, upper bound: 0.0009952
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009528, upper bound: 0.0009433
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009619, upper bound: 0.0009621
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009528, upper bound: 0.0009460
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009619, upper bound: 0.0009636
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009465, upper bound: 0.0009433
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009564, upper bound: 0.0009621
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009465, upper bound: 0.0009565
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009564, upper bound: 0.0009693
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008925, upper bound: 0.0010115
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010334
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008925, upper bound: 0.0010115
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010334
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008882, upper bound: 0.0010115
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009034, upper bound: 0.0010336
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0008882, upper bound: 0.0010196
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009034, upper bound: 0.0010391
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0010161
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009194, upper bound: 0.0010387
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009072, upper bound: 0.0010169
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009194, upper bound: 0.0010388
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009029, upper bound: 0.0010161
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009143, upper bound: 0.0010389
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009029, upper bound: 0.0010220
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 0, lower bound: -0.0009143, upper bound: 0.0010426

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.22 + 597.54 = 600.76 seconds
