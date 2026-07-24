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
Threshold: 0.00031548


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0089238, -0.0060976, -0.0089238, -0.0060976, -0.0019380, 0.0019380)
1: (-0.0054546, -0.0046578, -0.0054546, -0.0046578, -0.0005464, 0.0005464)
2: (-0.0016854, 0.0041935, -0.0016854, 0.0041935, -0.0040315, 0.0040315)
3: (0.0014043, 0.0021822, 0.0014043, 0.0021822, -0.0005335, 0.0005335)
4: (0.0029579, 0.0073514, 0.0029579, 0.0073514, -0.0030129, 0.0030129)
5: (0.9963281, 0.9975487, 0.9963281, 0.9975487, -0.0008371, 0.0008371)
6: (0.0045506, 0.0056586, 0.0045506, 0.0056586, -0.0007598, 0.0007598)
7: (-0.0063994, -0.0022646, -0.0063994, -0.0022646, -0.0028355, 0.0028355)
8: (-0.0074303, -0.0042122, -0.0074303, -0.0042122, -0.0022068, 0.0022068)
9: (-0.0036463, -0.0033687, -0.0036463, -0.0033687, -0.0001904, 0.0001904)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 1.96 = 3.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0005262, upper bound: 0.0005262

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004992, upper bound: 0.0005024
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005025, upper bound: 0.0005025
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.77 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 5, lower bound: -0.0004992, upper bound: 0.0005024
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 5, lower bound: -0.0005025, upper bound: 0.0005025

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0089549, -0.0062400, -0.0089218, -0.0061448, -0.0018342, 0.0017651
1: -0.0054634, -0.0046979, -0.0054540, -0.0046711, -0.0005171, 0.0004976
2: -0.0017502, 0.0038974, -0.0016813, 0.0040955, -0.0038156, 0.0036717
3: 0.0013957, 0.0021431, 0.0014048, 0.0021693, -0.0005049, 0.0004859
4: 0.0031792, 0.0073999, 0.0030312, 0.0073484, -0.0027440, 0.0028515
5: 0.9963895, 0.9975622, 0.9963484, 0.9975479, -0.0007624, 0.0007922
6: 0.0046064, 0.0056708, 0.0045691, 0.0056578, -0.0006920, 0.0007191
7: -0.0061911, -0.0022190, -0.0063305, -0.0022675, -0.0025824, 0.0026836
8: -0.0074658, -0.0043743, -0.0074281, -0.0042659, -0.0020886, 0.0020099
9: -0.0036323, -0.0033656, -0.0036417, -0.0033689, -0.0001734, 0.0001802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004987, upper bound: 0.0004987
time: 0.79 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004987, upper bound: 0.0005024
time: 0.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0089213, -0.0061696, -0.0089233, -0.0061111, -0.0019277, 0.0017337
1: -0.0054539, -0.0046781, -0.0054545, -0.0046616, -0.0005435, 0.0004888
2: -0.0016802, 0.0040438, -0.0016844, 0.0041655, -0.0040101, 0.0036064
3: 0.0014050, 0.0021624, 0.0014044, 0.0021785, -0.0005307, 0.0004773
4: 0.0030698, 0.0073475, 0.0029788, 0.0073507, -0.0026952, 0.0029969
5: 0.9963591, 0.9975476, 0.9963338, 0.9975486, -0.0007488, 0.0008326
6: 0.0045788, 0.0056576, 0.0045559, 0.0056584, -0.0006797, 0.0007558
7: -0.0062941, -0.0022683, -0.0063798, -0.0022653, -0.0025365, 0.0028204
8: -0.0074274, -0.0042941, -0.0074298, -0.0042275, -0.0021951, 0.0019742
9: -0.0036393, -0.0033689, -0.0036450, -0.0033687, -0.0001703, 0.0001894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005024, upper bound: 0.0004987
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005024, upper bound: 0.0005025
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -0.0004987, upper bound: 0.0004987
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -0.0004987, upper bound: 0.0005024
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -0.0005024, upper bound: 0.0004987
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 5, lower bound: -0.0005024, upper bound: 0.0005025

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089549, -0.0062400, -0.0089549, -0.0062400, -0.0017233, 0.0017233
1: -0.0054634, -0.0046979, -0.0054634, -0.0046979, -0.0004859, 0.0004859
2: -0.0017502, 0.0038974, -0.0017502, 0.0038974, -0.0035848, 0.0035848
3: 0.0013957, 0.0021431, 0.0013957, 0.0021431, -0.0004744, 0.0004744
4: 0.0031792, 0.0073999, 0.0031792, 0.0073999, -0.0026791, 0.0026791
5: 0.9963895, 0.9975622, 0.9963895, 0.9975622, -0.0007443, 0.0007443
6: 0.0046064, 0.0056708, 0.0046064, 0.0056708, -0.0006756, 0.0006756
7: -0.0061911, -0.0022190, -0.0061911, -0.0022190, -0.0025213, 0.0025213
8: -0.0074658, -0.0043743, -0.0074658, -0.0043743, -0.0019623, 0.0019623
9: -0.0036323, -0.0033656, -0.0036323, -0.0033656, -0.0001693, 0.0001693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004618, upper bound: 0.0004690
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004812
time: 0.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089549, -0.0062400, -0.0089213, -0.0061696, -0.0018563, 0.0017645
1: -0.0054634, -0.0046979, -0.0054539, -0.0046781, -0.0005234, 0.0004975
2: -0.0017502, 0.0038974, -0.0016802, 0.0040438, -0.0038615, 0.0036705
3: 0.0013957, 0.0021431, 0.0014050, 0.0021624, -0.0005110, 0.0004857
4: 0.0031792, 0.0073999, 0.0030698, 0.0073475, -0.0027431, 0.0028858
5: 0.9963895, 0.9975622, 0.9963591, 0.9975476, -0.0007621, 0.0008018
6: 0.0046064, 0.0056708, 0.0045788, 0.0056576, -0.0006918, 0.0007278
7: -0.0061911, -0.0022190, -0.0062941, -0.0022683, -0.0025815, 0.0027159
8: -0.0074658, -0.0043743, -0.0074274, -0.0042941, -0.0021138, 0.0020092
9: -0.0036323, -0.0033656, -0.0036393, -0.0033689, -0.0001733, 0.0001824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004618, upper bound: 0.0004735
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004846
time: 0.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089213, -0.0061696, -0.0089549, -0.0062400, -0.0017645, 0.0018563
1: -0.0054539, -0.0046781, -0.0054634, -0.0046979, -0.0004975, 0.0005234
2: -0.0016802, 0.0040438, -0.0017502, 0.0038974, -0.0036705, 0.0038615
3: 0.0014050, 0.0021624, 0.0013957, 0.0021431, -0.0004857, 0.0005110
4: 0.0030698, 0.0073475, 0.0031792, 0.0073999, -0.0028858, 0.0027431
5: 0.9963591, 0.9975476, 0.9963895, 0.9975622, -0.0008018, 0.0007621
6: 0.0045788, 0.0056576, 0.0046064, 0.0056708, -0.0007278, 0.0006918
7: -0.0062941, -0.0022683, -0.0061911, -0.0022190, -0.0027159, 0.0025815
8: -0.0074274, -0.0042941, -0.0074658, -0.0043743, -0.0020092, 0.0021138
9: -0.0036393, -0.0033689, -0.0036323, -0.0033656, -0.0001824, 0.0001733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004659, upper bound: 0.0004691
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004846, upper bound: 0.0004813
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089213, -0.0061696, -0.0089213, -0.0061696, -0.0017314, 0.0017314
1: -0.0054539, -0.0046781, -0.0054539, -0.0046781, -0.0004882, 0.0004882
2: -0.0016802, 0.0040438, -0.0016802, 0.0040438, -0.0036017, 0.0036017
3: 0.0014050, 0.0021624, 0.0014050, 0.0021624, -0.0004766, 0.0004766
4: 0.0030698, 0.0073475, 0.0030698, 0.0073475, -0.0026917, 0.0026917
5: 0.9963591, 0.9975476, 0.9963591, 0.9975476, -0.0007478, 0.0007478
6: 0.0045788, 0.0056576, 0.0045788, 0.0056576, -0.0006788, 0.0006788
7: -0.0062941, -0.0022683, -0.0062941, -0.0022683, -0.0025332, 0.0025332
8: -0.0074274, -0.0042941, -0.0074274, -0.0042941, -0.0019716, 0.0019716
9: -0.0036393, -0.0033689, -0.0036393, -0.0033689, -0.0001701, 0.0001701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004659, upper bound: 0.0004697
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004846, upper bound: 0.0004813
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004618, upper bound: 0.0004690
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004812
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004618, upper bound: 0.0004735
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004819, upper bound: 0.0004846
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004659, upper bound: 0.0004691
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004846, upper bound: 0.0004813
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004659, upper bound: 0.0004697
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 5, lower bound: -0.0004846, upper bound: 0.0004813

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090806, -0.0064212, -0.0089515, -0.0062903, -0.0016756, 0.0014660
1: -0.0054988, -0.0047490, -0.0054624, -0.0047121, -0.0004724, 0.0004133
2: -0.0020117, 0.0035204, -0.0017432, 0.0037928, -0.0034856, 0.0030496
3: 0.0013611, 0.0020932, 0.0013966, 0.0021292, -0.0004613, 0.0004036
4: 0.0034610, 0.0075953, 0.0032573, 0.0073946, -0.0022791, 0.0026049
5: 0.9964679, 0.9976164, 0.9964113, 0.9975607, -0.0006332, 0.0007237
6: 0.0046775, 0.0057201, 0.0046261, 0.0056695, -0.0005747, 0.0006569
7: -0.0059260, -0.0020351, -0.0061176, -0.0022240, -0.0021448, 0.0024515
8: -0.0076089, -0.0045807, -0.0074619, -0.0044315, -0.0019080, 0.0016693
9: -0.0036145, -0.0033533, -0.0036274, -0.0033660, -0.0001440, 0.0001646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003676, upper bound: 0.0003781
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003729
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089522, -0.0062709, -0.0089549, -0.0062400, -0.0017208, 0.0015071
1: -0.0054626, -0.0047067, -0.0054634, -0.0046979, -0.0004851, 0.0004249
2: -0.0017445, 0.0038332, -0.0017502, 0.0038974, -0.0035796, 0.0031351
3: 0.0013964, 0.0021346, 0.0013957, 0.0021431, -0.0004737, 0.0004149
4: 0.0032272, 0.0073956, 0.0031792, 0.0073999, -0.0023430, 0.0026751
5: 0.9964029, 0.9975609, 0.9963895, 0.9975622, -0.0006509, 0.0007432
6: 0.0046185, 0.0056697, 0.0046064, 0.0056708, -0.0005909, 0.0006746
7: -0.0061460, -0.0022230, -0.0061911, -0.0022190, -0.0022050, 0.0025176
8: -0.0074627, -0.0044094, -0.0074658, -0.0043743, -0.0019595, 0.0017162
9: -0.0036293, -0.0033659, -0.0036323, -0.0033656, -0.0001481, 0.0001691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004691, upper bound: 0.0004618
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004691, upper bound: 0.0004819
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090806, -0.0064212, -0.0089179, -0.0062185, -0.0018121, 0.0015085
1: -0.0054988, -0.0047490, -0.0054529, -0.0046919, -0.0005109, 0.0004253
2: -0.0020117, 0.0035204, -0.0016732, 0.0039420, -0.0037696, 0.0031379
3: 0.0013611, 0.0020932, 0.0014059, 0.0021490, -0.0004988, 0.0004153
4: 0.0034610, 0.0075953, 0.0031459, 0.0073423, -0.0023451, 0.0028172
5: 0.9964679, 0.9976164, 0.9963802, 0.9975461, -0.0006515, 0.0007827
6: 0.0046775, 0.0057201, 0.0045980, 0.0056563, -0.0005914, 0.0007105
7: -0.0059260, -0.0020351, -0.0062225, -0.0022732, -0.0022070, 0.0026513
8: -0.0076089, -0.0045807, -0.0074236, -0.0043499, -0.0020635, 0.0017177
9: -0.0036145, -0.0033533, -0.0036345, -0.0033693, -0.0001482, 0.0001780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003676, upper bound: 0.0003853
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003808
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089522, -0.0062709, -0.0089213, -0.0061696, -0.0018538, 0.0015743
1: -0.0054626, -0.0047067, -0.0054539, -0.0046781, -0.0005226, 0.0004439
2: -0.0017445, 0.0038332, -0.0016802, 0.0040438, -0.0038562, 0.0032749
3: 0.0013964, 0.0021346, 0.0014050, 0.0021624, -0.0005103, 0.0004334
4: 0.0032272, 0.0073956, 0.0030698, 0.0073475, -0.0024474, 0.0028819
5: 0.9964029, 0.9975609, 0.9963591, 0.9975476, -0.0006800, 0.0008007
6: 0.0046185, 0.0056697, 0.0045788, 0.0056576, -0.0006172, 0.0007268
7: -0.0061460, -0.0022230, -0.0062941, -0.0022683, -0.0023033, 0.0027122
8: -0.0074627, -0.0044094, -0.0074274, -0.0042941, -0.0021109, 0.0017927
9: -0.0036293, -0.0033659, -0.0036393, -0.0033689, -0.0001547, 0.0001821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004693, upper bound: 0.0004659
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004693, upper bound: 0.0004846
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090508, -0.0063474, -0.0089515, -0.0062903, -0.0017435, 0.0016151
1: -0.0054904, -0.0047282, -0.0054624, -0.0047121, -0.0004916, 0.0004553
2: -0.0019498, 0.0036739, -0.0017432, 0.0037928, -0.0036268, 0.0033597
3: 0.0013693, 0.0021135, 0.0013966, 0.0021292, -0.0004800, 0.0004446
4: 0.0033462, 0.0075490, 0.0032573, 0.0073946, -0.0025108, 0.0027105
5: 0.9964359, 0.9976036, 0.9964113, 0.9975607, -0.0006976, 0.0007530
6: 0.0046485, 0.0057084, 0.0046261, 0.0056695, -0.0006332, 0.0006835
7: -0.0060340, -0.0020787, -0.0061176, -0.0022240, -0.0023630, 0.0025508
8: -0.0075750, -0.0044966, -0.0074619, -0.0044315, -0.0019853, 0.0018391
9: -0.0036218, -0.0033562, -0.0036274, -0.0033660, -0.0001587, 0.0001713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003780
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003588, upper bound: 0.0003729
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089186, -0.0062004, -0.0089549, -0.0062400, -0.0017622, 0.0016538
1: -0.0054531, -0.0046868, -0.0054634, -0.0046979, -0.0004968, 0.0004663
2: -0.0016747, 0.0039797, -0.0017502, 0.0038974, -0.0036658, 0.0034402
3: 0.0014057, 0.0021539, 0.0013957, 0.0021431, -0.0004851, 0.0004553
4: 0.0031177, 0.0073434, 0.0031792, 0.0073999, -0.0025710, 0.0027396
5: 0.9963724, 0.9975464, 0.9963895, 0.9975622, -0.0007143, 0.0007611
6: 0.0045909, 0.0056566, 0.0046064, 0.0056708, -0.0006484, 0.0006909
7: -0.0062490, -0.0022722, -0.0061911, -0.0022190, -0.0024196, 0.0025783
8: -0.0074244, -0.0043292, -0.0074658, -0.0043743, -0.0020067, 0.0018832
9: -0.0036362, -0.0033692, -0.0036323, -0.0033656, -0.0001625, 0.0001731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004618
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004819
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090508, -0.0063474, -0.0089179, -0.0062185, -0.0016859, 0.0014748
1: -0.0054904, -0.0047282, -0.0054529, -0.0046919, -0.0004753, 0.0004158
2: -0.0019498, 0.0036739, -0.0016732, 0.0039420, -0.0035070, 0.0030679
3: 0.0013693, 0.0021135, 0.0014059, 0.0021490, -0.0004641, 0.0004060
4: 0.0033462, 0.0075490, 0.0031459, 0.0073423, -0.0022928, 0.0026209
5: 0.9964359, 0.9976036, 0.9963802, 0.9975461, -0.0006370, 0.0007282
6: 0.0046485, 0.0057084, 0.0045980, 0.0056563, -0.0005782, 0.0006610
7: -0.0060340, -0.0020787, -0.0062225, -0.0022732, -0.0021577, 0.0024665
8: -0.0075750, -0.0044966, -0.0074236, -0.0043499, -0.0019197, 0.0016794
9: -0.0036218, -0.0033562, -0.0036345, -0.0033693, -0.0001449, 0.0001656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003843
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003619, upper bound: 0.0003801
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089186, -0.0062004, -0.0089213, -0.0061696, -0.0017291, 0.0015189
1: -0.0054531, -0.0046868, -0.0054539, -0.0046781, -0.0004875, 0.0004282
2: -0.0016747, 0.0039797, -0.0016802, 0.0040438, -0.0035968, 0.0031596
3: 0.0014057, 0.0021539, 0.0014050, 0.0021624, -0.0004760, 0.0004181
4: 0.0031177, 0.0073434, 0.0030698, 0.0073475, -0.0023613, 0.0026880
5: 0.9963724, 0.9975464, 0.9963591, 0.9975476, -0.0006560, 0.0007468
6: 0.0045909, 0.0056566, 0.0045788, 0.0056576, -0.0005955, 0.0006779
7: -0.0062490, -0.0022722, -0.0062941, -0.0022683, -0.0022222, 0.0025297
8: -0.0074244, -0.0043292, -0.0074274, -0.0042941, -0.0019689, 0.0017296
9: -0.0036362, -0.0033692, -0.0036393, -0.0033689, -0.0001492, 0.0001699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004737, upper bound: 0.0004625
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004737, upper bound: 0.0004815
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.00 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003676, upper bound: 0.0003781
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003729
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004691, upper bound: 0.0004618
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004691, upper bound: 0.0004819
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003676, upper bound: 0.0003853
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003808
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004693, upper bound: 0.0004659
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004693, upper bound: 0.0004846
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003780
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003588, upper bound: 0.0003729
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004618
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004734, upper bound: 0.0004819
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003754, upper bound: 0.0003843
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0003619, upper bound: 0.0003801
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004737, upper bound: 0.0004625
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 5, lower bound: -0.0004737, upper bound: 0.0004815

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090806, -0.0064212, -0.0089506, -0.0063330, -0.0016238, 0.0014632
1: -0.0054988, -0.0047490, -0.0054622, -0.0047242, -0.0004578, 0.0004125
2: -0.0020117, 0.0035204, -0.0017412, 0.0037039, -0.0033778, 0.0030438
3: 0.0013611, 0.0020932, 0.0013969, 0.0021174, -0.0004470, 0.0004028
4: 0.0034610, 0.0075953, 0.0033238, 0.0073931, -0.0022748, 0.0025243
5: 0.9964679, 0.9976164, 0.9964296, 0.9975603, -0.0006320, 0.0007013
6: 0.0046775, 0.0057201, 0.0046429, 0.0056691, -0.0005737, 0.0006366
7: -0.0059260, -0.0020351, -0.0060550, -0.0022254, -0.0021408, 0.0023757
8: -0.0076089, -0.0045807, -0.0074608, -0.0044802, -0.0018490, 0.0016662
9: -0.0036145, -0.0033533, -0.0036232, -0.0033661, -0.0001438, 0.0001595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003729
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003729
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090803, -0.0064518, -0.0090847, -0.0063735, -0.0016532, 0.0016257
1: -0.0054987, -0.0047577, -0.0055000, -0.0047356, -0.0004661, 0.0004583
2: -0.0020111, 0.0034568, -0.0020202, 0.0036196, -0.0034391, 0.0033818
3: 0.0013612, 0.0020847, 0.0013599, 0.0021063, -0.0004551, 0.0004475
4: 0.0035085, 0.0075949, 0.0033868, 0.0076017, -0.0025273, 0.0025702
5: 0.9964809, 0.9976162, 0.9964471, 0.9976183, -0.0007022, 0.0007141
6: 0.0046895, 0.0057200, 0.0046588, 0.0057217, -0.0006374, 0.0006482
7: -0.0058813, -0.0020355, -0.0059958, -0.0020291, -0.0023785, 0.0024188
8: -0.0076086, -0.0046155, -0.0076136, -0.0045263, -0.0018826, 0.0018512
9: -0.0036115, -0.0033533, -0.0036192, -0.0033529, -0.0001597, 0.0001624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003313, upper bound: 0.0003338
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003642
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089522, -0.0062709, -0.0090806, -0.0064212, -0.0014663, 0.0017396
1: -0.0054626, -0.0047067, -0.0054988, -0.0047490, -0.0004134, 0.0004905
2: -0.0017445, 0.0038332, -0.0020117, 0.0035204, -0.0030502, 0.0036188
3: 0.0013964, 0.0021346, 0.0013611, 0.0020932, -0.0004037, 0.0004789
4: 0.0032272, 0.0073956, 0.0034610, 0.0075953, -0.0027045, 0.0022796
5: 0.9964029, 0.9975609, 0.9964679, 0.9976164, -0.0007514, 0.0006333
6: 0.0046185, 0.0056697, 0.0046775, 0.0057201, -0.0006820, 0.0005749
7: -0.0061460, -0.0022230, -0.0059260, -0.0020351, -0.0025452, 0.0021453
8: -0.0074627, -0.0044094, -0.0076089, -0.0045807, -0.0016697, 0.0019809
9: -0.0036293, -0.0033659, -0.0036145, -0.0033533, -0.0001709, 0.0001441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0003676
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003537
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089522, -0.0062709, -0.0089522, -0.0062709, -0.0015050, 0.0015050
1: -0.0054626, -0.0047067, -0.0054626, -0.0047067, -0.0004243, 0.0004243
2: -0.0017445, 0.0038332, -0.0017445, 0.0038332, -0.0031307, 0.0031307
3: 0.0013964, 0.0021346, 0.0013964, 0.0021346, -0.0004143, 0.0004143
4: 0.0032272, 0.0073956, 0.0032272, 0.0073956, -0.0023397, 0.0023397
5: 0.9964029, 0.9975609, 0.9964029, 0.9975609, -0.0006500, 0.0006500
6: 0.0046185, 0.0056697, 0.0046185, 0.0056697, -0.0005900, 0.0005900
7: -0.0061460, -0.0022230, -0.0061460, -0.0022230, -0.0022019, 0.0022019
8: -0.0074627, -0.0044094, -0.0074627, -0.0044094, -0.0017138, 0.0017138
9: -0.0036293, -0.0033659, -0.0036293, -0.0033659, -0.0001479, 0.0001479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0004576
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0004575
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090806, -0.0064212, -0.0089161, -0.0062571, -0.0017686, 0.0015061
1: -0.0054988, -0.0047490, -0.0054524, -0.0047028, -0.0004986, 0.0004246
2: -0.0020117, 0.0035204, -0.0016694, 0.0038617, -0.0036791, 0.0031330
3: 0.0013611, 0.0020932, 0.0014064, 0.0021383, -0.0004869, 0.0004146
4: 0.0034610, 0.0075953, 0.0032059, 0.0073395, -0.0023414, 0.0027496
5: 0.9964679, 0.9976164, 0.9963970, 0.9975454, -0.0006505, 0.0007639
6: 0.0046775, 0.0057201, 0.0046131, 0.0056556, -0.0005905, 0.0006934
7: -0.0059260, -0.0020351, -0.0061661, -0.0022759, -0.0022036, 0.0025876
8: -0.0076089, -0.0045807, -0.0074216, -0.0043938, -0.0020140, 0.0017150
9: -0.0036145, -0.0033533, -0.0036307, -0.0033694, -0.0001480, 0.0001738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003808
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003808
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090803, -0.0064518, -0.0090450, -0.0063063, -0.0018006, 0.0016407
1: -0.0054987, -0.0047577, -0.0054888, -0.0047166, -0.0005077, 0.0004626
2: -0.0020111, 0.0034568, -0.0019376, 0.0037594, -0.0037456, 0.0034130
3: 0.0013612, 0.0020847, 0.0013709, 0.0021248, -0.0004957, 0.0004517
4: 0.0035085, 0.0075949, 0.0032823, 0.0075399, -0.0025506, 0.0027992
5: 0.9964809, 0.9976162, 0.9964182, 0.9976010, -0.0007086, 0.0007777
6: 0.0046895, 0.0057200, 0.0046324, 0.0057061, -0.0006432, 0.0007059
7: -0.0058813, -0.0020355, -0.0060941, -0.0020872, -0.0024004, 0.0026344
8: -0.0076086, -0.0046155, -0.0075684, -0.0044498, -0.0020503, 0.0018683
9: -0.0036115, -0.0033533, -0.0036258, -0.0033568, -0.0001612, 0.0001769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003312, upper bound: 0.0003391
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003721
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089522, -0.0062709, -0.0090508, -0.0063474, -0.0016154, 0.0018062
1: -0.0054626, -0.0047067, -0.0054904, -0.0047282, -0.0004554, 0.0005092
2: -0.0017445, 0.0038332, -0.0019498, 0.0036739, -0.0033604, 0.0037573
3: 0.0013964, 0.0021346, 0.0013693, 0.0021135, -0.0004447, 0.0004972
4: 0.0032272, 0.0073956, 0.0033462, 0.0075490, -0.0028080, 0.0025113
5: 0.9964029, 0.9975609, 0.9964359, 0.9976036, -0.0007801, 0.0006977
6: 0.0046185, 0.0056697, 0.0046485, 0.0057084, -0.0007081, 0.0006333
7: -0.0061460, -0.0022230, -0.0060340, -0.0020787, -0.0026426, 0.0023634
8: -0.0074627, -0.0044094, -0.0075750, -0.0044966, -0.0018395, 0.0020567
9: -0.0036293, -0.0033659, -0.0036218, -0.0033562, -0.0001774, 0.0001587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0003729
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003588
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089522, -0.0062709, -0.0089186, -0.0062004, -0.0016517, 0.0015721
1: -0.0054626, -0.0047067, -0.0054531, -0.0046868, -0.0004657, 0.0004432
2: -0.0017445, 0.0038332, -0.0016747, 0.0039797, -0.0034358, 0.0032704
3: 0.0013964, 0.0021346, 0.0014057, 0.0021539, -0.0004547, 0.0004328
4: 0.0032272, 0.0073956, 0.0031177, 0.0073434, -0.0024441, 0.0025677
5: 0.9964029, 0.9975609, 0.9963724, 0.9975464, -0.0006790, 0.0007134
6: 0.0046185, 0.0056697, 0.0045909, 0.0056566, -0.0006164, 0.0006475
7: -0.0061460, -0.0022230, -0.0062490, -0.0022722, -0.0023002, 0.0024165
8: -0.0074627, -0.0044094, -0.0074244, -0.0043292, -0.0018808, 0.0017902
9: -0.0036293, -0.0033659, -0.0036362, -0.0033692, -0.0001545, 0.0001623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0004603
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0004604
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090508, -0.0063474, -0.0089506, -0.0063330, -0.0016934, 0.0016123
1: -0.0054904, -0.0047282, -0.0054622, -0.0047242, -0.0004774, 0.0004546
2: -0.0019498, 0.0036739, -0.0017412, 0.0037039, -0.0035225, 0.0033540
3: 0.0013693, 0.0021135, 0.0013969, 0.0021174, -0.0004661, 0.0004438
4: 0.0033462, 0.0075490, 0.0033238, 0.0073931, -0.0025065, 0.0026325
5: 0.9964359, 0.9976036, 0.9964296, 0.9975603, -0.0006964, 0.0007314
6: 0.0046485, 0.0057084, 0.0046429, 0.0056691, -0.0006321, 0.0006639
7: -0.0060340, -0.0020787, -0.0060550, -0.0022254, -0.0023589, 0.0024775
8: -0.0075750, -0.0044966, -0.0074608, -0.0044802, -0.0019282, 0.0018360
9: -0.0036218, -0.0033562, -0.0036232, -0.0033661, -0.0001584, 0.0001664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003588, upper bound: 0.0003729
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003588, upper bound: 0.0003729
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090501, -0.0063816, -0.0090847, -0.0063735, -0.0017187, 0.0017649
1: -0.0054902, -0.0047379, -0.0055000, -0.0047356, -0.0004846, 0.0004976
2: -0.0019482, 0.0036028, -0.0020202, 0.0036196, -0.0035752, 0.0036715
3: 0.0013695, 0.0021041, 0.0013599, 0.0021063, -0.0004731, 0.0004859
4: 0.0033994, 0.0075478, 0.0033868, 0.0076017, -0.0027438, 0.0026719
5: 0.9964507, 0.9976032, 0.9964471, 0.9976183, -0.0007623, 0.0007423
6: 0.0046619, 0.0057081, 0.0046588, 0.0057217, -0.0006919, 0.0006738
7: -0.0059840, -0.0020798, -0.0059958, -0.0020291, -0.0025822, 0.0025145
8: -0.0075741, -0.0045355, -0.0076136, -0.0045263, -0.0019571, 0.0020098
9: -0.0036184, -0.0033563, -0.0036192, -0.0033529, -0.0001734, 0.0001688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003369, upper bound: 0.0003338
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003491, upper bound: 0.0003642
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089186, -0.0062004, -0.0090806, -0.0064212, -0.0015089, 0.0018694
1: -0.0054531, -0.0046868, -0.0054988, -0.0047490, -0.0004254, 0.0005270
2: -0.0016747, 0.0039797, -0.0020117, 0.0035204, -0.0031388, 0.0038886
3: 0.0014057, 0.0021539, 0.0013611, 0.0020932, -0.0004154, 0.0005146
4: 0.0031177, 0.0073434, 0.0034610, 0.0075953, -0.0029061, 0.0023458
5: 0.9963724, 0.9975464, 0.9964679, 0.9976164, -0.0008074, 0.0006517
6: 0.0045909, 0.0056566, 0.0046775, 0.0057201, -0.0007329, 0.0005916
7: -0.0062490, -0.0022722, -0.0059260, -0.0020351, -0.0027350, 0.0022076
8: -0.0074244, -0.0043292, -0.0076089, -0.0045807, -0.0017182, 0.0021286
9: -0.0036362, -0.0033692, -0.0036145, -0.0033533, -0.0001836, 0.0001482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003853, upper bound: 0.0003676
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0003537
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089186, -0.0062004, -0.0089522, -0.0062709, -0.0015721, 0.0016517
1: -0.0054531, -0.0046868, -0.0054626, -0.0047067, -0.0004432, 0.0004657
2: -0.0016747, 0.0039797, -0.0017445, 0.0038332, -0.0032704, 0.0034358
3: 0.0014057, 0.0021539, 0.0013964, 0.0021346, -0.0004328, 0.0004547
4: 0.0031177, 0.0073434, 0.0032272, 0.0073956, -0.0025677, 0.0024441
5: 0.9963724, 0.9975464, 0.9964029, 0.9975609, -0.0007134, 0.0006790
6: 0.0045909, 0.0056566, 0.0046185, 0.0056697, -0.0006475, 0.0006164
7: -0.0062490, -0.0022722, -0.0061460, -0.0022230, -0.0024165, 0.0023002
8: -0.0074244, -0.0043292, -0.0074627, -0.0044094, -0.0017902, 0.0018808
9: -0.0036362, -0.0033692, -0.0036293, -0.0033659, -0.0001623, 0.0001545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003853, upper bound: 0.0004575
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0004576
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090508, -0.0063474, -0.0089161, -0.0062571, -0.0016359, 0.0014725
1: -0.0054904, -0.0047282, -0.0054524, -0.0047028, -0.0004612, 0.0004151
2: -0.0019498, 0.0036739, -0.0016694, 0.0038617, -0.0034030, 0.0030631
3: 0.0013693, 0.0021135, 0.0014064, 0.0021383, -0.0004503, 0.0004053
4: 0.0033462, 0.0075490, 0.0032059, 0.0073395, -0.0022891, 0.0025432
5: 0.9964359, 0.9976036, 0.9963970, 0.9975454, -0.0006360, 0.0007066
6: 0.0046485, 0.0057084, 0.0046131, 0.0056556, -0.0005773, 0.0006414
7: -0.0060340, -0.0020787, -0.0061661, -0.0022759, -0.0021543, 0.0023934
8: -0.0075750, -0.0044966, -0.0074216, -0.0043938, -0.0018628, 0.0016767
9: -0.0036218, -0.0033562, -0.0036307, -0.0033694, -0.0001447, 0.0001607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003619, upper bound: 0.0003801
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003619, upper bound: 0.0003801
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090501, -0.0063816, -0.0090450, -0.0063063, -0.0016585, 0.0016323
1: -0.0054902, -0.0047379, -0.0054888, -0.0047166, -0.0004676, 0.0004602
2: -0.0019482, 0.0036028, -0.0019376, 0.0037594, -0.0034499, 0.0033954
3: 0.0013695, 0.0021041, 0.0013709, 0.0021248, -0.0004565, 0.0004493
4: 0.0033994, 0.0075478, 0.0032823, 0.0075399, -0.0025375, 0.0025783
5: 0.9964507, 0.9976032, 0.9964182, 0.9976010, -0.0007050, 0.0007163
6: 0.0046619, 0.0057081, 0.0046324, 0.0057061, -0.0006399, 0.0006502
7: -0.0059840, -0.0020798, -0.0060941, -0.0020872, -0.0023881, 0.0024264
8: -0.0075741, -0.0045355, -0.0075684, -0.0044498, -0.0018885, 0.0018587
9: -0.0036184, -0.0033563, -0.0036258, -0.0033568, -0.0001604, 0.0001629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003424
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003522, upper bound: 0.0003714
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0089186, -0.0062004, -0.0090508, -0.0063474, -0.0014752, 0.0017490
1: -0.0054531, -0.0046868, -0.0054904, -0.0047282, -0.0004159, 0.0004931
2: -0.0016747, 0.0039797, -0.0019498, 0.0036739, -0.0030687, 0.0036382
3: 0.0014057, 0.0021539, 0.0013693, 0.0021135, -0.0004061, 0.0004815
4: 0.0031177, 0.0073434, 0.0033462, 0.0075490, -0.0027190, 0.0022934
5: 0.9963724, 0.9975464, 0.9964359, 0.9976036, -0.0007554, 0.0006372
6: 0.0045909, 0.0056566, 0.0046485, 0.0057084, -0.0006857, 0.0005784
7: -0.0062490, -0.0022722, -0.0060340, -0.0020787, -0.0025588, 0.0021583
8: -0.0074244, -0.0043292, -0.0075750, -0.0044966, -0.0016798, 0.0019916
9: -0.0036362, -0.0033692, -0.0036218, -0.0033562, -0.0001718, 0.0001449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003893, upper bound: 0.0003744
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0003606
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089186, -0.0062004, -0.0089186, -0.0062004, -0.0015168, 0.0015168
1: -0.0054531, -0.0046868, -0.0054531, -0.0046868, -0.0004277, 0.0004277
2: -0.0016747, 0.0039797, -0.0016747, 0.0039797, -0.0031553, 0.0031553
3: 0.0014057, 0.0021539, 0.0014057, 0.0021539, -0.0004176, 0.0004176
4: 0.0031177, 0.0073434, 0.0031177, 0.0073434, -0.0023581, 0.0023581
5: 0.9963724, 0.9975464, 0.9963724, 0.9975464, -0.0006552, 0.0006552
6: 0.0045909, 0.0056566, 0.0045909, 0.0056566, -0.0005947, 0.0005947
7: -0.0062490, -0.0022722, -0.0062490, -0.0022722, -0.0022192, 0.0022192
8: -0.0074244, -0.0043292, -0.0074244, -0.0043292, -0.0017272, 0.0017272
9: -0.0036362, -0.0033692, -0.0036362, -0.0033692, -0.0001490, 0.0001490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003893, upper bound: 0.0004574
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0004574
time: 1.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.54 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003729
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003729
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003313, upper bound: 0.0003338
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003642
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0003676
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003537
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0004576
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0004575
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003808
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003538, upper bound: 0.0003808
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003312, upper bound: 0.0003391
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003721
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0003729
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003588
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003781, upper bound: 0.0004603
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0004604
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003588, upper bound: 0.0003729
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003588, upper bound: 0.0003729
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003369, upper bound: 0.0003338
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003491, upper bound: 0.0003642
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003853, upper bound: 0.0003676
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0003537
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003853, upper bound: 0.0004575
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0004576
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003619, upper bound: 0.0003801
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003619, upper bound: 0.0003801
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003401, upper bound: 0.0003424
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003522, upper bound: 0.0003714
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003893, upper bound: 0.0003744
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0003606
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003893, upper bound: 0.0004574
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0004574

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090801, -0.0064605, -0.0089506, -0.0063330, -0.0016197, 0.0014203
1: -0.0054987, -0.0047601, -0.0054622, -0.0047242, -0.0004567, 0.0004004
2: -0.0020105, 0.0034388, -0.0017412, 0.0037039, -0.0033693, 0.0029544
3: 0.0013612, 0.0020824, 0.0013969, 0.0021174, -0.0004459, 0.0003910
4: 0.0035219, 0.0075944, 0.0033238, 0.0073931, -0.0022079, 0.0025180
5: 0.9964848, 0.9976162, 0.9964296, 0.9975603, -0.0006134, 0.0006996
6: 0.0046929, 0.0057199, 0.0046429, 0.0056691, -0.0005568, 0.0006350
7: -0.0058686, -0.0020360, -0.0060550, -0.0022254, -0.0020779, 0.0023697
8: -0.0076083, -0.0046253, -0.0074608, -0.0044802, -0.0018443, 0.0016172
9: -0.0036107, -0.0033533, -0.0036232, -0.0033661, -0.0001395, 0.0001591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003439, upper bound: 0.0003685
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003693
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0092025, -0.0065088, -0.0089506, -0.0063330, -0.0017775, 0.0014058
1: -0.0055332, -0.0047737, -0.0054622, -0.0047242, -0.0005012, 0.0003964
2: -0.0022651, 0.0033383, -0.0017412, 0.0037039, -0.0036977, 0.0029244
3: 0.0013275, 0.0020691, 0.0013969, 0.0021174, -0.0004893, 0.0003870
4: 0.0035970, 0.0077847, 0.0033238, 0.0073931, -0.0021855, 0.0027634
5: 0.9965056, 0.9976690, 0.9964296, 0.9975603, -0.0006072, 0.0007678
6: 0.0047118, 0.0057679, 0.0046429, 0.0056691, -0.0005512, 0.0006969
7: -0.0057979, -0.0018569, -0.0060550, -0.0022254, -0.0020568, 0.0026007
8: -0.0077476, -0.0046803, -0.0074608, -0.0044802, -0.0020241, 0.0016008
9: -0.0036059, -0.0033413, -0.0036232, -0.0033661, -0.0001381, 0.0001746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003439, upper bound: 0.0003685
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003693
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090229, -0.0064532, -0.0090699, -0.0063739, -0.0015949, 0.0016115
1: -0.0054826, -0.0047581, -0.0054958, -0.0047357, -0.0004497, 0.0004543
2: -0.0018917, 0.0034538, -0.0019895, 0.0036188, -0.0033178, 0.0033522
3: 0.0013770, 0.0020843, 0.0013640, 0.0021062, -0.0004391, 0.0004436
4: 0.0035107, 0.0075056, 0.0033874, 0.0075787, -0.0025052, 0.0024795
5: 0.9964817, 0.9975916, 0.9964474, 0.9976118, -0.0006960, 0.0006889
6: 0.0046900, 0.0056975, 0.0046589, 0.0057159, -0.0006318, 0.0006253
7: -0.0058792, -0.0021196, -0.0059952, -0.0020508, -0.0023577, 0.0023335
8: -0.0075432, -0.0046171, -0.0075967, -0.0045268, -0.0018161, 0.0018350
9: -0.0036114, -0.0033589, -0.0036192, -0.0033543, -0.0001583, 0.0001567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003128
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003079
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090429, -0.0064158, -0.0090704, -0.0063739, -0.0016055, 0.0016492
1: -0.0054882, -0.0047475, -0.0054959, -0.0047357, -0.0004527, 0.0004650
2: -0.0019332, 0.0035317, -0.0019904, 0.0036188, -0.0033398, 0.0034307
3: 0.0013715, 0.0020947, 0.0013639, 0.0021062, -0.0004420, 0.0004540
4: 0.0034525, 0.0075366, 0.0033874, 0.0075793, -0.0025639, 0.0024960
5: 0.9964654, 0.9976001, 0.9964474, 0.9976121, -0.0007123, 0.0006935
6: 0.0046753, 0.0057053, 0.0046589, 0.0057161, -0.0006466, 0.0006295
7: -0.0059340, -0.0020904, -0.0059952, -0.0020501, -0.0024129, 0.0023490
8: -0.0075659, -0.0045744, -0.0075972, -0.0045268, -0.0018282, 0.0018780
9: -0.0036151, -0.0033570, -0.0036192, -0.0033543, -0.0001620, 0.0001577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003476
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003250, upper bound: 0.0003446
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0090806, -0.0064212, -0.0014636, 0.0016847
1: -0.0054623, -0.0047189, -0.0054988, -0.0047490, -0.0004126, 0.0004750
2: -0.0017425, 0.0037425, -0.0020117, 0.0035204, -0.0030445, 0.0035045
3: 0.0013967, 0.0021226, 0.0013611, 0.0020932, -0.0004029, 0.0004638
4: 0.0032949, 0.0073941, 0.0034610, 0.0075953, -0.0026191, 0.0022753
5: 0.9964217, 0.9975605, 0.9964679, 0.9976164, -0.0007277, 0.0006321
6: 0.0046356, 0.0056694, 0.0046775, 0.0057201, -0.0006605, 0.0005738
7: -0.0060822, -0.0022245, -0.0059260, -0.0020351, -0.0024648, 0.0021413
8: -0.0074616, -0.0044591, -0.0076089, -0.0045807, -0.0016666, 0.0019184
9: -0.0036250, -0.0033660, -0.0036145, -0.0033533, -0.0001655, 0.0001438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003538
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003538
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090853, -0.0063548, -0.0090803, -0.0064518, -0.0016260, 0.0017114
1: -0.0055001, -0.0047303, -0.0054987, -0.0047577, -0.0004584, 0.0004825
2: -0.0020214, 0.0036586, -0.0020111, 0.0034568, -0.0033824, 0.0035601
3: 0.0013598, 0.0021115, 0.0013612, 0.0020847, -0.0004476, 0.0004711
4: 0.0033577, 0.0076025, 0.0035085, 0.0075949, -0.0026606, 0.0025278
5: 0.9964391, 0.9976184, 0.9964809, 0.9976162, -0.0007392, 0.0007023
6: 0.0046514, 0.0057219, 0.0046895, 0.0057200, -0.0006710, 0.0006375
7: -0.0060232, -0.0020283, -0.0058813, -0.0020355, -0.0025039, 0.0023790
8: -0.0076142, -0.0045050, -0.0076086, -0.0046155, -0.0018515, 0.0019488
9: -0.0036211, -0.0033528, -0.0036115, -0.0033533, -0.0001681, 0.0001597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003338, upper bound: 0.0003313
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003642, upper bound: 0.0003440
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0089522, -0.0062709, -0.0015011, 0.0014597
1: -0.0054623, -0.0047189, -0.0054626, -0.0047067, -0.0004232, 0.0004115
2: -0.0017425, 0.0037425, -0.0017445, 0.0038332, -0.0031226, 0.0030364
3: 0.0013967, 0.0021226, 0.0013964, 0.0021346, -0.0004132, 0.0004018
4: 0.0032949, 0.0073941, 0.0032272, 0.0073956, -0.0022692, 0.0023337
5: 0.9964217, 0.9975605, 0.9964029, 0.9975609, -0.0006305, 0.0006484
6: 0.0046356, 0.0056694, 0.0046185, 0.0056697, -0.0005723, 0.0005885
7: -0.0060822, -0.0022245, -0.0061460, -0.0022230, -0.0021356, 0.0021962
8: -0.0074616, -0.0044591, -0.0074627, -0.0044094, -0.0017093, 0.0016622
9: -0.0036250, -0.0033660, -0.0036293, -0.0033659, -0.0001434, 0.0001475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004543
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004576
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090853, -0.0063548, -0.0089517, -0.0062997, -0.0016422, 0.0014974
1: -0.0055001, -0.0047303, -0.0054625, -0.0047148, -0.0004630, 0.0004222
2: -0.0020214, 0.0036586, -0.0017435, 0.0037731, -0.0034161, 0.0031148
3: 0.0013598, 0.0021115, 0.0013966, 0.0021266, -0.0004521, 0.0004122
4: 0.0033577, 0.0076025, 0.0032721, 0.0073949, -0.0023278, 0.0025530
5: 0.9964391, 0.9976184, 0.9964153, 0.9975607, -0.0006467, 0.0007093
6: 0.0046514, 0.0057219, 0.0046298, 0.0056696, -0.0005870, 0.0006438
7: -0.0060232, -0.0020283, -0.0061038, -0.0022237, -0.0021907, 0.0024026
8: -0.0076142, -0.0045050, -0.0074621, -0.0044423, -0.0018700, 0.0017051
9: -0.0036211, -0.0033528, -0.0036265, -0.0033659, -0.0001471, 0.0001613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004483
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004528, upper bound: 0.0004483
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090801, -0.0064605, -0.0089161, -0.0062571, -0.0017646, 0.0014639
1: -0.0054987, -0.0047601, -0.0054524, -0.0047028, -0.0004975, 0.0004127
2: -0.0020105, 0.0034388, -0.0016694, 0.0038617, -0.0036706, 0.0030451
3: 0.0013612, 0.0020824, 0.0014064, 0.0021383, -0.0004858, 0.0004030
4: 0.0035219, 0.0075944, 0.0032059, 0.0073395, -0.0022757, 0.0027432
5: 0.9964848, 0.9976162, 0.9963970, 0.9975454, -0.0006323, 0.0007621
6: 0.0046929, 0.0057199, 0.0046131, 0.0056556, -0.0005739, 0.0006918
7: -0.0058686, -0.0020360, -0.0061661, -0.0022759, -0.0021417, 0.0025817
8: -0.0076083, -0.0046253, -0.0074216, -0.0043938, -0.0020093, 0.0016669
9: -0.0036107, -0.0033533, -0.0036307, -0.0033694, -0.0001438, 0.0001734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003765
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003770
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0092025, -0.0065088, -0.0089161, -0.0062571, -0.0019224, 0.0014501
1: -0.0055332, -0.0047737, -0.0054524, -0.0047028, -0.0005420, 0.0004088
2: -0.0022651, 0.0033383, -0.0016694, 0.0038617, -0.0039990, 0.0030165
3: 0.0013275, 0.0020691, 0.0014064, 0.0021383, -0.0005292, 0.0003992
4: 0.0035970, 0.0077847, 0.0032059, 0.0073395, -0.0022543, 0.0029886
5: 0.9965056, 0.9976690, 0.9963970, 0.9975454, -0.0006263, 0.0008303
6: 0.0047118, 0.0057679, 0.0046131, 0.0056556, -0.0005685, 0.0007537
7: -0.0057979, -0.0018569, -0.0061661, -0.0022759, -0.0021216, 0.0028126
8: -0.0077476, -0.0046803, -0.0074216, -0.0043938, -0.0021891, 0.0016512
9: -0.0036059, -0.0033413, -0.0036307, -0.0033694, -0.0001425, 0.0001889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003765
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003770
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090229, -0.0064532, -0.0090304, -0.0063067, -0.0017423, 0.0016251
1: -0.0054826, -0.0047581, -0.0054847, -0.0047168, -0.0004912, 0.0004582
2: -0.0018917, 0.0034538, -0.0019071, 0.0037586, -0.0036243, 0.0033804
3: 0.0013770, 0.0020843, 0.0013749, 0.0021247, -0.0004796, 0.0004473
4: 0.0035107, 0.0075056, 0.0032829, 0.0075171, -0.0025263, 0.0027086
5: 0.9964817, 0.9975916, 0.9964184, 0.9975947, -0.0007019, 0.0007525
6: 0.0046900, 0.0056975, 0.0046326, 0.0057004, -0.0006371, 0.0006831
7: -0.0058792, -0.0021196, -0.0060935, -0.0021087, -0.0023776, 0.0025491
8: -0.0075432, -0.0046171, -0.0075517, -0.0044503, -0.0019839, 0.0018505
9: -0.0036114, -0.0033589, -0.0036258, -0.0033582, -0.0001596, 0.0001712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003196
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003168
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090429, -0.0064158, -0.0090304, -0.0063067, -0.0017529, 0.0016670
1: -0.0054882, -0.0047475, -0.0054847, -0.0047168, -0.0004942, 0.0004700
2: -0.0019332, 0.0035317, -0.0019072, 0.0037586, -0.0036463, 0.0034677
3: 0.0013715, 0.0020947, 0.0013749, 0.0021247, -0.0004825, 0.0004589
4: 0.0034525, 0.0075366, 0.0032830, 0.0075172, -0.0025916, 0.0027250
5: 0.9964654, 0.9976001, 0.9964184, 0.9975948, -0.0007200, 0.0007571
6: 0.0046753, 0.0057053, 0.0046326, 0.0057004, -0.0006536, 0.0006872
7: -0.0059340, -0.0020904, -0.0060935, -0.0021086, -0.0024390, 0.0025646
8: -0.0075659, -0.0045744, -0.0075517, -0.0044503, -0.0019960, 0.0018982
9: -0.0036151, -0.0033570, -0.0036258, -0.0033582, -0.0001638, 0.0001722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003561
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003250, upper bound: 0.0003554
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0090508, -0.0063474, -0.0016126, 0.0017534
1: -0.0054623, -0.0047189, -0.0054904, -0.0047282, -0.0004547, 0.0004943
2: -0.0017425, 0.0037425, -0.0019498, 0.0036739, -0.0033546, 0.0036474
3: 0.0013967, 0.0021226, 0.0013693, 0.0021135, -0.0004439, 0.0004827
4: 0.0032949, 0.0073941, 0.0033462, 0.0075490, -0.0027258, 0.0025070
5: 0.9964217, 0.9975605, 0.9964359, 0.9976036, -0.0007573, 0.0006965
6: 0.0046356, 0.0056694, 0.0046485, 0.0057084, -0.0006874, 0.0006322
7: -0.0060822, -0.0022245, -0.0060340, -0.0020787, -0.0025653, 0.0023594
8: -0.0074616, -0.0044591, -0.0075750, -0.0044966, -0.0018363, 0.0019966
9: -0.0036250, -0.0033660, -0.0036218, -0.0033562, -0.0001723, 0.0001584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003588
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003588
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090853, -0.0063548, -0.0090501, -0.0063816, -0.0017653, 0.0017775
1: -0.0055001, -0.0047303, -0.0054902, -0.0047379, -0.0004977, 0.0005011
2: -0.0020214, 0.0036586, -0.0019482, 0.0036028, -0.0036721, 0.0036976
3: 0.0013598, 0.0021115, 0.0013695, 0.0021041, -0.0004859, 0.0004893
4: 0.0033577, 0.0076025, 0.0033994, 0.0075478, -0.0027634, 0.0027443
5: 0.9964391, 0.9976184, 0.9964507, 0.9976032, -0.0007677, 0.0007625
6: 0.0046514, 0.0057219, 0.0046619, 0.0057081, -0.0006969, 0.0006921
7: -0.0060232, -0.0020283, -0.0059840, -0.0020798, -0.0026006, 0.0025827
8: -0.0076142, -0.0045050, -0.0075741, -0.0045355, -0.0020101, 0.0020241
9: -0.0036211, -0.0033528, -0.0036184, -0.0033563, -0.0001746, 0.0001734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003338, upper bound: 0.0003369
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003642, upper bound: 0.0003491
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0089186, -0.0062004, -0.0016478, 0.0015282
1: -0.0054623, -0.0047189, -0.0054531, -0.0046868, -0.0004646, 0.0004309
2: -0.0017425, 0.0037425, -0.0016747, 0.0039797, -0.0034277, 0.0031790
3: 0.0013967, 0.0021226, 0.0014057, 0.0021539, -0.0004536, 0.0004207
4: 0.0032949, 0.0073941, 0.0031177, 0.0073434, -0.0023758, 0.0025617
5: 0.9964217, 0.9975605, 0.9963724, 0.9975464, -0.0006601, 0.0007117
6: 0.0046356, 0.0056694, 0.0045909, 0.0056566, -0.0005991, 0.0006460
7: -0.0060822, -0.0022245, -0.0062490, -0.0022722, -0.0022359, 0.0024108
8: -0.0074616, -0.0044591, -0.0074244, -0.0043292, -0.0018763, 0.0017402
9: -0.0036250, -0.0033660, -0.0036362, -0.0033692, -0.0001501, 0.0001619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004582
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004603
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090853, -0.0063548, -0.0089177, -0.0062302, -0.0017785, 0.0015601
1: -0.0055001, -0.0047303, -0.0054529, -0.0046952, -0.0005014, 0.0004399
2: -0.0020214, 0.0036586, -0.0016727, 0.0039177, -0.0036996, 0.0032454
3: 0.0013598, 0.0021115, 0.0014059, 0.0021457, -0.0004896, 0.0004295
4: 0.0033577, 0.0076025, 0.0031640, 0.0073419, -0.0024254, 0.0027648
5: 0.9964391, 0.9976184, 0.9963853, 0.9975461, -0.0006739, 0.0007682
6: 0.0046514, 0.0057219, 0.0046026, 0.0056562, -0.0006117, 0.0006973
7: -0.0060232, -0.0020283, -0.0062055, -0.0022736, -0.0022826, 0.0026020
8: -0.0076142, -0.0045050, -0.0074234, -0.0043632, -0.0020252, 0.0017766
9: -0.0036211, -0.0033528, -0.0036333, -0.0033693, -0.0001533, 0.0001747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004512
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004528, upper bound: 0.0004512
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090494, -0.0063830, -0.0089506, -0.0063330, -0.0016907, 0.0015764
1: -0.0054900, -0.0047383, -0.0054622, -0.0047242, -0.0004767, 0.0004444
2: -0.0019468, 0.0035999, -0.0017412, 0.0037039, -0.0035170, 0.0032792
3: 0.0013697, 0.0021037, 0.0013969, 0.0021174, -0.0004654, 0.0004340
4: 0.0034015, 0.0075468, 0.0033238, 0.0073931, -0.0024507, 0.0026284
5: 0.9964513, 0.9976029, 0.9964296, 0.9975603, -0.0006809, 0.0007302
6: 0.0046625, 0.0057079, 0.0046429, 0.0056691, -0.0006180, 0.0006628
7: -0.0059819, -0.0020808, -0.0060550, -0.0022254, -0.0023064, 0.0024736
8: -0.0075734, -0.0045371, -0.0074608, -0.0044802, -0.0019252, 0.0017950
9: -0.0036183, -0.0033563, -0.0036232, -0.0033661, -0.0001549, 0.0001661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003471, upper bound: 0.0003685
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003626, upper bound: 0.0003693
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0091638, -0.0064432, -0.0089506, -0.0063330, -0.0018203, 0.0015260
1: -0.0055223, -0.0047552, -0.0054622, -0.0047242, -0.0005132, 0.0004302
2: -0.0021847, 0.0034747, -0.0017412, 0.0037039, -0.0037866, 0.0031743
3: 0.0013382, 0.0020871, 0.0013969, 0.0021174, -0.0005011, 0.0004201
4: 0.0034951, 0.0077246, 0.0033238, 0.0073931, -0.0023723, 0.0028298
5: 0.9964772, 0.9976524, 0.9964296, 0.9975603, -0.0006591, 0.0007862
6: 0.0046861, 0.0057527, 0.0046429, 0.0056691, -0.0005983, 0.0007136
7: -0.0058939, -0.0019134, -0.0060550, -0.0022254, -0.0022326, 0.0026632
8: -0.0077036, -0.0046056, -0.0074608, -0.0044802, -0.0020728, 0.0017376
9: -0.0036124, -0.0033451, -0.0036232, -0.0033661, -0.0001499, 0.0001788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003471, upper bound: 0.0003685
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003626, upper bound: 0.0003693
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089900, -0.0063830, -0.0090699, -0.0063739, -0.0016551, 0.0017508
1: -0.0054733, -0.0047383, -0.0054958, -0.0047357, -0.0004666, 0.0004936
2: -0.0018232, 0.0036000, -0.0019895, 0.0036188, -0.0034430, 0.0036420
3: 0.0013860, 0.0021037, 0.0013640, 0.0021062, -0.0004556, 0.0004820
4: 0.0034014, 0.0074544, 0.0033874, 0.0075787, -0.0027218, 0.0025731
5: 0.9964513, 0.9975773, 0.9964474, 0.9976118, -0.0007562, 0.0007149
6: 0.0046625, 0.0056846, 0.0046589, 0.0057159, -0.0006864, 0.0006489
7: -0.0059820, -0.0021677, -0.0059952, -0.0020508, -0.0025615, 0.0024216
8: -0.0075058, -0.0045371, -0.0075967, -0.0045268, -0.0018847, 0.0019936
9: -0.0036183, -0.0033622, -0.0036192, -0.0033543, -0.0001720, 0.0001626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003178, upper bound: 0.0003143
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003169, upper bound: 0.0003091
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090145, -0.0063393, -0.0090704, -0.0063739, -0.0016675, 0.0018070
1: -0.0054802, -0.0047260, -0.0054959, -0.0047357, -0.0004701, 0.0005095
2: -0.0018740, 0.0036908, -0.0019904, 0.0036188, -0.0034687, 0.0037589
3: 0.0013793, 0.0021157, 0.0013639, 0.0021062, -0.0004590, 0.0004974
4: 0.0033336, 0.0074924, 0.0033874, 0.0075793, -0.0028091, 0.0025923
5: 0.9964324, 0.9975879, 0.9964474, 0.9976121, -0.0007805, 0.0007202
6: 0.0046454, 0.0056941, 0.0046589, 0.0057161, -0.0007084, 0.0006537
7: -0.0060458, -0.0021319, -0.0059952, -0.0020501, -0.0026437, 0.0024396
8: -0.0075336, -0.0044874, -0.0075972, -0.0045268, -0.0018988, 0.0020576
9: -0.0036226, -0.0033598, -0.0036192, -0.0033543, -0.0001775, 0.0001638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003315, upper bound: 0.0003476
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003314, upper bound: 0.0003446
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0090806, -0.0064212, -0.0015065, 0.0018215
1: -0.0054526, -0.0046983, -0.0054988, -0.0047490, -0.0004248, 0.0005135
2: -0.0016709, 0.0038945, -0.0020117, 0.0035204, -0.0031339, 0.0037891
3: 0.0014062, 0.0021427, 0.0013611, 0.0020932, -0.0004147, 0.0005014
4: 0.0031814, 0.0073406, 0.0034610, 0.0075953, -0.0028317, 0.0023421
5: 0.9963901, 0.9975457, 0.9964679, 0.9976164, -0.0007867, 0.0006507
6: 0.0046070, 0.0056559, 0.0046775, 0.0057201, -0.0007141, 0.0005906
7: -0.0061891, -0.0022748, -0.0059260, -0.0020351, -0.0026649, 0.0022042
8: -0.0074224, -0.0043759, -0.0076089, -0.0045807, -0.0017155, 0.0020741
9: -0.0036322, -0.0033694, -0.0036145, -0.0033533, -0.0001789, 0.0001480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0003538
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0003538
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090456, -0.0062859, -0.0090803, -0.0064518, -0.0016410, 0.0018501
1: -0.0054890, -0.0047109, -0.0054987, -0.0047577, -0.0004627, 0.0005216
2: -0.0019388, 0.0038018, -0.0020111, 0.0034568, -0.0034137, 0.0038486
3: 0.0013707, 0.0021304, 0.0013612, 0.0020847, -0.0004517, 0.0005093
4: 0.0032506, 0.0075408, 0.0035085, 0.0075949, -0.0028762, 0.0025512
5: 0.9964093, 0.9976013, 0.9964809, 0.9976162, -0.0007991, 0.0007088
6: 0.0046244, 0.0057064, 0.0046895, 0.0057200, -0.0007253, 0.0006434
7: -0.0061239, -0.0020864, -0.0058813, -0.0020355, -0.0027068, 0.0024009
8: -0.0075690, -0.0044266, -0.0076086, -0.0046155, -0.0018687, 0.0021067
9: -0.0036278, -0.0033567, -0.0036115, -0.0033533, -0.0001818, 0.0001612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003312
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003721, upper bound: 0.0003440
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0089522, -0.0062709, -0.0015698, 0.0016148
1: -0.0054526, -0.0046983, -0.0054626, -0.0047067, -0.0004426, 0.0004553
2: -0.0016709, 0.0038945, -0.0017445, 0.0038332, -0.0032654, 0.0033592
3: 0.0014062, 0.0021427, 0.0013964, 0.0021346, -0.0004321, 0.0004445
4: 0.0031814, 0.0073406, 0.0032272, 0.0073956, -0.0025105, 0.0024404
5: 0.9963901, 0.9975457, 0.9964029, 0.9975609, -0.0006975, 0.0006780
6: 0.0046070, 0.0056559, 0.0046185, 0.0056697, -0.0006331, 0.0006154
7: -0.0061891, -0.0022748, -0.0061460, -0.0022230, -0.0023626, 0.0022967
8: -0.0074224, -0.0043759, -0.0074627, -0.0044094, -0.0017875, 0.0018388
9: -0.0036322, -0.0033694, -0.0036293, -0.0033659, -0.0001586, 0.0001542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004607, upper bound: 0.0004543
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004607, upper bound: 0.0004576
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090456, -0.0062859, -0.0089517, -0.0062997, -0.0016811, 0.0016542
1: -0.0054890, -0.0047109, -0.0054625, -0.0047148, -0.0004740, 0.0004664
2: -0.0019388, 0.0038018, -0.0017435, 0.0037731, -0.0034970, 0.0034411
3: 0.0013707, 0.0021304, 0.0013966, 0.0021266, -0.0004628, 0.0004554
4: 0.0032506, 0.0075408, 0.0032721, 0.0073949, -0.0025717, 0.0026134
5: 0.9964093, 0.9976013, 0.9964153, 0.9975607, -0.0007145, 0.0007261
6: 0.0046244, 0.0057064, 0.0046298, 0.0056696, -0.0006485, 0.0006591
7: -0.0061239, -0.0020864, -0.0061038, -0.0022237, -0.0024202, 0.0024595
8: -0.0075690, -0.0044266, -0.0074621, -0.0044423, -0.0019143, 0.0018837
9: -0.0036278, -0.0033567, -0.0036265, -0.0033659, -0.0001625, 0.0001652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004483, upper bound: 0.0004483
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004563, upper bound: 0.0004483
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090494, -0.0063830, -0.0089161, -0.0062571, -0.0016332, 0.0014305
1: -0.0054900, -0.0047383, -0.0054524, -0.0047028, -0.0004605, 0.0004033
2: -0.0019468, 0.0035999, -0.0016694, 0.0038617, -0.0033974, 0.0029757
3: 0.0013697, 0.0021037, 0.0014064, 0.0021383, -0.0004496, 0.0003938
4: 0.0034015, 0.0075468, 0.0032059, 0.0073395, -0.0022238, 0.0025390
5: 0.9964513, 0.9976029, 0.9963970, 0.9975454, -0.0006178, 0.0007054
6: 0.0046625, 0.0057079, 0.0046131, 0.0056556, -0.0005608, 0.0006403
7: -0.0059819, -0.0020808, -0.0061661, -0.0022759, -0.0020929, 0.0023895
8: -0.0075734, -0.0045371, -0.0074216, -0.0043938, -0.0018597, 0.0016289
9: -0.0036183, -0.0033563, -0.0036307, -0.0033694, -0.0001405, 0.0001604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003513, upper bound: 0.0003746
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003650, upper bound: 0.0003755
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0091638, -0.0064432, -0.0089161, -0.0062571, -0.0017875, 0.0014165
1: -0.0055223, -0.0047552, -0.0054524, -0.0047028, -0.0005040, 0.0003994
2: -0.0021847, 0.0034747, -0.0016694, 0.0038617, -0.0037185, 0.0029465
3: 0.0013382, 0.0020871, 0.0014064, 0.0021383, -0.0004921, 0.0003899
4: 0.0034951, 0.0077246, 0.0032059, 0.0073395, -0.0022021, 0.0027789
5: 0.9964772, 0.9976524, 0.9963970, 0.9975454, -0.0006118, 0.0007721
6: 0.0046861, 0.0057527, 0.0046131, 0.0056556, -0.0005553, 0.0007008
7: -0.0058939, -0.0019134, -0.0061661, -0.0022759, -0.0020724, 0.0026153
8: -0.0077036, -0.0046056, -0.0074216, -0.0043938, -0.0020355, 0.0016129
9: -0.0036124, -0.0033451, -0.0036307, -0.0033694, -0.0001392, 0.0001756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003513, upper bound: 0.0003746
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003650, upper bound: 0.0003755
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089900, -0.0063830, -0.0090304, -0.0063067, -0.0015995, 0.0016180
1: -0.0054733, -0.0047383, -0.0054847, -0.0047168, -0.0004509, 0.0004562
2: -0.0018232, 0.0036000, -0.0019071, 0.0037586, -0.0033272, 0.0033658
3: 0.0013860, 0.0021037, 0.0013749, 0.0021247, -0.0004403, 0.0004454
4: 0.0034014, 0.0074544, 0.0032829, 0.0075171, -0.0025154, 0.0024865
5: 0.9964513, 0.9975773, 0.9964184, 0.9975947, -0.0006988, 0.0006908
6: 0.0046625, 0.0056846, 0.0046326, 0.0057004, -0.0006343, 0.0006271
7: -0.0059820, -0.0021677, -0.0060935, -0.0021087, -0.0023672, 0.0023401
8: -0.0075058, -0.0045371, -0.0075517, -0.0044503, -0.0018213, 0.0018424
9: -0.0036183, -0.0033622, -0.0036258, -0.0033582, -0.0001590, 0.0001571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003211, upper bound: 0.0003247
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003204, upper bound: 0.0003214
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090145, -0.0063393, -0.0090304, -0.0063067, -0.0016123, 0.0016568
1: -0.0054802, -0.0047260, -0.0054847, -0.0047168, -0.0004546, 0.0004671
2: -0.0018740, 0.0036908, -0.0019072, 0.0037586, -0.0033539, 0.0034466
3: 0.0013793, 0.0021157, 0.0013749, 0.0021247, -0.0004438, 0.0004561
4: 0.0033336, 0.0074924, 0.0032830, 0.0075172, -0.0025758, 0.0025065
5: 0.9964324, 0.9975879, 0.9964184, 0.9975948, -0.0007156, 0.0006964
6: 0.0046454, 0.0056941, 0.0046326, 0.0057004, -0.0006496, 0.0006321
7: -0.0060458, -0.0021319, -0.0060935, -0.0021086, -0.0024241, 0.0023589
8: -0.0075336, -0.0044874, -0.0075517, -0.0044503, -0.0018359, 0.0018867
9: -0.0036226, -0.0033598, -0.0036258, -0.0033582, -0.0001628, 0.0001584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003352, upper bound: 0.0003562
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003348, upper bound: 0.0003545
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0090508, -0.0063474, -0.0014729, 0.0016963
1: -0.0054526, -0.0046983, -0.0054904, -0.0047282, -0.0004153, 0.0004782
2: -0.0016709, 0.0038945, -0.0019498, 0.0036739, -0.0030639, 0.0035286
3: 0.0014062, 0.0021427, 0.0013693, 0.0021135, -0.0004055, 0.0004670
4: 0.0031814, 0.0073406, 0.0033462, 0.0075490, -0.0026371, 0.0022898
5: 0.9963901, 0.9975457, 0.9964359, 0.9976036, -0.0007327, 0.0006362
6: 0.0046070, 0.0056559, 0.0046485, 0.0057084, -0.0006650, 0.0005774
7: -0.0061891, -0.0022748, -0.0060340, -0.0020787, -0.0024818, 0.0021549
8: -0.0074224, -0.0043759, -0.0075750, -0.0044966, -0.0016772, 0.0019316
9: -0.0036322, -0.0033694, -0.0036218, -0.0033562, -0.0001666, 0.0001447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0003606
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0003606
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090456, -0.0062859, -0.0090501, -0.0063816, -0.0016326, 0.0017175
1: -0.0054890, -0.0047109, -0.0054902, -0.0047379, -0.0004603, 0.0004842
2: -0.0019388, 0.0038018, -0.0019482, 0.0036028, -0.0033962, 0.0035727
3: 0.0013707, 0.0021304, 0.0013695, 0.0021041, -0.0004494, 0.0004728
4: 0.0032506, 0.0075408, 0.0033994, 0.0075478, -0.0026700, 0.0025381
5: 0.9964093, 0.9976013, 0.9964507, 0.9976032, -0.0007418, 0.0007052
6: 0.0046244, 0.0057064, 0.0046619, 0.0057081, -0.0006733, 0.0006401
7: -0.0061239, -0.0020864, -0.0059840, -0.0020798, -0.0025128, 0.0023886
8: -0.0075690, -0.0044266, -0.0075741, -0.0045355, -0.0018591, 0.0019557
9: -0.0036278, -0.0033567, -0.0036184, -0.0033563, -0.0001687, 0.0001604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003458, upper bound: 0.0003387
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003508
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0089186, -0.0062004, -0.0015143, 0.0014732
1: -0.0054526, -0.0046983, -0.0054531, -0.0046868, -0.0004269, 0.0004153
2: -0.0016709, 0.0038945, -0.0016747, 0.0039797, -0.0031501, 0.0030646
3: 0.0014062, 0.0021427, 0.0014057, 0.0021539, -0.0004169, 0.0004055
4: 0.0031814, 0.0073406, 0.0031177, 0.0073434, -0.0022903, 0.0023542
5: 0.9963901, 0.9975457, 0.9963724, 0.9975464, -0.0006363, 0.0006541
6: 0.0046070, 0.0056559, 0.0045909, 0.0056566, -0.0005776, 0.0005937
7: -0.0061891, -0.0022748, -0.0062490, -0.0022722, -0.0021554, 0.0022156
8: -0.0074224, -0.0043759, -0.0074244, -0.0043292, -0.0017244, 0.0016775
9: -0.0036322, -0.0033694, -0.0036362, -0.0033692, -0.0001447, 0.0001488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004608, upper bound: 0.0004546
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004608, upper bound: 0.0004574
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090456, -0.0062859, -0.0089177, -0.0062302, -0.0016519, 0.0015036
1: -0.0054890, -0.0047109, -0.0054529, -0.0046952, -0.0004657, 0.0004239
2: -0.0019388, 0.0038018, -0.0016727, 0.0039177, -0.0034364, 0.0031277
3: 0.0013707, 0.0021304, 0.0014059, 0.0021457, -0.0004547, 0.0004139
4: 0.0032506, 0.0075408, 0.0031640, 0.0073419, -0.0023375, 0.0025681
5: 0.9964093, 0.9976013, 0.9963853, 0.9975461, -0.0006494, 0.0007135
6: 0.0046244, 0.0057064, 0.0046026, 0.0056562, -0.0005895, 0.0006476
7: -0.0061239, -0.0020864, -0.0062055, -0.0022736, -0.0021998, 0.0024169
8: -0.0075690, -0.0044266, -0.0074234, -0.0043632, -0.0018811, 0.0017121
9: -0.0036278, -0.0033567, -0.0036333, -0.0033693, -0.0001477, 0.0001623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004487, upper bound: 0.0004482
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004563, upper bound: 0.0004482
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003439, upper bound: 0.0003685
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003693
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003439, upper bound: 0.0003685
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003693
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003128
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003079
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003476
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003250, upper bound: 0.0003446
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003538
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003538
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003338, upper bound: 0.0003313
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003642, upper bound: 0.0003440
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004543
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004576
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004483
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004528, upper bound: 0.0004483
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003765
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003770
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003440, upper bound: 0.0003765
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003572, upper bound: 0.0003770
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003196
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003168
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003255, upper bound: 0.0003561
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003250, upper bound: 0.0003554
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003588
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003729, upper bound: 0.0003588
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003338, upper bound: 0.0003369
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003642, upper bound: 0.0003491
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004582
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004562, upper bound: 0.0004603
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004461, upper bound: 0.0004512
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004528, upper bound: 0.0004512
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003471, upper bound: 0.0003685
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003626, upper bound: 0.0003693
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003471, upper bound: 0.0003685
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003626, upper bound: 0.0003693
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003178, upper bound: 0.0003143
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003169, upper bound: 0.0003091
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003315, upper bound: 0.0003476
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003314, upper bound: 0.0003446
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0003538
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003808, upper bound: 0.0003538
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003312
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003721, upper bound: 0.0003440
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004607, upper bound: 0.0004543
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004607, upper bound: 0.0004576
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004483, upper bound: 0.0004483
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004563, upper bound: 0.0004483
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003513, upper bound: 0.0003746
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003650, upper bound: 0.0003755
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003513, upper bound: 0.0003746
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003650, upper bound: 0.0003755
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003211, upper bound: 0.0003247
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003204, upper bound: 0.0003214
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003352, upper bound: 0.0003562
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003348, upper bound: 0.0003545
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0003606
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003852, upper bound: 0.0003606
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003458, upper bound: 0.0003387
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003508
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004608, upper bound: 0.0004546
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004608, upper bound: 0.0004574
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004487, upper bound: 0.0004482
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 5, lower bound: -0.0004563, upper bound: 0.0004482

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090640, -0.0064608, -0.0088961, -0.0063344, -0.0016016, 0.0013684
1: -0.0054941, -0.0047602, -0.0054468, -0.0047246, -0.0004515, 0.0003858
2: -0.0019770, 0.0034380, -0.0016279, 0.0037011, -0.0033316, 0.0028465
3: 0.0013657, 0.0020823, 0.0014119, 0.0021171, -0.0004409, 0.0003767
4: 0.0035225, 0.0075694, 0.0033259, 0.0073085, -0.0021273, 0.0024898
5: 0.9964849, 0.9976092, 0.9964303, 0.9975367, -0.0005910, 0.0006917
6: 0.0046930, 0.0057136, 0.0046434, 0.0056478, -0.0005365, 0.0006279
7: -0.0058680, -0.0020595, -0.0060531, -0.0023051, -0.0020020, 0.0023432
8: -0.0075899, -0.0046258, -0.0073988, -0.0044817, -0.0018237, 0.0015582
9: -0.0036106, -0.0033549, -0.0036231, -0.0033714, -0.0001344, 0.0001573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004233, upper bound: 0.0004288
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004232, upper bound: 0.0004366
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090669, -0.0064608, -0.0089091, -0.0062932, -0.0016534, 0.0013799
1: -0.0054950, -0.0047602, -0.0054505, -0.0047129, -0.0004661, 0.0003891
2: -0.0019831, 0.0034380, -0.0016549, 0.0037868, -0.0034394, 0.0028706
3: 0.0013649, 0.0020823, 0.0014083, 0.0021284, -0.0004551, 0.0003799
4: 0.0035225, 0.0075739, 0.0032619, 0.0073286, -0.0021453, 0.0025704
5: 0.9964849, 0.9976104, 0.9964125, 0.9975423, -0.0005960, 0.0007141
6: 0.0046930, 0.0057147, 0.0046273, 0.0056528, -0.0005410, 0.0006482
7: -0.0058680, -0.0020552, -0.0061134, -0.0022861, -0.0020189, 0.0024190
8: -0.0075933, -0.0046258, -0.0074136, -0.0044348, -0.0018827, 0.0015713
9: -0.0036106, -0.0033546, -0.0036271, -0.0033701, -0.0001356, 0.0001624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004308, upper bound: 0.0004287
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004308, upper bound: 0.0004365
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0091874, -0.0065092, -0.0088961, -0.0063344, -0.0017640, 0.0013538
1: -0.0055289, -0.0047738, -0.0054468, -0.0047246, -0.0004973, 0.0003817
2: -0.0022339, 0.0033374, -0.0016279, 0.0037011, -0.0036696, 0.0028163
3: 0.0013317, 0.0020689, 0.0014119, 0.0021171, -0.0004856, 0.0003727
4: 0.0035977, 0.0077613, 0.0033259, 0.0073085, -0.0021047, 0.0027424
5: 0.9965058, 0.9976625, 0.9964303, 0.9975367, -0.0005847, 0.0007619
6: 0.0047120, 0.0057620, 0.0046434, 0.0056478, -0.0005308, 0.0006916
7: -0.0057973, -0.0018789, -0.0060531, -0.0023051, -0.0019808, 0.0025809
8: -0.0077305, -0.0046808, -0.0073988, -0.0044817, -0.0020087, 0.0015416
9: -0.0036059, -0.0033428, -0.0036231, -0.0033714, -0.0001330, 0.0001733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003261, upper bound: 0.0003481
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003231, upper bound: 0.0003491
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0091882, -0.0065092, -0.0089091, -0.0062932, -0.0018058, 0.0013654
1: -0.0055292, -0.0047738, -0.0054505, -0.0047129, -0.0005091, 0.0003850
2: -0.0022354, 0.0033374, -0.0016549, 0.0037868, -0.0037564, 0.0028404
3: 0.0013315, 0.0020690, 0.0014083, 0.0021284, -0.0004971, 0.0003759
4: 0.0035977, 0.0077625, 0.0032619, 0.0073286, -0.0021227, 0.0028073
5: 0.9965058, 0.9976628, 0.9964125, 0.9975423, -0.0005897, 0.0007799
6: 0.0047119, 0.0057623, 0.0046273, 0.0056528, -0.0005353, 0.0007080
7: -0.0057973, -0.0018778, -0.0061134, -0.0022861, -0.0019977, 0.0026420
8: -0.0077314, -0.0046808, -0.0074136, -0.0044348, -0.0020562, 0.0015548
9: -0.0036059, -0.0033427, -0.0036271, -0.0033701, -0.0001341, 0.0001774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003488
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003501
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090329, -0.0064192, -0.0090174, -0.0063895, -0.0015733, 0.0015864
1: -0.0054854, -0.0047485, -0.0054810, -0.0047401, -0.0004436, 0.0004473
2: -0.0019124, 0.0035247, -0.0018801, 0.0035865, -0.0032728, 0.0033001
3: 0.0013742, 0.0020937, 0.0013785, 0.0021019, -0.0004331, 0.0004367
4: 0.0034577, 0.0075211, 0.0034116, 0.0074969, -0.0024663, 0.0024459
5: 0.9964669, 0.9975958, 0.9964541, 0.9975891, -0.0006852, 0.0006795
6: 0.0046767, 0.0057014, 0.0046650, 0.0056953, -0.0006220, 0.0006168
7: -0.0059290, -0.0021050, -0.0059725, -0.0021277, -0.0023211, 0.0023018
8: -0.0075546, -0.0045783, -0.0075369, -0.0045445, -0.0017915, 0.0018065
9: -0.0036147, -0.0033580, -0.0036177, -0.0033595, -0.0001559, 0.0001546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002936, upper bound: 0.0003008
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003183, upper bound: 0.0003406
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090231, -0.0064205, -0.0090061, -0.0063652, -0.0016146, 0.0015889
1: -0.0054826, -0.0047488, -0.0054778, -0.0047332, -0.0004552, 0.0004480
2: -0.0018920, 0.0035219, -0.0018567, 0.0036370, -0.0033587, 0.0033053
3: 0.0013769, 0.0020934, 0.0013816, 0.0021086, -0.0004445, 0.0004374
4: 0.0034598, 0.0075058, 0.0033738, 0.0074794, -0.0024702, 0.0025101
5: 0.9964675, 0.9975916, 0.9964436, 0.9975842, -0.0006863, 0.0006974
6: 0.0046772, 0.0056975, 0.0046555, 0.0056909, -0.0006229, 0.0006330
7: -0.0059271, -0.0021194, -0.0060080, -0.0021442, -0.0023247, 0.0023623
8: -0.0075434, -0.0045798, -0.0075240, -0.0045168, -0.0018386, 0.0018093
9: -0.0036146, -0.0033589, -0.0036200, -0.0033606, -0.0001561, 0.0001586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002925, upper bound: 0.0002944
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003181, upper bound: 0.0003377
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0090801, -0.0064605, -0.0014206, 0.0016806
1: -0.0054623, -0.0047189, -0.0054987, -0.0047601, -0.0004005, 0.0004738
2: -0.0017425, 0.0037425, -0.0020105, 0.0034388, -0.0029551, 0.0034960
3: 0.0013967, 0.0021226, 0.0013612, 0.0020824, -0.0003911, 0.0004626
4: 0.0032949, 0.0073941, 0.0035219, 0.0075944, -0.0026127, 0.0022085
5: 0.9964217, 0.9975605, 0.9964848, 0.9976162, -0.0007259, 0.0006136
6: 0.0046356, 0.0056694, 0.0046929, 0.0057199, -0.0006589, 0.0005569
7: -0.0060822, -0.0022245, -0.0058686, -0.0020360, -0.0024589, 0.0020784
8: -0.0074616, -0.0044591, -0.0076083, -0.0046253, -0.0016176, 0.0019137
9: -0.0036250, -0.0033660, -0.0036107, -0.0033533, -0.0001651, 0.0001396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003439
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003572
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0092025, -0.0065088, -0.0014062, 0.0018385
1: -0.0054623, -0.0047189, -0.0055332, -0.0047737, -0.0003964, 0.0005183
2: -0.0017425, 0.0037425, -0.0022651, 0.0033383, -0.0029251, 0.0038244
3: 0.0013967, 0.0021226, 0.0013275, 0.0020691, -0.0003871, 0.0005061
4: 0.0032949, 0.0073941, 0.0035970, 0.0077847, -0.0028581, 0.0021860
5: 0.9964217, 0.9975605, 0.9965056, 0.9976690, -0.0007941, 0.0006073
6: 0.0046356, 0.0056694, 0.0047118, 0.0057679, -0.0007208, 0.0005513
7: -0.0060822, -0.0022245, -0.0057979, -0.0018569, -0.0026898, 0.0020573
8: -0.0074616, -0.0044591, -0.0077476, -0.0046803, -0.0016012, 0.0020935
9: -0.0036250, -0.0033660, -0.0036059, -0.0033413, -0.0001806, 0.0001381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003439
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003572
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090705, -0.0063552, -0.0090229, -0.0064532, -0.0016118, 0.0016530
1: -0.0054960, -0.0047304, -0.0054826, -0.0047581, -0.0004544, 0.0004661
2: -0.0019906, 0.0036577, -0.0018917, 0.0034538, -0.0033528, 0.0034387
3: 0.0013639, 0.0021113, 0.0013770, 0.0020843, -0.0004437, 0.0004551
4: 0.0033583, 0.0075795, 0.0035107, 0.0075056, -0.0025698, 0.0025057
5: 0.9964393, 0.9976120, 0.9964817, 0.9975916, -0.0007140, 0.0006962
6: 0.0046516, 0.0057161, 0.0046900, 0.0056975, -0.0006481, 0.0006319
7: -0.0060226, -0.0020500, -0.0058792, -0.0021196, -0.0024185, 0.0023582
8: -0.0075974, -0.0045055, -0.0075432, -0.0046171, -0.0018354, 0.0018823
9: -0.0036210, -0.0033543, -0.0036114, -0.0033589, -0.0001624, 0.0001583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003128, upper bound: 0.0003095
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003079, upper bound: 0.0003082
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090709, -0.0063552, -0.0090429, -0.0064158, -0.0016495, 0.0016637
1: -0.0054961, -0.0047304, -0.0054882, -0.0047475, -0.0004651, 0.0004690
2: -0.0019915, 0.0036577, -0.0019332, 0.0035317, -0.0034314, 0.0034607
3: 0.0013637, 0.0021113, 0.0013715, 0.0020947, -0.0004541, 0.0004580
4: 0.0033583, 0.0075802, 0.0034525, 0.0075366, -0.0025863, 0.0025644
5: 0.9964393, 0.9976123, 0.9964654, 0.9976001, -0.0007186, 0.0007125
6: 0.0046516, 0.0057163, 0.0046753, 0.0057053, -0.0006522, 0.0006467
7: -0.0060226, -0.0020493, -0.0059340, -0.0020904, -0.0024340, 0.0024134
8: -0.0075979, -0.0045055, -0.0075659, -0.0045744, -0.0018784, 0.0018944
9: -0.0036210, -0.0033542, -0.0036151, -0.0033570, -0.0001634, 0.0001621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003476, upper bound: 0.0003255
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003446, upper bound: 0.0003250
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0089512, -0.0063144, -0.0014558, 0.0014558
1: -0.0054623, -0.0047189, -0.0054623, -0.0047189, -0.0004104, 0.0004104
2: -0.0017425, 0.0037425, -0.0017425, 0.0037425, -0.0030284, 0.0030284
3: 0.0013967, 0.0021226, 0.0013967, 0.0021226, -0.0004008, 0.0004008
4: 0.0032949, 0.0073941, 0.0032949, 0.0073941, -0.0022632, 0.0022632
5: 0.9964217, 0.9975605, 0.9964217, 0.9975605, -0.0006288, 0.0006288
6: 0.0046356, 0.0056694, 0.0046356, 0.0056694, -0.0005708, 0.0005708
7: -0.0060822, -0.0022245, -0.0060822, -0.0022245, -0.0021299, 0.0021299
8: -0.0074616, -0.0044591, -0.0074616, -0.0044591, -0.0016577, 0.0016577
9: -0.0036250, -0.0033660, -0.0036250, -0.0033660, -0.0001430, 0.0001430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004386
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004460
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0090853, -0.0063548, -0.0014463, 0.0016237
1: -0.0054623, -0.0047189, -0.0055001, -0.0047303, -0.0004078, 0.0004578
2: -0.0017425, 0.0037425, -0.0020214, 0.0036586, -0.0030087, 0.0033777
3: 0.0013967, 0.0021226, 0.0013598, 0.0021115, -0.0003981, 0.0004470
4: 0.0032949, 0.0073941, 0.0033577, 0.0076025, -0.0025243, 0.0022485
5: 0.9964217, 0.9975605, 0.9964391, 0.9976184, -0.0007013, 0.0006247
6: 0.0046356, 0.0056694, 0.0046514, 0.0057219, -0.0006366, 0.0005670
7: -0.0060822, -0.0022245, -0.0060232, -0.0020283, -0.0023756, 0.0021161
8: -0.0074616, -0.0044591, -0.0076142, -0.0045050, -0.0016469, 0.0018490
9: -0.0036250, -0.0033660, -0.0036211, -0.0033528, -0.0001595, 0.0001421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004416
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004483
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090705, -0.0063552, -0.0088972, -0.0063011, -0.0016293, 0.0014391
1: -0.0054960, -0.0047304, -0.0054471, -0.0047152, -0.0004594, 0.0004057
2: -0.0019906, 0.0036577, -0.0016302, 0.0037703, -0.0033892, 0.0029936
3: 0.0013639, 0.0021113, 0.0014116, 0.0021262, -0.0004485, 0.0003961
4: 0.0033583, 0.0075795, 0.0032742, 0.0073102, -0.0022372, 0.0025329
5: 0.9964393, 0.9976120, 0.9964159, 0.9975373, -0.0006216, 0.0007037
6: 0.0046516, 0.0057161, 0.0046304, 0.0056482, -0.0005642, 0.0006388
7: -0.0060226, -0.0020500, -0.0061018, -0.0023034, -0.0021055, 0.0023837
8: -0.0075974, -0.0045055, -0.0074001, -0.0044439, -0.0018553, 0.0016387
9: -0.0036210, -0.0033543, -0.0036263, -0.0033713, -0.0001414, 0.0001601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004320
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004351
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090709, -0.0063552, -0.0089102, -0.0062685, -0.0016619, 0.0014510
1: -0.0054961, -0.0047304, -0.0054508, -0.0047060, -0.0004686, 0.0004091
2: -0.0019915, 0.0036577, -0.0016573, 0.0038381, -0.0034571, 0.0030185
3: 0.0013637, 0.0021113, 0.0014080, 0.0021352, -0.0004575, 0.0003994
4: 0.0033583, 0.0075802, 0.0032235, 0.0073304, -0.0022558, 0.0025836
5: 0.9964393, 0.9976123, 0.9964018, 0.9975429, -0.0006267, 0.0007178
6: 0.0046516, 0.0057163, 0.0046176, 0.0056533, -0.0005689, 0.0006515
7: -0.0060226, -0.0020493, -0.0061495, -0.0022844, -0.0021230, 0.0024315
8: -0.0075979, -0.0045055, -0.0074149, -0.0044067, -0.0018924, 0.0016523
9: -0.0036210, -0.0033542, -0.0036295, -0.0033700, -0.0001426, 0.0001633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004320
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004351
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090640, -0.0064608, -0.0088591, -0.0062584, -0.0017464, 0.0014037
1: -0.0054941, -0.0047602, -0.0054364, -0.0047031, -0.0004924, 0.0003958
2: -0.0019770, 0.0034380, -0.0015509, 0.0038592, -0.0036330, 0.0029201
3: 0.0013657, 0.0020823, 0.0014221, 0.0021380, -0.0004808, 0.0003864
4: 0.0035225, 0.0075694, 0.0032078, 0.0072509, -0.0021823, 0.0027150
5: 0.9964849, 0.9976092, 0.9963975, 0.9975207, -0.0006063, 0.0007543
6: 0.0046930, 0.0057136, 0.0046136, 0.0056332, -0.0005503, 0.0006847
7: -0.0058680, -0.0020595, -0.0061643, -0.0023593, -0.0020538, 0.0025552
8: -0.0075899, -0.0046258, -0.0073567, -0.0043952, -0.0019887, 0.0015984
9: -0.0036106, -0.0033549, -0.0036305, -0.0033750, -0.0001379, 0.0001716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004233, upper bound: 0.0004332
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004233, upper bound: 0.0004415
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090669, -0.0064608, -0.0088781, -0.0062109, -0.0018093, 0.0014175
1: -0.0054950, -0.0047602, -0.0054417, -0.0046897, -0.0005101, 0.0003996
2: -0.0019831, 0.0034380, -0.0015904, 0.0039579, -0.0037638, 0.0029487
3: 0.0013649, 0.0020823, 0.0014168, 0.0021511, -0.0004981, 0.0003902
4: 0.0035225, 0.0075739, 0.0031340, 0.0072804, -0.0022037, 0.0028128
5: 0.9964849, 0.9976104, 0.9963769, 0.9975290, -0.0006122, 0.0007815
6: 0.0046930, 0.0057147, 0.0045950, 0.0056407, -0.0005557, 0.0007093
7: -0.0058680, -0.0020552, -0.0062337, -0.0023314, -0.0020739, 0.0026472
8: -0.0075933, -0.0046258, -0.0073783, -0.0043411, -0.0020603, 0.0016141
9: -0.0036106, -0.0033546, -0.0036352, -0.0033732, -0.0001393, 0.0001778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004307, upper bound: 0.0004332
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004308, upper bound: 0.0004415
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0091874, -0.0065092, -0.0088591, -0.0062584, -0.0019089, 0.0013899
1: -0.0055289, -0.0047738, -0.0054364, -0.0047031, -0.0005382, 0.0003919
2: -0.0022339, 0.0033374, -0.0015509, 0.0038592, -0.0039710, 0.0028912
3: 0.0013317, 0.0020689, 0.0014221, 0.0021380, -0.0005255, 0.0003826
4: 0.0035977, 0.0077613, 0.0032078, 0.0072509, -0.0021607, 0.0029676
5: 0.9965058, 0.9976625, 0.9963975, 0.9975207, -0.0006003, 0.0008245
6: 0.0047120, 0.0057620, 0.0046136, 0.0056332, -0.0005449, 0.0007484
7: -0.0057973, -0.0018789, -0.0061643, -0.0023593, -0.0020335, 0.0027929
8: -0.0077305, -0.0046808, -0.0073567, -0.0043952, -0.0021737, 0.0015826
9: -0.0036059, -0.0033428, -0.0036305, -0.0033750, -0.0001365, 0.0001875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003263, upper bound: 0.0003596
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003235, upper bound: 0.0003595
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0091882, -0.0065092, -0.0088781, -0.0062109, -0.0019617, 0.0014036
1: -0.0055292, -0.0047738, -0.0054417, -0.0046897, -0.0005531, 0.0003957
2: -0.0022354, 0.0033374, -0.0015904, 0.0039579, -0.0040808, 0.0029199
3: 0.0013315, 0.0020690, 0.0014168, 0.0021511, -0.0005400, 0.0003864
4: 0.0035977, 0.0077625, 0.0031340, 0.0072804, -0.0021821, 0.0030497
5: 0.9965058, 0.9976628, 0.9963769, 0.9975290, -0.0006063, 0.0008473
6: 0.0047119, 0.0057623, 0.0045950, 0.0056407, -0.0005503, 0.0007691
7: -0.0057973, -0.0018778, -0.0062337, -0.0023314, -0.0020536, 0.0028701
8: -0.0077314, -0.0046808, -0.0073783, -0.0043411, -0.0022338, 0.0015983
9: -0.0036059, -0.0033427, -0.0036352, -0.0033732, -0.0001379, 0.0001927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003603
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003602
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090126, -0.0064565, -0.0089807, -0.0063224, -0.0017097, 0.0015651
1: -0.0054796, -0.0047590, -0.0054707, -0.0047212, -0.0004820, 0.0004413
2: -0.0018701, 0.0034471, -0.0018038, 0.0037259, -0.0035565, 0.0032557
3: 0.0013798, 0.0020835, 0.0013886, 0.0021204, -0.0004706, 0.0004308
4: 0.0035157, 0.0074895, 0.0033074, 0.0074399, -0.0024331, 0.0026579
5: 0.9964830, 0.9975870, 0.9964252, 0.9975732, -0.0006760, 0.0007385
6: 0.0046913, 0.0056934, 0.0046387, 0.0056809, -0.0006136, 0.0006703
7: -0.0058744, -0.0021347, -0.0060705, -0.0021813, -0.0022898, 0.0025014
8: -0.0075314, -0.0046208, -0.0074951, -0.0044682, -0.0019468, 0.0017822
9: -0.0036111, -0.0033600, -0.0036242, -0.0033631, -0.0001538, 0.0001680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003196
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003196
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090048, -0.0064578, -0.0089639, -0.0063010, -0.0017387, 0.0015687
1: -0.0054774, -0.0047593, -0.0054659, -0.0047152, -0.0004902, 0.0004423
2: -0.0018539, 0.0034443, -0.0017688, 0.0037704, -0.0036169, 0.0032633
3: 0.0013820, 0.0020831, 0.0013932, 0.0021263, -0.0004786, 0.0004318
4: 0.0035178, 0.0074774, 0.0032741, 0.0074138, -0.0024388, 0.0027030
5: 0.9964836, 0.9975837, 0.9964159, 0.9975660, -0.0006776, 0.0007510
6: 0.0046918, 0.0056904, 0.0046303, 0.0056743, -0.0006150, 0.0006817
7: -0.0058725, -0.0021461, -0.0061019, -0.0022060, -0.0022951, 0.0025438
8: -0.0075226, -0.0046223, -0.0074760, -0.0044438, -0.0019799, 0.0017863
9: -0.0036109, -0.0033607, -0.0036263, -0.0033647, -0.0001541, 0.0001708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003168
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003168
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090329, -0.0064192, -0.0089816, -0.0063225, -0.0017203, 0.0016072
1: -0.0054854, -0.0047485, -0.0054709, -0.0047212, -0.0004850, 0.0004531
2: -0.0019124, 0.0035247, -0.0018056, 0.0037258, -0.0035785, 0.0033434
3: 0.0013742, 0.0020937, 0.0013884, 0.0021204, -0.0004736, 0.0004424
4: 0.0034577, 0.0075211, 0.0033074, 0.0074413, -0.0024986, 0.0026743
5: 0.9964669, 0.9975958, 0.9964252, 0.9975736, -0.0006942, 0.0007430
6: 0.0046767, 0.0057014, 0.0046387, 0.0056812, -0.0006301, 0.0006744
7: -0.0059290, -0.0021050, -0.0060705, -0.0021801, -0.0023515, 0.0025168
8: -0.0075546, -0.0045783, -0.0074961, -0.0044682, -0.0019589, 0.0018302
9: -0.0036147, -0.0033580, -0.0036242, -0.0033630, -0.0001579, 0.0001690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002936, upper bound: 0.0003104
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003183, upper bound: 0.0003493
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090231, -0.0064205, -0.0089640, -0.0063011, -0.0017487, 0.0016099
1: -0.0054826, -0.0047488, -0.0054659, -0.0047152, -0.0004930, 0.0004539
2: -0.0018920, 0.0035219, -0.0017690, 0.0037704, -0.0036376, 0.0033489
3: 0.0013769, 0.0020934, 0.0013932, 0.0021262, -0.0004814, 0.0004432
4: 0.0034598, 0.0075058, 0.0032741, 0.0074139, -0.0025027, 0.0027185
5: 0.9964675, 0.9975916, 0.9964159, 0.9975660, -0.0006953, 0.0007553
6: 0.0046772, 0.0056975, 0.0046304, 0.0056744, -0.0006312, 0.0006856
7: -0.0059271, -0.0021194, -0.0061018, -0.0022058, -0.0023553, 0.0025584
8: -0.0075434, -0.0045798, -0.0074761, -0.0044438, -0.0019912, 0.0018332
9: -0.0036146, -0.0033589, -0.0036263, -0.0033647, -0.0001582, 0.0001718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002925, upper bound: 0.0003073
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003181, upper bound: 0.0003486
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0090494, -0.0063830, -0.0015767, 0.0017507
1: -0.0054623, -0.0047189, -0.0054900, -0.0047383, -0.0004445, 0.0004936
2: -0.0017425, 0.0037425, -0.0019468, 0.0035999, -0.0032799, 0.0036418
3: 0.0013967, 0.0021226, 0.0013697, 0.0021037, -0.0004340, 0.0004819
4: 0.0032949, 0.0073941, 0.0034015, 0.0075468, -0.0027217, 0.0024512
5: 0.9964217, 0.9975605, 0.9964513, 0.9976029, -0.0007562, 0.0006810
6: 0.0046356, 0.0056694, 0.0046625, 0.0057079, -0.0006864, 0.0006182
7: -0.0060822, -0.0022245, -0.0059819, -0.0020808, -0.0025614, 0.0023068
8: -0.0074616, -0.0044591, -0.0075734, -0.0045371, -0.0017954, 0.0019935
9: -0.0036250, -0.0033660, -0.0036183, -0.0033563, -0.0001720, 0.0001549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003471
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003626
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0091638, -0.0064432, -0.0015263, 0.0018803
1: -0.0054623, -0.0047189, -0.0055223, -0.0047552, -0.0004303, 0.0005301
2: -0.0017425, 0.0037425, -0.0021847, 0.0034747, -0.0031750, 0.0039114
3: 0.0013967, 0.0021226, 0.0013382, 0.0020871, -0.0004202, 0.0005176
4: 0.0032949, 0.0073941, 0.0034951, 0.0077246, -0.0029231, 0.0023728
5: 0.9964217, 0.9975605, 0.9964772, 0.9976524, -0.0008121, 0.0006592
6: 0.0046356, 0.0056694, 0.0046861, 0.0057527, -0.0007372, 0.0005984
7: -0.0060822, -0.0022245, -0.0058939, -0.0019134, -0.0027510, 0.0022331
8: -0.0074616, -0.0044591, -0.0077036, -0.0046056, -0.0017380, 0.0021411
9: -0.0036250, -0.0033660, -0.0036124, -0.0033451, -0.0001847, 0.0001499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003471
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003626
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090705, -0.0063552, -0.0089900, -0.0063830, -0.0017511, 0.0017140
1: -0.0054960, -0.0047304, -0.0054733, -0.0047383, -0.0004937, 0.0004832
2: -0.0019906, 0.0036577, -0.0018232, 0.0036000, -0.0036426, 0.0035654
3: 0.0013639, 0.0021113, 0.0013860, 0.0021037, -0.0004820, 0.0004718
4: 0.0033583, 0.0075795, 0.0034014, 0.0074544, -0.0026645, 0.0027223
5: 0.9964393, 0.9976120, 0.9964513, 0.9975773, -0.0007403, 0.0007563
6: 0.0046516, 0.0057161, 0.0046625, 0.0056846, -0.0006720, 0.0006865
7: -0.0060226, -0.0020500, -0.0059820, -0.0021677, -0.0025076, 0.0025620
8: -0.0075974, -0.0045055, -0.0075058, -0.0045371, -0.0019940, 0.0019517
9: -0.0036210, -0.0033543, -0.0036183, -0.0033622, -0.0001684, 0.0001720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003143, upper bound: 0.0003178
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003091, upper bound: 0.0003169
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090709, -0.0063552, -0.0090145, -0.0063393, -0.0018073, 0.0017263
1: -0.0054961, -0.0047304, -0.0054802, -0.0047260, -0.0005095, 0.0004867
2: -0.0019915, 0.0036577, -0.0018740, 0.0036908, -0.0037596, 0.0035911
3: 0.0013637, 0.0021113, 0.0013793, 0.0021157, -0.0004975, 0.0004752
4: 0.0033583, 0.0075802, 0.0033336, 0.0074924, -0.0026837, 0.0028097
5: 0.9964393, 0.9976123, 0.9964324, 0.9975879, -0.0007456, 0.0007806
6: 0.0046516, 0.0057163, 0.0046454, 0.0056941, -0.0006768, 0.0007086
7: -0.0060226, -0.0020493, -0.0060458, -0.0021319, -0.0025257, 0.0026442
8: -0.0075979, -0.0045055, -0.0075336, -0.0044874, -0.0020580, 0.0019657
9: -0.0036210, -0.0033542, -0.0036226, -0.0033598, -0.0001696, 0.0001776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003476, upper bound: 0.0003315
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003446, upper bound: 0.0003314
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0089168, -0.0062414, -0.0016110, 0.0015259
1: -0.0054623, -0.0047189, -0.0054526, -0.0046983, -0.0004542, 0.0004302
2: -0.0017425, 0.0037425, -0.0016709, 0.0038945, -0.0033511, 0.0031741
3: 0.0013967, 0.0021226, 0.0014062, 0.0021427, -0.0004435, 0.0004200
4: 0.0032949, 0.0073941, 0.0031814, 0.0073406, -0.0023721, 0.0025044
5: 0.9964217, 0.9975605, 0.9963901, 0.9975457, -0.0006590, 0.0006958
6: 0.0046356, 0.0056694, 0.0046070, 0.0056559, -0.0005982, 0.0006316
7: -0.0060822, -0.0022245, -0.0061891, -0.0022748, -0.0022324, 0.0023569
8: -0.0074616, -0.0044591, -0.0074224, -0.0043759, -0.0018344, 0.0017375
9: -0.0036250, -0.0033660, -0.0036322, -0.0033694, -0.0001499, 0.0001583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004392
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004496
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089512, -0.0063144, -0.0090456, -0.0062859, -0.0015663, 0.0016632
1: -0.0054623, -0.0047189, -0.0054890, -0.0047109, -0.0004416, 0.0004689
2: -0.0017425, 0.0037425, -0.0019388, 0.0038018, -0.0032582, 0.0034598
3: 0.0013967, 0.0021226, 0.0013707, 0.0021304, -0.0004312, 0.0004579
4: 0.0032949, 0.0073941, 0.0032506, 0.0075408, -0.0025856, 0.0024349
5: 0.9964217, 0.9975605, 0.9964093, 0.9976013, -0.0007184, 0.0006765
6: 0.0046356, 0.0056694, 0.0046244, 0.0057064, -0.0006521, 0.0006141
7: -0.0060822, -0.0022245, -0.0061239, -0.0020864, -0.0024334, 0.0022916
8: -0.0074616, -0.0044591, -0.0075690, -0.0044266, -0.0017835, 0.0018939
9: -0.0036250, -0.0033660, -0.0036278, -0.0033567, -0.0001634, 0.0001539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004423
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004512
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090705, -0.0063552, -0.0088606, -0.0062315, -0.0017656, 0.0014974
1: -0.0054960, -0.0047304, -0.0054368, -0.0046956, -0.0004978, 0.0004222
2: -0.0019906, 0.0036577, -0.0015541, 0.0039150, -0.0036728, 0.0031149
3: 0.0013639, 0.0021113, 0.0014216, 0.0021454, -0.0004860, 0.0004122
4: 0.0033583, 0.0075795, 0.0031661, 0.0072533, -0.0023279, 0.0027448
5: 0.9964393, 0.9976120, 0.9963859, 0.9975214, -0.0006468, 0.0007626
6: 0.0046516, 0.0057161, 0.0046031, 0.0056338, -0.0005871, 0.0006922
7: -0.0060226, -0.0020500, -0.0062035, -0.0023570, -0.0021908, 0.0025832
8: -0.0075974, -0.0045055, -0.0073584, -0.0043647, -0.0020105, 0.0017051
9: -0.0036210, -0.0033543, -0.0036332, -0.0033749, -0.0001471, 0.0001735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004361
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004376
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090709, -0.0063552, -0.0088797, -0.0061942, -0.0018200, 0.0015110
1: -0.0054961, -0.0047304, -0.0054422, -0.0046850, -0.0005131, 0.0004260
2: -0.0019915, 0.0036577, -0.0015938, 0.0039926, -0.0037860, 0.0031433
3: 0.0013637, 0.0021113, 0.0014164, 0.0021557, -0.0005010, 0.0004160
4: 0.0033583, 0.0075802, 0.0031080, 0.0072829, -0.0023491, 0.0028294
5: 0.9964393, 0.9976123, 0.9963698, 0.9975296, -0.0006526, 0.0007861
6: 0.0046516, 0.0057163, 0.0045885, 0.0056413, -0.0005924, 0.0007135
7: -0.0060226, -0.0020493, -0.0062581, -0.0023291, -0.0022107, 0.0026628
8: -0.0075979, -0.0045055, -0.0073801, -0.0043221, -0.0020725, 0.0017206
9: -0.0036210, -0.0033542, -0.0036368, -0.0033730, -0.0001484, 0.0001788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004394, upper bound: 0.0004361
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004394, upper bound: 0.0004376
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090327, -0.0063834, -0.0088961, -0.0063344, -0.0016716, 0.0015245
1: -0.0054853, -0.0047384, -0.0054468, -0.0047246, -0.0004713, 0.0004298
2: -0.0019121, 0.0035992, -0.0016279, 0.0037011, -0.0034773, 0.0031713
3: 0.0013743, 0.0021036, 0.0014119, 0.0021171, -0.0004602, 0.0004197
4: 0.0034021, 0.0075208, 0.0033259, 0.0073085, -0.0023700, 0.0025987
5: 0.9964514, 0.9975958, 0.9964303, 0.9975367, -0.0006585, 0.0007220
6: 0.0046626, 0.0057013, 0.0046434, 0.0056478, -0.0005977, 0.0006554
7: -0.0059814, -0.0021052, -0.0060531, -0.0023051, -0.0022305, 0.0024457
8: -0.0075544, -0.0045375, -0.0073988, -0.0044817, -0.0019035, 0.0017360
9: -0.0036183, -0.0033580, -0.0036231, -0.0033714, -0.0001498, 0.0001642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004248, upper bound: 0.0004287
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004247, upper bound: 0.0004366
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090371, -0.0063834, -0.0089091, -0.0062932, -0.0017250, 0.0015360
1: -0.0054866, -0.0047384, -0.0054505, -0.0047129, -0.0004863, 0.0004331
2: -0.0019211, 0.0035991, -0.0016549, 0.0037868, -0.0035883, 0.0031953
3: 0.0013731, 0.0021036, 0.0014083, 0.0021284, -0.0004749, 0.0004228
4: 0.0034021, 0.0075276, 0.0032619, 0.0073286, -0.0023879, 0.0026817
5: 0.9964515, 0.9975976, 0.9964125, 0.9975423, -0.0006634, 0.0007450
6: 0.0046626, 0.0057030, 0.0046273, 0.0056528, -0.0006022, 0.0006763
7: -0.0059813, -0.0020988, -0.0061134, -0.0022861, -0.0022473, 0.0025237
8: -0.0075593, -0.0045376, -0.0074136, -0.0044348, -0.0019642, 0.0017491
9: -0.0036183, -0.0033576, -0.0036271, -0.0033701, -0.0001509, 0.0001695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004361, upper bound: 0.0004288
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004363, upper bound: 0.0004366
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0091481, -0.0064436, -0.0088961, -0.0063344, -0.0018037, 0.0014740
1: -0.0055179, -0.0047554, -0.0054468, -0.0047246, -0.0005085, 0.0004156
2: -0.0021521, 0.0034738, -0.0016279, 0.0037011, -0.0037521, 0.0030663
3: 0.0013425, 0.0020870, 0.0014119, 0.0021171, -0.0004965, 0.0004058
4: 0.0034957, 0.0077002, 0.0033259, 0.0073085, -0.0022915, 0.0028041
5: 0.9964774, 0.9976456, 0.9964303, 0.9975367, -0.0006367, 0.0007790
6: 0.0046862, 0.0057466, 0.0046434, 0.0056478, -0.0005779, 0.0007071
7: -0.0058933, -0.0019364, -0.0060531, -0.0023051, -0.0021566, 0.0026389
8: -0.0076858, -0.0046061, -0.0073988, -0.0044817, -0.0020539, 0.0016785
9: -0.0036123, -0.0033466, -0.0036231, -0.0033714, -0.0001448, 0.0001772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003300, upper bound: 0.0003481
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003278, upper bound: 0.0003491
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0091503, -0.0064436, -0.0089091, -0.0062932, -0.0018512, 0.0014856
1: -0.0055185, -0.0047554, -0.0054505, -0.0047129, -0.0005219, 0.0004188
2: -0.0021567, 0.0034739, -0.0016549, 0.0037868, -0.0038510, 0.0030903
3: 0.0013419, 0.0020870, 0.0014083, 0.0021284, -0.0005096, 0.0004090
4: 0.0034957, 0.0077036, 0.0032619, 0.0073286, -0.0023095, 0.0028780
5: 0.9964774, 0.9976466, 0.9964125, 0.9975423, -0.0006416, 0.0007996
6: 0.0046862, 0.0057474, 0.0046273, 0.0056528, -0.0005824, 0.0007258
7: -0.0058933, -0.0019332, -0.0061134, -0.0022861, -0.0021735, 0.0027085
8: -0.0076883, -0.0046061, -0.0074136, -0.0044348, -0.0021080, 0.0016916
9: -0.0036123, -0.0033464, -0.0036271, -0.0033701, -0.0001459, 0.0001819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003454, upper bound: 0.0003488
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003445, upper bound: 0.0003501
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089804, -0.0063862, -0.0090172, -0.0063894, -0.0016271, 0.0016880
1: -0.0054706, -0.0047392, -0.0054809, -0.0047401, -0.0004587, 0.0004759
2: -0.0018032, 0.0035932, -0.0018797, 0.0035865, -0.0033847, 0.0035114
3: 0.0013887, 0.0021028, 0.0013785, 0.0021019, -0.0004479, 0.0004647
4: 0.0034065, 0.0074395, 0.0034115, 0.0074967, -0.0026242, 0.0025295
5: 0.9964527, 0.9975731, 0.9964541, 0.9975890, -0.0007291, 0.0007028
6: 0.0046637, 0.0056808, 0.0046650, 0.0056952, -0.0006618, 0.0006379
7: -0.0059772, -0.0021818, -0.0059725, -0.0021279, -0.0024697, 0.0023806
8: -0.0074948, -0.0045408, -0.0075367, -0.0045445, -0.0018528, 0.0019222
9: -0.0036180, -0.0033631, -0.0036177, -0.0033595, -0.0001658, 0.0001599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002771, upper bound: 0.0002617
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003098, upper bound: 0.0003060
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089695, -0.0063876, -0.0090056, -0.0063652, -0.0016610, 0.0016906
1: -0.0054675, -0.0047396, -0.0054777, -0.0047332, -0.0004683, 0.0004767
2: -0.0017805, 0.0035904, -0.0018556, 0.0036370, -0.0034551, 0.0035169
3: 0.0013917, 0.0021024, 0.0013817, 0.0021086, -0.0004572, 0.0004654
4: 0.0034086, 0.0074225, 0.0033738, 0.0074787, -0.0026283, 0.0025822
5: 0.9964533, 0.9975684, 0.9964436, 0.9975840, -0.0007302, 0.0007174
6: 0.0046643, 0.0056765, 0.0046555, 0.0056907, -0.0006628, 0.0006512
7: -0.0059752, -0.0021977, -0.0060080, -0.0021449, -0.0024735, 0.0024301
8: -0.0074824, -0.0045423, -0.0075235, -0.0045168, -0.0018914, 0.0019251
9: -0.0036178, -0.0033642, -0.0036200, -0.0033606, -0.0001661, 0.0001632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002741, upper bound: 0.0002528
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003090, upper bound: 0.0003008
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090049, -0.0063428, -0.0090174, -0.0063895, -0.0016395, 0.0017441
1: -0.0054775, -0.0047269, -0.0054810, -0.0047401, -0.0004623, 0.0004917
2: -0.0018542, 0.0036836, -0.0018801, 0.0035865, -0.0034106, 0.0036281
3: 0.0013819, 0.0021148, 0.0013785, 0.0021019, -0.0004513, 0.0004801
4: 0.0033390, 0.0074776, 0.0034116, 0.0074969, -0.0027114, 0.0025489
5: 0.9964339, 0.9975837, 0.9964541, 0.9975891, -0.0007533, 0.0007082
6: 0.0046467, 0.0056904, 0.0046650, 0.0056953, -0.0006838, 0.0006428
7: -0.0060408, -0.0021459, -0.0059725, -0.0021277, -0.0025518, 0.0023988
8: -0.0075227, -0.0044913, -0.0075369, -0.0045445, -0.0018670, 0.0019860
9: -0.0036222, -0.0033607, -0.0036177, -0.0033595, -0.0001713, 0.0001611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003017, upper bound: 0.0003023
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003241, upper bound: 0.0003406
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089936, -0.0063441, -0.0090061, -0.0063652, -0.0016728, 0.0017467
1: -0.0054743, -0.0047273, -0.0054778, -0.0047332, -0.0004716, 0.0004925
2: -0.0018306, 0.0036808, -0.0018567, 0.0036370, -0.0034798, 0.0036335
3: 0.0013850, 0.0021144, 0.0013816, 0.0021086, -0.0004605, 0.0004808
4: 0.0033411, 0.0074599, 0.0033738, 0.0074794, -0.0027155, 0.0026006
5: 0.9964346, 0.9975789, 0.9964436, 0.9975842, -0.0007544, 0.0007225
6: 0.0046472, 0.0056860, 0.0046555, 0.0056909, -0.0006848, 0.0006558
7: -0.0060388, -0.0021625, -0.0060080, -0.0021442, -0.0025556, 0.0024474
8: -0.0075098, -0.0044929, -0.0075240, -0.0045168, -0.0019048, 0.0019890
9: -0.0036221, -0.0033618, -0.0036200, -0.0033606, -0.0001716, 0.0001643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002994, upper bound: 0.0002951
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003241, upper bound: 0.0003377
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0090801, -0.0064605, -0.0014643, 0.0018174
1: -0.0054526, -0.0046983, -0.0054987, -0.0047601, -0.0004128, 0.0005124
2: -0.0016709, 0.0038945, -0.0020105, 0.0034388, -0.0030460, 0.0037805
3: 0.0014062, 0.0021427, 0.0013612, 0.0020824, -0.0004031, 0.0005003
4: 0.0031814, 0.0073406, 0.0035219, 0.0075944, -0.0028253, 0.0022764
5: 0.9963901, 0.9975457, 0.9964848, 0.9976162, -0.0007850, 0.0006324
6: 0.0046070, 0.0056559, 0.0046929, 0.0057199, -0.0007125, 0.0005741
7: -0.0061891, -0.0022748, -0.0058686, -0.0020360, -0.0026590, 0.0021423
8: -0.0074224, -0.0043759, -0.0076083, -0.0046253, -0.0016674, 0.0020695
9: -0.0036322, -0.0033694, -0.0036107, -0.0033533, -0.0001785, 0.0001439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003441
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003770, upper bound: 0.0003572
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0092025, -0.0065088, -0.0014505, 0.0019753
1: -0.0054526, -0.0046983, -0.0055332, -0.0047737, -0.0004090, 0.0005569
2: -0.0016709, 0.0038945, -0.0022651, 0.0033383, -0.0030174, 0.0041089
3: 0.0014062, 0.0021427, 0.0013275, 0.0020691, -0.0003993, 0.0005438
4: 0.0031814, 0.0073406, 0.0035970, 0.0077847, -0.0030708, 0.0022550
5: 0.9963901, 0.9975457, 0.9965056, 0.9976690, -0.0008531, 0.0006265
6: 0.0046070, 0.0056559, 0.0047118, 0.0057679, -0.0007744, 0.0005687
7: -0.0061891, -0.0022748, -0.0057979, -0.0018569, -0.0028899, 0.0021222
8: -0.0074224, -0.0043759, -0.0077476, -0.0046803, -0.0016517, 0.0022492
9: -0.0036322, -0.0033694, -0.0036059, -0.0033413, -0.0001941, 0.0001425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003441
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003770, upper bound: 0.0003572
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090309, -0.0062864, -0.0090229, -0.0064532, -0.0016254, 0.0017918
1: -0.0054848, -0.0047110, -0.0054826, -0.0047581, -0.0004583, 0.0005052
2: -0.0019084, 0.0038009, -0.0018917, 0.0034538, -0.0033812, 0.0037272
3: 0.0013748, 0.0021303, 0.0013770, 0.0020843, -0.0004474, 0.0004932
4: 0.0032513, 0.0075181, 0.0035107, 0.0075056, -0.0027855, 0.0025269
5: 0.9964095, 0.9975950, 0.9964817, 0.9975916, -0.0007739, 0.0007020
6: 0.0046246, 0.0057006, 0.0046900, 0.0056975, -0.0007025, 0.0006372
7: -0.0061233, -0.0021078, -0.0058792, -0.0021196, -0.0026215, 0.0023781
8: -0.0075524, -0.0044271, -0.0075432, -0.0046171, -0.0018509, 0.0020403
9: -0.0036278, -0.0033582, -0.0036114, -0.0033589, -0.0001760, 0.0001597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003196, upper bound: 0.0003095
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003168, upper bound: 0.0003082
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090310, -0.0062864, -0.0090429, -0.0064158, -0.0016674, 0.0018024
1: -0.0054848, -0.0047110, -0.0054882, -0.0047475, -0.0004701, 0.0005082
2: -0.0019084, 0.0038009, -0.0019332, 0.0035317, -0.0034685, 0.0037493
3: 0.0013747, 0.0021303, 0.0013715, 0.0020947, -0.0004590, 0.0004962
4: 0.0032513, 0.0075181, 0.0034525, 0.0075366, -0.0028020, 0.0025921
5: 0.9964095, 0.9975950, 0.9964654, 0.9976001, -0.0007785, 0.0007202
6: 0.0046246, 0.0057006, 0.0046753, 0.0057053, -0.0007066, 0.0006537
7: -0.0061233, -0.0021077, -0.0059340, -0.0020904, -0.0026370, 0.0024395
8: -0.0075524, -0.0044271, -0.0075659, -0.0045744, -0.0018986, 0.0020524
9: -0.0036278, -0.0033582, -0.0036151, -0.0033570, -0.0001771, 0.0001638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003561, upper bound: 0.0003255
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003554, upper bound: 0.0003250
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0089512, -0.0063144, -0.0015259, 0.0016110
1: -0.0054526, -0.0046983, -0.0054623, -0.0047189, -0.0004302, 0.0004542
2: -0.0016709, 0.0038945, -0.0017425, 0.0037425, -0.0031741, 0.0033511
3: 0.0014062, 0.0021427, 0.0013967, 0.0021226, -0.0004200, 0.0004435
4: 0.0031814, 0.0073406, 0.0032949, 0.0073941, -0.0025044, 0.0023721
5: 0.9963901, 0.9975457, 0.9964217, 0.9975605, -0.0006958, 0.0006590
6: 0.0046070, 0.0056559, 0.0046356, 0.0056694, -0.0006316, 0.0005982
7: -0.0061891, -0.0022748, -0.0060822, -0.0022245, -0.0023570, 0.0022324
8: -0.0074224, -0.0043759, -0.0074616, -0.0044591, -0.0017375, 0.0018344
9: -0.0036322, -0.0033694, -0.0036250, -0.0033660, -0.0001583, 0.0001499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004386
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004460
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0090853, -0.0063548, -0.0015164, 0.0017789
1: -0.0054526, -0.0046983, -0.0055001, -0.0047303, -0.0004275, 0.0005015
2: -0.0016709, 0.0038945, -0.0020214, 0.0036586, -0.0031545, 0.0037005
3: 0.0014062, 0.0021427, 0.0013598, 0.0021115, -0.0004174, 0.0004897
4: 0.0031814, 0.0073406, 0.0033577, 0.0076025, -0.0027655, 0.0023575
5: 0.9963901, 0.9975457, 0.9964391, 0.9976184, -0.0007683, 0.0006550
6: 0.0046070, 0.0056559, 0.0046514, 0.0057219, -0.0006974, 0.0005945
7: -0.0061891, -0.0022748, -0.0060232, -0.0020283, -0.0026027, 0.0022187
8: -0.0074224, -0.0043759, -0.0076142, -0.0045050, -0.0017268, 0.0020257
9: -0.0036322, -0.0033694, -0.0036211, -0.0033528, -0.0001748, 0.0001490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004417
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004483
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090309, -0.0062864, -0.0088972, -0.0063011, -0.0016650, 0.0015959
1: -0.0054848, -0.0047110, -0.0054471, -0.0047152, -0.0004694, 0.0004499
2: -0.0019084, 0.0038009, -0.0016302, 0.0037703, -0.0034635, 0.0033198
3: 0.0013748, 0.0021303, 0.0014116, 0.0021262, -0.0004583, 0.0004393
4: 0.0032513, 0.0075181, 0.0032742, 0.0073102, -0.0024810, 0.0025884
5: 0.9964095, 0.9975950, 0.9964159, 0.9975373, -0.0006893, 0.0007191
6: 0.0046246, 0.0057006, 0.0046304, 0.0056482, -0.0006257, 0.0006528
7: -0.0061233, -0.0021078, -0.0061018, -0.0023034, -0.0023349, 0.0024360
8: -0.0075524, -0.0044271, -0.0074001, -0.0044439, -0.0018959, 0.0018173
9: -0.0036278, -0.0033582, -0.0036263, -0.0033713, -0.0001568, 0.0001636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004347, upper bound: 0.0004320
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004347, upper bound: 0.0004351
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090310, -0.0062864, -0.0089102, -0.0062685, -0.0017054, 0.0016079
1: -0.0054848, -0.0047110, -0.0054508, -0.0047060, -0.0004808, 0.0004533
2: -0.0019084, 0.0038009, -0.0016573, 0.0038381, -0.0035476, 0.0033447
3: 0.0013747, 0.0021303, 0.0014080, 0.0021352, -0.0004695, 0.0004426
4: 0.0032513, 0.0075181, 0.0032235, 0.0073304, -0.0024996, 0.0026512
5: 0.9964095, 0.9975950, 0.9964018, 0.9975429, -0.0006945, 0.0007366
6: 0.0046246, 0.0057006, 0.0046176, 0.0056533, -0.0006304, 0.0006686
7: -0.0061233, -0.0021077, -0.0061495, -0.0022844, -0.0023524, 0.0024951
8: -0.0075524, -0.0044271, -0.0074149, -0.0044067, -0.0019419, 0.0018309
9: -0.0036278, -0.0033582, -0.0036295, -0.0033700, -0.0001580, 0.0001675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004428, upper bound: 0.0004320
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004428, upper bound: 0.0004351
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0090327, -0.0063834, -0.0088591, -0.0062584, -0.0016149, 0.0013779
1: -0.0054853, -0.0047384, -0.0054364, -0.0047031, -0.0004553, 0.0003885
2: -0.0019121, 0.0035992, -0.0015509, 0.0038592, -0.0033594, 0.0028663
3: 0.0013743, 0.0021036, 0.0014221, 0.0021380, -0.0004446, 0.0003793
4: 0.0034021, 0.0075208, 0.0032078, 0.0072509, -0.0021421, 0.0025106
5: 0.9964514, 0.9975958, 0.9963975, 0.9975207, -0.0005951, 0.0006975
6: 0.0046626, 0.0057013, 0.0046136, 0.0056332, -0.0005402, 0.0006331
7: -0.0059814, -0.0021052, -0.0061643, -0.0023593, -0.0020159, 0.0023627
8: -0.0075544, -0.0045375, -0.0073567, -0.0043952, -0.0018389, 0.0015690
9: -0.0036183, -0.0033580, -0.0036305, -0.0033750, -0.0001354, 0.0001587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004292
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004369
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0090371, -0.0063834, -0.0088781, -0.0062109, -0.0016674, 0.0013911
1: -0.0054866, -0.0047384, -0.0054417, -0.0046897, -0.0004701, 0.0003922
2: -0.0019211, 0.0035991, -0.0015904, 0.0039579, -0.0034685, 0.0028937
3: 0.0013731, 0.0021036, 0.0014168, 0.0021511, -0.0004590, 0.0003829
4: 0.0034021, 0.0075276, 0.0031340, 0.0072804, -0.0021626, 0.0025921
5: 0.9964515, 0.9975976, 0.9963769, 0.9975290, -0.0006008, 0.0007202
6: 0.0046626, 0.0057030, 0.0045950, 0.0056407, -0.0005454, 0.0006537
7: -0.0059813, -0.0020988, -0.0062337, -0.0023314, -0.0020352, 0.0024395
8: -0.0075593, -0.0045376, -0.0073783, -0.0043411, -0.0018987, 0.0015840
9: -0.0036183, -0.0033576, -0.0036352, -0.0033732, -0.0001367, 0.0001638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004364, upper bound: 0.0004292
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004365, upper bound: 0.0004369
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0091481, -0.0064436, -0.0088591, -0.0062584, -0.0017735, 0.0013638
1: -0.0055179, -0.0047554, -0.0054364, -0.0047031, -0.0005000, 0.0003845
2: -0.0021521, 0.0034738, -0.0015509, 0.0038592, -0.0036893, 0.0028369
3: 0.0013425, 0.0020870, 0.0014221, 0.0021380, -0.0004882, 0.0003754
4: 0.0034957, 0.0077002, 0.0032078, 0.0072509, -0.0021201, 0.0027572
5: 0.9964774, 0.9976456, 0.9963975, 0.9975207, -0.0005890, 0.0007660
6: 0.0046862, 0.0057466, 0.0046136, 0.0056332, -0.0005347, 0.0006953
7: -0.0058933, -0.0019364, -0.0061643, -0.0023593, -0.0019953, 0.0025948
8: -0.0076858, -0.0046061, -0.0073567, -0.0043952, -0.0020195, 0.0015529
9: -0.0036123, -0.0033466, -0.0036305, -0.0033750, -0.0001340, 0.0001742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003340, upper bound: 0.0003559
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003571
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0091503, -0.0064436, -0.0088781, -0.0062109, -0.0018159, 0.0013770
1: -0.0055185, -0.0047554, -0.0054417, -0.0046897, -0.0005120, 0.0003882
2: -0.0021567, 0.0034739, -0.0015904, 0.0039579, -0.0037774, 0.0028644
3: 0.0013419, 0.0020870, 0.0014168, 0.0021511, -0.0004999, 0.0003791
4: 0.0034957, 0.0077036, 0.0031340, 0.0072804, -0.0021407, 0.0028230
5: 0.9964774, 0.9976466, 0.9963769, 0.9975290, -0.0005948, 0.0007843
6: 0.0046862, 0.0057474, 0.0045950, 0.0056407, -0.0005399, 0.0007119
7: -0.0058933, -0.0019332, -0.0062337, -0.0023314, -0.0020146, 0.0026567
8: -0.0076883, -0.0046061, -0.0073783, -0.0043411, -0.0020678, 0.0015680
9: -0.0036123, -0.0033464, -0.0036352, -0.0033732, -0.0001353, 0.0001784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003569
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003462, upper bound: 0.0003583
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089804, -0.0063862, -0.0089807, -0.0063224, -0.0015689, 0.0015556
1: -0.0054706, -0.0047392, -0.0054707, -0.0047212, -0.0004423, 0.0004386
2: -0.0018032, 0.0035932, -0.0018038, 0.0037259, -0.0032637, 0.0032360
3: 0.0013887, 0.0021028, 0.0013886, 0.0021204, -0.0004319, 0.0004282
4: 0.0034065, 0.0074395, 0.0033074, 0.0074399, -0.0024184, 0.0024391
5: 0.9964527, 0.9975731, 0.9964252, 0.9975732, -0.0006719, 0.0006776
6: 0.0046637, 0.0056808, 0.0046387, 0.0056809, -0.0006099, 0.0006151
7: -0.0059772, -0.0021818, -0.0060705, -0.0021813, -0.0022760, 0.0022954
8: -0.0074948, -0.0045408, -0.0074951, -0.0044682, -0.0017865, 0.0017714
9: -0.0036180, -0.0033631, -0.0036242, -0.0033631, -0.0001528, 0.0001541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002847, upper bound: 0.0002810
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003132, upper bound: 0.0003164
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089695, -0.0063876, -0.0089639, -0.0063010, -0.0016099, 0.0015550
1: -0.0054675, -0.0047396, -0.0054659, -0.0047152, -0.0004539, 0.0004384
2: -0.0017805, 0.0035904, -0.0017688, 0.0037704, -0.0033489, 0.0032346
3: 0.0013917, 0.0021024, 0.0013932, 0.0021263, -0.0004432, 0.0004281
4: 0.0034086, 0.0074225, 0.0032741, 0.0074138, -0.0024174, 0.0025027
5: 0.9964533, 0.9975684, 0.9964159, 0.9975660, -0.0006716, 0.0006953
6: 0.0046643, 0.0056765, 0.0046303, 0.0056743, -0.0006096, 0.0006312
7: -0.0059752, -0.0021977, -0.0061019, -0.0022060, -0.0022750, 0.0023554
8: -0.0074824, -0.0045423, -0.0074760, -0.0044438, -0.0018332, 0.0017706
9: -0.0036178, -0.0033642, -0.0036263, -0.0033647, -0.0001528, 0.0001582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002830, upper bound: 0.0002759
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003127, upper bound: 0.0003131
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090049, -0.0063428, -0.0089816, -0.0063225, -0.0015817, 0.0015938
1: -0.0054775, -0.0047269, -0.0054709, -0.0047212, -0.0004459, 0.0004494
2: -0.0018542, 0.0036836, -0.0018056, 0.0037258, -0.0032902, 0.0033155
3: 0.0013819, 0.0021148, 0.0013884, 0.0021204, -0.0004354, 0.0004387
4: 0.0033390, 0.0074776, 0.0033074, 0.0074413, -0.0024778, 0.0024589
5: 0.9964339, 0.9975837, 0.9964252, 0.9975736, -0.0006884, 0.0006832
6: 0.0046467, 0.0056904, 0.0046387, 0.0056812, -0.0006249, 0.0006201
7: -0.0060408, -0.0021459, -0.0060705, -0.0021801, -0.0023319, 0.0023141
8: -0.0075227, -0.0044913, -0.0074961, -0.0044682, -0.0018011, 0.0018149
9: -0.0036222, -0.0033607, -0.0036242, -0.0033630, -0.0001566, 0.0001554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003069, upper bound: 0.0003155
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003280, upper bound: 0.0003493
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0089936, -0.0063441, -0.0089640, -0.0063011, -0.0016225, 0.0015911
1: -0.0054743, -0.0047273, -0.0054659, -0.0047152, -0.0004574, 0.0004486
2: -0.0018306, 0.0036808, -0.0017690, 0.0037704, -0.0033751, 0.0033098
3: 0.0013850, 0.0021144, 0.0013932, 0.0021262, -0.0004466, 0.0004380
4: 0.0033411, 0.0074599, 0.0032741, 0.0074139, -0.0024735, 0.0025223
5: 0.9964346, 0.9975789, 0.9964159, 0.9975660, -0.0006872, 0.0007008
6: 0.0046472, 0.0056860, 0.0046304, 0.0056744, -0.0006238, 0.0006361
7: -0.0060388, -0.0021625, -0.0061018, -0.0022058, -0.0023279, 0.0023738
8: -0.0075098, -0.0044929, -0.0074761, -0.0044438, -0.0018475, 0.0018118
9: -0.0036221, -0.0033618, -0.0036263, -0.0033647, -0.0001563, 0.0001594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003057, upper bound: 0.0003100
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003279, upper bound: 0.0003475
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0090494, -0.0063830, -0.0014309, 0.0016936
1: -0.0054526, -0.0046983, -0.0054900, -0.0047383, -0.0004034, 0.0004775
2: -0.0016709, 0.0038945, -0.0019468, 0.0035999, -0.0029765, 0.0035230
3: 0.0014062, 0.0021427, 0.0013697, 0.0021037, -0.0003939, 0.0004662
4: 0.0031814, 0.0073406, 0.0034015, 0.0075468, -0.0026329, 0.0022244
5: 0.9963901, 0.9975457, 0.9964513, 0.9976029, -0.0007315, 0.0006180
6: 0.0046070, 0.0056559, 0.0046625, 0.0057079, -0.0006640, 0.0005610
7: -0.0061891, -0.0022748, -0.0059819, -0.0020808, -0.0024778, 0.0020935
8: -0.0074224, -0.0043759, -0.0075734, -0.0045371, -0.0016293, 0.0019285
9: -0.0036322, -0.0033694, -0.0036183, -0.0033563, -0.0001664, 0.0001406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003807, upper bound: 0.0003510
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003811, upper bound: 0.0003639
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0091638, -0.0064432, -0.0014169, 0.0018479
1: -0.0054526, -0.0046983, -0.0055223, -0.0047552, -0.0003995, 0.0005210
2: -0.0016709, 0.0038945, -0.0021847, 0.0034747, -0.0029474, 0.0038441
3: 0.0014062, 0.0021427, 0.0013382, 0.0020871, -0.0003900, 0.0005087
4: 0.0031814, 0.0073406, 0.0034951, 0.0077246, -0.0028728, 0.0022027
5: 0.9963901, 0.9975457, 0.9964772, 0.9976524, -0.0007982, 0.0006120
6: 0.0046070, 0.0056559, 0.0046861, 0.0057527, -0.0007245, 0.0005555
7: -0.0061891, -0.0022748, -0.0058939, -0.0019134, -0.0027037, 0.0020730
8: -0.0074224, -0.0043759, -0.0077036, -0.0046056, -0.0016134, 0.0021043
9: -0.0036322, -0.0033694, -0.0036124, -0.0033451, -0.0001815, 0.0001392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003807, upper bound: 0.0003510
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003811, upper bound: 0.0003639
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090309, -0.0062864, -0.0089900, -0.0063830, -0.0016183, 0.0016584
1: -0.0054848, -0.0047110, -0.0054733, -0.0047383, -0.0004563, 0.0004676
2: -0.0019084, 0.0038009, -0.0018232, 0.0036000, -0.0033665, 0.0034499
3: 0.0013748, 0.0021303, 0.0013860, 0.0021037, -0.0004455, 0.0004565
4: 0.0032513, 0.0075181, 0.0034014, 0.0074544, -0.0025782, 0.0025159
5: 0.9964095, 0.9975950, 0.9964513, 0.9975773, -0.0007163, 0.0006990
6: 0.0046246, 0.0057006, 0.0046625, 0.0056846, -0.0006502, 0.0006345
7: -0.0061233, -0.0021078, -0.0059820, -0.0021677, -0.0024264, 0.0023677
8: -0.0075524, -0.0044271, -0.0075058, -0.0045371, -0.0018428, 0.0018885
9: -0.0036278, -0.0033582, -0.0036183, -0.0033622, -0.0001629, 0.0001590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003280, upper bound: 0.0003193
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003258, upper bound: 0.0003190
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090310, -0.0062864, -0.0090145, -0.0063393, -0.0016572, 0.0016713
1: -0.0054848, -0.0047110, -0.0054802, -0.0047260, -0.0004672, 0.0004712
2: -0.0019084, 0.0038009, -0.0018740, 0.0036908, -0.0034473, 0.0034766
3: 0.0013747, 0.0021303, 0.0013793, 0.0021157, -0.0004562, 0.0004601
4: 0.0032513, 0.0075181, 0.0033336, 0.0074924, -0.0025982, 0.0025763
5: 0.9964095, 0.9975950, 0.9964324, 0.9975879, -0.0007219, 0.0007158
6: 0.0046246, 0.0057006, 0.0046454, 0.0056941, -0.0006552, 0.0006497
7: -0.0061233, -0.0021077, -0.0060458, -0.0021319, -0.0024452, 0.0024246
8: -0.0075524, -0.0044271, -0.0075336, -0.0044874, -0.0018871, 0.0019031
9: -0.0036278, -0.0033582, -0.0036226, -0.0033598, -0.0001642, 0.0001628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003613, upper bound: 0.0003335
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003606, upper bound: 0.0003338
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0089168, -0.0062414, -0.0014707, 0.0014707
1: -0.0054526, -0.0046983, -0.0054526, -0.0046983, -0.0004146, 0.0004146
2: -0.0016709, 0.0038945, -0.0016709, 0.0038945, -0.0030593, 0.0030593
3: 0.0014062, 0.0021427, 0.0014062, 0.0021427, -0.0004049, 0.0004049
4: 0.0031814, 0.0073406, 0.0031814, 0.0073406, -0.0022864, 0.0022864
5: 0.9963901, 0.9975457, 0.9963901, 0.9975457, -0.0006352, 0.0006352
6: 0.0046070, 0.0056559, 0.0046070, 0.0056559, -0.0005766, 0.0005766
7: -0.0061891, -0.0022748, -0.0061891, -0.0022748, -0.0021517, 0.0021517
8: -0.0074224, -0.0043759, -0.0074224, -0.0043759, -0.0016747, 0.0016747
9: -0.0036322, -0.0033694, -0.0036322, -0.0033694, -0.0001445, 0.0001445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004525, upper bound: 0.0004378
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004524, upper bound: 0.0004461
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0089168, -0.0062414, -0.0090456, -0.0062859, -0.0014608, 0.0016344
1: -0.0054526, -0.0046983, -0.0054890, -0.0047109, -0.0004119, 0.0004608
2: -0.0016709, 0.0038945, -0.0019388, 0.0038018, -0.0030388, 0.0033999
3: 0.0014062, 0.0021427, 0.0013707, 0.0021304, -0.0004021, 0.0004499
4: 0.0031814, 0.0073406, 0.0032506, 0.0075408, -0.0025409, 0.0022710
5: 0.9963901, 0.9975457, 0.9964093, 0.9976013, -0.0007059, 0.0006310
6: 0.0046070, 0.0056559, 0.0046244, 0.0057064, -0.0006408, 0.0005727
7: -0.0061891, -0.0022748, -0.0061239, -0.0020864, -0.0023913, 0.0021373
8: -0.0074224, -0.0043759, -0.0075690, -0.0044266, -0.0016635, 0.0018611
9: -0.0036322, -0.0033694, -0.0036278, -0.0033567, -0.0001606, 0.0001435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004525, upper bound: 0.0004407
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004524, upper bound: 0.0004482
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0090309, -0.0062864, -0.0088606, -0.0062315, -0.0016385, 0.0014450
1: -0.0054848, -0.0047110, -0.0054368, -0.0046956, -0.0004620, 0.0004074
2: -0.0019084, 0.0038009, -0.0015541, 0.0039150, -0.0034084, 0.0030060
3: 0.0013748, 0.0021303, 0.0014216, 0.0021454, -0.0004510, 0.0003978
4: 0.0032513, 0.0075181, 0.0031661, 0.0072533, -0.0022465, 0.0025472
5: 0.9964095, 0.9975950, 0.9963859, 0.9975214, -0.0006241, 0.0007077
6: 0.0046246, 0.0057006, 0.0046031, 0.0056338, -0.0005665, 0.0006424
7: -0.0061233, -0.0021078, -0.0062035, -0.0023570, -0.0021142, 0.0023972
8: -0.0075524, -0.0044271, -0.0073584, -0.0043647, -0.0018658, 0.0016455
9: -0.0036278, -0.0033582, -0.0036332, -0.0033749, -0.0001420, 0.0001610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004357, upper bound: 0.0004329
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004357, upper bound: 0.0004355
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0090310, -0.0062864, -0.0088797, -0.0061942, -0.0016738, 0.0014573
1: -0.0054848, -0.0047110, -0.0054422, -0.0046850, -0.0004719, 0.0004109
2: -0.0019084, 0.0038009, -0.0015938, 0.0039926, -0.0034819, 0.0030314
3: 0.0013747, 0.0021303, 0.0014164, 0.0021557, -0.0004608, 0.0004012
4: 0.0032513, 0.0075181, 0.0031080, 0.0072829, -0.0022655, 0.0026021
5: 0.9964095, 0.9975950, 0.9963698, 0.9975296, -0.0006294, 0.0007230
6: 0.0046246, 0.0057006, 0.0045885, 0.0056413, -0.0005713, 0.0006562
7: -0.0061233, -0.0021077, -0.0062581, -0.0023291, -0.0021321, 0.0024489
8: -0.0075524, -0.0044271, -0.0073801, -0.0043221, -0.0019060, 0.0016594
9: -0.0036278, -0.0033582, -0.0036368, -0.0033730, -0.0001432, 0.0001644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004434, upper bound: 0.0004329
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004355
time: 0.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.53 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004233, upper bound: 0.0004288
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004232, upper bound: 0.0004366
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004308, upper bound: 0.0004287
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004308, upper bound: 0.0004365
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003261, upper bound: 0.0003481
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003231, upper bound: 0.0003491
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003488
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003501
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002936, upper bound: 0.0003008
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003183, upper bound: 0.0003406
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002925, upper bound: 0.0002944
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003181, upper bound: 0.0003377
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003439
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003572
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003439
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003572
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003128, upper bound: 0.0003095
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003079, upper bound: 0.0003082
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003476, upper bound: 0.0003255
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003446, upper bound: 0.0003250
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004386
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004460
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004416
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004483
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004320
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004351
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004320
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004395, upper bound: 0.0004351
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004233, upper bound: 0.0004332
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004233, upper bound: 0.0004415
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004307, upper bound: 0.0004332
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004308, upper bound: 0.0004415
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003263, upper bound: 0.0003596
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003235, upper bound: 0.0003595
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003408, upper bound: 0.0003603
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003391, upper bound: 0.0003602
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003196
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003095, upper bound: 0.0003196
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003168
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003168
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002936, upper bound: 0.0003104
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003183, upper bound: 0.0003493
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002925, upper bound: 0.0003073
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003181, upper bound: 0.0003486
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003471
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003626
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003685, upper bound: 0.0003471
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003693, upper bound: 0.0003626
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003143, upper bound: 0.0003178
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003091, upper bound: 0.0003169
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003476, upper bound: 0.0003315
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003446, upper bound: 0.0003314
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004392
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004496
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004423
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004473, upper bound: 0.0004512
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004361
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004329, upper bound: 0.0004376
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004394, upper bound: 0.0004361
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004394, upper bound: 0.0004376
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004248, upper bound: 0.0004287
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004247, upper bound: 0.0004366
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004361, upper bound: 0.0004288
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004363, upper bound: 0.0004366
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003300, upper bound: 0.0003481
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003278, upper bound: 0.0003491
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003454, upper bound: 0.0003488
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003445, upper bound: 0.0003501
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002771, upper bound: 0.0002617
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003098, upper bound: 0.0003060
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002741, upper bound: 0.0002528
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003090, upper bound: 0.0003008
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003017, upper bound: 0.0003023
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003241, upper bound: 0.0003406
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002994, upper bound: 0.0002951
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003241, upper bound: 0.0003377
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003441
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003770, upper bound: 0.0003572
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003765, upper bound: 0.0003441
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003770, upper bound: 0.0003572
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003196, upper bound: 0.0003095
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003168, upper bound: 0.0003082
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003561, upper bound: 0.0003255
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003554, upper bound: 0.0003250
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004386
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004460
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004417
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004523, upper bound: 0.0004483
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004347, upper bound: 0.0004320
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004347, upper bound: 0.0004351
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004428, upper bound: 0.0004320
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004428, upper bound: 0.0004351
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004292
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004369
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004364, upper bound: 0.0004292
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004365, upper bound: 0.0004369
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003340, upper bound: 0.0003559
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003318, upper bound: 0.0003571
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003487, upper bound: 0.0003569
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003462, upper bound: 0.0003583
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002847, upper bound: 0.0002810
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003132, upper bound: 0.0003164
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0002830, upper bound: 0.0002759
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003127, upper bound: 0.0003131
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003069, upper bound: 0.0003155
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003280, upper bound: 0.0003493
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003057, upper bound: 0.0003100
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003279, upper bound: 0.0003475
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003807, upper bound: 0.0003510
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003811, upper bound: 0.0003639
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003807, upper bound: 0.0003510
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003811, upper bound: 0.0003639
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003280, upper bound: 0.0003193
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003258, upper bound: 0.0003190
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003613, upper bound: 0.0003335
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0003606, upper bound: 0.0003338
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004525, upper bound: 0.0004378
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004524, upper bound: 0.0004461
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004525, upper bound: 0.0004407
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004524, upper bound: 0.0004482
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004357, upper bound: 0.0004329
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004357, upper bound: 0.0004355
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004434, upper bound: 0.0004329
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 5, lower bound: -0.0004435, upper bound: 0.0004355

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090067, -0.0064931, -0.0088807, -0.0063358, -0.0015323, 0.0012906
1: -0.0054780, -0.0047693, -0.0054425, -0.0047250, -0.0004320, 0.0003639
2: -0.0018579, 0.0033710, -0.0015958, 0.0036981, -0.0031874, 0.0026847
3: 0.0013814, 0.0020734, 0.0014161, 0.0021167, -0.0004218, 0.0003553
4: 0.0035726, 0.0074803, 0.0033282, 0.0072845, -0.0020064, 0.0023821
5: 0.9964988, 0.9975846, 0.9964309, 0.9975301, -0.0005574, 0.0006618
6: 0.0047056, 0.0056911, 0.0046440, 0.0056417, -0.0005060, 0.0006007
7: -0.0058209, -0.0021433, -0.0060510, -0.0023276, -0.0018882, 0.0022418
8: -0.0075247, -0.0046624, -0.0073813, -0.0044834, -0.0017448, 0.0014696
9: -0.0036075, -0.0033605, -0.0036229, -0.0033729, -0.0001268, 0.0001505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004064, upper bound: 0.0004164
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004104, upper bound: 0.0004164
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090330, -0.0064637, -0.0088879, -0.0063351, -0.0015333, 0.0013595
1: -0.0054854, -0.0047610, -0.0054445, -0.0047248, -0.0004323, 0.0003833
2: -0.0019127, 0.0034320, -0.0016108, 0.0036995, -0.0031895, 0.0028280
3: 0.0013742, 0.0020815, 0.0014141, 0.0021169, -0.0004221, 0.0003742
4: 0.0035270, 0.0075213, 0.0033271, 0.0072957, -0.0021135, 0.0023836
5: 0.9964861, 0.9975958, 0.9964306, 0.9975332, -0.0005872, 0.0006622
6: 0.0046941, 0.0057014, 0.0046437, 0.0056445, -0.0005330, 0.0006011
7: -0.0058639, -0.0021048, -0.0060520, -0.0023171, -0.0019890, 0.0022432
8: -0.0075547, -0.0046290, -0.0073895, -0.0044826, -0.0017459, 0.0015481
9: -0.0036104, -0.0033580, -0.0036230, -0.0033722, -0.0001336, 0.0001506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004063, upper bound: 0.0004240
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004104, upper bound: 0.0004240
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090091, -0.0064931, -0.0088936, -0.0062947, -0.0015830, 0.0013021
1: -0.0054787, -0.0047693, -0.0054461, -0.0047134, -0.0004463, 0.0003671
2: -0.0018630, 0.0033709, -0.0016227, 0.0037836, -0.0032929, 0.0027087
3: 0.0013808, 0.0020734, 0.0014126, 0.0021280, -0.0004358, 0.0003585
4: 0.0035727, 0.0074842, 0.0032643, 0.0073046, -0.0020243, 0.0024609
5: 0.9964988, 0.9975855, 0.9964132, 0.9975357, -0.0005624, 0.0006837
6: 0.0047056, 0.0056921, 0.0046279, 0.0056468, -0.0005105, 0.0006206
7: -0.0058209, -0.0021397, -0.0061111, -0.0023087, -0.0019051, 0.0023160
8: -0.0075275, -0.0046625, -0.0073960, -0.0044366, -0.0018025, 0.0014828
9: -0.0036075, -0.0033603, -0.0036270, -0.0033716, -0.0001279, 0.0001555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004156, upper bound: 0.0004164
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004163
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090368, -0.0064637, -0.0089009, -0.0062939, -0.0015796, 0.0013712
1: -0.0054865, -0.0047610, -0.0054482, -0.0047132, -0.0004453, 0.0003866
2: -0.0019206, 0.0034321, -0.0016379, 0.0037852, -0.0032859, 0.0028523
3: 0.0013731, 0.0020815, 0.0014105, 0.0021282, -0.0004348, 0.0003775
4: 0.0035270, 0.0075272, 0.0032630, 0.0073160, -0.0021317, 0.0024557
5: 0.9964861, 0.9975975, 0.9964128, 0.9975389, -0.0005922, 0.0006823
6: 0.0046941, 0.0057029, 0.0046276, 0.0056496, -0.0005376, 0.0006193
7: -0.0058639, -0.0020992, -0.0061122, -0.0022980, -0.0020061, 0.0023111
8: -0.0075590, -0.0046290, -0.0074043, -0.0044357, -0.0017987, 0.0015614
9: -0.0036104, -0.0033576, -0.0036270, -0.0033709, -0.0001347, 0.0001552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004156, upper bound: 0.0004240
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004240
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0091345, -0.0065251, -0.0088855, -0.0063376, -0.0017035, 0.0013236
1: -0.0055140, -0.0047783, -0.0054438, -0.0047255, -0.0004803, 0.0003732
2: -0.0021237, 0.0033044, -0.0016059, 0.0036943, -0.0035436, 0.0027534
3: 0.0013463, 0.0020646, 0.0014148, 0.0021162, -0.0004689, 0.0003644
4: 0.0036224, 0.0076790, 0.0033310, 0.0072920, -0.0020577, 0.0026483
5: 0.9965127, 0.9976397, 0.9964317, 0.9975321, -0.0005717, 0.0007358
6: 0.0047182, 0.0057412, 0.0046447, 0.0056436, -0.0005189, 0.0006679
7: -0.0057741, -0.0019563, -0.0060483, -0.0023206, -0.0019365, 0.0024923
8: -0.0076702, -0.0046989, -0.0073868, -0.0044854, -0.0019398, 0.0015072
9: -0.0036043, -0.0033480, -0.0036228, -0.0033724, -0.0001300, 0.0001674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003122, upper bound: 0.0003085
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003122, upper bound: 0.0003481
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0091261, -0.0064992, -0.0088769, -0.0063390, -0.0017059, 0.0013527
1: -0.0055117, -0.0047710, -0.0054414, -0.0047259, -0.0004810, 0.0003814
2: -0.0021063, 0.0033582, -0.0015878, 0.0036915, -0.0035486, 0.0028139
3: 0.0013486, 0.0020717, 0.0014172, 0.0021158, -0.0004696, 0.0003724
4: 0.0035822, 0.0076660, 0.0033331, 0.0072785, -0.0021029, 0.0026520
5: 0.9965015, 0.9976361, 0.9964322, 0.9975284, -0.0005843, 0.0007368
6: 0.0047080, 0.0057379, 0.0046452, 0.0056402, -0.0005303, 0.0006688
7: -0.0058119, -0.0019686, -0.0060463, -0.0023333, -0.0019791, 0.0024958
8: -0.0076607, -0.0046694, -0.0073769, -0.0044870, -0.0019425, 0.0015403
9: -0.0036069, -0.0033488, -0.0036226, -0.0033733, -0.0001329, 0.0001676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003078, upper bound: 0.0003093
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003078, upper bound: 0.0003491
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0091354, -0.0065251, -0.0088989, -0.0062966, -0.0017460, 0.0013353
1: -0.0055143, -0.0047783, -0.0054476, -0.0047139, -0.0004923, 0.0003765
2: -0.0021257, 0.0033044, -0.0016338, 0.0037797, -0.0036320, 0.0027778
3: 0.0013460, 0.0020646, 0.0014111, 0.0021275, -0.0004806, 0.0003676
4: 0.0036224, 0.0076805, 0.0032671, 0.0073128, -0.0020759, 0.0027143
5: 0.9965127, 0.9976401, 0.9964139, 0.9975380, -0.0005768, 0.0007541
6: 0.0047182, 0.0057416, 0.0046286, 0.0056489, -0.0005235, 0.0006845
7: -0.0057741, -0.0019549, -0.0061084, -0.0023009, -0.0019537, 0.0025545
8: -0.0076713, -0.0046989, -0.0074020, -0.0044387, -0.0019881, 0.0015206
9: -0.0036043, -0.0033479, -0.0036268, -0.0033711, -0.0001312, 0.0001715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003268, upper bound: 0.0003116
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003268, upper bound: 0.0003488
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0091270, -0.0064992, -0.0088890, -0.0062980, -0.0017481, 0.0013639
1: -0.0055119, -0.0047710, -0.0054448, -0.0047143, -0.0004929, 0.0003845
2: -0.0021081, 0.0033582, -0.0016131, 0.0037767, -0.0036365, 0.0028372
3: 0.0013483, 0.0020717, 0.0014138, 0.0021271, -0.0004812, 0.0003755
4: 0.0035822, 0.0076673, 0.0032694, 0.0072974, -0.0021204, 0.0027177
5: 0.9965015, 0.9976364, 0.9964145, 0.9975336, -0.0005891, 0.0007550
6: 0.0047080, 0.0057383, 0.0046292, 0.0056450, -0.0005347, 0.0006854
7: -0.0058119, -0.0019674, -0.0061063, -0.0023155, -0.0019955, 0.0025576
8: -0.0076617, -0.0046694, -0.0073907, -0.0044403, -0.0019906, 0.0015531
9: -0.0036069, -0.0033487, -0.0036266, -0.0033721, -0.0001340, 0.0001717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003124
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003501
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090036, -0.0064221, -0.0090092, -0.0063901, -0.0015056, 0.0015769
1: -0.0054771, -0.0047493, -0.0054787, -0.0047403, -0.0004245, 0.0004446
2: -0.0018515, 0.0035186, -0.0018632, 0.0035851, -0.0031319, 0.0032804
3: 0.0013823, 0.0020929, 0.0013807, 0.0021017, -0.0004145, 0.0004341
4: 0.0034623, 0.0074756, 0.0034126, 0.0074843, -0.0024515, 0.0023406
5: 0.9964682, 0.9975832, 0.9964544, 0.9975856, -0.0006811, 0.0006503
6: 0.0046778, 0.0056899, 0.0046653, 0.0056921, -0.0006182, 0.0005903
7: -0.0059247, -0.0021478, -0.0059715, -0.0021396, -0.0023072, 0.0022027
8: -0.0075212, -0.0045816, -0.0075276, -0.0045452, -0.0017144, 0.0017957
9: -0.0036145, -0.0033608, -0.0036176, -0.0033603, -0.0001549, 0.0001479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002659, upper bound: 0.0002690
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002659, upper bound: 0.0003406
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089942, -0.0064234, -0.0089979, -0.0063658, -0.0015517, 0.0015795
1: -0.0054745, -0.0047497, -0.0054755, -0.0047334, -0.0004375, 0.0004453
2: -0.0018319, 0.0035159, -0.0018396, 0.0036356, -0.0032278, 0.0032856
3: 0.0013849, 0.0020926, 0.0013839, 0.0021084, -0.0004272, 0.0004348
4: 0.0034643, 0.0074609, 0.0033748, 0.0074667, -0.0024554, 0.0024123
5: 0.9964687, 0.9975791, 0.9964438, 0.9975807, -0.0006822, 0.0006702
6: 0.0046783, 0.0056862, 0.0046558, 0.0056877, -0.0006192, 0.0006083
7: -0.0059228, -0.0021616, -0.0060070, -0.0021562, -0.0023109, 0.0022702
8: -0.0075105, -0.0045831, -0.0075147, -0.0045176, -0.0017669, 0.0017985
9: -0.0036143, -0.0033618, -0.0036200, -0.0033614, -0.0001552, 0.0001524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002658, upper bound: 0.0002658
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002658, upper bound: 0.0003377
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0090640, -0.0064608, -0.0013687, 0.0016624
1: -0.0054470, -0.0047193, -0.0054941, -0.0047602, -0.0003859, 0.0004687
2: -0.0016293, 0.0037397, -0.0019770, 0.0034380, -0.0028472, 0.0034581
3: 0.0014117, 0.0021222, 0.0013657, 0.0020823, -0.0003768, 0.0004576
4: 0.0032970, 0.0073095, 0.0035225, 0.0075694, -0.0025844, 0.0021278
5: 0.9964223, 0.9975371, 0.9964849, 0.9976092, -0.0007180, 0.0005912
6: 0.0046361, 0.0056480, 0.0046930, 0.0057136, -0.0006517, 0.0005366
7: -0.0060803, -0.0023041, -0.0058680, -0.0020595, -0.0024322, 0.0020025
8: -0.0073996, -0.0044606, -0.0075899, -0.0046258, -0.0015586, 0.0018930
9: -0.0036249, -0.0033713, -0.0036106, -0.0033549, -0.0001633, 0.0001345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004286, upper bound: 0.0004233
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004366, upper bound: 0.0004234
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0090669, -0.0064608, -0.0013802, 0.0017102
1: -0.0054506, -0.0047089, -0.0054950, -0.0047602, -0.0003891, 0.0004822
2: -0.0016560, 0.0038166, -0.0019831, 0.0034380, -0.0028711, 0.0035575
3: 0.0014081, 0.0021324, 0.0013649, 0.0020823, -0.0003799, 0.0004708
4: 0.0032396, 0.0073295, 0.0035225, 0.0075739, -0.0026587, 0.0021457
5: 0.9964063, 0.9975426, 0.9964849, 0.9976104, -0.0007387, 0.0005961
6: 0.0046217, 0.0056531, 0.0046930, 0.0057147, -0.0006705, 0.0005411
7: -0.0061343, -0.0022853, -0.0058680, -0.0020552, -0.0025021, 0.0020193
8: -0.0074142, -0.0044185, -0.0075933, -0.0046258, -0.0015716, 0.0019474
9: -0.0036285, -0.0033701, -0.0036106, -0.0033546, -0.0001680, 0.0001356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004287, upper bound: 0.0004308
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004365, upper bound: 0.0004308
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0091874, -0.0065092, -0.0013542, 0.0018249
1: -0.0054470, -0.0047193, -0.0055289, -0.0047738, -0.0003818, 0.0005145
2: -0.0016293, 0.0037397, -0.0022339, 0.0033374, -0.0028170, 0.0037961
3: 0.0014117, 0.0021222, 0.0013317, 0.0020689, -0.0003728, 0.0005024
4: 0.0032970, 0.0073095, 0.0035977, 0.0077613, -0.0028370, 0.0021052
5: 0.9964223, 0.9975371, 0.9965058, 0.9976625, -0.0007882, 0.0005849
6: 0.0046361, 0.0056480, 0.0047120, 0.0057620, -0.0007154, 0.0005309
7: -0.0060803, -0.0023041, -0.0057973, -0.0018789, -0.0026699, 0.0019813
8: -0.0073996, -0.0044606, -0.0077305, -0.0046808, -0.0015420, 0.0020780
9: -0.0036249, -0.0033713, -0.0036059, -0.0033428, -0.0001793, 0.0001330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003481, upper bound: 0.0003261
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003491, upper bound: 0.0003231
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0091882, -0.0065092, -0.0013657, 0.0018626
1: -0.0054506, -0.0047089, -0.0055292, -0.0047738, -0.0003850, 0.0005251
2: -0.0016560, 0.0038166, -0.0022354, 0.0033374, -0.0028409, 0.0038745
3: 0.0014081, 0.0021324, 0.0013315, 0.0020690, -0.0003759, 0.0005127
4: 0.0032396, 0.0073295, 0.0035977, 0.0077625, -0.0028956, 0.0021231
5: 0.9964063, 0.9975426, 0.9965058, 0.9976628, -0.0008045, 0.0005899
6: 0.0046217, 0.0056531, 0.0047119, 0.0057623, -0.0007302, 0.0005354
7: -0.0061343, -0.0022853, -0.0057973, -0.0018778, -0.0027251, 0.0019981
8: -0.0074142, -0.0044185, -0.0077314, -0.0046808, -0.0015551, 0.0021209
9: -0.0036285, -0.0033701, -0.0036059, -0.0033427, -0.0001830, 0.0001342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003488, upper bound: 0.0003408
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003501, upper bound: 0.0003391
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090180, -0.0063704, -0.0090329, -0.0064192, -0.0015869, 0.0016325
1: -0.0054812, -0.0047347, -0.0054854, -0.0047485, -0.0004474, 0.0004603
2: -0.0018815, 0.0036260, -0.0019124, 0.0035247, -0.0033011, 0.0033960
3: 0.0013783, 0.0021071, 0.0013742, 0.0020937, -0.0004369, 0.0004494
4: 0.0033820, 0.0074980, 0.0034577, 0.0075211, -0.0025380, 0.0024671
5: 0.9964458, 0.9975893, 0.9964669, 0.9975958, -0.0007051, 0.0006854
6: 0.0046576, 0.0056956, 0.0046767, 0.0057014, -0.0006400, 0.0006222
7: -0.0060003, -0.0021267, -0.0059290, -0.0021050, -0.0023885, 0.0023218
8: -0.0075377, -0.0045228, -0.0075546, -0.0045783, -0.0018070, 0.0018590
9: -0.0036195, -0.0033594, -0.0036147, -0.0033580, -0.0001604, 0.0001559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003008, upper bound: 0.0002937
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003406, upper bound: 0.0003183
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090066, -0.0063443, -0.0090231, -0.0064205, -0.0015891, 0.0016734
1: -0.0054779, -0.0047274, -0.0054826, -0.0047488, -0.0004480, 0.0004718
2: -0.0018576, 0.0036804, -0.0018920, 0.0035219, -0.0033057, 0.0034811
3: 0.0013815, 0.0021143, 0.0013769, 0.0020934, -0.0004375, 0.0004607
4: 0.0033414, 0.0074801, 0.0034598, 0.0075058, -0.0026015, 0.0024705
5: 0.9964346, 0.9975845, 0.9964675, 0.9975916, -0.0007228, 0.0006864
6: 0.0046473, 0.0056910, 0.0046772, 0.0056975, -0.0006561, 0.0006230
7: -0.0060385, -0.0021435, -0.0059271, -0.0021194, -0.0024483, 0.0023250
8: -0.0075246, -0.0044931, -0.0075434, -0.0045798, -0.0018095, 0.0019055
9: -0.0036221, -0.0033606, -0.0036146, -0.0033589, -0.0001644, 0.0001561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002943, upper bound: 0.0002926
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003377, upper bound: 0.0003181
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0089361, -0.0063148, -0.0013977, 0.0014379
1: -0.0054470, -0.0047193, -0.0054581, -0.0047190, -0.0003941, 0.0004054
2: -0.0016293, 0.0037397, -0.0017110, 0.0037417, -0.0029075, 0.0029911
3: 0.0014117, 0.0021222, 0.0014009, 0.0021225, -0.0003848, 0.0003958
4: 0.0032970, 0.0073095, 0.0032955, 0.0073706, -0.0022353, 0.0021729
5: 0.9964223, 0.9975371, 0.9964219, 0.9975540, -0.0006210, 0.0006037
6: 0.0046361, 0.0056480, 0.0046358, 0.0056634, -0.0005637, 0.0005480
7: -0.0060803, -0.0023041, -0.0060817, -0.0022466, -0.0021037, 0.0020449
8: -0.0073996, -0.0044606, -0.0074443, -0.0044595, -0.0015916, 0.0016373
9: -0.0036249, -0.0033713, -0.0036250, -0.0033675, -0.0001413, 0.0001373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004370, upper bound: 0.0004300
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004398, upper bound: 0.0004300
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0089370, -0.0063148, -0.0014105, 0.0014946
1: -0.0054506, -0.0047089, -0.0054583, -0.0047190, -0.0003977, 0.0004214
2: -0.0016560, 0.0038166, -0.0017129, 0.0037417, -0.0029341, 0.0031090
3: 0.0014081, 0.0021324, 0.0014006, 0.0021224, -0.0003883, 0.0004114
4: 0.0032396, 0.0073295, 0.0032955, 0.0073720, -0.0023235, 0.0021927
5: 0.9964063, 0.9975426, 0.9964218, 0.9975544, -0.0006455, 0.0006092
6: 0.0046217, 0.0056531, 0.0046358, 0.0056638, -0.0005859, 0.0005530
7: -0.0061343, -0.0022853, -0.0060817, -0.0022453, -0.0021866, 0.0020636
8: -0.0074142, -0.0044185, -0.0074454, -0.0044595, -0.0016061, 0.0017019
9: -0.0036285, -0.0033701, -0.0036250, -0.0033674, -0.0001468, 0.0001386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004371, upper bound: 0.0004370
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004399, upper bound: 0.0004371
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0090705, -0.0063552, -0.0013881, 0.0016109
1: -0.0054470, -0.0047193, -0.0054960, -0.0047304, -0.0003914, 0.0004542
2: -0.0016293, 0.0037397, -0.0019906, 0.0036577, -0.0028875, 0.0033511
3: 0.0014117, 0.0021222, 0.0013639, 0.0021113, -0.0003821, 0.0004435
4: 0.0032970, 0.0073095, 0.0033583, 0.0075795, -0.0025044, 0.0021580
5: 0.9964223, 0.9975371, 0.9964393, 0.9976120, -0.0006958, 0.0005995
6: 0.0046361, 0.0056480, 0.0046516, 0.0057161, -0.0006316, 0.0005442
7: -0.0060803, -0.0023041, -0.0060226, -0.0020500, -0.0023569, 0.0020309
8: -0.0073996, -0.0044606, -0.0075974, -0.0045055, -0.0015806, 0.0018344
9: -0.0036249, -0.0033713, -0.0036210, -0.0033543, -0.0001583, 0.0001364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004311, upper bound: 0.0004288
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004340, upper bound: 0.0004288
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0090709, -0.0063552, -0.0014009, 0.0016568
1: -0.0054506, -0.0047089, -0.0054961, -0.0047304, -0.0003950, 0.0004671
2: -0.0016560, 0.0038166, -0.0019915, 0.0036577, -0.0029142, 0.0034464
3: 0.0014081, 0.0021324, 0.0013637, 0.0021113, -0.0003856, 0.0004561
4: 0.0032396, 0.0073295, 0.0033583, 0.0075802, -0.0025757, 0.0021779
5: 0.9964063, 0.9975426, 0.9964393, 0.9976123, -0.0007156, 0.0006051
6: 0.0046217, 0.0056531, 0.0046516, 0.0057163, -0.0006495, 0.0005492
7: -0.0061343, -0.0022853, -0.0060226, -0.0020493, -0.0024240, 0.0020496
8: -0.0074142, -0.0044185, -0.0075979, -0.0045055, -0.0015952, 0.0018866
9: -0.0036285, -0.0033701, -0.0036210, -0.0033542, -0.0001628, 0.0001376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004311, upper bound: 0.0004351
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004340, upper bound: 0.0004351
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090179, -0.0063704, -0.0088866, -0.0063042, -0.0015677, 0.0014053
1: -0.0054811, -0.0047347, -0.0054441, -0.0047161, -0.0004420, 0.0003962
2: -0.0018811, 0.0036261, -0.0016080, 0.0037638, -0.0032610, 0.0029233
3: 0.0013784, 0.0021071, 0.0014145, 0.0021254, -0.0004315, 0.0003869
4: 0.0033820, 0.0074977, 0.0032790, 0.0072936, -0.0021847, 0.0024371
5: 0.9964458, 0.9975892, 0.9964173, 0.9975326, -0.0006070, 0.0006771
6: 0.0046576, 0.0056955, 0.0046316, 0.0056440, -0.0005509, 0.0006146
7: -0.0060003, -0.0021270, -0.0060972, -0.0023190, -0.0020560, 0.0022936
8: -0.0075375, -0.0045228, -0.0073880, -0.0044474, -0.0017851, 0.0016002
9: -0.0036195, -0.0033594, -0.0036260, -0.0033723, -0.0001381, 0.0001540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004198, upper bound: 0.0004248
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004248
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090061, -0.0063443, -0.0088779, -0.0063055, -0.0015675, 0.0014459
1: -0.0054778, -0.0047274, -0.0054417, -0.0047164, -0.0004419, 0.0004076
2: -0.0018566, 0.0036804, -0.0015901, 0.0037611, -0.0032607, 0.0030077
3: 0.0013816, 0.0021143, 0.0014169, 0.0021250, -0.0004315, 0.0003980
4: 0.0033414, 0.0074794, 0.0032811, 0.0072802, -0.0022477, 0.0024368
5: 0.9964346, 0.9975842, 0.9964179, 0.9975289, -0.0006245, 0.0006770
6: 0.0046473, 0.0056909, 0.0046321, 0.0056406, -0.0005668, 0.0006145
7: -0.0060385, -0.0021442, -0.0060953, -0.0023317, -0.0021154, 0.0022933
8: -0.0075240, -0.0044931, -0.0073781, -0.0044489, -0.0017849, 0.0016464
9: -0.0036221, -0.0033606, -0.0036259, -0.0033732, -0.0001420, 0.0001540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004198, upper bound: 0.0004277
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004277
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090180, -0.0063704, -0.0089000, -0.0062716, -0.0016019, 0.0014171
1: -0.0054812, -0.0047347, -0.0054479, -0.0047069, -0.0004516, 0.0003995
2: -0.0018815, 0.0036260, -0.0016360, 0.0038316, -0.0033322, 0.0029479
3: 0.0013783, 0.0021071, 0.0014108, 0.0021343, -0.0004410, 0.0003901
4: 0.0033820, 0.0074980, 0.0032284, 0.0073145, -0.0022031, 0.0024903
5: 0.9964458, 0.9975893, 0.9964032, 0.9975384, -0.0006121, 0.0006919
6: 0.0046576, 0.0056956, 0.0046188, 0.0056493, -0.0005556, 0.0006280
7: -0.0060003, -0.0021267, -0.0061449, -0.0022993, -0.0020733, 0.0023436
8: -0.0075377, -0.0045228, -0.0074033, -0.0044103, -0.0018240, 0.0016137
9: -0.0036195, -0.0033594, -0.0036292, -0.0033710, -0.0001392, 0.0001574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004241, upper bound: 0.0004247
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004320, upper bound: 0.0004248
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090066, -0.0063443, -0.0088901, -0.0062730, -0.0016017, 0.0014583
1: -0.0054779, -0.0047274, -0.0054451, -0.0047072, -0.0004516, 0.0004112
2: -0.0018576, 0.0036804, -0.0016153, 0.0038288, -0.0033319, 0.0030336
3: 0.0013815, 0.0021143, 0.0014135, 0.0021340, -0.0004409, 0.0004015
4: 0.0033414, 0.0074801, 0.0032305, 0.0072990, -0.0022672, 0.0024901
5: 0.9964346, 0.9975845, 0.9964037, 0.9975342, -0.0006299, 0.0006918
6: 0.0046473, 0.0056910, 0.0046193, 0.0056454, -0.0005717, 0.0006280
7: -0.0060385, -0.0021435, -0.0061429, -0.0023139, -0.0021336, 0.0023434
8: -0.0075246, -0.0044931, -0.0073919, -0.0044118, -0.0018239, 0.0016606
9: -0.0036221, -0.0033606, -0.0036291, -0.0033720, -0.0001433, 0.0001574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004240, upper bound: 0.0004277
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004320, upper bound: 0.0004277
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090067, -0.0064931, -0.0088435, -0.0062597, -0.0016771, 0.0013291
1: -0.0054780, -0.0047693, -0.0054320, -0.0047035, -0.0004728, 0.0003747
2: -0.0018579, 0.0033710, -0.0015184, 0.0038563, -0.0034888, 0.0027648
3: 0.0013814, 0.0020734, 0.0014264, 0.0021376, -0.0004617, 0.0003659
4: 0.0035726, 0.0074803, 0.0032099, 0.0072266, -0.0020662, 0.0026073
5: 0.9964988, 0.9975846, 0.9963980, 0.9975140, -0.0005741, 0.0007244
6: 0.0047056, 0.0056911, 0.0046142, 0.0056271, -0.0005211, 0.0006575
7: -0.0058209, -0.0021433, -0.0061623, -0.0023821, -0.0019446, 0.0024537
8: -0.0075247, -0.0046624, -0.0073389, -0.0043968, -0.0019098, 0.0015135
9: -0.0036075, -0.0033605, -0.0036304, -0.0033766, -0.0001306, 0.0001648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004063, upper bound: 0.0004211
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004104, upper bound: 0.0004211
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090330, -0.0064637, -0.0088511, -0.0062591, -0.0016811, 0.0013935
1: -0.0054854, -0.0047610, -0.0054341, -0.0047033, -0.0004740, 0.0003929
2: -0.0019127, 0.0034320, -0.0015342, 0.0038577, -0.0034971, 0.0028989
3: 0.0013742, 0.0020815, 0.0014243, 0.0021378, -0.0004628, 0.0003836
4: 0.0035270, 0.0075213, 0.0032089, 0.0072384, -0.0021664, 0.0026135
5: 0.9964861, 0.9975958, 0.9963977, 0.9975173, -0.0006019, 0.0007261
6: 0.0046941, 0.0057014, 0.0046139, 0.0056301, -0.0005463, 0.0006591
7: -0.0058639, -0.0021048, -0.0061632, -0.0023710, -0.0020389, 0.0024596
8: -0.0075547, -0.0046290, -0.0073475, -0.0043960, -0.0019143, 0.0015868
9: -0.0036104, -0.0033580, -0.0036305, -0.0033758, -0.0001369, 0.0001652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004063, upper bound: 0.0004288
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004104, upper bound: 0.0004287
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090091, -0.0064931, -0.0088622, -0.0062124, -0.0017388, 0.0013423
1: -0.0054787, -0.0047693, -0.0054372, -0.0046902, -0.0004902, 0.0003784
2: -0.0018630, 0.0033709, -0.0015573, 0.0039549, -0.0036171, 0.0027922
3: 0.0013808, 0.0020734, 0.0014212, 0.0021507, -0.0004787, 0.0003695
4: 0.0035727, 0.0074842, 0.0031363, 0.0072557, -0.0020867, 0.0027032
5: 0.9964988, 0.9975855, 0.9963776, 0.9975221, -0.0005797, 0.0007510
6: 0.0047056, 0.0056921, 0.0045956, 0.0056345, -0.0005262, 0.0006817
7: -0.0058209, -0.0021397, -0.0062316, -0.0023547, -0.0019638, 0.0025440
8: -0.0075275, -0.0046625, -0.0073602, -0.0043428, -0.0019800, 0.0015284
9: -0.0036075, -0.0033603, -0.0036351, -0.0033747, -0.0001319, 0.0001708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004155, upper bound: 0.0004211
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004211
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090368, -0.0064637, -0.0088703, -0.0062116, -0.0017425, 0.0014074
1: -0.0054865, -0.0047610, -0.0054395, -0.0046899, -0.0004913, 0.0003968
2: -0.0019206, 0.0034321, -0.0015742, 0.0039564, -0.0036248, 0.0029277
3: 0.0013731, 0.0020815, 0.0014190, 0.0021509, -0.0004797, 0.0003874
4: 0.0035270, 0.0075272, 0.0031351, 0.0072683, -0.0021880, 0.0027089
5: 0.9964861, 0.9975975, 0.9963773, 0.9975256, -0.0006079, 0.0007526
6: 0.0046941, 0.0057029, 0.0045953, 0.0056376, -0.0005518, 0.0006831
7: -0.0058639, -0.0020992, -0.0062327, -0.0023429, -0.0020591, 0.0025494
8: -0.0075590, -0.0046290, -0.0073694, -0.0043420, -0.0019842, 0.0016026
9: -0.0036104, -0.0033576, -0.0036351, -0.0033739, -0.0001383, 0.0001712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004156, upper bound: 0.0004288
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004288
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0091345, -0.0065251, -0.0088493, -0.0062618, -0.0018482, 0.0013611
1: -0.0055140, -0.0047783, -0.0054336, -0.0047041, -0.0005211, 0.0003838
2: -0.0021237, 0.0033044, -0.0015305, 0.0038521, -0.0038446, 0.0028315
3: 0.0013463, 0.0020646, 0.0014248, 0.0021371, -0.0005088, 0.0003747
4: 0.0036224, 0.0076790, 0.0032131, 0.0072357, -0.0021161, 0.0028732
5: 0.9965127, 0.9976397, 0.9963989, 0.9975165, -0.0005879, 0.0007983
6: 0.0047182, 0.0057412, 0.0046150, 0.0056294, -0.0005336, 0.0007246
7: -0.0057741, -0.0019563, -0.0061593, -0.0023735, -0.0019914, 0.0027040
8: -0.0076702, -0.0046989, -0.0073455, -0.0043991, -0.0021046, 0.0015499
9: -0.0036043, -0.0033480, -0.0036302, -0.0033760, -0.0001337, 0.0001816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003122, upper bound: 0.0003160
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003122, upper bound: 0.0003596
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0091261, -0.0064992, -0.0088382, -0.0062633, -0.0018506, 0.0013877
1: -0.0055117, -0.0047710, -0.0054305, -0.0047045, -0.0005218, 0.0003913
2: -0.0021063, 0.0033582, -0.0015074, 0.0038490, -0.0038497, 0.0028868
3: 0.0013486, 0.0020717, 0.0014278, 0.0021366, -0.0005094, 0.0003820
4: 0.0035822, 0.0076660, 0.0032154, 0.0072184, -0.0021574, 0.0028770
5: 0.9965015, 0.9976361, 0.9963996, 0.9975117, -0.0005994, 0.0007993
6: 0.0047080, 0.0057379, 0.0046155, 0.0056250, -0.0005441, 0.0007255
7: -0.0058119, -0.0019686, -0.0061571, -0.0023898, -0.0020303, 0.0027076
8: -0.0076607, -0.0046694, -0.0073329, -0.0044008, -0.0021073, 0.0015802
9: -0.0036069, -0.0033488, -0.0036301, -0.0033771, -0.0001363, 0.0001818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003163
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003082, upper bound: 0.0003595
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0091354, -0.0065251, -0.0088684, -0.0062143, -0.0019019, 0.0013750
1: -0.0055143, -0.0047783, -0.0054390, -0.0046907, -0.0005362, 0.0003877
2: -0.0021257, 0.0033044, -0.0015702, 0.0039508, -0.0039563, 0.0028604
3: 0.0013460, 0.0020646, 0.0014195, 0.0021501, -0.0005236, 0.0003785
4: 0.0036224, 0.0076805, 0.0031393, 0.0072653, -0.0021377, 0.0029567
5: 0.9965127, 0.9976401, 0.9963784, 0.9975248, -0.0005939, 0.0008215
6: 0.0047182, 0.0057416, 0.0045964, 0.0056369, -0.0005391, 0.0007456
7: -0.0057741, -0.0019549, -0.0062287, -0.0023456, -0.0020118, 0.0027826
8: -0.0076713, -0.0046989, -0.0073672, -0.0043451, -0.0021657, 0.0015658
9: -0.0036043, -0.0033479, -0.0036349, -0.0033741, -0.0001351, 0.0001868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003269, upper bound: 0.0003195
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003269, upper bound: 0.0003603
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0091270, -0.0064992, -0.0088572, -0.0062159, -0.0019042, 0.0014011
1: -0.0055119, -0.0047710, -0.0054358, -0.0046911, -0.0005369, 0.0003950
2: -0.0021081, 0.0033582, -0.0015469, 0.0039476, -0.0039611, 0.0029145
3: 0.0013483, 0.0020717, 0.0014226, 0.0021497, -0.0005242, 0.0003857
4: 0.0035822, 0.0076673, 0.0031417, 0.0072479, -0.0021781, 0.0029603
5: 0.9965015, 0.9976364, 0.9963791, 0.9975200, -0.0006051, 0.0008225
6: 0.0047080, 0.0057383, 0.0045970, 0.0056325, -0.0005493, 0.0007465
7: -0.0058119, -0.0019674, -0.0062265, -0.0023620, -0.0020499, 0.0027859
8: -0.0076617, -0.0046694, -0.0073545, -0.0043468, -0.0021683, 0.0015954
9: -0.0036069, -0.0033487, -0.0036347, -0.0033752, -0.0001376, 0.0001871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003198
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003244, upper bound: 0.0003601
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090124, -0.0064652, -0.0089807, -0.0063224, -0.0016502, 0.0015503
1: -0.0054796, -0.0047615, -0.0054707, -0.0047212, -0.0004653, 0.0004371
2: -0.0018697, 0.0034288, -0.0018038, 0.0037259, -0.0034328, 0.0032249
3: 0.0013799, 0.0020810, 0.0013886, 0.0021204, -0.0004543, 0.0004268
4: 0.0035294, 0.0074892, 0.0033074, 0.0074399, -0.0024101, 0.0025655
5: 0.9964868, 0.9975870, 0.9964252, 0.9975732, -0.0006696, 0.0007128
6: 0.0046947, 0.0056933, 0.0046387, 0.0056809, -0.0006078, 0.0006470
7: -0.0058616, -0.0021350, -0.0060705, -0.0021813, -0.0022681, 0.0024144
8: -0.0075312, -0.0046308, -0.0074951, -0.0044682, -0.0018791, 0.0017653
9: -0.0036102, -0.0033600, -0.0036242, -0.0033631, -0.0001523, 0.0001621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002521, upper bound: 0.0002390
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002521, upper bound: 0.0003196
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0091367, -0.0065135, -0.0089807, -0.0063224, -0.0017239, 0.0014371
1: -0.0055146, -0.0047751, -0.0054707, -0.0047212, -0.0004860, 0.0004052
2: -0.0021283, 0.0033284, -0.0018038, 0.0037259, -0.0035861, 0.0029895
3: 0.0013456, 0.0020678, 0.0013886, 0.0021204, -0.0004746, 0.0003956
4: 0.0036044, 0.0076824, 0.0033074, 0.0074399, -0.0022342, 0.0026800
5: 0.9965077, 0.9976407, 0.9964252, 0.9975732, -0.0006207, 0.0007446
6: 0.0047136, 0.0057421, 0.0046387, 0.0056809, -0.0005634, 0.0006759
7: -0.0057910, -0.0019531, -0.0060705, -0.0021813, -0.0021026, 0.0025222
8: -0.0076728, -0.0046857, -0.0074951, -0.0044682, -0.0019630, 0.0016365
9: -0.0036055, -0.0033478, -0.0036242, -0.0033631, -0.0001412, 0.0001694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002521, upper bound: 0.0002390
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002521, upper bound: 0.0003196
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090046, -0.0064666, -0.0089639, -0.0063010, -0.0016822, 0.0015538
1: -0.0054774, -0.0047618, -0.0054659, -0.0047152, -0.0004743, 0.0004381
2: -0.0018536, 0.0034260, -0.0017688, 0.0037704, -0.0034993, 0.0032322
3: 0.0013820, 0.0020807, 0.0013932, 0.0021263, -0.0004631, 0.0004277
4: 0.0035315, 0.0074771, 0.0032741, 0.0074138, -0.0024155, 0.0026152
5: 0.9964874, 0.9975836, 0.9964159, 0.9975660, -0.0006711, 0.0007266
6: 0.0046953, 0.0056903, 0.0046303, 0.0056743, -0.0006092, 0.0006595
7: -0.0058596, -0.0021464, -0.0061019, -0.0022060, -0.0022733, 0.0024612
8: -0.0075224, -0.0046323, -0.0074760, -0.0044438, -0.0019155, 0.0017693
9: -0.0036101, -0.0033607, -0.0036263, -0.0033647, -0.0001526, 0.0001653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002512, upper bound: 0.0002337
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002512, upper bound: 0.0002337
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0091076, -0.0065149, -0.0089639, -0.0063010, -0.0017344, 0.0014426
1: -0.0055064, -0.0047755, -0.0054659, -0.0047152, -0.0004890, 0.0004067
2: -0.0020679, 0.0033255, -0.0017688, 0.0037704, -0.0036078, 0.0030009
3: 0.0013536, 0.0020674, 0.0013932, 0.0021263, -0.0004774, 0.0003971
4: 0.0036066, 0.0076373, 0.0032741, 0.0074138, -0.0022427, 0.0026963
5: 0.9965083, 0.9976282, 0.9964159, 0.9975660, -0.0006231, 0.0007491
6: 0.0047142, 0.0057307, 0.0046303, 0.0056743, -0.0005656, 0.0006800
7: -0.0057890, -0.0019956, -0.0061019, -0.0022060, -0.0021106, 0.0025375
8: -0.0076397, -0.0046873, -0.0074760, -0.0044438, -0.0019749, 0.0016427
9: -0.0036053, -0.0033506, -0.0036263, -0.0033647, -0.0001417, 0.0001704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002512, upper bound: 0.0002337
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002512, upper bound: 0.0003168
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090036, -0.0064221, -0.0089735, -0.0063231, -0.0016561, 0.0015971
1: -0.0054771, -0.0047493, -0.0054686, -0.0047214, -0.0004669, 0.0004503
2: -0.0018515, 0.0035186, -0.0017888, 0.0037245, -0.0034450, 0.0033223
3: 0.0013823, 0.0020929, 0.0013906, 0.0021202, -0.0004559, 0.0004397
4: 0.0034623, 0.0074756, 0.0033084, 0.0074287, -0.0024829, 0.0025746
5: 0.9964682, 0.9975832, 0.9964254, 0.9975702, -0.0006898, 0.0007153
6: 0.0046778, 0.0056899, 0.0046390, 0.0056781, -0.0006262, 0.0006493
7: -0.0059247, -0.0021478, -0.0060695, -0.0021919, -0.0023367, 0.0024229
8: -0.0075212, -0.0045816, -0.0074869, -0.0044689, -0.0018858, 0.0018186
9: -0.0036145, -0.0033608, -0.0036242, -0.0033638, -0.0001569, 0.0001627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002672, upper bound: 0.0002747
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002672, upper bound: 0.0003493
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089942, -0.0064234, -0.0089558, -0.0063017, -0.0016893, 0.0016001
1: -0.0054745, -0.0047497, -0.0054636, -0.0047153, -0.0004763, 0.0004511
2: -0.0018319, 0.0035159, -0.0017520, 0.0037690, -0.0035142, 0.0033284
3: 0.0013849, 0.0020926, 0.0013954, 0.0021261, -0.0004650, 0.0004405
4: 0.0034643, 0.0074609, 0.0032751, 0.0074012, -0.0024875, 0.0026263
5: 0.9964687, 0.9975791, 0.9964162, 0.9975626, -0.0006911, 0.0007297
6: 0.0046783, 0.0056862, 0.0046306, 0.0056711, -0.0006273, 0.0006623
7: -0.0059228, -0.0021616, -0.0061009, -0.0022178, -0.0023410, 0.0024716
8: -0.0075105, -0.0045831, -0.0074668, -0.0044446, -0.0019237, 0.0018220
9: -0.0036143, -0.0033618, -0.0036263, -0.0033655, -0.0001572, 0.0001660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002669, upper bound: 0.0002715
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002669, upper bound: 0.0003486
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0090327, -0.0063834, -0.0015249, 0.0017315
1: -0.0054470, -0.0047193, -0.0054853, -0.0047384, -0.0004299, 0.0004882
2: -0.0016293, 0.0037397, -0.0019121, 0.0035992, -0.0031720, 0.0036019
3: 0.0014117, 0.0021222, 0.0013743, 0.0021036, -0.0004198, 0.0004767
4: 0.0032970, 0.0073095, 0.0034021, 0.0075208, -0.0026918, 0.0023706
5: 0.9964223, 0.9975371, 0.9964514, 0.9975958, -0.0007479, 0.0006586
6: 0.0046361, 0.0056480, 0.0046626, 0.0057013, -0.0006788, 0.0005978
7: -0.0060803, -0.0023041, -0.0059814, -0.0021052, -0.0025333, 0.0022310
8: -0.0073996, -0.0044606, -0.0075544, -0.0045375, -0.0017364, 0.0019717
9: -0.0036249, -0.0033713, -0.0036183, -0.0033580, -0.0001701, 0.0001498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004286, upper bound: 0.0004249
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004364, upper bound: 0.0004248
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0090371, -0.0063834, -0.0015363, 0.0017810
1: -0.0054506, -0.0047089, -0.0054866, -0.0047384, -0.0004331, 0.0005021
2: -0.0016560, 0.0038166, -0.0019211, 0.0035991, -0.0031958, 0.0037048
3: 0.0014081, 0.0021324, 0.0013731, 0.0021036, -0.0004229, 0.0004903
4: 0.0032396, 0.0073295, 0.0034021, 0.0075276, -0.0027688, 0.0023884
5: 0.9964063, 0.9975426, 0.9964515, 0.9975976, -0.0007692, 0.0006636
6: 0.0046217, 0.0056531, 0.0046626, 0.0057030, -0.0006982, 0.0006023
7: -0.0061343, -0.0022853, -0.0059813, -0.0020988, -0.0026057, 0.0022477
8: -0.0074142, -0.0044185, -0.0075593, -0.0045376, -0.0017494, 0.0020280
9: -0.0036285, -0.0033701, -0.0036183, -0.0033576, -0.0001750, 0.0001509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004287, upper bound: 0.0004362
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004364, upper bound: 0.0004362
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0091481, -0.0064436, -0.0014744, 0.0018636
1: -0.0054470, -0.0047193, -0.0055179, -0.0047554, -0.0004157, 0.0005254
2: -0.0016293, 0.0037397, -0.0021521, 0.0034738, -0.0030670, 0.0038767
3: 0.0014117, 0.0021222, 0.0013425, 0.0020870, -0.0004059, 0.0005130
4: 0.0032970, 0.0073095, 0.0034957, 0.0077002, -0.0028972, 0.0022921
5: 0.9964223, 0.9975371, 0.9964774, 0.9976456, -0.0008049, 0.0006368
6: 0.0046361, 0.0056480, 0.0046862, 0.0057466, -0.0007306, 0.0005780
7: -0.0060803, -0.0023041, -0.0058933, -0.0019364, -0.0027266, 0.0021571
8: -0.0073996, -0.0044606, -0.0076858, -0.0046061, -0.0016789, 0.0021221
9: -0.0036249, -0.0033713, -0.0036123, -0.0033466, -0.0001831, 0.0001448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003481, upper bound: 0.0003299
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003491, upper bound: 0.0003278
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0091503, -0.0064436, -0.0014858, 0.0019073
1: -0.0054506, -0.0047089, -0.0055185, -0.0047554, -0.0004189, 0.0005377
2: -0.0016560, 0.0038166, -0.0021567, 0.0034739, -0.0030908, 0.0039675
3: 0.0014081, 0.0021324, 0.0013419, 0.0020870, -0.0004090, 0.0005250
4: 0.0032396, 0.0073295, 0.0034957, 0.0077036, -0.0029651, 0.0023099
5: 0.9964063, 0.9975426, 0.9964774, 0.9976466, -0.0008238, 0.0006418
6: 0.0046217, 0.0056531, 0.0046862, 0.0057474, -0.0007478, 0.0005825
7: -0.0061343, -0.0022853, -0.0058933, -0.0019332, -0.0027905, 0.0021739
8: -0.0074142, -0.0044185, -0.0076883, -0.0046061, -0.0016919, 0.0021718
9: -0.0036285, -0.0033701, -0.0036123, -0.0033464, -0.0001874, 0.0001460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003488, upper bound: 0.0003454
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003501, upper bound: 0.0003445
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090179, -0.0063704, -0.0089804, -0.0063862, -0.0016885, 0.0016866
1: -0.0054811, -0.0047347, -0.0054706, -0.0047392, -0.0004760, 0.0004755
2: -0.0018811, 0.0036261, -0.0018032, 0.0035932, -0.0035124, 0.0035084
3: 0.0013784, 0.0021071, 0.0013887, 0.0021028, -0.0004648, 0.0004643
4: 0.0033820, 0.0074977, 0.0034065, 0.0074395, -0.0026220, 0.0026249
5: 0.9964458, 0.9975892, 0.9964527, 0.9975731, -0.0007285, 0.0007293
6: 0.0046576, 0.0056955, 0.0046637, 0.0056808, -0.0006612, 0.0006620
7: -0.0060003, -0.0021270, -0.0059772, -0.0021818, -0.0024676, 0.0024704
8: -0.0075375, -0.0045228, -0.0074948, -0.0045408, -0.0019227, 0.0019205
9: -0.0036195, -0.0033594, -0.0036180, -0.0033631, -0.0001657, 0.0001659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002616, upper bound: 0.0002772
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003059, upper bound: 0.0003098
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090061, -0.0063443, -0.0089695, -0.0063876, -0.0016908, 0.0017193
1: -0.0054778, -0.0047274, -0.0054675, -0.0047396, -0.0004767, 0.0004847
2: -0.0018566, 0.0036804, -0.0017805, 0.0035904, -0.0035173, 0.0035764
3: 0.0013816, 0.0021143, 0.0013917, 0.0021024, -0.0004655, 0.0004733
4: 0.0033414, 0.0074794, 0.0034086, 0.0074225, -0.0026728, 0.0026286
5: 0.9964346, 0.9975842, 0.9964533, 0.9975684, -0.0007426, 0.0007303
6: 0.0046473, 0.0056909, 0.0046643, 0.0056765, -0.0006740, 0.0006629
7: -0.0060385, -0.0021442, -0.0059752, -0.0021977, -0.0025154, 0.0024738
8: -0.0075240, -0.0044931, -0.0074824, -0.0045423, -0.0019254, 0.0019577
9: -0.0036221, -0.0033606, -0.0036178, -0.0033642, -0.0001689, 0.0001661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002527, upper bound: 0.0002741
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003008, upper bound: 0.0003091
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090180, -0.0063704, -0.0090049, -0.0063428, -0.0017446, 0.0016990
1: -0.0054812, -0.0047347, -0.0054775, -0.0047269, -0.0004919, 0.0004790
2: -0.0018815, 0.0036260, -0.0018542, 0.0036836, -0.0036291, 0.0035343
3: 0.0013783, 0.0021071, 0.0013819, 0.0021148, -0.0004803, 0.0004677
4: 0.0033820, 0.0074980, 0.0033390, 0.0074776, -0.0026413, 0.0027122
5: 0.9964458, 0.9975893, 0.9964339, 0.9975837, -0.0007338, 0.0007535
6: 0.0046576, 0.0056956, 0.0046467, 0.0056904, -0.0006661, 0.0006840
7: -0.0060003, -0.0021267, -0.0060408, -0.0021459, -0.0024858, 0.0025525
8: -0.0075377, -0.0045228, -0.0075227, -0.0044913, -0.0019866, 0.0019347
9: -0.0036195, -0.0033594, -0.0036222, -0.0033607, -0.0001669, 0.0001714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003023, upper bound: 0.0003017
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003406, upper bound: 0.0003241
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090066, -0.0063443, -0.0089936, -0.0063441, -0.0017469, 0.0017311
1: -0.0054779, -0.0047274, -0.0054743, -0.0047273, -0.0004925, 0.0004881
2: -0.0018576, 0.0036804, -0.0018306, 0.0036808, -0.0036339, 0.0036010
3: 0.0013815, 0.0021143, 0.0013850, 0.0021144, -0.0004809, 0.0004765
4: 0.0033414, 0.0074801, 0.0033411, 0.0074599, -0.0026912, 0.0027158
5: 0.9964346, 0.9975845, 0.9964346, 0.9975789, -0.0007477, 0.0007545
6: 0.0046473, 0.0056910, 0.0046472, 0.0056860, -0.0006787, 0.0006849
7: -0.0060385, -0.0021435, -0.0060388, -0.0021625, -0.0025327, 0.0025559
8: -0.0075246, -0.0044931, -0.0075098, -0.0044929, -0.0019892, 0.0019712
9: -0.0036221, -0.0033606, -0.0036221, -0.0033618, -0.0001701, 0.0001716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002950, upper bound: 0.0002995
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0003376, upper bound: 0.0003241
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0089010, -0.0062417, -0.0015529, 0.0015069
1: -0.0054470, -0.0047193, -0.0054482, -0.0046984, -0.0004378, 0.0004249
2: -0.0016293, 0.0037397, -0.0016380, 0.0038938, -0.0032303, 0.0031348
3: 0.0014117, 0.0021222, 0.0014105, 0.0021426, -0.0004275, 0.0004148
4: 0.0032970, 0.0073095, 0.0031819, 0.0073160, -0.0023427, 0.0024141
5: 0.9964223, 0.9975371, 0.9963902, 0.9975388, -0.0006509, 0.0006707
6: 0.0046361, 0.0056480, 0.0046071, 0.0056497, -0.0005908, 0.0006088
7: -0.0060803, -0.0023041, -0.0061886, -0.0022979, -0.0022048, 0.0022719
8: -0.0073996, -0.0044606, -0.0074044, -0.0043763, -0.0017682, 0.0017160
9: -0.0036249, -0.0033713, -0.0036322, -0.0033709, -0.0001480, 0.0001526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004371, upper bound: 0.0004303
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004398, upper bound: 0.0004303
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0089034, -0.0062418, -0.0015656, 0.0015681
1: -0.0054506, -0.0047089, -0.0054489, -0.0046985, -0.0004414, 0.0004421
2: -0.0016560, 0.0038166, -0.0016430, 0.0038937, -0.0032568, 0.0032620
3: 0.0014081, 0.0021324, 0.0014099, 0.0021426, -0.0004310, 0.0004317
4: 0.0032396, 0.0073295, 0.0031820, 0.0073197, -0.0024378, 0.0024339
5: 0.9964063, 0.9975426, 0.9963903, 0.9975399, -0.0006773, 0.0006762
6: 0.0046217, 0.0056531, 0.0046071, 0.0056506, -0.0006148, 0.0006138
7: -0.0061343, -0.0022853, -0.0061885, -0.0022944, -0.0022943, 0.0022906
8: -0.0074142, -0.0044185, -0.0074071, -0.0043763, -0.0017828, 0.0017856
9: -0.0036285, -0.0033701, -0.0036322, -0.0033707, -0.0001541, 0.0001538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004371, upper bound: 0.0004408
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004398, upper bound: 0.0004407
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0088968, -0.0063158, -0.0090309, -0.0062864, -0.0015081, 0.0016472
1: -0.0054470, -0.0047193, -0.0054848, -0.0047110, -0.0004252, 0.0004644
2: -0.0016293, 0.0037397, -0.0019084, 0.0038009, -0.0031371, 0.0034265
3: 0.0014117, 0.0021222, 0.0013748, 0.0021303, -0.0004151, 0.0004534
4: 0.0032970, 0.0073095, 0.0032513, 0.0075181, -0.0025608, 0.0023444
5: 0.9964223, 0.9975371, 0.9964095, 0.9975950, -0.0007115, 0.0006514
6: 0.0046361, 0.0056480, 0.0046246, 0.0057006, -0.0006458, 0.0005912
7: -0.0060803, -0.0023041, -0.0061233, -0.0021078, -0.0024100, 0.0022064
8: -0.0073996, -0.0044606, -0.0075524, -0.0044271, -0.0017172, 0.0018757
9: -0.0036249, -0.0033713, -0.0036278, -0.0033582, -0.0001618, 0.0001482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004309, upper bound: 0.0004291
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004340, upper bound: 0.0004291
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0089096, -0.0062788, -0.0090310, -0.0062864, -0.0015208, 0.0017017
1: -0.0054506, -0.0047089, -0.0054848, -0.0047110, -0.0004288, 0.0004798
2: -0.0016560, 0.0038166, -0.0019084, 0.0038009, -0.0031637, 0.0035399
3: 0.0014081, 0.0021324, 0.0013747, 0.0021303, -0.0004187, 0.0004684
4: 0.0032396, 0.0073295, 0.0032513, 0.0075181, -0.0026455, 0.0023643
5: 0.9964063, 0.9975426, 0.9964095, 0.9975950, -0.0007350, 0.0006569
6: 0.0046217, 0.0056531, 0.0046246, 0.0057006, -0.0006672, 0.0005962
7: -0.0061343, -0.0022853, -0.0061233, -0.0021077, -0.0024897, 0.0022251
8: -0.0074142, -0.0044185, -0.0075524, -0.0044271, -0.0017318, 0.0019377
9: -0.0036285, -0.0033701, -0.0036278, -0.0033582, -0.0001672, 0.0001494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004310, upper bound: 0.0004377
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004340, upper bound: 0.0004377
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090179, -0.0063704, -0.0088508, -0.0062347, -0.0017037, 0.0014669
1: -0.0054811, -0.0047347, -0.0054340, -0.0046965, -0.0004803, 0.0004136
2: -0.0018811, 0.0036261, -0.0015336, 0.0039083, -0.0035440, 0.0030514
3: 0.0013784, 0.0021071, 0.0014243, 0.0021445, -0.0004690, 0.0004038
4: 0.0033820, 0.0074977, 0.0031710, 0.0072380, -0.0022804, 0.0026485
5: 0.9964458, 0.9975892, 0.9963872, 0.9975171, -0.0006336, 0.0007358
6: 0.0046576, 0.0056955, 0.0046044, 0.0056300, -0.0005751, 0.0006679
7: -0.0060003, -0.0021270, -0.0061989, -0.0023714, -0.0021461, 0.0024926
8: -0.0075375, -0.0045228, -0.0073472, -0.0043683, -0.0019400, 0.0016703
9: -0.0036195, -0.0033594, -0.0036329, -0.0033759, -0.0001441, 0.0001674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004198, upper bound: 0.0004288
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004288
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090061, -0.0063443, -0.0088397, -0.0062361, -0.0017034, 0.0015011
1: -0.0054778, -0.0047274, -0.0054309, -0.0046969, -0.0004802, 0.0004232
2: -0.0018566, 0.0036804, -0.0015105, 0.0039055, -0.0035434, 0.0031226
3: 0.0013816, 0.0021143, 0.0014274, 0.0021441, -0.0004689, 0.0004132
4: 0.0033414, 0.0074794, 0.0031732, 0.0072207, -0.0023336, 0.0026481
5: 0.9964346, 0.9975842, 0.9963878, 0.9975123, -0.0006484, 0.0007357
6: 0.0046473, 0.0056909, 0.0046049, 0.0056256, -0.0005885, 0.0006678
7: -0.0060385, -0.0021442, -0.0061968, -0.0023876, -0.0021962, 0.0024922
8: -0.0075240, -0.0044931, -0.0073346, -0.0043699, -0.0019397, 0.0017093
9: -0.0036221, -0.0033606, -0.0036327, -0.0033769, -0.0001475, 0.0001673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004198, upper bound: 0.0004304
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004250, upper bound: 0.0004305
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0090180, -0.0063704, -0.0088699, -0.0061975, -0.0017598, 0.0014805
1: -0.0054812, -0.0047347, -0.0054394, -0.0046860, -0.0004962, 0.0004174
2: -0.0018815, 0.0036260, -0.0015734, 0.0039857, -0.0036608, 0.0030796
3: 0.0013783, 0.0021071, 0.0014191, 0.0021547, -0.0004844, 0.0004075
4: 0.0033820, 0.0074980, 0.0031132, 0.0072678, -0.0023015, 0.0027358
5: 0.9964458, 0.9975893, 0.9963712, 0.9975255, -0.0006394, 0.0007601
6: 0.0046576, 0.0056956, 0.0045898, 0.0056375, -0.0005804, 0.0006899
7: -0.0060003, -0.0021267, -0.0062533, -0.0023434, -0.0021660, 0.0025747
8: -0.0075377, -0.0045228, -0.0073690, -0.0043259, -0.0020039, 0.0016858
9: -0.0036195, -0.0033594, -0.0036365, -0.0033740, -0.0001454, 0.0001729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004241, upper bound: 0.0004288
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004320, upper bound: 0.0004288
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090066, -0.0063443, -0.0088587, -0.0061990, -0.0017597, 0.0015144
1: -0.0054779, -0.0047274, -0.0054363, -0.0046864, -0.0004961, 0.0004270
2: -0.0018576, 0.0036804, -0.0015501, 0.0039826, -0.0036606, 0.0031503
3: 0.0013815, 0.0021143, 0.0014222, 0.0021543, -0.0004844, 0.0004169
4: 0.0033414, 0.0074801, 0.0031155, 0.0072503, -0.0023543, 0.0027357
5: 0.9964346, 0.9975845, 0.9963718, 0.9975206, -0.0006541, 0.0007601
6: 0.0046473, 0.0056910, 0.0045904, 0.0056331, -0.0005937, 0.0006899
7: -0.0060385, -0.0021435, -0.0062511, -0.0023598, -0.0022157, 0.0025746
8: -0.0075246, -0.0044931, -0.0073563, -0.0043276, -0.0020038, 0.0017245
9: -0.0036221, -0.0033606, -0.0036364, -0.0033751, -0.0001488, 0.0001729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004241, upper bound: 0.0004305
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004320, upper bound: 0.0004304
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0089727, -0.0064207, -0.0088807, -0.0063358, -0.0016090, 0.0014553
1: -0.0054684, -0.0047489, -0.0054425, -0.0047250, -0.0004536, 0.0004103
2: -0.0017871, 0.0035214, -0.0015958, 0.0036981, -0.0033470, 0.0030274
3: 0.0013908, 0.0020933, 0.0014161, 0.0021167, -0.0004429, 0.0004006
4: 0.0034602, 0.0074274, 0.0033282, 0.0072845, -0.0022625, 0.0025013
5: 0.9964676, 0.9975698, 0.9964309, 0.9975301, -0.0006286, 0.0006949
6: 0.0046773, 0.0056778, 0.0046440, 0.0056417, -0.0005706, 0.0006308
7: -0.0059267, -0.0021931, -0.0060510, -0.0023276, -0.0021293, 0.0023540
8: -0.0074860, -0.0045801, -0.0073813, -0.0044834, -0.0018321, 0.0016572
9: -0.0036146, -0.0033639, -0.0036229, -0.0033729, -0.0001430, 0.0001581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004098, upper bound: 0.0004164
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004120, upper bound: 0.0004164
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0090039, -0.0063861, -0.0088879, -0.0063351, -0.0016042, 0.0015155
1: -0.0054772, -0.0047392, -0.0054445, -0.0047248, -0.0004523, 0.0004273
2: -0.0018520, 0.0035934, -0.0016108, 0.0036995, -0.0033371, 0.0031525
3: 0.0013822, 0.0021028, 0.0014141, 0.0021169, -0.0004416, 0.0004172
4: 0.0034064, 0.0074759, 0.0033271, 0.0072957, -0.0023560, 0.0024939
5: 0.9964527, 0.9975833, 0.9964306, 0.9975332, -0.0006546, 0.0006929
6: 0.0046637, 0.0056900, 0.0046437, 0.0056445, -0.0005942, 0.0006289
7: -0.0059773, -0.0021475, -0.0060520, -0.0023171, -0.0022173, 0.0023471
8: -0.0075215, -0.0045407, -0.0073895, -0.0044826, -0.0018267, 0.0017257
9: -0.0036180, -0.0033608, -0.0036230, -0.0033722, -0.0001489, 0.0001576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004099, upper bound: 0.0004240
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004120, upper bound: 0.0004240
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0089755, -0.0064208, -0.0088936, -0.0062947, -0.0016617, 0.0014668
1: -0.0054692, -0.0047489, -0.0054461, -0.0047134, -0.0004685, 0.0004136
2: -0.0017929, 0.0035213, -0.0016227, 0.0037836, -0.0034566, 0.0030513
3: 0.0013900, 0.0020933, 0.0014126, 0.0021280, -0.0004574, 0.0004038
4: 0.0034603, 0.0074318, 0.0032643, 0.0073046, -0.0022804, 0.0025833
5: 0.9964676, 0.9975711, 0.9964132, 0.9975357, -0.0006336, 0.0007177
6: 0.0046773, 0.0056789, 0.0046279, 0.0056468, -0.0005751, 0.0006515
7: -0.0059266, -0.0021890, -0.0061111, -0.0023087, -0.0021461, 0.0024311
8: -0.0074892, -0.0045802, -0.0073960, -0.0044366, -0.0018922, 0.0016703
9: -0.0036146, -0.0033636, -0.0036270, -0.0033716, -0.0001441, 0.0001632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004222, upper bound: 0.0004164
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004231, upper bound: 0.0004164
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0090084, -0.0063862, -0.0089009, -0.0062939, -0.0016521, 0.0015271
1: -0.0054785, -0.0047392, -0.0054482, -0.0047132, -0.0004658, 0.0004306
2: -0.0018615, 0.0035933, -0.0016379, 0.0037852, -0.0034366, 0.0031767
3: 0.0013810, 0.0021028, 0.0014105, 0.0021282, -0.0004548, 0.0004204
4: 0.0034065, 0.0074830, 0.0032630, 0.0073160, -0.0023741, 0.0025683
5: 0.9964527, 0.9975852, 0.9964128, 0.9975389, -0.0006596, 0.0007136
6: 0.0046637, 0.0056918, 0.0046276, 0.0056496, -0.0005987, 0.0006477
7: -0.0059773, -0.0021408, -0.0061122, -0.0022980, -0.0022343, 0.0024171
8: -0.0075267, -0.0045407, -0.0074043, -0.0044357, -0.0018812, 0.0017390
9: -0.0036180, -0.0033604, -0.0036270, -0.0033709, -0.0001500, 0.0001623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004221, upper bound: 0.0004240
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0004231, upper bound: 0.0004240
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0090987, -0.0064595, -0.0088855, -0.0063376, -0.0017445, 0.0014430
1: -0.0055039, -0.0047598, -0.0054438, -0.0047255, -0.0004918, 0.0004068
2: -0.0020493, 0.0034408, -0.0016059, 0.0036943, -0.0036289, 0.0030018
3: 0.0013561, 0.0020826, 0.0014148, 0.0021162, -0.0004802, 0.0003972
4: 0.0035204, 0.0076234, 0.0033310, 0.0072920, -0.0022433, 0.0027120
5: 0.9964844, 0.9976243, 0.9964317, 0.9975321, -0.0006233, 0.0007535
6: 0.0046925, 0.0057272, 0.0046447, 0.0056436, -0.0005657, 0.0006839
7: -0.0058700, -0.0020087, -0.0060483, -0.0023206, -0.0021112, 0.0025523
8: -0.0076295, -0.0046242, -0.0073868, -0.0044854, -0.0019865, 0.0016432
9: -0.0036108, -0.0033515, -0.0036228, -0.0033724, -0.0001418, 0.0001714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.44 + 597.39 = 600.83 seconds
