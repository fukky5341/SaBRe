## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00059895


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9906135, 0.9926143, 0.9906135, 0.9926143, -0.0010450, 0.0010450)
1: (-0.0036028, -0.0031043, -0.0036028, -0.0031043, -0.0002604, 0.0002604)
2: (0.0063971, 0.0090390, 0.0063971, 0.0090390, -0.0013799, 0.0013799)
3: (-0.0053873, -0.0041848, -0.0053873, -0.0041848, -0.0006281, 0.0006281)
4: (0.0017660, 0.0022774, 0.0017660, 0.0022774, -0.0002671, 0.0002671)
5: (0.0070053, 0.0103281, 0.0070053, 0.0103281, -0.0017356, 0.0017356)
6: (-0.0010806, -0.0002372, -0.0010806, -0.0002372, -0.0004405, 0.0004405)
7: (-0.0059334, -0.0037513, -0.0059334, -0.0037513, -0.0011397, 0.0011397)
8: (-0.0026844, -0.0015369, -0.0026844, -0.0015369, -0.0005994, 0.0005994)
9: (-0.0000817, 0.0012489, -0.0000817, 0.0012489, -0.0006950, 0.0006950)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.38 = 2.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006989, upper bound: 0.0006989

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006769, upper bound: 0.0006470
time: 0.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006769, upper bound: 0.0006769
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -0.0006769, upper bound: 0.0006470
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -0.0006769, upper bound: 0.0006769

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9906262, 0.9925235, 0.9906136, 0.9925845, -0.0009588, 0.0009480
1: -0.0035996, -0.0031269, -0.0036028, -0.0031117, -0.0002389, 0.0002362
2: 0.0065170, 0.0090222, 0.0064364, 0.0090389, -0.0012518, 0.0012661
3: -0.0053797, -0.0042394, -0.0053872, -0.0042027, -0.0005763, 0.0005698
4: 0.0017892, 0.0022741, 0.0017736, 0.0022773, -0.0002423, 0.0002451
5: 0.0071561, 0.0103070, 0.0070548, 0.0103280, -0.0015744, 0.0015924
6: -0.0010752, -0.0002755, -0.0010805, -0.0002497, -0.0004042, 0.0003996
7: -0.0059195, -0.0038503, -0.0059333, -0.0037838, -0.0010457, 0.0010339
8: -0.0026772, -0.0015890, -0.0026844, -0.0015540, -0.0005499, 0.0005437
9: -0.0000213, 0.0012404, -0.0000619, 0.0012488, -0.0006305, 0.0006377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006454, upper bound: 0.0006105
time: 0.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006108
time: 0.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9906138, 0.9925810, 0.9906136, 0.9926062, -0.0010342, 0.0009204
1: -0.0036027, -0.0031126, -0.0036028, -0.0031063, -0.0002577, 0.0002293
2: 0.0064411, 0.0090386, 0.0064078, 0.0090389, -0.0012153, 0.0013656
3: -0.0053871, -0.0042048, -0.0053872, -0.0041897, -0.0006216, 0.0005532
4: 0.0017746, 0.0022773, 0.0017681, 0.0022774, -0.0002352, 0.0002643
5: 0.0070607, 0.0103277, 0.0070188, 0.0103280, -0.0015286, 0.0017176
6: -0.0010804, -0.0002513, -0.0010805, -0.0002406, -0.0004359, 0.0003880
7: -0.0059330, -0.0037877, -0.0059333, -0.0037601, -0.0011279, 0.0010038
8: -0.0026843, -0.0015561, -0.0026844, -0.0015416, -0.0005932, 0.0005279
9: -0.0000595, 0.0012487, -0.0000763, 0.0012488, -0.0006121, 0.0006878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006471, upper bound: 0.0006769
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006471, upper bound: 0.0006769
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0006454, upper bound: 0.0006105
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006108
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0006471, upper bound: 0.0006769
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0006471, upper bound: 0.0006769

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906262, 0.9925235, 0.9906144, 0.9925627, -0.0009247, 0.0009440
1: -0.0035996, -0.0031269, -0.0036026, -0.0031171, -0.0002304, 0.0002352
2: 0.0065170, 0.0090222, 0.0064652, 0.0090378, -0.0012466, 0.0012211
3: -0.0053797, -0.0042394, -0.0053867, -0.0042158, -0.0005558, 0.0005674
4: 0.0017892, 0.0022741, 0.0017792, 0.0022771, -0.0002413, 0.0002363
5: 0.0071561, 0.0103070, 0.0070909, 0.0103266, -0.0015679, 0.0015358
6: -0.0010752, -0.0002755, -0.0010802, -0.0002589, -0.0003898, 0.0003979
7: -0.0059195, -0.0038503, -0.0059323, -0.0038075, -0.0010085, 0.0010296
8: -0.0026772, -0.0015890, -0.0026839, -0.0015665, -0.0005304, 0.0005414
9: -0.0000213, 0.0012404, -0.0000474, 0.0012483, -0.0006278, 0.0006150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006105
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006105
time: 0.58 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906270, 0.9924915, 0.9905533, 0.9924822, -0.0009255, 0.0010320
1: -0.0035995, -0.0031349, -0.0036178, -0.0031372, -0.0002306, 0.0002571
2: 0.0065592, 0.0090213, 0.0065715, 0.0091185, -0.0013627, 0.0012221
3: -0.0053792, -0.0042586, -0.0054235, -0.0042642, -0.0005563, 0.0006202
4: 0.0017974, 0.0022739, 0.0017998, 0.0022928, -0.0002637, 0.0002365
5: 0.0072092, 0.0103059, 0.0072246, 0.0104282, -0.0017139, 0.0015371
6: -0.0010749, -0.0002889, -0.0011060, -0.0002929, -0.0003901, 0.0004350
7: -0.0059187, -0.0038852, -0.0059991, -0.0038953, -0.0010094, 0.0011255
8: -0.0026768, -0.0016073, -0.0027190, -0.0016127, -0.0005308, 0.0005919
9: -0.0000001, 0.0012400, 0.0000061, 0.0012890, -0.0006863, 0.0006155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006108
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006108
time: 0.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906138, 0.9925810, 0.9906262, 0.9925235, -0.0009467, 0.0009750
1: -0.0036027, -0.0031126, -0.0035996, -0.0031269, -0.0002359, 0.0002429
2: 0.0064411, 0.0090386, 0.0065170, 0.0090222, -0.0012875, 0.0012500
3: -0.0053871, -0.0042048, -0.0053797, -0.0042394, -0.0005690, 0.0005860
4: 0.0017746, 0.0022773, 0.0017892, 0.0022741, -0.0002492, 0.0002419
5: 0.0070607, 0.0103277, 0.0071561, 0.0103070, -0.0016193, 0.0015722
6: -0.0010804, -0.0002513, -0.0010752, -0.0002755, -0.0003990, 0.0004110
7: -0.0059330, -0.0037877, -0.0059195, -0.0038503, -0.0010325, 0.0010634
8: -0.0026843, -0.0015561, -0.0026772, -0.0015890, -0.0005430, 0.0005592
9: -0.0000595, 0.0012487, -0.0000213, 0.0012404, -0.0006484, 0.0006296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006454
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
time: 0.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906138, 0.9925810, 0.9906138, 0.9925810, -0.0009196, 0.0009196
1: -0.0036027, -0.0031126, -0.0036027, -0.0031126, -0.0002291, 0.0002291
2: 0.0064411, 0.0090386, 0.0064411, 0.0090386, -0.0012143, 0.0012143
3: -0.0053871, -0.0042048, -0.0053871, -0.0042048, -0.0005527, 0.0005527
4: 0.0017746, 0.0022773, 0.0017746, 0.0022773, -0.0002350, 0.0002350
5: 0.0070607, 0.0103277, 0.0070607, 0.0103277, -0.0015273, 0.0015273
6: -0.0010804, -0.0002513, -0.0010804, -0.0002513, -0.0003876, 0.0003876
7: -0.0059330, -0.0037877, -0.0059330, -0.0037877, -0.0010029, 0.0010029
8: -0.0026843, -0.0015561, -0.0026843, -0.0015561, -0.0005274, 0.0005274
9: -0.0000595, 0.0012487, -0.0000595, 0.0012487, -0.0006116, 0.0006116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006454
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006105
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006105
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006108
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006425, upper bound: 0.0006108
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006454
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006454
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906271, 0.9925026, 0.9906144, 0.9925627, -0.0009209, 0.0009095
1: -0.0035994, -0.0031321, -0.0036026, -0.0031171, -0.0002295, 0.0002266
2: 0.0065445, 0.0090211, 0.0064652, 0.0090378, -0.0012010, 0.0012161
3: -0.0053791, -0.0042519, -0.0053867, -0.0042158, -0.0005535, 0.0005466
4: 0.0017946, 0.0022739, 0.0017792, 0.0022771, -0.0002325, 0.0002354
5: 0.0071907, 0.0103057, 0.0070909, 0.0103266, -0.0015105, 0.0015295
6: -0.0010749, -0.0002842, -0.0010802, -0.0002589, -0.0003882, 0.0003834
7: -0.0059186, -0.0038730, -0.0059323, -0.0038075, -0.0010044, 0.0009919
8: -0.0026767, -0.0016009, -0.0026839, -0.0015665, -0.0005282, 0.0005217
9: -0.0000075, 0.0012399, -0.0000474, 0.0012483, -0.0006049, 0.0006125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905726, 0.9924197, 0.9906144, 0.9925627, -0.0010594, 0.0008996
1: -0.0036130, -0.0031528, -0.0036026, -0.0031171, -0.0002640, 0.0002242
2: 0.0066539, 0.0090930, 0.0064652, 0.0090378, -0.0011880, 0.0013989
3: -0.0054119, -0.0043017, -0.0053867, -0.0042158, -0.0006367, 0.0005407
4: 0.0018157, 0.0022878, 0.0017792, 0.0022771, -0.0002299, 0.0002708
5: 0.0073284, 0.0103961, 0.0070909, 0.0103266, -0.0014941, 0.0017595
6: -0.0010978, -0.0003192, -0.0010802, -0.0002589, -0.0004466, 0.0003792
7: -0.0059780, -0.0039635, -0.0059323, -0.0038075, -0.0011554, 0.0009812
8: -0.0027079, -0.0016485, -0.0026839, -0.0015665, -0.0006076, 0.0005160
9: 0.0000477, 0.0012761, -0.0000474, 0.0012483, -0.0005983, 0.0007046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906271, 0.9925026, 0.9905533, 0.9924822, -0.0009025, 0.0010153
1: -0.0035994, -0.0031321, -0.0036178, -0.0031372, -0.0002249, 0.0002530
2: 0.0065445, 0.0090211, 0.0065715, 0.0091185, -0.0013407, 0.0011917
3: -0.0053791, -0.0042519, -0.0054235, -0.0042642, -0.0005424, 0.0006102
4: 0.0017946, 0.0022739, 0.0017998, 0.0022928, -0.0002595, 0.0002307
5: 0.0071907, 0.0103057, 0.0072246, 0.0104282, -0.0016862, 0.0014989
6: -0.0010749, -0.0002842, -0.0011060, -0.0002929, -0.0003804, 0.0004280
7: -0.0059186, -0.0038730, -0.0059991, -0.0038953, -0.0009843, 0.0011073
8: -0.0026767, -0.0016009, -0.0027190, -0.0016127, -0.0005176, 0.0005823
9: -0.0000075, 0.0012399, 0.0000061, 0.0012890, -0.0006752, 0.0006002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006108
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006108
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905726, 0.9924197, 0.9905533, 0.9924822, -0.0009192, 0.0009104
1: -0.0036130, -0.0031528, -0.0036178, -0.0031372, -0.0002290, 0.0002268
2: 0.0066539, 0.0090930, 0.0065715, 0.0091185, -0.0012021, 0.0012138
3: -0.0054119, -0.0043017, -0.0054235, -0.0042642, -0.0005524, 0.0005472
4: 0.0018157, 0.0022878, 0.0017998, 0.0022928, -0.0002327, 0.0002349
5: 0.0073284, 0.0103961, 0.0072246, 0.0104282, -0.0015120, 0.0015266
6: -0.0010978, -0.0003192, -0.0011060, -0.0002929, -0.0003875, 0.0003838
7: -0.0059780, -0.0039635, -0.0059991, -0.0038953, -0.0010025, 0.0009929
8: -0.0027079, -0.0016485, -0.0027190, -0.0016127, -0.0005272, 0.0005221
9: 0.0000477, 0.0012761, 0.0000061, 0.0012890, -0.0006055, 0.0006113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006105
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006105
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906147, 0.9925582, 0.9906262, 0.9925235, -0.0009427, 0.0009452
1: -0.0036025, -0.0031183, -0.0035996, -0.0031269, -0.0002349, 0.0002355
2: 0.0064711, 0.0090375, 0.0065170, 0.0090222, -0.0012481, 0.0012448
3: -0.0053866, -0.0042185, -0.0053797, -0.0042394, -0.0005666, 0.0005681
4: 0.0017804, 0.0022771, 0.0017892, 0.0022741, -0.0002416, 0.0002409
5: 0.0070984, 0.0103262, 0.0071561, 0.0103070, -0.0015698, 0.0015657
6: -0.0010801, -0.0002608, -0.0010752, -0.0002755, -0.0003974, 0.0003984
7: -0.0059321, -0.0038124, -0.0059195, -0.0038503, -0.0010282, 0.0010309
8: -0.0026838, -0.0015691, -0.0026772, -0.0015890, -0.0005407, 0.0005421
9: -0.0000444, 0.0012481, -0.0000213, 0.0012404, -0.0006286, 0.0006270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905536, 0.9924812, 0.9906270, 0.9924915, -0.0010310, 0.0009469
1: -0.0036178, -0.0031374, -0.0035995, -0.0031349, -0.0002569, 0.0002359
2: 0.0065728, 0.0091183, 0.0065592, 0.0090213, -0.0012503, 0.0013614
3: -0.0054234, -0.0042648, -0.0053792, -0.0042586, -0.0006196, 0.0005691
4: 0.0018000, 0.0022927, 0.0017974, 0.0022739, -0.0002420, 0.0002635
5: 0.0072263, 0.0104278, 0.0072092, 0.0103059, -0.0015726, 0.0017123
6: -0.0011059, -0.0002933, -0.0010749, -0.0002889, -0.0004346, 0.0003991
7: -0.0059988, -0.0038965, -0.0059187, -0.0038852, -0.0011244, 0.0010327
8: -0.0027189, -0.0016133, -0.0026768, -0.0016073, -0.0005913, 0.0005431
9: 0.0000068, 0.0012888, -0.0000001, 0.0012400, -0.0006297, 0.0006857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906147, 0.9925582, 0.9906138, 0.9925810, -0.0009158, 0.0008851
1: -0.0036025, -0.0031183, -0.0036027, -0.0031126, -0.0002282, 0.0002205
2: 0.0064711, 0.0090375, 0.0064411, 0.0090386, -0.0011688, 0.0012093
3: -0.0053866, -0.0042185, -0.0053871, -0.0042048, -0.0005504, 0.0005320
4: 0.0017804, 0.0022771, 0.0017746, 0.0022773, -0.0002262, 0.0002341
5: 0.0070984, 0.0103262, 0.0070607, 0.0103277, -0.0014700, 0.0015210
6: -0.0010801, -0.0002608, -0.0010804, -0.0002513, -0.0003860, 0.0003731
7: -0.0059321, -0.0038124, -0.0059330, -0.0037877, -0.0009988, 0.0009653
8: -0.0026838, -0.0015691, -0.0026843, -0.0015561, -0.0005253, 0.0005077
9: -0.0000444, 0.0012481, -0.0000595, 0.0012487, -0.0005887, 0.0006091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905536, 0.9924812, 0.9906145, 0.9925510, -0.0010367, 0.0008929
1: -0.0036178, -0.0031374, -0.0036026, -0.0031200, -0.0002583, 0.0002225
2: 0.0065728, 0.0091183, 0.0064806, 0.0090377, -0.0011791, 0.0013690
3: -0.0054234, -0.0042648, -0.0053867, -0.0042228, -0.0006231, 0.0005367
4: 0.0018000, 0.0022927, 0.0017822, 0.0022771, -0.0002282, 0.0002650
5: 0.0072263, 0.0104278, 0.0071103, 0.0103265, -0.0014830, 0.0017218
6: -0.0011059, -0.0002933, -0.0010801, -0.0002638, -0.0004370, 0.0003764
7: -0.0059988, -0.0038965, -0.0059323, -0.0038203, -0.0011307, 0.0009739
8: -0.0027189, -0.0016133, -0.0026839, -0.0015732, -0.0005946, 0.0005122
9: 0.0000068, 0.0012888, -0.0000397, 0.0012482, -0.0005939, 0.0006895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
time: 0.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006289, upper bound: 0.0006105
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006108
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006108
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006105
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006230, upper bound: 0.0006105
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006105, upper bound: 0.0006425
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.0006108, upper bound: 0.0006425

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906271, 0.9925026, 0.9906271, 0.9925026, -0.0008644, 0.0008644
1: -0.0035994, -0.0031321, -0.0035994, -0.0031321, -0.0002154, 0.0002154
2: 0.0065445, 0.0090211, 0.0065445, 0.0090211, -0.0011414, 0.0011414
3: -0.0053791, -0.0042519, -0.0053791, -0.0042519, -0.0005195, 0.0005195
4: 0.0017946, 0.0022739, 0.0017946, 0.0022739, -0.0002209, 0.0002209
5: 0.0071907, 0.0103057, 0.0071907, 0.0103057, -0.0014356, 0.0014356
6: -0.0010749, -0.0002842, -0.0010749, -0.0002842, -0.0003644, 0.0003644
7: -0.0059186, -0.0038730, -0.0059186, -0.0038730, -0.0009427, 0.0009427
8: -0.0026767, -0.0016009, -0.0026767, -0.0016009, -0.0004958, 0.0004958
9: -0.0000075, 0.0012399, -0.0000075, 0.0012399, -0.0005749, 0.0005749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0006080
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0006080
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906271, 0.9925026, 0.9906147, 0.9925582, -0.0009414, 0.0009082
1: -0.0035994, -0.0031321, -0.0036025, -0.0031183, -0.0002346, 0.0002263
2: 0.0065445, 0.0090211, 0.0064711, 0.0090375, -0.0011993, 0.0012431
3: -0.0053791, -0.0042519, -0.0053866, -0.0042185, -0.0005658, 0.0005459
4: 0.0017946, 0.0022739, 0.0017804, 0.0022771, -0.0002321, 0.0002406
5: 0.0071907, 0.0103057, 0.0070984, 0.0103262, -0.0015084, 0.0015635
6: -0.0010749, -0.0002842, -0.0010801, -0.0002608, -0.0003968, 0.0003828
7: -0.0059186, -0.0038730, -0.0059321, -0.0038124, -0.0010267, 0.0009905
8: -0.0026767, -0.0016009, -0.0026838, -0.0015691, -0.0005399, 0.0005209
9: -0.0000075, 0.0012399, -0.0000444, 0.0012481, -0.0006040, 0.0006261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0006080
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0006080
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905726, 0.9924197, 0.9906271, 0.9925026, -0.0010028, 0.0008545
1: -0.0036130, -0.0031528, -0.0035994, -0.0031321, -0.0002499, 0.0002129
2: 0.0066539, 0.0090930, 0.0065445, 0.0090211, -0.0011284, 0.0013243
3: -0.0054119, -0.0043017, -0.0053791, -0.0042519, -0.0006027, 0.0005136
4: 0.0018157, 0.0022878, 0.0017946, 0.0022739, -0.0002184, 0.0002563
5: 0.0073284, 0.0103961, 0.0071907, 0.0103057, -0.0014192, 0.0016656
6: -0.0010978, -0.0003192, -0.0010749, -0.0002842, -0.0004227, 0.0003602
7: -0.0059780, -0.0039635, -0.0059186, -0.0038730, -0.0010938, 0.0009320
8: -0.0027079, -0.0016485, -0.0026767, -0.0016009, -0.0005752, 0.0004901
9: 0.0000477, 0.0012761, -0.0000075, 0.0012399, -0.0005683, 0.0006670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0005998
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0005998
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905726, 0.9924197, 0.9906147, 0.9925582, -0.0010799, 0.0008983
1: -0.0036130, -0.0031528, -0.0036025, -0.0031183, -0.0002691, 0.0002238
2: 0.0066539, 0.0090930, 0.0064711, 0.0090375, -0.0011862, 0.0014260
3: -0.0054119, -0.0043017, -0.0053866, -0.0042185, -0.0006490, 0.0005399
4: 0.0018157, 0.0022878, 0.0017804, 0.0022771, -0.0002296, 0.0002760
5: 0.0073284, 0.0103961, 0.0070984, 0.0103262, -0.0014920, 0.0017935
6: -0.0010978, -0.0003192, -0.0010801, -0.0002608, -0.0004552, 0.0003787
7: -0.0059780, -0.0039635, -0.0059321, -0.0038124, -0.0011778, 0.0009798
8: -0.0027079, -0.0016485, -0.0026838, -0.0015691, -0.0006194, 0.0005152
9: 0.0000477, 0.0012761, -0.0000444, 0.0012481, -0.0005975, 0.0007182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0005998
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0005998
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906271, 0.9925026, 0.9905726, 0.9924197, -0.0008545, 0.0010028
1: -0.0035994, -0.0031321, -0.0036130, -0.0031528, -0.0002129, 0.0002499
2: 0.0065445, 0.0090211, 0.0066539, 0.0090930, -0.0013243, 0.0011284
3: -0.0053791, -0.0042519, -0.0054119, -0.0043017, -0.0005136, 0.0006027
4: 0.0017946, 0.0022739, 0.0018157, 0.0022878, -0.0002563, 0.0002184
5: 0.0071907, 0.0103057, 0.0073284, 0.0103961, -0.0016656, 0.0014192
6: -0.0010749, -0.0002842, -0.0010978, -0.0003192, -0.0003602, 0.0004227
7: -0.0059186, -0.0038730, -0.0059780, -0.0039635, -0.0009320, 0.0010937
8: -0.0026767, -0.0016009, -0.0027079, -0.0016485, -0.0004901, 0.0005752
9: -0.0000075, 0.0012399, 0.0000477, 0.0012761, -0.0006670, 0.0005683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005883, upper bound: 0.0006079
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006128, upper bound: 0.0006079
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906271, 0.9925026, 0.9905536, 0.9924812, -0.0009088, 0.0010143
1: -0.0035994, -0.0031321, -0.0036178, -0.0031374, -0.0002264, 0.0002527
2: 0.0065445, 0.0090211, 0.0065728, 0.0091183, -0.0013393, 0.0012000
3: -0.0053791, -0.0042519, -0.0054234, -0.0042648, -0.0005462, 0.0006096
4: 0.0017946, 0.0022739, 0.0018000, 0.0022927, -0.0002592, 0.0002323
5: 0.0071907, 0.0103057, 0.0072263, 0.0104278, -0.0016845, 0.0015093
6: -0.0010749, -0.0002842, -0.0011059, -0.0002933, -0.0003831, 0.0004276
7: -0.0059186, -0.0038730, -0.0059988, -0.0038965, -0.0009911, 0.0011062
8: -0.0026767, -0.0016009, -0.0027189, -0.0016133, -0.0005212, 0.0005817
9: -0.0000075, 0.0012399, 0.0000068, 0.0012888, -0.0006746, 0.0006044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005883, upper bound: 0.0006079
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006128, upper bound: 0.0006079
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905726, 0.9924197, 0.9905726, 0.9924197, -0.0008639, 0.0008639
1: -0.0036130, -0.0031528, -0.0036130, -0.0031528, -0.0002153, 0.0002153
2: 0.0066539, 0.0090930, 0.0066539, 0.0090930, -0.0011408, 0.0011408
3: -0.0054119, -0.0043017, -0.0054119, -0.0043017, -0.0005192, 0.0005192
4: 0.0018157, 0.0022878, 0.0018157, 0.0022878, -0.0002208, 0.0002208
5: 0.0073284, 0.0103961, 0.0073284, 0.0103961, -0.0014348, 0.0014348
6: -0.0010978, -0.0003192, -0.0010978, -0.0003192, -0.0003642, 0.0003642
7: -0.0059780, -0.0039635, -0.0059780, -0.0039635, -0.0009422, 0.0009422
8: -0.0027079, -0.0016485, -0.0027079, -0.0016485, -0.0004955, 0.0004955
9: 0.0000477, 0.0012761, 0.0000477, 0.0012761, -0.0005746, 0.0005746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005884, upper bound: 0.0005998
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0005998
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905726, 0.9924197, 0.9905536, 0.9924812, -0.0009405, 0.0009091
1: -0.0036130, -0.0031528, -0.0036178, -0.0031374, -0.0002344, 0.0002265
2: 0.0066539, 0.0090930, 0.0065728, 0.0091183, -0.0012005, 0.0012420
3: -0.0054119, -0.0043017, -0.0054234, -0.0042648, -0.0005653, 0.0005464
4: 0.0018157, 0.0022878, 0.0018000, 0.0022927, -0.0002323, 0.0002404
5: 0.0073284, 0.0103961, 0.0072263, 0.0104278, -0.0015099, 0.0015621
6: -0.0010978, -0.0003192, -0.0011059, -0.0002933, -0.0003965, 0.0003832
7: -0.0059780, -0.0039635, -0.0059988, -0.0038965, -0.0010258, 0.0009915
8: -0.0027079, -0.0016485, -0.0027189, -0.0016133, -0.0005394, 0.0005214
9: 0.0000477, 0.0012761, 0.0000068, 0.0012888, -0.0006046, 0.0006255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005884, upper bound: 0.0005998
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0005998
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906147, 0.9925582, 0.9906271, 0.9925026, -0.0009082, 0.0009414
1: -0.0036025, -0.0031183, -0.0035994, -0.0031321, -0.0002263, 0.0002346
2: 0.0064711, 0.0090375, 0.0065445, 0.0090211, -0.0012431, 0.0011993
3: -0.0053866, -0.0042185, -0.0053791, -0.0042519, -0.0005459, 0.0005658
4: 0.0017804, 0.0022771, 0.0017946, 0.0022739, -0.0002406, 0.0002321
5: 0.0070984, 0.0103262, 0.0071907, 0.0103057, -0.0015635, 0.0015084
6: -0.0010801, -0.0002608, -0.0010749, -0.0002842, -0.0003828, 0.0003968
7: -0.0059321, -0.0038124, -0.0059186, -0.0038730, -0.0009905, 0.0010267
8: -0.0026838, -0.0015691, -0.0026767, -0.0016009, -0.0005209, 0.0005399
9: -0.0000444, 0.0012481, -0.0000075, 0.0012399, -0.0006261, 0.0006040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906147, 0.9925582, 0.9905726, 0.9924197, -0.0008983, 0.0010799
1: -0.0036025, -0.0031183, -0.0036130, -0.0031528, -0.0002238, 0.0002691
2: 0.0064711, 0.0090375, 0.0066539, 0.0090930, -0.0014260, 0.0011862
3: -0.0053866, -0.0042185, -0.0054119, -0.0043017, -0.0005399, 0.0006490
4: 0.0017804, 0.0022771, 0.0018157, 0.0022878, -0.0002760, 0.0002296
5: 0.0070984, 0.0103262, 0.0073284, 0.0103961, -0.0017935, 0.0014920
6: -0.0010801, -0.0002608, -0.0010978, -0.0003192, -0.0003787, 0.0004552
7: -0.0059321, -0.0038124, -0.0059780, -0.0039635, -0.0009798, 0.0011778
8: -0.0026838, -0.0015691, -0.0027079, -0.0016485, -0.0005152, 0.0006194
9: -0.0000444, 0.0012481, 0.0000477, 0.0012761, -0.0007182, 0.0005975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905536, 0.9924812, 0.9906271, 0.9925026, -0.0010143, 0.0009088
1: -0.0036178, -0.0031374, -0.0035994, -0.0031321, -0.0002527, 0.0002264
2: 0.0065728, 0.0091183, 0.0065445, 0.0090211, -0.0012000, 0.0013393
3: -0.0054234, -0.0042648, -0.0053791, -0.0042519, -0.0006096, 0.0005462
4: 0.0018000, 0.0022927, 0.0017946, 0.0022739, -0.0002323, 0.0002592
5: 0.0072263, 0.0104278, 0.0071907, 0.0103057, -0.0015093, 0.0016845
6: -0.0011059, -0.0002933, -0.0010749, -0.0002842, -0.0004276, 0.0003831
7: -0.0059988, -0.0038965, -0.0059186, -0.0038730, -0.0011062, 0.0009911
8: -0.0027189, -0.0016133, -0.0026767, -0.0016009, -0.0005817, 0.0005212
9: 0.0000068, 0.0012888, -0.0000075, 0.0012399, -0.0006044, 0.0006746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006322
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905536, 0.9924812, 0.9905726, 0.9924197, -0.0009091, 0.0009405
1: -0.0036178, -0.0031374, -0.0036130, -0.0031528, -0.0002265, 0.0002344
2: 0.0065728, 0.0091183, 0.0066539, 0.0090930, -0.0012420, 0.0012005
3: -0.0054234, -0.0042648, -0.0054119, -0.0043017, -0.0005464, 0.0005653
4: 0.0018000, 0.0022927, 0.0018157, 0.0022878, -0.0002404, 0.0002323
5: 0.0072263, 0.0104278, 0.0073284, 0.0103961, -0.0015621, 0.0015099
6: -0.0011059, -0.0002933, -0.0010978, -0.0003192, -0.0003832, 0.0003965
7: -0.0059988, -0.0038965, -0.0059780, -0.0039635, -0.0009915, 0.0010258
8: -0.0027189, -0.0016133, -0.0027079, -0.0016485, -0.0005214, 0.0005394
9: 0.0000068, 0.0012888, 0.0000477, 0.0012761, -0.0006255, 0.0006046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006323
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906147, 0.9925582, 0.9906147, 0.9925582, -0.0008813, 0.0008813
1: -0.0036025, -0.0031183, -0.0036025, -0.0031183, -0.0002196, 0.0002196
2: 0.0064711, 0.0090375, 0.0064711, 0.0090375, -0.0011638, 0.0011638
3: -0.0053866, -0.0042185, -0.0053866, -0.0042185, -0.0005297, 0.0005297
4: 0.0017804, 0.0022771, 0.0017804, 0.0022771, -0.0002252, 0.0002252
5: 0.0070984, 0.0103262, 0.0070984, 0.0103262, -0.0014637, 0.0014637
6: -0.0010801, -0.0002608, -0.0010801, -0.0002608, -0.0003715, 0.0003715
7: -0.0059321, -0.0038124, -0.0059321, -0.0038124, -0.0009612, 0.0009612
8: -0.0026838, -0.0015691, -0.0026838, -0.0015691, -0.0005055, 0.0005055
9: -0.0000444, 0.0012481, -0.0000444, 0.0012481, -0.0005861, 0.0005861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906147, 0.9925582, 0.9905536, 0.9924812, -0.0008716, 0.0010198
1: -0.0036025, -0.0031183, -0.0036178, -0.0031374, -0.0002172, 0.0002541
2: 0.0064711, 0.0090375, 0.0065728, 0.0091183, -0.0013467, 0.0011509
3: -0.0053866, -0.0042185, -0.0054234, -0.0042648, -0.0005238, 0.0006130
4: 0.0017804, 0.0022771, 0.0018000, 0.0022927, -0.0002606, 0.0002228
5: 0.0070984, 0.0103262, 0.0072263, 0.0104278, -0.0016938, 0.0014475
6: -0.0010801, -0.0002608, -0.0011059, -0.0002933, -0.0003674, 0.0004299
7: -0.0059321, -0.0038124, -0.0059988, -0.0038965, -0.0009506, 0.0011123
8: -0.0026838, -0.0015691, -0.0027189, -0.0016133, -0.0004999, 0.0005849
9: -0.0000444, 0.0012481, 0.0000068, 0.0012888, -0.0006783, 0.0005796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905536, 0.9924812, 0.9906147, 0.9925582, -0.0010198, 0.0008716
1: -0.0036178, -0.0031374, -0.0036025, -0.0031183, -0.0002541, 0.0002172
2: 0.0065728, 0.0091183, 0.0064711, 0.0090375, -0.0011509, 0.0013467
3: -0.0054234, -0.0042648, -0.0053866, -0.0042185, -0.0006130, 0.0005238
4: 0.0018000, 0.0022927, 0.0017804, 0.0022771, -0.0002228, 0.0002606
5: 0.0072263, 0.0104278, 0.0070984, 0.0103262, -0.0014475, 0.0016938
6: -0.0011059, -0.0002933, -0.0010801, -0.0002608, -0.0004299, 0.0003674
7: -0.0059988, -0.0038965, -0.0059321, -0.0038124, -0.0011123, 0.0009506
8: -0.0027189, -0.0016133, -0.0026838, -0.0015691, -0.0005849, 0.0004999
9: 0.0000068, 0.0012888, -0.0000444, 0.0012481, -0.0005796, 0.0006783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006322
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905536, 0.9924812, 0.9905536, 0.9924812, -0.0008866, 0.0008866
1: -0.0036178, -0.0031374, -0.0036178, -0.0031374, -0.0002209, 0.0002209
2: 0.0065728, 0.0091183, 0.0065728, 0.0091183, -0.0011707, 0.0011707
3: -0.0054234, -0.0042648, -0.0054234, -0.0042648, -0.0005329, 0.0005329
4: 0.0018000, 0.0022927, 0.0018000, 0.0022927, -0.0002266, 0.0002266
5: 0.0072263, 0.0104278, 0.0072263, 0.0104278, -0.0014725, 0.0014725
6: -0.0011059, -0.0002933, -0.0011059, -0.0002933, -0.0003737, 0.0003737
7: -0.0059988, -0.0038965, -0.0059988, -0.0038965, -0.0009669, 0.0009669
8: -0.0027189, -0.0016133, -0.0027189, -0.0016133, -0.0005085, 0.0005085
9: 0.0000068, 0.0012888, 0.0000068, 0.0012888, -0.0005896, 0.0005896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006322
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0006080
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0006080
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0006080
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0006080
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0005998
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0005998
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005972, upper bound: 0.0005998
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006186, upper bound: 0.0005998
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005883, upper bound: 0.0006079
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006128, upper bound: 0.0006079
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005883, upper bound: 0.0006079
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006128, upper bound: 0.0006079
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005884, upper bound: 0.0005998
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0005998
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005884, upper bound: 0.0005998
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006127, upper bound: 0.0005998
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006322
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006323
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006347
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0006348
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006322
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0005778, upper bound: 0.0006320
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0006000, upper bound: 0.0006322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906738, 0.9925048, 0.9906433, 0.9925026, -0.0008211, 0.0008533
1: -0.0035878, -0.0031315, -0.0035954, -0.0031321, -0.0002046, 0.0002126
2: 0.0065415, 0.0089594, 0.0065445, 0.0089998, -0.0011268, 0.0010843
3: -0.0053511, -0.0042505, -0.0053694, -0.0042519, -0.0004935, 0.0005129
4: 0.0017940, 0.0022620, 0.0017946, 0.0022698, -0.0002181, 0.0002099
5: 0.0071869, 0.0102281, 0.0071907, 0.0102788, -0.0014172, 0.0013637
6: -0.0010552, -0.0002833, -0.0010680, -0.0002842, -0.0003461, 0.0003597
7: -0.0058676, -0.0038706, -0.0059009, -0.0038730, -0.0008956, 0.0009307
8: -0.0026499, -0.0015996, -0.0026674, -0.0016009, -0.0004710, 0.0004894
9: -0.0000090, 0.0012088, -0.0000075, 0.0012291, -0.0005675, 0.0005461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005849, upper bound: 0.0006035
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005838, upper bound: 0.0006042
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906431, 0.9925026, 0.9906299, 0.9925026, -0.0008319, 0.0008616
1: -0.0035954, -0.0031321, -0.0035987, -0.0031321, -0.0002073, 0.0002147
2: 0.0065445, 0.0089998, 0.0065445, 0.0090173, -0.0011377, 0.0010985
3: -0.0053694, -0.0042519, -0.0053774, -0.0042519, -0.0005000, 0.0005178
4: 0.0017946, 0.0022698, 0.0017946, 0.0022732, -0.0002202, 0.0002126
5: 0.0071907, 0.0102789, 0.0071907, 0.0103009, -0.0014309, 0.0013817
6: -0.0010681, -0.0002842, -0.0010736, -0.0002842, -0.0003507, 0.0003632
7: -0.0059010, -0.0038730, -0.0059155, -0.0038730, -0.0009073, 0.0009397
8: -0.0026674, -0.0016009, -0.0026750, -0.0016009, -0.0004771, 0.0004942
9: -0.0000075, 0.0012292, -0.0000075, 0.0012380, -0.0005730, 0.0005533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0006035
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006042, upper bound: 0.0006042
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906738, 0.9925048, 0.9906310, 0.9925582, -0.0008982, 0.0008977
1: -0.0035878, -0.0031315, -0.0035984, -0.0031183, -0.0002238, 0.0002237
2: 0.0065415, 0.0089594, 0.0064711, 0.0090159, -0.0011855, 0.0011860
3: -0.0053511, -0.0042505, -0.0053768, -0.0042185, -0.0005398, 0.0005396
4: 0.0017940, 0.0022620, 0.0017804, 0.0022729, -0.0002294, 0.0002295
5: 0.0071869, 0.0102281, 0.0070984, 0.0102990, -0.0014910, 0.0014917
6: -0.0010552, -0.0002833, -0.0010732, -0.0002608, -0.0003786, 0.0003784
7: -0.0058676, -0.0038706, -0.0059143, -0.0038124, -0.0009796, 0.0009791
8: -0.0026499, -0.0015996, -0.0026744, -0.0015691, -0.0005151, 0.0005149
9: -0.0000090, 0.0012088, -0.0000444, 0.0012372, -0.0005971, 0.0005973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006026, upper bound: 0.0005924
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0005930
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906431, 0.9925026, 0.9906175, 0.9925582, -0.0009090, 0.0009053
1: -0.0035954, -0.0031321, -0.0036018, -0.0031183, -0.0002265, 0.0002256
2: 0.0065445, 0.0089998, 0.0064711, 0.0090338, -0.0011955, 0.0012003
3: -0.0053694, -0.0042519, -0.0053849, -0.0042185, -0.0005463, 0.0005441
4: 0.0017946, 0.0022698, 0.0017804, 0.0022764, -0.0002314, 0.0002323
5: 0.0071907, 0.0102789, 0.0070984, 0.0103216, -0.0015036, 0.0015096
6: -0.0010681, -0.0002842, -0.0010789, -0.0002608, -0.0003832, 0.0003816
7: -0.0059010, -0.0038730, -0.0059291, -0.0038124, -0.0009913, 0.0009874
8: -0.0026674, -0.0016009, -0.0026822, -0.0015691, -0.0005213, 0.0005193
9: -0.0000075, 0.0012292, -0.0000444, 0.0012463, -0.0006021, 0.0006045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006227, upper bound: 0.0005923
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006213, upper bound: 0.0005931
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906197, 0.9924191, 0.9906433, 0.9925026, -0.0009595, 0.0008395
1: -0.0036013, -0.0031529, -0.0035954, -0.0031321, -0.0002391, 0.0002092
2: 0.0066548, 0.0090310, 0.0065445, 0.0089998, -0.0011085, 0.0012670
3: -0.0053836, -0.0043021, -0.0053694, -0.0042519, -0.0005767, 0.0005046
4: 0.0018159, 0.0022758, 0.0017946, 0.0022698, -0.0002146, 0.0002452
5: 0.0073295, 0.0103180, 0.0071907, 0.0102788, -0.0013943, 0.0015935
6: -0.0010780, -0.0003195, -0.0010680, -0.0002842, -0.0004045, 0.0003539
7: -0.0059267, -0.0039642, -0.0059009, -0.0038730, -0.0010465, 0.0009156
8: -0.0026809, -0.0016489, -0.0026674, -0.0016009, -0.0005503, 0.0004815
9: 0.0000481, 0.0012448, -0.0000075, 0.0012291, -0.0005583, 0.0006381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005843, upper bound: 0.0005979
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005838, upper bound: 0.0005985
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905881, 0.9924197, 0.9906299, 0.9925026, -0.0009707, 0.0008517
1: -0.0036091, -0.0031528, -0.0035987, -0.0031321, -0.0002419, 0.0002122
2: 0.0066539, 0.0090726, 0.0065445, 0.0090173, -0.0011246, 0.0012818
3: -0.0054026, -0.0043017, -0.0053774, -0.0042519, -0.0005834, 0.0005119
4: 0.0018157, 0.0022839, 0.0017946, 0.0022732, -0.0002177, 0.0002481
5: 0.0073284, 0.0103704, 0.0071907, 0.0103009, -0.0014145, 0.0016121
6: -0.0010913, -0.0003192, -0.0010736, -0.0002842, -0.0004092, 0.0003590
7: -0.0059611, -0.0039635, -0.0059155, -0.0038730, -0.0010587, 0.0009289
8: -0.0026990, -0.0016485, -0.0026750, -0.0016009, -0.0005567, 0.0004885
9: 0.0000477, 0.0012658, -0.0000075, 0.0012380, -0.0005664, 0.0006456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006061, upper bound: 0.0005982
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006038, upper bound: 0.0005989
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906197, 0.9924191, 0.9906310, 0.9925582, -0.0010365, 0.0008839
1: -0.0036013, -0.0031529, -0.0035984, -0.0031183, -0.0002583, 0.0002203
2: 0.0066548, 0.0090310, 0.0064711, 0.0090159, -0.0011672, 0.0013687
3: -0.0053836, -0.0043021, -0.0053768, -0.0042185, -0.0006230, 0.0005313
4: 0.0018159, 0.0022758, 0.0017804, 0.0022729, -0.0002259, 0.0002649
5: 0.0073295, 0.0103180, 0.0070984, 0.0102990, -0.0014681, 0.0017215
6: -0.0010780, -0.0003195, -0.0010732, -0.0002608, -0.0004369, 0.0003726
7: -0.0059267, -0.0039642, -0.0059143, -0.0038124, -0.0011305, 0.0009641
8: -0.0026809, -0.0016489, -0.0026744, -0.0015691, -0.0005945, 0.0005070
9: 0.0000481, 0.0012448, -0.0000444, 0.0012372, -0.0005879, 0.0006894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006016, upper bound: 0.0005863
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006021, upper bound: 0.0005871
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905881, 0.9924197, 0.9906175, 0.9925582, -0.0010477, 0.0008955
1: -0.0036091, -0.0031528, -0.0036018, -0.0031183, -0.0002611, 0.0002231
2: 0.0066539, 0.0090726, 0.0064711, 0.0090338, -0.0011825, 0.0013835
3: -0.0054026, -0.0043017, -0.0053849, -0.0042185, -0.0006297, 0.0005382
4: 0.0018157, 0.0022839, 0.0017804, 0.0022764, -0.0002289, 0.0002678
5: 0.0073284, 0.0103704, 0.0070984, 0.0103216, -0.0014872, 0.0017401
6: -0.0010913, -0.0003192, -0.0010789, -0.0002608, -0.0004417, 0.0003775
7: -0.0059611, -0.0039635, -0.0059291, -0.0038124, -0.0011427, 0.0009766
8: -0.0026990, -0.0016485, -0.0026822, -0.0015691, -0.0006009, 0.0005136
9: 0.0000477, 0.0012658, -0.0000444, 0.0012463, -0.0005955, 0.0006968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006214, upper bound: 0.0005865
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006203, upper bound: 0.0005872
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906738, 0.9925048, 0.9905887, 0.9924197, -0.0008113, 0.0009924
1: -0.0035878, -0.0031315, -0.0036090, -0.0031528, -0.0002021, 0.0002473
2: 0.0065415, 0.0089594, 0.0066539, 0.0090718, -0.0013105, 0.0010713
3: -0.0053511, -0.0042505, -0.0054022, -0.0043017, -0.0004876, 0.0005965
4: 0.0017940, 0.0022620, 0.0018157, 0.0022837, -0.0002536, 0.0002073
5: 0.0071869, 0.0102281, 0.0073284, 0.0103694, -0.0016482, 0.0013474
6: -0.0010552, -0.0002833, -0.0010910, -0.0003192, -0.0003420, 0.0004183
7: -0.0058676, -0.0038706, -0.0059605, -0.0039635, -0.0008848, 0.0010824
8: -0.0026499, -0.0015996, -0.0026987, -0.0016485, -0.0004653, 0.0005692
9: -0.0000090, 0.0012088, 0.0000477, 0.0012654, -0.0006600, 0.0005395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005734, upper bound: 0.0006022
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0006035
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906431, 0.9925026, 0.9905754, 0.9924197, -0.0008204, 0.0010001
1: -0.0035954, -0.0031321, -0.0036123, -0.0031528, -0.0002044, 0.0002492
2: 0.0065445, 0.0089998, 0.0066539, 0.0090894, -0.0013207, 0.0010833
3: -0.0053694, -0.0042519, -0.0054102, -0.0043017, -0.0004931, 0.0006011
4: 0.0017946, 0.0022698, 0.0018157, 0.0022871, -0.0002556, 0.0002097
5: 0.0071907, 0.0102789, 0.0073284, 0.0103915, -0.0016611, 0.0013625
6: -0.0010681, -0.0002842, -0.0010967, -0.0003192, -0.0003458, 0.0004216
7: -0.0059010, -0.0038730, -0.0059750, -0.0039635, -0.0008947, 0.0010908
8: -0.0026674, -0.0016009, -0.0027063, -0.0016485, -0.0004705, 0.0005736
9: -0.0000075, 0.0012292, 0.0000477, 0.0012743, -0.0006652, 0.0005456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005984, upper bound: 0.0006025
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0006039
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906738, 0.9925048, 0.9905697, 0.9924812, -0.0008655, 0.0010037
1: -0.0035878, -0.0031315, -0.0036138, -0.0031374, -0.0002157, 0.0002501
2: 0.0065415, 0.0089594, 0.0065728, 0.0090970, -0.0013254, 0.0011429
3: -0.0053511, -0.0042505, -0.0054137, -0.0042648, -0.0005202, 0.0006032
4: 0.0017940, 0.0022620, 0.0018000, 0.0022886, -0.0002565, 0.0002212
5: 0.0071869, 0.0102281, 0.0072263, 0.0104011, -0.0016670, 0.0014375
6: -0.0010552, -0.0002833, -0.0010991, -0.0002933, -0.0003649, 0.0004231
7: -0.0058676, -0.0038706, -0.0059813, -0.0038965, -0.0009440, 0.0010947
8: -0.0026499, -0.0015996, -0.0027096, -0.0016133, -0.0004964, 0.0005757
9: -0.0000090, 0.0012088, 0.0000068, 0.0012781, -0.0006675, 0.0005756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005952, upper bound: 0.0005923
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005970, upper bound: 0.0005930
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906431, 0.9925026, 0.9905563, 0.9924812, -0.0008762, 0.0010114
1: -0.0035954, -0.0031321, -0.0036171, -0.0031374, -0.0002183, 0.0002520
2: 0.0065445, 0.0089998, 0.0065728, 0.0091146, -0.0013355, 0.0011570
3: -0.0053694, -0.0042519, -0.0054217, -0.0042648, -0.0005266, 0.0006079
4: 0.0017946, 0.0022698, 0.0018000, 0.0022920, -0.0002585, 0.0002239
5: 0.0071907, 0.0102789, 0.0072263, 0.0104232, -0.0016797, 0.0014552
6: -0.0010681, -0.0002842, -0.0011047, -0.0002933, -0.0003693, 0.0004263
7: -0.0059010, -0.0038730, -0.0059958, -0.0038965, -0.0009556, 0.0011030
8: -0.0026674, -0.0016009, -0.0027173, -0.0016133, -0.0005025, 0.0005801
9: -0.0000075, 0.0012292, 0.0000068, 0.0012870, -0.0006726, 0.0005827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006180, upper bound: 0.0005923
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006188, upper bound: 0.0005930
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906197, 0.9924191, 0.9905887, 0.9924197, -0.0008205, 0.0008528
1: -0.0036013, -0.0031529, -0.0036090, -0.0031528, -0.0002044, 0.0002125
2: 0.0066548, 0.0090310, 0.0066539, 0.0090718, -0.0011261, 0.0010835
3: -0.0053836, -0.0043021, -0.0054022, -0.0043017, -0.0004931, 0.0005125
4: 0.0018159, 0.0022758, 0.0018157, 0.0022837, -0.0002179, 0.0002097
5: 0.0073295, 0.0103180, 0.0073284, 0.0103694, -0.0014163, 0.0013627
6: -0.0010780, -0.0003195, -0.0010910, -0.0003192, -0.0003459, 0.0003595
7: -0.0059267, -0.0039642, -0.0059605, -0.0039635, -0.0008949, 0.0009301
8: -0.0026809, -0.0016489, -0.0026987, -0.0016485, -0.0004706, 0.0004891
9: 0.0000481, 0.0012448, 0.0000477, 0.0012654, -0.0005671, 0.0005457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005736, upper bound: 0.0005979
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0005985
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905881, 0.9924197, 0.9905754, 0.9924197, -0.0008312, 0.0008611
1: -0.0036091, -0.0031528, -0.0036123, -0.0031528, -0.0002071, 0.0002146
2: 0.0066539, 0.0090726, 0.0066539, 0.0090894, -0.0011371, 0.0010976
3: -0.0054026, -0.0043017, -0.0054102, -0.0043017, -0.0004996, 0.0005176
4: 0.0018157, 0.0022839, 0.0018157, 0.0022871, -0.0002201, 0.0002124
5: 0.0073284, 0.0103704, 0.0073284, 0.0103915, -0.0014302, 0.0013805
6: -0.0010913, -0.0003192, -0.0010967, -0.0003192, -0.0003504, 0.0003630
7: -0.0059611, -0.0039635, -0.0059750, -0.0039635, -0.0009066, 0.0009392
8: -0.0026990, -0.0016485, -0.0027063, -0.0016485, -0.0004768, 0.0004939
9: 0.0000477, 0.0012658, 0.0000477, 0.0012743, -0.0005727, 0.0005528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005984, upper bound: 0.0005982
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005985, upper bound: 0.0005990
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906197, 0.9924191, 0.9905697, 0.9924812, -0.0008971, 0.0008985
1: -0.0036013, -0.0031529, -0.0036138, -0.0031374, -0.0002235, 0.0002239
2: 0.0066548, 0.0090310, 0.0065728, 0.0090970, -0.0011864, 0.0011846
3: -0.0053836, -0.0043021, -0.0054137, -0.0042648, -0.0005392, 0.0005400
4: 0.0018159, 0.0022758, 0.0018000, 0.0022886, -0.0002296, 0.0002293
5: 0.0073295, 0.0103180, 0.0072263, 0.0104011, -0.0014922, 0.0014900
6: -0.0010780, -0.0003195, -0.0010991, -0.0002933, -0.0003782, 0.0003787
7: -0.0059267, -0.0039642, -0.0059813, -0.0038965, -0.0009784, 0.0009799
8: -0.0026809, -0.0016489, -0.0027096, -0.0016133, -0.0005145, 0.0005153
9: 0.0000481, 0.0012448, 0.0000068, 0.0012781, -0.0005976, 0.0005966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005952, upper bound: 0.0005863
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005969, upper bound: 0.0005871
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905881, 0.9924197, 0.9905563, 0.9924812, -0.0009083, 0.0009062
1: -0.0036091, -0.0031528, -0.0036171, -0.0031374, -0.0002263, 0.0002258
2: 0.0066539, 0.0090726, 0.0065728, 0.0091146, -0.0011966, 0.0011994
3: -0.0054026, -0.0043017, -0.0054217, -0.0042648, -0.0005459, 0.0005447
4: 0.0018157, 0.0022839, 0.0018000, 0.0022920, -0.0002316, 0.0002321
5: 0.0073284, 0.0103704, 0.0072263, 0.0104232, -0.0015051, 0.0015085
6: -0.0010913, -0.0003192, -0.0011047, -0.0002933, -0.0003829, 0.0003820
7: -0.0059611, -0.0039635, -0.0059958, -0.0038965, -0.0009906, 0.0009884
8: -0.0026990, -0.0016485, -0.0027173, -0.0016133, -0.0005210, 0.0005198
9: 0.0000477, 0.0012658, 0.0000068, 0.0012870, -0.0006027, 0.0006041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006180, upper bound: 0.0005865
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006182, upper bound: 0.0005872
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906631, 0.9925573, 0.9906433, 0.9925026, -0.0008630, 0.0009285
1: -0.0035905, -0.0031185, -0.0035954, -0.0031321, -0.0002150, 0.0002313
2: 0.0064722, 0.0089736, 0.0065445, 0.0089998, -0.0012260, 0.0011396
3: -0.0053575, -0.0042190, -0.0053694, -0.0042519, -0.0005187, 0.0005580
4: 0.0017806, 0.0022647, 0.0017946, 0.0022698, -0.0002373, 0.0002206
5: 0.0070998, 0.0102458, 0.0071907, 0.0102788, -0.0015420, 0.0014334
6: -0.0010597, -0.0002612, -0.0010680, -0.0002842, -0.0003638, 0.0003914
7: -0.0058793, -0.0038134, -0.0059009, -0.0038730, -0.0009413, 0.0010126
8: -0.0026560, -0.0015696, -0.0026674, -0.0016009, -0.0004950, 0.0005325
9: -0.0000439, 0.0012159, -0.0000075, 0.0012291, -0.0006175, 0.0005740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005725, upper bound: 0.0006213
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0006212
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906305, 0.9925582, 0.9906299, 0.9925026, -0.0008760, 0.0009386
1: -0.0035986, -0.0031183, -0.0035987, -0.0031321, -0.0002183, 0.0002339
2: 0.0064711, 0.0090166, 0.0065445, 0.0090173, -0.0012394, 0.0011568
3: -0.0053771, -0.0042185, -0.0053774, -0.0042519, -0.0005265, 0.0005641
4: 0.0017804, 0.0022730, 0.0017946, 0.0022732, -0.0002399, 0.0002239
5: 0.0070984, 0.0103000, 0.0071907, 0.0103009, -0.0015588, 0.0014549
6: -0.0010734, -0.0002608, -0.0010736, -0.0002842, -0.0003693, 0.0003956
7: -0.0059149, -0.0038124, -0.0059155, -0.0038730, -0.0009554, 0.0010237
8: -0.0026747, -0.0015691, -0.0026750, -0.0016009, -0.0005024, 0.0005383
9: -0.0000444, 0.0012376, -0.0000075, 0.0012380, -0.0006242, 0.0005826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005969, upper bound: 0.0006213
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005931, upper bound: 0.0006213
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906631, 0.9925573, 0.9905887, 0.9924197, -0.0008532, 0.0010676
1: -0.0035905, -0.0031185, -0.0036090, -0.0031528, -0.0002126, 0.0002660
2: 0.0064722, 0.0089736, 0.0066539, 0.0090718, -0.0014097, 0.0011266
3: -0.0053575, -0.0042190, -0.0054022, -0.0043017, -0.0005128, 0.0006416
4: 0.0017806, 0.0022647, 0.0018157, 0.0022837, -0.0002728, 0.0002181
5: 0.0070998, 0.0102458, 0.0073284, 0.0103694, -0.0017730, 0.0014170
6: -0.0010597, -0.0002612, -0.0010910, -0.0003192, -0.0003596, 0.0004500
7: -0.0058793, -0.0038134, -0.0059605, -0.0039635, -0.0009305, 0.0011643
8: -0.0026560, -0.0015696, -0.0026987, -0.0016485, -0.0004893, 0.0006123
9: -0.0000439, 0.0012159, 0.0000477, 0.0012654, -0.0007100, 0.0005674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0006192
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006201
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906305, 0.9925582, 0.9905754, 0.9924197, -0.0008645, 0.0010772
1: -0.0035986, -0.0031183, -0.0036123, -0.0031528, -0.0002154, 0.0002684
2: 0.0064711, 0.0090166, 0.0066539, 0.0090894, -0.0014224, 0.0011415
3: -0.0053771, -0.0042185, -0.0054102, -0.0043017, -0.0005196, 0.0006474
4: 0.0017804, 0.0022730, 0.0018157, 0.0022871, -0.0002753, 0.0002209
5: 0.0070984, 0.0103000, 0.0073284, 0.0103915, -0.0017890, 0.0014357
6: -0.0010734, -0.0002608, -0.0010967, -0.0003192, -0.0003644, 0.0004541
7: -0.0059149, -0.0038124, -0.0059750, -0.0039635, -0.0009428, 0.0011748
8: -0.0026747, -0.0015691, -0.0027063, -0.0016485, -0.0004958, 0.0006178
9: -0.0000444, 0.0012376, 0.0000477, 0.0012743, -0.0007164, 0.0005749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0006193
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005872, upper bound: 0.0006203
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906010, 0.9924784, 0.9906433, 0.9925026, -0.0009689, 0.0008927
1: -0.0036059, -0.0031381, -0.0035954, -0.0031321, -0.0002414, 0.0002224
2: 0.0065765, 0.0090556, 0.0065445, 0.0089998, -0.0011787, 0.0012795
3: -0.0053949, -0.0042665, -0.0053694, -0.0042519, -0.0005824, 0.0005365
4: 0.0018008, 0.0022806, 0.0017946, 0.0022698, -0.0002281, 0.0002476
5: 0.0072309, 0.0103491, 0.0071907, 0.0102788, -0.0014825, 0.0016092
6: -0.0010859, -0.0002945, -0.0010680, -0.0002842, -0.0004084, 0.0003763
7: -0.0059471, -0.0038995, -0.0059009, -0.0038730, -0.0010568, 0.0009736
8: -0.0026917, -0.0016148, -0.0026674, -0.0016009, -0.0005557, 0.0005120
9: 0.0000086, 0.0012573, -0.0000075, 0.0012291, -0.0005937, 0.0006444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005724, upper bound: 0.0006180
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0006187
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905692, 0.9924812, 0.9906299, 0.9925026, -0.0009818, 0.0009060
1: -0.0036139, -0.0031374, -0.0035987, -0.0031321, -0.0002446, 0.0002257
2: 0.0065728, 0.0090976, 0.0065445, 0.0090173, -0.0011963, 0.0012964
3: -0.0054140, -0.0042648, -0.0053774, -0.0042519, -0.0005901, 0.0005445
4: 0.0018000, 0.0022887, 0.0017946, 0.0022732, -0.0002315, 0.0002509
5: 0.0072263, 0.0104019, 0.0071907, 0.0103009, -0.0015046, 0.0016306
6: -0.0010993, -0.0002933, -0.0010736, -0.0002842, -0.0004139, 0.0003819
7: -0.0059818, -0.0038965, -0.0059155, -0.0038730, -0.0010708, 0.0009881
8: -0.0027099, -0.0016133, -0.0026750, -0.0016009, -0.0005631, 0.0005196
9: 0.0000068, 0.0012784, -0.0000075, 0.0012380, -0.0006025, 0.0006530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005968, upper bound: 0.0006183
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005930, upper bound: 0.0006189
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906010, 0.9924784, 0.9905887, 0.9924197, -0.0008635, 0.0009285
1: -0.0036059, -0.0031381, -0.0036090, -0.0031528, -0.0002151, 0.0002314
2: 0.0065765, 0.0090556, 0.0066539, 0.0090718, -0.0012261, 0.0011402
3: -0.0053949, -0.0042665, -0.0054022, -0.0043017, -0.0005190, 0.0005581
4: 0.0018008, 0.0022806, 0.0018157, 0.0022837, -0.0002373, 0.0002207
5: 0.0072309, 0.0103491, 0.0073284, 0.0103694, -0.0015421, 0.0014340
6: -0.0010859, -0.0002945, -0.0010910, -0.0003192, -0.0003640, 0.0003914
7: -0.0059471, -0.0038995, -0.0059605, -0.0039635, -0.0009417, 0.0010127
8: -0.0026917, -0.0016148, -0.0026987, -0.0016485, -0.0004952, 0.0005325
9: 0.0000086, 0.0012573, 0.0000477, 0.0012654, -0.0006175, 0.0005743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005648, upper bound: 0.0006180
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006186
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905692, 0.9924812, 0.9905754, 0.9924197, -0.0008765, 0.0009377
1: -0.0036139, -0.0031374, -0.0036123, -0.0031528, -0.0002184, 0.0002337
2: 0.0065728, 0.0090976, 0.0066539, 0.0090894, -0.0012383, 0.0011574
3: -0.0054140, -0.0042648, -0.0054102, -0.0043017, -0.0005268, 0.0005636
4: 0.0018000, 0.0022887, 0.0018157, 0.0022871, -0.0002397, 0.0002240
5: 0.0072263, 0.0104019, 0.0073284, 0.0103915, -0.0015574, 0.0014557
6: -0.0010993, -0.0002933, -0.0010967, -0.0003192, -0.0003695, 0.0003953
7: -0.0059818, -0.0038965, -0.0059750, -0.0039635, -0.0009559, 0.0010227
8: -0.0027099, -0.0016133, -0.0027063, -0.0016485, -0.0005027, 0.0005379
9: 0.0000068, 0.0012784, 0.0000477, 0.0012743, -0.0006237, 0.0005829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006183
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005873, upper bound: 0.0006189
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906631, 0.9925573, 0.9906310, 0.9925582, -0.0008380, 0.0008705
1: -0.0035905, -0.0031185, -0.0035984, -0.0031183, -0.0002088, 0.0002169
2: 0.0064722, 0.0089736, 0.0064711, 0.0090159, -0.0011495, 0.0011065
3: -0.0053575, -0.0042190, -0.0053768, -0.0042185, -0.0005036, 0.0005232
4: 0.0017806, 0.0022647, 0.0017804, 0.0022729, -0.0002225, 0.0002142
5: 0.0070998, 0.0102458, 0.0070984, 0.0102990, -0.0014457, 0.0013917
6: -0.0010597, -0.0002612, -0.0010732, -0.0002608, -0.0003532, 0.0003669
7: -0.0058793, -0.0038134, -0.0059143, -0.0038124, -0.0009139, 0.0009494
8: -0.0026560, -0.0015696, -0.0026744, -0.0015691, -0.0004806, 0.0004993
9: -0.0000439, 0.0012159, -0.0000444, 0.0012372, -0.0005789, 0.0005573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005725, upper bound: 0.0006212
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005696, upper bound: 0.0006212
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906305, 0.9925582, 0.9906175, 0.9925582, -0.0008487, 0.0008785
1: -0.0035986, -0.0031183, -0.0036018, -0.0031183, -0.0002115, 0.0002189
2: 0.0064711, 0.0090166, 0.0064711, 0.0090338, -0.0011601, 0.0011207
3: -0.0053771, -0.0042185, -0.0053849, -0.0042185, -0.0005101, 0.0005280
4: 0.0017804, 0.0022730, 0.0017804, 0.0022764, -0.0002245, 0.0002169
5: 0.0070984, 0.0103000, 0.0070984, 0.0103216, -0.0014590, 0.0014096
6: -0.0010734, -0.0002608, -0.0010789, -0.0002608, -0.0003578, 0.0003703
7: -0.0059149, -0.0038124, -0.0059291, -0.0038124, -0.0009257, 0.0009581
8: -0.0026747, -0.0015691, -0.0026822, -0.0015691, -0.0004868, 0.0005039
9: -0.0000444, 0.0012376, -0.0000444, 0.0012463, -0.0005843, 0.0005645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005969, upper bound: 0.0006212
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005931, upper bound: 0.0006212
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906631, 0.9925573, 0.9905697, 0.9924812, -0.0008282, 0.0010096
1: -0.0035905, -0.0031185, -0.0036138, -0.0031374, -0.0002064, 0.0002516
2: 0.0064722, 0.0089736, 0.0065728, 0.0090970, -0.0013332, 0.0010936
3: -0.0053575, -0.0042190, -0.0054137, -0.0042648, -0.0004978, 0.0006068
4: 0.0017806, 0.0022647, 0.0018000, 0.0022886, -0.0002580, 0.0002117
5: 0.0070998, 0.0102458, 0.0072263, 0.0104011, -0.0016768, 0.0013755
6: -0.0010597, -0.0002612, -0.0010991, -0.0002933, -0.0003491, 0.0004256
7: -0.0058793, -0.0038134, -0.0059813, -0.0038965, -0.0009033, 0.0011011
8: -0.0026560, -0.0015696, -0.0027096, -0.0016133, -0.0004750, 0.0005791
9: -0.0000439, 0.0012159, 0.0000068, 0.0012781, -0.0006715, 0.0005508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0006192
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006201
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906305, 0.9925582, 0.9905563, 0.9924812, -0.0008374, 0.0010171
1: -0.0035986, -0.0031183, -0.0036171, -0.0031374, -0.0002087, 0.0002534
2: 0.0064711, 0.0090166, 0.0065728, 0.0091146, -0.0013431, 0.0011058
3: -0.0053771, -0.0042185, -0.0054217, -0.0042648, -0.0005033, 0.0006113
4: 0.0017804, 0.0022730, 0.0018000, 0.0022920, -0.0002600, 0.0002140
5: 0.0070984, 0.0103000, 0.0072263, 0.0104232, -0.0016893, 0.0013908
6: -0.0010734, -0.0002608, -0.0011047, -0.0002933, -0.0003530, 0.0004288
7: -0.0059149, -0.0038124, -0.0059958, -0.0038965, -0.0009133, 0.0011093
8: -0.0026747, -0.0015691, -0.0027173, -0.0016133, -0.0004803, 0.0005834
9: -0.0000444, 0.0012376, 0.0000068, 0.0012870, -0.0006765, 0.0005570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0006193
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005872, upper bound: 0.0006203
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906010, 0.9924784, 0.9906310, 0.9925582, -0.0009765, 0.0008570
1: -0.0036059, -0.0031381, -0.0035984, -0.0031183, -0.0002433, 0.0002135
2: 0.0065765, 0.0090556, 0.0064711, 0.0090159, -0.0011316, 0.0012895
3: -0.0053949, -0.0042665, -0.0053768, -0.0042185, -0.0005869, 0.0005151
4: 0.0018008, 0.0022806, 0.0017804, 0.0022729, -0.0002190, 0.0002496
5: 0.0072309, 0.0103491, 0.0070984, 0.0102990, -0.0014233, 0.0016218
6: -0.0010859, -0.0002945, -0.0010732, -0.0002608, -0.0004116, 0.0003612
7: -0.0059471, -0.0038995, -0.0059143, -0.0038124, -0.0010650, 0.0009347
8: -0.0026917, -0.0016148, -0.0026744, -0.0015691, -0.0005601, 0.0004915
9: 0.0000086, 0.0012573, -0.0000444, 0.0012372, -0.0005699, 0.0006495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005724, upper bound: 0.0006180
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0006186
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905692, 0.9924812, 0.9906175, 0.9925582, -0.0009875, 0.0008687
1: -0.0036139, -0.0031374, -0.0036018, -0.0031183, -0.0002461, 0.0002165
2: 0.0065728, 0.0090976, 0.0064711, 0.0090338, -0.0011472, 0.0013040
3: -0.0054140, -0.0042648, -0.0053849, -0.0042185, -0.0005935, 0.0005221
4: 0.0018000, 0.0022887, 0.0017804, 0.0022764, -0.0002220, 0.0002524
5: 0.0072263, 0.0104019, 0.0070984, 0.0103216, -0.0014428, 0.0016401
6: -0.0010993, -0.0002933, -0.0010789, -0.0002608, -0.0004163, 0.0003662
7: -0.0059818, -0.0038965, -0.0059291, -0.0038124, -0.0010770, 0.0009475
8: -0.0027099, -0.0016133, -0.0026822, -0.0015691, -0.0005664, 0.0004983
9: 0.0000068, 0.0012784, -0.0000444, 0.0012463, -0.0005778, 0.0006567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005968, upper bound: 0.0006183
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005930, upper bound: 0.0006189
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906010, 0.9924784, 0.9905697, 0.9924812, -0.0008430, 0.0008755
1: -0.0036059, -0.0031381, -0.0036138, -0.0031374, -0.0002101, 0.0002182
2: 0.0065765, 0.0090556, 0.0065728, 0.0090970, -0.0011561, 0.0011132
3: -0.0053949, -0.0042665, -0.0054137, -0.0042648, -0.0005067, 0.0005262
4: 0.0018008, 0.0022806, 0.0018000, 0.0022886, -0.0002238, 0.0002155
5: 0.0072309, 0.0103491, 0.0072263, 0.0104011, -0.0014541, 0.0014001
6: -0.0010859, -0.0002945, -0.0010991, -0.0002933, -0.0003554, 0.0003691
7: -0.0059471, -0.0038995, -0.0059813, -0.0038965, -0.0009194, 0.0009549
8: -0.0026917, -0.0016148, -0.0027096, -0.0016133, -0.0004835, 0.0005022
9: 0.0000086, 0.0012573, 0.0000068, 0.0012781, -0.0005823, 0.0005607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005649, upper bound: 0.0006180
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006186
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905692, 0.9924812, 0.9905563, 0.9924812, -0.0008533, 0.0008838
1: -0.0036139, -0.0031374, -0.0036171, -0.0031374, -0.0002126, 0.0002202
2: 0.0065728, 0.0090976, 0.0065728, 0.0091146, -0.0011671, 0.0011268
3: -0.0054140, -0.0042648, -0.0054217, -0.0042648, -0.0005129, 0.0005312
4: 0.0018000, 0.0022887, 0.0018000, 0.0022920, -0.0002259, 0.0002181
5: 0.0072263, 0.0104019, 0.0072263, 0.0104232, -0.0014679, 0.0014173
6: -0.0010993, -0.0002933, -0.0011047, -0.0002933, -0.0003597, 0.0003726
7: -0.0059818, -0.0038965, -0.0059958, -0.0038965, -0.0009307, 0.0009639
8: -0.0027099, -0.0016133, -0.0027173, -0.0016133, -0.0004894, 0.0005069
9: 0.0000068, 0.0012784, 0.0000068, 0.0012870, -0.0005878, 0.0005675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005863, upper bound: 0.0006183
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005873, upper bound: 0.0006189
time: 0.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005849, upper bound: 0.0006035
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005838, upper bound: 0.0006042
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006072, upper bound: 0.0006035
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006042, upper bound: 0.0006042
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006026, upper bound: 0.0005924
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0005930
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006227, upper bound: 0.0005923
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006213, upper bound: 0.0005931
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005843, upper bound: 0.0005979
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005838, upper bound: 0.0005985
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006061, upper bound: 0.0005982
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006038, upper bound: 0.0005989
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006016, upper bound: 0.0005863
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006021, upper bound: 0.0005871
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006214, upper bound: 0.0005865
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006203, upper bound: 0.0005872
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005734, upper bound: 0.0006022
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0006035
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005984, upper bound: 0.0006025
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0006039
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005952, upper bound: 0.0005923
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005970, upper bound: 0.0005930
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006180, upper bound: 0.0005923
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006188, upper bound: 0.0005930
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005736, upper bound: 0.0005979
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0005985
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005984, upper bound: 0.0005982
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005985, upper bound: 0.0005990
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005952, upper bound: 0.0005863
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005969, upper bound: 0.0005871
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006180, upper bound: 0.0005865
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0006182, upper bound: 0.0005872
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005725, upper bound: 0.0006213
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0006212
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005969, upper bound: 0.0006213
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005931, upper bound: 0.0006213
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0006192
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006201
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0006193
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005872, upper bound: 0.0006203
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005724, upper bound: 0.0006180
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0006187
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005968, upper bound: 0.0006183
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005930, upper bound: 0.0006189
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005648, upper bound: 0.0006180
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006186
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005864, upper bound: 0.0006183
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005873, upper bound: 0.0006189
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005725, upper bound: 0.0006212
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005696, upper bound: 0.0006212
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005969, upper bound: 0.0006212
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005931, upper bound: 0.0006212
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0006192
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006201
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0006193
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005872, upper bound: 0.0006203
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005724, upper bound: 0.0006180
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0006186
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005968, upper bound: 0.0006183
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005930, upper bound: 0.0006189
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005649, upper bound: 0.0006180
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005656, upper bound: 0.0006186
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005863, upper bound: 0.0006183
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -0.0005873, upper bound: 0.0006189

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906748, 0.9924985, 0.9906468, 0.9924810, -0.0007914, 0.0008431
1: -0.0035876, -0.0031331, -0.0035945, -0.0031375, -0.0001972, 0.0002101
2: 0.0065499, 0.0089581, 0.0065731, 0.0089951, -0.0011133, 0.0010451
3: -0.0053505, -0.0042544, -0.0053673, -0.0042649, -0.0004757, 0.0005067
4: 0.0017956, 0.0022617, 0.0018001, 0.0022689, -0.0002155, 0.0002023
5: 0.0071975, 0.0102264, 0.0072266, 0.0102729, -0.0014002, 0.0013144
6: -0.0010547, -0.0002860, -0.0010665, -0.0002934, -0.0003336, 0.0003554
7: -0.0058666, -0.0038775, -0.0058971, -0.0038966, -0.0008631, 0.0009195
8: -0.0026493, -0.0016033, -0.0026654, -0.0016134, -0.0004539, 0.0004835
9: -0.0000047, 0.0012082, 0.0000069, 0.0012268, -0.0005607, 0.0005263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005650, upper bound: 0.0005806
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005671, upper bound: 0.0005841
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906744, 0.9924943, 0.9906429, 0.9924688, -0.0007907, 0.0008779
1: -0.0035877, -0.0031342, -0.0035955, -0.0031405, -0.0001970, 0.0002187
2: 0.0065555, 0.0089587, 0.0065892, 0.0090002, -0.0011592, 0.0010441
3: -0.0053507, -0.0042569, -0.0053696, -0.0042722, -0.0004752, 0.0005276
4: 0.0017967, 0.0022618, 0.0018032, 0.0022699, -0.0002244, 0.0002021
5: 0.0072045, 0.0102271, 0.0072469, 0.0102794, -0.0014580, 0.0013132
6: -0.0010549, -0.0002877, -0.0010682, -0.0002985, -0.0003333, 0.0003700
7: -0.0058670, -0.0038821, -0.0059013, -0.0039100, -0.0008624, 0.0009574
8: -0.0026496, -0.0016057, -0.0026676, -0.0016204, -0.0004535, 0.0005035
9: -0.0000019, 0.0012084, 0.0000150, 0.0012294, -0.0005838, 0.0005259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005642, upper bound: 0.0005808
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005659, upper bound: 0.0005847
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906442, 0.9924966, 0.9906334, 0.9924810, -0.0007995, 0.0008520
1: -0.0035952, -0.0031336, -0.0035979, -0.0031375, -0.0001992, 0.0002123
2: 0.0065524, 0.0089986, 0.0065731, 0.0090128, -0.0011251, 0.0010557
3: -0.0053689, -0.0042555, -0.0053754, -0.0042649, -0.0004805, 0.0005121
4: 0.0017961, 0.0022695, 0.0018001, 0.0022723, -0.0002178, 0.0002043
5: 0.0072007, 0.0102773, 0.0072266, 0.0102952, -0.0014151, 0.0013278
6: -0.0010677, -0.0002868, -0.0010722, -0.0002934, -0.0003370, 0.0003592
7: -0.0059000, -0.0038796, -0.0059117, -0.0038966, -0.0008719, 0.0009293
8: -0.0026669, -0.0016044, -0.0026731, -0.0016134, -0.0004585, 0.0004887
9: -0.0000035, 0.0012285, 0.0000069, 0.0012357, -0.0005667, 0.0005317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005904, upper bound: 0.0005813
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005881, upper bound: 0.0005841
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906438, 0.9924916, 0.9906303, 0.9924688, -0.0008010, 0.0008838
1: -0.0035953, -0.0031348, -0.0035986, -0.0031405, -0.0001996, 0.0002202
2: 0.0065590, 0.0089991, 0.0065892, 0.0090169, -0.0011670, 0.0010578
3: -0.0053691, -0.0042585, -0.0053772, -0.0042722, -0.0004815, 0.0005312
4: 0.0017974, 0.0022697, 0.0018032, 0.0022731, -0.0002259, 0.0002047
5: 0.0072090, 0.0102780, 0.0072469, 0.0103003, -0.0014678, 0.0013304
6: -0.0010678, -0.0002889, -0.0010735, -0.0002985, -0.0003377, 0.0003726
7: -0.0059004, -0.0038851, -0.0059151, -0.0039100, -0.0008737, 0.0009639
8: -0.0026671, -0.0016073, -0.0026748, -0.0016204, -0.0004594, 0.0005069
9: -0.0000002, 0.0012288, 0.0000150, 0.0012377, -0.0005878, 0.0005327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005874, upper bound: 0.0005816
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005847
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906748, 0.9924985, 0.9906345, 0.9925352, -0.0008630, 0.0008867
1: -0.0035876, -0.0031331, -0.0035976, -0.0031240, -0.0002150, 0.0002209
2: 0.0065499, 0.0089581, 0.0065014, 0.0090113, -0.0011709, 0.0011395
3: -0.0053505, -0.0042544, -0.0053747, -0.0042323, -0.0005187, 0.0005329
4: 0.0017956, 0.0022617, 0.0017862, 0.0022720, -0.0002266, 0.0002206
5: 0.0071975, 0.0102264, 0.0071365, 0.0102933, -0.0014727, 0.0014332
6: -0.0010547, -0.0002860, -0.0010717, -0.0002705, -0.0003638, 0.0003738
7: -0.0058666, -0.0038775, -0.0059104, -0.0038375, -0.0009412, 0.0009671
8: -0.0026493, -0.0016033, -0.0026724, -0.0015822, -0.0004950, 0.0005086
9: -0.0000047, 0.0012082, -0.0000292, 0.0012349, -0.0005897, 0.0005739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005823, upper bound: 0.0005744
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005839, upper bound: 0.0005751
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906744, 0.9924943, 0.9906290, 0.9925265, -0.0008674, 0.0009194
1: -0.0035877, -0.0031342, -0.0035990, -0.0031261, -0.0002161, 0.0002291
2: 0.0065555, 0.0089587, 0.0065129, 0.0090186, -0.0012140, 0.0011454
3: -0.0053507, -0.0042569, -0.0053780, -0.0042375, -0.0005214, 0.0005526
4: 0.0017967, 0.0022618, 0.0017885, 0.0022734, -0.0002350, 0.0002217
5: 0.0072045, 0.0102271, 0.0071510, 0.0103025, -0.0015270, 0.0014407
6: -0.0010549, -0.0002877, -0.0010740, -0.0002742, -0.0003657, 0.0003876
7: -0.0058670, -0.0038821, -0.0059165, -0.0038470, -0.0009461, 0.0010027
8: -0.0026496, -0.0016057, -0.0026756, -0.0015872, -0.0004975, 0.0005273
9: -0.0000019, 0.0012084, -0.0000234, 0.0012386, -0.0006115, 0.0005769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005832, upper bound: 0.0005746
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005837, upper bound: 0.0005754
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906442, 0.9924966, 0.9906209, 0.9925352, -0.0008744, 0.0008951
1: -0.0035952, -0.0031336, -0.0036010, -0.0031240, -0.0002179, 0.0002230
2: 0.0065524, 0.0089986, 0.0065014, 0.0090293, -0.0011820, 0.0011546
3: -0.0053689, -0.0042555, -0.0053829, -0.0042323, -0.0005255, 0.0005380
4: 0.0017961, 0.0022695, 0.0017862, 0.0022755, -0.0002288, 0.0002235
5: 0.0072007, 0.0102773, 0.0071365, 0.0103159, -0.0014866, 0.0014522
6: -0.0010677, -0.0002868, -0.0010775, -0.0002705, -0.0003686, 0.0003773
7: -0.0059000, -0.0038796, -0.0059254, -0.0038375, -0.0009536, 0.0009763
8: -0.0026669, -0.0016044, -0.0026802, -0.0015822, -0.0005015, 0.0005134
9: -0.0000035, 0.0012285, -0.0000292, 0.0012440, -0.0005953, 0.0005815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006049, upper bound: 0.0005748
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006032, upper bound: 0.0005751
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906438, 0.9924916, 0.9906155, 0.9925265, -0.0008779, 0.0009260
1: -0.0035953, -0.0031348, -0.0036023, -0.0031261, -0.0002187, 0.0002307
2: 0.0065590, 0.0089991, 0.0065129, 0.0090363, -0.0012228, 0.0011593
3: -0.0053691, -0.0042585, -0.0053861, -0.0042375, -0.0005276, 0.0005565
4: 0.0017974, 0.0022697, 0.0017885, 0.0022768, -0.0002367, 0.0002244
5: 0.0072090, 0.0102780, 0.0071510, 0.0103248, -0.0015379, 0.0014580
6: -0.0010678, -0.0002889, -0.0010797, -0.0002742, -0.0003701, 0.0003903
7: -0.0059004, -0.0038851, -0.0059311, -0.0038470, -0.0009575, 0.0010099
8: -0.0026671, -0.0016073, -0.0026833, -0.0015872, -0.0005035, 0.0005311
9: -0.0000002, 0.0012288, -0.0000234, 0.0012475, -0.0006158, 0.0005839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006038, upper bound: 0.0005752
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006014, upper bound: 0.0005754
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905890, 0.9924133, 0.9906334, 0.9924810, -0.0009377, 0.0008418
1: -0.0036089, -0.0031544, -0.0035979, -0.0031375, -0.0002337, 0.0002097
2: 0.0066625, 0.0090715, 0.0065731, 0.0090128, -0.0011115, 0.0012383
3: -0.0054021, -0.0043056, -0.0053754, -0.0042649, -0.0005636, 0.0005059
4: 0.0018174, 0.0022837, 0.0018001, 0.0022723, -0.0002151, 0.0002397
5: 0.0073391, 0.0103690, 0.0072266, 0.0102952, -0.0013980, 0.0015574
6: -0.0010909, -0.0003219, -0.0010722, -0.0002934, -0.0003953, 0.0003548
7: -0.0059602, -0.0039705, -0.0059117, -0.0038966, -0.0010227, 0.0009181
8: -0.0026986, -0.0016522, -0.0026731, -0.0016134, -0.0005379, 0.0004828
9: 0.0000520, 0.0012653, 0.0000069, 0.0012357, -0.0005598, 0.0006237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005756, upper bound: 0.0005502
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005881, upper bound: 0.0005811
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905887, 0.9924096, 0.9906303, 0.9924688, -0.0009394, 0.0008731
1: -0.0036090, -0.0031553, -0.0035986, -0.0031405, -0.0002341, 0.0002176
2: 0.0066673, 0.0090719, 0.0065892, 0.0090169, -0.0011530, 0.0012405
3: -0.0054022, -0.0043078, -0.0053772, -0.0042722, -0.0005646, 0.0005248
4: 0.0018183, 0.0022837, 0.0018032, 0.0022731, -0.0002232, 0.0002401
5: 0.0073452, 0.0103695, 0.0072469, 0.0103003, -0.0014501, 0.0015603
6: -0.0010911, -0.0003235, -0.0010735, -0.0002985, -0.0003960, 0.0003681
7: -0.0059605, -0.0039745, -0.0059151, -0.0039100, -0.0010246, 0.0009523
8: -0.0026987, -0.0016543, -0.0026748, -0.0016204, -0.0005388, 0.0005008
9: 0.0000544, 0.0012654, 0.0000150, 0.0012377, -0.0005807, 0.0006248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005724, upper bound: 0.0005502
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005811
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906204, 0.9924127, 0.9906345, 0.9925352, -0.0010008, 0.0008728
1: -0.0036011, -0.0031545, -0.0035976, -0.0031240, -0.0002494, 0.0002175
2: 0.0066633, 0.0090298, 0.0065014, 0.0090113, -0.0011526, 0.0013215
3: -0.0053831, -0.0043059, -0.0053747, -0.0042323, -0.0006015, 0.0005246
4: 0.0018175, 0.0022756, 0.0017862, 0.0022720, -0.0002231, 0.0002558
5: 0.0073401, 0.0103166, 0.0071365, 0.0102933, -0.0014496, 0.0016621
6: -0.0010776, -0.0003222, -0.0010717, -0.0002705, -0.0004219, 0.0003679
7: -0.0059258, -0.0039711, -0.0059104, -0.0038375, -0.0010915, 0.0009520
8: -0.0026805, -0.0016525, -0.0026724, -0.0015822, -0.0005740, 0.0005006
9: 0.0000523, 0.0012443, -0.0000292, 0.0012349, -0.0005805, 0.0006656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005562, upper bound: 0.0005339
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005839, upper bound: 0.0005694
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906201, 0.9924093, 0.9906290, 0.9925265, -0.0010055, 0.0009045
1: -0.0036012, -0.0031553, -0.0035990, -0.0031261, -0.0002505, 0.0002254
2: 0.0066677, 0.0090303, 0.0065129, 0.0090186, -0.0011944, 0.0013277
3: -0.0053833, -0.0043079, -0.0053780, -0.0042375, -0.0006043, 0.0005437
4: 0.0018184, 0.0022757, 0.0017885, 0.0022734, -0.0002312, 0.0002570
5: 0.0073456, 0.0103172, 0.0071510, 0.0103025, -0.0015023, 0.0016699
6: -0.0010778, -0.0003236, -0.0010740, -0.0002742, -0.0004238, 0.0003813
7: -0.0059262, -0.0039748, -0.0059165, -0.0038470, -0.0010966, 0.0009865
8: -0.0026807, -0.0016544, -0.0026756, -0.0015872, -0.0005767, 0.0005188
9: 0.0000546, 0.0012445, -0.0000234, 0.0012386, -0.0006016, 0.0006687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005561, upper bound: 0.0005339
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005837, upper bound: 0.0005694
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905890, 0.9924133, 0.9906209, 0.9925352, -0.0010126, 0.0008848
1: -0.0036089, -0.0031544, -0.0036010, -0.0031240, -0.0002523, 0.0002205
2: 0.0066625, 0.0090715, 0.0065014, 0.0090293, -0.0011684, 0.0013372
3: -0.0054021, -0.0043056, -0.0053829, -0.0042323, -0.0006086, 0.0005318
4: 0.0018174, 0.0022837, 0.0017862, 0.0022755, -0.0002261, 0.0002588
5: 0.0073391, 0.0103690, 0.0071365, 0.0103159, -0.0014696, 0.0016818
6: -0.0010909, -0.0003219, -0.0010775, -0.0002705, -0.0004269, 0.0003730
7: -0.0059602, -0.0039705, -0.0059254, -0.0038375, -0.0011044, 0.0009650
8: -0.0026986, -0.0016522, -0.0026802, -0.0015822, -0.0005808, 0.0005075
9: 0.0000520, 0.0012653, -0.0000292, 0.0012440, -0.0005885, 0.0006735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005882, upper bound: 0.0005425
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0005694
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905887, 0.9924096, 0.9906155, 0.9925265, -0.0010163, 0.0009153
1: -0.0036090, -0.0031553, -0.0036023, -0.0031261, -0.0002532, 0.0002281
2: 0.0066673, 0.0090719, 0.0065129, 0.0090363, -0.0012087, 0.0013420
3: -0.0054022, -0.0043078, -0.0053861, -0.0042375, -0.0006108, 0.0005501
4: 0.0018183, 0.0022837, 0.0017885, 0.0022768, -0.0002339, 0.0002597
5: 0.0073452, 0.0103695, 0.0071510, 0.0103248, -0.0015202, 0.0016879
6: -0.0010911, -0.0003235, -0.0010797, -0.0002742, -0.0004284, 0.0003858
7: -0.0059605, -0.0039745, -0.0059311, -0.0038470, -0.0011084, 0.0009983
8: -0.0026987, -0.0016543, -0.0026833, -0.0015872, -0.0005829, 0.0005250
9: 0.0000544, 0.0012654, -0.0000234, 0.0012475, -0.0006088, 0.0006759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005862, upper bound: 0.0005425
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006014, upper bound: 0.0005694
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906748, 0.9924985, 0.9905916, 0.9923964, -0.0007779, 0.0009805
1: -0.0035876, -0.0031331, -0.0036083, -0.0031586, -0.0001938, 0.0002443
2: 0.0065499, 0.0089581, 0.0066848, 0.0090679, -0.0012947, 0.0010273
3: -0.0053505, -0.0042544, -0.0054005, -0.0043157, -0.0004676, 0.0005893
4: 0.0017956, 0.0022617, 0.0018217, 0.0022830, -0.0002506, 0.0001988
5: 0.0071975, 0.0102264, 0.0073671, 0.0103645, -0.0016284, 0.0012920
6: -0.0010547, -0.0002860, -0.0010898, -0.0003290, -0.0003279, 0.0004133
7: -0.0058666, -0.0038775, -0.0059573, -0.0039889, -0.0008485, 0.0010693
8: -0.0026493, -0.0016033, -0.0026970, -0.0016619, -0.0004462, 0.0005623
9: -0.0000047, 0.0012082, 0.0000632, 0.0012635, -0.0006521, 0.0005174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005161, upper bound: 0.0005555
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005572, upper bound: 0.0005840
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906744, 0.9924943, 0.9905849, 0.9923916, -0.0007808, 0.0010142
1: -0.0035877, -0.0031342, -0.0036100, -0.0031598, -0.0001946, 0.0002527
2: 0.0065555, 0.0089587, 0.0066911, 0.0090769, -0.0013393, 0.0010311
3: -0.0053507, -0.0042569, -0.0054045, -0.0043186, -0.0004693, 0.0006096
4: 0.0017967, 0.0022618, 0.0018229, 0.0022847, -0.0002592, 0.0001996
5: 0.0072045, 0.0102271, 0.0073751, 0.0103758, -0.0016845, 0.0012968
6: -0.0010549, -0.0002877, -0.0010927, -0.0003310, -0.0003292, 0.0004275
7: -0.0058670, -0.0038821, -0.0059647, -0.0039941, -0.0008516, 0.0011062
8: -0.0026496, -0.0016057, -0.0027009, -0.0016646, -0.0004479, 0.0005817
9: -0.0000019, 0.0012084, 0.0000664, 0.0012680, -0.0006745, 0.0005193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005200, upper bound: 0.0005555
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005590, upper bound: 0.0005847
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906442, 0.9924966, 0.9905782, 0.9923964, -0.0007860, 0.0009891
1: -0.0035952, -0.0031336, -0.0036116, -0.0031586, -0.0001959, 0.0002465
2: 0.0065524, 0.0089986, 0.0066848, 0.0090856, -0.0013062, 0.0010379
3: -0.0053689, -0.0042555, -0.0054085, -0.0043157, -0.0004724, 0.0005945
4: 0.0017961, 0.0022695, 0.0018217, 0.0022864, -0.0002528, 0.0002009
5: 0.0072007, 0.0102773, 0.0073671, 0.0103868, -0.0016428, 0.0013054
6: -0.0010677, -0.0002868, -0.0010954, -0.0003290, -0.0003313, 0.0004170
7: -0.0059000, -0.0038796, -0.0059718, -0.0039889, -0.0008573, 0.0010788
8: -0.0026669, -0.0016044, -0.0027047, -0.0016619, -0.0004508, 0.0005673
9: -0.0000035, 0.0012285, 0.0000632, 0.0012724, -0.0006578, 0.0005228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005640
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005813, upper bound: 0.0005840
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906438, 0.9924916, 0.9905720, 0.9923916, -0.0007894, 0.0010200
1: -0.0035953, -0.0031348, -0.0036132, -0.0031598, -0.0001967, 0.0002542
2: 0.0065590, 0.0089991, 0.0066911, 0.0090939, -0.0013469, 0.0010424
3: -0.0053691, -0.0042585, -0.0054123, -0.0043186, -0.0004744, 0.0006131
4: 0.0017974, 0.0022697, 0.0018229, 0.0022880, -0.0002607, 0.0002017
5: 0.0072090, 0.0102780, 0.0073751, 0.0103972, -0.0016941, 0.0013110
6: -0.0010678, -0.0002889, -0.0010981, -0.0003310, -0.0003327, 0.0004300
7: -0.0059004, -0.0038851, -0.0059787, -0.0039941, -0.0008609, 0.0011125
8: -0.0026671, -0.0016073, -0.0027083, -0.0016646, -0.0004527, 0.0005850
9: -0.0000002, 0.0012288, 0.0000664, 0.0012765, -0.0006784, 0.0005250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005642
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005811, upper bound: 0.0005847
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906442, 0.9924966, 0.9905592, 0.9924576, -0.0008392, 0.0010007
1: -0.0035952, -0.0031336, -0.0036164, -0.0031433, -0.0002091, 0.0002494
2: 0.0065524, 0.0089986, 0.0066040, 0.0091108, -0.0013215, 0.0011081
3: -0.0053689, -0.0042555, -0.0054200, -0.0042790, -0.0005044, 0.0006015
4: 0.0017961, 0.0022695, 0.0018061, 0.0022913, -0.0002558, 0.0002145
5: 0.0072007, 0.0102773, 0.0072656, 0.0104185, -0.0016621, 0.0013937
6: -0.0010677, -0.0002868, -0.0011035, -0.0003033, -0.0003537, 0.0004218
7: -0.0059000, -0.0038796, -0.0059927, -0.0039222, -0.0009152, 0.0010914
8: -0.0026669, -0.0016044, -0.0027156, -0.0016268, -0.0004813, 0.0005740
9: -0.0000035, 0.0012285, 0.0000225, 0.0012851, -0.0006656, 0.0005581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005737, upper bound: 0.0005566
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006005, upper bound: 0.0005750
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906438, 0.9924916, 0.9905514, 0.9924538, -0.0008449, 0.0010342
1: -0.0035953, -0.0031348, -0.0036183, -0.0031443, -0.0002105, 0.0002577
2: 0.0065590, 0.0089991, 0.0066090, 0.0091212, -0.0013656, 0.0011156
3: -0.0053691, -0.0042585, -0.0054247, -0.0042813, -0.0005078, 0.0006216
4: 0.0017974, 0.0022697, 0.0018071, 0.0022933, -0.0002643, 0.0002159
5: 0.0072090, 0.0102780, 0.0072719, 0.0104315, -0.0017176, 0.0014032
6: -0.0010678, -0.0002889, -0.0011068, -0.0003048, -0.0003561, 0.0004359
7: -0.0059004, -0.0038851, -0.0060012, -0.0039263, -0.0009214, 0.0011279
8: -0.0026671, -0.0016073, -0.0027201, -0.0016290, -0.0004846, 0.0005932
9: -0.0000002, 0.0012288, 0.0000250, 0.0012903, -0.0006878, 0.0005619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005768, upper bound: 0.0005573
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006006, upper bound: 0.0005753
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905887, 0.9924096, 0.9905720, 0.9923916, -0.0008024, 0.0008804
1: -0.0036090, -0.0031553, -0.0036132, -0.0031598, -0.0001999, 0.0002194
2: 0.0066673, 0.0090719, 0.0066911, 0.0090939, -0.0011626, 0.0010596
3: -0.0054022, -0.0043078, -0.0054123, -0.0043186, -0.0004823, 0.0005292
4: 0.0018183, 0.0022837, 0.0018229, 0.0022880, -0.0002250, 0.0002051
5: 0.0073452, 0.0103695, 0.0073751, 0.0103972, -0.0014622, 0.0013326
6: -0.0010911, -0.0003235, -0.0010981, -0.0003310, -0.0003382, 0.0003711
7: -0.0059605, -0.0039745, -0.0059787, -0.0039941, -0.0008751, 0.0009602
8: -0.0026987, -0.0016543, -0.0027083, -0.0016646, -0.0004602, 0.0005050
9: 0.0000544, 0.0012654, 0.0000664, 0.0012765, -0.0005855, 0.0005336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005502
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005802, upper bound: 0.0005811
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905890, 0.9924133, 0.9905592, 0.9924576, -0.0008747, 0.0008964
1: -0.0036089, -0.0031544, -0.0036164, -0.0031433, -0.0002180, 0.0002234
2: 0.0066625, 0.0090715, 0.0066040, 0.0091108, -0.0011837, 0.0011551
3: -0.0054021, -0.0043056, -0.0054200, -0.0042790, -0.0005257, 0.0005388
4: 0.0018174, 0.0022837, 0.0018061, 0.0022913, -0.0002291, 0.0002236
5: 0.0073391, 0.0103690, 0.0072656, 0.0104185, -0.0014887, 0.0014528
6: -0.0010909, -0.0003219, -0.0011035, -0.0003033, -0.0003687, 0.0003779
7: -0.0059602, -0.0039705, -0.0059927, -0.0039222, -0.0009540, 0.0009776
8: -0.0026986, -0.0016522, -0.0027156, -0.0016268, -0.0005017, 0.0005141
9: 0.0000520, 0.0012653, 0.0000225, 0.0012851, -0.0005962, 0.0005818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005737, upper bound: 0.0005425
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006006, upper bound: 0.0005694
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905887, 0.9924096, 0.9905514, 0.9924538, -0.0008791, 0.0009247
1: -0.0036090, -0.0031553, -0.0036183, -0.0031443, -0.0002190, 0.0002304
2: 0.0066673, 0.0090719, 0.0066090, 0.0091212, -0.0012211, 0.0011608
3: -0.0054022, -0.0043078, -0.0054247, -0.0042813, -0.0005283, 0.0005558
4: 0.0018183, 0.0022837, 0.0018071, 0.0022933, -0.0002363, 0.0002247
5: 0.0073452, 0.0103695, 0.0072719, 0.0104315, -0.0015358, 0.0014600
6: -0.0010911, -0.0003235, -0.0011068, -0.0003048, -0.0003706, 0.0003898
7: -0.0059605, -0.0039745, -0.0060012, -0.0039263, -0.0009587, 0.0010085
8: -0.0026987, -0.0016543, -0.0027201, -0.0016290, -0.0005042, 0.0005304
9: 0.0000544, 0.0012654, 0.0000250, 0.0012903, -0.0006150, 0.0005846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005767, upper bound: 0.0005425
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0005694
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925509, 0.9906468, 0.9924810, -0.0008331, 0.0009179
1: -0.0035902, -0.0031201, -0.0035945, -0.0031375, -0.0002076, 0.0002287
2: 0.0064807, 0.0089723, 0.0065731, 0.0089951, -0.0012121, 0.0011001
3: -0.0053569, -0.0042229, -0.0053673, -0.0042649, -0.0005007, 0.0005517
4: 0.0017822, 0.0022645, 0.0018001, 0.0022689, -0.0002346, 0.0002129
5: 0.0071105, 0.0102443, 0.0072266, 0.0102729, -0.0015245, 0.0013836
6: -0.0010593, -0.0002639, -0.0010665, -0.0002934, -0.0003512, 0.0003869
7: -0.0058783, -0.0038204, -0.0058971, -0.0038966, -0.0009086, 0.0010011
8: -0.0026555, -0.0015733, -0.0026654, -0.0016134, -0.0004778, 0.0005265
9: -0.0000396, 0.0012153, 0.0000069, 0.0012268, -0.0006105, 0.0005541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005523, upper bound: 0.0005927
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005563, upper bound: 0.0006009
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906636, 0.9925468, 0.9906429, 0.9924688, -0.0008325, 0.0009522
1: -0.0035903, -0.0031211, -0.0035955, -0.0031405, -0.0002074, 0.0002373
2: 0.0064862, 0.0089728, 0.0065892, 0.0090002, -0.0012574, 0.0010993
3: -0.0053572, -0.0042253, -0.0053696, -0.0042722, -0.0005003, 0.0005723
4: 0.0017833, 0.0022646, 0.0018032, 0.0022699, -0.0002434, 0.0002128
5: 0.0071173, 0.0102449, 0.0072469, 0.0102794, -0.0015814, 0.0013826
6: -0.0010594, -0.0002656, -0.0010682, -0.0002985, -0.0003509, 0.0004014
7: -0.0058787, -0.0038249, -0.0059013, -0.0039100, -0.0009079, 0.0010385
8: -0.0026557, -0.0015756, -0.0026676, -0.0016204, -0.0004775, 0.0005461
9: -0.0000369, 0.0012156, 0.0000150, 0.0012294, -0.0006333, 0.0005536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005494, upper bound: 0.0005930
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005536, upper bound: 0.0006015
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906315, 0.9925519, 0.9906334, 0.9924810, -0.0008434, 0.0009280
1: -0.0035984, -0.0031198, -0.0035979, -0.0031375, -0.0002101, 0.0002312
2: 0.0064794, 0.0090154, 0.0065731, 0.0090128, -0.0012255, 0.0011136
3: -0.0053765, -0.0042223, -0.0053754, -0.0042649, -0.0005069, 0.0005578
4: 0.0017820, 0.0022728, 0.0018001, 0.0022723, -0.0002372, 0.0002155
5: 0.0071088, 0.0102984, 0.0072266, 0.0102952, -0.0015413, 0.0014007
6: -0.0010730, -0.0002635, -0.0010722, -0.0002934, -0.0003555, 0.0003912
7: -0.0059139, -0.0038193, -0.0059117, -0.0038966, -0.0009198, 0.0010122
8: -0.0026742, -0.0015727, -0.0026731, -0.0016134, -0.0004837, 0.0005323
9: -0.0000403, 0.0012370, 0.0000069, 0.0012357, -0.0006172, 0.0005609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005799, upper bound: 0.0005931
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005790, upper bound: 0.0006009
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906309, 0.9925478, 0.9906303, 0.9924688, -0.0008450, 0.0009608
1: -0.0035985, -0.0031208, -0.0035986, -0.0031405, -0.0002106, 0.0002394
2: 0.0064848, 0.0090159, 0.0065892, 0.0090169, -0.0012687, 0.0011158
3: -0.0053768, -0.0042247, -0.0053772, -0.0042722, -0.0005079, 0.0005775
4: 0.0017830, 0.0022729, 0.0018032, 0.0022731, -0.0002456, 0.0002160
5: 0.0071156, 0.0102991, 0.0072469, 0.0103003, -0.0015958, 0.0014034
6: -0.0010732, -0.0002652, -0.0010735, -0.0002985, -0.0003562, 0.0004050
7: -0.0059143, -0.0038237, -0.0059151, -0.0039100, -0.0009216, 0.0010479
8: -0.0026744, -0.0015750, -0.0026748, -0.0016204, -0.0004847, 0.0005511
9: -0.0000376, 0.0012373, 0.0000150, 0.0012377, -0.0006390, 0.0005620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0005932
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005753, upper bound: 0.0006015
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925509, 0.9905916, 0.9923964, -0.0008196, 0.0010553
1: -0.0035902, -0.0031201, -0.0036083, -0.0031586, -0.0002042, 0.0002630
2: 0.0064807, 0.0089723, 0.0066848, 0.0090679, -0.0013935, 0.0010823
3: -0.0053569, -0.0042229, -0.0054005, -0.0043157, -0.0004926, 0.0006343
4: 0.0017822, 0.0022645, 0.0018217, 0.0022830, -0.0002697, 0.0002095
5: 0.0071105, 0.0102443, 0.0073671, 0.0103645, -0.0017527, 0.0013612
6: -0.0010593, -0.0002639, -0.0010898, -0.0003290, -0.0003455, 0.0004449
7: -0.0058783, -0.0038204, -0.0059573, -0.0039889, -0.0008939, 0.0011510
8: -0.0026555, -0.0015733, -0.0026970, -0.0016619, -0.0004701, 0.0006053
9: -0.0000396, 0.0012153, 0.0000632, 0.0012635, -0.0007019, 0.0005451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004980, upper bound: 0.0005631
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005488, upper bound: 0.0006008
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906636, 0.9925468, 0.9905849, 0.9923916, -0.0008226, 0.0010886
1: -0.0035903, -0.0031211, -0.0036100, -0.0031598, -0.0002050, 0.0002712
2: 0.0064862, 0.0089728, 0.0066911, 0.0090769, -0.0014375, 0.0010862
3: -0.0053572, -0.0042253, -0.0054045, -0.0043186, -0.0004944, 0.0006543
4: 0.0017833, 0.0022646, 0.0018229, 0.0022847, -0.0002782, 0.0002102
5: 0.0071173, 0.0102449, 0.0073751, 0.0103758, -0.0018080, 0.0013662
6: -0.0010594, -0.0002656, -0.0010927, -0.0003310, -0.0003468, 0.0004589
7: -0.0058787, -0.0038249, -0.0059647, -0.0039941, -0.0008972, 0.0011873
8: -0.0026557, -0.0015756, -0.0027009, -0.0016646, -0.0004718, 0.0006244
9: -0.0000369, 0.0012156, 0.0000664, 0.0012680, -0.0007240, 0.0005471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004984, upper bound: 0.0005631
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005498, upper bound: 0.0006014
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906315, 0.9925519, 0.9905782, 0.9923964, -0.0008299, 0.0010651
1: -0.0035984, -0.0031198, -0.0036116, -0.0031586, -0.0002068, 0.0002654
2: 0.0064794, 0.0090154, 0.0066848, 0.0090856, -0.0014065, 0.0010959
3: -0.0053765, -0.0042223, -0.0054085, -0.0043157, -0.0004988, 0.0006402
4: 0.0017820, 0.0022728, 0.0018217, 0.0022864, -0.0002722, 0.0002121
5: 0.0071088, 0.0102984, 0.0073671, 0.0103868, -0.0017690, 0.0013783
6: -0.0010730, -0.0002635, -0.0010954, -0.0003290, -0.0003498, 0.0004490
7: -0.0059139, -0.0038193, -0.0059718, -0.0039889, -0.0009051, 0.0011617
8: -0.0026742, -0.0015727, -0.0027047, -0.0016619, -0.0004760, 0.0006109
9: -0.0000403, 0.0012370, 0.0000632, 0.0012724, -0.0007084, 0.0005519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005710
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005689, upper bound: 0.0006008
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906309, 0.9925478, 0.9905720, 0.9923916, -0.0008333, 0.0010970
1: -0.0035985, -0.0031208, -0.0036132, -0.0031598, -0.0002076, 0.0002734
2: 0.0064848, 0.0090159, 0.0066911, 0.0090939, -0.0014486, 0.0011004
3: -0.0053768, -0.0042247, -0.0054123, -0.0043186, -0.0005009, 0.0006594
4: 0.0017830, 0.0022729, 0.0018229, 0.0022880, -0.0002804, 0.0002130
5: 0.0071156, 0.0102991, 0.0073751, 0.0103972, -0.0018220, 0.0013840
6: -0.0010732, -0.0002652, -0.0010981, -0.0003310, -0.0003513, 0.0004624
7: -0.0059143, -0.0038237, -0.0059787, -0.0039941, -0.0009089, 0.0011965
8: -0.0026744, -0.0015750, -0.0027083, -0.0016646, -0.0004780, 0.0006292
9: -0.0000376, 0.0012373, 0.0000664, 0.0012765, -0.0007296, 0.0005542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005711
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0006014
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906018, 0.9924718, 0.9906468, 0.9924810, -0.0009388, 0.0008818
1: -0.0036057, -0.0031398, -0.0035945, -0.0031375, -0.0002339, 0.0002197
2: 0.0065851, 0.0090545, 0.0065731, 0.0089951, -0.0011644, 0.0012397
3: -0.0053943, -0.0042704, -0.0053673, -0.0042649, -0.0005643, 0.0005300
4: 0.0018024, 0.0022804, 0.0018001, 0.0022689, -0.0002254, 0.0002399
5: 0.0072418, 0.0103476, 0.0072266, 0.0102729, -0.0014646, 0.0015592
6: -0.0010855, -0.0002972, -0.0010665, -0.0002934, -0.0003958, 0.0003717
7: -0.0059461, -0.0039066, -0.0058971, -0.0038966, -0.0010239, 0.0009618
8: -0.0026912, -0.0016186, -0.0026654, -0.0016134, -0.0005385, 0.0005058
9: 0.0000130, 0.0012567, 0.0000069, 0.0012268, -0.0005865, 0.0006244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005277, upper bound: 0.0005461
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005563, upper bound: 0.0006006
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906015, 0.9924687, 0.9906429, 0.9924688, -0.0009383, 0.0009167
1: -0.0036058, -0.0031406, -0.0035955, -0.0031405, -0.0002338, 0.0002284
2: 0.0065894, 0.0090549, 0.0065892, 0.0090002, -0.0012105, 0.0012390
3: -0.0053945, -0.0042723, -0.0053696, -0.0042722, -0.0005639, 0.0005510
4: 0.0018033, 0.0022804, 0.0018032, 0.0022699, -0.0002343, 0.0002398
5: 0.0072472, 0.0103482, 0.0072469, 0.0102794, -0.0015225, 0.0015583
6: -0.0010856, -0.0002986, -0.0010682, -0.0002985, -0.0003955, 0.0003864
7: -0.0059465, -0.0039101, -0.0059013, -0.0039100, -0.0010233, 0.0009998
8: -0.0026914, -0.0016205, -0.0026676, -0.0016204, -0.0005381, 0.0005258
9: 0.0000151, 0.0012569, 0.0000150, 0.0012294, -0.0006097, 0.0006240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005238, upper bound: 0.0005461
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0006006
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905699, 0.9924747, 0.9906334, 0.9924810, -0.0009490, 0.0008951
1: -0.0036137, -0.0031391, -0.0035979, -0.0031375, -0.0002365, 0.0002230
2: 0.0065813, 0.0090966, 0.0065731, 0.0090128, -0.0011819, 0.0012532
3: -0.0054135, -0.0042687, -0.0053754, -0.0042649, -0.0005704, 0.0005380
4: 0.0018017, 0.0022885, 0.0018001, 0.0022723, -0.0002288, 0.0002426
5: 0.0072370, 0.0104005, 0.0072266, 0.0102952, -0.0014866, 0.0015762
6: -0.0010989, -0.0002960, -0.0010722, -0.0002934, -0.0004000, 0.0003773
7: -0.0059809, -0.0039035, -0.0059117, -0.0038966, -0.0010350, 0.0009762
8: -0.0027094, -0.0016170, -0.0026731, -0.0016134, -0.0005443, 0.0005134
9: 0.0000111, 0.0012779, 0.0000069, 0.0012357, -0.0005953, 0.0006312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005546
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005789, upper bound: 0.0006006
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905698, 0.9924716, 0.9906303, 0.9924688, -0.0009507, 0.0009287
1: -0.0036137, -0.0031398, -0.0035986, -0.0031405, -0.0002369, 0.0002314
2: 0.0065854, 0.0090969, 0.0065892, 0.0090169, -0.0012263, 0.0012554
3: -0.0054136, -0.0042705, -0.0053772, -0.0042722, -0.0005714, 0.0005582
4: 0.0018025, 0.0022886, 0.0018032, 0.0022731, -0.0002373, 0.0002430
5: 0.0072422, 0.0104010, 0.0072469, 0.0103003, -0.0015423, 0.0015789
6: -0.0010990, -0.0002973, -0.0010735, -0.0002985, -0.0004007, 0.0003915
7: -0.0059812, -0.0039069, -0.0059151, -0.0039100, -0.0010368, 0.0010128
8: -0.0027096, -0.0016187, -0.0026748, -0.0016204, -0.0005453, 0.0005326
9: 0.0000131, 0.0012781, 0.0000150, 0.0012377, -0.0006176, 0.0006323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005599, upper bound: 0.0005546
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005753, upper bound: 0.0006006
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906018, 0.9924718, 0.9905916, 0.9923964, -0.0008333, 0.0009180
1: -0.0036057, -0.0031398, -0.0036083, -0.0031586, -0.0002076, 0.0002287
2: 0.0065851, 0.0090545, 0.0066848, 0.0090679, -0.0012122, 0.0011004
3: -0.0053943, -0.0042704, -0.0054005, -0.0043157, -0.0005009, 0.0005518
4: 0.0018024, 0.0022804, 0.0018217, 0.0022830, -0.0002346, 0.0002130
5: 0.0072418, 0.0103476, 0.0073671, 0.0103645, -0.0015247, 0.0013841
6: -0.0010855, -0.0002972, -0.0010898, -0.0003290, -0.0003513, 0.0003870
7: -0.0059461, -0.0039066, -0.0059573, -0.0039889, -0.0009089, 0.0010012
8: -0.0026912, -0.0016186, -0.0026970, -0.0016619, -0.0004780, 0.0005265
9: 0.0000130, 0.0012567, 0.0000632, 0.0012635, -0.0006105, 0.0005542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004979, upper bound: 0.0005458
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005491, upper bound: 0.0006005
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906015, 0.9924687, 0.9905849, 0.9923916, -0.0008346, 0.0009501
1: -0.0036058, -0.0031406, -0.0036100, -0.0031598, -0.0002080, 0.0002367
2: 0.0065894, 0.0090549, 0.0066911, 0.0090769, -0.0012546, 0.0011021
3: -0.0053945, -0.0042723, -0.0054045, -0.0043186, -0.0005016, 0.0005710
4: 0.0018033, 0.0022804, 0.0018229, 0.0022847, -0.0002428, 0.0002133
5: 0.0072472, 0.0103482, 0.0073751, 0.0103758, -0.0015779, 0.0013862
6: -0.0010856, -0.0002986, -0.0010927, -0.0003310, -0.0003518, 0.0004005
7: -0.0059465, -0.0039101, -0.0059647, -0.0039941, -0.0009103, 0.0010362
8: -0.0026914, -0.0016205, -0.0027009, -0.0016646, -0.0004787, 0.0005449
9: 0.0000151, 0.0012569, 0.0000664, 0.0012680, -0.0006319, 0.0005551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005458
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005499, upper bound: 0.0006006
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905699, 0.9924747, 0.9905782, 0.9923964, -0.0008439, 0.0009273
1: -0.0036137, -0.0031391, -0.0036116, -0.0031586, -0.0002103, 0.0002311
2: 0.0065813, 0.0090966, 0.0066848, 0.0090856, -0.0012244, 0.0011144
3: -0.0054135, -0.0042687, -0.0054085, -0.0043157, -0.0005072, 0.0005573
4: 0.0018017, 0.0022885, 0.0018217, 0.0022864, -0.0002370, 0.0002157
5: 0.0072370, 0.0104005, 0.0073671, 0.0103868, -0.0015400, 0.0014016
6: -0.0010989, -0.0002960, -0.0010954, -0.0003290, -0.0003557, 0.0003909
7: -0.0059809, -0.0039035, -0.0059718, -0.0039889, -0.0009204, 0.0010113
8: -0.0027094, -0.0016170, -0.0027047, -0.0016619, -0.0004840, 0.0005318
9: 0.0000111, 0.0012779, 0.0000632, 0.0012724, -0.0006167, 0.0005613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005545
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005700, upper bound: 0.0006006
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905698, 0.9924716, 0.9905720, 0.9923916, -0.0008475, 0.0009577
1: -0.0036137, -0.0031398, -0.0036132, -0.0031598, -0.0002112, 0.0002386
2: 0.0065854, 0.0090969, 0.0066911, 0.0090939, -0.0012646, 0.0011192
3: -0.0054136, -0.0042705, -0.0054123, -0.0043186, -0.0005094, 0.0005756
4: 0.0018025, 0.0022886, 0.0018229, 0.0022880, -0.0002448, 0.0002166
5: 0.0072422, 0.0104010, 0.0073751, 0.0103972, -0.0015905, 0.0014076
6: -0.0010990, -0.0002973, -0.0010981, -0.0003310, -0.0003573, 0.0004037
7: -0.0059812, -0.0039069, -0.0059787, -0.0039941, -0.0009244, 0.0010445
8: -0.0027096, -0.0016187, -0.0027083, -0.0016646, -0.0004861, 0.0005493
9: 0.0000131, 0.0012781, 0.0000664, 0.0012765, -0.0006369, 0.0005637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005545
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005696, upper bound: 0.0006006
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925509, 0.9906345, 0.9925352, -0.0008084, 0.0008604
1: -0.0035902, -0.0031201, -0.0035976, -0.0031240, -0.0002014, 0.0002144
2: 0.0064807, 0.0089723, 0.0065014, 0.0090113, -0.0011361, 0.0010675
3: -0.0053569, -0.0042229, -0.0053747, -0.0042323, -0.0004859, 0.0005171
4: 0.0017822, 0.0022645, 0.0017862, 0.0022720, -0.0002199, 0.0002066
5: 0.0071105, 0.0102443, 0.0071365, 0.0102933, -0.0014289, 0.0013426
6: -0.0010593, -0.0002639, -0.0010717, -0.0002705, -0.0003408, 0.0003627
7: -0.0058783, -0.0038204, -0.0059104, -0.0038375, -0.0008817, 0.0009383
8: -0.0026555, -0.0015733, -0.0026724, -0.0015822, -0.0004637, 0.0004935
9: -0.0000396, 0.0012153, -0.0000292, 0.0012349, -0.0005722, 0.0005376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005523, upper bound: 0.0005928
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005564, upper bound: 0.0006010
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906636, 0.9925468, 0.9906290, 0.9925265, -0.0008090, 0.0008950
1: -0.0035903, -0.0031211, -0.0035990, -0.0031261, -0.0002016, 0.0002230
2: 0.0064862, 0.0089728, 0.0065129, 0.0090186, -0.0011818, 0.0010683
3: -0.0053572, -0.0042253, -0.0053780, -0.0042375, -0.0004862, 0.0005379
4: 0.0017833, 0.0022646, 0.0017885, 0.0022734, -0.0002287, 0.0002068
5: 0.0071173, 0.0102449, 0.0071510, 0.0103025, -0.0014864, 0.0013436
6: -0.0010594, -0.0002656, -0.0010740, -0.0002742, -0.0003410, 0.0003773
7: -0.0058787, -0.0038249, -0.0059165, -0.0038470, -0.0008823, 0.0009761
8: -0.0026557, -0.0015756, -0.0026756, -0.0015872, -0.0004640, 0.0005133
9: -0.0000369, 0.0012156, -0.0000234, 0.0012386, -0.0005952, 0.0005380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005494, upper bound: 0.0005930
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005536, upper bound: 0.0006015
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906315, 0.9925519, 0.9906209, 0.9925352, -0.0008163, 0.0008690
1: -0.0035984, -0.0031198, -0.0036010, -0.0031240, -0.0002034, 0.0002165
2: 0.0064794, 0.0090154, 0.0065014, 0.0090293, -0.0011475, 0.0010779
3: -0.0053765, -0.0042223, -0.0053829, -0.0042323, -0.0004906, 0.0005223
4: 0.0017820, 0.0022728, 0.0017862, 0.0022755, -0.0002221, 0.0002086
5: 0.0071088, 0.0102984, 0.0071365, 0.0103159, -0.0014433, 0.0013557
6: -0.0010730, -0.0002635, -0.0010775, -0.0002705, -0.0003441, 0.0003663
7: -0.0059139, -0.0038193, -0.0059254, -0.0038375, -0.0008903, 0.0009478
8: -0.0026742, -0.0015727, -0.0026802, -0.0015822, -0.0004682, 0.0004984
9: -0.0000403, 0.0012370, -0.0000292, 0.0012440, -0.0005780, 0.0005429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005799, upper bound: 0.0005931
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005790, upper bound: 0.0006008
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906309, 0.9925478, 0.9906155, 0.9925265, -0.0008194, 0.0009009
1: -0.0035985, -0.0031208, -0.0036023, -0.0031261, -0.0002042, 0.0002245
2: 0.0064848, 0.0090159, 0.0065129, 0.0090363, -0.0011896, 0.0010820
3: -0.0053768, -0.0042247, -0.0053861, -0.0042375, -0.0004925, 0.0005415
4: 0.0017830, 0.0022729, 0.0017885, 0.0022768, -0.0002303, 0.0002094
5: 0.0071156, 0.0102991, 0.0071510, 0.0103248, -0.0014962, 0.0013609
6: -0.0010732, -0.0002652, -0.0010797, -0.0002742, -0.0003454, 0.0003798
7: -0.0059143, -0.0038237, -0.0059311, -0.0038470, -0.0008937, 0.0009826
8: -0.0026744, -0.0015750, -0.0026833, -0.0015872, -0.0004700, 0.0005167
9: -0.0000376, 0.0012373, -0.0000234, 0.0012475, -0.0005992, 0.0005450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0005932
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005754, upper bound: 0.0006015
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925509, 0.9905726, 0.9924576, -0.0007951, 0.0009978
1: -0.0035902, -0.0031201, -0.0036130, -0.0031433, -0.0001981, 0.0002486
2: 0.0064807, 0.0089723, 0.0066040, 0.0090930, -0.0013176, 0.0010499
3: -0.0053569, -0.0042229, -0.0054119, -0.0042790, -0.0004779, 0.0005997
4: 0.0017822, 0.0022645, 0.0018061, 0.0022878, -0.0002550, 0.0002032
5: 0.0071105, 0.0102443, 0.0072656, 0.0103961, -0.0016572, 0.0013205
6: -0.0010593, -0.0002639, -0.0010978, -0.0003033, -0.0003352, 0.0004206
7: -0.0058783, -0.0038204, -0.0059780, -0.0039222, -0.0008671, 0.0010883
8: -0.0026555, -0.0015733, -0.0027079, -0.0016268, -0.0004560, 0.0005723
9: -0.0000396, 0.0012153, 0.0000225, 0.0012761, -0.0006636, 0.0005288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004980, upper bound: 0.0005631
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005488, upper bound: 0.0006008
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906636, 0.9925468, 0.9905646, 0.9924538, -0.0007992, 0.0010314
1: -0.0035903, -0.0031211, -0.0036150, -0.0031443, -0.0001991, 0.0002570
2: 0.0064862, 0.0089728, 0.0066090, 0.0091036, -0.0013619, 0.0010554
3: -0.0053572, -0.0042253, -0.0054167, -0.0042813, -0.0004804, 0.0006199
4: 0.0017833, 0.0022646, 0.0018071, 0.0022899, -0.0002636, 0.0002043
5: 0.0071173, 0.0102449, 0.0072719, 0.0104094, -0.0017129, 0.0013274
6: -0.0010594, -0.0002656, -0.0011012, -0.0003048, -0.0003369, 0.0004348
7: -0.0058787, -0.0038249, -0.0059867, -0.0039263, -0.0008717, 0.0011248
8: -0.0026557, -0.0015756, -0.0027125, -0.0016290, -0.0004584, 0.0005915
9: -0.0000369, 0.0012156, 0.0000250, 0.0012814, -0.0006859, 0.0005315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004984, upper bound: 0.0005631
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005498, upper bound: 0.0006014
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906315, 0.9925519, 0.9905592, 0.9924576, -0.0008030, 0.0010062
1: -0.0035984, -0.0031198, -0.0036164, -0.0031433, -0.0002001, 0.0002507
2: 0.0064794, 0.0090154, 0.0066040, 0.0091108, -0.0013287, 0.0010603
3: -0.0053765, -0.0042223, -0.0054200, -0.0042790, -0.0004826, 0.0006048
4: 0.0017820, 0.0022728, 0.0018061, 0.0022913, -0.0002572, 0.0002052
5: 0.0071088, 0.0102984, 0.0072656, 0.0104185, -0.0016712, 0.0013336
6: -0.0010730, -0.0002635, -0.0011035, -0.0003033, -0.0003385, 0.0004242
7: -0.0059139, -0.0038193, -0.0059927, -0.0039222, -0.0008758, 0.0010974
8: -0.0026742, -0.0015727, -0.0027156, -0.0016268, -0.0004606, 0.0005771
9: -0.0000403, 0.0012370, 0.0000225, 0.0012851, -0.0006692, 0.0005340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005710
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005689, upper bound: 0.0006008
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906309, 0.9925478, 0.9905514, 0.9924538, -0.0008079, 0.0010371
1: -0.0035985, -0.0031208, -0.0036183, -0.0031443, -0.0002013, 0.0002584
2: 0.0064848, 0.0090159, 0.0066090, 0.0091212, -0.0013695, 0.0010669
3: -0.0053768, -0.0042247, -0.0054247, -0.0042813, -0.0004856, 0.0006233
4: 0.0017830, 0.0022729, 0.0018071, 0.0022933, -0.0002651, 0.0002065
5: 0.0071156, 0.0102991, 0.0072719, 0.0104315, -0.0017224, 0.0013419
6: -0.0010732, -0.0002652, -0.0011068, -0.0003048, -0.0003406, 0.0004372
7: -0.0059143, -0.0038237, -0.0060012, -0.0039263, -0.0008812, 0.0011311
8: -0.0026744, -0.0015750, -0.0027201, -0.0016290, -0.0004634, 0.0005948
9: -0.0000376, 0.0012373, 0.0000250, 0.0012903, -0.0006897, 0.0005373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005711
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0006014
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906018, 0.9924718, 0.9906345, 0.9925352, -0.0009464, 0.0008467
1: -0.0036057, -0.0031398, -0.0035976, -0.0031240, -0.0002358, 0.0002110
2: 0.0065851, 0.0090545, 0.0065014, 0.0090113, -0.0011180, 0.0012497
3: -0.0053943, -0.0042704, -0.0053747, -0.0042323, -0.0005688, 0.0005089
4: 0.0018024, 0.0022804, 0.0017862, 0.0022720, -0.0002164, 0.0002419
5: 0.0072418, 0.0103476, 0.0071365, 0.0102933, -0.0014062, 0.0015718
6: -0.0010855, -0.0002972, -0.0010717, -0.0002705, -0.0003989, 0.0003569
7: -0.0059461, -0.0039066, -0.0059104, -0.0038375, -0.0010322, 0.0009234
8: -0.0026912, -0.0016186, -0.0026724, -0.0015822, -0.0005428, 0.0004856
9: 0.0000130, 0.0012567, -0.0000292, 0.0012349, -0.0005631, 0.0006294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005277, upper bound: 0.0005460
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005563, upper bound: 0.0006006
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906015, 0.9924687, 0.9906290, 0.9925265, -0.0009472, 0.0008804
1: -0.0036058, -0.0031406, -0.0035990, -0.0031261, -0.0002360, 0.0002194
2: 0.0065894, 0.0090549, 0.0065129, 0.0090186, -0.0011626, 0.0012508
3: -0.0053945, -0.0042723, -0.0053780, -0.0042375, -0.0005693, 0.0005292
4: 0.0018033, 0.0022804, 0.0017885, 0.0022734, -0.0002250, 0.0002421
5: 0.0072472, 0.0103482, 0.0071510, 0.0103025, -0.0014622, 0.0015732
6: -0.0010856, -0.0002986, -0.0010740, -0.0002742, -0.0003993, 0.0003711
7: -0.0059465, -0.0039101, -0.0059165, -0.0038470, -0.0010331, 0.0009602
8: -0.0026914, -0.0016205, -0.0026756, -0.0015872, -0.0005433, 0.0005050
9: 0.0000151, 0.0012569, -0.0000234, 0.0012386, -0.0005855, 0.0006300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005238, upper bound: 0.0005460
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0006006
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905699, 0.9924747, 0.9906209, 0.9925352, -0.0009545, 0.0008589
1: -0.0036137, -0.0031391, -0.0036010, -0.0031240, -0.0002378, 0.0002140
2: 0.0065813, 0.0090966, 0.0065014, 0.0090293, -0.0011341, 0.0012605
3: -0.0054135, -0.0042687, -0.0053829, -0.0042323, -0.0005737, 0.0005162
4: 0.0018017, 0.0022885, 0.0017862, 0.0022755, -0.0002195, 0.0002440
5: 0.0072370, 0.0104005, 0.0071365, 0.0103159, -0.0014265, 0.0015853
6: -0.0010989, -0.0002960, -0.0010775, -0.0002705, -0.0004024, 0.0003620
7: -0.0059809, -0.0039035, -0.0059254, -0.0038375, -0.0010411, 0.0009367
8: -0.0027094, -0.0016170, -0.0026802, -0.0015822, -0.0005475, 0.0004926
9: 0.0000111, 0.0012779, -0.0000292, 0.0012440, -0.0005712, 0.0006348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005546
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005790, upper bound: 0.0006006
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905698, 0.9924716, 0.9906155, 0.9925265, -0.0009578, 0.0008902
1: -0.0036137, -0.0031398, -0.0036023, -0.0031261, -0.0002387, 0.0002218
2: 0.0065854, 0.0090969, 0.0065129, 0.0090363, -0.0011756, 0.0012648
3: -0.0054136, -0.0042705, -0.0053861, -0.0042375, -0.0005757, 0.0005351
4: 0.0018025, 0.0022886, 0.0017885, 0.0022768, -0.0002275, 0.0002448
5: 0.0072422, 0.0104010, 0.0071510, 0.0103248, -0.0014786, 0.0015907
6: -0.0010990, -0.0002973, -0.0010797, -0.0002742, -0.0004037, 0.0003753
7: -0.0059812, -0.0039069, -0.0059311, -0.0038470, -0.0010446, 0.0009709
8: -0.0027096, -0.0016187, -0.0026833, -0.0015872, -0.0005494, 0.0005106
9: 0.0000131, 0.0012781, -0.0000234, 0.0012475, -0.0005921, 0.0006370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005599, upper bound: 0.0005546
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005753, upper bound: 0.0006006
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906018, 0.9924718, 0.9905726, 0.9924576, -0.0008132, 0.0008657
1: -0.0036057, -0.0031398, -0.0036130, -0.0031433, -0.0002026, 0.0002157
2: 0.0065851, 0.0090545, 0.0066040, 0.0090930, -0.0011432, 0.0010739
3: -0.0053943, -0.0042704, -0.0054119, -0.0042790, -0.0004888, 0.0005203
4: 0.0018024, 0.0022804, 0.0018061, 0.0022878, -0.0002213, 0.0002078
5: 0.0072418, 0.0103476, 0.0072656, 0.0103961, -0.0014378, 0.0013506
6: -0.0010855, -0.0002972, -0.0010978, -0.0003033, -0.0003428, 0.0003649
7: -0.0059461, -0.0039066, -0.0059780, -0.0039222, -0.0008869, 0.0009442
8: -0.0026912, -0.0016186, -0.0027079, -0.0016268, -0.0004664, 0.0004965
9: 0.0000130, 0.0012567, 0.0000225, 0.0012761, -0.0005758, 0.0005409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004979, upper bound: 0.0005457
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005491, upper bound: 0.0006006
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906015, 0.9924687, 0.9905646, 0.9924538, -0.0008149, 0.0008963
1: -0.0036058, -0.0031406, -0.0036150, -0.0031443, -0.0002031, 0.0002233
2: 0.0065894, 0.0090549, 0.0066090, 0.0091036, -0.0011836, 0.0010761
3: -0.0053945, -0.0042723, -0.0054167, -0.0042813, -0.0004898, 0.0005387
4: 0.0018033, 0.0022804, 0.0018071, 0.0022899, -0.0002291, 0.0002083
5: 0.0072472, 0.0103482, 0.0072719, 0.0104094, -0.0014886, 0.0013534
6: -0.0010856, -0.0002986, -0.0011012, -0.0003048, -0.0003435, 0.0003778
7: -0.0059465, -0.0039101, -0.0059867, -0.0039263, -0.0008888, 0.0009776
8: -0.0026914, -0.0016205, -0.0027125, -0.0016290, -0.0004674, 0.0005141
9: 0.0000151, 0.0012569, 0.0000250, 0.0012814, -0.0005961, 0.0005420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005457
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005499, upper bound: 0.0006006
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905699, 0.9924747, 0.9905592, 0.9924576, -0.0008210, 0.0008744
1: -0.0036137, -0.0031391, -0.0036164, -0.0031433, -0.0002046, 0.0002179
2: 0.0065813, 0.0090966, 0.0066040, 0.0091108, -0.0011547, 0.0010841
3: -0.0054135, -0.0042687, -0.0054200, -0.0042790, -0.0004935, 0.0005256
4: 0.0018017, 0.0022885, 0.0018061, 0.0022913, -0.0002235, 0.0002098
5: 0.0072370, 0.0104005, 0.0072656, 0.0104185, -0.0014523, 0.0013636
6: -0.0010989, -0.0002960, -0.0011035, -0.0003033, -0.0003461, 0.0003686
7: -0.0059809, -0.0039035, -0.0059927, -0.0039222, -0.0008954, 0.0009537
8: -0.0027094, -0.0016170, -0.0027156, -0.0016268, -0.0004709, 0.0005015
9: 0.0000111, 0.0012779, 0.0000225, 0.0012851, -0.0005816, 0.0005460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005545
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005700, upper bound: 0.0006006
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905698, 0.9924716, 0.9905514, 0.9924538, -0.0008248, 0.0009030
1: -0.0036137, -0.0031398, -0.0036183, -0.0031443, -0.0002055, 0.0002250
2: 0.0065854, 0.0090969, 0.0066090, 0.0091212, -0.0011924, 0.0010892
3: -0.0054136, -0.0042705, -0.0054247, -0.0042813, -0.0004957, 0.0005427
4: 0.0018025, 0.0022886, 0.0018071, 0.0022933, -0.0002308, 0.0002108
5: 0.0072422, 0.0104010, 0.0072719, 0.0104315, -0.0014997, 0.0013699
6: -0.0010990, -0.0002973, -0.0011068, -0.0003048, -0.0003477, 0.0003806
7: -0.0059812, -0.0039069, -0.0060012, -0.0039263, -0.0008996, 0.0009848
8: -0.0027096, -0.0016187, -0.0027201, -0.0016290, -0.0004731, 0.0005179
9: 0.0000131, 0.0012781, 0.0000250, 0.0012903, -0.0006005, 0.0005486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005545
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005696, upper bound: 0.0006006
time: 0.62 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.87 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005650, upper bound: 0.0005806
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005671, upper bound: 0.0005841
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005642, upper bound: 0.0005808
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005659, upper bound: 0.0005847
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005904, upper bound: 0.0005813
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005881, upper bound: 0.0005841
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005874, upper bound: 0.0005816
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005847
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005823, upper bound: 0.0005744
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005839, upper bound: 0.0005751
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005832, upper bound: 0.0005746
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005837, upper bound: 0.0005754
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006049, upper bound: 0.0005748
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006032, upper bound: 0.0005751
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006038, upper bound: 0.0005752
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006014, upper bound: 0.0005754
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005756, upper bound: 0.0005502
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005881, upper bound: 0.0005811
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005724, upper bound: 0.0005502
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005811
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005562, upper bound: 0.0005339
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005839, upper bound: 0.0005694
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005561, upper bound: 0.0005339
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005837, upper bound: 0.0005694
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005882, upper bound: 0.0005425
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006031, upper bound: 0.0005694
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005862, upper bound: 0.0005425
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006014, upper bound: 0.0005694
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005161, upper bound: 0.0005555
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005572, upper bound: 0.0005840
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005200, upper bound: 0.0005555
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005590, upper bound: 0.0005847
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005640
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005813, upper bound: 0.0005840
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005642
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005811, upper bound: 0.0005847
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005737, upper bound: 0.0005566
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006005, upper bound: 0.0005750
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005768, upper bound: 0.0005573
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006006, upper bound: 0.0005753
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005502
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005802, upper bound: 0.0005811
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005737, upper bound: 0.0005425
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0006006, upper bound: 0.0005694
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005767, upper bound: 0.0005425
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005998, upper bound: 0.0005694
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005523, upper bound: 0.0005927
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005563, upper bound: 0.0006009
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005494, upper bound: 0.0005930
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005536, upper bound: 0.0006015
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005799, upper bound: 0.0005931
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005790, upper bound: 0.0006009
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0005932
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005753, upper bound: 0.0006015
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004980, upper bound: 0.0005631
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005488, upper bound: 0.0006008
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004984, upper bound: 0.0005631
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005498, upper bound: 0.0006014
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005710
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005689, upper bound: 0.0006008
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005711
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0006014
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005277, upper bound: 0.0005461
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005563, upper bound: 0.0006006
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005238, upper bound: 0.0005461
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0006006
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005546
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005789, upper bound: 0.0006006
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005599, upper bound: 0.0005546
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005753, upper bound: 0.0006006
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004979, upper bound: 0.0005458
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005491, upper bound: 0.0006005
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005458
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005499, upper bound: 0.0006006
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005545
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005700, upper bound: 0.0006006
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005545
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005696, upper bound: 0.0006006
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005523, upper bound: 0.0005928
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005564, upper bound: 0.0006010
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005494, upper bound: 0.0005930
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005536, upper bound: 0.0006015
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005799, upper bound: 0.0005931
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005790, upper bound: 0.0006008
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005759, upper bound: 0.0005932
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005754, upper bound: 0.0006015
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004980, upper bound: 0.0005631
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005488, upper bound: 0.0006008
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004984, upper bound: 0.0005631
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005498, upper bound: 0.0006014
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005710
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005689, upper bound: 0.0006008
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005711
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0006014
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005277, upper bound: 0.0005460
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005563, upper bound: 0.0006006
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005238, upper bound: 0.0005460
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005535, upper bound: 0.0006006
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005546
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005790, upper bound: 0.0006006
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005599, upper bound: 0.0005546
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005753, upper bound: 0.0006006
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004979, upper bound: 0.0005457
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005491, upper bound: 0.0006006
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005457
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005499, upper bound: 0.0006006
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005452, upper bound: 0.0005545
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005700, upper bound: 0.0006006
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005460, upper bound: 0.0005545
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0005696, upper bound: 0.0006006

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906892, 0.9925329, 0.9906341, 0.9925352, -0.0008132, 0.0008951
1: -0.0035840, -0.0031246, -0.0035977, -0.0031240, -0.0002026, 0.0002230
2: 0.0065045, 0.0089391, 0.0065014, 0.0090119, -0.0011819, 0.0010739
3: -0.0053418, -0.0042337, -0.0053749, -0.0042323, -0.0004888, 0.0005380
4: 0.0017868, 0.0022580, 0.0017862, 0.0022721, -0.0002288, 0.0002078
5: 0.0071404, 0.0102025, 0.0071365, 0.0102940, -0.0014865, 0.0013507
6: -0.0010487, -0.0002715, -0.0010719, -0.0002705, -0.0003428, 0.0003773
7: -0.0058508, -0.0038400, -0.0059110, -0.0038375, -0.0008870, 0.0009762
8: -0.0026410, -0.0015836, -0.0026727, -0.0015822, -0.0004664, 0.0005134
9: -0.0000276, 0.0011986, -0.0000292, 0.0012352, -0.0005953, 0.0005409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005818, upper bound: 0.0005417
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0005506
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906604, 0.9924966, 0.9906209, 0.9925352, -0.0008519, 0.0008899
1: -0.0035911, -0.0031336, -0.0036010, -0.0031240, -0.0002123, 0.0002217
2: 0.0065524, 0.0089772, 0.0065014, 0.0090293, -0.0011751, 0.0011249
3: -0.0053591, -0.0042555, -0.0053829, -0.0042323, -0.0005120, 0.0005349
4: 0.0017961, 0.0022654, 0.0017862, 0.0022755, -0.0002274, 0.0002177
5: 0.0072007, 0.0102503, 0.0071365, 0.0103159, -0.0014780, 0.0014148
6: -0.0010608, -0.0002868, -0.0010775, -0.0002705, -0.0003591, 0.0003751
7: -0.0058823, -0.0038796, -0.0059254, -0.0038375, -0.0009291, 0.0009706
8: -0.0026576, -0.0016044, -0.0026802, -0.0015822, -0.0004886, 0.0005104
9: -0.0000035, 0.0012177, -0.0000292, 0.0012440, -0.0005919, 0.0005666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005924, upper bound: 0.0005521
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005924, upper bound: 0.0005639
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906888, 0.9925288, 0.9906279, 0.9925265, -0.0008164, 0.0009295
1: -0.0035841, -0.0031256, -0.0035992, -0.0031261, -0.0002034, 0.0002316
2: 0.0065099, 0.0089396, 0.0065129, 0.0090200, -0.0012274, 0.0010781
3: -0.0053421, -0.0042362, -0.0053786, -0.0042375, -0.0004907, 0.0005587
4: 0.0017879, 0.0022581, 0.0017885, 0.0022737, -0.0002376, 0.0002087
5: 0.0071472, 0.0102032, 0.0071510, 0.0103043, -0.0015438, 0.0013560
6: -0.0010488, -0.0002732, -0.0010745, -0.0002742, -0.0003442, 0.0003918
7: -0.0058513, -0.0038445, -0.0059177, -0.0038470, -0.0008905, 0.0010138
8: -0.0026413, -0.0015859, -0.0026762, -0.0015872, -0.0004683, 0.0005331
9: -0.0000249, 0.0011989, -0.0000234, 0.0012393, -0.0006182, 0.0005430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005806, upper bound: 0.0005424
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005806, upper bound: 0.0005506
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906600, 0.9924916, 0.9906155, 0.9925265, -0.0008558, 0.0009209
1: -0.0035912, -0.0031348, -0.0036023, -0.0031261, -0.0002132, 0.0002295
2: 0.0065590, 0.0089777, 0.0065129, 0.0090363, -0.0012161, 0.0011300
3: -0.0053594, -0.0042585, -0.0053861, -0.0042375, -0.0005143, 0.0005535
4: 0.0017974, 0.0022655, 0.0017885, 0.0022768, -0.0002354, 0.0002187
5: 0.0072090, 0.0102511, 0.0071510, 0.0103248, -0.0015295, 0.0014213
6: -0.0010610, -0.0002889, -0.0010797, -0.0002742, -0.0003607, 0.0003882
7: -0.0058827, -0.0038851, -0.0059311, -0.0038470, -0.0009333, 0.0010044
8: -0.0026578, -0.0016073, -0.0026833, -0.0015872, -0.0004908, 0.0005282
9: -0.0000002, 0.0012180, -0.0000234, 0.0012475, -0.0006125, 0.0005691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005906, upper bound: 0.0005538
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005906, upper bound: 0.0005639
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906055, 0.9924133, 0.9906209, 0.9925352, -0.0009938, 0.0008801
1: -0.0036048, -0.0031544, -0.0036010, -0.0031240, -0.0002476, 0.0002193
2: 0.0066625, 0.0090496, 0.0065014, 0.0090293, -0.0011622, 0.0013122
3: -0.0053921, -0.0043056, -0.0053829, -0.0042323, -0.0005973, 0.0005290
4: 0.0018174, 0.0022794, 0.0017862, 0.0022755, -0.0002249, 0.0002540
5: 0.0073391, 0.0103414, 0.0071365, 0.0103159, -0.0014617, 0.0016505
6: -0.0010839, -0.0003219, -0.0010775, -0.0002705, -0.0004189, 0.0003710
7: -0.0059421, -0.0039705, -0.0059254, -0.0038375, -0.0010838, 0.0009599
8: -0.0026890, -0.0016522, -0.0026802, -0.0015822, -0.0005700, 0.0005048
9: 0.0000520, 0.0012542, -0.0000292, 0.0012440, -0.0005853, 0.0006609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005812, upper bound: 0.0005331
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005389
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906052, 0.9924096, 0.9906155, 0.9925265, -0.0009979, 0.0009107
1: -0.0036049, -0.0031553, -0.0036023, -0.0031261, -0.0002486, 0.0002269
2: 0.0066673, 0.0090499, 0.0065129, 0.0090363, -0.0012025, 0.0013177
3: -0.0053923, -0.0043078, -0.0053861, -0.0042375, -0.0005997, 0.0005473
4: 0.0018183, 0.0022795, 0.0017885, 0.0022768, -0.0002327, 0.0002550
5: 0.0073452, 0.0103419, 0.0071510, 0.0103248, -0.0015124, 0.0016573
6: -0.0010841, -0.0003235, -0.0010797, -0.0002742, -0.0004206, 0.0003839
7: -0.0059424, -0.0039745, -0.0059311, -0.0038470, -0.0010883, 0.0009932
8: -0.0026892, -0.0016543, -0.0026833, -0.0015872, -0.0005723, 0.0005223
9: 0.0000544, 0.0012544, -0.0000234, 0.0012475, -0.0006056, 0.0006637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0005337
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005780, upper bound: 0.0005391
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906604, 0.9924966, 0.9905592, 0.9924576, -0.0008192, 0.0009955
1: -0.0035911, -0.0031336, -0.0036164, -0.0031433, -0.0002041, 0.0002481
2: 0.0065524, 0.0089772, 0.0066040, 0.0091108, -0.0013146, 0.0010818
3: -0.0053591, -0.0042555, -0.0054200, -0.0042790, -0.0004924, 0.0005984
4: 0.0017961, 0.0022654, 0.0018061, 0.0022913, -0.0002544, 0.0002094
5: 0.0072007, 0.0102503, 0.0072656, 0.0104185, -0.0016534, 0.0013606
6: -0.0010608, -0.0002868, -0.0011035, -0.0003033, -0.0003453, 0.0004197
7: -0.0058823, -0.0038796, -0.0059927, -0.0039222, -0.0008935, 0.0010858
8: -0.0026576, -0.0016044, -0.0027156, -0.0016268, -0.0004699, 0.0005710
9: -0.0000035, 0.0012177, 0.0000225, 0.0012851, -0.0006621, 0.0005448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005596
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005750
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906600, 0.9924916, 0.9905514, 0.9924538, -0.0008236, 0.0010291
1: -0.0035912, -0.0031348, -0.0036183, -0.0031443, -0.0002052, 0.0002564
2: 0.0065590, 0.0089777, 0.0066090, 0.0091212, -0.0013589, 0.0010876
3: -0.0053594, -0.0042585, -0.0054247, -0.0042813, -0.0004950, 0.0006185
4: 0.0017974, 0.0022655, 0.0018071, 0.0022933, -0.0002630, 0.0002105
5: 0.0072090, 0.0102511, 0.0072719, 0.0104315, -0.0017092, 0.0013679
6: -0.0010610, -0.0002889, -0.0011068, -0.0003048, -0.0003472, 0.0004338
7: -0.0058827, -0.0038851, -0.0060012, -0.0039263, -0.0008983, 0.0011224
8: -0.0026578, -0.0016073, -0.0027201, -0.0016290, -0.0004724, 0.0005903
9: -0.0000002, 0.0012180, 0.0000250, 0.0012903, -0.0006844, 0.0005478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005469
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005708, upper bound: 0.0005534
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906055, 0.9924133, 0.9905592, 0.9924576, -0.0008485, 0.0008910
1: -0.0036048, -0.0031544, -0.0036164, -0.0031433, -0.0002114, 0.0002220
2: 0.0066625, 0.0090496, 0.0066040, 0.0091108, -0.0011766, 0.0011205
3: -0.0053921, -0.0043056, -0.0054200, -0.0042790, -0.0005100, 0.0005355
4: 0.0018174, 0.0022794, 0.0018061, 0.0022913, -0.0002277, 0.0002169
5: 0.0073391, 0.0103414, 0.0072656, 0.0104185, -0.0014799, 0.0014092
6: -0.0010839, -0.0003219, -0.0011035, -0.0003033, -0.0003577, 0.0003756
7: -0.0059421, -0.0039705, -0.0059927, -0.0039222, -0.0009254, 0.0009718
8: -0.0026890, -0.0016522, -0.0027156, -0.0016268, -0.0004867, 0.0005111
9: 0.0000520, 0.0012542, 0.0000225, 0.0012851, -0.0005926, 0.0005643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005460
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005694
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906052, 0.9924096, 0.9905514, 0.9924538, -0.0008538, 0.0009195
1: -0.0036049, -0.0031553, -0.0036183, -0.0031443, -0.0002127, 0.0002291
2: 0.0066673, 0.0090499, 0.0066090, 0.0091212, -0.0012142, 0.0011274
3: -0.0053923, -0.0043078, -0.0054247, -0.0042813, -0.0005132, 0.0005527
4: 0.0018183, 0.0022795, 0.0018071, 0.0022933, -0.0002350, 0.0002182
5: 0.0073452, 0.0103419, 0.0072719, 0.0104315, -0.0015271, 0.0014180
6: -0.0010841, -0.0003235, -0.0011068, -0.0003048, -0.0003599, 0.0003876
7: -0.0059424, -0.0039745, -0.0060012, -0.0039263, -0.0009312, 0.0010029
8: -0.0026892, -0.0016543, -0.0027201, -0.0016290, -0.0004897, 0.0005274
9: 0.0000544, 0.0012544, 0.0000250, 0.0012903, -0.0006115, 0.0005678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005337
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005708, upper bound: 0.0005391
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906468, 0.9924810, -0.0008084, 0.0009134
1: -0.0035860, -0.0031201, -0.0035945, -0.0031375, -0.0002014, 0.0002276
2: 0.0064807, 0.0089497, 0.0065731, 0.0089951, -0.0012061, 0.0010675
3: -0.0053466, -0.0042229, -0.0053673, -0.0042649, -0.0004859, 0.0005490
4: 0.0017822, 0.0022601, 0.0018001, 0.0022689, -0.0002334, 0.0002066
5: 0.0071105, 0.0102159, 0.0072266, 0.0102729, -0.0015170, 0.0013427
6: -0.0010521, -0.0002639, -0.0010665, -0.0002934, -0.0003408, 0.0003850
7: -0.0058596, -0.0038204, -0.0058971, -0.0038966, -0.0008817, 0.0009962
8: -0.0026457, -0.0015733, -0.0026654, -0.0016134, -0.0004637, 0.0005239
9: -0.0000396, 0.0012039, 0.0000069, 0.0012268, -0.0006075, 0.0005377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006007
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006008
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9906429, 0.9924688, -0.0008120, 0.0009477
1: -0.0035861, -0.0031211, -0.0035955, -0.0031405, -0.0002023, 0.0002361
2: 0.0064862, 0.0089502, 0.0065892, 0.0090002, -0.0012514, 0.0010722
3: -0.0053469, -0.0042253, -0.0053696, -0.0042722, -0.0004880, 0.0005696
4: 0.0017833, 0.0022602, 0.0018032, 0.0022699, -0.0002422, 0.0002075
5: 0.0071173, 0.0102165, 0.0072469, 0.0102794, -0.0015739, 0.0013486
6: -0.0010522, -0.0002656, -0.0010682, -0.0002985, -0.0003423, 0.0003995
7: -0.0058601, -0.0038249, -0.0059013, -0.0039100, -0.0008856, 0.0010336
8: -0.0026459, -0.0015756, -0.0026676, -0.0016204, -0.0004657, 0.0005435
9: -0.0000369, 0.0012042, 0.0000150, 0.0012294, -0.0006303, 0.0005400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005520, upper bound: 0.0006013
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005520, upper bound: 0.0006015
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906482, 0.9925519, 0.9906334, 0.9924810, -0.0008158, 0.0009240
1: -0.0035942, -0.0031198, -0.0035979, -0.0031375, -0.0002033, 0.0002302
2: 0.0064794, 0.0089933, 0.0065731, 0.0090128, -0.0012201, 0.0010772
3: -0.0053665, -0.0042223, -0.0053754, -0.0042649, -0.0004903, 0.0005554
4: 0.0017820, 0.0022685, 0.0018001, 0.0022723, -0.0002362, 0.0002085
5: 0.0071088, 0.0102706, 0.0072266, 0.0102952, -0.0015346, 0.0013548
6: -0.0010660, -0.0002635, -0.0010722, -0.0002934, -0.0003439, 0.0003895
7: -0.0058956, -0.0038193, -0.0059117, -0.0038966, -0.0008897, 0.0010078
8: -0.0026646, -0.0015727, -0.0026731, -0.0016134, -0.0004679, 0.0005300
9: -0.0000403, 0.0012258, 0.0000069, 0.0012357, -0.0006145, 0.0005425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005794
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005902
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906479, 0.9925478, 0.9906303, 0.9924688, -0.0008194, 0.0009568
1: -0.0035943, -0.0031208, -0.0035986, -0.0031405, -0.0002042, 0.0002384
2: 0.0064848, 0.0089938, 0.0065892, 0.0090169, -0.0012634, 0.0010820
3: -0.0053667, -0.0042247, -0.0053772, -0.0042722, -0.0004925, 0.0005751
4: 0.0017830, 0.0022686, 0.0018032, 0.0022731, -0.0002445, 0.0002094
5: 0.0071156, 0.0102713, 0.0072469, 0.0103003, -0.0015890, 0.0013609
6: -0.0010661, -0.0002652, -0.0010735, -0.0002985, -0.0003454, 0.0004033
7: -0.0058960, -0.0038237, -0.0059151, -0.0039100, -0.0008937, 0.0010435
8: -0.0026648, -0.0015750, -0.0026748, -0.0016204, -0.0004700, 0.0005488
9: -0.0000376, 0.0012261, 0.0000150, 0.0012377, -0.0006363, 0.0005450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005639, upper bound: 0.0005794
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005639, upper bound: 0.0005906
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9905916, 0.9923964, -0.0007996, 0.0010508
1: -0.0035860, -0.0031201, -0.0036083, -0.0031586, -0.0001993, 0.0002618
2: 0.0064807, 0.0089497, 0.0066848, 0.0090679, -0.0013875, 0.0010559
3: -0.0053466, -0.0042229, -0.0054005, -0.0043157, -0.0004806, 0.0006315
4: 0.0017822, 0.0022601, 0.0018217, 0.0022830, -0.0002686, 0.0002044
5: 0.0071105, 0.0102159, 0.0073671, 0.0103645, -0.0017452, 0.0013281
6: -0.0010521, -0.0002639, -0.0010898, -0.0003290, -0.0003371, 0.0004429
7: -0.0058596, -0.0038204, -0.0059573, -0.0039889, -0.0008721, 0.0011460
8: -0.0026457, -0.0015733, -0.0026970, -0.0016619, -0.0004586, 0.0006027
9: -0.0000396, 0.0012039, 0.0000632, 0.0012635, -0.0006988, 0.0005318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0005824
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0006008
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9905849, 0.9923916, -0.0008036, 0.0010840
1: -0.0035861, -0.0031211, -0.0036100, -0.0031598, -0.0002002, 0.0002701
2: 0.0064862, 0.0089502, 0.0066911, 0.0090769, -0.0014315, 0.0010612
3: -0.0053469, -0.0042253, -0.0054045, -0.0043186, -0.0004830, 0.0006515
4: 0.0017833, 0.0022602, 0.0018229, 0.0022847, -0.0002771, 0.0002054
5: 0.0071173, 0.0102165, 0.0073751, 0.0103758, -0.0018004, 0.0013347
6: -0.0010522, -0.0002656, -0.0010927, -0.0003310, -0.0003388, 0.0004570
7: -0.0058601, -0.0038249, -0.0059647, -0.0039941, -0.0008765, 0.0011823
8: -0.0026459, -0.0015756, -0.0027009, -0.0016646, -0.0004609, 0.0006218
9: -0.0000369, 0.0012042, 0.0000664, 0.0012680, -0.0007210, 0.0005345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005824
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006014
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906482, 0.9925519, 0.9905782, 0.9923964, -0.0008063, 0.0010611
1: -0.0035942, -0.0031198, -0.0036116, -0.0031586, -0.0002009, 0.0002644
2: 0.0064794, 0.0089933, 0.0066848, 0.0090856, -0.0014012, 0.0010647
3: -0.0053665, -0.0042223, -0.0054085, -0.0043157, -0.0004846, 0.0006378
4: 0.0017820, 0.0022685, 0.0018217, 0.0022864, -0.0002712, 0.0002061
5: 0.0071088, 0.0102706, 0.0073671, 0.0103868, -0.0017623, 0.0013392
6: -0.0010660, -0.0002635, -0.0010954, -0.0003290, -0.0003399, 0.0004473
7: -0.0058956, -0.0038193, -0.0059718, -0.0039889, -0.0008794, 0.0011573
8: -0.0026646, -0.0015727, -0.0027047, -0.0016619, -0.0004625, 0.0006086
9: -0.0000403, 0.0012258, 0.0000632, 0.0012724, -0.0007057, 0.0005363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005861
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006008
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906479, 0.9925478, 0.9905720, 0.9923916, -0.0008109, 0.0010930
1: -0.0035943, -0.0031208, -0.0036132, -0.0031598, -0.0002021, 0.0002723
2: 0.0064848, 0.0089938, 0.0066911, 0.0090939, -0.0014433, 0.0010708
3: -0.0053667, -0.0042247, -0.0054123, -0.0043186, -0.0004874, 0.0006569
4: 0.0017830, 0.0022686, 0.0018229, 0.0022880, -0.0002793, 0.0002072
5: 0.0071156, 0.0102713, 0.0073751, 0.0103972, -0.0018153, 0.0013467
6: -0.0010661, -0.0002652, -0.0010981, -0.0003310, -0.0003418, 0.0004607
7: -0.0058960, -0.0038237, -0.0059787, -0.0039941, -0.0008844, 0.0011921
8: -0.0026648, -0.0015750, -0.0027083, -0.0016646, -0.0004651, 0.0006269
9: -0.0000376, 0.0012261, 0.0000664, 0.0012765, -0.0007269, 0.0005393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005862
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006014
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9906468, 0.9924810, -0.0009152, 0.0008779
1: -0.0036014, -0.0031398, -0.0035945, -0.0031375, -0.0002280, 0.0002188
2: 0.0065851, 0.0090316, 0.0065731, 0.0089951, -0.0011593, 0.0012085
3: -0.0053839, -0.0042704, -0.0053673, -0.0042649, -0.0005501, 0.0005277
4: 0.0018024, 0.0022759, 0.0018001, 0.0022689, -0.0002244, 0.0002339
5: 0.0072418, 0.0103188, 0.0072266, 0.0102729, -0.0014581, 0.0015200
6: -0.0010782, -0.0002972, -0.0010665, -0.0002934, -0.0003858, 0.0003701
7: -0.0059272, -0.0039066, -0.0058971, -0.0038966, -0.0009982, 0.0009575
8: -0.0026812, -0.0016186, -0.0026654, -0.0016134, -0.0005249, 0.0005035
9: 0.0000130, 0.0012452, 0.0000069, 0.0012268, -0.0005839, 0.0006087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0005715
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0006006
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9906429, 0.9924688, -0.0009189, 0.0009129
1: -0.0036015, -0.0031406, -0.0035955, -0.0031405, -0.0002290, 0.0002275
2: 0.0065894, 0.0090321, 0.0065892, 0.0090002, -0.0012055, 0.0012133
3: -0.0053841, -0.0042723, -0.0053696, -0.0042722, -0.0005523, 0.0005487
4: 0.0018033, 0.0022760, 0.0018032, 0.0022699, -0.0002333, 0.0002348
5: 0.0072472, 0.0103195, 0.0072469, 0.0102794, -0.0015162, 0.0015261
6: -0.0010784, -0.0002986, -0.0010682, -0.0002985, -0.0003873, 0.0003848
7: -0.0059277, -0.0039101, -0.0059013, -0.0039100, -0.0010021, 0.0009957
8: -0.0026814, -0.0016205, -0.0026676, -0.0016204, -0.0005270, 0.0005236
9: 0.0000151, 0.0012454, 0.0000150, 0.0012294, -0.0006072, 0.0006111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0005715
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0006006
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905870, 0.9924747, 0.9906334, 0.9924810, -0.0009233, 0.0008911
1: -0.0036094, -0.0031391, -0.0035979, -0.0031375, -0.0002301, 0.0002220
2: 0.0065813, 0.0090741, 0.0065731, 0.0090128, -0.0011767, 0.0012192
3: -0.0054032, -0.0042687, -0.0053754, -0.0042649, -0.0005549, 0.0005356
4: 0.0018017, 0.0022842, 0.0018001, 0.0022723, -0.0002278, 0.0002360
5: 0.0072370, 0.0103723, 0.0072266, 0.0102952, -0.0014800, 0.0015335
6: -0.0010918, -0.0002960, -0.0010722, -0.0002934, -0.0003892, 0.0003756
7: -0.0059623, -0.0039035, -0.0059117, -0.0038966, -0.0010070, 0.0009719
8: -0.0026997, -0.0016170, -0.0026731, -0.0016134, -0.0005296, 0.0005111
9: 0.0000111, 0.0012666, 0.0000069, 0.0012357, -0.0005927, 0.0006141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005574, upper bound: 0.0005661
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005558, upper bound: 0.0005704
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905867, 0.9924716, 0.9906303, 0.9924688, -0.0009270, 0.0009248
1: -0.0036095, -0.0031398, -0.0035986, -0.0031405, -0.0002310, 0.0002304
2: 0.0065854, 0.0090745, 0.0065892, 0.0090169, -0.0012212, 0.0012240
3: -0.0054034, -0.0042705, -0.0053772, -0.0042722, -0.0005571, 0.0005558
4: 0.0018025, 0.0022842, 0.0018032, 0.0022731, -0.0002364, 0.0002369
5: 0.0072422, 0.0103727, 0.0072469, 0.0103003, -0.0015360, 0.0015395
6: -0.0010919, -0.0002973, -0.0010735, -0.0002985, -0.0003907, 0.0003898
7: -0.0059626, -0.0039069, -0.0059151, -0.0039100, -0.0010110, 0.0010086
8: -0.0026998, -0.0016187, -0.0026748, -0.0016204, -0.0005317, 0.0005304
9: 0.0000131, 0.0012667, 0.0000150, 0.0012377, -0.0006151, 0.0006165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005553, upper bound: 0.0005670
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005534, upper bound: 0.0005708
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9905916, 0.9923964, -0.0008074, 0.0009134
1: -0.0036014, -0.0031398, -0.0036083, -0.0031586, -0.0002012, 0.0002276
2: 0.0065851, 0.0090316, 0.0066848, 0.0090679, -0.0012062, 0.0010662
3: -0.0053839, -0.0042704, -0.0054005, -0.0043157, -0.0004853, 0.0005490
4: 0.0018024, 0.0022759, 0.0018217, 0.0022830, -0.0002334, 0.0002064
5: 0.0072418, 0.0103188, 0.0073671, 0.0103645, -0.0015170, 0.0013410
6: -0.0010782, -0.0002972, -0.0010898, -0.0003290, -0.0003404, 0.0003850
7: -0.0059272, -0.0039066, -0.0059573, -0.0039889, -0.0008806, 0.0009962
8: -0.0026812, -0.0016186, -0.0026970, -0.0016619, -0.0004631, 0.0005239
9: 0.0000130, 0.0012452, 0.0000632, 0.0012635, -0.0006075, 0.0005370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005712
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006006
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9905849, 0.9923916, -0.0008107, 0.0009454
1: -0.0036015, -0.0031406, -0.0036100, -0.0031598, -0.0002020, 0.0002356
2: 0.0065894, 0.0090321, 0.0066911, 0.0090769, -0.0012484, 0.0010706
3: -0.0053841, -0.0042723, -0.0054045, -0.0043186, -0.0004873, 0.0005682
4: 0.0018033, 0.0022760, 0.0018229, 0.0022847, -0.0002416, 0.0002072
5: 0.0072472, 0.0103195, 0.0073751, 0.0103758, -0.0015702, 0.0013465
6: -0.0010784, -0.0002986, -0.0010927, -0.0003310, -0.0003418, 0.0003985
7: -0.0059277, -0.0039101, -0.0059647, -0.0039941, -0.0008842, 0.0010311
8: -0.0026814, -0.0016205, -0.0027009, -0.0016646, -0.0004650, 0.0005423
9: 0.0000151, 0.0012454, 0.0000664, 0.0012680, -0.0006288, 0.0005392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0005712
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0006006
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905870, 0.9924747, 0.9905782, 0.9923964, -0.0008161, 0.0009231
1: -0.0036094, -0.0031391, -0.0036116, -0.0031586, -0.0002034, 0.0002300
2: 0.0065813, 0.0090741, 0.0066848, 0.0090856, -0.0012190, 0.0010777
3: -0.0054032, -0.0042687, -0.0054085, -0.0043157, -0.0004905, 0.0005548
4: 0.0018017, 0.0022842, 0.0018217, 0.0022864, -0.0002359, 0.0002086
5: 0.0072370, 0.0103723, 0.0073671, 0.0103868, -0.0015332, 0.0013555
6: -0.0010918, -0.0002960, -0.0010954, -0.0003290, -0.0003440, 0.0003891
7: -0.0059623, -0.0039035, -0.0059718, -0.0039889, -0.0008901, 0.0010068
8: -0.0026997, -0.0016170, -0.0027047, -0.0016619, -0.0004681, 0.0005295
9: 0.0000111, 0.0012666, 0.0000632, 0.0012724, -0.0006139, 0.0005428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005767
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006006
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905867, 0.9924716, 0.9905720, 0.9923916, -0.0008197, 0.0009536
1: -0.0036095, -0.0031398, -0.0036132, -0.0031598, -0.0002042, 0.0002376
2: 0.0065854, 0.0090745, 0.0066911, 0.0090939, -0.0012592, 0.0010824
3: -0.0054034, -0.0042705, -0.0054123, -0.0043186, -0.0004927, 0.0005732
4: 0.0018025, 0.0022842, 0.0018229, 0.0022880, -0.0002437, 0.0002095
5: 0.0072422, 0.0103727, 0.0073751, 0.0103972, -0.0015838, 0.0013614
6: -0.0010919, -0.0002973, -0.0010981, -0.0003310, -0.0003455, 0.0004020
7: -0.0059626, -0.0039069, -0.0059787, -0.0039941, -0.0008940, 0.0010401
8: -0.0026998, -0.0016187, -0.0027083, -0.0016646, -0.0004701, 0.0005470
9: 0.0000131, 0.0012667, 0.0000664, 0.0012765, -0.0006342, 0.0005452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005767
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006006
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906345, 0.9925352, -0.0007623, 0.0008547
1: -0.0035860, -0.0031201, -0.0035976, -0.0031240, -0.0001899, 0.0002130
2: 0.0064807, 0.0089497, 0.0065014, 0.0090113, -0.0011286, 0.0010066
3: -0.0053466, -0.0042229, -0.0053747, -0.0042323, -0.0004581, 0.0005137
4: 0.0017822, 0.0022601, 0.0017862, 0.0022720, -0.0002184, 0.0001948
5: 0.0071105, 0.0102159, 0.0071365, 0.0102933, -0.0014195, 0.0012660
6: -0.0010521, -0.0002639, -0.0010717, -0.0002705, -0.0003213, 0.0003603
7: -0.0058596, -0.0038204, -0.0059104, -0.0038375, -0.0008314, 0.0009322
8: -0.0026457, -0.0015733, -0.0026724, -0.0015822, -0.0004372, 0.0004902
9: -0.0000396, 0.0012039, -0.0000292, 0.0012349, -0.0005684, 0.0005070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006007
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006009
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9906290, 0.9925265, -0.0007664, 0.0008895
1: -0.0035861, -0.0031211, -0.0035990, -0.0031261, -0.0001910, 0.0002217
2: 0.0064862, 0.0089502, 0.0065129, 0.0090186, -0.0011746, 0.0010121
3: -0.0053469, -0.0042253, -0.0053780, -0.0042375, -0.0004606, 0.0005346
4: 0.0017833, 0.0022602, 0.0017885, 0.0022734, -0.0002273, 0.0001959
5: 0.0071173, 0.0102165, 0.0071510, 0.0103025, -0.0014774, 0.0012729
6: -0.0010522, -0.0002656, -0.0010740, -0.0002742, -0.0003231, 0.0003750
7: -0.0058601, -0.0038249, -0.0059165, -0.0038470, -0.0008359, 0.0009702
8: -0.0026459, -0.0015756, -0.0026756, -0.0015872, -0.0004396, 0.0005102
9: -0.0000369, 0.0012042, -0.0000234, 0.0012386, -0.0005916, 0.0005097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005411, upper bound: 0.0005795
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005411, upper bound: 0.0005907
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906482, 0.9925519, 0.9906209, 0.9925352, -0.0007699, 0.0008638
1: -0.0035942, -0.0031198, -0.0036010, -0.0031240, -0.0001918, 0.0002152
2: 0.0064794, 0.0089933, 0.0065014, 0.0090293, -0.0011406, 0.0010166
3: -0.0053665, -0.0042223, -0.0053829, -0.0042323, -0.0004627, 0.0005191
4: 0.0017820, 0.0022685, 0.0017862, 0.0022755, -0.0002208, 0.0001968
5: 0.0071088, 0.0102706, 0.0071365, 0.0103159, -0.0014345, 0.0012786
6: -0.0010660, -0.0002635, -0.0010775, -0.0002705, -0.0003245, 0.0003641
7: -0.0058956, -0.0038193, -0.0059254, -0.0038375, -0.0008397, 0.0009420
8: -0.0026646, -0.0015727, -0.0026802, -0.0015822, -0.0004416, 0.0004954
9: -0.0000403, 0.0012258, -0.0000292, 0.0012440, -0.0005745, 0.0005120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005794
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005902
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906479, 0.9925478, 0.9906155, 0.9925265, -0.0007749, 0.0008959
1: -0.0035943, -0.0031208, -0.0036023, -0.0031261, -0.0001931, 0.0002232
2: 0.0064848, 0.0089938, 0.0065129, 0.0090363, -0.0011830, 0.0010232
3: -0.0053667, -0.0042247, -0.0053861, -0.0042375, -0.0004657, 0.0005384
4: 0.0017830, 0.0022686, 0.0017885, 0.0022768, -0.0002290, 0.0001980
5: 0.0071156, 0.0102713, 0.0071510, 0.0103248, -0.0014879, 0.0012869
6: -0.0010661, -0.0002652, -0.0010797, -0.0002742, -0.0003266, 0.0003776
7: -0.0058960, -0.0038237, -0.0059311, -0.0038470, -0.0008451, 0.0009771
8: -0.0026648, -0.0015750, -0.0026833, -0.0015872, -0.0004444, 0.0005138
9: -0.0000376, 0.0012261, -0.0000234, 0.0012475, -0.0005958, 0.0005153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005795
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005907
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9905726, 0.9924576, -0.0007538, 0.0009922
1: -0.0035860, -0.0031201, -0.0036130, -0.0031433, -0.0001878, 0.0002472
2: 0.0064807, 0.0089497, 0.0066040, 0.0090930, -0.0013102, 0.0009954
3: -0.0053466, -0.0042229, -0.0054119, -0.0042790, -0.0004531, 0.0005963
4: 0.0017822, 0.0022601, 0.0018061, 0.0022878, -0.0002536, 0.0001927
5: 0.0071105, 0.0102159, 0.0072656, 0.0103961, -0.0016478, 0.0012519
6: -0.0010521, -0.0002639, -0.0010978, -0.0003033, -0.0003178, 0.0004182
7: -0.0058596, -0.0038204, -0.0059780, -0.0039222, -0.0008221, 0.0010821
8: -0.0026457, -0.0015733, -0.0027079, -0.0016268, -0.0004323, 0.0005691
9: -0.0000396, 0.0012039, 0.0000225, 0.0012761, -0.0006599, 0.0005013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0005824
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0006008
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9905646, 0.9924538, -0.0007583, 0.0010259
1: -0.0035861, -0.0031211, -0.0036150, -0.0031443, -0.0001890, 0.0002556
2: 0.0064862, 0.0089502, 0.0066090, 0.0091036, -0.0013547, 0.0010014
3: -0.0053469, -0.0042253, -0.0054167, -0.0042813, -0.0004558, 0.0006166
4: 0.0017833, 0.0022602, 0.0018071, 0.0022899, -0.0002622, 0.0001938
5: 0.0071173, 0.0102165, 0.0072719, 0.0104094, -0.0017039, 0.0012595
6: -0.0010522, -0.0002656, -0.0011012, -0.0003048, -0.0003197, 0.0004325
7: -0.0058601, -0.0038249, -0.0059867, -0.0039263, -0.0008271, 0.0011189
8: -0.0026459, -0.0015756, -0.0027125, -0.0016290, -0.0004350, 0.0005884
9: -0.0000369, 0.0012042, 0.0000250, 0.0012814, -0.0006823, 0.0005044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005824
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006015
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906482, 0.9925519, 0.9905592, 0.9924576, -0.0007609, 0.0010010
1: -0.0035942, -0.0031198, -0.0036164, -0.0031433, -0.0001896, 0.0002494
2: 0.0064794, 0.0089933, 0.0066040, 0.0091108, -0.0013217, 0.0010048
3: -0.0053665, -0.0042223, -0.0054200, -0.0042790, -0.0004573, 0.0006016
4: 0.0017820, 0.0022685, 0.0018061, 0.0022913, -0.0002558, 0.0001945
5: 0.0071088, 0.0102706, 0.0072656, 0.0104185, -0.0016624, 0.0012638
6: -0.0010660, -0.0002635, -0.0011035, -0.0003033, -0.0003208, 0.0004219
7: -0.0058956, -0.0038193, -0.0059927, -0.0039222, -0.0008299, 0.0010917
8: -0.0026646, -0.0015727, -0.0027156, -0.0016268, -0.0004364, 0.0005741
9: -0.0000403, 0.0012258, 0.0000225, 0.0012851, -0.0006657, 0.0005061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005861
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006008
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906479, 0.9925478, 0.9905514, 0.9924538, -0.0007668, 0.0010320
1: -0.0035943, -0.0031208, -0.0036183, -0.0031443, -0.0001911, 0.0002572
2: 0.0064848, 0.0089938, 0.0066090, 0.0091212, -0.0013628, 0.0010125
3: -0.0053667, -0.0042247, -0.0054247, -0.0042813, -0.0004608, 0.0006203
4: 0.0017830, 0.0022686, 0.0018071, 0.0022933, -0.0002638, 0.0001960
5: 0.0071156, 0.0102713, 0.0072719, 0.0104315, -0.0017140, 0.0012735
6: -0.0010661, -0.0002652, -0.0011068, -0.0003048, -0.0003232, 0.0004350
7: -0.0058960, -0.0038237, -0.0060012, -0.0039263, -0.0008363, 0.0011256
8: -0.0026648, -0.0015750, -0.0027201, -0.0016290, -0.0004398, 0.0005919
9: -0.0000376, 0.0012261, 0.0000250, 0.0012903, -0.0006864, 0.0005099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005466, upper bound: 0.0005727
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005391, upper bound: 0.0005780
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9906345, 0.9925352, -0.0009063, 0.0008416
1: -0.0036014, -0.0031398, -0.0035976, -0.0031240, -0.0002258, 0.0002097
2: 0.0065851, 0.0090316, 0.0065014, 0.0090113, -0.0011114, 0.0011968
3: -0.0053839, -0.0042704, -0.0053747, -0.0042323, -0.0005447, 0.0005059
4: 0.0018024, 0.0022759, 0.0017862, 0.0022720, -0.0002151, 0.0002316
5: 0.0072418, 0.0103188, 0.0071365, 0.0102933, -0.0013978, 0.0015052
6: -0.0010782, -0.0002972, -0.0010717, -0.0002705, -0.0003820, 0.0003548
7: -0.0059272, -0.0039066, -0.0059104, -0.0038375, -0.0009885, 0.0009179
8: -0.0026812, -0.0016186, -0.0026724, -0.0015822, -0.0005198, 0.0004827
9: 0.0000130, 0.0012452, -0.0000292, 0.0012349, -0.0005598, 0.0006028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0005715
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0006006
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9906290, 0.9925265, -0.0009108, 0.0008755
1: -0.0036015, -0.0031406, -0.0035990, -0.0031261, -0.0002269, 0.0002182
2: 0.0065894, 0.0090321, 0.0065129, 0.0090186, -0.0011561, 0.0012026
3: -0.0053841, -0.0042723, -0.0053780, -0.0042375, -0.0005474, 0.0005262
4: 0.0018033, 0.0022760, 0.0017885, 0.0022734, -0.0002238, 0.0002328
5: 0.0072472, 0.0103195, 0.0071510, 0.0103025, -0.0014541, 0.0015126
6: -0.0010784, -0.0002986, -0.0010740, -0.0002742, -0.0003839, 0.0003691
7: -0.0059277, -0.0039101, -0.0059165, -0.0038470, -0.0009933, 0.0009549
8: -0.0026814, -0.0016205, -0.0026756, -0.0015872, -0.0005224, 0.0005022
9: 0.0000151, 0.0012454, -0.0000234, 0.0012386, -0.0005823, 0.0006057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005297, upper bound: 0.0005643
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005261, upper bound: 0.0005675
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905870, 0.9924747, 0.9906209, 0.9925352, -0.0009121, 0.0008541
1: -0.0036094, -0.0031391, -0.0036010, -0.0031240, -0.0002273, 0.0002128
2: 0.0065813, 0.0090741, 0.0065014, 0.0090293, -0.0011278, 0.0012044
3: -0.0054032, -0.0042687, -0.0053829, -0.0042323, -0.0005482, 0.0005133
4: 0.0018017, 0.0022842, 0.0017862, 0.0022755, -0.0002183, 0.0002331
5: 0.0072370, 0.0103723, 0.0071365, 0.0103159, -0.0014185, 0.0015148
6: -0.0010918, -0.0002960, -0.0010775, -0.0002705, -0.0003845, 0.0003600
7: -0.0059623, -0.0039035, -0.0059254, -0.0038375, -0.0009947, 0.0009315
8: -0.0026997, -0.0016170, -0.0026802, -0.0015822, -0.0005231, 0.0004899
9: 0.0000111, 0.0012666, -0.0000292, 0.0012440, -0.0005680, 0.0006066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005574, upper bound: 0.0005661
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005559, upper bound: 0.0005704
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905867, 0.9924716, 0.9906155, 0.9925265, -0.0009173, 0.0008856
1: -0.0036095, -0.0031398, -0.0036023, -0.0031261, -0.0002286, 0.0002207
2: 0.0065854, 0.0090745, 0.0065129, 0.0090363, -0.0011694, 0.0012112
3: -0.0054034, -0.0042705, -0.0053861, -0.0042375, -0.0005513, 0.0005323
4: 0.0018025, 0.0022842, 0.0017885, 0.0022768, -0.0002263, 0.0002344
5: 0.0072422, 0.0103727, 0.0071510, 0.0103248, -0.0014708, 0.0015234
6: -0.0010919, -0.0002973, -0.0010797, -0.0002742, -0.0003867, 0.0003733
7: -0.0059626, -0.0039069, -0.0059311, -0.0038470, -0.0010004, 0.0009658
8: -0.0026998, -0.0016187, -0.0026833, -0.0015872, -0.0005261, 0.0005079
9: 0.0000131, 0.0012667, -0.0000234, 0.0012475, -0.0005890, 0.0006100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005555, upper bound: 0.0005670
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005534, upper bound: 0.0005708
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9905726, 0.9924576, -0.0007667, 0.0008601
1: -0.0036014, -0.0031398, -0.0036130, -0.0031433, -0.0001910, 0.0002143
2: 0.0065851, 0.0090316, 0.0066040, 0.0090930, -0.0011358, 0.0010125
3: -0.0053839, -0.0042704, -0.0054119, -0.0042790, -0.0004608, 0.0005170
4: 0.0018024, 0.0022759, 0.0018061, 0.0022878, -0.0002198, 0.0001960
5: 0.0072418, 0.0103188, 0.0072656, 0.0103961, -0.0014286, 0.0012734
6: -0.0010782, -0.0002972, -0.0010978, -0.0003033, -0.0003232, 0.0003626
7: -0.0059272, -0.0039066, -0.0059780, -0.0039222, -0.0008362, 0.0009381
8: -0.0026812, -0.0016186, -0.0027079, -0.0016268, -0.0004398, 0.0004933
9: 0.0000130, 0.0012452, 0.0000225, 0.0012761, -0.0005721, 0.0005099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005711
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006006
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9905646, 0.9924538, -0.0007700, 0.0008908
1: -0.0036015, -0.0031406, -0.0036150, -0.0031443, -0.0001919, 0.0002220
2: 0.0065894, 0.0090321, 0.0066090, 0.0091036, -0.0011763, 0.0010168
3: -0.0053841, -0.0042723, -0.0054167, -0.0042813, -0.0004628, 0.0005354
4: 0.0018033, 0.0022760, 0.0018071, 0.0022899, -0.0002277, 0.0001968
5: 0.0072472, 0.0103195, 0.0072719, 0.0104094, -0.0014795, 0.0012788
6: -0.0010784, -0.0002986, -0.0011012, -0.0003048, -0.0003246, 0.0003755
7: -0.0059277, -0.0039101, -0.0059867, -0.0039263, -0.0008398, 0.0009716
8: -0.0026814, -0.0016205, -0.0027125, -0.0016290, -0.0004416, 0.0005109
9: 0.0000151, 0.0012454, 0.0000250, 0.0012814, -0.0005925, 0.0005121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0005711
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0006006
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905870, 0.9924747, 0.9905592, 0.9924576, -0.0007736, 0.0008691
1: -0.0036094, -0.0031391, -0.0036164, -0.0031433, -0.0001928, 0.0002166
2: 0.0065813, 0.0090741, 0.0066040, 0.0091108, -0.0011477, 0.0010216
3: -0.0054032, -0.0042687, -0.0054200, -0.0042790, -0.0004650, 0.0005224
4: 0.0018017, 0.0022842, 0.0018061, 0.0022913, -0.0002221, 0.0001977
5: 0.0072370, 0.0103723, 0.0072656, 0.0104185, -0.0014435, 0.0012849
6: -0.0010918, -0.0002960, -0.0011035, -0.0003033, -0.0003261, 0.0003664
7: -0.0059623, -0.0039035, -0.0059927, -0.0039222, -0.0008437, 0.0009479
8: -0.0026997, -0.0016170, -0.0027156, -0.0016268, -0.0004437, 0.0004985
9: 0.0000111, 0.0012666, 0.0000225, 0.0012851, -0.0005780, 0.0005145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005767
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006006
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9905867, 0.9924716, 0.9905514, 0.9924538, -0.0007782, 0.0008978
1: -0.0036095, -0.0031398, -0.0036183, -0.0031443, -0.0001939, 0.0002237
2: 0.0065854, 0.0090745, 0.0066090, 0.0091212, -0.0011855, 0.0010276
3: -0.0054034, -0.0042705, -0.0054247, -0.0042813, -0.0004677, 0.0005396
4: 0.0018025, 0.0022842, 0.0018071, 0.0022933, -0.0002295, 0.0001989
5: 0.0072422, 0.0103727, 0.0072719, 0.0104315, -0.0014911, 0.0012924
6: -0.0010919, -0.0002973, -0.0011068, -0.0003048, -0.0003280, 0.0003785
7: -0.0059626, -0.0039069, -0.0060012, -0.0039263, -0.0008487, 0.0009792
8: -0.0026998, -0.0016187, -0.0027201, -0.0016290, -0.0004463, 0.0005149
9: 0.0000131, 0.0012667, 0.0000250, 0.0012903, -0.0005971, 0.0005175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005466, upper bound: 0.0005670
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005391, upper bound: 0.0005708
time: 0.70 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005818, upper bound: 0.0005417
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0005506
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005924, upper bound: 0.0005521
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005924, upper bound: 0.0005639
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005806, upper bound: 0.0005424
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005806, upper bound: 0.0005506
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005906, upper bound: 0.0005538
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005906, upper bound: 0.0005639
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005812, upper bound: 0.0005331
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005389
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0005337
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005780, upper bound: 0.0005391
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005596
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005750
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005469
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005708, upper bound: 0.0005534
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005460
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005548, upper bound: 0.0005694
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005337
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005708, upper bound: 0.0005391
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006007
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006008
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005520, upper bound: 0.0006013
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005520, upper bound: 0.0006015
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005794
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005902
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005639, upper bound: 0.0005794
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005639, upper bound: 0.0005906
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0005824
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0006008
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005824
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006014
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005861
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006008
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005862
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006014
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0005715
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0006006
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0005715
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0006006
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005574, upper bound: 0.0005661
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005558, upper bound: 0.0005704
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005553, upper bound: 0.0005670
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005534, upper bound: 0.0005708
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005712
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006006
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0005712
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0006006
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005767
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006006
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0005767
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006006
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006007
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006009
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005411, upper bound: 0.0005795
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005411, upper bound: 0.0005907
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005794
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005677, upper bound: 0.0005902
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005795
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005907
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0005824
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005092, upper bound: 0.0006008
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005824
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006015
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005861
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006008
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005466, upper bound: 0.0005727
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005391, upper bound: 0.0005780
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0005715
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005326, upper bound: 0.0006006
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005297, upper bound: 0.0005643
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005261, upper bound: 0.0005675
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005574, upper bound: 0.0005661
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005559, upper bound: 0.0005704
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005555, upper bound: 0.0005670
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005534, upper bound: 0.0005708
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0005711
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005086, upper bound: 0.0006006
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0005711
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005077, upper bound: 0.0006006
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0005767
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005448, upper bound: 0.0006006
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005466, upper bound: 0.0005670
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 0, lower bound: -0.0005391, upper bound: 0.0005708

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906927, 0.9925168, -0.0008323, 0.0008561
1: -0.0035860, -0.0031201, -0.0035831, -0.0031286, -0.0002074, 0.0002133
2: 0.0064807, 0.0089497, 0.0065258, 0.0089346, -0.0011305, 0.0010991
3: -0.0053466, -0.0042229, -0.0053398, -0.0042434, -0.0005003, 0.0005145
4: 0.0017822, 0.0022601, 0.0017909, 0.0022572, -0.0002188, 0.0002127
5: 0.0071105, 0.0102159, 0.0071672, 0.0101968, -0.0014219, 0.0013824
6: -0.0010521, -0.0002639, -0.0010472, -0.0002783, -0.0003509, 0.0003609
7: -0.0058596, -0.0038204, -0.0058471, -0.0038576, -0.0009078, 0.0009337
8: -0.0026457, -0.0015733, -0.0026391, -0.0015928, -0.0004774, 0.0004910
9: -0.0000396, 0.0012039, -0.0000169, 0.0011963, -0.0005694, 0.0005536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005136, upper bound: 0.0005775
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005150, upper bound: 0.0005748
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906629, 0.9924810, -0.0008054, 0.0008930
1: -0.0035860, -0.0031201, -0.0035905, -0.0031375, -0.0002007, 0.0002225
2: 0.0064807, 0.0089497, 0.0065731, 0.0089739, -0.0011792, 0.0010635
3: -0.0053466, -0.0042229, -0.0053576, -0.0042649, -0.0004841, 0.0005367
4: 0.0017822, 0.0022601, 0.0018001, 0.0022648, -0.0002282, 0.0002058
5: 0.0071105, 0.0102159, 0.0072266, 0.0102462, -0.0014831, 0.0013376
6: -0.0010521, -0.0002639, -0.0010598, -0.0002934, -0.0003395, 0.0003764
7: -0.0058596, -0.0038204, -0.0058796, -0.0038966, -0.0008784, 0.0009739
8: -0.0026457, -0.0015733, -0.0026561, -0.0016134, -0.0004619, 0.0005122
9: -0.0000396, 0.0012039, 0.0000069, 0.0012161, -0.0005939, 0.0005356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005136, upper bound: 0.0005902
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005150, upper bound: 0.0005902
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9906869, 0.9925076, -0.0008362, 0.0008937
1: -0.0035861, -0.0031211, -0.0035845, -0.0031309, -0.0002084, 0.0002227
2: 0.0064862, 0.0089502, 0.0065379, 0.0089420, -0.0011801, 0.0011042
3: -0.0053469, -0.0042253, -0.0053432, -0.0042489, -0.0005026, 0.0005371
4: 0.0017833, 0.0022602, 0.0017933, 0.0022586, -0.0002284, 0.0002137
5: 0.0071173, 0.0102165, 0.0071824, 0.0102062, -0.0014842, 0.0013888
6: -0.0010522, -0.0002656, -0.0010496, -0.0002821, -0.0003525, 0.0003767
7: -0.0058601, -0.0038249, -0.0058533, -0.0038676, -0.0009120, 0.0009747
8: -0.0026459, -0.0015756, -0.0026423, -0.0015981, -0.0004796, 0.0005126
9: -0.0000369, 0.0012042, -0.0000108, 0.0012001, -0.0005944, 0.0005561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005087, upper bound: 0.0005775
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005093, upper bound: 0.0005748
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9906591, 0.9924688, -0.0008093, 0.0009314
1: -0.0035861, -0.0031211, -0.0035915, -0.0031405, -0.0002017, 0.0002321
2: 0.0064862, 0.0089502, 0.0065892, 0.0089789, -0.0012299, 0.0010687
3: -0.0053469, -0.0042253, -0.0053600, -0.0042722, -0.0004864, 0.0005598
4: 0.0017833, 0.0022602, 0.0018032, 0.0022657, -0.0002380, 0.0002068
5: 0.0071173, 0.0102165, 0.0072469, 0.0102526, -0.0015469, 0.0013441
6: -0.0010522, -0.0002656, -0.0010614, -0.0002985, -0.0003411, 0.0003926
7: -0.0058601, -0.0038249, -0.0058838, -0.0039100, -0.0008827, 0.0010158
8: -0.0026459, -0.0015756, -0.0026584, -0.0016204, -0.0004642, 0.0005342
9: -0.0000369, 0.0012042, 0.0000150, 0.0012186, -0.0006194, 0.0005382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005087, upper bound: 0.0005906
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005093, upper bound: 0.0005906
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906080, 0.9923964, -0.0007964, 0.0010329
1: -0.0035860, -0.0031201, -0.0036042, -0.0031586, -0.0001984, 0.0002574
2: 0.0064807, 0.0089497, 0.0066848, 0.0090463, -0.0013639, 0.0010516
3: -0.0053466, -0.0042229, -0.0053906, -0.0043157, -0.0004786, 0.0006208
4: 0.0017822, 0.0022601, 0.0018217, 0.0022788, -0.0002640, 0.0002035
5: 0.0071105, 0.0102159, 0.0073671, 0.0103373, -0.0017155, 0.0013226
6: -0.0010521, -0.0002639, -0.0010829, -0.0003290, -0.0003357, 0.0004354
7: -0.0058596, -0.0038204, -0.0059394, -0.0039889, -0.0008685, 0.0011265
8: -0.0026457, -0.0015733, -0.0026876, -0.0016619, -0.0004568, 0.0005924
9: -0.0000396, 0.0012039, 0.0000632, 0.0012526, -0.0006870, 0.0005296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003275, upper bound: 0.0005797
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003156, upper bound: 0.0005756
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9906011, 0.9923916, -0.0008004, 0.0010683
1: -0.0035861, -0.0031211, -0.0036059, -0.0031598, -0.0001994, 0.0002662
2: 0.0064862, 0.0089502, 0.0066911, 0.0090555, -0.0014106, 0.0010570
3: -0.0053469, -0.0042253, -0.0053948, -0.0043186, -0.0004811, 0.0006421
4: 0.0017833, 0.0022602, 0.0018229, 0.0022806, -0.0002730, 0.0002046
5: 0.0071173, 0.0102165, 0.0073751, 0.0103488, -0.0017742, 0.0013294
6: -0.0010522, -0.0002656, -0.0010858, -0.0003310, -0.0003374, 0.0004503
7: -0.0058601, -0.0038249, -0.0059469, -0.0039941, -0.0008730, 0.0011651
8: -0.0026459, -0.0015756, -0.0026916, -0.0016646, -0.0004591, 0.0006127
9: -0.0000369, 0.0012042, 0.0000664, 0.0012572, -0.0007105, 0.0005323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003406, upper bound: 0.0005814
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003327, upper bound: 0.0005762
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906482, 0.9925519, 0.9905949, 0.9923964, -0.0008030, 0.0010481
1: -0.0035942, -0.0031198, -0.0036075, -0.0031586, -0.0002001, 0.0002612
2: 0.0064794, 0.0089933, 0.0066848, 0.0090636, -0.0013841, 0.0010604
3: -0.0053665, -0.0042223, -0.0053985, -0.0043157, -0.0004826, 0.0006300
4: 0.0017820, 0.0022685, 0.0018217, 0.0022821, -0.0002679, 0.0002052
5: 0.0071088, 0.0102706, 0.0073671, 0.0103591, -0.0017408, 0.0013337
6: -0.0010660, -0.0002635, -0.0010884, -0.0003290, -0.0003385, 0.0004418
7: -0.0058956, -0.0038193, -0.0059537, -0.0039889, -0.0008758, 0.0011431
8: -0.0026646, -0.0015727, -0.0026951, -0.0016619, -0.0004606, 0.0006012
9: -0.0000403, 0.0012258, 0.0000632, 0.0012613, -0.0006971, 0.0005341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004458, upper bound: 0.0005356
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004443, upper bound: 0.0005774
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906479, 0.9925478, 0.9905881, 0.9923916, -0.0008076, 0.0010825
1: -0.0035943, -0.0031208, -0.0036091, -0.0031598, -0.0002012, 0.0002697
2: 0.0064848, 0.0089938, 0.0066911, 0.0090725, -0.0014294, 0.0010664
3: -0.0053667, -0.0042247, -0.0054025, -0.0043186, -0.0004854, 0.0006506
4: 0.0017830, 0.0022686, 0.0018229, 0.0022839, -0.0002767, 0.0002064
5: 0.0071156, 0.0102713, 0.0073751, 0.0103703, -0.0017978, 0.0013413
6: -0.0010661, -0.0002652, -0.0010913, -0.0003310, -0.0003404, 0.0004563
7: -0.0058960, -0.0038237, -0.0059610, -0.0039941, -0.0008808, 0.0011806
8: -0.0026648, -0.0015750, -0.0026990, -0.0016646, -0.0004632, 0.0006208
9: -0.0000376, 0.0012261, 0.0000664, 0.0012658, -0.0007199, 0.0005371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004526, upper bound: 0.0005816
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004514, upper bound: 0.0005780
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9906629, 0.9924810, -0.0009122, 0.0008603
1: -0.0036014, -0.0031398, -0.0035905, -0.0031375, -0.0002273, 0.0002144
2: 0.0065851, 0.0090316, 0.0065731, 0.0089739, -0.0011360, 0.0012045
3: -0.0053839, -0.0042704, -0.0053576, -0.0042649, -0.0005482, 0.0005170
4: 0.0018024, 0.0022759, 0.0018001, 0.0022648, -0.0002199, 0.0002331
5: 0.0072418, 0.0103188, 0.0072266, 0.0102462, -0.0014288, 0.0015150
6: -0.0010782, -0.0002972, -0.0010598, -0.0002934, -0.0003845, 0.0003626
7: -0.0059272, -0.0039066, -0.0058796, -0.0038966, -0.0009949, 0.0009382
8: -0.0026812, -0.0016186, -0.0026561, -0.0016134, -0.0005232, 0.0004934
9: 0.0000130, 0.0012452, 0.0000069, 0.0012161, -0.0005721, 0.0006067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003903, upper bound: 0.0005758
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003782, upper bound: 0.0005666
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9906591, 0.9924688, -0.0009162, 0.0008984
1: -0.0036015, -0.0031406, -0.0035915, -0.0031405, -0.0002283, 0.0002239
2: 0.0065894, 0.0090321, 0.0065892, 0.0089789, -0.0011864, 0.0012098
3: -0.0053841, -0.0042723, -0.0053600, -0.0042722, -0.0005506, 0.0005400
4: 0.0018033, 0.0022760, 0.0018032, 0.0022657, -0.0002296, 0.0002342
5: 0.0072472, 0.0103195, 0.0072469, 0.0102526, -0.0014921, 0.0015216
6: -0.0010784, -0.0002986, -0.0010614, -0.0002985, -0.0003862, 0.0003787
7: -0.0059277, -0.0039101, -0.0058838, -0.0039100, -0.0009992, 0.0009799
8: -0.0026814, -0.0016205, -0.0026584, -0.0016204, -0.0005255, 0.0005153
9: 0.0000151, 0.0012454, 0.0000150, 0.0012186, -0.0005975, 0.0006093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003881, upper bound: 0.0005772
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003743, upper bound: 0.0005675
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9906080, 0.9923964, -0.0008045, 0.0008894
1: -0.0036014, -0.0031398, -0.0036042, -0.0031586, -0.0002005, 0.0002216
2: 0.0065851, 0.0090316, 0.0066848, 0.0090463, -0.0011745, 0.0010623
3: -0.0053839, -0.0042704, -0.0053906, -0.0043157, -0.0004835, 0.0005346
4: 0.0018024, 0.0022759, 0.0018217, 0.0022788, -0.0002273, 0.0002056
5: 0.0072418, 0.0103188, 0.0073671, 0.0103373, -0.0014772, 0.0013361
6: -0.0010782, -0.0002972, -0.0010829, -0.0003290, -0.0003391, 0.0003749
7: -0.0059272, -0.0039066, -0.0059394, -0.0039889, -0.0008774, 0.0009701
8: -0.0026812, -0.0016186, -0.0026876, -0.0016619, -0.0004614, 0.0005101
9: 0.0000130, 0.0012452, 0.0000632, 0.0012526, -0.0005915, 0.0005350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003219, upper bound: 0.0005758
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003007, upper bound: 0.0005665
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9906011, 0.9923916, -0.0008080, 0.0009271
1: -0.0036015, -0.0031406, -0.0036059, -0.0031598, -0.0002013, 0.0002310
2: 0.0065894, 0.0090321, 0.0066911, 0.0090555, -0.0012243, 0.0010669
3: -0.0053841, -0.0042723, -0.0053948, -0.0043186, -0.0004856, 0.0005572
4: 0.0018033, 0.0022760, 0.0018229, 0.0022806, -0.0002370, 0.0002065
5: 0.0072472, 0.0103195, 0.0073751, 0.0103488, -0.0015398, 0.0013419
6: -0.0010784, -0.0002986, -0.0010858, -0.0003310, -0.0003406, 0.0003908
7: -0.0059277, -0.0039101, -0.0059469, -0.0039941, -0.0008812, 0.0010112
8: -0.0026814, -0.0016205, -0.0026916, -0.0016646, -0.0004634, 0.0005318
9: 0.0000151, 0.0012454, 0.0000664, 0.0012572, -0.0006166, 0.0005373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003321, upper bound: 0.0004728
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003132, upper bound: 0.0005673
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905870, 0.9924747, 0.9905949, 0.9923964, -0.0008131, 0.0009062
1: -0.0036094, -0.0031391, -0.0036075, -0.0031586, -0.0002026, 0.0002258
2: 0.0065813, 0.0090741, 0.0066848, 0.0090636, -0.0011966, 0.0010737
3: -0.0054032, -0.0042687, -0.0053985, -0.0043157, -0.0004887, 0.0005446
4: 0.0018017, 0.0022842, 0.0018217, 0.0022821, -0.0002316, 0.0002078
5: 0.0072370, 0.0103723, 0.0073671, 0.0103591, -0.0015050, 0.0013504
6: -0.0010918, -0.0002960, -0.0010884, -0.0003290, -0.0003427, 0.0003820
7: -0.0059623, -0.0039035, -0.0059537, -0.0039889, -0.0008868, 0.0009883
8: -0.0026997, -0.0016170, -0.0026951, -0.0016619, -0.0004663, 0.0005198
9: 0.0000111, 0.0012666, 0.0000632, 0.0012613, -0.0006027, 0.0005407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0005760
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004340, upper bound: 0.0005704
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905867, 0.9924716, 0.9905881, 0.9923916, -0.0008166, 0.0009417
1: -0.0036095, -0.0031398, -0.0036091, -0.0031598, -0.0002035, 0.0002347
2: 0.0065854, 0.0090745, 0.0066911, 0.0090725, -0.0012435, 0.0010783
3: -0.0054034, -0.0042705, -0.0054025, -0.0043186, -0.0004908, 0.0005660
4: 0.0018025, 0.0022842, 0.0018229, 0.0022839, -0.0002407, 0.0002087
5: 0.0072422, 0.0103727, 0.0073751, 0.0103703, -0.0015641, 0.0013562
6: -0.0010919, -0.0002973, -0.0010913, -0.0003310, -0.0003442, 0.0003970
7: -0.0059626, -0.0039069, -0.0059610, -0.0039941, -0.0008906, 0.0010271
8: -0.0026998, -0.0016187, -0.0026990, -0.0016646, -0.0004684, 0.0005401
9: 0.0000131, 0.0012667, 0.0000664, 0.0012658, -0.0006263, 0.0005431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004443, upper bound: 0.0005776
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004411, upper bound: 0.0004704
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906772, 0.9925734, -0.0008122, 0.0007973
1: -0.0035860, -0.0031201, -0.0035869, -0.0031145, -0.0002024, 0.0001987
2: 0.0064807, 0.0089497, 0.0064511, 0.0089549, -0.0010528, 0.0010725
3: -0.0053466, -0.0042229, -0.0053490, -0.0042094, -0.0004882, 0.0004792
4: 0.0017822, 0.0022601, 0.0017765, 0.0022611, -0.0002038, 0.0002076
5: 0.0071105, 0.0102159, 0.0070732, 0.0102223, -0.0013241, 0.0013489
6: -0.0010521, -0.0002639, -0.0010537, -0.0002544, -0.0003424, 0.0003361
7: -0.0058596, -0.0038204, -0.0058639, -0.0037959, -0.0008858, 0.0008695
8: -0.0026457, -0.0015733, -0.0026479, -0.0015604, -0.0004658, 0.0004573
9: -0.0000396, 0.0012039, -0.0000545, 0.0012065, -0.0005302, 0.0005402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005138, upper bound: 0.0005775
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005154, upper bound: 0.0005748
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9906514, 0.9925352, -0.0007592, 0.0008134
1: -0.0035860, -0.0031201, -0.0035934, -0.0031240, -0.0001892, 0.0002027
2: 0.0064807, 0.0089497, 0.0065014, 0.0089891, -0.0010741, 0.0010025
3: -0.0053466, -0.0042229, -0.0053646, -0.0042323, -0.0004563, 0.0004889
4: 0.0017822, 0.0022601, 0.0017862, 0.0022677, -0.0002079, 0.0001940
5: 0.0071105, 0.0102159, 0.0071365, 0.0102654, -0.0013509, 0.0012609
6: -0.0010521, -0.0002639, -0.0010646, -0.0002705, -0.0003200, 0.0003429
7: -0.0058596, -0.0038204, -0.0058922, -0.0038375, -0.0008280, 0.0008871
8: -0.0026457, -0.0015733, -0.0026628, -0.0015822, -0.0004355, 0.0004665
9: -0.0000396, 0.0012039, -0.0000292, 0.0012238, -0.0005410, 0.0005049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005138, upper bound: 0.0005902
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005154, upper bound: 0.0005902
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906812, 0.9925509, 0.9905896, 0.9924576, -0.0007505, 0.0009536
1: -0.0035860, -0.0031201, -0.0036088, -0.0031433, -0.0001870, 0.0002376
2: 0.0064807, 0.0089497, 0.0066040, 0.0090705, -0.0012593, 0.0009910
3: -0.0053466, -0.0042229, -0.0054016, -0.0042790, -0.0004511, 0.0005732
4: 0.0017822, 0.0022601, 0.0018061, 0.0022835, -0.0002437, 0.0001918
5: 0.0071105, 0.0102159, 0.0072656, 0.0103678, -0.0015838, 0.0012464
6: -0.0010521, -0.0002639, -0.0010906, -0.0003033, -0.0003164, 0.0004020
7: -0.0058596, -0.0038204, -0.0059594, -0.0039222, -0.0008185, 0.0010401
8: -0.0026457, -0.0015733, -0.0026981, -0.0016268, -0.0004305, 0.0005470
9: -0.0000396, 0.0012039, 0.0000225, 0.0012648, -0.0006342, 0.0004991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003252, upper bound: 0.0005797
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003141, upper bound: 0.0005756
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906808, 0.9925468, 0.9905822, 0.9924538, -0.0007551, 0.0009903
1: -0.0035861, -0.0031211, -0.0036106, -0.0031443, -0.0001882, 0.0002468
2: 0.0064862, 0.0089502, 0.0066090, 0.0090804, -0.0013077, 0.0009971
3: -0.0053469, -0.0042253, -0.0054061, -0.0042813, -0.0004539, 0.0005952
4: 0.0017833, 0.0022602, 0.0018071, 0.0022854, -0.0002531, 0.0001930
5: 0.0071173, 0.0102165, 0.0072719, 0.0103802, -0.0016448, 0.0012541
6: -0.0010522, -0.0002656, -0.0010938, -0.0003048, -0.0003183, 0.0004175
7: -0.0058601, -0.0038249, -0.0059676, -0.0039263, -0.0008236, 0.0010801
8: -0.0026459, -0.0015756, -0.0027024, -0.0016290, -0.0004331, 0.0005680
9: -0.0000369, 0.0012042, 0.0000250, 0.0012698, -0.0006586, 0.0005022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003378, upper bound: 0.0005814
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003296, upper bound: 0.0005762
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906482, 0.9925519, 0.9905761, 0.9924576, -0.0007576, 0.0009683
1: -0.0035942, -0.0031198, -0.0036121, -0.0031433, -0.0001888, 0.0002413
2: 0.0064794, 0.0089933, 0.0066040, 0.0090884, -0.0012786, 0.0010004
3: -0.0053665, -0.0042223, -0.0054098, -0.0042790, -0.0004553, 0.0005820
4: 0.0017820, 0.0022685, 0.0018061, 0.0022869, -0.0002475, 0.0001936
5: 0.0071088, 0.0102706, 0.0072656, 0.0103903, -0.0016081, 0.0012583
6: -0.0010660, -0.0002635, -0.0010963, -0.0003033, -0.0003194, 0.0004082
7: -0.0058956, -0.0038193, -0.0059742, -0.0039222, -0.0008263, 0.0010560
8: -0.0026646, -0.0015727, -0.0027059, -0.0016268, -0.0004345, 0.0005554
9: -0.0000403, 0.0012258, 0.0000225, 0.0012738, -0.0006440, 0.0005039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004450, upper bound: 0.0005797
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004430, upper bound: 0.0005775
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9906514, 0.9925352, -0.0009033, 0.0008025
1: -0.0036014, -0.0031398, -0.0035934, -0.0031240, -0.0002251, 0.0002000
2: 0.0065851, 0.0090316, 0.0065014, 0.0089891, -0.0010597, 0.0011927
3: -0.0053839, -0.0042704, -0.0053646, -0.0042323, -0.0005429, 0.0004823
4: 0.0018024, 0.0022759, 0.0017862, 0.0022677, -0.0002051, 0.0002309
5: 0.0072418, 0.0103188, 0.0071365, 0.0102654, -0.0013328, 0.0015002
6: -0.0010782, -0.0002972, -0.0010646, -0.0002705, -0.0003808, 0.0003383
7: -0.0059272, -0.0039066, -0.0058922, -0.0038375, -0.0009851, 0.0008752
8: -0.0026812, -0.0016186, -0.0026628, -0.0015822, -0.0005181, 0.0004603
9: 0.0000130, 0.0012452, -0.0000292, 0.0012238, -0.0005337, 0.0006007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003896, upper bound: 0.0005758
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003770, upper bound: 0.0005666
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906191, 0.9924718, 0.9905896, 0.9924576, -0.0007637, 0.0008170
1: -0.0036014, -0.0031398, -0.0036088, -0.0031433, -0.0001903, 0.0002036
2: 0.0065851, 0.0090316, 0.0066040, 0.0090705, -0.0010789, 0.0010085
3: -0.0053839, -0.0042704, -0.0054016, -0.0042790, -0.0004590, 0.0004911
4: 0.0018024, 0.0022759, 0.0018061, 0.0022835, -0.0002088, 0.0001952
5: 0.0072418, 0.0103188, 0.0072656, 0.0103678, -0.0013570, 0.0012684
6: -0.0010782, -0.0002972, -0.0010906, -0.0003033, -0.0003219, 0.0003444
7: -0.0059272, -0.0039066, -0.0059594, -0.0039222, -0.0008330, 0.0008911
8: -0.0026812, -0.0016186, -0.0026981, -0.0016268, -0.0004380, 0.0004686
9: 0.0000130, 0.0012452, 0.0000225, 0.0012648, -0.0005434, 0.0005079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003174, upper bound: 0.0005758
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002985, upper bound: 0.0005665
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906188, 0.9924687, 0.9905822, 0.9924538, -0.0007672, 0.0008557
1: -0.0036015, -0.0031406, -0.0036106, -0.0031443, -0.0001912, 0.0002132
2: 0.0065894, 0.0090321, 0.0066090, 0.0090804, -0.0011300, 0.0010131
3: -0.0053841, -0.0042723, -0.0054061, -0.0042813, -0.0004611, 0.0005143
4: 0.0018033, 0.0022760, 0.0018071, 0.0022854, -0.0002187, 0.0001961
5: 0.0072472, 0.0103195, 0.0072719, 0.0103802, -0.0014212, 0.0012742
6: -0.0010784, -0.0002986, -0.0010938, -0.0003048, -0.0003234, 0.0003607
7: -0.0059277, -0.0039101, -0.0059676, -0.0039263, -0.0008368, 0.0009333
8: -0.0026814, -0.0016205, -0.0027024, -0.0016290, -0.0004400, 0.0004908
9: 0.0000151, 0.0012454, 0.0000250, 0.0012698, -0.0005691, 0.0005102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003290, upper bound: 0.0005772
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003108, upper bound: 0.0005673
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905870, 0.9924747, 0.9905761, 0.9924576, -0.0007705, 0.0008321
1: -0.0036094, -0.0031391, -0.0036121, -0.0031433, -0.0001920, 0.0002073
2: 0.0065813, 0.0090741, 0.0066040, 0.0090884, -0.0010988, 0.0010174
3: -0.0054032, -0.0042687, -0.0054098, -0.0042790, -0.0004631, 0.0005001
4: 0.0018017, 0.0022842, 0.0018061, 0.0022869, -0.0002127, 0.0001969
5: 0.0072370, 0.0103723, 0.0072656, 0.0103903, -0.0013820, 0.0012797
6: -0.0010918, -0.0002960, -0.0010963, -0.0003033, -0.0003248, 0.0003508
7: -0.0059623, -0.0039035, -0.0059742, -0.0039222, -0.0008403, 0.0009075
8: -0.0026997, -0.0016170, -0.0027059, -0.0016268, -0.0004419, 0.0004773
9: 0.0000111, 0.0012666, 0.0000225, 0.0012738, -0.0005534, 0.0005124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004379, upper bound: 0.0005760
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0005704
time: 0.63 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.94 seconds
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005136, upper bound: 0.0005775
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005150, upper bound: 0.0005748
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005136, upper bound: 0.0005902
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005150, upper bound: 0.0005902
IS_A2_B1_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005087, upper bound: 0.0005775
IS_A2_B1_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005093, upper bound: 0.0005748
IS_A2_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005087, upper bound: 0.0005906
IS_A2_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005093, upper bound: 0.0005906
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003275, upper bound: 0.0005797
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003156, upper bound: 0.0005756
IS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003406, upper bound: 0.0005814
IS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003327, upper bound: 0.0005762
IS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004458, upper bound: 0.0005356
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004443, upper bound: 0.0005774
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004526, upper bound: 0.0005816
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004514, upper bound: 0.0005780
IS_A2_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003903, upper bound: 0.0005758
IS_A2_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003782, upper bound: 0.0005666
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003881, upper bound: 0.0005772
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003743, upper bound: 0.0005675
IS_A2_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003219, upper bound: 0.0005758
IS_A2_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003007, upper bound: 0.0005665
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003321, upper bound: 0.0004728
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003132, upper bound: 0.0005673
IS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0005760
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004340, upper bound: 0.0005704
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004443, upper bound: 0.0005776
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004411, upper bound: 0.0004704
IS_A2_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005138, upper bound: 0.0005775
IS_A2_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005154, upper bound: 0.0005748
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005138, upper bound: 0.0005902
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0005154, upper bound: 0.0005902
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003252, upper bound: 0.0005797
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003141, upper bound: 0.0005756
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003378, upper bound: 0.0005814
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003296, upper bound: 0.0005762
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004450, upper bound: 0.0005797
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004430, upper bound: 0.0005775
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003896, upper bound: 0.0005758
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003770, upper bound: 0.0005666
IS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003174, upper bound: 0.0005758
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0002985, upper bound: 0.0005665
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003290, upper bound: 0.0005772
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0003108, upper bound: 0.0005673
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004379, upper bound: 0.0005760
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.94
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0005704

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.80 + 509.24 = 512.04 seconds
