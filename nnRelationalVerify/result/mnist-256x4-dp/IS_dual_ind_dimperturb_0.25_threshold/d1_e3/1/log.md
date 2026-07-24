## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018112


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040990, -0.0040846, -0.0040990, -0.0040846, -0.0000065, 0.0000065)
1: (-0.0061632, -0.0056225, -0.0061632, -0.0056225, -0.0002452, 0.0002452)
2: (0.9690673, 0.9697163, 0.9690673, 0.9697163, -0.0002943, 0.0002943)
3: (0.0181521, 0.0229378, 0.0181521, 0.0229378, -0.0021706, 0.0021706)
4: (-0.0024376, -0.0020736, -0.0024376, -0.0020736, -0.0001651, 0.0001651)
5: (0.0148067, 0.0151746, 0.0148067, 0.0151746, -0.0001669, 0.0001669)
6: (0.0045259, 0.0047048, 0.0045259, 0.0047048, -0.0000812, 0.0000812)
7: (-0.0137228, -0.0124825, -0.0137228, -0.0124825, -0.0005625, 0.0005625)
8: (0.0058421, 0.0068261, 0.0058421, 0.0068261, -0.0004463, 0.0004463)
9: (0.0082323, 0.0100020, 0.0082323, 0.0100020, -0.0008027, 0.0008027)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.33 = 2.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0002254, upper bound: 0.0002255

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002126, upper bound: 0.0002176
time: 0.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002176, upper bound: 0.0002176
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 2, lower bound: -0.0002126, upper bound: 0.0002176
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 2, lower bound: -0.0002176, upper bound: 0.0002176

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040986, -0.0040846, -0.0040989, -0.0040846, -0.0000061, 0.0000064
1: -0.0061469, -0.0056226, -0.0061593, -0.0056225, -0.0002282, 0.0002405
2: 0.9690868, 0.9697161, 0.9690720, 0.9697162, -0.0002738, 0.0002887
3: 0.0182960, 0.0229368, 0.0181865, 0.0229376, -0.0020199, 0.0021292
4: -0.0024375, -0.0020845, -0.0024376, -0.0020762, -0.0001619, 0.0001536
5: 0.0148068, 0.0151635, 0.0148068, 0.0151720, -0.0001637, 0.0001553
6: 0.0045312, 0.0047048, 0.0045272, 0.0047048, -0.0000755, 0.0000796
7: -0.0137225, -0.0125198, -0.0137227, -0.0124914, -0.0005518, 0.0005235
8: 0.0058424, 0.0067965, 0.0058422, 0.0068190, -0.0004378, 0.0004153
9: 0.0082327, 0.0099488, 0.0082323, 0.0099893, -0.0007874, 0.0007469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002063, upper bound: 0.0002033
time: 0.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002063, upper bound: 0.0002116
time: 0.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0040987, -0.0040844, -0.0040989, -0.0040846, -0.0000062, 0.0000068
1: -0.0061511, -0.0056153, -0.0061597, -0.0056226, -0.0002306, 0.0002556
2: 0.9690819, 0.9697249, 0.9690715, 0.9697162, -0.0002767, 0.0003067
3: 0.0182593, 0.0230015, 0.0181829, 0.0229375, -0.0020412, 0.0022624
4: -0.0024424, -0.0020818, -0.0024376, -0.0020759, -0.0001721, 0.0001552
5: 0.0148018, 0.0151664, 0.0148068, 0.0151722, -0.0001739, 0.0001569
6: 0.0045299, 0.0047072, 0.0045270, 0.0047048, -0.0000763, 0.0000846
7: -0.0137393, -0.0125103, -0.0137227, -0.0124905, -0.0005863, 0.0005290
8: 0.0058290, 0.0068041, 0.0058422, 0.0068198, -0.0004652, 0.0004197
9: 0.0082087, 0.0099624, 0.0082324, 0.0099906, -0.0008366, 0.0007548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002116, upper bound: 0.0002034
time: 0.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002116, upper bound: 0.0002116
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 2, lower bound: -0.0002063, upper bound: 0.0002033
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 2, lower bound: -0.0002063, upper bound: 0.0002116
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 2, lower bound: -0.0002116, upper bound: 0.0002034
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 2, lower bound: -0.0002116, upper bound: 0.0002116

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040983, -0.0040845, -0.0000056, 0.0000058
1: -0.0061395, -0.0056227, -0.0061370, -0.0056186, -0.0002105, 0.0002179
2: 0.9690958, 0.9697160, 0.9690987, 0.9697209, -0.0002526, 0.0002615
3: 0.0183618, 0.0229363, 0.0183836, 0.0229722, -0.0018633, 0.0019290
4: -0.0024375, -0.0020895, -0.0024402, -0.0020912, -0.0001467, 0.0001417
5: 0.0148068, 0.0151585, 0.0148041, 0.0151568, -0.0001483, 0.0001432
6: 0.0045337, 0.0047047, 0.0045345, 0.0047061, -0.0000697, 0.0000721
7: -0.0137224, -0.0125369, -0.0137317, -0.0125425, -0.0004999, 0.0004829
8: 0.0058424, 0.0067830, 0.0058351, 0.0067785, -0.0003966, 0.0003831
9: 0.0082328, 0.0099245, 0.0082195, 0.0099164, -0.0007133, 0.0006891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002034
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002034
time: 0.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040846, -0.0040987, -0.0040846, -0.0000061, 0.0000058
1: -0.0061465, -0.0056226, -0.0061531, -0.0056226, -0.0002269, 0.0002168
2: 0.9690874, 0.9697161, 0.9690795, 0.9697161, -0.0002722, 0.0002602
3: 0.0182998, 0.0229367, 0.0182413, 0.0229371, -0.0020080, 0.0019193
4: -0.0024375, -0.0020848, -0.0024375, -0.0020804, -0.0001460, 0.0001527
5: 0.0148068, 0.0151633, 0.0148068, 0.0151677, -0.0001475, 0.0001544
6: 0.0045314, 0.0047048, 0.0045292, 0.0047048, -0.0000751, 0.0000718
7: -0.0137225, -0.0125208, -0.0137226, -0.0125057, -0.0004974, 0.0005204
8: 0.0058424, 0.0067957, 0.0058423, 0.0068078, -0.0003946, 0.0004129
9: 0.0082327, 0.0099474, 0.0082325, 0.0099690, -0.0007098, 0.0007426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002116
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002116
time: 0.55 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040983, -0.0040845, -0.0000057, 0.0000062
1: -0.0061437, -0.0056154, -0.0061375, -0.0056186, -0.0002129, 0.0002325
2: 0.9690907, 0.9697248, 0.9690982, 0.9697209, -0.0002554, 0.0002790
3: 0.0183244, 0.0230011, 0.0183795, 0.0229722, -0.0018840, 0.0020578
4: -0.0024424, -0.0020867, -0.0024402, -0.0020909, -0.0001565, 0.0001433
5: 0.0148019, 0.0151614, 0.0148041, 0.0151571, -0.0001582, 0.0001448
6: 0.0045323, 0.0047072, 0.0045344, 0.0047061, -0.0000704, 0.0000769
7: -0.0137392, -0.0125272, -0.0137317, -0.0125415, -0.0005333, 0.0004883
8: 0.0058291, 0.0067907, 0.0058351, 0.0067793, -0.0004231, 0.0003874
9: 0.0082089, 0.0099383, 0.0082196, 0.0099179, -0.0007610, 0.0006967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002034
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002033
time: 0.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040987, -0.0040844, -0.0040987, -0.0040846, -0.0000061, 0.0000063
1: -0.0061506, -0.0056153, -0.0061535, -0.0056226, -0.0002293, 0.0002360
2: 0.9690824, 0.9697249, 0.9690790, 0.9697161, -0.0002751, 0.0002833
3: 0.0182632, 0.0230015, 0.0182377, 0.0229370, -0.0020292, 0.0020893
4: -0.0024424, -0.0020821, -0.0024375, -0.0020801, -0.0001589, 0.0001543
5: 0.0148018, 0.0151661, 0.0148068, 0.0151680, -0.0001606, 0.0001560
6: 0.0045300, 0.0047072, 0.0045291, 0.0047048, -0.0000759, 0.0000781
7: -0.0137393, -0.0125113, -0.0137226, -0.0125047, -0.0005415, 0.0005259
8: 0.0058290, 0.0068033, 0.0058423, 0.0068085, -0.0004296, 0.0004172
9: 0.0082087, 0.0099609, 0.0082326, 0.0099703, -0.0007726, 0.0007504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002033, upper bound: 0.0002116
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002033, upper bound: 0.0002116
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002034
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002034
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002116
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002116
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002034
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002033
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002033, upper bound: 0.0002116
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 2, lower bound: -0.0002033, upper bound: 0.0002116

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040980, -0.0040845, -0.0040983, -0.0040845, -0.0000053, 0.0000056
1: -0.0061247, -0.0056187, -0.0061370, -0.0056186, -0.0001971, 0.0002103
2: 0.9691136, 0.9697208, 0.9690987, 0.9697209, -0.0002365, 0.0002524
3: 0.0184931, 0.0229715, 0.0183836, 0.0229722, -0.0017445, 0.0018613
4: -0.0024401, -0.0020995, -0.0024402, -0.0020912, -0.0001416, 0.0001327
5: 0.0148041, 0.0151484, 0.0148041, 0.0151568, -0.0001431, 0.0001341
6: 0.0045386, 0.0047061, 0.0045345, 0.0047061, -0.0000652, 0.0000696
7: -0.0137315, -0.0125709, -0.0137317, -0.0125425, -0.0004824, 0.0004521
8: 0.0058352, 0.0067560, 0.0058351, 0.0067785, -0.0003827, 0.0003587
9: 0.0082198, 0.0098759, 0.0082195, 0.0099164, -0.0006883, 0.0006451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002013
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002034
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040983, -0.0040845, -0.0000058, 0.0000058
1: -0.0061409, -0.0056227, -0.0061370, -0.0056186, -0.0002167, 0.0002170
2: 0.9690941, 0.9697160, 0.9690987, 0.9697209, -0.0002600, 0.0002604
3: 0.0183489, 0.0229363, 0.0183836, 0.0229722, -0.0019177, 0.0019208
4: -0.0024375, -0.0020886, -0.0024402, -0.0020912, -0.0001461, 0.0001459
5: 0.0148069, 0.0151595, 0.0148041, 0.0151568, -0.0001476, 0.0001474
6: 0.0045332, 0.0047047, 0.0045345, 0.0047061, -0.0000717, 0.0000718
7: -0.0137224, -0.0125335, -0.0137317, -0.0125425, -0.0004978, 0.0004970
8: 0.0058425, 0.0067856, 0.0058351, 0.0067785, -0.0003949, 0.0003943
9: 0.0082328, 0.0099292, 0.0082195, 0.0099164, -0.0007103, 0.0007092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002013
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002033
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040980, -0.0040845, -0.0040987, -0.0040846, -0.0000054, 0.0000061
1: -0.0061247, -0.0056187, -0.0061531, -0.0056226, -0.0002039, 0.0002288
2: 0.9691136, 0.9697208, 0.9690795, 0.9697161, -0.0002446, 0.0002746
3: 0.0184931, 0.0229715, 0.0182413, 0.0229371, -0.0018044, 0.0020255
4: -0.0024401, -0.0020995, -0.0024375, -0.0020804, -0.0001540, 0.0001372
5: 0.0148041, 0.0151484, 0.0148068, 0.0151677, -0.0001557, 0.0001387
6: 0.0045386, 0.0047061, 0.0045292, 0.0047048, -0.0000675, 0.0000757
7: -0.0137315, -0.0125709, -0.0137226, -0.0125057, -0.0005249, 0.0004676
8: 0.0058352, 0.0067560, 0.0058423, 0.0068078, -0.0004164, 0.0003710
9: 0.0082198, 0.0098759, 0.0082325, 0.0099690, -0.0007490, 0.0006673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002083
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002116
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040987, -0.0040846, -0.0000054, 0.0000058
1: -0.0061409, -0.0056227, -0.0061531, -0.0056226, -0.0002030, 0.0002163
2: 0.9690941, 0.9697160, 0.9690795, 0.9697161, -0.0002436, 0.0002596
3: 0.0183489, 0.0229363, 0.0182413, 0.0229371, -0.0017970, 0.0019146
4: -0.0024375, -0.0020886, -0.0024375, -0.0020804, -0.0001456, 0.0001367
5: 0.0148069, 0.0151595, 0.0148068, 0.0151677, -0.0001472, 0.0001381
6: 0.0045332, 0.0047047, 0.0045292, 0.0047048, -0.0000672, 0.0000716
7: -0.0137224, -0.0125335, -0.0137226, -0.0125057, -0.0004962, 0.0004657
8: 0.0058425, 0.0067856, 0.0058423, 0.0068078, -0.0003936, 0.0003695
9: 0.0082328, 0.0099292, 0.0082325, 0.0099690, -0.0007080, 0.0006645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002013
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002034
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040981, -0.0040842, -0.0040983, -0.0040845, -0.0000053, 0.0000061
1: -0.0061288, -0.0056110, -0.0061375, -0.0056186, -0.0001995, 0.0002295
2: 0.9691086, 0.9697300, 0.9690982, 0.9697209, -0.0002395, 0.0002754
3: 0.0184562, 0.0230393, 0.0183795, 0.0229722, -0.0017662, 0.0020315
4: -0.0024453, -0.0020967, -0.0024402, -0.0020909, -0.0001545, 0.0001343
5: 0.0147989, 0.0151512, 0.0148041, 0.0151571, -0.0001562, 0.0001358
6: 0.0045372, 0.0047086, 0.0045344, 0.0047061, -0.0000660, 0.0000760
7: -0.0137491, -0.0125613, -0.0137317, -0.0125415, -0.0005265, 0.0004577
8: 0.0058213, 0.0067636, 0.0058351, 0.0067793, -0.0004177, 0.0003631
9: 0.0081947, 0.0098895, 0.0082196, 0.0099179, -0.0007513, 0.0006531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040983, -0.0040845, -0.0000058, 0.0000062
1: -0.0061448, -0.0056154, -0.0061375, -0.0056186, -0.0002191, 0.0002317
2: 0.9690894, 0.9697248, 0.9690982, 0.9697209, -0.0002629, 0.0002781
3: 0.0183145, 0.0230010, 0.0183795, 0.0229722, -0.0019390, 0.0020512
4: -0.0024424, -0.0020860, -0.0024402, -0.0020909, -0.0001560, 0.0001475
5: 0.0148019, 0.0151621, 0.0148041, 0.0151571, -0.0001577, 0.0001490
6: 0.0045319, 0.0047072, 0.0045344, 0.0047061, -0.0000725, 0.0000767
7: -0.0137392, -0.0125246, -0.0137317, -0.0125415, -0.0005316, 0.0005025
8: 0.0058291, 0.0067927, 0.0058351, 0.0067793, -0.0004217, 0.0003987
9: 0.0082089, 0.0099419, 0.0082196, 0.0099179, -0.0007585, 0.0007170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040981, -0.0040842, -0.0040987, -0.0040846, -0.0000055, 0.0000066
1: -0.0061288, -0.0056110, -0.0061535, -0.0056226, -0.0002063, 0.0002480
2: 0.9691086, 0.9697300, 0.9690790, 0.9697161, -0.0002476, 0.0002976
3: 0.0184562, 0.0230393, 0.0182377, 0.0229370, -0.0018261, 0.0021952
4: -0.0024453, -0.0020967, -0.0024375, -0.0020801, -0.0001670, 0.0001389
5: 0.0147989, 0.0151512, 0.0148068, 0.0151680, -0.0001687, 0.0001404
6: 0.0045372, 0.0047086, 0.0045291, 0.0047048, -0.0000683, 0.0000821
7: -0.0137491, -0.0125613, -0.0137226, -0.0125047, -0.0005689, 0.0004732
8: 0.0058213, 0.0067636, 0.0058423, 0.0068085, -0.0004513, 0.0003755
9: 0.0081947, 0.0098895, 0.0082326, 0.0099703, -0.0008118, 0.0006753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002063
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002063
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040987, -0.0040846, -0.0000055, 0.0000063
1: -0.0061448, -0.0056154, -0.0061535, -0.0056226, -0.0002055, 0.0002356
2: 0.9690894, 0.9697248, 0.9690790, 0.9697161, -0.0002466, 0.0002827
3: 0.0183145, 0.0230010, 0.0182377, 0.0229370, -0.0018186, 0.0020851
4: -0.0024424, -0.0020860, -0.0024375, -0.0020801, -0.0001586, 0.0001383
5: 0.0148019, 0.0151621, 0.0148068, 0.0151680, -0.0001603, 0.0001398
6: 0.0045319, 0.0047072, 0.0045291, 0.0047048, -0.0000680, 0.0000780
7: -0.0137392, -0.0125246, -0.0137226, -0.0125047, -0.0005404, 0.0004713
8: 0.0058291, 0.0067927, 0.0058423, 0.0068085, -0.0004287, 0.0003739
9: 0.0082089, 0.0099419, 0.0082326, 0.0099703, -0.0007711, 0.0006725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002009
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002009
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.39 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002013
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002034
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002013
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002026, upper bound: 0.0002033
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002083
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002116
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002013
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002008, upper bound: 0.0002034
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002071, upper bound: 0.0002008
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002063
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002063
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002009
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 2, lower bound: -0.0002034, upper bound: 0.0002009

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040980, -0.0040845, -0.0040980, -0.0040845, -0.0000053, 0.0000053
1: -0.0061247, -0.0056187, -0.0061247, -0.0056187, -0.0001971, 0.0001971
2: 0.9691136, 0.9697208, 0.9691136, 0.9697208, -0.0002365, 0.0002365
3: 0.0184931, 0.0229715, 0.0184931, 0.0229715, -0.0017445, 0.0017445
4: -0.0024401, -0.0020995, -0.0024401, -0.0020995, -0.0001327, 0.0001327
5: 0.0148041, 0.0151484, 0.0148041, 0.0151484, -0.0001341, 0.0001341
6: 0.0045386, 0.0047061, 0.0045386, 0.0047061, -0.0000652, 0.0000652
7: -0.0137315, -0.0125709, -0.0137315, -0.0125709, -0.0004521, 0.0004521
8: 0.0058352, 0.0067560, 0.0058352, 0.0067560, -0.0003587, 0.0003587
9: 0.0082198, 0.0098759, 0.0082198, 0.0098759, -0.0006451, 0.0006451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.99 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001824
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001829
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040980, -0.0040845, -0.0040981, -0.0040842, -0.0000057, 0.0000055
1: -0.0061247, -0.0056187, -0.0061288, -0.0056110, -0.0002142, 0.0002078
2: 0.9691136, 0.9697208, 0.9691086, 0.9697300, -0.0002571, 0.0002493
3: 0.0184931, 0.0229715, 0.0184562, 0.0230393, -0.0018961, 0.0018389
4: -0.0024401, -0.0020995, -0.0024453, -0.0020967, -0.0001399, 0.0001442
5: 0.0148041, 0.0151484, 0.0147989, 0.0151512, -0.0001414, 0.0001457
6: 0.0045386, 0.0047061, 0.0045372, 0.0047086, -0.0000709, 0.0000688
7: -0.0137315, -0.0125709, -0.0137491, -0.0125613, -0.0004766, 0.0004914
8: 0.0058352, 0.0067560, 0.0058213, 0.0067636, -0.0003781, 0.0003898
9: 0.0082198, 0.0098759, 0.0081947, 0.0098895, -0.0006800, 0.0007012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.93 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001842
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001855
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040980, -0.0040845, -0.0000058, 0.0000054
1: -0.0061409, -0.0056227, -0.0061247, -0.0056187, -0.0002167, 0.0002038
2: 0.9690941, 0.9697160, 0.9691136, 0.9697208, -0.0002600, 0.0002446
3: 0.0183489, 0.0229363, 0.0184931, 0.0229715, -0.0019177, 0.0018039
4: -0.0024375, -0.0020886, -0.0024401, -0.0020995, -0.0001372, 0.0001459
5: 0.0148069, 0.0151595, 0.0148041, 0.0151484, -0.0001387, 0.0001474
6: 0.0045332, 0.0047047, 0.0045386, 0.0047061, -0.0000717, 0.0000674
7: -0.0137224, -0.0125335, -0.0137315, -0.0125709, -0.0004675, 0.0004970
8: 0.0058425, 0.0067856, 0.0058352, 0.0067560, -0.0003709, 0.0003943
9: 0.0082328, 0.0099292, 0.0082198, 0.0098759, -0.0006671, 0.0007092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.95 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001816
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001817
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040981, -0.0040842, -0.0000062, 0.0000057
1: -0.0061409, -0.0056227, -0.0061288, -0.0056110, -0.0002338, 0.0002145
2: 0.9690941, 0.9697160, 0.9691086, 0.9697300, -0.0002805, 0.0002574
3: 0.0183489, 0.0229363, 0.0184562, 0.0230393, -0.0020693, 0.0018983
4: -0.0024375, -0.0020886, -0.0024453, -0.0020967, -0.0001444, 0.0001574
5: 0.0148069, 0.0151595, 0.0147989, 0.0151512, -0.0001459, 0.0001591
6: 0.0045332, 0.0047047, 0.0045372, 0.0047086, -0.0000774, 0.0000710
7: -0.0137224, -0.0125335, -0.0137491, -0.0125613, -0.0004920, 0.0005363
8: 0.0058425, 0.0067856, 0.0058213, 0.0067636, -0.0003903, 0.0004255
9: 0.0082328, 0.0099292, 0.0081947, 0.0098895, -0.0007020, 0.0007652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.95 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001834
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001837
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040980, -0.0040845, -0.0040984, -0.0040846, -0.0000054, 0.0000058
1: -0.0061247, -0.0056187, -0.0061409, -0.0056227, -0.0002038, 0.0002167
2: 0.9691136, 0.9697208, 0.9690941, 0.9697160, -0.0002446, 0.0002600
3: 0.0184931, 0.0229715, 0.0183489, 0.0229363, -0.0018039, 0.0019177
4: -0.0024401, -0.0020995, -0.0024375, -0.0020886, -0.0001459, 0.0001372
5: 0.0148041, 0.0151484, 0.0148069, 0.0151595, -0.0001474, 0.0001387
6: 0.0045386, 0.0047061, 0.0045332, 0.0047047, -0.0000674, 0.0000717
7: -0.0137315, -0.0125709, -0.0137224, -0.0125335, -0.0004970, 0.0004675
8: 0.0058352, 0.0067560, 0.0058425, 0.0067856, -0.0003943, 0.0003709
9: 0.0082198, 0.0098759, 0.0082328, 0.0099292, -0.0007092, 0.0006671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001827, upper bound: 0.0001830
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040980, -0.0040845, -0.0040985, -0.0040844, -0.0000058, 0.0000060
1: -0.0061247, -0.0056187, -0.0061448, -0.0056154, -0.0002164, 0.0002264
2: 0.9691136, 0.9697208, 0.9690894, 0.9697248, -0.0002597, 0.0002717
3: 0.0184931, 0.0229715, 0.0183145, 0.0230010, -0.0019157, 0.0020039
4: -0.0024401, -0.0020995, -0.0024424, -0.0020860, -0.0001524, 0.0001457
5: 0.0148041, 0.0151484, 0.0148019, 0.0151621, -0.0001540, 0.0001473
6: 0.0045386, 0.0047061, 0.0045319, 0.0047072, -0.0000716, 0.0000749
7: -0.0137315, -0.0125709, -0.0137392, -0.0125246, -0.0005193, 0.0004965
8: 0.0058352, 0.0067560, 0.0058291, 0.0067927, -0.0004120, 0.0003939
9: 0.0082198, 0.0098759, 0.0082089, 0.0099419, -0.0007410, 0.0007084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.03 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001827, upper bound: 0.0001850
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001883
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040984, -0.0040846, -0.0000054, 0.0000054
1: -0.0061409, -0.0056227, -0.0061409, -0.0056227, -0.0002030, 0.0002030
2: 0.9690941, 0.9697160, 0.9690941, 0.9697160, -0.0002436, 0.0002436
3: 0.0183489, 0.0229363, 0.0183489, 0.0229363, -0.0017969, 0.0017969
4: -0.0024375, -0.0020886, -0.0024375, -0.0020886, -0.0001367, 0.0001367
5: 0.0148069, 0.0151595, 0.0148069, 0.0151595, -0.0001381, 0.0001381
6: 0.0045332, 0.0047047, 0.0045332, 0.0047047, -0.0000672, 0.0000672
7: -0.0137224, -0.0125335, -0.0137224, -0.0125335, -0.0004657, 0.0004657
8: 0.0058425, 0.0067856, 0.0058425, 0.0067856, -0.0003695, 0.0003695
9: 0.0082328, 0.0099292, 0.0082328, 0.0099292, -0.0006645, 0.0006645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001622, upper bound: 0.0001648
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001407, upper bound: 0.0001317
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040984, -0.0040846, -0.0040985, -0.0040844, -0.0000059, 0.0000057
1: -0.0061409, -0.0056227, -0.0061448, -0.0056154, -0.0002201, 0.0002138
2: 0.9690941, 0.9697160, 0.9690894, 0.9697248, -0.0002641, 0.0002566
3: 0.0183489, 0.0229363, 0.0183145, 0.0230010, -0.0019478, 0.0018923
4: -0.0024375, -0.0020886, -0.0024424, -0.0020860, -0.0001439, 0.0001481
5: 0.0148069, 0.0151595, 0.0148019, 0.0151621, -0.0001455, 0.0001497
6: 0.0045332, 0.0047047, 0.0045319, 0.0047072, -0.0000728, 0.0000708
7: -0.0137224, -0.0125335, -0.0137392, -0.0125246, -0.0004904, 0.0005048
8: 0.0058425, 0.0067856, 0.0058291, 0.0067927, -0.0003891, 0.0004005
9: 0.0082328, 0.0099292, 0.0082089, 0.0099419, -0.0006998, 0.0007203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001622, upper bound: 0.0001653
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001407, upper bound: 0.0001317
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040981, -0.0040842, -0.0040980, -0.0040845, -0.0000055, 0.0000057
1: -0.0061288, -0.0056110, -0.0061247, -0.0056187, -0.0002078, 0.0002142
2: 0.9691086, 0.9697300, 0.9691136, 0.9697208, -0.0002493, 0.0002571
3: 0.0184562, 0.0230393, 0.0184931, 0.0229715, -0.0018389, 0.0018961
4: -0.0024453, -0.0020967, -0.0024401, -0.0020995, -0.0001442, 0.0001399
5: 0.0147989, 0.0151512, 0.0148041, 0.0151484, -0.0001457, 0.0001414
6: 0.0045372, 0.0047086, 0.0045386, 0.0047061, -0.0000688, 0.0000709
7: -0.0137491, -0.0125613, -0.0137315, -0.0125709, -0.0004914, 0.0004766
8: 0.0058213, 0.0067636, 0.0058352, 0.0067560, -0.0003898, 0.0003781
9: 0.0081947, 0.0098895, 0.0082198, 0.0098759, -0.0007012, 0.0006800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.00 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001825
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040981, -0.0040842, -0.0040981, -0.0040842, -0.0000053, 0.0000053
1: -0.0061288, -0.0056110, -0.0061288, -0.0056110, -0.0001997, 0.0001997
2: 0.9691086, 0.9697300, 0.9691086, 0.9697300, -0.0002396, 0.0002396
3: 0.0184562, 0.0230393, 0.0184562, 0.0230393, -0.0017673, 0.0017673
4: -0.0024453, -0.0020967, -0.0024453, -0.0020967, -0.0001344, 0.0001344
5: 0.0147989, 0.0151512, 0.0147989, 0.0151512, -0.0001358, 0.0001358
6: 0.0045372, 0.0047086, 0.0045372, 0.0047086, -0.0000661, 0.0000661
7: -0.0137491, -0.0125613, -0.0137491, -0.0125613, -0.0004580, 0.0004580
8: 0.0058213, 0.0067636, 0.0058213, 0.0067636, -0.0003634, 0.0003634
9: 0.0081947, 0.0098895, 0.0081947, 0.0098895, -0.0006536, 0.0006536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.95 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001826
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040980, -0.0040845, -0.0000060, 0.0000058
1: -0.0061448, -0.0056154, -0.0061247, -0.0056187, -0.0002264, 0.0002164
2: 0.9690894, 0.9697248, 0.9691136, 0.9697208, -0.0002717, 0.0002597
3: 0.0183145, 0.0230010, 0.0184931, 0.0229715, -0.0020039, 0.0019157
4: -0.0024424, -0.0020860, -0.0024401, -0.0020995, -0.0001457, 0.0001524
5: 0.0148019, 0.0151621, 0.0148041, 0.0151484, -0.0001473, 0.0001540
6: 0.0045319, 0.0047072, 0.0045386, 0.0047061, -0.0000749, 0.0000716
7: -0.0137392, -0.0125246, -0.0137315, -0.0125709, -0.0004965, 0.0005193
8: 0.0058291, 0.0067927, 0.0058352, 0.0067560, -0.0003939, 0.0004120
9: 0.0082089, 0.0099419, 0.0082198, 0.0098759, -0.0007084, 0.0007410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.98 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040981, -0.0040842, -0.0000059, 0.0000055
1: -0.0061448, -0.0056154, -0.0061288, -0.0056110, -0.0002192, 0.0002064
2: 0.9690894, 0.9697248, 0.9691086, 0.9697300, -0.0002630, 0.0002477
3: 0.0183145, 0.0230010, 0.0184562, 0.0230393, -0.0019401, 0.0018270
4: -0.0024424, -0.0020860, -0.0024453, -0.0020967, -0.0001390, 0.0001476
5: 0.0148019, 0.0151621, 0.0147989, 0.0151512, -0.0001404, 0.0001491
6: 0.0045319, 0.0047072, 0.0045372, 0.0047086, -0.0000725, 0.0000683
7: -0.0137392, -0.0125246, -0.0137491, -0.0125613, -0.0004735, 0.0005028
8: 0.0058291, 0.0067927, 0.0058213, 0.0067636, -0.0003756, 0.0003989
9: 0.0082089, 0.0099419, 0.0081947, 0.0098895, -0.0006756, 0.0007174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 1.96 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040981, -0.0040842, -0.0040984, -0.0040846, -0.0000057, 0.0000062
1: -0.0061288, -0.0056110, -0.0061409, -0.0056227, -0.0002145, 0.0002338
2: 0.9691086, 0.9697300, 0.9690941, 0.9697160, -0.0002574, 0.0002805
3: 0.0184562, 0.0230393, 0.0183489, 0.0229363, -0.0018983, 0.0020693
4: -0.0024453, -0.0020967, -0.0024375, -0.0020886, -0.0001574, 0.0001444
5: 0.0147989, 0.0151512, 0.0148069, 0.0151595, -0.0001591, 0.0001459
6: 0.0045372, 0.0047086, 0.0045332, 0.0047047, -0.0000710, 0.0000774
7: -0.0137491, -0.0125613, -0.0137224, -0.0125335, -0.0005363, 0.0004920
8: 0.0058213, 0.0067636, 0.0058425, 0.0067856, -0.0004255, 0.0003903
9: 0.0081947, 0.0098895, 0.0082328, 0.0099292, -0.0007652, 0.0007020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040981, -0.0040842, -0.0040985, -0.0040844, -0.0000055, 0.0000059
1: -0.0061288, -0.0056110, -0.0061448, -0.0056154, -0.0002064, 0.0002192
2: 0.9691086, 0.9697300, 0.9690894, 0.9697248, -0.0002477, 0.0002630
3: 0.0184562, 0.0230393, 0.0183145, 0.0230010, -0.0018270, 0.0019401
4: -0.0024453, -0.0020967, -0.0024424, -0.0020860, -0.0001476, 0.0001390
5: 0.0147989, 0.0151512, 0.0148019, 0.0151621, -0.0001491, 0.0001404
6: 0.0045372, 0.0047086, 0.0045319, 0.0047072, -0.0000683, 0.0000725
7: -0.0137491, -0.0125613, -0.0137392, -0.0125246, -0.0005028, 0.0004735
8: 0.0058213, 0.0067636, 0.0058291, 0.0067927, -0.0003989, 0.0003756
9: 0.0081947, 0.0098895, 0.0082089, 0.0099419, -0.0007174, 0.0006756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.19 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040984, -0.0040846, -0.0000057, 0.0000059
1: -0.0061448, -0.0056154, -0.0061409, -0.0056227, -0.0002138, 0.0002201
2: 0.9690894, 0.9697248, 0.9690941, 0.9697160, -0.0002566, 0.0002641
3: 0.0183145, 0.0230010, 0.0183489, 0.0229363, -0.0018923, 0.0019478
4: -0.0024424, -0.0020860, -0.0024375, -0.0020886, -0.0001481, 0.0001439
5: 0.0148019, 0.0151621, 0.0148069, 0.0151595, -0.0001497, 0.0001455
6: 0.0045319, 0.0047072, 0.0045332, 0.0047047, -0.0000708, 0.0000728
7: -0.0137392, -0.0125246, -0.0137224, -0.0125335, -0.0005048, 0.0004904
8: 0.0058291, 0.0067927, 0.0058425, 0.0067856, -0.0004005, 0.0003891
9: 0.0082089, 0.0099419, 0.0082328, 0.0099292, -0.0007203, 0.0006998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001641, upper bound: 0.0001644
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001404, upper bound: 0.0001315
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040985, -0.0040844, -0.0040985, -0.0040844, -0.0000055, 0.0000055
1: -0.0061448, -0.0056154, -0.0061448, -0.0056154, -0.0002056, 0.0002056
2: 0.9690894, 0.9697248, 0.9690894, 0.9697248, -0.0002467, 0.0002467
3: 0.0183145, 0.0230010, 0.0183145, 0.0230010, -0.0018200, 0.0018200
4: -0.0024424, -0.0020860, -0.0024424, -0.0020860, -0.0001384, 0.0001384
5: 0.0148019, 0.0151621, 0.0148019, 0.0151621, -0.0001399, 0.0001399
6: 0.0045319, 0.0047072, 0.0045319, 0.0047072, -0.0000680, 0.0000680
7: -0.0137392, -0.0125246, -0.0137392, -0.0125246, -0.0004717, 0.0004717
8: 0.0058291, 0.0067927, 0.0058291, 0.0067927, -0.0003742, 0.0003742
9: 0.0082089, 0.0099419, 0.0082089, 0.0099419, -0.0006730, 0.0006730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001641, upper bound: 0.0001644
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001404, upper bound: 0.0001315
time: 0.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001824
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001829
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001842
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001855
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001816
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001817
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001834
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001837
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001827, upper bound: 0.0001830
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001827, upper bound: 0.0001850
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001883
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001622, upper bound: 0.0001648
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001407, upper bound: 0.0001317
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001622, upper bound: 0.0001653
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001407, upper bound: 0.0001317
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001825
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001826
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001641, upper bound: 0.0001644
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001404, upper bound: 0.0001315
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001641, upper bound: 0.0001644
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.52
Output dim: 2, lower bound: -0.0001404, upper bound: 0.0001315

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040979, -0.0040845, -0.0000049, 0.0000052
1: -0.0061151, -0.0056187, -0.0061234, -0.0056187, -0.0001834, 0.0001951
2: 0.9691251, 0.9697208, 0.9691151, 0.9697208, -0.0002201, 0.0002341
3: 0.0185775, 0.0229715, 0.0185044, 0.0229715, -0.0016233, 0.0017268
4: -0.0024401, -0.0021060, -0.0024401, -0.0021004, -0.0001313, 0.0001235
5: 0.0148041, 0.0151419, 0.0148041, 0.0151475, -0.0001327, 0.0001248
6: 0.0045418, 0.0047061, 0.0045390, 0.0047061, -0.0000607, 0.0000646
7: -0.0137315, -0.0125928, -0.0137315, -0.0125738, -0.0004475, 0.0004207
8: 0.0058352, 0.0067386, 0.0058352, 0.0067537, -0.0003550, 0.0003338
9: 0.0082198, 0.0098447, 0.0082198, 0.0098717, -0.0006386, 0.0006003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.00 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040978, -0.0040845, -0.0000048, 0.0000060
1: -0.0061043, -0.0056125, -0.0061183, -0.0056187, -0.0001803, 0.0002239
2: 0.9691381, 0.9697283, 0.9691213, 0.9697208, -0.0002164, 0.0002686
3: 0.0186735, 0.0230266, 0.0185495, 0.0229715, -0.0015959, 0.0019815
4: -0.0024443, -0.0021133, -0.0024401, -0.0021038, -0.0001507, 0.0001214
5: 0.0147999, 0.0151345, 0.0148041, 0.0151441, -0.0001523, 0.0001227
6: 0.0045454, 0.0047081, 0.0045407, 0.0047061, -0.0000597, 0.0000741
7: -0.0137458, -0.0126177, -0.0137315, -0.0125855, -0.0005135, 0.0004136
8: 0.0058239, 0.0067189, 0.0058352, 0.0067444, -0.0004074, 0.0003281
9: 0.0081994, 0.0098092, 0.0082198, 0.0098551, -0.0007328, 0.0005902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040981, -0.0040842, -0.0000054, 0.0000055
1: -0.0061151, -0.0056187, -0.0061285, -0.0056110, -0.0002005, 0.0002072
2: 0.9691251, 0.9697208, 0.9691091, 0.9697300, -0.0002406, 0.0002487
3: 0.0185775, 0.0229715, 0.0184591, 0.0230393, -0.0017749, 0.0018343
4: -0.0024401, -0.0021060, -0.0024453, -0.0020969, -0.0001395, 0.0001350
5: 0.0148041, 0.0151419, 0.0147989, 0.0151510, -0.0001410, 0.0001364
6: 0.0045418, 0.0047061, 0.0045373, 0.0047086, -0.0000664, 0.0000686
7: -0.0137315, -0.0125928, -0.0137491, -0.0125621, -0.0004754, 0.0004600
8: 0.0058352, 0.0067386, 0.0058213, 0.0067630, -0.0003771, 0.0003649
9: 0.0082198, 0.0098447, 0.0081947, 0.0098885, -0.0006783, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.09 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040979, -0.0040842, -0.0000054, 0.0000063
1: -0.0061043, -0.0056125, -0.0061237, -0.0056110, -0.0002005, 0.0002352
2: 0.9691381, 0.9697283, 0.9691148, 0.9697300, -0.0002406, 0.0002823
3: 0.0186735, 0.0230266, 0.0185018, 0.0230393, -0.0017750, 0.0020818
4: -0.0024443, -0.0021133, -0.0024453, -0.0021002, -0.0001583, 0.0001350
5: 0.0147999, 0.0151345, 0.0147989, 0.0151477, -0.0001600, 0.0001364
6: 0.0045454, 0.0047081, 0.0045389, 0.0047086, -0.0000664, 0.0000778
7: -0.0137458, -0.0126177, -0.0137491, -0.0125732, -0.0005395, 0.0004600
8: 0.0058239, 0.0067189, 0.0058213, 0.0067542, -0.0004280, 0.0003649
9: 0.0081994, 0.0098092, 0.0081947, 0.0098727, -0.0007699, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040980, -0.0040845, -0.0000056, 0.0000054
1: -0.0061337, -0.0056227, -0.0061247, -0.0056187, -0.0002106, 0.0002038
2: 0.9691027, 0.9697160, 0.9691136, 0.9697208, -0.0002527, 0.0002446
3: 0.0184131, 0.0229363, 0.0184931, 0.0229715, -0.0018641, 0.0018039
4: -0.0024375, -0.0020935, -0.0024401, -0.0020995, -0.0001372, 0.0001418
5: 0.0148069, 0.0151545, 0.0148041, 0.0151484, -0.0001387, 0.0001433
6: 0.0045356, 0.0047047, 0.0045386, 0.0047061, -0.0000697, 0.0000674
7: -0.0137224, -0.0125502, -0.0137315, -0.0125709, -0.0004675, 0.0004831
8: 0.0058425, 0.0067724, 0.0058352, 0.0067560, -0.0003709, 0.0003833
9: 0.0082328, 0.0099055, 0.0082198, 0.0098759, -0.0006671, 0.0006893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040979, -0.0040845, -0.0000052, 0.0000058
1: -0.0061137, -0.0056180, -0.0061207, -0.0056187, -0.0001955, 0.0002182
2: 0.9691268, 0.9697217, 0.9691184, 0.9697208, -0.0002346, 0.0002618
3: 0.0185903, 0.0229782, 0.0185286, 0.0229715, -0.0017306, 0.0019310
4: -0.0024407, -0.0021069, -0.0024401, -0.0021022, -0.0001469, 0.0001316
5: 0.0148036, 0.0151409, 0.0148041, 0.0151457, -0.0001484, 0.0001330
6: 0.0045422, 0.0047063, 0.0045399, 0.0047061, -0.0000647, 0.0000722
7: -0.0137332, -0.0125961, -0.0137315, -0.0125801, -0.0005004, 0.0004485
8: 0.0058338, 0.0067360, 0.0058352, 0.0067487, -0.0003970, 0.0003558
9: 0.0082173, 0.0098400, 0.0082198, 0.0098628, -0.0007141, 0.0006400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.08 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040981, -0.0040842, -0.0000061, 0.0000057
1: -0.0061337, -0.0056227, -0.0061288, -0.0056110, -0.0002277, 0.0002145
2: 0.9691027, 0.9697160, 0.9691086, 0.9697300, -0.0002733, 0.0002574
3: 0.0184131, 0.0229363, 0.0184562, 0.0230393, -0.0020157, 0.0018983
4: -0.0024375, -0.0020935, -0.0024453, -0.0020967, -0.0001444, 0.0001533
5: 0.0148069, 0.0151545, 0.0147989, 0.0151512, -0.0001459, 0.0001549
6: 0.0045356, 0.0047047, 0.0045372, 0.0047086, -0.0000754, 0.0000710
7: -0.0137224, -0.0125502, -0.0137491, -0.0125613, -0.0004920, 0.0005224
8: 0.0058425, 0.0067724, 0.0058213, 0.0067636, -0.0003903, 0.0004144
9: 0.0082328, 0.0099055, 0.0081947, 0.0098895, -0.0007020, 0.0007454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.07 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040980, -0.0040842, -0.0000058, 0.0000061
1: -0.0061137, -0.0056180, -0.0061258, -0.0056110, -0.0002157, 0.0002294
2: 0.9691268, 0.9697217, 0.9691122, 0.9697300, -0.0002589, 0.0002753
3: 0.0185903, 0.0229782, 0.0184831, 0.0230393, -0.0019096, 0.0020304
4: -0.0024407, -0.0021069, -0.0024453, -0.0020988, -0.0001544, 0.0001452
5: 0.0148036, 0.0151409, 0.0147989, 0.0151492, -0.0001561, 0.0001468
6: 0.0045422, 0.0047063, 0.0045382, 0.0047086, -0.0000714, 0.0000759
7: -0.0137332, -0.0125961, -0.0137491, -0.0125683, -0.0005262, 0.0004949
8: 0.0058338, 0.0067360, 0.0058213, 0.0067580, -0.0004175, 0.0003926
9: 0.0082173, 0.0098400, 0.0081947, 0.0098796, -0.0007508, 0.0007062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.07 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040984, -0.0040846, -0.0000051, 0.0000058
1: -0.0061151, -0.0056187, -0.0061409, -0.0056227, -0.0001901, 0.0002167
2: 0.9691251, 0.9697208, 0.9690941, 0.9697160, -0.0002281, 0.0002600
3: 0.0185775, 0.0229715, 0.0183489, 0.0229363, -0.0016828, 0.0019177
4: -0.0024401, -0.0021060, -0.0024375, -0.0020886, -0.0001459, 0.0001280
5: 0.0148041, 0.0151419, 0.0148069, 0.0151595, -0.0001474, 0.0001293
6: 0.0045418, 0.0047061, 0.0045332, 0.0047047, -0.0000629, 0.0000717
7: -0.0137315, -0.0125928, -0.0137224, -0.0125335, -0.0004970, 0.0004361
8: 0.0058352, 0.0067386, 0.0058425, 0.0067856, -0.0003943, 0.0003460
9: 0.0082198, 0.0098447, 0.0082328, 0.0099292, -0.0007092, 0.0006223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040983, -0.0040846, -0.0000049, 0.0000065
1: -0.0061043, -0.0056125, -0.0061372, -0.0056227, -0.0001822, 0.0002428
2: 0.9691381, 0.9697283, 0.9690986, 0.9697160, -0.0002187, 0.0002914
3: 0.0186735, 0.0230266, 0.0183823, 0.0229363, -0.0016130, 0.0021492
4: -0.0024443, -0.0021133, -0.0024375, -0.0020911, -0.0001635, 0.0001227
5: 0.0147999, 0.0151345, 0.0148069, 0.0151569, -0.0001652, 0.0001240
6: 0.0045454, 0.0047081, 0.0045345, 0.0047047, -0.0000603, 0.0000804
7: -0.0137458, -0.0126177, -0.0137224, -0.0125422, -0.0005570, 0.0004180
8: 0.0058239, 0.0067189, 0.0058425, 0.0067788, -0.0004419, 0.0003316
9: 0.0081994, 0.0098092, 0.0082328, 0.0099169, -0.0007948, 0.0005965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 1.97 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001851
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001851
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040985, -0.0040844, -0.0000054, 0.0000060
1: -0.0061151, -0.0056187, -0.0061448, -0.0056154, -0.0002027, 0.0002264
2: 0.9691251, 0.9697208, 0.9690894, 0.9697248, -0.0002433, 0.0002717
3: 0.0185775, 0.0229715, 0.0183145, 0.0230010, -0.0017946, 0.0020039
4: -0.0024401, -0.0021060, -0.0024424, -0.0020860, -0.0001524, 0.0001365
5: 0.0148041, 0.0151419, 0.0148019, 0.0151621, -0.0001540, 0.0001379
6: 0.0045418, 0.0047061, 0.0045319, 0.0047072, -0.0000671, 0.0000749
7: -0.0137315, -0.0125928, -0.0137392, -0.0125246, -0.0005193, 0.0004651
8: 0.0058352, 0.0067386, 0.0058291, 0.0067927, -0.0004120, 0.0003690
9: 0.0082198, 0.0098447, 0.0082089, 0.0099419, -0.0007410, 0.0006636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040984, -0.0040844, -0.0000053, 0.0000068
1: -0.0061043, -0.0056125, -0.0061413, -0.0056154, -0.0001996, 0.0002530
2: 0.9691381, 0.9697283, 0.9690937, 0.9697248, -0.0002395, 0.0003036
3: 0.0186735, 0.0230266, 0.0183460, 0.0230010, -0.0017667, 0.0022397
4: -0.0024443, -0.0021133, -0.0024424, -0.0020883, -0.0001703, 0.0001344
5: 0.0147999, 0.0151345, 0.0148019, 0.0151597, -0.0001722, 0.0001358
6: 0.0045454, 0.0047081, 0.0045331, 0.0047072, -0.0000661, 0.0000837
7: -0.0137458, -0.0126177, -0.0137392, -0.0125328, -0.0005804, 0.0004579
8: 0.0058239, 0.0067189, 0.0058291, 0.0067862, -0.0004605, 0.0003632
9: 0.0081994, 0.0098092, 0.0082089, 0.0099303, -0.0008282, 0.0006533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.04 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040980, -0.0040845, -0.0000052, 0.0000057
1: -0.0061191, -0.0056110, -0.0061244, -0.0056187, -0.0001936, 0.0002138
2: 0.9691203, 0.9697300, 0.9691139, 0.9697208, -0.0002323, 0.0002565
3: 0.0185424, 0.0230393, 0.0184956, 0.0229715, -0.0017135, 0.0018922
4: -0.0024453, -0.0021033, -0.0024401, -0.0020997, -0.0001439, 0.0001303
5: 0.0147989, 0.0151446, 0.0148041, 0.0151482, -0.0001454, 0.0001317
6: 0.0045405, 0.0047086, 0.0045387, 0.0047061, -0.0000641, 0.0000707
7: -0.0137491, -0.0125837, -0.0137315, -0.0125715, -0.0004904, 0.0004441
8: 0.0058213, 0.0067458, 0.0058352, 0.0067555, -0.0003890, 0.0003523
9: 0.0081947, 0.0098577, 0.0082198, 0.0098750, -0.0006997, 0.0006337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.07 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040978, -0.0040845, -0.0000051, 0.0000063
1: -0.0061072, -0.0056053, -0.0061195, -0.0056187, -0.0001927, 0.0002376
2: 0.9691346, 0.9697368, 0.9691198, 0.9697208, -0.0002313, 0.0002851
3: 0.0186478, 0.0230899, 0.0185388, 0.0229715, -0.0017057, 0.0021027
4: -0.0024492, -0.0021113, -0.0024401, -0.0021030, -0.0001599, 0.0001297
5: 0.0147950, 0.0151365, 0.0148041, 0.0151449, -0.0001616, 0.0001311
6: 0.0045444, 0.0047105, 0.0045403, 0.0047061, -0.0000638, 0.0000786
7: -0.0137622, -0.0126110, -0.0137315, -0.0125827, -0.0005449, 0.0004421
8: 0.0058109, 0.0067242, 0.0058352, 0.0067466, -0.0004323, 0.0003507
9: 0.0081760, 0.0098187, 0.0082198, 0.0098590, -0.0007776, 0.0006308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.01 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001825
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001825
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040980, -0.0040842, -0.0000050, 0.0000053
1: -0.0061191, -0.0056110, -0.0061276, -0.0056110, -0.0001858, 0.0001977
2: 0.9691203, 0.9697300, 0.9691101, 0.9697300, -0.0002230, 0.0002373
3: 0.0185424, 0.0230393, 0.0184671, 0.0230393, -0.0016445, 0.0017500
4: -0.0024453, -0.0021033, -0.0024453, -0.0020976, -0.0001331, 0.0001251
5: 0.0147989, 0.0151446, 0.0147989, 0.0151504, -0.0001345, 0.0001264
6: 0.0045405, 0.0047086, 0.0045376, 0.0047086, -0.0000615, 0.0000654
7: -0.0137491, -0.0125837, -0.0137491, -0.0125642, -0.0004535, 0.0004262
8: 0.0058213, 0.0067458, 0.0058213, 0.0067613, -0.0003598, 0.0003381
9: 0.0081947, 0.0098577, 0.0081947, 0.0098855, -0.0006471, 0.0006081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.08 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040979, -0.0040842, -0.0000049, 0.0000061
1: -0.0061072, -0.0056053, -0.0061223, -0.0056110, -0.0001825, 0.0002274
2: 0.9691346, 0.9697368, 0.9691164, 0.9697300, -0.0002190, 0.0002729
3: 0.0186478, 0.0230899, 0.0185140, 0.0230393, -0.0016154, 0.0020127
4: -0.0024492, -0.0021113, -0.0024453, -0.0021011, -0.0001531, 0.0001229
5: 0.0147950, 0.0151365, 0.0147989, 0.0151468, -0.0001547, 0.0001242
6: 0.0045444, 0.0047105, 0.0045394, 0.0047086, -0.0000604, 0.0000753
7: -0.0137622, -0.0126110, -0.0137491, -0.0125763, -0.0005216, 0.0004186
8: 0.0058109, 0.0067242, 0.0058213, 0.0067517, -0.0004138, 0.0003321
9: 0.0081760, 0.0098187, 0.0081947, 0.0098682, -0.0007443, 0.0005974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.03 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001826
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001826
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040983, -0.0040844, -0.0040980, -0.0040845, -0.0000058, 0.0000058
1: -0.0061372, -0.0056154, -0.0061247, -0.0056187, -0.0002182, 0.0002164
2: 0.9690986, 0.9697248, 0.9691136, 0.9697208, -0.0002618, 0.0002597
3: 0.0183822, 0.0230010, 0.0184931, 0.0229715, -0.0019311, 0.0019157
4: -0.0024424, -0.0020911, -0.0024401, -0.0020995, -0.0001457, 0.0001469
5: 0.0148019, 0.0151569, 0.0148041, 0.0151484, -0.0001473, 0.0001484
6: 0.0045345, 0.0047072, 0.0045386, 0.0047061, -0.0000722, 0.0000716
7: -0.0137392, -0.0125422, -0.0137315, -0.0125709, -0.0004965, 0.0005005
8: 0.0058291, 0.0067788, 0.0058352, 0.0067560, -0.0003939, 0.0003970
9: 0.0082089, 0.0099169, 0.0082198, 0.0098759, -0.0007084, 0.0007141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.08 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040979, -0.0040845, -0.0000055, 0.0000061
1: -0.0061172, -0.0056099, -0.0061207, -0.0056187, -0.0002065, 0.0002300
2: 0.9691226, 0.9697313, 0.9691182, 0.9697208, -0.0002479, 0.0002760
3: 0.0185592, 0.0230492, 0.0185279, 0.0229715, -0.0018282, 0.0020355
4: -0.0024461, -0.0021046, -0.0024401, -0.0021022, -0.0001548, 0.0001390
5: 0.0147982, 0.0151433, 0.0148041, 0.0151457, -0.0001565, 0.0001405
6: 0.0045411, 0.0047090, 0.0045399, 0.0047061, -0.0000684, 0.0000761
7: -0.0137517, -0.0125880, -0.0137315, -0.0125799, -0.0005275, 0.0004738
8: 0.0058192, 0.0067424, 0.0058352, 0.0067488, -0.0004185, 0.0003759
9: 0.0081911, 0.0098515, 0.0082198, 0.0098631, -0.0007527, 0.0006760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.07 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040983, -0.0040844, -0.0040981, -0.0040842, -0.0000057, 0.0000055
1: -0.0061372, -0.0056154, -0.0061288, -0.0056110, -0.0002131, 0.0002064
2: 0.9690986, 0.9697248, 0.9691086, 0.9697300, -0.0002557, 0.0002477
3: 0.0183822, 0.0230010, 0.0184562, 0.0230393, -0.0018860, 0.0018270
4: -0.0024424, -0.0020911, -0.0024453, -0.0020967, -0.0001390, 0.0001434
5: 0.0148019, 0.0151569, 0.0147989, 0.0151512, -0.0001404, 0.0001450
6: 0.0045345, 0.0047072, 0.0045372, 0.0047086, -0.0000705, 0.0000683
7: -0.0137392, -0.0125422, -0.0137491, -0.0125613, -0.0004735, 0.0004888
8: 0.0058291, 0.0067788, 0.0058213, 0.0067636, -0.0003756, 0.0003878
9: 0.0082089, 0.0099169, 0.0081947, 0.0098895, -0.0006756, 0.0006974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040980, -0.0040842, -0.0000053, 0.0000059
1: -0.0061172, -0.0056099, -0.0061247, -0.0056110, -0.0001978, 0.0002213
2: 0.9691226, 0.9697313, 0.9691135, 0.9697300, -0.0002373, 0.0002656
3: 0.0185592, 0.0230492, 0.0184924, 0.0230393, -0.0017506, 0.0019587
4: -0.0024461, -0.0021046, -0.0024453, -0.0020995, -0.0001490, 0.0001331
5: 0.0147982, 0.0151433, 0.0147989, 0.0151484, -0.0001506, 0.0001346
6: 0.0045411, 0.0047090, 0.0045386, 0.0047086, -0.0000655, 0.0000732
7: -0.0137517, -0.0125880, -0.0137491, -0.0125707, -0.0005076, 0.0004537
8: 0.0058192, 0.0067424, 0.0058213, 0.0067561, -0.0004027, 0.0003599
9: 0.0081911, 0.0098515, 0.0081947, 0.0098762, -0.0007243, 0.0006474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.04 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001849, upper bound: 0.0001815
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001849, upper bound: 0.0001815
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040984, -0.0040846, -0.0000053, 0.0000062
1: -0.0061191, -0.0056110, -0.0061409, -0.0056227, -0.0002003, 0.0002338
2: 0.9691203, 0.9697300, 0.9690941, 0.9697160, -0.0002404, 0.0002805
3: 0.0185424, 0.0230393, 0.0183489, 0.0229363, -0.0017729, 0.0020693
4: -0.0024453, -0.0021033, -0.0024375, -0.0020886, -0.0001574, 0.0001348
5: 0.0147989, 0.0151446, 0.0148069, 0.0151595, -0.0001591, 0.0001363
6: 0.0045405, 0.0047086, 0.0045332, 0.0047047, -0.0000663, 0.0000774
7: -0.0137491, -0.0125837, -0.0137224, -0.0125335, -0.0005363, 0.0004595
8: 0.0058213, 0.0067458, 0.0058425, 0.0067856, -0.0004255, 0.0003645
9: 0.0081947, 0.0098577, 0.0082328, 0.0099292, -0.0007652, 0.0006556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040983, -0.0040846, -0.0000052, 0.0000069
1: -0.0061072, -0.0056053, -0.0061380, -0.0056227, -0.0001946, 0.0002568
2: 0.9691346, 0.9697368, 0.9690976, 0.9697160, -0.0002336, 0.0003082
3: 0.0186478, 0.0230899, 0.0183748, 0.0229363, -0.0017228, 0.0022732
4: -0.0024492, -0.0021113, -0.0024375, -0.0020905, -0.0001729, 0.0001310
5: 0.0147950, 0.0151365, 0.0148069, 0.0151575, -0.0001747, 0.0001324
6: 0.0045444, 0.0047105, 0.0045342, 0.0047047, -0.0000644, 0.0000850
7: -0.0137622, -0.0126110, -0.0137224, -0.0125403, -0.0005891, 0.0004465
8: 0.0058109, 0.0067242, 0.0058425, 0.0067803, -0.0004674, 0.0003542
9: 0.0081760, 0.0098187, 0.0082328, 0.0099196, -0.0008406, 0.0006371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.03 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040985, -0.0040844, -0.0000051, 0.0000059
1: -0.0061191, -0.0056110, -0.0061448, -0.0056154, -0.0001925, 0.0002192
2: 0.9691203, 0.9697300, 0.9690894, 0.9697248, -0.0002310, 0.0002630
3: 0.0185424, 0.0230393, 0.0183145, 0.0230010, -0.0017041, 0.0019401
4: -0.0024453, -0.0021033, -0.0024424, -0.0020860, -0.0001476, 0.0001296
5: 0.0147989, 0.0151446, 0.0148019, 0.0151621, -0.0001491, 0.0001310
6: 0.0045405, 0.0047086, 0.0045319, 0.0047072, -0.0000637, 0.0000725
7: -0.0137491, -0.0125837, -0.0137392, -0.0125246, -0.0005028, 0.0004416
8: 0.0058213, 0.0067458, 0.0058291, 0.0067927, -0.0003989, 0.0003504
9: 0.0081947, 0.0098577, 0.0082089, 0.0099419, -0.0007174, 0.0006302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.00 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040984, -0.0040844, -0.0000049, 0.0000066
1: -0.0061072, -0.0056053, -0.0061411, -0.0056154, -0.0001845, 0.0002460
2: 0.9691346, 0.9697368, 0.9690939, 0.9697248, -0.0002214, 0.0002952
3: 0.0186478, 0.0230899, 0.0183478, 0.0230010, -0.0016328, 0.0021773
4: -0.0024492, -0.0021113, -0.0024424, -0.0020885, -0.0001656, 0.0001242
5: 0.0147950, 0.0151365, 0.0148019, 0.0151596, -0.0001674, 0.0001255
6: 0.0045444, 0.0047105, 0.0045332, 0.0047072, -0.0000610, 0.0000814
7: -0.0137622, -0.0126110, -0.0137392, -0.0125333, -0.0005643, 0.0004232
8: 0.0058109, 0.0067242, 0.0058291, 0.0067859, -0.0004477, 0.0003357
9: 0.0081760, 0.0098187, 0.0082089, 0.0099296, -0.0008052, 0.0006038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.16 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001842
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.74 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001851
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001851
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001825
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001825
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001826
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001826
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001849, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001849, upper bound: 0.0001815
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001842
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040977, -0.0040845, -0.0000049, 0.0000049
1: -0.0061151, -0.0056187, -0.0061151, -0.0056187, -0.0001834, 0.0001834
2: 0.9691251, 0.9697208, 0.9691251, 0.9697208, -0.0002201, 0.0002201
3: 0.0185775, 0.0229715, 0.0185775, 0.0229715, -0.0016233, 0.0016233
4: -0.0024401, -0.0021060, -0.0024401, -0.0021060, -0.0001235, 0.0001235
5: 0.0148041, 0.0151419, 0.0148041, 0.0151419, -0.0001248, 0.0001248
6: 0.0045418, 0.0047061, 0.0045418, 0.0047061, -0.0000607, 0.0000607
7: -0.0137315, -0.0125928, -0.0137315, -0.0125928, -0.0004207, 0.0004207
8: 0.0058352, 0.0067386, 0.0058352, 0.0067386, -0.0003338, 0.0003338
9: 0.0082198, 0.0098447, 0.0082198, 0.0098447, -0.0006003, 0.0006003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001863, upper bound: 0.0001824
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040974, -0.0040843, -0.0000057, 0.0000051
1: -0.0061151, -0.0056187, -0.0061043, -0.0056125, -0.0002131, 0.0001907
2: 0.9691251, 0.9697208, 0.9691381, 0.9697283, -0.0002557, 0.0002289
3: 0.0185775, 0.0229715, 0.0186735, 0.0230266, -0.0018860, 0.0016882
4: -0.0024401, -0.0021060, -0.0024443, -0.0021133, -0.0001284, 0.0001434
5: 0.0148041, 0.0151419, 0.0147999, 0.0151345, -0.0001298, 0.0001450
6: 0.0045418, 0.0047061, 0.0045454, 0.0047081, -0.0000705, 0.0000631
7: -0.0137315, -0.0125928, -0.0137458, -0.0126177, -0.0004375, 0.0004888
8: 0.0058352, 0.0067386, 0.0058239, 0.0067189, -0.0003471, 0.0003878
9: 0.0082198, 0.0098447, 0.0081994, 0.0098092, -0.0006243, 0.0006974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001863, upper bound: 0.0001824
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001824
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040977, -0.0040845, -0.0000051, 0.0000057
1: -0.0061043, -0.0056125, -0.0061151, -0.0056187, -0.0001907, 0.0002131
2: 0.9691381, 0.9697283, 0.9691251, 0.9697208, -0.0002289, 0.0002557
3: 0.0186735, 0.0230266, 0.0185775, 0.0229715, -0.0016882, 0.0018860
4: -0.0024443, -0.0021133, -0.0024401, -0.0021060, -0.0001434, 0.0001284
5: 0.0147999, 0.0151345, 0.0148041, 0.0151419, -0.0001450, 0.0001298
6: 0.0045454, 0.0047081, 0.0045418, 0.0047061, -0.0000631, 0.0000705
7: -0.0137458, -0.0126177, -0.0137315, -0.0125928, -0.0004888, 0.0004375
8: 0.0058239, 0.0067189, 0.0058352, 0.0067386, -0.0003878, 0.0003471
9: 0.0081994, 0.0098092, 0.0082198, 0.0098447, -0.0006974, 0.0006243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040974, -0.0040843, -0.0000048, 0.0000048
1: -0.0061043, -0.0056125, -0.0061043, -0.0056125, -0.0001803, 0.0001803
2: 0.9691381, 0.9697283, 0.9691381, 0.9697283, -0.0002164, 0.0002164
3: 0.0186735, 0.0230266, 0.0186735, 0.0230266, -0.0015959, 0.0015959
4: -0.0024443, -0.0021133, -0.0024443, -0.0021133, -0.0001214, 0.0001214
5: 0.0147999, 0.0151345, 0.0147999, 0.0151345, -0.0001227, 0.0001227
6: 0.0045454, 0.0047081, 0.0045454, 0.0047081, -0.0000597, 0.0000597
7: -0.0137458, -0.0126177, -0.0137458, -0.0126177, -0.0004136, 0.0004136
8: 0.0058239, 0.0067189, 0.0058239, 0.0067189, -0.0003281, 0.0003281
9: 0.0081994, 0.0098092, 0.0081994, 0.0098092, -0.0005902, 0.0005902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001824
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001829
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040978, -0.0040842, -0.0000054, 0.0000052
1: -0.0061151, -0.0056187, -0.0061191, -0.0056110, -0.0002005, 0.0001936
2: 0.9691251, 0.9697208, 0.9691203, 0.9697300, -0.0002406, 0.0002323
3: 0.0185775, 0.0229715, 0.0185424, 0.0230393, -0.0017749, 0.0017135
4: -0.0024401, -0.0021060, -0.0024453, -0.0021033, -0.0001303, 0.0001350
5: 0.0148041, 0.0151419, 0.0147989, 0.0151446, -0.0001317, 0.0001364
6: 0.0045418, 0.0047061, 0.0045405, 0.0047086, -0.0000664, 0.0000641
7: -0.0137315, -0.0125928, -0.0137491, -0.0125837, -0.0004441, 0.0004600
8: 0.0058352, 0.0067386, 0.0058213, 0.0067458, -0.0003523, 0.0003649
9: 0.0082198, 0.0098447, 0.0081947, 0.0098577, -0.0006337, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001842
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040975, -0.0040841, -0.0000060, 0.0000053
1: -0.0061151, -0.0056187, -0.0061072, -0.0056053, -0.0002263, 0.0002001
2: 0.9691251, 0.9697208, 0.9691346, 0.9697368, -0.0002716, 0.0002402
3: 0.0185775, 0.0229715, 0.0186478, 0.0230899, -0.0020032, 0.0017714
4: -0.0024401, -0.0021060, -0.0024492, -0.0021113, -0.0001347, 0.0001524
5: 0.0148041, 0.0151419, 0.0147950, 0.0151365, -0.0001362, 0.0001540
6: 0.0045418, 0.0047061, 0.0045444, 0.0047105, -0.0000749, 0.0000662
7: -0.0137315, -0.0125928, -0.0137622, -0.0126110, -0.0004591, 0.0005191
8: 0.0058352, 0.0067386, 0.0058109, 0.0067242, -0.0003642, 0.0004119
9: 0.0082198, 0.0098447, 0.0081760, 0.0098187, -0.0006551, 0.0007408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001842
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001842
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040978, -0.0040842, -0.0000056, 0.0000060
1: -0.0061043, -0.0056125, -0.0061191, -0.0056110, -0.0002079, 0.0002233
2: 0.9691381, 0.9697283, 0.9691203, 0.9697300, -0.0002494, 0.0002679
3: 0.0186735, 0.0230266, 0.0185424, 0.0230393, -0.0018398, 0.0019762
4: -0.0024443, -0.0021133, -0.0024453, -0.0021033, -0.0001503, 0.0001399
5: 0.0147999, 0.0151345, 0.0147989, 0.0151446, -0.0001519, 0.0001414
6: 0.0045454, 0.0047081, 0.0045405, 0.0047086, -0.0000688, 0.0000739
7: -0.0137458, -0.0126177, -0.0137491, -0.0125837, -0.0005121, 0.0004768
8: 0.0058239, 0.0067189, 0.0058213, 0.0067458, -0.0004063, 0.0003783
9: 0.0081994, 0.0098092, 0.0081947, 0.0098577, -0.0007308, 0.0006804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040975, -0.0040841, -0.0000054, 0.0000051
1: -0.0061043, -0.0056125, -0.0061072, -0.0056053, -0.0002005, 0.0001927
2: 0.9691381, 0.9697283, 0.9691346, 0.9697368, -0.0002406, 0.0002313
3: 0.0186735, 0.0230266, 0.0186478, 0.0230899, -0.0017750, 0.0017057
4: -0.0024443, -0.0021133, -0.0024492, -0.0021113, -0.0001297, 0.0001350
5: 0.0147999, 0.0151345, 0.0147950, 0.0151365, -0.0001311, 0.0001364
6: 0.0045454, 0.0047081, 0.0045444, 0.0047105, -0.0000664, 0.0000638
7: -0.0137458, -0.0126177, -0.0137622, -0.0126110, -0.0004421, 0.0004600
8: 0.0058239, 0.0067189, 0.0058109, 0.0067242, -0.0003507, 0.0003649
9: 0.0081994, 0.0098092, 0.0081760, 0.0098187, -0.0006308, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.07 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001842
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001855
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040977, -0.0040845, -0.0000056, 0.0000051
1: -0.0061337, -0.0056227, -0.0061151, -0.0056187, -0.0002106, 0.0001901
2: 0.9691027, 0.9697160, 0.9691251, 0.9697208, -0.0002527, 0.0002281
3: 0.0184131, 0.0229363, 0.0185775, 0.0229715, -0.0018641, 0.0016828
4: -0.0024375, -0.0020935, -0.0024401, -0.0021060, -0.0001280, 0.0001418
5: 0.0148069, 0.0151545, 0.0148041, 0.0151419, -0.0001293, 0.0001433
6: 0.0045356, 0.0047047, 0.0045418, 0.0047061, -0.0000697, 0.0000629
7: -0.0137224, -0.0125502, -0.0137315, -0.0125928, -0.0004361, 0.0004831
8: 0.0058425, 0.0067724, 0.0058352, 0.0067386, -0.0003460, 0.0003833
9: 0.0082328, 0.0099055, 0.0082198, 0.0098447, -0.0006223, 0.0006893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001923, upper bound: 0.0001816
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040974, -0.0040843, -0.0000064, 0.0000053
1: -0.0061337, -0.0056227, -0.0061043, -0.0056125, -0.0002403, 0.0001974
2: 0.9691027, 0.9697160, 0.9691381, 0.9697283, -0.0002883, 0.0002369
3: 0.0184131, 0.0229363, 0.0186735, 0.0230266, -0.0021267, 0.0017476
4: -0.0024375, -0.0020935, -0.0024443, -0.0021133, -0.0001329, 0.0001618
5: 0.0148069, 0.0151545, 0.0147999, 0.0151345, -0.0001343, 0.0001635
6: 0.0045356, 0.0047047, 0.0045454, 0.0047081, -0.0000795, 0.0000653
7: -0.0137224, -0.0125502, -0.0137458, -0.0126177, -0.0004529, 0.0005512
8: 0.0058425, 0.0067724, 0.0058239, 0.0067189, -0.0003593, 0.0004373
9: 0.0082328, 0.0099055, 0.0081994, 0.0098092, -0.0006463, 0.0007865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001923, upper bound: 0.0001816
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001816
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040977, -0.0040845, -0.0000053, 0.0000055
1: -0.0061137, -0.0056180, -0.0061151, -0.0056187, -0.0001992, 0.0002064
2: 0.9691268, 0.9697217, 0.9691251, 0.9697208, -0.0002391, 0.0002477
3: 0.0185903, 0.0229782, 0.0185775, 0.0229715, -0.0017633, 0.0018271
4: -0.0024407, -0.0021069, -0.0024401, -0.0021060, -0.0001390, 0.0001341
5: 0.0148036, 0.0151409, 0.0148041, 0.0151419, -0.0001404, 0.0001355
6: 0.0045422, 0.0047063, 0.0045418, 0.0047061, -0.0000659, 0.0000683
7: -0.0137332, -0.0125961, -0.0137315, -0.0125928, -0.0004735, 0.0004570
8: 0.0058338, 0.0067360, 0.0058352, 0.0067386, -0.0003757, 0.0003625
9: 0.0082173, 0.0098400, 0.0082198, 0.0098447, -0.0006757, 0.0006521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.07 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040974, -0.0040843, -0.0000052, 0.0000049
1: -0.0061137, -0.0056180, -0.0061043, -0.0056125, -0.0001955, 0.0001822
2: 0.9691268, 0.9697217, 0.9691381, 0.9697283, -0.0002346, 0.0002187
3: 0.0185903, 0.0229782, 0.0186735, 0.0230266, -0.0017306, 0.0016130
4: -0.0024407, -0.0021069, -0.0024443, -0.0021133, -0.0001227, 0.0001316
5: 0.0148036, 0.0151409, 0.0147999, 0.0151345, -0.0001240, 0.0001330
6: 0.0045422, 0.0047063, 0.0045454, 0.0047081, -0.0000647, 0.0000603
7: -0.0137332, -0.0125961, -0.0137458, -0.0126177, -0.0004180, 0.0004485
8: 0.0058338, 0.0067360, 0.0058239, 0.0067189, -0.0003316, 0.0003558
9: 0.0082173, 0.0098400, 0.0081994, 0.0098092, -0.0005965, 0.0006400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.01 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001816
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001817
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040978, -0.0040842, -0.0000061, 0.0000053
1: -0.0061337, -0.0056227, -0.0061191, -0.0056110, -0.0002277, 0.0002003
2: 0.9691027, 0.9697160, 0.9691203, 0.9697300, -0.0002733, 0.0002404
3: 0.0184131, 0.0229363, 0.0185424, 0.0230393, -0.0020157, 0.0017729
4: -0.0024375, -0.0020935, -0.0024453, -0.0021033, -0.0001348, 0.0001533
5: 0.0148069, 0.0151545, 0.0147989, 0.0151446, -0.0001363, 0.0001549
6: 0.0045356, 0.0047047, 0.0045405, 0.0047086, -0.0000754, 0.0000663
7: -0.0137224, -0.0125502, -0.0137491, -0.0125837, -0.0004595, 0.0005224
8: 0.0058425, 0.0067724, 0.0058213, 0.0067458, -0.0003645, 0.0004144
9: 0.0082328, 0.0099055, 0.0081947, 0.0098577, -0.0006556, 0.0007454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.11 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001834
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040975, -0.0040841, -0.0000068, 0.0000055
1: -0.0061337, -0.0056227, -0.0061072, -0.0056053, -0.0002535, 0.0002068
2: 0.9691027, 0.9697160, 0.9691346, 0.9697368, -0.0003042, 0.0002482
3: 0.0184131, 0.0229363, 0.0186478, 0.0230899, -0.0022439, 0.0018309
4: -0.0024375, -0.0020935, -0.0024492, -0.0021113, -0.0001392, 0.0001707
5: 0.0148069, 0.0151545, 0.0147950, 0.0151365, -0.0001407, 0.0001725
6: 0.0045356, 0.0047047, 0.0045444, 0.0047105, -0.0000839, 0.0000685
7: -0.0137224, -0.0125502, -0.0137622, -0.0126110, -0.0004745, 0.0005815
8: 0.0058425, 0.0067724, 0.0058109, 0.0067242, -0.0003764, 0.0004614
9: 0.0082328, 0.0099055, 0.0081760, 0.0098187, -0.0006770, 0.0008298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001834
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001834
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040978, -0.0040842, -0.0000058, 0.0000058
1: -0.0061137, -0.0056180, -0.0061191, -0.0056110, -0.0002163, 0.0002166
2: 0.9691268, 0.9697217, 0.9691203, 0.9697300, -0.0002596, 0.0002599
3: 0.0185903, 0.0229782, 0.0185424, 0.0230393, -0.0019149, 0.0019173
4: -0.0024407, -0.0021069, -0.0024453, -0.0021033, -0.0001458, 0.0001456
5: 0.0148036, 0.0151409, 0.0147989, 0.0151446, -0.0001474, 0.0001472
6: 0.0045422, 0.0047063, 0.0045405, 0.0047086, -0.0000716, 0.0000717
7: -0.0137332, -0.0125961, -0.0137491, -0.0125837, -0.0004969, 0.0004963
8: 0.0058338, 0.0067360, 0.0058213, 0.0067458, -0.0003942, 0.0003937
9: 0.0082173, 0.0098400, 0.0081947, 0.0098577, -0.0007090, 0.0007081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040975, -0.0040841, -0.0000058, 0.0000052
1: -0.0061137, -0.0056180, -0.0061072, -0.0056053, -0.0002157, 0.0001946
2: 0.9691268, 0.9697217, 0.9691346, 0.9697368, -0.0002589, 0.0002336
3: 0.0185903, 0.0229782, 0.0186478, 0.0230899, -0.0019096, 0.0017228
4: -0.0024407, -0.0021069, -0.0024492, -0.0021113, -0.0001310, 0.0001452
5: 0.0148036, 0.0151409, 0.0147950, 0.0151365, -0.0001324, 0.0001468
6: 0.0045422, 0.0047063, 0.0045444, 0.0047105, -0.0000714, 0.0000644
7: -0.0137332, -0.0125961, -0.0137622, -0.0126110, -0.0004465, 0.0004949
8: 0.0058338, 0.0067360, 0.0058109, 0.0067242, -0.0003542, 0.0003926
9: 0.0082173, 0.0098400, 0.0081760, 0.0098187, -0.0006371, 0.0007062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001834
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001836
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040982, -0.0040846, -0.0000051, 0.0000056
1: -0.0061151, -0.0056187, -0.0061337, -0.0056227, -0.0001901, 0.0002106
2: 0.9691251, 0.9697208, 0.9691027, 0.9697160, -0.0002281, 0.0002527
3: 0.0185775, 0.0229715, 0.0184131, 0.0229363, -0.0016828, 0.0018641
4: -0.0024401, -0.0021060, -0.0024375, -0.0020935, -0.0001418, 0.0001280
5: 0.0148041, 0.0151419, 0.0148069, 0.0151545, -0.0001433, 0.0001293
6: 0.0045418, 0.0047061, 0.0045356, 0.0047047, -0.0000629, 0.0000697
7: -0.0137315, -0.0125928, -0.0137224, -0.0125502, -0.0004831, 0.0004361
8: 0.0058352, 0.0067386, 0.0058425, 0.0067724, -0.0003833, 0.0003460
9: 0.0082198, 0.0098447, 0.0082328, 0.0099055, -0.0006893, 0.0006223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.03 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001833, upper bound: 0.0001830
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040977, -0.0040844, -0.0000055, 0.0000053
1: -0.0061151, -0.0056187, -0.0061137, -0.0056180, -0.0002064, 0.0001992
2: 0.9691251, 0.9697208, 0.9691268, 0.9697217, -0.0002477, 0.0002391
3: 0.0185775, 0.0229715, 0.0185903, 0.0229782, -0.0018271, 0.0017633
4: -0.0024401, -0.0021060, -0.0024407, -0.0021069, -0.0001341, 0.0001390
5: 0.0148041, 0.0151419, 0.0148036, 0.0151409, -0.0001355, 0.0001404
6: 0.0045418, 0.0047061, 0.0045422, 0.0047063, -0.0000683, 0.0000659
7: -0.0137315, -0.0125928, -0.0137332, -0.0125961, -0.0004570, 0.0004735
8: 0.0058352, 0.0067386, 0.0058338, 0.0067360, -0.0003625, 0.0003757
9: 0.0082198, 0.0098447, 0.0082173, 0.0098400, -0.0006521, 0.0006757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.01 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001833, upper bound: 0.0001830
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040982, -0.0040846, -0.0000053, 0.0000064
1: -0.0061043, -0.0056125, -0.0061337, -0.0056227, -0.0001974, 0.0002403
2: 0.9691381, 0.9697283, 0.9691027, 0.9697160, -0.0002369, 0.0002883
3: 0.0186735, 0.0230266, 0.0184131, 0.0229363, -0.0017476, 0.0021267
4: -0.0024443, -0.0021133, -0.0024375, -0.0020935, -0.0001618, 0.0001329
5: 0.0147999, 0.0151345, 0.0148069, 0.0151545, -0.0001635, 0.0001343
6: 0.0045454, 0.0047081, 0.0045356, 0.0047047, -0.0000653, 0.0000795
7: -0.0137458, -0.0126177, -0.0137224, -0.0125502, -0.0005512, 0.0004529
8: 0.0058239, 0.0067189, 0.0058425, 0.0067724, -0.0004373, 0.0003593
9: 0.0081994, 0.0098092, 0.0082328, 0.0099055, -0.0007865, 0.0006463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.08 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001851
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040977, -0.0040844, -0.0000049, 0.0000052
1: -0.0061043, -0.0056125, -0.0061137, -0.0056180, -0.0001822, 0.0001955
2: 0.9691381, 0.9697283, 0.9691268, 0.9697217, -0.0002187, 0.0002346
3: 0.0186735, 0.0230266, 0.0185903, 0.0229782, -0.0016130, 0.0017306
4: -0.0024443, -0.0021133, -0.0024407, -0.0021069, -0.0001316, 0.0001227
5: 0.0147999, 0.0151345, 0.0148036, 0.0151409, -0.0001330, 0.0001240
6: 0.0045454, 0.0047081, 0.0045422, 0.0047063, -0.0000603, 0.0000647
7: -0.0137458, -0.0126177, -0.0137332, -0.0125961, -0.0004485, 0.0004180
8: 0.0058239, 0.0067189, 0.0058338, 0.0067360, -0.0003558, 0.0003316
9: 0.0081994, 0.0098092, 0.0082173, 0.0098400, -0.0006400, 0.0005965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001817, upper bound: 0.0001830
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001850
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040983, -0.0040844, -0.0000054, 0.0000058
1: -0.0061151, -0.0056187, -0.0061372, -0.0056154, -0.0002027, 0.0002182
2: 0.9691251, 0.9697208, 0.9690986, 0.9697248, -0.0002433, 0.0002618
3: 0.0185775, 0.0229715, 0.0183822, 0.0230010, -0.0017946, 0.0019311
4: -0.0024401, -0.0021060, -0.0024424, -0.0020911, -0.0001469, 0.0001365
5: 0.0148041, 0.0151419, 0.0148019, 0.0151569, -0.0001484, 0.0001379
6: 0.0045418, 0.0047061, 0.0045345, 0.0047072, -0.0000671, 0.0000722
7: -0.0137315, -0.0125928, -0.0137392, -0.0125422, -0.0005005, 0.0004651
8: 0.0058352, 0.0067386, 0.0058291, 0.0067788, -0.0003970, 0.0003690
9: 0.0082198, 0.0098447, 0.0082089, 0.0099169, -0.0007141, 0.0006636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001827, upper bound: 0.0001850
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040978, -0.0040842, -0.0000058, 0.0000055
1: -0.0061151, -0.0056187, -0.0061172, -0.0056099, -0.0002182, 0.0002072
2: 0.9691251, 0.9697208, 0.9691226, 0.9697313, -0.0002618, 0.0002487
3: 0.0185775, 0.0229715, 0.0185592, 0.0230492, -0.0019313, 0.0018341
4: -0.0024401, -0.0021060, -0.0024461, -0.0021046, -0.0001395, 0.0001469
5: 0.0148041, 0.0151419, 0.0147982, 0.0151433, -0.0001410, 0.0001485
6: 0.0045418, 0.0047061, 0.0045411, 0.0047090, -0.0000722, 0.0000686
7: -0.0137315, -0.0125928, -0.0137517, -0.0125880, -0.0004753, 0.0005005
8: 0.0058352, 0.0067386, 0.0058192, 0.0067424, -0.0003771, 0.0003971
9: 0.0082198, 0.0098447, 0.0081911, 0.0098515, -0.0006782, 0.0007142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001828, upper bound: 0.0001850
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040983, -0.0040844, -0.0000056, 0.0000066
1: -0.0061043, -0.0056125, -0.0061372, -0.0056154, -0.0002101, 0.0002478
2: 0.9691381, 0.9697283, 0.9690986, 0.9697248, -0.0002521, 0.0002974
3: 0.0186735, 0.0230266, 0.0183822, 0.0230010, -0.0018594, 0.0021938
4: -0.0024443, -0.0021133, -0.0024424, -0.0020911, -0.0001669, 0.0001414
5: 0.0147999, 0.0151345, 0.0148019, 0.0151569, -0.0001686, 0.0001429
6: 0.0045454, 0.0047081, 0.0045345, 0.0047072, -0.0000695, 0.0000820
7: -0.0137458, -0.0126177, -0.0137392, -0.0125422, -0.0005685, 0.0004819
8: 0.0058239, 0.0067189, 0.0058291, 0.0067788, -0.0004511, 0.0003823
9: 0.0081994, 0.0098092, 0.0082089, 0.0099169, -0.0008113, 0.0006876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040978, -0.0040842, -0.0000053, 0.0000055
1: -0.0061043, -0.0056125, -0.0061172, -0.0056099, -0.0001996, 0.0002065
2: 0.9691381, 0.9697283, 0.9691226, 0.9697313, -0.0002395, 0.0002479
3: 0.0186735, 0.0230266, 0.0185592, 0.0230492, -0.0017667, 0.0018282
4: -0.0024443, -0.0021133, -0.0024461, -0.0021046, -0.0001390, 0.0001344
5: 0.0147999, 0.0151345, 0.0147982, 0.0151433, -0.0001405, 0.0001358
6: 0.0045454, 0.0047081, 0.0045411, 0.0047090, -0.0000661, 0.0000684
7: -0.0137458, -0.0126177, -0.0137517, -0.0125880, -0.0004738, 0.0004579
8: 0.0058239, 0.0067189, 0.0058192, 0.0067424, -0.0003759, 0.0003632
9: 0.0081994, 0.0098092, 0.0081911, 0.0098515, -0.0006760, 0.0006533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040977, -0.0040845, -0.0000052, 0.0000054
1: -0.0061191, -0.0056110, -0.0061151, -0.0056187, -0.0001936, 0.0002005
2: 0.9691203, 0.9697300, 0.9691251, 0.9697208, -0.0002323, 0.0002406
3: 0.0185424, 0.0230393, 0.0185775, 0.0229715, -0.0017135, 0.0017749
4: -0.0024453, -0.0021033, -0.0024401, -0.0021060, -0.0001350, 0.0001303
5: 0.0147989, 0.0151446, 0.0148041, 0.0151419, -0.0001364, 0.0001317
6: 0.0045405, 0.0047086, 0.0045418, 0.0047061, -0.0000641, 0.0000664
7: -0.0137491, -0.0125837, -0.0137315, -0.0125928, -0.0004600, 0.0004441
8: 0.0058213, 0.0067458, 0.0058352, 0.0067386, -0.0003649, 0.0003523
9: 0.0081947, 0.0098577, 0.0082198, 0.0098447, -0.0006564, 0.0006337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.01 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040974, -0.0040843, -0.0000060, 0.0000056
1: -0.0061191, -0.0056110, -0.0061043, -0.0056125, -0.0002233, 0.0002079
2: 0.9691203, 0.9697300, 0.9691381, 0.9697283, -0.0002679, 0.0002494
3: 0.0185424, 0.0230393, 0.0186735, 0.0230266, -0.0019762, 0.0018398
4: -0.0024453, -0.0021033, -0.0024443, -0.0021133, -0.0001399, 0.0001503
5: 0.0147989, 0.0151446, 0.0147999, 0.0151345, -0.0001414, 0.0001519
6: 0.0045405, 0.0047086, 0.0045454, 0.0047081, -0.0000739, 0.0000688
7: -0.0137491, -0.0125837, -0.0137458, -0.0126177, -0.0004768, 0.0005121
8: 0.0058213, 0.0067458, 0.0058239, 0.0067189, -0.0003783, 0.0004063
9: 0.0081947, 0.0098577, 0.0081994, 0.0098092, -0.0006804, 0.0007308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040977, -0.0040845, -0.0000053, 0.0000060
1: -0.0061072, -0.0056053, -0.0061151, -0.0056187, -0.0002001, 0.0002263
2: 0.9691346, 0.9697368, 0.9691251, 0.9697208, -0.0002402, 0.0002716
3: 0.0186478, 0.0230899, 0.0185775, 0.0229715, -0.0017714, 0.0020032
4: -0.0024492, -0.0021113, -0.0024401, -0.0021060, -0.0001524, 0.0001347
5: 0.0147950, 0.0151365, 0.0148041, 0.0151419, -0.0001540, 0.0001362
6: 0.0045444, 0.0047105, 0.0045418, 0.0047061, -0.0000662, 0.0000749
7: -0.0137622, -0.0126110, -0.0137315, -0.0125928, -0.0005191, 0.0004591
8: 0.0058109, 0.0067242, 0.0058352, 0.0067386, -0.0004119, 0.0003642
9: 0.0081760, 0.0098187, 0.0082198, 0.0098447, -0.0007408, 0.0006551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001825
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040974, -0.0040843, -0.0000051, 0.0000054
1: -0.0061072, -0.0056053, -0.0061043, -0.0056125, -0.0001927, 0.0002005
2: 0.9691346, 0.9697368, 0.9691381, 0.9697283, -0.0002313, 0.0002406
3: 0.0186478, 0.0230899, 0.0186735, 0.0230266, -0.0017057, 0.0017750
4: -0.0024492, -0.0021113, -0.0024443, -0.0021133, -0.0001350, 0.0001297
5: 0.0147950, 0.0151365, 0.0147999, 0.0151345, -0.0001364, 0.0001311
6: 0.0045444, 0.0047105, 0.0045454, 0.0047081, -0.0000638, 0.0000664
7: -0.0137622, -0.0126110, -0.0137458, -0.0126177, -0.0004600, 0.0004421
8: 0.0058109, 0.0067242, 0.0058239, 0.0067189, -0.0003649, 0.0003507
9: 0.0081760, 0.0098187, 0.0081994, 0.0098092, -0.0006564, 0.0006308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001825
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040978, -0.0040842, -0.0000050, 0.0000050
1: -0.0061191, -0.0056110, -0.0061191, -0.0056110, -0.0001858, 0.0001858
2: 0.9691203, 0.9697300, 0.9691203, 0.9697300, -0.0002230, 0.0002230
3: 0.0185424, 0.0230393, 0.0185424, 0.0230393, -0.0016445, 0.0016445
4: -0.0024453, -0.0021033, -0.0024453, -0.0021033, -0.0001251, 0.0001251
5: 0.0147989, 0.0151446, 0.0147989, 0.0151446, -0.0001264, 0.0001264
6: 0.0045405, 0.0047086, 0.0045405, 0.0047086, -0.0000615, 0.0000615
7: -0.0137491, -0.0125837, -0.0137491, -0.0125837, -0.0004262, 0.0004262
8: 0.0058213, 0.0067458, 0.0058213, 0.0067458, -0.0003381, 0.0003381
9: 0.0081947, 0.0098577, 0.0081947, 0.0098577, -0.0006081, 0.0006081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040975, -0.0040841, -0.0000058, 0.0000052
1: -0.0061191, -0.0056110, -0.0061072, -0.0056053, -0.0002162, 0.0001936
2: 0.9691203, 0.9697300, 0.9691346, 0.9697368, -0.0002594, 0.0002323
3: 0.0185424, 0.0230393, 0.0186478, 0.0230899, -0.0019135, 0.0017137
4: -0.0024453, -0.0021033, -0.0024492, -0.0021113, -0.0001303, 0.0001455
5: 0.0147989, 0.0151446, 0.0147950, 0.0151365, -0.0001317, 0.0001471
6: 0.0045405, 0.0047086, 0.0045444, 0.0047105, -0.0000715, 0.0000641
7: -0.0137491, -0.0125837, -0.0137622, -0.0126110, -0.0004441, 0.0004959
8: 0.0058213, 0.0067458, 0.0058109, 0.0067242, -0.0003524, 0.0003934
9: 0.0081947, 0.0098577, 0.0081760, 0.0098187, -0.0006337, 0.0007076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040978, -0.0040842, -0.0000052, 0.0000058
1: -0.0061072, -0.0056053, -0.0061191, -0.0056110, -0.0001936, 0.0002162
2: 0.9691346, 0.9697368, 0.9691203, 0.9697300, -0.0002323, 0.0002594
3: 0.0186478, 0.0230899, 0.0185424, 0.0230393, -0.0017137, 0.0019135
4: -0.0024492, -0.0021113, -0.0024453, -0.0021033, -0.0001455, 0.0001303
5: 0.0147950, 0.0151365, 0.0147989, 0.0151446, -0.0001471, 0.0001317
6: 0.0045444, 0.0047105, 0.0045405, 0.0047086, -0.0000641, 0.0000715
7: -0.0137622, -0.0126110, -0.0137491, -0.0125837, -0.0004959, 0.0004441
8: 0.0058109, 0.0067242, 0.0058213, 0.0067458, -0.0003934, 0.0003524
9: 0.0081760, 0.0098187, 0.0081947, 0.0098577, -0.0007076, 0.0006337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001826
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040975, -0.0040841, -0.0000049, 0.0000049
1: -0.0061072, -0.0056053, -0.0061072, -0.0056053, -0.0001825, 0.0001825
2: 0.9691346, 0.9697368, 0.9691346, 0.9697368, -0.0002190, 0.0002190
3: 0.0186478, 0.0230899, 0.0186478, 0.0230899, -0.0016154, 0.0016154
4: -0.0024492, -0.0021113, -0.0024492, -0.0021113, -0.0001229, 0.0001229
5: 0.0147950, 0.0151365, 0.0147950, 0.0151365, -0.0001242, 0.0001242
6: 0.0045444, 0.0047105, 0.0045444, 0.0047105, -0.0000604, 0.0000604
7: -0.0137622, -0.0126110, -0.0137622, -0.0126110, -0.0004186, 0.0004186
8: 0.0058109, 0.0067242, 0.0058109, 0.0067242, -0.0003321, 0.0003321
9: 0.0081760, 0.0098187, 0.0081760, 0.0098187, -0.0005974, 0.0005974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.37 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001826
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040983, -0.0040844, -0.0040977, -0.0040845, -0.0000058, 0.0000054
1: -0.0061372, -0.0056154, -0.0061151, -0.0056187, -0.0002182, 0.0002027
2: 0.9690986, 0.9697248, 0.9691251, 0.9697208, -0.0002618, 0.0002433
3: 0.0183822, 0.0230010, 0.0185775, 0.0229715, -0.0019311, 0.0017946
4: -0.0024424, -0.0020911, -0.0024401, -0.0021060, -0.0001365, 0.0001469
5: 0.0148019, 0.0151569, 0.0148041, 0.0151419, -0.0001379, 0.0001484
6: 0.0045345, 0.0047072, 0.0045418, 0.0047061, -0.0000722, 0.0000671
7: -0.0137392, -0.0125422, -0.0137315, -0.0125928, -0.0004651, 0.0005005
8: 0.0058291, 0.0067788, 0.0058352, 0.0067386, -0.0003690, 0.0003970
9: 0.0082089, 0.0099169, 0.0082198, 0.0098447, -0.0006636, 0.0007141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.53 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040983, -0.0040844, -0.0040974, -0.0040843, -0.0000066, 0.0000056
1: -0.0061372, -0.0056154, -0.0061043, -0.0056125, -0.0002478, 0.0002101
2: 0.9690986, 0.9697248, 0.9691381, 0.9697283, -0.0002974, 0.0002521
3: 0.0183822, 0.0230010, 0.0186735, 0.0230266, -0.0021938, 0.0018594
4: -0.0024424, -0.0020911, -0.0024443, -0.0021133, -0.0001414, 0.0001669
5: 0.0148019, 0.0151569, 0.0147999, 0.0151345, -0.0001429, 0.0001686
6: 0.0045345, 0.0047072, 0.0045454, 0.0047081, -0.0000820, 0.0000695
7: -0.0137392, -0.0125422, -0.0137458, -0.0126177, -0.0004819, 0.0005685
8: 0.0058291, 0.0067788, 0.0058239, 0.0067189, -0.0003823, 0.0004511
9: 0.0082089, 0.0099169, 0.0081994, 0.0098092, -0.0006876, 0.0008113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040977, -0.0040845, -0.0000055, 0.0000058
1: -0.0061172, -0.0056099, -0.0061151, -0.0056187, -0.0002072, 0.0002182
2: 0.9691226, 0.9697313, 0.9691251, 0.9697208, -0.0002487, 0.0002618
3: 0.0185592, 0.0230492, 0.0185775, 0.0229715, -0.0018341, 0.0019313
4: -0.0024461, -0.0021046, -0.0024401, -0.0021060, -0.0001469, 0.0001395
5: 0.0147982, 0.0151433, 0.0148041, 0.0151419, -0.0001485, 0.0001410
6: 0.0045411, 0.0047090, 0.0045418, 0.0047061, -0.0000686, 0.0000722
7: -0.0137517, -0.0125880, -0.0137315, -0.0125928, -0.0005005, 0.0004753
8: 0.0058192, 0.0067424, 0.0058352, 0.0067386, -0.0003971, 0.0003771
9: 0.0081911, 0.0098515, 0.0082198, 0.0098447, -0.0007142, 0.0006782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.27 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040974, -0.0040843, -0.0000055, 0.0000053
1: -0.0061172, -0.0056099, -0.0061043, -0.0056125, -0.0002065, 0.0001996
2: 0.9691226, 0.9697313, 0.9691381, 0.9697283, -0.0002479, 0.0002395
3: 0.0185592, 0.0230492, 0.0186735, 0.0230266, -0.0018282, 0.0017667
4: -0.0024461, -0.0021046, -0.0024443, -0.0021133, -0.0001344, 0.0001390
5: 0.0147982, 0.0151433, 0.0147999, 0.0151345, -0.0001358, 0.0001405
6: 0.0045411, 0.0047090, 0.0045454, 0.0047081, -0.0000684, 0.0000661
7: -0.0137517, -0.0125880, -0.0137458, -0.0126177, -0.0004579, 0.0004738
8: 0.0058192, 0.0067424, 0.0058239, 0.0067189, -0.0003632, 0.0003759
9: 0.0081911, 0.0098515, 0.0081994, 0.0098092, -0.0006533, 0.0006760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040983, -0.0040844, -0.0040978, -0.0040842, -0.0000057, 0.0000051
1: -0.0061372, -0.0056154, -0.0061191, -0.0056110, -0.0002131, 0.0001925
2: 0.9690986, 0.9697248, 0.9691203, 0.9697300, -0.0002557, 0.0002310
3: 0.0183822, 0.0230010, 0.0185424, 0.0230393, -0.0018860, 0.0017041
4: -0.0024424, -0.0020911, -0.0024453, -0.0021033, -0.0001296, 0.0001434
5: 0.0148019, 0.0151569, 0.0147989, 0.0151446, -0.0001310, 0.0001450
6: 0.0045345, 0.0047072, 0.0045405, 0.0047086, -0.0000705, 0.0000637
7: -0.0137392, -0.0125422, -0.0137491, -0.0125837, -0.0004416, 0.0004888
8: 0.0058291, 0.0067788, 0.0058213, 0.0067458, -0.0003504, 0.0003878
9: 0.0082089, 0.0099169, 0.0081947, 0.0098577, -0.0006302, 0.0006974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.19 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040983, -0.0040844, -0.0040975, -0.0040841, -0.0000065, 0.0000054
1: -0.0061372, -0.0056154, -0.0061072, -0.0056053, -0.0002435, 0.0002004
2: 0.9690986, 0.9697248, 0.9691346, 0.9697368, -0.0002922, 0.0002404
3: 0.0183822, 0.0230010, 0.0186478, 0.0230899, -0.0021550, 0.0017734
4: -0.0024424, -0.0020911, -0.0024492, -0.0021113, -0.0001349, 0.0001639
5: 0.0148019, 0.0151569, 0.0147950, 0.0151365, -0.0001363, 0.0001656
6: 0.0045345, 0.0047072, 0.0045444, 0.0047105, -0.0000806, 0.0000663
7: -0.0137392, -0.0125422, -0.0137622, -0.0126110, -0.0004596, 0.0005585
8: 0.0058291, 0.0067788, 0.0058109, 0.0067242, -0.0003646, 0.0004431
9: 0.0082089, 0.0099169, 0.0081760, 0.0098187, -0.0006558, 0.0007969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.13 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040978, -0.0040842, -0.0000054, 0.0000056
1: -0.0061172, -0.0056099, -0.0061191, -0.0056110, -0.0002016, 0.0002092
2: 0.9691226, 0.9697313, 0.9691203, 0.9697300, -0.0002419, 0.0002510
3: 0.0185592, 0.0230492, 0.0185424, 0.0230393, -0.0017844, 0.0018516
4: -0.0024461, -0.0021046, -0.0024453, -0.0021033, -0.0001408, 0.0001357
5: 0.0147982, 0.0151433, 0.0147989, 0.0151446, -0.0001423, 0.0001372
6: 0.0045411, 0.0047090, 0.0045405, 0.0047086, -0.0000667, 0.0000692
7: -0.0137517, -0.0125880, -0.0137491, -0.0125837, -0.0004799, 0.0004624
8: 0.0058192, 0.0067424, 0.0058213, 0.0067458, -0.0003807, 0.0003669
9: 0.0081911, 0.0098515, 0.0081947, 0.0098577, -0.0006847, 0.0006599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.24 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040975, -0.0040841, -0.0000053, 0.0000049
1: -0.0061172, -0.0056099, -0.0061072, -0.0056053, -0.0001978, 0.0001845
2: 0.9691226, 0.9697313, 0.9691346, 0.9697368, -0.0002373, 0.0002214
3: 0.0185592, 0.0230492, 0.0186478, 0.0230899, -0.0017506, 0.0016328
4: -0.0024461, -0.0021046, -0.0024492, -0.0021113, -0.0001242, 0.0001331
5: 0.0147982, 0.0151433, 0.0147950, 0.0151365, -0.0001255, 0.0001346
6: 0.0045411, 0.0047090, 0.0045444, 0.0047105, -0.0000655, 0.0000610
7: -0.0137517, -0.0125880, -0.0137622, -0.0126110, -0.0004232, 0.0004537
8: 0.0058192, 0.0067424, 0.0058109, 0.0067242, -0.0003357, 0.0003599
9: 0.0081911, 0.0098515, 0.0081760, 0.0098187, -0.0006038, 0.0006474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.20 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040982, -0.0040846, -0.0000053, 0.0000061
1: -0.0061191, -0.0056110, -0.0061337, -0.0056227, -0.0002003, 0.0002277
2: 0.9691203, 0.9697300, 0.9691027, 0.9697160, -0.0002404, 0.0002733
3: 0.0185424, 0.0230393, 0.0184131, 0.0229363, -0.0017729, 0.0020157
4: -0.0024453, -0.0021033, -0.0024375, -0.0020935, -0.0001533, 0.0001348
5: 0.0147989, 0.0151446, 0.0148069, 0.0151545, -0.0001549, 0.0001363
6: 0.0045405, 0.0047086, 0.0045356, 0.0047047, -0.0000663, 0.0000754
7: -0.0137491, -0.0125837, -0.0137224, -0.0125502, -0.0005224, 0.0004595
8: 0.0058213, 0.0067458, 0.0058425, 0.0067724, -0.0004144, 0.0003645
9: 0.0081947, 0.0098577, 0.0082328, 0.0099055, -0.0007454, 0.0006556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.15 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040977, -0.0040844, -0.0000058, 0.0000058
1: -0.0061191, -0.0056110, -0.0061137, -0.0056180, -0.0002166, 0.0002163
2: 0.9691203, 0.9697300, 0.9691268, 0.9697217, -0.0002599, 0.0002596
3: 0.0185424, 0.0230393, 0.0185903, 0.0229782, -0.0019173, 0.0019149
4: -0.0024453, -0.0021033, -0.0024407, -0.0021069, -0.0001456, 0.0001458
5: 0.0147989, 0.0151446, 0.0148036, 0.0151409, -0.0001472, 0.0001474
6: 0.0045405, 0.0047086, 0.0045422, 0.0047063, -0.0000717, 0.0000716
7: -0.0137491, -0.0125837, -0.0137332, -0.0125961, -0.0004963, 0.0004969
8: 0.0058213, 0.0067458, 0.0058338, 0.0067360, -0.0003937, 0.0003942
9: 0.0081947, 0.0098577, 0.0082173, 0.0098400, -0.0007081, 0.0007090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.15 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001837, upper bound: 0.0001829
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040982, -0.0040846, -0.0000055, 0.0000068
1: -0.0061072, -0.0056053, -0.0061337, -0.0056227, -0.0002068, 0.0002535
2: 0.9691346, 0.9697368, 0.9691027, 0.9697160, -0.0002482, 0.0003042
3: 0.0186478, 0.0230899, 0.0184131, 0.0229363, -0.0018309, 0.0022439
4: -0.0024492, -0.0021113, -0.0024375, -0.0020935, -0.0001707, 0.0001392
5: 0.0147950, 0.0151365, 0.0148069, 0.0151545, -0.0001725, 0.0001407
6: 0.0045444, 0.0047105, 0.0045356, 0.0047047, -0.0000685, 0.0000839
7: -0.0137622, -0.0126110, -0.0137224, -0.0125502, -0.0005815, 0.0004745
8: 0.0058109, 0.0067242, 0.0058425, 0.0067724, -0.0004614, 0.0003764
9: 0.0081760, 0.0098187, 0.0082328, 0.0099055, -0.0008298, 0.0006770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040977, -0.0040844, -0.0000052, 0.0000058
1: -0.0061072, -0.0056053, -0.0061137, -0.0056180, -0.0001946, 0.0002157
2: 0.9691346, 0.9697368, 0.9691268, 0.9697217, -0.0002336, 0.0002589
3: 0.0186478, 0.0230899, 0.0185903, 0.0229782, -0.0017228, 0.0019096
4: -0.0024492, -0.0021113, -0.0024407, -0.0021069, -0.0001452, 0.0001310
5: 0.0147950, 0.0151365, 0.0148036, 0.0151409, -0.0001468, 0.0001324
6: 0.0045444, 0.0047105, 0.0045422, 0.0047063, -0.0000644, 0.0000714
7: -0.0137622, -0.0126110, -0.0137332, -0.0125961, -0.0004949, 0.0004465
8: 0.0058109, 0.0067242, 0.0058338, 0.0067360, -0.0003926, 0.0003542
9: 0.0081760, 0.0098187, 0.0082173, 0.0098400, -0.0007062, 0.0006371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001837, upper bound: 0.0001829
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040983, -0.0040844, -0.0000051, 0.0000057
1: -0.0061191, -0.0056110, -0.0061372, -0.0056154, -0.0001925, 0.0002131
2: 0.9691203, 0.9697300, 0.9690986, 0.9697248, -0.0002310, 0.0002557
3: 0.0185424, 0.0230393, 0.0183822, 0.0230010, -0.0017041, 0.0018860
4: -0.0024453, -0.0021033, -0.0024424, -0.0020911, -0.0001434, 0.0001296
5: 0.0147989, 0.0151446, 0.0148019, 0.0151569, -0.0001450, 0.0001310
6: 0.0045405, 0.0047086, 0.0045345, 0.0047072, -0.0000637, 0.0000705
7: -0.0137491, -0.0125837, -0.0137392, -0.0125422, -0.0004888, 0.0004416
8: 0.0058213, 0.0067458, 0.0058291, 0.0067788, -0.0003878, 0.0003504
9: 0.0081947, 0.0098577, 0.0082089, 0.0099169, -0.0006974, 0.0006302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040978, -0.0040842, -0.0040978, -0.0040842, -0.0000056, 0.0000054
1: -0.0061191, -0.0056110, -0.0061172, -0.0056099, -0.0002092, 0.0002016
2: 0.9691203, 0.9697300, 0.9691226, 0.9697313, -0.0002510, 0.0002419
3: 0.0185424, 0.0230393, 0.0185592, 0.0230492, -0.0018516, 0.0017844
4: -0.0024453, -0.0021033, -0.0024461, -0.0021046, -0.0001357, 0.0001408
5: 0.0147989, 0.0151446, 0.0147982, 0.0151433, -0.0001372, 0.0001423
6: 0.0045405, 0.0047086, 0.0045411, 0.0047090, -0.0000692, 0.0000667
7: -0.0137491, -0.0125837, -0.0137517, -0.0125880, -0.0004624, 0.0004799
8: 0.0058213, 0.0067458, 0.0058192, 0.0067424, -0.0003669, 0.0003807
9: 0.0081947, 0.0098577, 0.0081911, 0.0098515, -0.0006599, 0.0006847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001829
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040983, -0.0040844, -0.0000054, 0.0000065
1: -0.0061072, -0.0056053, -0.0061372, -0.0056154, -0.0002004, 0.0002435
2: 0.9691346, 0.9697368, 0.9690986, 0.9697248, -0.0002404, 0.0002922
3: 0.0186478, 0.0230899, 0.0183822, 0.0230010, -0.0017734, 0.0021550
4: -0.0024492, -0.0021113, -0.0024424, -0.0020911, -0.0001639, 0.0001349
5: 0.0147950, 0.0151365, 0.0148019, 0.0151569, -0.0001656, 0.0001363
6: 0.0045444, 0.0047105, 0.0045345, 0.0047072, -0.0000663, 0.0000806
7: -0.0137622, -0.0126110, -0.0137392, -0.0125422, -0.0005585, 0.0004596
8: 0.0058109, 0.0067242, 0.0058291, 0.0067788, -0.0004431, 0.0003646
9: 0.0081760, 0.0098187, 0.0082089, 0.0099169, -0.0007969, 0.0006558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.12 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040975, -0.0040841, -0.0040978, -0.0040842, -0.0000049, 0.0000053
1: -0.0061072, -0.0056053, -0.0061172, -0.0056099, -0.0001845, 0.0001978
2: 0.9691346, 0.9697368, 0.9691226, 0.9697313, -0.0002214, 0.0002373
3: 0.0186478, 0.0230899, 0.0185592, 0.0230492, -0.0016328, 0.0017506
4: -0.0024492, -0.0021113, -0.0024461, -0.0021046, -0.0001331, 0.0001242
5: 0.0147950, 0.0151365, 0.0147982, 0.0151433, -0.0001346, 0.0001255
6: 0.0045444, 0.0047105, 0.0045411, 0.0047090, -0.0000610, 0.0000655
7: -0.0137622, -0.0126110, -0.0137517, -0.0125880, -0.0004537, 0.0004232
8: 0.0058109, 0.0067242, 0.0058192, 0.0067424, -0.0003599, 0.0003357
9: 0.0081760, 0.0098187, 0.0081911, 0.0098515, -0.0006474, 0.0006038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 157

Time for candidate selection: 2.13 seconds

### Candidate
type: A, layer: 3, pos: 79

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001829
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
time: 0.59 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.70 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001863, upper bound: 0.0001824
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001863, upper bound: 0.0001824
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001824
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001824
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001829
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001842
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001848, upper bound: 0.0001842
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001842
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001842
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001855
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001923, upper bound: 0.0001816
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001923, upper bound: 0.0001816
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001816
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001816
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001817
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001834
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001900, upper bound: 0.0001834
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001834
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001834
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001836
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001833, upper bound: 0.0001830
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001833, upper bound: 0.0001830
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001851
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001817, upper bound: 0.0001830
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001850
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001827, upper bound: 0.0001850
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001828, upper bound: 0.0001850
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001850
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001815, upper bound: 0.0001884
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001825
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001825
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001823
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001823
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001842, upper bound: 0.0001826
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001823
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001855, upper bound: 0.0001826
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001955, upper bound: 0.0001815
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001850, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001815
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001837, upper bound: 0.0001829
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001837, upper bound: 0.0001829
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001853, upper bound: 0.0001829
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001829
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001829
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001834, upper bound: 0.0001841
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001829
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.70
Output dim: 2, lower bound: -0.0001836, upper bound: 0.0001841

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040977, -0.0040845, -0.0000049, 0.0000049
1: -0.0061151, -0.0056187, -0.0061151, -0.0056187, -0.0001834, 0.0001834
2: 0.9691251, 0.9697208, 0.9691251, 0.9697208, -0.0002201, 0.0002201
3: 0.0185775, 0.0229715, 0.0185775, 0.0229715, -0.0016233, 0.0016233
4: -0.0024401, -0.0021060, -0.0024401, -0.0021060, -0.0001235, 0.0001235
5: 0.0148041, 0.0151419, 0.0148041, 0.0151419, -0.0001248, 0.0001248
6: 0.0045418, 0.0047061, 0.0045418, 0.0047061, -0.0000607, 0.0000607
7: -0.0137315, -0.0125928, -0.0137315, -0.0125928, -0.0004207, 0.0004207
8: 0.0058352, 0.0067386, 0.0058352, 0.0067386, -0.0003338, 0.0003338
9: 0.0082198, 0.0098447, 0.0082198, 0.0098447, -0.0006003, 0.0006003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.12 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001863
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040977, -0.0040845, -0.0000051, 0.0000057
1: -0.0061043, -0.0056125, -0.0061151, -0.0056187, -0.0001907, 0.0002131
2: 0.9691381, 0.9697283, 0.9691251, 0.9697208, -0.0002289, 0.0002557
3: 0.0186735, 0.0230266, 0.0185775, 0.0229715, -0.0016882, 0.0018860
4: -0.0024443, -0.0021133, -0.0024401, -0.0021060, -0.0001434, 0.0001284
5: 0.0147999, 0.0151345, 0.0148041, 0.0151419, -0.0001450, 0.0001298
6: 0.0045454, 0.0047081, 0.0045418, 0.0047061, -0.0000631, 0.0000705
7: -0.0137458, -0.0126177, -0.0137315, -0.0125928, -0.0004888, 0.0004375
8: 0.0058239, 0.0067189, 0.0058352, 0.0067386, -0.0003878, 0.0003471
9: 0.0081994, 0.0098092, 0.0082198, 0.0098447, -0.0006974, 0.0006243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.16 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001863
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040974, -0.0040843, -0.0000057, 0.0000051
1: -0.0061151, -0.0056187, -0.0061043, -0.0056125, -0.0002131, 0.0001907
2: 0.9691251, 0.9697208, 0.9691381, 0.9697283, -0.0002557, 0.0002289
3: 0.0185775, 0.0229715, 0.0186735, 0.0230266, -0.0018860, 0.0016882
4: -0.0024401, -0.0021060, -0.0024443, -0.0021133, -0.0001284, 0.0001434
5: 0.0148041, 0.0151419, 0.0147999, 0.0151345, -0.0001298, 0.0001450
6: 0.0045418, 0.0047061, 0.0045454, 0.0047081, -0.0000705, 0.0000631
7: -0.0137315, -0.0125928, -0.0137458, -0.0126177, -0.0004375, 0.0004888
8: 0.0058352, 0.0067386, 0.0058239, 0.0067189, -0.0003471, 0.0003878
9: 0.0082198, 0.0098447, 0.0081994, 0.0098092, -0.0006243, 0.0006974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040974, -0.0040843, -0.0000048, 0.0000048
1: -0.0061043, -0.0056125, -0.0061043, -0.0056125, -0.0001803, 0.0001803
2: 0.9691381, 0.9697283, 0.9691381, 0.9697283, -0.0002164, 0.0002164
3: 0.0186735, 0.0230266, 0.0186735, 0.0230266, -0.0015959, 0.0015959
4: -0.0024443, -0.0021133, -0.0024443, -0.0021133, -0.0001214, 0.0001214
5: 0.0147999, 0.0151345, 0.0147999, 0.0151345, -0.0001227, 0.0001227
6: 0.0045454, 0.0047081, 0.0045454, 0.0047081, -0.0000597, 0.0000597
7: -0.0137458, -0.0126177, -0.0137458, -0.0126177, -0.0004136, 0.0004136
8: 0.0058239, 0.0067189, 0.0058239, 0.0067189, -0.0003281, 0.0003281
9: 0.0081994, 0.0098092, 0.0081994, 0.0098092, -0.0005902, 0.0005902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.15 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040977, -0.0040845, -0.0000049, 0.0000049
1: -0.0061151, -0.0056187, -0.0061151, -0.0056187, -0.0001834, 0.0001834
2: 0.9691251, 0.9697208, 0.9691251, 0.9697208, -0.0002201, 0.0002201
3: 0.0185775, 0.0229715, 0.0185775, 0.0229715, -0.0016233, 0.0016233
4: -0.0024401, -0.0021060, -0.0024401, -0.0021060, -0.0001235, 0.0001235
5: 0.0148041, 0.0151419, 0.0148041, 0.0151419, -0.0001248, 0.0001248
6: 0.0045418, 0.0047061, 0.0045418, 0.0047061, -0.0000607, 0.0000607
7: -0.0137315, -0.0125928, -0.0137315, -0.0125928, -0.0004207, 0.0004207
8: 0.0058352, 0.0067386, 0.0058352, 0.0067386, -0.0003338, 0.0003338
9: 0.0082198, 0.0098447, 0.0082198, 0.0098447, -0.0006003, 0.0006003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.11 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001862
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040977, -0.0040845, -0.0000051, 0.0000057
1: -0.0061043, -0.0056125, -0.0061151, -0.0056187, -0.0001907, 0.0002131
2: 0.9691381, 0.9697283, 0.9691251, 0.9697208, -0.0002289, 0.0002557
3: 0.0186735, 0.0230266, 0.0185775, 0.0229715, -0.0016882, 0.0018860
4: -0.0024443, -0.0021133, -0.0024401, -0.0021060, -0.0001434, 0.0001284
5: 0.0147999, 0.0151345, 0.0148041, 0.0151419, -0.0001450, 0.0001298
6: 0.0045454, 0.0047081, 0.0045418, 0.0047061, -0.0000631, 0.0000705
7: -0.0137458, -0.0126177, -0.0137315, -0.0125928, -0.0004888, 0.0004375
8: 0.0058239, 0.0067189, 0.0058352, 0.0067386, -0.0003878, 0.0003471
9: 0.0081994, 0.0098092, 0.0082198, 0.0098447, -0.0006974, 0.0006243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001862
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040974, -0.0040843, -0.0000057, 0.0000051
1: -0.0061151, -0.0056187, -0.0061043, -0.0056125, -0.0002131, 0.0001907
2: 0.9691251, 0.9697208, 0.9691381, 0.9697283, -0.0002557, 0.0002289
3: 0.0185775, 0.0229715, 0.0186735, 0.0230266, -0.0018860, 0.0016882
4: -0.0024401, -0.0021060, -0.0024443, -0.0021133, -0.0001284, 0.0001434
5: 0.0148041, 0.0151419, 0.0147999, 0.0151345, -0.0001298, 0.0001450
6: 0.0045418, 0.0047061, 0.0045454, 0.0047081, -0.0000705, 0.0000631
7: -0.0137315, -0.0125928, -0.0137458, -0.0126177, -0.0004375, 0.0004888
8: 0.0058352, 0.0067386, 0.0058239, 0.0067189, -0.0003471, 0.0003878
9: 0.0082198, 0.0098447, 0.0081994, 0.0098092, -0.0006243, 0.0006974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.20 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001824
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040974, -0.0040843, -0.0000048, 0.0000048
1: -0.0061043, -0.0056125, -0.0061043, -0.0056125, -0.0001803, 0.0001803
2: 0.9691381, 0.9697283, 0.9691381, 0.9697283, -0.0002164, 0.0002164
3: 0.0186735, 0.0230266, 0.0186735, 0.0230266, -0.0015959, 0.0015959
4: -0.0024443, -0.0021133, -0.0024443, -0.0021133, -0.0001214, 0.0001214
5: 0.0147999, 0.0151345, 0.0147999, 0.0151345, -0.0001227, 0.0001227
6: 0.0045454, 0.0047081, 0.0045454, 0.0047081, -0.0000597, 0.0000597
7: -0.0137458, -0.0126177, -0.0137458, -0.0126177, -0.0004136, 0.0004136
8: 0.0058239, 0.0067189, 0.0058239, 0.0067189, -0.0003281, 0.0003281
9: 0.0081994, 0.0098092, 0.0081994, 0.0098092, -0.0005902, 0.0005902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001824, upper bound: 0.0001829
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001829
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040978, -0.0040842, -0.0000054, 0.0000052
1: -0.0061151, -0.0056187, -0.0061191, -0.0056110, -0.0002005, 0.0001936
2: 0.9691251, 0.9697208, 0.9691203, 0.9697300, -0.0002406, 0.0002323
3: 0.0185775, 0.0229715, 0.0185424, 0.0230393, -0.0017749, 0.0017135
4: -0.0024401, -0.0021060, -0.0024453, -0.0021033, -0.0001303, 0.0001350
5: 0.0148041, 0.0151419, 0.0147989, 0.0151446, -0.0001317, 0.0001364
6: 0.0045418, 0.0047061, 0.0045405, 0.0047086, -0.0000664, 0.0000641
7: -0.0137315, -0.0125928, -0.0137491, -0.0125837, -0.0004441, 0.0004600
8: 0.0058352, 0.0067386, 0.0058213, 0.0067458, -0.0003523, 0.0003649
9: 0.0082198, 0.0098447, 0.0081947, 0.0098577, -0.0006337, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.19 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001897
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040978, -0.0040842, -0.0000056, 0.0000060
1: -0.0061043, -0.0056125, -0.0061191, -0.0056110, -0.0002079, 0.0002233
2: 0.9691381, 0.9697283, 0.9691203, 0.9697300, -0.0002494, 0.0002679
3: 0.0186735, 0.0230266, 0.0185424, 0.0230393, -0.0018398, 0.0019762
4: -0.0024443, -0.0021133, -0.0024453, -0.0021033, -0.0001503, 0.0001399
5: 0.0147999, 0.0151345, 0.0147989, 0.0151446, -0.0001519, 0.0001414
6: 0.0045454, 0.0047081, 0.0045405, 0.0047086, -0.0000688, 0.0000739
7: -0.0137458, -0.0126177, -0.0137491, -0.0125837, -0.0005121, 0.0004768
8: 0.0058239, 0.0067189, 0.0058213, 0.0067458, -0.0004063, 0.0003783
9: 0.0081994, 0.0098092, 0.0081947, 0.0098577, -0.0007308, 0.0006804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.16 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001897
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040975, -0.0040841, -0.0000060, 0.0000053
1: -0.0061151, -0.0056187, -0.0061072, -0.0056053, -0.0002263, 0.0002001
2: 0.9691251, 0.9697208, 0.9691346, 0.9697368, -0.0002716, 0.0002402
3: 0.0185775, 0.0229715, 0.0186478, 0.0230899, -0.0020032, 0.0017714
4: -0.0024401, -0.0021060, -0.0024492, -0.0021113, -0.0001347, 0.0001524
5: 0.0148041, 0.0151419, 0.0147950, 0.0151365, -0.0001362, 0.0001540
6: 0.0045418, 0.0047061, 0.0045444, 0.0047105, -0.0000749, 0.0000662
7: -0.0137315, -0.0125928, -0.0137622, -0.0126110, -0.0004591, 0.0005191
8: 0.0058352, 0.0067386, 0.0058109, 0.0067242, -0.0003642, 0.0004119
9: 0.0082198, 0.0098447, 0.0081760, 0.0098187, -0.0006551, 0.0007408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040975, -0.0040841, -0.0000054, 0.0000051
1: -0.0061043, -0.0056125, -0.0061072, -0.0056053, -0.0002005, 0.0001927
2: 0.9691381, 0.9697283, 0.9691346, 0.9697368, -0.0002406, 0.0002313
3: 0.0186735, 0.0230266, 0.0186478, 0.0230899, -0.0017750, 0.0017057
4: -0.0024443, -0.0021133, -0.0024492, -0.0021113, -0.0001297, 0.0001350
5: 0.0147999, 0.0151345, 0.0147950, 0.0151365, -0.0001311, 0.0001364
6: 0.0045454, 0.0047081, 0.0045444, 0.0047105, -0.0000664, 0.0000638
7: -0.0137458, -0.0126177, -0.0137622, -0.0126110, -0.0004421, 0.0004600
8: 0.0058239, 0.0067189, 0.0058109, 0.0067242, -0.0003507, 0.0003649
9: 0.0081994, 0.0098092, 0.0081760, 0.0098187, -0.0006308, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040978, -0.0040842, -0.0000054, 0.0000052
1: -0.0061151, -0.0056187, -0.0061191, -0.0056110, -0.0002005, 0.0001936
2: 0.9691251, 0.9697208, 0.9691203, 0.9697300, -0.0002406, 0.0002323
3: 0.0185775, 0.0229715, 0.0185424, 0.0230393, -0.0017749, 0.0017135
4: -0.0024401, -0.0021060, -0.0024453, -0.0021033, -0.0001303, 0.0001350
5: 0.0148041, 0.0151419, 0.0147989, 0.0151446, -0.0001317, 0.0001364
6: 0.0045418, 0.0047061, 0.0045405, 0.0047086, -0.0000664, 0.0000641
7: -0.0137315, -0.0125928, -0.0137491, -0.0125837, -0.0004441, 0.0004600
8: 0.0058352, 0.0067386, 0.0058213, 0.0067458, -0.0003523, 0.0003649
9: 0.0082198, 0.0098447, 0.0081947, 0.0098577, -0.0006337, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.13 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001897
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040978, -0.0040842, -0.0000056, 0.0000060
1: -0.0061043, -0.0056125, -0.0061191, -0.0056110, -0.0002079, 0.0002233
2: 0.9691381, 0.9697283, 0.9691203, 0.9697300, -0.0002494, 0.0002679
3: 0.0186735, 0.0230266, 0.0185424, 0.0230393, -0.0018398, 0.0019762
4: -0.0024443, -0.0021133, -0.0024453, -0.0021033, -0.0001503, 0.0001399
5: 0.0147999, 0.0151345, 0.0147989, 0.0151446, -0.0001519, 0.0001414
6: 0.0045454, 0.0047081, 0.0045405, 0.0047086, -0.0000688, 0.0000739
7: -0.0137458, -0.0126177, -0.0137491, -0.0125837, -0.0005121, 0.0004768
8: 0.0058239, 0.0067189, 0.0058213, 0.0067458, -0.0004063, 0.0003783
9: 0.0081994, 0.0098092, 0.0081947, 0.0098577, -0.0007308, 0.0006804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.15 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001897
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040975, -0.0040841, -0.0000060, 0.0000053
1: -0.0061151, -0.0056187, -0.0061072, -0.0056053, -0.0002263, 0.0002001
2: 0.9691251, 0.9697208, 0.9691346, 0.9697368, -0.0002716, 0.0002402
3: 0.0185775, 0.0229715, 0.0186478, 0.0230899, -0.0020032, 0.0017714
4: -0.0024401, -0.0021060, -0.0024492, -0.0021113, -0.0001347, 0.0001524
5: 0.0148041, 0.0151419, 0.0147950, 0.0151365, -0.0001362, 0.0001540
6: 0.0045418, 0.0047061, 0.0045444, 0.0047105, -0.0000749, 0.0000662
7: -0.0137315, -0.0125928, -0.0137622, -0.0126110, -0.0004591, 0.0005191
8: 0.0058352, 0.0067386, 0.0058109, 0.0067242, -0.0003642, 0.0004119
9: 0.0082198, 0.0098447, 0.0081760, 0.0098187, -0.0006551, 0.0007408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001842
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040975, -0.0040841, -0.0000054, 0.0000051
1: -0.0061043, -0.0056125, -0.0061072, -0.0056053, -0.0002005, 0.0001927
2: 0.9691381, 0.9697283, 0.9691346, 0.9697368, -0.0002406, 0.0002313
3: 0.0186735, 0.0230266, 0.0186478, 0.0230899, -0.0017750, 0.0017057
4: -0.0024443, -0.0021133, -0.0024492, -0.0021113, -0.0001297, 0.0001350
5: 0.0147999, 0.0151345, 0.0147950, 0.0151365, -0.0001311, 0.0001364
6: 0.0045454, 0.0047081, 0.0045444, 0.0047105, -0.0000664, 0.0000638
7: -0.0137458, -0.0126177, -0.0137622, -0.0126110, -0.0004421, 0.0004600
8: 0.0058239, 0.0067189, 0.0058109, 0.0067242, -0.0003507, 0.0003649
9: 0.0081994, 0.0098092, 0.0081760, 0.0098187, -0.0006308, 0.0006564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.23 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001823, upper bound: 0.0001855
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001825, upper bound: 0.0001855
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040977, -0.0040845, -0.0000056, 0.0000051
1: -0.0061337, -0.0056227, -0.0061151, -0.0056187, -0.0002106, 0.0001901
2: 0.9691027, 0.9697160, 0.9691251, 0.9697208, -0.0002527, 0.0002281
3: 0.0184131, 0.0229363, 0.0185775, 0.0229715, -0.0018641, 0.0016828
4: -0.0024375, -0.0020935, -0.0024401, -0.0021060, -0.0001280, 0.0001418
5: 0.0148069, 0.0151545, 0.0148041, 0.0151419, -0.0001293, 0.0001433
6: 0.0045356, 0.0047047, 0.0045418, 0.0047061, -0.0000697, 0.0000629
7: -0.0137224, -0.0125502, -0.0137315, -0.0125928, -0.0004361, 0.0004831
8: 0.0058425, 0.0067724, 0.0058352, 0.0067386, -0.0003460, 0.0003833
9: 0.0082328, 0.0099055, 0.0082198, 0.0098447, -0.0006223, 0.0006893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.20 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001833
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040977, -0.0040845, -0.0000053, 0.0000055
1: -0.0061137, -0.0056180, -0.0061151, -0.0056187, -0.0001992, 0.0002064
2: 0.9691268, 0.9697217, 0.9691251, 0.9697208, -0.0002391, 0.0002477
3: 0.0185903, 0.0229782, 0.0185775, 0.0229715, -0.0017633, 0.0018271
4: -0.0024407, -0.0021069, -0.0024401, -0.0021060, -0.0001390, 0.0001341
5: 0.0148036, 0.0151409, 0.0148041, 0.0151419, -0.0001404, 0.0001355
6: 0.0045422, 0.0047063, 0.0045418, 0.0047061, -0.0000659, 0.0000683
7: -0.0137332, -0.0125961, -0.0137315, -0.0125928, -0.0004735, 0.0004570
8: 0.0058338, 0.0067360, 0.0058352, 0.0067386, -0.0003757, 0.0003625
9: 0.0082173, 0.0098400, 0.0082198, 0.0098447, -0.0006757, 0.0006521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.21 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001833
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040974, -0.0040843, -0.0000064, 0.0000053
1: -0.0061337, -0.0056227, -0.0061043, -0.0056125, -0.0002403, 0.0001974
2: 0.9691027, 0.9697160, 0.9691381, 0.9697283, -0.0002883, 0.0002369
3: 0.0184131, 0.0229363, 0.0186735, 0.0230266, -0.0021267, 0.0017476
4: -0.0024375, -0.0020935, -0.0024443, -0.0021133, -0.0001329, 0.0001618
5: 0.0148069, 0.0151545, 0.0147999, 0.0151345, -0.0001343, 0.0001635
6: 0.0045356, 0.0047047, 0.0045454, 0.0047081, -0.0000795, 0.0000653
7: -0.0137224, -0.0125502, -0.0137458, -0.0126177, -0.0004529, 0.0005512
8: 0.0058425, 0.0067724, 0.0058239, 0.0067189, -0.0003593, 0.0004373
9: 0.0082328, 0.0099055, 0.0081994, 0.0098092, -0.0006463, 0.0007865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.26 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040974, -0.0040843, -0.0000052, 0.0000049
1: -0.0061137, -0.0056180, -0.0061043, -0.0056125, -0.0001955, 0.0001822
2: 0.9691268, 0.9697217, 0.9691381, 0.9697283, -0.0002346, 0.0002187
3: 0.0185903, 0.0229782, 0.0186735, 0.0230266, -0.0017306, 0.0016130
4: -0.0024407, -0.0021069, -0.0024443, -0.0021133, -0.0001227, 0.0001316
5: 0.0148036, 0.0151409, 0.0147999, 0.0151345, -0.0001240, 0.0001330
6: 0.0045422, 0.0047063, 0.0045454, 0.0047081, -0.0000647, 0.0000603
7: -0.0137332, -0.0125961, -0.0137458, -0.0126177, -0.0004180, 0.0004485
8: 0.0058338, 0.0067360, 0.0058239, 0.0067189, -0.0003316, 0.0003558
9: 0.0082173, 0.0098400, 0.0081994, 0.0098092, -0.0005965, 0.0006400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.26 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040977, -0.0040845, -0.0000056, 0.0000051
1: -0.0061337, -0.0056227, -0.0061151, -0.0056187, -0.0002106, 0.0001901
2: 0.9691027, 0.9697160, 0.9691251, 0.9697208, -0.0002527, 0.0002281
3: 0.0184131, 0.0229363, 0.0185775, 0.0229715, -0.0018641, 0.0016828
4: -0.0024375, -0.0020935, -0.0024401, -0.0021060, -0.0001280, 0.0001418
5: 0.0148069, 0.0151545, 0.0148041, 0.0151419, -0.0001293, 0.0001433
6: 0.0045356, 0.0047047, 0.0045418, 0.0047061, -0.0000697, 0.0000629
7: -0.0137224, -0.0125502, -0.0137315, -0.0125928, -0.0004361, 0.0004831
8: 0.0058425, 0.0067724, 0.0058352, 0.0067386, -0.0003460, 0.0003833
9: 0.0082328, 0.0099055, 0.0082198, 0.0098447, -0.0006223, 0.0006893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.13 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001833
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040977, -0.0040845, -0.0000053, 0.0000055
1: -0.0061137, -0.0056180, -0.0061151, -0.0056187, -0.0001992, 0.0002064
2: 0.9691268, 0.9697217, 0.9691251, 0.9697208, -0.0002391, 0.0002477
3: 0.0185903, 0.0229782, 0.0185775, 0.0229715, -0.0017633, 0.0018271
4: -0.0024407, -0.0021069, -0.0024401, -0.0021060, -0.0001390, 0.0001341
5: 0.0148036, 0.0151409, 0.0148041, 0.0151419, -0.0001404, 0.0001355
6: 0.0045422, 0.0047063, 0.0045418, 0.0047061, -0.0000659, 0.0000683
7: -0.0137332, -0.0125961, -0.0137315, -0.0125928, -0.0004735, 0.0004570
8: 0.0058338, 0.0067360, 0.0058352, 0.0067386, -0.0003757, 0.0003625
9: 0.0082173, 0.0098400, 0.0082198, 0.0098447, -0.0006757, 0.0006521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001833
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040974, -0.0040843, -0.0000064, 0.0000053
1: -0.0061337, -0.0056227, -0.0061043, -0.0056125, -0.0002403, 0.0001974
2: 0.9691027, 0.9697160, 0.9691381, 0.9697283, -0.0002883, 0.0002369
3: 0.0184131, 0.0229363, 0.0186735, 0.0230266, -0.0021267, 0.0017476
4: -0.0024375, -0.0020935, -0.0024443, -0.0021133, -0.0001329, 0.0001618
5: 0.0148069, 0.0151545, 0.0147999, 0.0151345, -0.0001343, 0.0001635
6: 0.0045356, 0.0047047, 0.0045454, 0.0047081, -0.0000795, 0.0000653
7: -0.0137224, -0.0125502, -0.0137458, -0.0126177, -0.0004529, 0.0005512
8: 0.0058425, 0.0067724, 0.0058239, 0.0067189, -0.0003593, 0.0004373
9: 0.0082328, 0.0099055, 0.0081994, 0.0098092, -0.0006463, 0.0007865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001816
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040974, -0.0040843, -0.0000052, 0.0000049
1: -0.0061137, -0.0056180, -0.0061043, -0.0056125, -0.0001955, 0.0001822
2: 0.9691268, 0.9697217, 0.9691381, 0.9697283, -0.0002346, 0.0002187
3: 0.0185903, 0.0229782, 0.0186735, 0.0230266, -0.0017306, 0.0016130
4: -0.0024407, -0.0021069, -0.0024443, -0.0021133, -0.0001227, 0.0001316
5: 0.0148036, 0.0151409, 0.0147999, 0.0151345, -0.0001240, 0.0001330
6: 0.0045422, 0.0047063, 0.0045454, 0.0047081, -0.0000647, 0.0000603
7: -0.0137332, -0.0125961, -0.0137458, -0.0126177, -0.0004180, 0.0004485
8: 0.0058338, 0.0067360, 0.0058239, 0.0067189, -0.0003316, 0.0003558
9: 0.0082173, 0.0098400, 0.0081994, 0.0098092, -0.0005965, 0.0006400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.21 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001830, upper bound: 0.0001817
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001851, upper bound: 0.0001817
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040978, -0.0040842, -0.0000061, 0.0000053
1: -0.0061337, -0.0056227, -0.0061191, -0.0056110, -0.0002277, 0.0002003
2: 0.9691027, 0.9697160, 0.9691203, 0.9697300, -0.0002733, 0.0002404
3: 0.0184131, 0.0229363, 0.0185424, 0.0230393, -0.0020157, 0.0017729
4: -0.0024375, -0.0020935, -0.0024453, -0.0021033, -0.0001348, 0.0001533
5: 0.0148069, 0.0151545, 0.0147989, 0.0151446, -0.0001363, 0.0001549
6: 0.0045356, 0.0047047, 0.0045405, 0.0047086, -0.0000754, 0.0000663
7: -0.0137224, -0.0125502, -0.0137491, -0.0125837, -0.0004595, 0.0005224
8: 0.0058425, 0.0067724, 0.0058213, 0.0067458, -0.0003645, 0.0004144
9: 0.0082328, 0.0099055, 0.0081947, 0.0098577, -0.0006556, 0.0007454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001853
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040978, -0.0040842, -0.0000058, 0.0000058
1: -0.0061137, -0.0056180, -0.0061191, -0.0056110, -0.0002163, 0.0002166
2: 0.9691268, 0.9697217, 0.9691203, 0.9697300, -0.0002596, 0.0002599
3: 0.0185903, 0.0229782, 0.0185424, 0.0230393, -0.0019149, 0.0019173
4: -0.0024407, -0.0021069, -0.0024453, -0.0021033, -0.0001458, 0.0001456
5: 0.0148036, 0.0151409, 0.0147989, 0.0151446, -0.0001474, 0.0001472
6: 0.0045422, 0.0047063, 0.0045405, 0.0047086, -0.0000716, 0.0000717
7: -0.0137332, -0.0125961, -0.0137491, -0.0125837, -0.0004969, 0.0004963
8: 0.0058338, 0.0067360, 0.0058213, 0.0067458, -0.0003942, 0.0003937
9: 0.0082173, 0.0098400, 0.0081947, 0.0098577, -0.0007090, 0.0007081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.11 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001853
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040975, -0.0040841, -0.0000068, 0.0000055
1: -0.0061337, -0.0056227, -0.0061072, -0.0056053, -0.0002535, 0.0002068
2: 0.9691027, 0.9697160, 0.9691346, 0.9697368, -0.0003042, 0.0002482
3: 0.0184131, 0.0229363, 0.0186478, 0.0230899, -0.0022439, 0.0018309
4: -0.0024375, -0.0020935, -0.0024492, -0.0021113, -0.0001392, 0.0001707
5: 0.0148069, 0.0151545, 0.0147950, 0.0151365, -0.0001407, 0.0001725
6: 0.0045356, 0.0047047, 0.0045444, 0.0047105, -0.0000839, 0.0000685
7: -0.0137224, -0.0125502, -0.0137622, -0.0126110, -0.0004745, 0.0005815
8: 0.0058425, 0.0067724, 0.0058109, 0.0067242, -0.0003764, 0.0004614
9: 0.0082328, 0.0099055, 0.0081760, 0.0098187, -0.0006770, 0.0008298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.11 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040975, -0.0040841, -0.0000058, 0.0000052
1: -0.0061137, -0.0056180, -0.0061072, -0.0056053, -0.0002157, 0.0001946
2: 0.9691268, 0.9697217, 0.9691346, 0.9697368, -0.0002589, 0.0002336
3: 0.0185903, 0.0229782, 0.0186478, 0.0230899, -0.0019096, 0.0017228
4: -0.0024407, -0.0021069, -0.0024492, -0.0021113, -0.0001310, 0.0001452
5: 0.0148036, 0.0151409, 0.0147950, 0.0151365, -0.0001324, 0.0001468
6: 0.0045422, 0.0047063, 0.0045444, 0.0047105, -0.0000714, 0.0000644
7: -0.0137332, -0.0125961, -0.0137622, -0.0126110, -0.0004465, 0.0004949
8: 0.0058338, 0.0067360, 0.0058109, 0.0067242, -0.0003542, 0.0003926
9: 0.0082173, 0.0098400, 0.0081760, 0.0098187, -0.0006371, 0.0007062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.17 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040978, -0.0040842, -0.0000061, 0.0000053
1: -0.0061337, -0.0056227, -0.0061191, -0.0056110, -0.0002277, 0.0002003
2: 0.9691027, 0.9697160, 0.9691203, 0.9697300, -0.0002733, 0.0002404
3: 0.0184131, 0.0229363, 0.0185424, 0.0230393, -0.0020157, 0.0017729
4: -0.0024375, -0.0020935, -0.0024453, -0.0021033, -0.0001348, 0.0001533
5: 0.0148069, 0.0151545, 0.0147989, 0.0151446, -0.0001363, 0.0001549
6: 0.0045356, 0.0047047, 0.0045405, 0.0047086, -0.0000754, 0.0000663
7: -0.0137224, -0.0125502, -0.0137491, -0.0125837, -0.0004595, 0.0005224
8: 0.0058425, 0.0067724, 0.0058213, 0.0067458, -0.0003645, 0.0004144
9: 0.0082328, 0.0099055, 0.0081947, 0.0098577, -0.0006556, 0.0007454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.17 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001853
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040978, -0.0040842, -0.0000058, 0.0000058
1: -0.0061137, -0.0056180, -0.0061191, -0.0056110, -0.0002163, 0.0002166
2: 0.9691268, 0.9697217, 0.9691203, 0.9697300, -0.0002596, 0.0002599
3: 0.0185903, 0.0229782, 0.0185424, 0.0230393, -0.0019149, 0.0019173
4: -0.0024407, -0.0021069, -0.0024453, -0.0021033, -0.0001458, 0.0001456
5: 0.0148036, 0.0151409, 0.0147989, 0.0151446, -0.0001474, 0.0001472
6: 0.0045422, 0.0047063, 0.0045405, 0.0047086, -0.0000716, 0.0000717
7: -0.0137332, -0.0125961, -0.0137491, -0.0125837, -0.0004969, 0.0004963
8: 0.0058338, 0.0067360, 0.0058213, 0.0067458, -0.0003942, 0.0003937
9: 0.0082173, 0.0098400, 0.0081947, 0.0098577, -0.0007090, 0.0007081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.26 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001853
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001836
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040982, -0.0040846, -0.0040975, -0.0040841, -0.0000068, 0.0000055
1: -0.0061337, -0.0056227, -0.0061072, -0.0056053, -0.0002535, 0.0002068
2: 0.9691027, 0.9697160, 0.9691346, 0.9697368, -0.0003042, 0.0002482
3: 0.0184131, 0.0229363, 0.0186478, 0.0230899, -0.0022439, 0.0018309
4: -0.0024375, -0.0020935, -0.0024492, -0.0021113, -0.0001392, 0.0001707
5: 0.0148069, 0.0151545, 0.0147950, 0.0151365, -0.0001407, 0.0001725
6: 0.0045356, 0.0047047, 0.0045444, 0.0047105, -0.0000839, 0.0000685
7: -0.0137224, -0.0125502, -0.0137622, -0.0126110, -0.0004745, 0.0005815
8: 0.0058425, 0.0067724, 0.0058109, 0.0067242, -0.0003764, 0.0004614
9: 0.0082328, 0.0099055, 0.0081760, 0.0098187, -0.0006770, 0.0008298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001834
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040844, -0.0040975, -0.0040841, -0.0000058, 0.0000052
1: -0.0061137, -0.0056180, -0.0061072, -0.0056053, -0.0002157, 0.0001946
2: 0.9691268, 0.9697217, 0.9691346, 0.9697368, -0.0002589, 0.0002336
3: 0.0185903, 0.0229782, 0.0186478, 0.0230899, -0.0019096, 0.0017228
4: -0.0024407, -0.0021069, -0.0024492, -0.0021113, -0.0001310, 0.0001452
5: 0.0148036, 0.0151409, 0.0147950, 0.0151365, -0.0001324, 0.0001468
6: 0.0045422, 0.0047063, 0.0045444, 0.0047105, -0.0000714, 0.0000644
7: -0.0137332, -0.0125961, -0.0137622, -0.0126110, -0.0004465, 0.0004949
8: 0.0058338, 0.0067360, 0.0058109, 0.0067242, -0.0003542, 0.0003926
9: 0.0082173, 0.0098400, 0.0081760, 0.0098187, -0.0006371, 0.0007062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.20 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001829, upper bound: 0.0001837
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001841, upper bound: 0.0001837
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040982, -0.0040846, -0.0000051, 0.0000056
1: -0.0061151, -0.0056187, -0.0061337, -0.0056227, -0.0001901, 0.0002106
2: 0.9691251, 0.9697208, 0.9691027, 0.9697160, -0.0002281, 0.0002527
3: 0.0185775, 0.0229715, 0.0184131, 0.0229363, -0.0016828, 0.0018641
4: -0.0024401, -0.0021060, -0.0024375, -0.0020935, -0.0001418, 0.0001280
5: 0.0148041, 0.0151419, 0.0148069, 0.0151545, -0.0001433, 0.0001293
6: 0.0045418, 0.0047061, 0.0045356, 0.0047047, -0.0000629, 0.0000697
7: -0.0137315, -0.0125928, -0.0137224, -0.0125502, -0.0004831, 0.0004361
8: 0.0058352, 0.0067386, 0.0058425, 0.0067724, -0.0003833, 0.0003460
9: 0.0082198, 0.0098447, 0.0082328, 0.0099055, -0.0006893, 0.0006223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001924
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040982, -0.0040846, -0.0000053, 0.0000064
1: -0.0061043, -0.0056125, -0.0061337, -0.0056227, -0.0001974, 0.0002403
2: 0.9691381, 0.9697283, 0.9691027, 0.9697160, -0.0002369, 0.0002883
3: 0.0186735, 0.0230266, 0.0184131, 0.0229363, -0.0017476, 0.0021267
4: -0.0024443, -0.0021133, -0.0024375, -0.0020935, -0.0001618, 0.0001329
5: 0.0147999, 0.0151345, 0.0148069, 0.0151545, -0.0001635, 0.0001343
6: 0.0045454, 0.0047081, 0.0045356, 0.0047047, -0.0000653, 0.0000795
7: -0.0137458, -0.0126177, -0.0137224, -0.0125502, -0.0005512, 0.0004529
8: 0.0058239, 0.0067189, 0.0058425, 0.0067724, -0.0004373, 0.0003593
9: 0.0081994, 0.0098092, 0.0082328, 0.0099055, -0.0007865, 0.0006463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001924
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040977, -0.0040845, -0.0040977, -0.0040844, -0.0000055, 0.0000053
1: -0.0061151, -0.0056187, -0.0061137, -0.0056180, -0.0002064, 0.0001992
2: 0.9691251, 0.9697208, 0.9691268, 0.9697217, -0.0002477, 0.0002391
3: 0.0185775, 0.0229715, 0.0185903, 0.0229782, -0.0018271, 0.0017633
4: -0.0024401, -0.0021060, -0.0024407, -0.0021069, -0.0001341, 0.0001390
5: 0.0148041, 0.0151419, 0.0148036, 0.0151409, -0.0001355, 0.0001404
6: 0.0045418, 0.0047061, 0.0045422, 0.0047063, -0.0000683, 0.0000659
7: -0.0137315, -0.0125928, -0.0137332, -0.0125961, -0.0004570, 0.0004735
8: 0.0058352, 0.0067386, 0.0058338, 0.0067360, -0.0003625, 0.0003757
9: 0.0082198, 0.0098447, 0.0082173, 0.0098400, -0.0006521, 0.0006757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 157

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 79

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001816, upper bound: 0.0001830
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040974, -0.0040843, -0.0040977, -0.0040844, -0.0000049, 0.0000052
1: -0.0061043, -0.0056125, -0.0061137, -0.0056180, -0.0001822, 0.0001955
2: 0.9691381, 0.9697283, 0.9691268, 0.9697217, -0.0002187, 0.0002346
3: 0.0186735, 0.0230266, 0.0185903, 0.0229782, -0.0016130, 0.0017306
4: -0.0024443, -0.0021133, -0.0024407, -0.0021069, -0.0001316, 0.0001227
5: 0.0147999, 0.0151345, 0.0148036, 0.0151409, -0.0001330, 0.0001240
6: 0.0045454, 0.0047081, 0.0045422, 0.0047063, -0.0000603, 0.0000647
7: -0.0137458, -0.0126177, -0.0137332, -0.0125961, -0.0004485, 0.0004180
8: 0.0058239, 0.0067189, 0.0058338, 0.0067360, -0.0003558, 0.0003316
9: 0.0081994, 0.0098092, 0.0082173, 0.0098400, -0.0006400, 0.0005965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.58 + 597.91 = 600.49 seconds
