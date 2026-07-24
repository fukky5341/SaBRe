## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00164889


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0029833, 0.0051926, 0.0029833, 0.0051926, -0.0017495, 0.0017495)
1: (0.0017533, 0.0020725, 0.0017533, 0.0020725, -0.0002527, 0.0002527)
2: (0.0114890, 0.0127105, 0.0114890, 0.0127105, -0.0009672, 0.0009672)
3: (-0.0027980, -0.0015347, -0.0027980, -0.0015347, -0.0010004, 0.0010004)
4: (-0.0023756, -0.0010080, -0.0023756, -0.0010080, -0.0010829, 0.0010829)
5: (0.0050673, 0.0063614, 0.0050673, 0.0063614, -0.0010248, 0.0010248)
6: (-0.0021949, 0.0029400, -0.0021949, 0.0029400, -0.0040662, 0.0040662)
7: (-0.0065607, 0.0004326, -0.0065607, 0.0004326, -0.0055378, 0.0055378)
8: (0.9845924, 0.9895186, 0.9845924, 0.9895186, -0.0039010, 0.0039010)
9: (-0.0063730, -0.0019012, -0.0063730, -0.0019012, -0.0035410, 0.0035410)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.63 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0019904, upper bound: 0.0019904

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017486, upper bound: 0.0019081
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019087, upper bound: 0.0019087
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 8, lower bound: -0.0017486, upper bound: 0.0019081
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 8, lower bound: -0.0019087, upper bound: 0.0019087

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0032284, 0.0051741, 0.0030841, 0.0051853, -0.0014980, 0.0016282
1: 0.0017887, 0.0020698, 0.0017679, 0.0020714, -0.0002164, 0.0002352
2: 0.0114993, 0.0125750, 0.0114931, 0.0126548, -0.0009002, 0.0008282
3: -0.0027874, -0.0016749, -0.0027938, -0.0015923, -0.0009310, 0.0008566
4: -0.0022238, -0.0010195, -0.0023132, -0.0010125, -0.0009273, 0.0010079
5: 0.0050781, 0.0062178, 0.0050715, 0.0063024, -0.0009538, 0.0008775
6: -0.0021519, 0.0023703, -0.0021780, 0.0027058, -0.0037845, 0.0034818
7: -0.0057848, 0.0003740, -0.0062417, 0.0004095, -0.0047419, 0.0051541
8: 0.9851390, 0.9894773, 0.9848170, 0.9895023, -0.0033403, 0.0036307
9: -0.0063355, -0.0023974, -0.0063582, -0.0021052, -0.0032957, 0.0030321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016693, upper bound: 0.0017724
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016693, upper bound: 0.0018353
time: 0.62 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0030652, 0.0053460, 0.0030213, 0.0051879, -0.0015696, 0.0019237
1: 0.0017651, 0.0020946, 0.0017588, 0.0020718, -0.0002268, 0.0002779
2: 0.0114042, 0.0126652, 0.0114916, 0.0126895, -0.0010636, 0.0008678
3: -0.0028857, -0.0015815, -0.0027953, -0.0015564, -0.0011000, 0.0008975
4: -0.0023249, -0.0009130, -0.0023520, -0.0010109, -0.0009716, 0.0011908
5: 0.0049774, 0.0063135, 0.0050700, 0.0063392, -0.0011269, 0.0009195
6: -0.0025516, 0.0027496, -0.0021842, 0.0028516, -0.0044713, 0.0036483
7: -0.0063015, 0.0009184, -0.0064404, 0.0004179, -0.0049686, 0.0060895
8: 0.9847750, 0.9898608, 0.9846771, 0.9895083, -0.0035000, 0.0042895
9: -0.0066836, -0.0020670, -0.0063636, -0.0019782, -0.0038938, 0.0031771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018365, upper bound: 0.0017733
time: 0.65 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018365, upper bound: 0.0018365
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.99 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 8, lower bound: -0.0016693, upper bound: 0.0017724
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 8, lower bound: -0.0016693, upper bound: 0.0018353
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 8, lower bound: -0.0018365, upper bound: 0.0017733
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 8, lower bound: -0.0018365, upper bound: 0.0018365

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032871, 0.0051716, 0.0032294, 0.0051175, -0.0013377, 0.0014905
1: 0.0017972, 0.0020695, 0.0017889, 0.0020616, -0.0001933, 0.0002153
2: 0.0115006, 0.0125425, 0.0115305, 0.0125744, -0.0008241, 0.0007396
3: -0.0027860, -0.0017084, -0.0027550, -0.0016754, -0.0008523, 0.0007649
4: -0.0021875, -0.0010210, -0.0022232, -0.0010545, -0.0008280, 0.0009226
5: 0.0050795, 0.0061835, 0.0051112, 0.0062173, -0.0008731, 0.0007836
6: -0.0021463, 0.0022338, -0.0020204, 0.0023680, -0.0034644, 0.0031091
7: -0.0055990, 0.0003664, -0.0057817, 0.0001949, -0.0042344, 0.0047182
8: 0.9852698, 0.9894719, 0.9851410, 0.9893512, -0.0029828, 0.0033236
9: -0.0063306, -0.0025162, -0.0062210, -0.0023993, -0.0030169, 0.0027076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017017
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017017
time: 0.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032596, 0.0051722, 0.0031587, 0.0051807, -0.0014626, 0.0014318
1: 0.0017932, 0.0020695, 0.0017786, 0.0020708, -0.0002113, 0.0002069
2: 0.0115003, 0.0125577, 0.0114956, 0.0126135, -0.0007916, 0.0008086
3: -0.0027863, -0.0016927, -0.0027912, -0.0016350, -0.0008187, 0.0008363
4: -0.0022045, -0.0010206, -0.0022670, -0.0010153, -0.0009054, 0.0008863
5: 0.0050792, 0.0061996, 0.0050742, 0.0062587, -0.0008388, 0.0008568
6: -0.0021477, 0.0022977, -0.0021674, 0.0025323, -0.0033280, 0.0033994
7: -0.0056860, 0.0003683, -0.0060055, 0.0003951, -0.0046297, 0.0045324
8: 0.9852085, 0.9894733, 0.9849834, 0.9894922, -0.0032613, 0.0031927
9: -0.0063318, -0.0024606, -0.0063490, -0.0022562, -0.0028982, 0.0029604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017678
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017678
time: 0.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031241, 0.0053436, 0.0031659, 0.0051204, -0.0014105, 0.0017820
1: 0.0017736, 0.0020943, 0.0017797, 0.0020621, -0.0002038, 0.0002575
2: 0.0114055, 0.0126327, 0.0115289, 0.0126095, -0.0009852, 0.0007798
3: -0.0028843, -0.0016152, -0.0027567, -0.0016391, -0.0010190, 0.0008065
4: -0.0022884, -0.0009145, -0.0022625, -0.0010527, -0.0008731, 0.0011031
5: 0.0049788, 0.0062790, 0.0051095, 0.0062545, -0.0010439, 0.0008263
6: -0.0025460, 0.0026128, -0.0020272, 0.0025156, -0.0041419, 0.0032783
7: -0.0061152, 0.0009108, -0.0059827, 0.0002042, -0.0044648, 0.0056409
8: 0.9849062, 0.9898554, 0.9849995, 0.9893577, -0.0031451, 0.0039736
9: -0.0066787, -0.0021861, -0.0062269, -0.0022708, -0.0036070, 0.0028549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017128, upper bound: 0.0017030
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017690, upper bound: 0.0017030
time: 0.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0030983, 0.0053441, 0.0030948, 0.0051834, -0.0015347, 0.0017276
1: 0.0017699, 0.0020944, 0.0017694, 0.0020711, -0.0002217, 0.0002496
2: 0.0114052, 0.0126469, 0.0114941, 0.0126488, -0.0009552, 0.0008485
3: -0.0028846, -0.0016005, -0.0027927, -0.0015985, -0.0009879, 0.0008775
4: -0.0023044, -0.0009142, -0.0023065, -0.0010137, -0.0009500, 0.0010694
5: 0.0049785, 0.0062940, 0.0050726, 0.0062961, -0.0010120, 0.0008990
6: -0.0025473, 0.0026726, -0.0021736, 0.0026808, -0.0040155, 0.0035670
7: -0.0061966, 0.0009124, -0.0062077, 0.0004035, -0.0048579, 0.0054687
8: 0.9848489, 0.9898566, 0.9848410, 0.9894981, -0.0034220, 0.0038523
9: -0.0066798, -0.0021341, -0.0063544, -0.0021270, -0.0034968, 0.0031063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017128, upper bound: 0.0017689
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0017690
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017017
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017017
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017678
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017678
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0017128, upper bound: 0.0017030
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0017690, upper bound: 0.0017030
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0017128, upper bound: 0.0017689
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0017690

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0051640, 0.0032662, 0.0051149, -0.0012125, 0.0014358
1: 0.0018107, 0.0020683, 0.0017942, 0.0020613, -0.0001752, 0.0002074
2: 0.0115048, 0.0124907, 0.0115320, 0.0125541, -0.0007938, 0.0006704
3: -0.0027816, -0.0017620, -0.0027535, -0.0016964, -0.0008210, 0.0006933
4: -0.0021294, -0.0010257, -0.0022005, -0.0010561, -0.0007506, 0.0008888
5: 0.0050840, 0.0061285, 0.0051128, 0.0061957, -0.0008411, 0.0007103
6: -0.0021285, 0.0020159, -0.0020144, 0.0022825, -0.0033373, 0.0028182
7: -0.0053021, 0.0003420, -0.0056653, 0.0001867, -0.0038381, 0.0045450
8: 0.9854789, 0.9894549, 0.9852232, 0.9893454, -0.0027036, 0.0032016
9: -0.0063151, -0.0027060, -0.0062157, -0.0024738, -0.0029062, 0.0024542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0015733
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017017
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033749, 0.0052482, 0.0032727, 0.0051135, -0.0012336, 0.0015968
1: 0.0018099, 0.0020805, 0.0017951, 0.0020610, -0.0001782, 0.0002307
2: 0.0114583, 0.0124939, 0.0115328, 0.0125505, -0.0008828, 0.0006820
3: -0.0028298, -0.0017586, -0.0027527, -0.0017002, -0.0009131, 0.0007054
4: -0.0021331, -0.0009736, -0.0021964, -0.0010570, -0.0007636, 0.0009885
5: 0.0050347, 0.0061320, 0.0051136, 0.0061919, -0.0009354, 0.0007227
6: -0.0023241, 0.0020297, -0.0020111, 0.0022673, -0.0037115, 0.0028673
7: -0.0053210, 0.0006086, -0.0056445, 0.0001823, -0.0039050, 0.0050547
8: 0.9854656, 0.9896426, 0.9852378, 0.9893422, -0.0027507, 0.0035607
9: -0.0064855, -0.0026940, -0.0062129, -0.0024871, -0.0032321, 0.0024969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0015733
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017017
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033557, 0.0051645, 0.0031963, 0.0051780, -0.0013384, 0.0013780
1: 0.0018071, 0.0020684, 0.0017841, 0.0020704, -0.0001934, 0.0001991
2: 0.0115045, 0.0125046, 0.0114971, 0.0125927, -0.0007619, 0.0007400
3: -0.0027819, -0.0017476, -0.0027896, -0.0016565, -0.0007879, 0.0007653
4: -0.0021450, -0.0010254, -0.0022437, -0.0010170, -0.0008285, 0.0008530
5: 0.0050837, 0.0061433, 0.0050758, 0.0062367, -0.0008072, 0.0007840
6: -0.0021298, 0.0020744, -0.0021611, 0.0024449, -0.0032028, 0.0031108
7: -0.0053819, 0.0003439, -0.0058864, 0.0003866, -0.0042366, 0.0043620
8: 0.9854227, 0.9894561, 0.9850673, 0.9894862, -0.0029844, 0.0030727
9: -0.0063162, -0.0026550, -0.0063435, -0.0023324, -0.0027892, 0.0027090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0016142
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017678
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033512, 0.0052487, 0.0032010, 0.0051766, -0.0013597, 0.0015568
1: 0.0018065, 0.0020806, 0.0017848, 0.0020702, -0.0001964, 0.0002249
2: 0.0114580, 0.0125071, 0.0114978, 0.0125901, -0.0008607, 0.0007518
3: -0.0028301, -0.0017451, -0.0027888, -0.0016592, -0.0008902, 0.0007775
4: -0.0021478, -0.0009733, -0.0022408, -0.0010179, -0.0008417, 0.0009637
5: 0.0050344, 0.0061459, 0.0050766, 0.0062339, -0.0009120, 0.0007965
6: -0.0023254, 0.0020849, -0.0021579, 0.0024339, -0.0036185, 0.0031604
7: -0.0053962, 0.0006103, -0.0058715, 0.0003821, -0.0043041, 0.0049280
8: 0.9854128, 0.9896438, 0.9850779, 0.9894831, -0.0030319, 0.0034714
9: -0.0064866, -0.0026459, -0.0063407, -0.0023420, -0.0031511, 0.0027522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0016142
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017678
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032235, 0.0053365, 0.0032031, 0.0051178, -0.0012855, 0.0017282
1: 0.0017880, 0.0020933, 0.0017851, 0.0020617, -0.0001857, 0.0002497
2: 0.0114095, 0.0125777, 0.0115304, 0.0125890, -0.0009555, 0.0007107
3: -0.0028803, -0.0016720, -0.0027552, -0.0016604, -0.0009882, 0.0007350
4: -0.0022269, -0.0009189, -0.0022395, -0.0010543, -0.0007957, 0.0010698
5: 0.0049829, 0.0062207, 0.0051111, 0.0062327, -0.0010124, 0.0007530
6: -0.0025295, 0.0023817, -0.0020211, 0.0024291, -0.0040167, 0.0029878
7: -0.0058003, 0.0008882, -0.0058650, 0.0001959, -0.0040691, 0.0054705
8: 0.9851280, 0.9898395, 0.9850824, 0.9893519, -0.0028664, 0.0038535
9: -0.0066643, -0.0023875, -0.0062216, -0.0023461, -0.0034980, 0.0026019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0015713
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0015772
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032148, 0.0054325, 0.0032080, 0.0051164, -0.0013035, 0.0018740
1: 0.0017867, 0.0021071, 0.0017858, 0.0020615, -0.0001883, 0.0002707
2: 0.0113564, 0.0125825, 0.0115311, 0.0125862, -0.0010361, 0.0007207
3: -0.0029352, -0.0016671, -0.0027544, -0.0016632, -0.0010716, 0.0007454
4: -0.0022323, -0.0008595, -0.0022364, -0.0010551, -0.0008069, 0.0011601
5: 0.0049267, 0.0062258, 0.0051119, 0.0062298, -0.0010978, 0.0007636
6: -0.0027526, 0.0024019, -0.0020180, 0.0024176, -0.0043558, 0.0030298
7: -0.0058279, 0.0011921, -0.0058493, 0.0001916, -0.0041263, 0.0059322
8: 0.9851086, 0.9900536, 0.9850935, 0.9893488, -0.0029066, 0.0041788
9: -0.0068586, -0.0023698, -0.0062189, -0.0023562, -0.0037932, 0.0026385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0015713
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0015772
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0031990, 0.0053370, 0.0031332, 0.0051807, -0.0014141, 0.0016741
1: 0.0017845, 0.0020933, 0.0017750, 0.0020708, -0.0002043, 0.0002419
2: 0.0114092, 0.0125912, 0.0114956, 0.0126276, -0.0009256, 0.0007818
3: -0.0028805, -0.0016580, -0.0027912, -0.0016204, -0.0009573, 0.0008086
4: -0.0022421, -0.0009186, -0.0022828, -0.0010154, -0.0008753, 0.0010363
5: 0.0049827, 0.0062351, 0.0050742, 0.0062736, -0.0009807, 0.0008283
6: -0.0025306, 0.0024387, -0.0021673, 0.0025916, -0.0038912, 0.0032867
7: -0.0058780, 0.0008897, -0.0060862, 0.0003950, -0.0044761, 0.0052994
8: 0.9850733, 0.9898407, 0.9849267, 0.9894921, -0.0031531, 0.0037330
9: -0.0066653, -0.0023378, -0.0063489, -0.0022047, -0.0033886, 0.0028622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0015979
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0016039
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0031881, 0.0054330, 0.0031355, 0.0051793, -0.0014321, 0.0018246
1: 0.0017829, 0.0021072, 0.0017753, 0.0020706, -0.0002069, 0.0002636
2: 0.0113561, 0.0125972, 0.0114964, 0.0126263, -0.0010088, 0.0007918
3: -0.0029354, -0.0016518, -0.0027904, -0.0016217, -0.0010433, 0.0008189
4: -0.0022488, -0.0008592, -0.0022814, -0.0010162, -0.0008865, 0.0011295
5: 0.0049264, 0.0062414, 0.0050750, 0.0062723, -0.0010689, 0.0008389
6: -0.0027537, 0.0024639, -0.0021640, 0.0025862, -0.0042410, 0.0033286
7: -0.0059123, 0.0011936, -0.0060789, 0.0003905, -0.0045333, 0.0057758
8: 0.9850491, 0.9900546, 0.9849318, 0.9894890, -0.0031933, 0.0040686
9: -0.0068596, -0.0023159, -0.0063461, -0.0022093, -0.0036932, 0.0028987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0015979
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0016039
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.12 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0015733
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017017
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0015733
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017017
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0016142
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015595, upper bound: 0.0017678
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0016142
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0015979, upper bound: 0.0017678
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0015713
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0015772
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0015713
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0015772
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0015979
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017121, upper bound: 0.0016039
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0015979
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 8, lower bound: -0.0017678, upper bound: 0.0016039

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0051640, 0.0032467, 0.0052725, -0.0014205, 0.0014798
1: 0.0018107, 0.0020683, 0.0017914, 0.0020840, -0.0002052, 0.0002138
2: 0.0115048, 0.0124907, 0.0114448, 0.0125649, -0.0008182, 0.0007854
3: -0.0027816, -0.0017620, -0.0028437, -0.0016853, -0.0008462, 0.0008123
4: -0.0021294, -0.0010257, -0.0022125, -0.0009585, -0.0008793, 0.0009160
5: 0.0050840, 0.0061285, 0.0050204, 0.0062071, -0.0008669, 0.0008321
6: -0.0021285, 0.0020159, -0.0023808, 0.0023278, -0.0034396, 0.0033017
7: -0.0053021, 0.0003420, -0.0057270, 0.0006857, -0.0044966, 0.0046844
8: 0.9854789, 0.9894549, 0.9851796, 0.9896969, -0.0031675, 0.0032998
9: -0.0063151, -0.0027060, -0.0065348, -0.0024344, -0.0029953, 0.0028753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033749, 0.0052482, 0.0032530, 0.0052712, -0.0014419, 0.0016372
1: 0.0018099, 0.0020805, 0.0017923, 0.0020838, -0.0002083, 0.0002365
2: 0.0114583, 0.0124939, 0.0114455, 0.0125613, -0.0009052, 0.0007972
3: -0.0028298, -0.0017586, -0.0028429, -0.0016889, -0.0009362, 0.0008245
4: -0.0021331, -0.0009736, -0.0022086, -0.0009593, -0.0008926, 0.0010134
5: 0.0050347, 0.0061320, 0.0050212, 0.0062034, -0.0009591, 0.0008447
6: -0.0023241, 0.0020297, -0.0023778, 0.0023130, -0.0038052, 0.0033514
7: -0.0053210, 0.0006086, -0.0057068, 0.0006816, -0.0045644, 0.0051824
8: 0.9854656, 0.9896426, 0.9851938, 0.9896940, -0.0032152, 0.0036506
9: -0.0064855, -0.0026940, -0.0065322, -0.0024472, -0.0033138, 0.0029186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0017017
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0017017
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033557, 0.0051645, 0.0031823, 0.0053390, -0.0015457, 0.0014232
1: 0.0018071, 0.0020684, 0.0017821, 0.0020936, -0.0002233, 0.0002056
2: 0.0115045, 0.0125046, 0.0114081, 0.0126004, -0.0007869, 0.0008546
3: -0.0027819, -0.0017476, -0.0028817, -0.0016485, -0.0008138, 0.0008838
4: -0.0021450, -0.0010254, -0.0022524, -0.0009174, -0.0009568, 0.0008810
5: 0.0050837, 0.0061433, 0.0049815, 0.0062448, -0.0008337, 0.0009055
6: -0.0021298, 0.0020744, -0.0025352, 0.0024774, -0.0033080, 0.0035926
7: -0.0053819, 0.0003439, -0.0059307, 0.0008960, -0.0048929, 0.0045052
8: 0.9854227, 0.9894561, 0.9850362, 0.9898450, -0.0034466, 0.0031736
9: -0.0063162, -0.0026550, -0.0066693, -0.0023041, -0.0028808, 0.0031286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015320, upper bound: 0.0017678
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015320, upper bound: 0.0017017
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033512, 0.0052487, 0.0031842, 0.0053377, -0.0015672, 0.0015983
1: 0.0018065, 0.0020806, 0.0017823, 0.0020934, -0.0002264, 0.0002309
2: 0.0114580, 0.0125071, 0.0114088, 0.0125994, -0.0008836, 0.0008665
3: -0.0028301, -0.0017451, -0.0028809, -0.0016496, -0.0009139, 0.0008961
4: -0.0021478, -0.0009733, -0.0022512, -0.0009182, -0.0009701, 0.0009894
5: 0.0050344, 0.0061459, 0.0049822, 0.0062438, -0.0009363, 0.0009181
6: -0.0023254, 0.0020849, -0.0025322, 0.0024731, -0.0037148, 0.0036426
7: -0.0053962, 0.0006103, -0.0059248, 0.0008920, -0.0049609, 0.0050593
8: 0.9854128, 0.9896438, 0.9850402, 0.9898422, -0.0034945, 0.0035639
9: -0.0064866, -0.0026459, -0.0066667, -0.0023079, -0.0032351, 0.0031721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017677
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017017
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032235, 0.0053365, 0.0034031, 0.0051033, -0.0014017, 0.0015134
1: 0.0017880, 0.0020933, 0.0018140, 0.0020596, -0.0002025, 0.0002186
2: 0.0114095, 0.0125777, 0.0115384, 0.0124784, -0.0008367, 0.0007750
3: -0.0028803, -0.0016720, -0.0027469, -0.0017747, -0.0008654, 0.0008015
4: -0.0022269, -0.0009189, -0.0021157, -0.0010633, -0.0008677, 0.0009368
5: 0.0049829, 0.0062207, 0.0051196, 0.0061155, -0.0008866, 0.0008211
6: -0.0025295, 0.0023817, -0.0019874, 0.0019643, -0.0035176, 0.0032580
7: -0.0058003, 0.0008882, -0.0052319, 0.0001500, -0.0044371, 0.0047906
8: 0.9851280, 0.9898395, 0.9855284, 0.9893195, -0.0031256, 0.0033746
9: -0.0066643, -0.0023875, -0.0061922, -0.0027510, -0.0030633, 0.0028372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015713
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015713
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032235, 0.0053365, 0.0032467, 0.0052725, -0.0013170, 0.0014133
1: 0.0017880, 0.0020933, 0.0017914, 0.0020840, -0.0001903, 0.0002042
2: 0.0114095, 0.0125777, 0.0114448, 0.0125649, -0.0007814, 0.0007281
3: -0.0028803, -0.0016720, -0.0028437, -0.0016853, -0.0008082, 0.0007531
4: -0.0022269, -0.0009189, -0.0022125, -0.0009585, -0.0008152, 0.0008749
5: 0.0049829, 0.0062207, 0.0050204, 0.0062071, -0.0008279, 0.0007715
6: -0.0025295, 0.0023817, -0.0023808, 0.0023278, -0.0032850, 0.0030611
7: -0.0058003, 0.0008882, -0.0057270, 0.0006857, -0.0041689, 0.0044738
8: 0.9851280, 0.9898395, 0.9851796, 0.9896969, -0.0029367, 0.0031515
9: -0.0066643, -0.0023875, -0.0065348, -0.0024344, -0.0028607, 0.0026657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015772
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015772
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032148, 0.0054325, 0.0034107, 0.0051020, -0.0014229, 0.0016627
1: 0.0017867, 0.0021071, 0.0018150, 0.0020594, -0.0002056, 0.0002402
2: 0.0113564, 0.0125825, 0.0115391, 0.0124742, -0.0009192, 0.0007867
3: -0.0029352, -0.0016671, -0.0027462, -0.0017791, -0.0009507, 0.0008136
4: -0.0022323, -0.0008595, -0.0021110, -0.0010641, -0.0008808, 0.0010292
5: 0.0049267, 0.0062258, 0.0051203, 0.0061111, -0.0009740, 0.0008335
6: -0.0027526, 0.0024019, -0.0019845, 0.0019467, -0.0038645, 0.0033071
7: -0.0058279, 0.0011921, -0.0052080, 0.0001459, -0.0045040, 0.0052631
8: 0.9851086, 0.9900536, 0.9855453, 0.9893167, -0.0031727, 0.0037074
9: -0.0068586, -0.0023698, -0.0061897, -0.0027662, -0.0033654, 0.0028800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015713
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015713
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032148, 0.0054325, 0.0032530, 0.0052712, -0.0013352, 0.0015711
1: 0.0017867, 0.0021071, 0.0017923, 0.0020838, -0.0001929, 0.0002270
2: 0.0113564, 0.0125825, 0.0114455, 0.0125613, -0.0008686, 0.0007382
3: -0.0029352, -0.0016671, -0.0028429, -0.0016889, -0.0008984, 0.0007635
4: -0.0022323, -0.0008595, -0.0022086, -0.0009593, -0.0008265, 0.0009725
5: 0.0049267, 0.0062258, 0.0050212, 0.0062034, -0.0009203, 0.0007822
6: -0.0027526, 0.0024019, -0.0023778, 0.0023130, -0.0036517, 0.0031034
7: -0.0058279, 0.0011921, -0.0057068, 0.0006816, -0.0042266, 0.0049733
8: 0.9851086, 0.9900536, 0.9851938, 0.9896940, -0.0029773, 0.0035033
9: -0.0068586, -0.0023698, -0.0065322, -0.0024472, -0.0031800, 0.0027026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015772
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015772
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0031990, 0.0053370, 0.0033391, 0.0051669, -0.0015334, 0.0014579
1: 0.0017845, 0.0020933, 0.0018047, 0.0020688, -0.0002215, 0.0002106
2: 0.0114092, 0.0125912, 0.0115032, 0.0125138, -0.0008060, 0.0008478
3: -0.0028805, -0.0016580, -0.0027833, -0.0017381, -0.0008336, 0.0008768
4: -0.0022421, -0.0009186, -0.0021553, -0.0010239, -0.0009492, 0.0009025
5: 0.0049827, 0.0062351, 0.0050823, 0.0061530, -0.0008540, 0.0008983
6: -0.0025306, 0.0024387, -0.0021353, 0.0021131, -0.0033885, 0.0035640
7: -0.0058780, 0.0008897, -0.0054346, 0.0003513, -0.0048539, 0.0046149
8: 0.9850733, 0.9898407, 0.9853856, 0.9894614, -0.0034192, 0.0032508
9: -0.0066653, -0.0023378, -0.0063210, -0.0026213, -0.0029509, 0.0031037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0015979
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0015713
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0031990, 0.0053370, 0.0031823, 0.0053390, -0.0014502, 0.0013514
1: 0.0017845, 0.0020933, 0.0017821, 0.0020936, -0.0002095, 0.0001952
2: 0.0114092, 0.0125912, 0.0114081, 0.0126004, -0.0007472, 0.0008018
3: -0.0028805, -0.0016580, -0.0028817, -0.0016485, -0.0007727, 0.0008292
4: -0.0022421, -0.0009186, -0.0022524, -0.0009174, -0.0008977, 0.0008365
5: 0.0049827, 0.0062351, 0.0049815, 0.0062448, -0.0007917, 0.0008495
6: -0.0025306, 0.0024387, -0.0025352, 0.0024774, -0.0031410, 0.0033706
7: -0.0058780, 0.0008897, -0.0059307, 0.0008960, -0.0045905, 0.0042778
8: 0.9850733, 0.9898407, 0.9850362, 0.9898450, -0.0032336, 0.0030134
9: -0.0066653, -0.0023378, -0.0066693, -0.0023041, -0.0027354, 0.0029353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0016039
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0015771
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031881, 0.0054330, 0.0033449, 0.0051656, -0.0015544, 0.0016121
1: 0.0017829, 0.0021072, 0.0018055, 0.0020686, -0.0002246, 0.0002329
2: 0.0113561, 0.0125972, 0.0115039, 0.0125105, -0.0008913, 0.0008594
3: -0.0029354, -0.0016518, -0.0027825, -0.0017415, -0.0009218, 0.0008888
4: -0.0022488, -0.0008592, -0.0021517, -0.0010247, -0.0009622, 0.0009979
5: 0.0049264, 0.0062414, 0.0050831, 0.0061496, -0.0009444, 0.0009105
6: -0.0027537, 0.0024639, -0.0021322, 0.0020995, -0.0037469, 0.0036128
7: -0.0059123, 0.0011936, -0.0054160, 0.0003472, -0.0049203, 0.0051030
8: 0.9850491, 0.9900546, 0.9853987, 0.9894584, -0.0034659, 0.0035946
9: -0.0068596, -0.0023159, -0.0063184, -0.0026332, -0.0032630, 0.0031462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015979
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015713
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0031881, 0.0054330, 0.0031842, 0.0053377, -0.0014683, 0.0015256
1: 0.0017829, 0.0021072, 0.0017823, 0.0020934, -0.0002121, 0.0002204
2: 0.0113561, 0.0125972, 0.0114088, 0.0125994, -0.0008435, 0.0008118
3: -0.0029354, -0.0016518, -0.0028809, -0.0016496, -0.0008724, 0.0008396
4: -0.0022488, -0.0008592, -0.0022512, -0.0009182, -0.0009089, 0.0009444
5: 0.0049264, 0.0062414, 0.0049822, 0.0062438, -0.0008937, 0.0008602
6: -0.0027537, 0.0024639, -0.0025322, 0.0024731, -0.0035460, 0.0034128
7: -0.0059123, 0.0011936, -0.0059248, 0.0008920, -0.0046480, 0.0048294
8: 0.9850491, 0.9900546, 0.9850402, 0.9898422, -0.0032741, 0.0034019
9: -0.0068596, -0.0023159, -0.0066667, -0.0023079, -0.0030880, 0.0029720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0016039
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015772
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.04 seconds
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0017017
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0017017
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015320, upper bound: 0.0017678
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015320, upper bound: 0.0017017
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017677
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017017
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015713
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015713
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015772
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015772
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015713
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015713
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015772
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015772
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0015979
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0015713
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0016039
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0016687, upper bound: 0.0015771
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015979
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015713
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0016039
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015772

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034605, 0.0050987, 0.0032467, 0.0052725, -0.0013435, 0.0013836
1: 0.0018222, 0.0020589, 0.0017914, 0.0020840, -0.0001941, 0.0001999
2: 0.0115409, 0.0124466, 0.0114448, 0.0125649, -0.0007649, 0.0007428
3: -0.0027443, -0.0018076, -0.0028437, -0.0016853, -0.0007911, 0.0007682
4: -0.0020802, -0.0010661, -0.0022125, -0.0009585, -0.0008316, 0.0008565
5: 0.0051223, 0.0060819, 0.0050204, 0.0062071, -0.0008105, 0.0007870
6: -0.0019767, 0.0018309, -0.0023808, 0.0023278, -0.0032158, 0.0031226
7: -0.0050502, 0.0001354, -0.0057270, 0.0006857, -0.0042528, 0.0043797
8: 0.9856564, 0.9893093, 0.9851796, 0.9896969, -0.0029957, 0.0030851
9: -0.0061829, -0.0028671, -0.0065348, -0.0024344, -0.0028005, 0.0027193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0016691
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0034007, 0.0051621, 0.0032467, 0.0052725, -0.0014171, 0.0014785
1: 0.0018136, 0.0020681, 0.0017914, 0.0020840, -0.0002047, 0.0002136
2: 0.0115058, 0.0124797, 0.0114448, 0.0125649, -0.0008174, 0.0007835
3: -0.0027806, -0.0017734, -0.0028437, -0.0016853, -0.0008454, 0.0008103
4: -0.0021172, -0.0010268, -0.0022125, -0.0009585, -0.0008772, 0.0009152
5: 0.0050851, 0.0061169, 0.0050204, 0.0062071, -0.0008661, 0.0008301
6: -0.0021242, 0.0019698, -0.0023808, 0.0023278, -0.0034365, 0.0032938
7: -0.0052394, 0.0003363, -0.0057270, 0.0006857, -0.0044858, 0.0046803
8: 0.9855232, 0.9894508, 0.9851796, 0.9896969, -0.0031599, 0.0032969
9: -0.0063114, -0.0027462, -0.0065348, -0.0024344, -0.0029927, 0.0028684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0016691
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034555, 0.0051882, 0.0032530, 0.0052712, -0.0013651, 0.0015595
1: 0.0018215, 0.0020719, 0.0017923, 0.0020838, -0.0001972, 0.0002253
2: 0.0114914, 0.0124494, 0.0114455, 0.0125613, -0.0008622, 0.0007547
3: -0.0027955, -0.0018047, -0.0028429, -0.0016889, -0.0008917, 0.0007806
4: -0.0020833, -0.0010107, -0.0022086, -0.0009593, -0.0008450, 0.0009654
5: 0.0050698, 0.0060848, 0.0050212, 0.0062034, -0.0009136, 0.0007997
6: -0.0021849, 0.0018424, -0.0023778, 0.0023130, -0.0036247, 0.0031729
7: -0.0050659, 0.0004189, -0.0057068, 0.0006816, -0.0043212, 0.0049366
8: 0.9856453, 0.9895089, 0.9851938, 0.9896940, -0.0030439, 0.0034774
9: -0.0063642, -0.0028571, -0.0065322, -0.0024472, -0.0031566, 0.0027631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016688
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016688
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033941, 0.0052462, 0.0032530, 0.0052712, -0.0014386, 0.0016355
1: 0.0018126, 0.0020802, 0.0017923, 0.0020838, -0.0002078, 0.0002363
2: 0.0114594, 0.0124834, 0.0114455, 0.0125613, -0.0009042, 0.0007954
3: -0.0028286, -0.0017696, -0.0028429, -0.0016889, -0.0009352, 0.0008226
4: -0.0021213, -0.0009748, -0.0022086, -0.0009593, -0.0008905, 0.0010124
5: 0.0050358, 0.0061208, 0.0050212, 0.0062034, -0.0009580, 0.0008427
6: -0.0023197, 0.0019852, -0.0023778, 0.0023130, -0.0038013, 0.0033438
7: -0.0052604, 0.0006025, -0.0057068, 0.0006816, -0.0045539, 0.0051770
8: 0.9855083, 0.9896383, 0.9851938, 0.9896940, -0.0032079, 0.0036468
9: -0.0064816, -0.0027327, -0.0065322, -0.0024472, -0.0033103, 0.0029119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016687
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016688
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034605, 0.0050987, 0.0031823, 0.0053390, -0.0014399, 0.0014638
1: 0.0018222, 0.0020589, 0.0017821, 0.0020936, -0.0002080, 0.0002115
2: 0.0115409, 0.0124466, 0.0114081, 0.0126004, -0.0008093, 0.0007961
3: -0.0027443, -0.0018076, -0.0028817, -0.0016485, -0.0008370, 0.0008233
4: -0.0020802, -0.0010661, -0.0022524, -0.0009174, -0.0008913, 0.0009061
5: 0.0051223, 0.0060819, 0.0049815, 0.0062448, -0.0008575, 0.0008435
6: -0.0019767, 0.0018309, -0.0025352, 0.0024774, -0.0034022, 0.0033466
7: -0.0050502, 0.0001354, -0.0059307, 0.0008960, -0.0045578, 0.0046335
8: 0.9856564, 0.9893093, 0.9850362, 0.9898450, -0.0032106, 0.0032639
9: -0.0061829, -0.0028671, -0.0066693, -0.0023041, -0.0029628, 0.0029144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0017216
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0017678
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0034007, 0.0051621, 0.0031823, 0.0053390, -0.0013835, 0.0014210
1: 0.0018136, 0.0020681, 0.0017821, 0.0020936, -0.0001999, 0.0002053
2: 0.0115058, 0.0124797, 0.0114081, 0.0126004, -0.0007856, 0.0007649
3: -0.0027806, -0.0017734, -0.0028817, -0.0016485, -0.0008125, 0.0007911
4: -0.0021172, -0.0010268, -0.0022524, -0.0009174, -0.0008564, 0.0008796
5: 0.0050851, 0.0061169, 0.0049815, 0.0062448, -0.0008324, 0.0008105
6: -0.0021242, 0.0019698, -0.0025352, 0.0024774, -0.0033028, 0.0032157
7: -0.0052394, 0.0003363, -0.0059307, 0.0008960, -0.0043795, 0.0044981
8: 0.9855232, 0.9894508, 0.9850362, 0.9898450, -0.0030850, 0.0031685
9: -0.0063114, -0.0027462, -0.0066693, -0.0023041, -0.0028762, 0.0028003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0016691
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0017017
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034555, 0.0051882, 0.0031842, 0.0053377, -0.0014614, 0.0016329
1: 0.0018215, 0.0020719, 0.0017823, 0.0020934, -0.0002111, 0.0002359
2: 0.0114914, 0.0124494, 0.0114088, 0.0125994, -0.0009028, 0.0008080
3: -0.0027955, -0.0018047, -0.0028809, -0.0016496, -0.0009337, 0.0008357
4: -0.0020833, -0.0010107, -0.0022512, -0.0009182, -0.0009047, 0.0010108
5: 0.0050698, 0.0060848, 0.0049822, 0.0062438, -0.0009565, 0.0008561
6: -0.0021849, 0.0018424, -0.0025322, 0.0024731, -0.0037952, 0.0033968
7: -0.0050659, 0.0004189, -0.0059248, 0.0008920, -0.0046261, 0.0051688
8: 0.9856453, 0.9895089, 0.9850402, 0.9898422, -0.0032587, 0.0036410
9: -0.0063642, -0.0028571, -0.0066667, -0.0023079, -0.0033050, 0.0029581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017121
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017121
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033941, 0.0052462, 0.0031842, 0.0053377, -0.0014025, 0.0015957
1: 0.0018126, 0.0020802, 0.0017823, 0.0020934, -0.0002026, 0.0002305
2: 0.0114594, 0.0124834, 0.0114088, 0.0125994, -0.0008822, 0.0007754
3: -0.0028286, -0.0017696, -0.0028809, -0.0016496, -0.0009124, 0.0008020
4: -0.0021213, -0.0009748, -0.0022512, -0.0009182, -0.0008682, 0.0009877
5: 0.0050358, 0.0061208, 0.0049822, 0.0062438, -0.0009347, 0.0008216
6: -0.0023197, 0.0019852, -0.0025322, 0.0024731, -0.0037087, 0.0032598
7: -0.0052604, 0.0006025, -0.0059248, 0.0008920, -0.0044396, 0.0050510
8: 0.9855083, 0.9896383, 0.9850402, 0.9898422, -0.0031274, 0.0035580
9: -0.0064816, -0.0027327, -0.0066667, -0.0023079, -0.0032297, 0.0028388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0016687
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0016688
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0034031, 0.0051033, -0.0013150, 0.0014160
1: 0.0018003, 0.0020834, 0.0018140, 0.0020596, -0.0001900, 0.0002046
2: 0.0114471, 0.0125306, 0.0115384, 0.0124784, -0.0007829, 0.0007270
3: -0.0028413, -0.0017207, -0.0027469, -0.0017747, -0.0008097, 0.0007519
4: -0.0021742, -0.0009610, -0.0021157, -0.0010633, -0.0008140, 0.0008765
5: 0.0050228, 0.0061709, 0.0051196, 0.0061155, -0.0008295, 0.0007703
6: -0.0023713, 0.0021838, -0.0019874, 0.0019643, -0.0032912, 0.0030565
7: -0.0055309, 0.0006727, -0.0052319, 0.0001500, -0.0041626, 0.0044823
8: 0.9853178, 0.9896877, 0.9855284, 0.9893195, -0.0029323, 0.0031574
9: -0.0065265, -0.0025598, -0.0061922, -0.0027510, -0.0028661, 0.0026617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015328
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015712
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0034031, 0.0051033, -0.0014013, 0.0015119
1: 0.0017913, 0.0020930, 0.0018140, 0.0020596, -0.0002024, 0.0002184
2: 0.0114105, 0.0125649, 0.0115384, 0.0124784, -0.0008359, 0.0007747
3: -0.0028792, -0.0016853, -0.0027469, -0.0017747, -0.0008645, 0.0008013
4: -0.0022126, -0.0009201, -0.0021157, -0.0010633, -0.0008674, 0.0009359
5: 0.0049841, 0.0062072, 0.0051196, 0.0061155, -0.0008857, 0.0008209
6: -0.0025250, 0.0023279, -0.0019874, 0.0019643, -0.0035141, 0.0032569
7: -0.0057272, 0.0008821, -0.0052319, 0.0001500, -0.0044357, 0.0047858
8: 0.9851795, 0.9898352, 0.9855284, 0.9893195, -0.0031246, 0.0033712
9: -0.0066604, -0.0024342, -0.0061922, -0.0027510, -0.0030602, 0.0028363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015328
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015713
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0032467, 0.0052725, -0.0012371, 0.0013104
1: 0.0018003, 0.0020834, 0.0017914, 0.0020840, -0.0001787, 0.0001893
2: 0.0114471, 0.0125306, 0.0114448, 0.0125649, -0.0007245, 0.0006840
3: -0.0028413, -0.0017207, -0.0028437, -0.0016853, -0.0007493, 0.0007074
4: -0.0021742, -0.0009610, -0.0022125, -0.0009585, -0.0007658, 0.0008112
5: 0.0050228, 0.0061709, 0.0050204, 0.0062071, -0.0007676, 0.0007247
6: -0.0023713, 0.0021838, -0.0023808, 0.0023278, -0.0030458, 0.0028754
7: -0.0055309, 0.0006727, -0.0057270, 0.0006857, -0.0039160, 0.0041481
8: 0.9853178, 0.9896877, 0.9851796, 0.9896969, -0.0027585, 0.0029220
9: -0.0065265, -0.0025598, -0.0065348, -0.0024344, -0.0026524, 0.0025040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015380
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015772
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0032467, 0.0052725, -0.0013154, 0.0014118
1: 0.0017913, 0.0020930, 0.0017914, 0.0020840, -0.0001900, 0.0002040
2: 0.0114105, 0.0125649, 0.0114448, 0.0125649, -0.0007806, 0.0007272
3: -0.0028792, -0.0016853, -0.0028437, -0.0016853, -0.0008073, 0.0007522
4: -0.0022126, -0.0009201, -0.0022125, -0.0009585, -0.0008143, 0.0008739
5: 0.0049841, 0.0062072, 0.0050204, 0.0062071, -0.0008270, 0.0007706
6: -0.0025250, 0.0023279, -0.0023808, 0.0023278, -0.0032815, 0.0030573
7: -0.0057272, 0.0008821, -0.0057270, 0.0006857, -0.0041638, 0.0044691
8: 0.9851795, 0.9898352, 0.9851796, 0.9896969, -0.0029331, 0.0031481
9: -0.0066604, -0.0024342, -0.0065348, -0.0024344, -0.0028577, 0.0026625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015380
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015772
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0034107, 0.0051020, -0.0013361, 0.0015733
1: 0.0017990, 0.0020967, 0.0018150, 0.0020594, -0.0001930, 0.0002273
2: 0.0113964, 0.0125355, 0.0115391, 0.0124742, -0.0008698, 0.0007387
3: -0.0028937, -0.0017156, -0.0027462, -0.0017791, -0.0008996, 0.0007640
4: -0.0021797, -0.0009043, -0.0021110, -0.0010641, -0.0008270, 0.0009739
5: 0.0049691, 0.0061761, 0.0051203, 0.0061111, -0.0009216, 0.0007827
6: -0.0025842, 0.0022046, -0.0019845, 0.0019467, -0.0036568, 0.0031054
7: -0.0055591, 0.0009628, -0.0052080, 0.0001459, -0.0042292, 0.0049803
8: 0.9852980, 0.9898920, 0.9855453, 0.9893167, -0.0029792, 0.0035082
9: -0.0067120, -0.0025417, -0.0061897, -0.0027662, -0.0031845, 0.0027043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015320
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015319
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0034107, 0.0051020, -0.0014226, 0.0016610
1: 0.0017893, 0.0021068, 0.0018150, 0.0020594, -0.0002055, 0.0002400
2: 0.0113575, 0.0125729, 0.0115391, 0.0124742, -0.0009183, 0.0007865
3: -0.0029340, -0.0016770, -0.0027462, -0.0017791, -0.0009498, 0.0008134
4: -0.0022215, -0.0008607, -0.0021110, -0.0010641, -0.0008806, 0.0010282
5: 0.0049279, 0.0062156, 0.0051203, 0.0061111, -0.0009730, 0.0008333
6: -0.0027479, 0.0023614, -0.0019845, 0.0019467, -0.0038607, 0.0033065
7: -0.0057728, 0.0011857, -0.0052080, 0.0001459, -0.0045031, 0.0052580
8: 0.9851474, 0.9900490, 0.9855453, 0.9893167, -0.0031721, 0.0037038
9: -0.0068545, -0.0024051, -0.0061897, -0.0027662, -0.0033621, 0.0028794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015319
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015320
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0032530, 0.0052712, -0.0012555, 0.0014846
1: 0.0017990, 0.0020967, 0.0017923, 0.0020838, -0.0001814, 0.0002145
2: 0.0113964, 0.0125355, 0.0114455, 0.0125613, -0.0008208, 0.0006942
3: -0.0028937, -0.0017156, -0.0028429, -0.0016889, -0.0008489, 0.0007179
4: -0.0021797, -0.0009043, -0.0022086, -0.0009593, -0.0007772, 0.0009190
5: 0.0049691, 0.0061761, 0.0050212, 0.0062034, -0.0008697, 0.0007355
6: -0.0025842, 0.0022046, -0.0023778, 0.0023130, -0.0034506, 0.0029182
7: -0.0055591, 0.0009628, -0.0057068, 0.0006816, -0.0039744, 0.0046994
8: 0.9852980, 0.9898920, 0.9851938, 0.9896940, -0.0027996, 0.0033103
9: -0.0067120, -0.0025417, -0.0065322, -0.0024472, -0.0030049, 0.0025413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015373
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015374
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0032530, 0.0052712, -0.0013335, 0.0015694
1: 0.0017893, 0.0021068, 0.0017923, 0.0020838, -0.0001926, 0.0002267
2: 0.0113575, 0.0125729, 0.0114455, 0.0125613, -0.0008677, 0.0007372
3: -0.0029340, -0.0016770, -0.0028429, -0.0016889, -0.0008974, 0.0007625
4: -0.0022215, -0.0008607, -0.0022086, -0.0009593, -0.0008254, 0.0009715
5: 0.0049279, 0.0062156, 0.0050212, 0.0062034, -0.0009193, 0.0007811
6: -0.0027479, 0.0023614, -0.0023778, 0.0023130, -0.0036476, 0.0030993
7: -0.0057728, 0.0011857, -0.0057068, 0.0006816, -0.0042210, 0.0049677
8: 0.9851474, 0.9900490, 0.9851938, 0.9896940, -0.0029734, 0.0034994
9: -0.0068545, -0.0024051, -0.0065322, -0.0024472, -0.0031765, 0.0026990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015373
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015374
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0033391, 0.0051669, -0.0014106, 0.0014814
1: 0.0018003, 0.0020834, 0.0018047, 0.0020688, -0.0002038, 0.0002140
2: 0.0114471, 0.0125306, 0.0115032, 0.0125138, -0.0008190, 0.0007799
3: -0.0028413, -0.0017207, -0.0027833, -0.0017381, -0.0008471, 0.0008066
4: -0.0021742, -0.0009610, -0.0021553, -0.0010239, -0.0008732, 0.0009170
5: 0.0050228, 0.0061709, 0.0050823, 0.0061530, -0.0008678, 0.0008263
6: -0.0023713, 0.0021838, -0.0021353, 0.0021131, -0.0034432, 0.0032787
7: -0.0055309, 0.0006727, -0.0054346, 0.0003513, -0.0044653, 0.0046893
8: 0.9853178, 0.9896877, 0.9853856, 0.9894614, -0.0031455, 0.0033032
9: -0.0065265, -0.0025598, -0.0063210, -0.0026213, -0.0029984, 0.0028552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015643
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015979
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0033391, 0.0051669, -0.0013530, 0.0014555
1: 0.0017913, 0.0020930, 0.0018047, 0.0020688, -0.0001955, 0.0002103
2: 0.0114105, 0.0125649, 0.0115032, 0.0125138, -0.0008047, 0.0007480
3: -0.0028792, -0.0016853, -0.0027833, -0.0017381, -0.0008323, 0.0007737
4: -0.0022126, -0.0009201, -0.0021553, -0.0010239, -0.0008375, 0.0009010
5: 0.0049841, 0.0062072, 0.0050823, 0.0061530, -0.0008526, 0.0007926
6: -0.0025250, 0.0023279, -0.0021353, 0.0021131, -0.0033830, 0.0031447
7: -0.0057272, 0.0008821, -0.0054346, 0.0003513, -0.0042828, 0.0046073
8: 0.9851795, 0.9898352, 0.9853856, 0.9894614, -0.0030169, 0.0032455
9: -0.0066604, -0.0024342, -0.0063210, -0.0026213, -0.0029461, 0.0027386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015328
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015713
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0031823, 0.0053390, -0.0013392, 0.0013796
1: 0.0018003, 0.0020834, 0.0017821, 0.0020936, -0.0001935, 0.0001993
2: 0.0114471, 0.0125306, 0.0114081, 0.0126004, -0.0007628, 0.0007404
3: -0.0028413, -0.0017207, -0.0028817, -0.0016485, -0.0007889, 0.0007657
4: -0.0021742, -0.0009610, -0.0022524, -0.0009174, -0.0008290, 0.0008540
5: 0.0050228, 0.0061709, 0.0049815, 0.0062448, -0.0008082, 0.0007845
6: -0.0023713, 0.0021838, -0.0025352, 0.0024774, -0.0032067, 0.0031126
7: -0.0055309, 0.0006727, -0.0059307, 0.0008960, -0.0042391, 0.0043672
8: 0.9853178, 0.9896877, 0.9850362, 0.9898450, -0.0029861, 0.0030763
9: -0.0065265, -0.0025598, -0.0066693, -0.0023041, -0.0027925, 0.0027106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0015700
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0016039
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0031823, 0.0053390, -0.0012759, 0.0013490
1: 0.0017913, 0.0020930, 0.0017821, 0.0020936, -0.0001843, 0.0001949
2: 0.0114105, 0.0125649, 0.0114081, 0.0126004, -0.0007458, 0.0007054
3: -0.0028792, -0.0016853, -0.0028817, -0.0016485, -0.0007714, 0.0007296
4: -0.0022126, -0.0009201, -0.0022524, -0.0009174, -0.0007898, 0.0008351
5: 0.0049841, 0.0062072, 0.0049815, 0.0062448, -0.0007902, 0.0007474
6: -0.0025250, 0.0023279, -0.0025352, 0.0024774, -0.0031355, 0.0029655
7: -0.0057272, 0.0008821, -0.0059307, 0.0008960, -0.0040388, 0.0042702
8: 0.9851795, 0.9898352, 0.9850362, 0.9898450, -0.0028450, 0.0030080
9: -0.0066604, -0.0024342, -0.0066693, -0.0023041, -0.0027305, 0.0025825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0015380
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0015772
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0033449, 0.0051656, -0.0014317, 0.0016386
1: 0.0017990, 0.0020967, 0.0018055, 0.0020686, -0.0002068, 0.0002367
2: 0.0113964, 0.0125355, 0.0115039, 0.0125105, -0.0009059, 0.0007916
3: -0.0028937, -0.0017156, -0.0027825, -0.0017415, -0.0009370, 0.0008187
4: -0.0021797, -0.0009043, -0.0021517, -0.0010247, -0.0008863, 0.0010143
5: 0.0049691, 0.0061761, 0.0050831, 0.0061496, -0.0009599, 0.0008387
6: -0.0025842, 0.0022046, -0.0021322, 0.0020995, -0.0038086, 0.0033278
7: -0.0055591, 0.0009628, -0.0054160, 0.0003472, -0.0045321, 0.0051869
8: 0.9852980, 0.9898920, 0.9853987, 0.9894584, -0.0031925, 0.0036538
9: -0.0067120, -0.0025417, -0.0063184, -0.0026332, -0.0033167, 0.0028980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015595
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015595
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0033449, 0.0051656, -0.0013713, 0.0016096
1: 0.0017893, 0.0021068, 0.0018055, 0.0020686, -0.0001981, 0.0002325
2: 0.0113575, 0.0125729, 0.0115039, 0.0125105, -0.0008899, 0.0007582
3: -0.0029340, -0.0016770, -0.0027825, -0.0017415, -0.0009204, 0.0007841
4: -0.0022215, -0.0008607, -0.0021517, -0.0010247, -0.0008489, 0.0009963
5: 0.0049279, 0.0062156, 0.0050831, 0.0061496, -0.0009429, 0.0008033
6: -0.0027479, 0.0023614, -0.0021322, 0.0020995, -0.0037411, 0.0031873
7: -0.0057728, 0.0011857, -0.0054160, 0.0003472, -0.0043408, 0.0050950
8: 0.9851474, 0.9900490, 0.9853987, 0.9894584, -0.0030578, 0.0035890
9: -0.0068545, -0.0024051, -0.0063184, -0.0026332, -0.0032579, 0.0027757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015319
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015319
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0031842, 0.0053377, -0.0013576, 0.0015513
1: 0.0017990, 0.0020967, 0.0017823, 0.0020934, -0.0001961, 0.0002241
2: 0.0113964, 0.0125355, 0.0114088, 0.0125994, -0.0008577, 0.0007506
3: -0.0028937, -0.0017156, -0.0028809, -0.0016496, -0.0008871, 0.0007763
4: -0.0021797, -0.0009043, -0.0022512, -0.0009182, -0.0008404, 0.0009603
5: 0.0049691, 0.0061761, 0.0049822, 0.0062438, -0.0009088, 0.0007953
6: -0.0025842, 0.0022046, -0.0025322, 0.0024731, -0.0036057, 0.0031555
7: -0.0055591, 0.0009628, -0.0059248, 0.0008920, -0.0042975, 0.0049106
8: 0.9852980, 0.9898920, 0.9850402, 0.9898422, -0.0030272, 0.0034591
9: -0.0067120, -0.0025417, -0.0066667, -0.0023079, -0.0031400, 0.0027479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015656
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015656
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0031842, 0.0053377, -0.0012991, 0.0015230
1: 0.0017893, 0.0021068, 0.0017823, 0.0020934, -0.0001877, 0.0002200
2: 0.0113575, 0.0125729, 0.0114088, 0.0125994, -0.0008420, 0.0007182
3: -0.0029340, -0.0016770, -0.0028809, -0.0016496, -0.0008709, 0.0007428
4: -0.0022215, -0.0008607, -0.0022512, -0.0009182, -0.0008042, 0.0009428
5: 0.0049279, 0.0062156, 0.0049822, 0.0062438, -0.0008922, 0.0007610
6: -0.0027479, 0.0023614, -0.0025322, 0.0024731, -0.0035400, 0.0030195
7: -0.0057728, 0.0011857, -0.0059248, 0.0008920, -0.0041123, 0.0048211
8: 0.9851474, 0.9900490, 0.9850402, 0.9898422, -0.0028968, 0.0033961
9: -0.0068545, -0.0024051, -0.0066667, -0.0023079, -0.0030827, 0.0026295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015374
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015374
time: 0.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0016691
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0016691
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015323, upper bound: 0.0017017
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016688
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016688
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016687
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015716, upper bound: 0.0016688
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0017216
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0017678
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0016691
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015319, upper bound: 0.0017017
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017121
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0017121
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0016687
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0015713, upper bound: 0.0016688
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015328
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015712
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015328
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016761, upper bound: 0.0015713
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015380
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015772
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015380
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016769, upper bound: 0.0015772
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015320
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015319
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015319
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017202, upper bound: 0.0015320
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015373
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015374
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015373
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017207, upper bound: 0.0015374
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015643
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015979
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015328
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016688, upper bound: 0.0015713
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0015700
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0016039
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0015380
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0016697, upper bound: 0.0015772
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015595
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015595
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015319
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015319
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015656
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015656
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015374
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.27
Output dim: 8, lower bound: -0.0017030, upper bound: 0.0015374

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034605, 0.0050987, 0.0033086, 0.0052684, -0.0013401, 0.0013114
1: 0.0018222, 0.0020589, 0.0018003, 0.0020834, -0.0001936, 0.0001895
2: 0.0115409, 0.0124466, 0.0114471, 0.0125306, -0.0007250, 0.0007409
3: -0.0027443, -0.0018076, -0.0028413, -0.0017207, -0.0007499, 0.0007663
4: -0.0020802, -0.0010661, -0.0021742, -0.0009610, -0.0008295, 0.0008118
5: 0.0051223, 0.0060819, 0.0050228, 0.0061709, -0.0007682, 0.0007850
6: -0.0019767, 0.0018309, -0.0023713, 0.0021838, -0.0030480, 0.0031148
7: -0.0050502, 0.0001354, -0.0055309, 0.0006727, -0.0042421, 0.0041511
8: 0.9856564, 0.9893093, 0.9853178, 0.9896877, -0.0029882, 0.0029241
9: -0.0061829, -0.0028671, -0.0065265, -0.0025598, -0.0026543, 0.0027125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014680, upper bound: 0.0016087
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014559, upper bound: 0.0016087
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034605, 0.0050987, 0.0032997, 0.0053600, -0.0014741, 0.0013679
1: 0.0018222, 0.0020589, 0.0017990, 0.0020967, -0.0002130, 0.0001976
2: 0.0115409, 0.0124466, 0.0113964, 0.0125355, -0.0007563, 0.0008150
3: -0.0027443, -0.0018076, -0.0028937, -0.0017156, -0.0007822, 0.0008429
4: -0.0020802, -0.0010661, -0.0021797, -0.0009043, -0.0009125, 0.0008468
5: 0.0051223, 0.0060819, 0.0049691, 0.0061761, -0.0008013, 0.0008635
6: -0.0019767, 0.0018309, -0.0025842, 0.0022046, -0.0031794, 0.0034262
7: -0.0050502, 0.0001354, -0.0055591, 0.0009628, -0.0046662, 0.0043301
8: 0.9856564, 0.9893093, 0.9852980, 0.9898920, -0.0032869, 0.0030502
9: -0.0061829, -0.0028671, -0.0067120, -0.0025417, -0.0027688, 0.0029837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014680, upper bound: 0.0016461
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014559, upper bound: 0.0016461
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0034007, 0.0051621, 0.0033086, 0.0052684, -0.0014137, 0.0014063
1: 0.0018136, 0.0020681, 0.0018003, 0.0020834, -0.0002042, 0.0002032
2: 0.0115058, 0.0124797, 0.0114471, 0.0125306, -0.0007775, 0.0007816
3: -0.0027806, -0.0017734, -0.0028413, -0.0017207, -0.0008042, 0.0008084
4: -0.0021172, -0.0010268, -0.0021742, -0.0009610, -0.0008751, 0.0008705
5: 0.0050851, 0.0061169, 0.0050228, 0.0061709, -0.0008238, 0.0008282
6: -0.0021242, 0.0019698, -0.0023713, 0.0021838, -0.0032687, 0.0032859
7: -0.0052394, 0.0003363, -0.0055309, 0.0006727, -0.0044752, 0.0044517
8: 0.9855232, 0.9894508, 0.9853178, 0.9896877, -0.0031524, 0.0031359
9: -0.0063114, -0.0027462, -0.0065265, -0.0025598, -0.0028465, 0.0028615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0015946
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0015946
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0034007, 0.0051621, 0.0032997, 0.0053600, -0.0015477, 0.0014629
1: 0.0018136, 0.0020681, 0.0017990, 0.0020967, -0.0002236, 0.0002113
2: 0.0115058, 0.0124797, 0.0113964, 0.0125355, -0.0008088, 0.0008557
3: -0.0027806, -0.0017734, -0.0028937, -0.0017156, -0.0008365, 0.0008850
4: -0.0021172, -0.0010268, -0.0021797, -0.0009043, -0.0009581, 0.0009056
5: 0.0050851, 0.0061169, 0.0049691, 0.0061761, -0.0008570, 0.0009067
6: -0.0021242, 0.0019698, -0.0025842, 0.0022046, -0.0034002, 0.0035973
7: -0.0052394, 0.0003363, -0.0055591, 0.0009628, -0.0048993, 0.0046307
8: 0.9855232, 0.9894508, 0.9852980, 0.9898920, -0.0034511, 0.0032620
9: -0.0063114, -0.0027462, -0.0067120, -0.0025417, -0.0029610, 0.0031327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0016272
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0016272
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034555, 0.0051882, 0.0033086, 0.0052684, -0.0014136, 0.0014689
1: 0.0018215, 0.0020719, 0.0018003, 0.0020834, -0.0002042, 0.0002122
2: 0.0114914, 0.0124494, 0.0114471, 0.0125306, -0.0008121, 0.0007815
3: -0.0027955, -0.0018047, -0.0028413, -0.0017207, -0.0008399, 0.0008083
4: -0.0020833, -0.0010107, -0.0021742, -0.0009610, -0.0008750, 0.0009093
5: 0.0050698, 0.0060848, 0.0050228, 0.0061709, -0.0008605, 0.0008281
6: -0.0021849, 0.0018424, -0.0023713, 0.0021838, -0.0034141, 0.0032855
7: -0.0050659, 0.0004189, -0.0055309, 0.0006727, -0.0044746, 0.0046497
8: 0.9856453, 0.9895089, 0.9853178, 0.9896877, -0.0031520, 0.0032753
9: -0.0063642, -0.0028571, -0.0065265, -0.0025598, -0.0029731, 0.0028612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014995, upper bound: 0.0016024
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014964, upper bound: 0.0016024
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034555, 0.0051882, 0.0032997, 0.0053600, -0.0013675, 0.0013375
1: 0.0018215, 0.0020719, 0.0017990, 0.0020967, -0.0001976, 0.0001932
2: 0.0114914, 0.0124494, 0.0113964, 0.0125355, -0.0007395, 0.0007560
3: -0.0027955, -0.0018047, -0.0028937, -0.0017156, -0.0007648, 0.0007819
4: -0.0020833, -0.0010107, -0.0021797, -0.0009043, -0.0008465, 0.0008279
5: 0.0050698, 0.0060848, 0.0049691, 0.0061761, -0.0007835, 0.0008011
6: -0.0021849, 0.0018424, -0.0025842, 0.0022046, -0.0031087, 0.0031783
7: -0.0050659, 0.0004189, -0.0055591, 0.0009628, -0.0043286, 0.0042338
8: 0.9856453, 0.9895089, 0.9852980, 0.9898920, -0.0030492, 0.0029823
9: -0.0063642, -0.0028571, -0.0067120, -0.0025417, -0.0027072, 0.0027678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014995, upper bound: 0.0016024
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014964, upper bound: 0.0016024
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033941, 0.0052462, 0.0033086, 0.0052684, -0.0014790, 0.0015448
1: 0.0018126, 0.0020802, 0.0018003, 0.0020834, -0.0002137, 0.0002232
2: 0.0114594, 0.0124834, 0.0114471, 0.0125306, -0.0008541, 0.0008177
3: -0.0028286, -0.0017696, -0.0028413, -0.0017207, -0.0008833, 0.0008457
4: -0.0021213, -0.0009748, -0.0021742, -0.0009610, -0.0009155, 0.0009563
5: 0.0050358, 0.0061208, 0.0050228, 0.0061709, -0.0009050, 0.0008664
6: -0.0023197, 0.0019852, -0.0023713, 0.0021838, -0.0035906, 0.0034376
7: -0.0052604, 0.0006025, -0.0055309, 0.0006727, -0.0046817, 0.0048901
8: 0.9855083, 0.9896383, 0.9853178, 0.9896877, -0.0032979, 0.0034447
9: -0.0064816, -0.0027327, -0.0065265, -0.0025598, -0.0031268, 0.0029936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033941, 0.0052462, 0.0032997, 0.0053600, -0.0014410, 0.0014326
1: 0.0018126, 0.0020802, 0.0017990, 0.0020967, -0.0002082, 0.0002070
2: 0.0114594, 0.0124834, 0.0113964, 0.0125355, -0.0007921, 0.0007967
3: -0.0028286, -0.0017696, -0.0028937, -0.0017156, -0.0008192, 0.0008240
4: -0.0021213, -0.0009748, -0.0021797, -0.0009043, -0.0008920, 0.0008868
5: 0.0050358, 0.0061208, 0.0049691, 0.0061761, -0.0008392, 0.0008441
6: -0.0023197, 0.0019852, -0.0025842, 0.0022046, -0.0033298, 0.0033492
7: -0.0052604, 0.0006025, -0.0055591, 0.0009628, -0.0045614, 0.0045350
8: 0.9855083, 0.9896383, 0.9852980, 0.9898920, -0.0032131, 0.0031945
9: -0.0064816, -0.0027327, -0.0067120, -0.0025417, -0.0028998, 0.0029167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034605, 0.0050987, 0.0032466, 0.0053346, -0.0014360, 0.0013976
1: 0.0018222, 0.0020589, 0.0017913, 0.0020930, -0.0002075, 0.0002019
2: 0.0115409, 0.0124466, 0.0114105, 0.0125649, -0.0007727, 0.0007939
3: -0.0027443, -0.0018076, -0.0028792, -0.0016853, -0.0007992, 0.0008211
4: -0.0020802, -0.0010661, -0.0022126, -0.0009201, -0.0008889, 0.0008651
5: 0.0051223, 0.0060819, 0.0049841, 0.0062072, -0.0008187, 0.0008412
6: -0.0019767, 0.0018309, -0.0025250, 0.0023279, -0.0032485, 0.0033376
7: -0.0050502, 0.0001354, -0.0057272, 0.0008821, -0.0045456, 0.0044241
8: 0.9856564, 0.9893093, 0.9851795, 0.9898352, -0.0032020, 0.0031164
9: -0.0061829, -0.0028671, -0.0066604, -0.0024342, -0.0028289, 0.0029066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014657, upper bound: 0.0016456
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014544, upper bound: 0.0016456
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034605, 0.0050987, 0.0032322, 0.0054305, -0.0015618, 0.0014441
1: 0.0018222, 0.0020589, 0.0017893, 0.0021068, -0.0002256, 0.0002086
2: 0.0115409, 0.0124466, 0.0113575, 0.0125729, -0.0007984, 0.0008635
3: -0.0027443, -0.0018076, -0.0029340, -0.0016770, -0.0008258, 0.0008931
4: -0.0020802, -0.0010661, -0.0022215, -0.0008607, -0.0009668, 0.0008939
5: 0.0051223, 0.0060819, 0.0049279, 0.0062156, -0.0008460, 0.0009149
6: -0.0019767, 0.0018309, -0.0027479, 0.0023614, -0.0033565, 0.0036301
7: -0.0050502, 0.0001354, -0.0057728, 0.0011857, -0.0049439, 0.0045713
8: 0.9856564, 0.9893093, 0.9851474, 0.9900490, -0.0034826, 0.0032201
9: -0.0061829, -0.0028671, -0.0068545, -0.0024051, -0.0029230, 0.0031613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014657, upper bound: 0.0016916
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014544, upper bound: 0.0016916
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0034007, 0.0051621, 0.0032466, 0.0053346, -0.0013800, 0.0013493
1: 0.0018136, 0.0020681, 0.0017913, 0.0020930, -0.0001994, 0.0001949
2: 0.0115058, 0.0124797, 0.0114105, 0.0125649, -0.0007460, 0.0007629
3: -0.0027806, -0.0017734, -0.0028792, -0.0016853, -0.0007715, 0.0007891
4: -0.0021172, -0.0010268, -0.0022126, -0.0009201, -0.0008542, 0.0008352
5: 0.0050851, 0.0061169, 0.0049841, 0.0062072, -0.0007904, 0.0008084
6: -0.0021242, 0.0019698, -0.0025250, 0.0023279, -0.0031362, 0.0032074
7: -0.0052394, 0.0003363, -0.0057272, 0.0008821, -0.0043682, 0.0042712
8: 0.9855232, 0.9894508, 0.9851795, 0.9898352, -0.0030770, 0.0030087
9: -0.0063114, -0.0027462, -0.0066604, -0.0024342, -0.0027311, 0.0027931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0015946
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0015946
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0034007, 0.0051621, 0.0032322, 0.0054305, -0.0015103, 0.0014056
1: 0.0018136, 0.0020681, 0.0017893, 0.0021068, -0.0002182, 0.0002031
2: 0.0115058, 0.0124797, 0.0113575, 0.0125729, -0.0007771, 0.0008350
3: -0.0027806, -0.0017734, -0.0029340, -0.0016770, -0.0008037, 0.0008636
4: -0.0021172, -0.0010268, -0.0022215, -0.0008607, -0.0009349, 0.0008701
5: 0.0050851, 0.0061169, 0.0049279, 0.0062156, -0.0008234, 0.0008847
6: -0.0021242, 0.0019698, -0.0027479, 0.0023614, -0.0032669, 0.0035104
7: -0.0052394, 0.0003363, -0.0057728, 0.0011857, -0.0047809, 0.0044493
8: 0.9855232, 0.9894508, 0.9851474, 0.9900490, -0.0033678, 0.0031342
9: -0.0063114, -0.0027462, -0.0068545, -0.0024051, -0.0028450, 0.0030570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0016272
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0016272
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034555, 0.0051882, 0.0032466, 0.0053346, -0.0015095, 0.0015551
1: 0.0018215, 0.0020719, 0.0017913, 0.0020930, -0.0002181, 0.0002247
2: 0.0114914, 0.0124494, 0.0114105, 0.0125649, -0.0008598, 0.0008345
3: -0.0027955, -0.0018047, -0.0028792, -0.0016853, -0.0008892, 0.0008631
4: -0.0020833, -0.0010107, -0.0022126, -0.0009201, -0.0009344, 0.0009626
5: 0.0050698, 0.0060848, 0.0049841, 0.0062072, -0.0009110, 0.0008842
6: -0.0021849, 0.0018424, -0.0025250, 0.0023279, -0.0036145, 0.0035084
7: -0.0050659, 0.0004189, -0.0057272, 0.0008821, -0.0047781, 0.0049227
8: 0.9856453, 0.9895089, 0.9851795, 0.9898352, -0.0033658, 0.0034676
9: -0.0063642, -0.0028571, -0.0066604, -0.0024342, -0.0031477, 0.0030553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014987, upper bound: 0.0016355
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014956, upper bound: 0.0016356
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034555, 0.0051882, 0.0032322, 0.0054305, -0.0014629, 0.0014240
1: 0.0018215, 0.0020719, 0.0017893, 0.0021068, -0.0002113, 0.0002057
2: 0.0114914, 0.0124494, 0.0113575, 0.0125729, -0.0007873, 0.0008088
3: -0.0027955, -0.0018047, -0.0029340, -0.0016770, -0.0008143, 0.0008365
4: -0.0020833, -0.0010107, -0.0022215, -0.0008607, -0.0009056, 0.0008815
5: 0.0050698, 0.0060848, 0.0049279, 0.0062156, -0.0008342, 0.0008570
6: -0.0021849, 0.0018424, -0.0027479, 0.0023614, -0.0033098, 0.0034002
7: -0.0050659, 0.0004189, -0.0057728, 0.0011857, -0.0046308, 0.0045077
8: 0.9856453, 0.9895089, 0.9851474, 0.9900490, -0.0032620, 0.0031753
9: -0.0063642, -0.0028571, -0.0068545, -0.0024051, -0.0028823, 0.0029610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014987, upper bound: 0.0016355
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014956, upper bound: 0.0016356
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033941, 0.0052462, 0.0032466, 0.0053346, -0.0014538, 0.0015057
1: 0.0018126, 0.0020802, 0.0017913, 0.0020930, -0.0002100, 0.0002175
2: 0.0114594, 0.0124834, 0.0114105, 0.0125649, -0.0008325, 0.0008038
3: -0.0028286, -0.0017696, -0.0028792, -0.0016853, -0.0008610, 0.0008313
4: -0.0021213, -0.0009748, -0.0022126, -0.0009201, -0.0008999, 0.0009321
5: 0.0050358, 0.0061208, 0.0049841, 0.0062072, -0.0008821, 0.0008516
6: -0.0023197, 0.0019852, -0.0025250, 0.0023279, -0.0034997, 0.0033790
7: -0.0052604, 0.0006025, -0.0057272, 0.0008821, -0.0046019, 0.0047664
8: 0.9855083, 0.9896383, 0.9851795, 0.9898352, -0.0032417, 0.0033575
9: -0.0064816, -0.0027327, -0.0066604, -0.0024342, -0.0030477, 0.0029426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033941, 0.0052462, 0.0032322, 0.0054305, -0.0014044, 0.0013724
1: 0.0018126, 0.0020802, 0.0017893, 0.0021068, -0.0002029, 0.0001983
2: 0.0114594, 0.0124834, 0.0113575, 0.0125729, -0.0007588, 0.0007765
3: -0.0028286, -0.0017696, -0.0029340, -0.0016770, -0.0007847, 0.0008030
4: -0.0021213, -0.0009748, -0.0022215, -0.0008607, -0.0008693, 0.0008495
5: 0.0050358, 0.0061208, 0.0049279, 0.0062156, -0.0008039, 0.0008227
6: -0.0023197, 0.0019852, -0.0027479, 0.0023614, -0.0031898, 0.0032642
7: -0.0052604, 0.0006025, -0.0057728, 0.0011857, -0.0044456, 0.0043442
8: 0.9855083, 0.9896383, 0.9851474, 0.9900490, -0.0031315, 0.0030602
9: -0.0064816, -0.0027327, -0.0068545, -0.0024051, -0.0027778, 0.0028426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0034605, 0.0050987, -0.0013114, 0.0013401
1: 0.0018003, 0.0020834, 0.0018222, 0.0020589, -0.0001895, 0.0001936
2: 0.0114471, 0.0125306, 0.0115409, 0.0124466, -0.0007409, 0.0007250
3: -0.0028413, -0.0017207, -0.0027443, -0.0018076, -0.0007663, 0.0007499
4: -0.0021742, -0.0009610, -0.0020802, -0.0010661, -0.0008118, 0.0008295
5: 0.0050228, 0.0061709, 0.0051223, 0.0060819, -0.0007850, 0.0007682
6: -0.0023713, 0.0021838, -0.0019767, 0.0018309, -0.0031148, 0.0030480
7: -0.0055309, 0.0006727, -0.0050502, 0.0001354, -0.0041511, 0.0042421
8: 0.9853178, 0.9896877, 0.9856564, 0.9893093, -0.0029241, 0.0029882
9: -0.0065265, -0.0025598, -0.0061829, -0.0028671, -0.0027125, 0.0026543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016018, upper bound: 0.0014562
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016024, upper bound: 0.0014562
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0034555, 0.0051882, -0.0014689, 0.0014136
1: 0.0018003, 0.0020834, 0.0018215, 0.0020719, -0.0002122, 0.0002042
2: 0.0114471, 0.0125306, 0.0114914, 0.0124494, -0.0007815, 0.0008121
3: -0.0028413, -0.0017207, -0.0027955, -0.0018047, -0.0008083, 0.0008399
4: -0.0021742, -0.0009610, -0.0020833, -0.0010107, -0.0009093, 0.0008750
5: 0.0050228, 0.0061709, 0.0050698, 0.0060848, -0.0008281, 0.0008605
6: -0.0023713, 0.0021838, -0.0021849, 0.0018424, -0.0032855, 0.0034141
7: -0.0055309, 0.0006727, -0.0050659, 0.0004189, -0.0046497, 0.0044746
8: 0.9853178, 0.9896877, 0.9856453, 0.9895089, -0.0032753, 0.0031520
9: -0.0065265, -0.0025598, -0.0063642, -0.0028571, -0.0028612, 0.0029731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016018, upper bound: 0.0014964
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016024, upper bound: 0.0014964
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0034605, 0.0050987, -0.0013976, 0.0014360
1: 0.0017913, 0.0020930, 0.0018222, 0.0020589, -0.0002019, 0.0002075
2: 0.0114105, 0.0125649, 0.0115409, 0.0124466, -0.0007939, 0.0007727
3: -0.0028792, -0.0016853, -0.0027443, -0.0018076, -0.0008211, 0.0007992
4: -0.0022126, -0.0009201, -0.0020802, -0.0010661, -0.0008651, 0.0008889
5: 0.0049841, 0.0062072, 0.0051223, 0.0060819, -0.0008412, 0.0008187
6: -0.0025250, 0.0023279, -0.0019767, 0.0018309, -0.0033376, 0.0032485
7: -0.0057272, 0.0008821, -0.0050502, 0.0001354, -0.0044241, 0.0045456
8: 0.9851795, 0.9898352, 0.9856564, 0.9893093, -0.0031164, 0.0032020
9: -0.0066604, -0.0024342, -0.0061829, -0.0028671, -0.0029066, 0.0028289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014546
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014547
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0034555, 0.0051882, -0.0015551, 0.0015095
1: 0.0017913, 0.0020930, 0.0018215, 0.0020719, -0.0002247, 0.0002181
2: 0.0114105, 0.0125649, 0.0114914, 0.0124494, -0.0008345, 0.0008598
3: -0.0028792, -0.0016853, -0.0027955, -0.0018047, -0.0008631, 0.0008892
4: -0.0022126, -0.0009201, -0.0020833, -0.0010107, -0.0009626, 0.0009344
5: 0.0049841, 0.0062072, 0.0050698, 0.0060848, -0.0008842, 0.0009110
6: -0.0025250, 0.0023279, -0.0021849, 0.0018424, -0.0035084, 0.0036145
7: -0.0057272, 0.0008821, -0.0050659, 0.0004189, -0.0049227, 0.0047781
8: 0.9851795, 0.9898352, 0.9856453, 0.9895089, -0.0034676, 0.0033658
9: -0.0066604, -0.0024342, -0.0063642, -0.0028571, -0.0030553, 0.0031477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014956
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014956
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0033086, 0.0052684, -0.0012336, 0.0012336
1: 0.0018003, 0.0020834, 0.0018003, 0.0020834, -0.0001782, 0.0001782
2: 0.0114471, 0.0125306, 0.0114471, 0.0125306, -0.0006820, 0.0006820
3: -0.0028413, -0.0017207, -0.0028413, -0.0017207, -0.0007054, 0.0007054
4: -0.0021742, -0.0009610, -0.0021742, -0.0009610, -0.0007636, 0.0007636
5: 0.0050228, 0.0061709, 0.0050228, 0.0061709, -0.0007227, 0.0007227
6: -0.0023713, 0.0021838, -0.0023713, 0.0021838, -0.0028673, 0.0028673
7: -0.0055309, 0.0006727, -0.0055309, 0.0006727, -0.0039050, 0.0039050
8: 0.9853178, 0.9896877, 0.9853178, 0.9896877, -0.0027508, 0.0027508
9: -0.0065265, -0.0025598, -0.0065265, -0.0025598, -0.0024970, 0.0024970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016027, upper bound: 0.0014610
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016032, upper bound: 0.0014610
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0032997, 0.0053600, -0.0013857, 0.0013036
1: 0.0018003, 0.0020834, 0.0017990, 0.0020967, -0.0002002, 0.0001883
2: 0.0114471, 0.0125306, 0.0113964, 0.0125355, -0.0007207, 0.0007661
3: -0.0028413, -0.0017207, -0.0028937, -0.0017156, -0.0007454, 0.0007924
4: -0.0021742, -0.0009610, -0.0021797, -0.0009043, -0.0008578, 0.0008069
5: 0.0050228, 0.0061709, 0.0049691, 0.0061761, -0.0007636, 0.0008118
6: -0.0023713, 0.0021838, -0.0025842, 0.0022046, -0.0030298, 0.0032208
7: -0.0055309, 0.0006727, -0.0055591, 0.0009628, -0.0043865, 0.0041263
8: 0.9853178, 0.9896877, 0.9852980, 0.9898920, -0.0030900, 0.0029067
9: -0.0065265, -0.0025598, -0.0067120, -0.0025417, -0.0026385, 0.0028049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016027, upper bound: 0.0015012
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016032, upper bound: 0.0015012
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0033086, 0.0052684, -0.0013119, 0.0013350
1: 0.0017913, 0.0020930, 0.0018003, 0.0020834, -0.0001895, 0.0001929
2: 0.0114105, 0.0125649, 0.0114471, 0.0125306, -0.0007381, 0.0007253
3: -0.0028792, -0.0016853, -0.0028413, -0.0017207, -0.0007634, 0.0007502
4: -0.0022126, -0.0009201, -0.0021742, -0.0009610, -0.0008121, 0.0008264
5: 0.0049841, 0.0062072, 0.0050228, 0.0061709, -0.0007821, 0.0007685
6: -0.0025250, 0.0023279, -0.0023713, 0.0021838, -0.0031030, 0.0030493
7: -0.0057272, 0.0008821, -0.0055309, 0.0006727, -0.0041528, 0.0042260
8: 0.9851795, 0.9898352, 0.9853178, 0.9896877, -0.0029253, 0.0029769
9: -0.0066604, -0.0024342, -0.0065265, -0.0025598, -0.0027022, 0.0026554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0014600
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0014600
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0032997, 0.0053600, -0.0014640, 0.0014050
1: 0.0017913, 0.0020930, 0.0017990, 0.0020967, -0.0002115, 0.0002030
2: 0.0114105, 0.0125649, 0.0113964, 0.0125355, -0.0007768, 0.0008094
3: -0.0028792, -0.0016853, -0.0028937, -0.0017156, -0.0008034, 0.0008371
4: -0.0022126, -0.0009201, -0.0021797, -0.0009043, -0.0009063, 0.0008697
5: 0.0049841, 0.0062072, 0.0049691, 0.0061761, -0.0008230, 0.0008576
6: -0.0025250, 0.0023279, -0.0025842, 0.0022046, -0.0032655, 0.0034028
7: -0.0057272, 0.0008821, -0.0055591, 0.0009628, -0.0046343, 0.0044473
8: 0.9851795, 0.9898352, 0.9852980, 0.9898920, -0.0032645, 0.0031328
9: -0.0066604, -0.0024342, -0.0067120, -0.0025417, -0.0028438, 0.0029633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0015005
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0015005
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0034605, 0.0050987, -0.0013679, 0.0014741
1: 0.0017990, 0.0020967, 0.0018222, 0.0020589, -0.0001976, 0.0002130
2: 0.0113964, 0.0125355, 0.0115409, 0.0124466, -0.0008150, 0.0007563
3: -0.0028937, -0.0017156, -0.0027443, -0.0018076, -0.0008429, 0.0007822
4: -0.0021797, -0.0009043, -0.0020802, -0.0010661, -0.0008468, 0.0009125
5: 0.0049691, 0.0061761, 0.0051223, 0.0060819, -0.0008635, 0.0008013
6: -0.0025842, 0.0022046, -0.0019767, 0.0018309, -0.0034262, 0.0031794
7: -0.0055591, 0.0009628, -0.0050502, 0.0001354, -0.0043301, 0.0046662
8: 0.9852980, 0.9898920, 0.9856564, 0.9893093, -0.0030502, 0.0032869
9: -0.0067120, -0.0025417, -0.0061829, -0.0028671, -0.0029837, 0.0027688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016394, upper bound: 0.0014559
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016461, upper bound: 0.0014559
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0034555, 0.0051882, -0.0013375, 0.0013675
1: 0.0017990, 0.0020967, 0.0018215, 0.0020719, -0.0001932, 0.0001976
2: 0.0113964, 0.0125355, 0.0114914, 0.0124494, -0.0007560, 0.0007395
3: -0.0028937, -0.0017156, -0.0027955, -0.0018047, -0.0007819, 0.0007648
4: -0.0021797, -0.0009043, -0.0020833, -0.0010107, -0.0008279, 0.0008465
5: 0.0049691, 0.0061761, 0.0050698, 0.0060848, -0.0008011, 0.0007835
6: -0.0025842, 0.0022046, -0.0021849, 0.0018424, -0.0031783, 0.0031087
7: -0.0055591, 0.0009628, -0.0050659, 0.0004189, -0.0042338, 0.0043286
8: 0.9852980, 0.9898920, 0.9856453, 0.9895089, -0.0029823, 0.0030492
9: -0.0067120, -0.0025417, -0.0063642, -0.0028571, -0.0027678, 0.0027072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016394, upper bound: 0.0014559
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016461, upper bound: 0.0014559
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0034605, 0.0050987, -0.0014441, 0.0015618
1: 0.0017893, 0.0021068, 0.0018222, 0.0020589, -0.0002086, 0.0002256
2: 0.0113575, 0.0125729, 0.0115409, 0.0124466, -0.0008635, 0.0007984
3: -0.0029340, -0.0016770, -0.0027443, -0.0018076, -0.0008931, 0.0008258
4: -0.0022215, -0.0008607, -0.0020802, -0.0010661, -0.0008939, 0.0009668
5: 0.0049279, 0.0062156, 0.0051223, 0.0060819, -0.0009149, 0.0008460
6: -0.0027479, 0.0023614, -0.0019767, 0.0018309, -0.0036301, 0.0033565
7: -0.0057728, 0.0011857, -0.0050502, 0.0001354, -0.0045713, 0.0049439
8: 0.9851474, 0.9900490, 0.9856564, 0.9893093, -0.0032201, 0.0034826
9: -0.0068545, -0.0024051, -0.0061829, -0.0028671, -0.0031613, 0.0029230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0034555, 0.0051882, -0.0014240, 0.0014629
1: 0.0017893, 0.0021068, 0.0018215, 0.0020719, -0.0002057, 0.0002113
2: 0.0113575, 0.0125729, 0.0114914, 0.0124494, -0.0008088, 0.0007873
3: -0.0029340, -0.0016770, -0.0027955, -0.0018047, -0.0008365, 0.0008143
4: -0.0022215, -0.0008607, -0.0020833, -0.0010107, -0.0008815, 0.0009056
5: 0.0049279, 0.0062156, 0.0050698, 0.0060848, -0.0008570, 0.0008342
6: -0.0027479, 0.0023614, -0.0021849, 0.0018424, -0.0034002, 0.0033098
7: -0.0057728, 0.0011857, -0.0050659, 0.0004189, -0.0045076, 0.0046308
8: 0.9851474, 0.9900490, 0.9856453, 0.9895089, -0.0031753, 0.0032620
9: -0.0068545, -0.0024051, -0.0063642, -0.0028571, -0.0029610, 0.0028823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0033086, 0.0052684, -0.0013036, 0.0013857
1: 0.0017990, 0.0020967, 0.0018003, 0.0020834, -0.0001883, 0.0002002
2: 0.0113964, 0.0125355, 0.0114471, 0.0125306, -0.0007661, 0.0007207
3: -0.0028937, -0.0017156, -0.0028413, -0.0017207, -0.0007924, 0.0007454
4: -0.0021797, -0.0009043, -0.0021742, -0.0009610, -0.0008069, 0.0008578
5: 0.0049691, 0.0061761, 0.0050228, 0.0061709, -0.0008118, 0.0007636
6: -0.0025842, 0.0022046, -0.0023713, 0.0021838, -0.0032208, 0.0030298
7: -0.0055591, 0.0009628, -0.0055309, 0.0006727, -0.0041263, 0.0043865
8: 0.9852980, 0.9898920, 0.9853178, 0.9896877, -0.0029067, 0.0030900
9: -0.0067120, -0.0025417, -0.0065265, -0.0025598, -0.0028049, 0.0026385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016399, upper bound: 0.0014607
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016467, upper bound: 0.0014607
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0032997, 0.0053600, -0.0012575, 0.0012575
1: 0.0017990, 0.0020967, 0.0017990, 0.0020967, -0.0001817, 0.0001817
2: 0.0113964, 0.0125355, 0.0113964, 0.0125355, -0.0006952, 0.0006952
3: -0.0028937, -0.0017156, -0.0028937, -0.0017156, -0.0007190, 0.0007190
4: -0.0021797, -0.0009043, -0.0021797, -0.0009043, -0.0007784, 0.0007784
5: 0.0049691, 0.0061761, 0.0049691, 0.0061761, -0.0007366, 0.0007366
6: -0.0025842, 0.0022046, -0.0025842, 0.0022046, -0.0029227, 0.0029227
7: -0.0055591, 0.0009628, -0.0055591, 0.0009628, -0.0039804, 0.0039804
8: 0.9852980, 0.9898920, 0.9852980, 0.9898920, -0.0028039, 0.0028039
9: -0.0067120, -0.0025417, -0.0067120, -0.0025417, -0.0025452, 0.0025452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016399, upper bound: 0.0014607
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016467, upper bound: 0.0014607
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0033086, 0.0052684, -0.0013712, 0.0014705
1: 0.0017893, 0.0021068, 0.0018003, 0.0020834, -0.0001981, 0.0002124
2: 0.0113575, 0.0125729, 0.0114471, 0.0125306, -0.0008130, 0.0007581
3: -0.0029340, -0.0016770, -0.0028413, -0.0017207, -0.0008409, 0.0007841
4: -0.0022215, -0.0008607, -0.0021742, -0.0009610, -0.0008488, 0.0009103
5: 0.0049279, 0.0062156, 0.0050228, 0.0061709, -0.0008614, 0.0008033
6: -0.0027479, 0.0023614, -0.0023713, 0.0021838, -0.0034179, 0.0031872
7: -0.0057728, 0.0011857, -0.0055309, 0.0006727, -0.0043406, 0.0046549
8: 0.9851474, 0.9900490, 0.9853178, 0.9896877, -0.0030576, 0.0032790
9: -0.0068545, -0.0024051, -0.0065265, -0.0025598, -0.0029764, 0.0027755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0032997, 0.0053600, -0.0013354, 0.0013584
1: 0.0017893, 0.0021068, 0.0017990, 0.0020967, -0.0001929, 0.0001962
2: 0.0113575, 0.0125729, 0.0113964, 0.0125355, -0.0007510, 0.0007383
3: -0.0029340, -0.0016770, -0.0028937, -0.0017156, -0.0007767, 0.0007636
4: -0.0022215, -0.0008607, -0.0021797, -0.0009043, -0.0008266, 0.0008409
5: 0.0049279, 0.0062156, 0.0049691, 0.0061761, -0.0007957, 0.0007823
6: -0.0027479, 0.0023614, -0.0025842, 0.0022046, -0.0031572, 0.0031038
7: -0.0057728, 0.0011857, -0.0055591, 0.0009628, -0.0042271, 0.0042999
8: 0.9851474, 0.9900490, 0.9852980, 0.9898920, -0.0029776, 0.0030289
9: -0.0068545, -0.0024051, -0.0067120, -0.0025417, -0.0027495, 0.0027029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0034007, 0.0051621, -0.0014063, 0.0014137
1: 0.0018003, 0.0020834, 0.0018136, 0.0020681, -0.0002032, 0.0002042
2: 0.0114471, 0.0125306, 0.0115058, 0.0124797, -0.0007816, 0.0007775
3: -0.0028413, -0.0017207, -0.0027806, -0.0017734, -0.0008084, 0.0008042
4: -0.0021742, -0.0009610, -0.0021172, -0.0010268, -0.0008705, 0.0008751
5: 0.0050228, 0.0061709, 0.0050851, 0.0061169, -0.0008282, 0.0008238
6: -0.0023713, 0.0021838, -0.0021242, 0.0019698, -0.0032859, 0.0032687
7: -0.0055309, 0.0006727, -0.0052394, 0.0003363, -0.0044517, 0.0044752
8: 0.9853178, 0.9896877, 0.9855232, 0.9894508, -0.0031359, 0.0031524
9: -0.0065265, -0.0025598, -0.0063114, -0.0027462, -0.0028615, 0.0028465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015959, upper bound: 0.0014858
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015935, upper bound: 0.0014858
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0033941, 0.0052462, -0.0015448, 0.0014790
1: 0.0018003, 0.0020834, 0.0018126, 0.0020802, -0.0002232, 0.0002137
2: 0.0114471, 0.0125306, 0.0114594, 0.0124834, -0.0008177, 0.0008541
3: -0.0028413, -0.0017207, -0.0028286, -0.0017696, -0.0008457, 0.0008833
4: -0.0021742, -0.0009610, -0.0021213, -0.0009748, -0.0009563, 0.0009155
5: 0.0050228, 0.0061709, 0.0050358, 0.0061208, -0.0008664, 0.0009050
6: -0.0023713, 0.0021838, -0.0023197, 0.0019852, -0.0034376, 0.0035906
7: -0.0055309, 0.0006727, -0.0052604, 0.0006025, -0.0048901, 0.0046817
8: 0.9853178, 0.9896877, 0.9855083, 0.9896383, -0.0034447, 0.0032979
9: -0.0065265, -0.0025598, -0.0064816, -0.0027327, -0.0029936, 0.0031268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015959, upper bound: 0.0015208
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015935, upper bound: 0.0015208
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0034007, 0.0051621, -0.0013493, 0.0013800
1: 0.0017913, 0.0020930, 0.0018136, 0.0020681, -0.0001949, 0.0001994
2: 0.0114105, 0.0125649, 0.0115058, 0.0124797, -0.0007629, 0.0007460
3: -0.0028792, -0.0016853, -0.0027806, -0.0017734, -0.0007891, 0.0007715
4: -0.0022126, -0.0009201, -0.0021172, -0.0010268, -0.0008352, 0.0008542
5: 0.0049841, 0.0062072, 0.0050851, 0.0061169, -0.0008084, 0.0007904
6: -0.0025250, 0.0023279, -0.0021242, 0.0019698, -0.0032074, 0.0031362
7: -0.0057272, 0.0008821, -0.0052394, 0.0003363, -0.0042712, 0.0043682
8: 0.9851795, 0.9898352, 0.9855232, 0.9894508, -0.0030087, 0.0030770
9: -0.0066604, -0.0024342, -0.0063114, -0.0027462, -0.0027931, 0.0027311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014547
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014547
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0033941, 0.0052462, -0.0015057, 0.0014538
1: 0.0017913, 0.0020930, 0.0018126, 0.0020802, -0.0002175, 0.0002100
2: 0.0114105, 0.0125649, 0.0114594, 0.0124834, -0.0008038, 0.0008325
3: -0.0028792, -0.0016853, -0.0028286, -0.0017696, -0.0008313, 0.0008610
4: -0.0022126, -0.0009201, -0.0021213, -0.0009748, -0.0009321, 0.0008999
5: 0.0049841, 0.0062072, 0.0050358, 0.0061208, -0.0008516, 0.0008821
6: -0.0025250, 0.0023279, -0.0023197, 0.0019852, -0.0033790, 0.0034997
7: -0.0057272, 0.0008821, -0.0052604, 0.0006025, -0.0047664, 0.0046019
8: 0.9851795, 0.9898352, 0.9855083, 0.9896383, -0.0033575, 0.0032417
9: -0.0066604, -0.0024342, -0.0064816, -0.0027327, -0.0029426, 0.0030477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014956
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014956
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0032466, 0.0053346, -0.0013350, 0.0013119
1: 0.0018003, 0.0020834, 0.0017913, 0.0020930, -0.0001929, 0.0001895
2: 0.0114471, 0.0125306, 0.0114105, 0.0125649, -0.0007253, 0.0007381
3: -0.0028413, -0.0017207, -0.0028792, -0.0016853, -0.0007502, 0.0007634
4: -0.0021742, -0.0009610, -0.0022126, -0.0009201, -0.0008264, 0.0008121
5: 0.0050228, 0.0061709, 0.0049841, 0.0062072, -0.0007685, 0.0007821
6: -0.0023713, 0.0021838, -0.0025250, 0.0023279, -0.0030493, 0.0031030
7: -0.0055309, 0.0006727, -0.0057272, 0.0008821, -0.0042260, 0.0041528
8: 0.9853178, 0.9896877, 0.9851795, 0.9898352, -0.0029769, 0.0029253
9: -0.0065265, -0.0025598, -0.0066604, -0.0024342, -0.0026554, 0.0027022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015965, upper bound: 0.0014919
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015941, upper bound: 0.0014919
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0052684, 0.0032322, 0.0054305, -0.0014705, 0.0013712
1: 0.0018003, 0.0020834, 0.0017893, 0.0021068, -0.0002124, 0.0001981
2: 0.0114471, 0.0125306, 0.0113575, 0.0125729, -0.0007581, 0.0008130
3: -0.0028413, -0.0017207, -0.0029340, -0.0016770, -0.0007841, 0.0008409
4: -0.0021742, -0.0009610, -0.0022215, -0.0008607, -0.0009103, 0.0008488
5: 0.0050228, 0.0061709, 0.0049279, 0.0062156, -0.0008033, 0.0008614
6: -0.0023713, 0.0021838, -0.0027479, 0.0023614, -0.0031871, 0.0034179
7: -0.0055309, 0.0006727, -0.0057728, 0.0011857, -0.0046549, 0.0043406
8: 0.9853178, 0.9896877, 0.9851474, 0.9900490, -0.0032790, 0.0030576
9: -0.0065265, -0.0025598, -0.0068545, -0.0024051, -0.0027755, 0.0029764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015965, upper bound: 0.0015265
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015941, upper bound: 0.0015265
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0032466, 0.0053346, -0.0012723, 0.0012723
1: 0.0017913, 0.0020930, 0.0017913, 0.0020930, -0.0001838, 0.0001838
2: 0.0114105, 0.0125649, 0.0114105, 0.0125649, -0.0007034, 0.0007034
3: -0.0028792, -0.0016853, -0.0028792, -0.0016853, -0.0007275, 0.0007275
4: -0.0022126, -0.0009201, -0.0022126, -0.0009201, -0.0007876, 0.0007876
5: 0.0049841, 0.0062072, 0.0049841, 0.0062072, -0.0007453, 0.0007453
6: -0.0025250, 0.0023279, -0.0025250, 0.0023279, -0.0029573, 0.0029573
7: -0.0057272, 0.0008821, -0.0057272, 0.0008821, -0.0040275, 0.0040275
8: 0.9851795, 0.9898352, 0.9851795, 0.9898352, -0.0028371, 0.0028371
9: -0.0066604, -0.0024342, -0.0066604, -0.0024342, -0.0025753, 0.0025753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0014600
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0014600
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032466, 0.0053346, 0.0032322, 0.0054305, -0.0014245, 0.0013423
1: 0.0017913, 0.0020930, 0.0017893, 0.0021068, -0.0002058, 0.0001939
2: 0.0114105, 0.0125649, 0.0113575, 0.0125729, -0.0007421, 0.0007875
3: -0.0028792, -0.0016853, -0.0029340, -0.0016770, -0.0007676, 0.0008145
4: -0.0022126, -0.0009201, -0.0022215, -0.0008607, -0.0008818, 0.0008309
5: 0.0049841, 0.0062072, 0.0049279, 0.0062156, -0.0007863, 0.0008344
6: -0.0025250, 0.0023279, -0.0027479, 0.0023614, -0.0031199, 0.0033108
7: -0.0057272, 0.0008821, -0.0057728, 0.0011857, -0.0045091, 0.0042491
8: 0.9851795, 0.9898352, 0.9851474, 0.9900490, -0.0031763, 0.0029931
9: -0.0066604, -0.0024342, -0.0068545, -0.0024051, -0.0027170, 0.0028832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0015005
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0015005
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0034007, 0.0051621, -0.0014629, 0.0015477
1: 0.0017990, 0.0020967, 0.0018136, 0.0020681, -0.0002113, 0.0002236
2: 0.0113964, 0.0125355, 0.0115058, 0.0124797, -0.0008557, 0.0008088
3: -0.0028937, -0.0017156, -0.0027806, -0.0017734, -0.0008850, 0.0008365
4: -0.0021797, -0.0009043, -0.0021172, -0.0010268, -0.0009056, 0.0009581
5: 0.0049691, 0.0061761, 0.0050851, 0.0061169, -0.0009067, 0.0008570
6: -0.0025842, 0.0022046, -0.0021242, 0.0019698, -0.0035973, 0.0034002
7: -0.0055591, 0.0009628, -0.0052394, 0.0003363, -0.0046307, 0.0048993
8: 0.9852980, 0.9898920, 0.9855232, 0.9894508, -0.0032620, 0.0034511
9: -0.0067120, -0.0025417, -0.0063114, -0.0027462, -0.0031327, 0.0029610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016239, upper bound: 0.0014812
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016272, upper bound: 0.0014812
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0033941, 0.0052462, -0.0014326, 0.0014410
1: 0.0017990, 0.0020967, 0.0018126, 0.0020802, -0.0002070, 0.0002082
2: 0.0113964, 0.0125355, 0.0114594, 0.0124834, -0.0007967, 0.0007921
3: -0.0028937, -0.0017156, -0.0028286, -0.0017696, -0.0008240, 0.0008192
4: -0.0021797, -0.0009043, -0.0021213, -0.0009748, -0.0008868, 0.0008920
5: 0.0049691, 0.0061761, 0.0050358, 0.0061208, -0.0008441, 0.0008392
6: -0.0025842, 0.0022046, -0.0023197, 0.0019852, -0.0033492, 0.0033298
7: -0.0055591, 0.0009628, -0.0052604, 0.0006025, -0.0045350, 0.0045614
8: 0.9852980, 0.9898920, 0.9855083, 0.9896383, -0.0031945, 0.0032131
9: -0.0067120, -0.0025417, -0.0064816, -0.0027327, -0.0029167, 0.0028998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016239, upper bound: 0.0014812
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016272, upper bound: 0.0014812
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0034007, 0.0051621, -0.0014056, 0.0015103
1: 0.0017893, 0.0021068, 0.0018136, 0.0020681, -0.0002031, 0.0002182
2: 0.0113575, 0.0125729, 0.0115058, 0.0124797, -0.0008350, 0.0007771
3: -0.0029340, -0.0016770, -0.0027806, -0.0017734, -0.0008636, 0.0008037
4: -0.0022215, -0.0008607, -0.0021172, -0.0010268, -0.0008701, 0.0009349
5: 0.0049279, 0.0062156, 0.0050851, 0.0061169, -0.0008847, 0.0008234
6: -0.0027479, 0.0023614, -0.0021242, 0.0019698, -0.0035104, 0.0032669
7: -0.0057728, 0.0011857, -0.0052394, 0.0003363, -0.0044493, 0.0047809
8: 0.9851474, 0.9900490, 0.9855232, 0.9894508, -0.0031342, 0.0033678
9: -0.0068545, -0.0024051, -0.0063114, -0.0027462, -0.0030570, 0.0028450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0033941, 0.0052462, -0.0013724, 0.0014044
1: 0.0017893, 0.0021068, 0.0018126, 0.0020802, -0.0001983, 0.0002029
2: 0.0113575, 0.0125729, 0.0114594, 0.0124834, -0.0007765, 0.0007588
3: -0.0029340, -0.0016770, -0.0028286, -0.0017696, -0.0008030, 0.0007847
4: -0.0022215, -0.0008607, -0.0021213, -0.0009748, -0.0008495, 0.0008693
5: 0.0049279, 0.0062156, 0.0050358, 0.0061208, -0.0008227, 0.0008039
6: -0.0027479, 0.0023614, -0.0023197, 0.0019852, -0.0032642, 0.0031898
7: -0.0057728, 0.0011857, -0.0052604, 0.0006025, -0.0043442, 0.0044456
8: 0.9851474, 0.9900490, 0.9855083, 0.9896383, -0.0030602, 0.0031315
9: -0.0068545, -0.0024051, -0.0064816, -0.0027327, -0.0028426, 0.0027778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0032466, 0.0053346, -0.0014050, 0.0014640
1: 0.0017990, 0.0020967, 0.0017913, 0.0020930, -0.0002030, 0.0002115
2: 0.0113964, 0.0125355, 0.0114105, 0.0125649, -0.0008094, 0.0007768
3: -0.0028937, -0.0017156, -0.0028792, -0.0016853, -0.0008371, 0.0008034
4: -0.0021797, -0.0009043, -0.0022126, -0.0009201, -0.0008697, 0.0009063
5: 0.0049691, 0.0061761, 0.0049841, 0.0062072, -0.0008576, 0.0008230
6: -0.0025842, 0.0022046, -0.0025250, 0.0023279, -0.0034028, 0.0032655
7: -0.0055591, 0.0009628, -0.0057272, 0.0008821, -0.0044473, 0.0046343
8: 0.9852980, 0.9898920, 0.9851795, 0.9898352, -0.0031328, 0.0032645
9: -0.0067120, -0.0025417, -0.0066604, -0.0024342, -0.0029633, 0.0028438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016248, upper bound: 0.0014875
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016278, upper bound: 0.0014875
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032997, 0.0053600, 0.0032322, 0.0054305, -0.0013584, 0.0013354
1: 0.0017990, 0.0020967, 0.0017893, 0.0021068, -0.0001962, 0.0001929
2: 0.0113964, 0.0125355, 0.0113575, 0.0125729, -0.0007383, 0.0007510
3: -0.0028937, -0.0017156, -0.0029340, -0.0016770, -0.0007636, 0.0007767
4: -0.0021797, -0.0009043, -0.0022215, -0.0008607, -0.0008409, 0.0008266
5: 0.0049691, 0.0061761, 0.0049279, 0.0062156, -0.0007823, 0.0007957
6: -0.0025842, 0.0022046, -0.0027479, 0.0023614, -0.0031038, 0.0031572
7: -0.0055591, 0.0009628, -0.0057728, 0.0011857, -0.0042999, 0.0042271
8: 0.9852980, 0.9898920, 0.9851474, 0.9900490, -0.0030289, 0.0029776
9: -0.0067120, -0.0025417, -0.0068545, -0.0024051, -0.0027029, 0.0027495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016248, upper bound: 0.0014875
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016278, upper bound: 0.0014875
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0032466, 0.0053346, -0.0013423, 0.0014245
1: 0.0017893, 0.0021068, 0.0017913, 0.0020930, -0.0001939, 0.0002058
2: 0.0113575, 0.0125729, 0.0114105, 0.0125649, -0.0007875, 0.0007421
3: -0.0029340, -0.0016770, -0.0028792, -0.0016853, -0.0008145, 0.0007676
4: -0.0022215, -0.0008607, -0.0022126, -0.0009201, -0.0008309, 0.0008818
5: 0.0049279, 0.0062156, 0.0049841, 0.0062072, -0.0008344, 0.0007863
6: -0.0027479, 0.0023614, -0.0025250, 0.0023279, -0.0033108, 0.0031199
7: -0.0057728, 0.0011857, -0.0057272, 0.0008821, -0.0042491, 0.0045091
8: 0.9851474, 0.9900490, 0.9851795, 0.9898352, -0.0029931, 0.0031763
9: -0.0068545, -0.0024051, -0.0066604, -0.0024342, -0.0028832, 0.0027170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032322, 0.0054305, 0.0032322, 0.0054305, -0.0013005, 0.0013005
1: 0.0017893, 0.0021068, 0.0017893, 0.0021068, -0.0001879, 0.0001879
2: 0.0113575, 0.0125729, 0.0113575, 0.0125729, -0.0007190, 0.0007190
3: -0.0029340, -0.0016770, -0.0029340, -0.0016770, -0.0007436, 0.0007436
4: -0.0022215, -0.0008607, -0.0022215, -0.0008607, -0.0008050, 0.0008050
5: 0.0049279, 0.0062156, 0.0049279, 0.0062156, -0.0007618, 0.0007618
6: -0.0027479, 0.0023614, -0.0027479, 0.0023614, -0.0030226, 0.0030226
7: -0.0057728, 0.0011857, -0.0057728, 0.0011857, -0.0041165, 0.0041165
8: 0.9851474, 0.9900490, 0.9851474, 0.9900490, -0.0028998, 0.0028998
9: -0.0068545, -0.0024051, -0.0068545, -0.0024051, -0.0026322, 0.0026322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
time: 0.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.38 seconds
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014680, upper bound: 0.0016087
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014559, upper bound: 0.0016087
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014680, upper bound: 0.0016461
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014559, upper bound: 0.0016461
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0015946
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0015946
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0016272
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0016272
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014995, upper bound: 0.0016024
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014964, upper bound: 0.0016024
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014995, upper bound: 0.0016024
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014964, upper bound: 0.0016024
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014657, upper bound: 0.0016456
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014544, upper bound: 0.0016456
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014657, upper bound: 0.0016916
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014544, upper bound: 0.0016916
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0015946
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0015946
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014905, upper bound: 0.0016272
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014812, upper bound: 0.0016272
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014987, upper bound: 0.0016355
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014956, upper bound: 0.0016356
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014987, upper bound: 0.0016355
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0014956, upper bound: 0.0016356
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015246, upper bound: 0.0015935
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015208, upper bound: 0.0015935
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016018, upper bound: 0.0014562
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016024, upper bound: 0.0014562
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016018, upper bound: 0.0014964
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016024, upper bound: 0.0014964
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014546
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014547
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014956
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014956
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016027, upper bound: 0.0014610
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016032, upper bound: 0.0014610
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016027, upper bound: 0.0015012
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016032, upper bound: 0.0015012
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0014600
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0014600
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0015005
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0015005
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016394, upper bound: 0.0014559
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016461, upper bound: 0.0014559
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016394, upper bound: 0.0014559
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016461, upper bound: 0.0014559
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016399, upper bound: 0.0014607
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016467, upper bound: 0.0014607
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016399, upper bound: 0.0014607
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016467, upper bound: 0.0014607
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015959, upper bound: 0.0014858
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015935, upper bound: 0.0014858
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015959, upper bound: 0.0015208
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015935, upper bound: 0.0015208
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014547
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014547
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016364, upper bound: 0.0014956
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016355, upper bound: 0.0014956
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015965, upper bound: 0.0014919
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015941, upper bound: 0.0014919
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015965, upper bound: 0.0015265
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0015941, upper bound: 0.0015265
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0014600
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0014600
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016376, upper bound: 0.0015005
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016366, upper bound: 0.0015005
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016239, upper bound: 0.0014812
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016272, upper bound: 0.0014812
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016239, upper bound: 0.0014812
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016272, upper bound: 0.0014812
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016859, upper bound: 0.0014544
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016916, upper bound: 0.0014544
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016248, upper bound: 0.0014875
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016278, upper bound: 0.0014875
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016248, upper bound: 0.0014875
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016278, upper bound: 0.0014875
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016867, upper bound: 0.0014597
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0016924, upper bound: 0.0014597

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034881, 0.0050976, 0.0032384, 0.0054302, -0.0015177, 0.0014186
1: 0.0018262, 0.0020588, 0.0017902, 0.0021068, -0.0002193, 0.0002049
2: 0.0115415, 0.0124314, 0.0113577, 0.0125695, -0.0007843, 0.0008391
3: -0.0027437, -0.0018233, -0.0029338, -0.0016805, -0.0008112, 0.0008678
4: -0.0020631, -0.0010668, -0.0022177, -0.0008609, -0.0009395, 0.0008781
5: 0.0051229, 0.0060657, 0.0049281, 0.0062120, -0.0008310, 0.0008890
6: -0.0019742, 0.0017667, -0.0027472, 0.0023471, -0.0032972, 0.0035274
7: -0.0049629, 0.0001320, -0.0057533, 0.0011847, -0.0048041, 0.0044905
8: 0.9857179, 0.9893068, 0.9851612, 0.9900484, -0.0033841, 0.0031632
9: -0.0061808, -0.0029230, -0.0068539, -0.0024175, -0.0028713, 0.0030719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014178, upper bound: 0.0016327
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014178, upper bound: 0.0016476
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0035329, 0.0051470, 0.0032592, 0.0054291, -0.0015407, 0.0014587
1: 0.0018327, 0.0020659, 0.0017932, 0.0021066, -0.0002226, 0.0002107
2: 0.0115142, 0.0124066, 0.0113583, 0.0125579, -0.0008065, 0.0008518
3: -0.0027719, -0.0018490, -0.0029332, -0.0016924, -0.0008341, 0.0008810
4: -0.0020353, -0.0010362, -0.0022048, -0.0008616, -0.0009537, 0.0009030
5: 0.0050939, 0.0060395, 0.0049287, 0.0061998, -0.0008545, 0.0009026
6: -0.0020891, 0.0016625, -0.0027447, 0.0022988, -0.0033904, 0.0035811
7: -0.0048209, 0.0002885, -0.0056875, 0.0011813, -0.0048771, 0.0046175
8: 0.9858178, 0.9894171, 0.9852076, 0.9900460, -0.0034356, 0.0032526
9: -0.0062808, -0.0030137, -0.0068517, -0.0024596, -0.0029525, 0.0031186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014062, upper bound: 0.0016327
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014062, upper bound: 0.0016476
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0034661, 0.0050984, -0.0013938, 0.0015391
1: 0.0017936, 0.0021066, 0.0018230, 0.0020589, -0.0002014, 0.0002224
2: 0.0113583, 0.0125562, 0.0115411, 0.0124436, -0.0008509, 0.0007706
3: -0.0029332, -0.0016942, -0.0027442, -0.0018108, -0.0008801, 0.0007970
4: -0.0022029, -0.0008616, -0.0020767, -0.0010663, -0.0008628, 0.0009527
5: 0.0049287, 0.0061980, 0.0051224, 0.0060786, -0.0009016, 0.0008165
6: -0.0027445, 0.0022915, -0.0019762, 0.0018179, -0.0035772, 0.0032397
7: -0.0056775, 0.0011811, -0.0050325, 0.0001347, -0.0044122, 0.0048719
8: 0.9852145, 0.9900459, 0.9856689, 0.9893087, -0.0031080, 0.0034319
9: -0.0068516, -0.0024660, -0.0061825, -0.0028784, -0.0031152, 0.0028213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0034912, 0.0050977, -0.0014094, 0.0015794
1: 0.0017984, 0.0021152, 0.0018267, 0.0020588, -0.0002036, 0.0002282
2: 0.0113254, 0.0125377, 0.0115415, 0.0124297, -0.0008732, 0.0007792
3: -0.0029672, -0.0017134, -0.0027437, -0.0018251, -0.0009031, 0.0008059
4: -0.0021821, -0.0008247, -0.0020612, -0.0010667, -0.0008724, 0.0009777
5: 0.0048938, 0.0061784, 0.0051228, 0.0060639, -0.0009252, 0.0008256
6: -0.0028830, 0.0022137, -0.0019744, 0.0017595, -0.0036710, 0.0032758
7: -0.0055716, 0.0013697, -0.0049530, 0.0001323, -0.0044614, 0.0049996
8: 0.9852890, 0.9901786, 0.9857249, 0.9893071, -0.0031427, 0.0035218
9: -0.0069722, -0.0025337, -0.0061809, -0.0029293, -0.0031969, 0.0028527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0034611, 0.0051880, -0.0013812, 0.0014386
1: 0.0017936, 0.0021066, 0.0018223, 0.0020718, -0.0001995, 0.0002078
2: 0.0113583, 0.0125562, 0.0114916, 0.0124463, -0.0007954, 0.0007636
3: -0.0029332, -0.0016942, -0.0027954, -0.0018079, -0.0008226, 0.0007898
4: -0.0022029, -0.0008616, -0.0020798, -0.0010108, -0.0008550, 0.0008905
5: 0.0049287, 0.0061980, 0.0050699, 0.0060815, -0.0008427, 0.0008091
6: -0.0027445, 0.0022915, -0.0021843, 0.0018294, -0.0033436, 0.0032102
7: -0.0056775, 0.0011811, -0.0050482, 0.0004182, -0.0043721, 0.0045538
8: 0.9852145, 0.9900459, 0.9856579, 0.9895085, -0.0030798, 0.0032078
9: -0.0068516, -0.0024660, -0.0063637, -0.0028684, -0.0029118, 0.0027956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0034857, 0.0051872, -0.0014060, 0.0014664
1: 0.0017984, 0.0021152, 0.0018259, 0.0020717, -0.0002031, 0.0002119
2: 0.0113254, 0.0125377, 0.0114920, 0.0124327, -0.0008107, 0.0007773
3: -0.0029672, -0.0017134, -0.0027949, -0.0018219, -0.0008385, 0.0008040
4: -0.0021821, -0.0008247, -0.0020646, -0.0010113, -0.0008703, 0.0009077
5: 0.0048938, 0.0061784, 0.0050704, 0.0060672, -0.0008590, 0.0008236
6: -0.0028830, 0.0022137, -0.0021824, 0.0017724, -0.0034083, 0.0032679
7: -0.0055716, 0.0013697, -0.0049705, 0.0004155, -0.0044506, 0.0046418
8: 0.9852890, 0.9901786, 0.9857125, 0.9895066, -0.0031351, 0.0032698
9: -0.0069722, -0.0025337, -0.0063621, -0.0029180, -0.0029681, 0.0028458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0033149, 0.0052681, -0.0013190, 0.0014483
1: 0.0017936, 0.0021066, 0.0018012, 0.0020834, -0.0001906, 0.0002092
2: 0.0113583, 0.0125562, 0.0114473, 0.0125271, -0.0008007, 0.0007293
3: -0.0029332, -0.0016942, -0.0028412, -0.0017243, -0.0008281, 0.0007542
4: -0.0022029, -0.0008616, -0.0021703, -0.0009612, -0.0008165, 0.0008965
5: 0.0049287, 0.0061980, 0.0050230, 0.0061672, -0.0008484, 0.0007727
6: -0.0027445, 0.0022915, -0.0023706, 0.0021691, -0.0033662, 0.0030658
7: -0.0056775, 0.0011811, -0.0055109, 0.0006718, -0.0041754, 0.0045844
8: 0.9852145, 0.9900459, 0.9853318, 0.9896871, -0.0029412, 0.0032294
9: -0.0068516, -0.0024660, -0.0065259, -0.0025725, -0.0029314, 0.0026698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016449, upper bound: 0.0014110
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016449, upper bound: 0.0014150
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0033362, 0.0052671, -0.0013377, 0.0014913
1: 0.0017984, 0.0021152, 0.0018043, 0.0020832, -0.0001933, 0.0002155
2: 0.0113254, 0.0125377, 0.0114478, 0.0125154, -0.0008245, 0.0007396
3: -0.0029672, -0.0017134, -0.0028406, -0.0017365, -0.0008528, 0.0007649
4: -0.0021821, -0.0008247, -0.0021571, -0.0009619, -0.0008281, 0.0009232
5: 0.0048938, 0.0061784, 0.0050236, 0.0061547, -0.0008736, 0.0007836
6: -0.0028830, 0.0022137, -0.0023681, 0.0021197, -0.0034662, 0.0031092
7: -0.0055716, 0.0013697, -0.0054436, 0.0006685, -0.0042345, 0.0047207
8: 0.9852890, 0.9901786, 0.9853793, 0.9896848, -0.0029829, 0.0033254
9: -0.0069722, -0.0025337, -0.0065238, -0.0026156, -0.0030186, 0.0027077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0033059, 0.0053597, -0.0012945, 0.0013334
1: 0.0017936, 0.0021066, 0.0017999, 0.0020966, -0.0001870, 0.0001926
2: 0.0113583, 0.0125562, 0.0113966, 0.0125321, -0.0007372, 0.0007157
3: -0.0029332, -0.0016942, -0.0028936, -0.0017192, -0.0007625, 0.0007402
4: -0.0022029, -0.0008616, -0.0021758, -0.0009045, -0.0008013, 0.0008254
5: 0.0049287, 0.0061980, 0.0049693, 0.0061724, -0.0007811, 0.0007583
6: -0.0027445, 0.0022915, -0.0025835, 0.0021901, -0.0030993, 0.0030087
7: -0.0056775, 0.0011811, -0.0055394, 0.0009618, -0.0040976, 0.0042209
8: 0.9852145, 0.9900459, 0.9853119, 0.9898914, -0.0028864, 0.0029733
9: -0.0068516, -0.0024660, -0.0067114, -0.0025543, -0.0026990, 0.0026201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016449, upper bound: 0.0014110
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014150
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0033273, 0.0053586, -0.0013258, 0.0013603
1: 0.0017984, 0.0021152, 0.0018030, 0.0020965, -0.0001915, 0.0001965
2: 0.0113254, 0.0125377, 0.0113972, 0.0125203, -0.0007520, 0.0007330
3: -0.0029672, -0.0017134, -0.0028929, -0.0017314, -0.0007778, 0.0007581
4: -0.0021821, -0.0008247, -0.0021626, -0.0009052, -0.0008207, 0.0008420
5: 0.0048938, 0.0061784, 0.0049700, 0.0061599, -0.0007968, 0.0007767
6: -0.0028830, 0.0022137, -0.0025809, 0.0021405, -0.0031616, 0.0030816
7: -0.0055716, 0.0013697, -0.0054719, 0.0009583, -0.0041969, 0.0043058
8: 0.9852890, 0.9901786, 0.9853594, 0.9898888, -0.0029564, 0.0030331
9: -0.0069722, -0.0025337, -0.0067091, -0.0025975, -0.0027533, 0.0026836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0034063, 0.0051619, -0.0013540, 0.0014888
1: 0.0017936, 0.0021066, 0.0018144, 0.0020681, -0.0001956, 0.0002151
2: 0.0113583, 0.0125562, 0.0115060, 0.0124766, -0.0008231, 0.0007486
3: -0.0029332, -0.0016942, -0.0027805, -0.0017766, -0.0008513, 0.0007742
4: -0.0022029, -0.0008616, -0.0021137, -0.0010270, -0.0008381, 0.0009216
5: 0.0049287, 0.0061980, 0.0050852, 0.0061136, -0.0008721, 0.0007932
6: -0.0027445, 0.0022915, -0.0021238, 0.0019568, -0.0034604, 0.0031470
7: -0.0056775, 0.0011811, -0.0052217, 0.0003357, -0.0042860, 0.0047127
8: 0.9852145, 0.9900459, 0.9855356, 0.9894504, -0.0030191, 0.0033198
9: -0.0068516, -0.0024660, -0.0063110, -0.0027574, -0.0030135, 0.0027406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0034314, 0.0051612, -0.0013684, 0.0015362
1: 0.0017984, 0.0021152, 0.0018180, 0.0020679, -0.0001977, 0.0002219
2: 0.0113254, 0.0125377, 0.0115064, 0.0124627, -0.0008493, 0.0007565
3: -0.0029672, -0.0017134, -0.0027800, -0.0017909, -0.0008784, 0.0007824
4: -0.0021821, -0.0008247, -0.0020982, -0.0010274, -0.0008470, 0.0009509
5: 0.0048938, 0.0061784, 0.0050856, 0.0060989, -0.0008999, 0.0008016
6: -0.0028830, 0.0022137, -0.0021220, 0.0018985, -0.0035705, 0.0031804
7: -0.0055716, 0.0013697, -0.0051423, 0.0003332, -0.0043315, 0.0048627
8: 0.9852890, 0.9901786, 0.9855915, 0.9894487, -0.0030512, 0.0034254
9: -0.0069722, -0.0025337, -0.0063094, -0.0028082, -0.0031093, 0.0027697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0033996, 0.0052460, -0.0013262, 0.0013806
1: 0.0017936, 0.0021066, 0.0018135, 0.0020802, -0.0001916, 0.0001995
2: 0.0113583, 0.0125562, 0.0114595, 0.0124803, -0.0007633, 0.0007332
3: -0.0029332, -0.0016942, -0.0028285, -0.0017728, -0.0007894, 0.0007583
4: -0.0022029, -0.0008616, -0.0021178, -0.0009749, -0.0008210, 0.0008546
5: 0.0049287, 0.0061980, 0.0050360, 0.0061175, -0.0008088, 0.0007769
6: -0.0027445, 0.0022915, -0.0023191, 0.0019723, -0.0032089, 0.0030825
7: -0.0056775, 0.0011811, -0.0052428, 0.0006017, -0.0041981, 0.0043703
8: 0.9852145, 0.9900459, 0.9855207, 0.9896378, -0.0029572, 0.0030785
9: -0.0068516, -0.0024660, -0.0064811, -0.0027440, -0.0027945, 0.0026844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0034252, 0.0052451, -0.0013526, 0.0014113
1: 0.0017984, 0.0021152, 0.0018171, 0.0020801, -0.0001954, 0.0002039
2: 0.0113254, 0.0125377, 0.0114600, 0.0124661, -0.0007803, 0.0007478
3: -0.0029672, -0.0017134, -0.0028280, -0.0017874, -0.0008070, 0.0007734
4: -0.0021821, -0.0008247, -0.0021020, -0.0009755, -0.0008373, 0.0008736
5: 0.0048938, 0.0061784, 0.0050365, 0.0061026, -0.0008268, 0.0007923
6: -0.0028830, 0.0022137, -0.0023171, 0.0019128, -0.0032804, 0.0031438
7: -0.0055716, 0.0013697, -0.0051618, 0.0005990, -0.0042816, 0.0044676
8: 0.9852890, 0.9901786, 0.9855778, 0.9896359, -0.0030160, 0.0031470
9: -0.0069722, -0.0025337, -0.0064794, -0.0027957, -0.0028567, 0.0027378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0032529, 0.0053343, -0.0012903, 0.0014019
1: 0.0017936, 0.0021066, 0.0017923, 0.0020930, -0.0001864, 0.0002025
2: 0.0113583, 0.0125562, 0.0114107, 0.0125614, -0.0007751, 0.0007134
3: -0.0029332, -0.0016942, -0.0028790, -0.0016889, -0.0008016, 0.0007378
4: -0.0022029, -0.0008616, -0.0022087, -0.0009203, -0.0007987, 0.0008678
5: 0.0049287, 0.0061980, 0.0049842, 0.0062035, -0.0008212, 0.0007558
6: -0.0027445, 0.0022915, -0.0025244, 0.0023133, -0.0032583, 0.0029990
7: -0.0056775, 0.0011811, -0.0057073, 0.0008812, -0.0040844, 0.0044375
8: 0.9852145, 0.9900459, 0.9851935, 0.9898347, -0.0028771, 0.0031259
9: -0.0068516, -0.0024660, -0.0066598, -0.0024470, -0.0028375, 0.0026116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014110
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014150
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0032740, 0.0053333, -0.0013083, 0.0014535
1: 0.0017984, 0.0021152, 0.0017953, 0.0020928, -0.0001890, 0.0002100
2: 0.0113254, 0.0125377, 0.0114112, 0.0125498, -0.0008036, 0.0007233
3: -0.0029672, -0.0017134, -0.0028784, -0.0017009, -0.0008311, 0.0007481
4: -0.0021821, -0.0008247, -0.0021956, -0.0009209, -0.0008098, 0.0008998
5: 0.0048938, 0.0061784, 0.0049848, 0.0061912, -0.0008515, 0.0007664
6: -0.0028830, 0.0022137, -0.0025221, 0.0022644, -0.0033784, 0.0030407
7: -0.0055716, 0.0013697, -0.0056406, 0.0008781, -0.0041412, 0.0046011
8: 0.9852890, 0.9901786, 0.9852405, 0.9898325, -0.0029172, 0.0032411
9: -0.0069722, -0.0025337, -0.0066579, -0.0024896, -0.0029420, 0.0026480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032623, 0.0054290, 0.0032384, 0.0054302, -0.0012539, 0.0012762
1: 0.0017936, 0.0021066, 0.0017902, 0.0021068, -0.0001811, 0.0001844
2: 0.0113583, 0.0125562, 0.0113577, 0.0125695, -0.0007056, 0.0006932
3: -0.0029332, -0.0016942, -0.0029338, -0.0016805, -0.0007298, 0.0007170
4: -0.0022029, -0.0008616, -0.0022177, -0.0008609, -0.0007762, 0.0007900
5: 0.0049287, 0.0061980, 0.0049281, 0.0062120, -0.0007476, 0.0007345
6: -0.0027445, 0.0022915, -0.0027472, 0.0023471, -0.0029664, 0.0029144
7: -0.0056775, 0.0011811, -0.0057533, 0.0011847, -0.0039691, 0.0040399
8: 0.9852145, 0.9900459, 0.9851612, 0.9900484, -0.0027959, 0.0028458
9: -0.0068516, -0.0024660, -0.0068539, -0.0024175, -0.0025832, 0.0025380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014110
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014150
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032958, 0.0054886, 0.0032592, 0.0054291, -0.0012830, 0.0013201
1: 0.0017984, 0.0021152, 0.0017932, 0.0021066, -0.0001854, 0.0001907
2: 0.0113254, 0.0125377, 0.0113583, 0.0125579, -0.0007299, 0.0007093
3: -0.0029672, -0.0017134, -0.0029332, -0.0016924, -0.0007548, 0.0007336
4: -0.0021821, -0.0008247, -0.0022048, -0.0008616, -0.0007942, 0.0008172
5: 0.0048938, 0.0061784, 0.0049287, 0.0061998, -0.0007733, 0.0007516
6: -0.0028830, 0.0022137, -0.0027447, 0.0022988, -0.0030683, 0.0029820
7: -0.0055716, 0.0013697, -0.0056875, 0.0011813, -0.0040612, 0.0041788
8: 0.9852890, 0.9901786, 0.9852076, 0.9900460, -0.0028608, 0.0029436
9: -0.0069722, -0.0025337, -0.0068517, -0.0024596, -0.0026720, 0.0025969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
time: 0.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.05 seconds
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0014178, upper bound: 0.0016327
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0014178, upper bound: 0.0016476
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0014062, upper bound: 0.0016327
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0014062, upper bound: 0.0016476
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016449, upper bound: 0.0014110
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016449, upper bound: 0.0014150
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016449, upper bound: 0.0014110
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014150
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014021
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016419, upper bound: 0.0014062
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014022
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016476, upper bound: 0.0014062
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014110
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014150
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014110
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016448, upper bound: 0.0014150
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014111
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.05
Output dim: 8, lower bound: -0.0016509, upper bound: 0.0014150

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033056, 0.0054868, 0.0033686, 0.0052613, -0.0013257, 0.0014623
1: 0.0017999, 0.0021150, 0.0018090, 0.0020824, -0.0001915, 0.0002113
2: 0.0113264, 0.0125323, 0.0114510, 0.0124974, -0.0008085, 0.0007330
3: -0.0029662, -0.0017190, -0.0028373, -0.0017550, -0.0008362, 0.0007581
4: -0.0021761, -0.0008259, -0.0021371, -0.0009654, -0.0008206, 0.0009052
5: 0.0048949, 0.0061727, 0.0050270, 0.0061357, -0.0008566, 0.0007766
6: -0.0028788, 0.0021910, -0.0023548, 0.0020444, -0.0033988, 0.0030814
7: -0.0055406, 0.0013640, -0.0053410, 0.0006504, -0.0041966, 0.0046289
8: 0.9853109, 0.9901747, 0.9854515, 0.9896721, -0.0029561, 0.0032607
9: -0.0069685, -0.0025535, -0.0065122, -0.0026812, -0.0029598, 0.0026834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033155, 0.0054826, 0.0033833, 0.0053004, -0.0013357, 0.0014493
1: 0.0018013, 0.0021144, 0.0018111, 0.0020881, -0.0001930, 0.0002094
2: 0.0113287, 0.0125268, 0.0114294, 0.0124893, -0.0008013, 0.0007385
3: -0.0029638, -0.0017247, -0.0028596, -0.0017634, -0.0008287, 0.0007638
4: -0.0021699, -0.0008285, -0.0021279, -0.0009412, -0.0008268, 0.0008972
5: 0.0048974, 0.0061668, 0.0050041, 0.0061271, -0.0008490, 0.0007825
6: -0.0028690, 0.0021678, -0.0024456, 0.0020102, -0.0033687, 0.0031046
7: -0.0055090, 0.0013506, -0.0052945, 0.0007739, -0.0042282, 0.0045879
8: 0.9853333, 0.9901652, 0.9854843, 0.9897590, -0.0029785, 0.0032318
9: -0.0069600, -0.0025737, -0.0065912, -0.0027109, -0.0029336, 0.0027037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033056, 0.0054868, 0.0033590, 0.0053528, -0.0013138, 0.0013314
1: 0.0017999, 0.0021150, 0.0018076, 0.0020956, -0.0001898, 0.0001923
2: 0.0113264, 0.0125323, 0.0114005, 0.0125027, -0.0007361, 0.0007264
3: -0.0029662, -0.0017190, -0.0028896, -0.0017495, -0.0007613, 0.0007512
4: -0.0021761, -0.0008259, -0.0021430, -0.0009088, -0.0008133, 0.0008242
5: 0.0048949, 0.0061727, 0.0049734, 0.0061413, -0.0007799, 0.0007696
6: -0.0028788, 0.0021910, -0.0025673, 0.0020667, -0.0030945, 0.0030536
7: -0.0055406, 0.0013640, -0.0053713, 0.0009397, -0.0041588, 0.0042145
8: 0.9853109, 0.9901747, 0.9854302, 0.9898758, -0.0029295, 0.0029688
9: -0.0069685, -0.0025535, -0.0066972, -0.0026618, -0.0026948, 0.0026592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014110
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033155, 0.0054826, 0.0033741, 0.0053917, -0.0013154, 0.0013184
1: 0.0018013, 0.0021144, 0.0018098, 0.0021012, -0.0001900, 0.0001905
2: 0.0113287, 0.0125268, 0.0113789, 0.0124944, -0.0007289, 0.0007273
3: -0.0029638, -0.0017247, -0.0029118, -0.0017581, -0.0007539, 0.0007522
4: -0.0021699, -0.0008285, -0.0021337, -0.0008847, -0.0008143, 0.0008161
5: 0.0048974, 0.0061668, 0.0049506, 0.0061325, -0.0007723, 0.0007706
6: -0.0028690, 0.0021678, -0.0026579, 0.0020317, -0.0030644, 0.0030574
7: -0.0055090, 0.0013506, -0.0053237, 0.0010630, -0.0041639, 0.0041734
8: 0.9853333, 0.9901652, 0.9854638, 0.9899628, -0.0029331, 0.0029398
9: -0.0069600, -0.0025737, -0.0067761, -0.0026922, -0.0026686, 0.0026625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033056, 0.0054868, 0.0033065, 0.0053268, -0.0012959, 0.0014244
1: 0.0017999, 0.0021150, 0.0018000, 0.0020919, -0.0001872, 0.0002058
2: 0.0113264, 0.0125323, 0.0114148, 0.0125318, -0.0007875, 0.0007165
3: -0.0029662, -0.0017190, -0.0028747, -0.0017195, -0.0008145, 0.0007410
4: -0.0021761, -0.0008259, -0.0021755, -0.0009249, -0.0008022, 0.0008817
5: 0.0048949, 0.0061727, 0.0049886, 0.0061721, -0.0008344, 0.0007591
6: -0.0028788, 0.0021910, -0.0025069, 0.0021887, -0.0033106, 0.0030120
7: -0.0055406, 0.0013640, -0.0055375, 0.0008575, -0.0041020, 0.0045088
8: 0.9853109, 0.9901747, 0.9853131, 0.9898179, -0.0028896, 0.0031761
9: -0.0069685, -0.0025535, -0.0066447, -0.0025555, -0.0028830, 0.0026230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033155, 0.0054826, 0.0033193, 0.0053662, -0.0013025, 0.0014133
1: 0.0018013, 0.0021144, 0.0018018, 0.0020976, -0.0001882, 0.0002042
2: 0.0113287, 0.0125268, 0.0113930, 0.0125247, -0.0007814, 0.0007201
3: -0.0029638, -0.0017247, -0.0028972, -0.0017268, -0.0008081, 0.0007448
4: -0.0021699, -0.0008285, -0.0021676, -0.0009005, -0.0008063, 0.0008749
5: 0.0048974, 0.0061668, 0.0049655, 0.0061646, -0.0008279, 0.0007630
6: -0.0028690, 0.0021678, -0.0025985, 0.0021591, -0.0032849, 0.0030274
7: -0.0055090, 0.0013506, -0.0054973, 0.0009822, -0.0041230, 0.0044738
8: 0.9853333, 0.9901652, 0.9853415, 0.9899058, -0.0029043, 0.0031514
9: -0.0069600, -0.0025737, -0.0067244, -0.0025813, -0.0028607, 0.0026364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033056, 0.0054868, 0.0032922, 0.0054225, -0.0012705, 0.0012911
1: 0.0017999, 0.0021150, 0.0017979, 0.0021057, -0.0001835, 0.0001865
2: 0.0113264, 0.0125323, 0.0113619, 0.0125397, -0.0007138, 0.0007024
3: -0.0029662, -0.0017190, -0.0029294, -0.0017113, -0.0007383, 0.0007265
4: -0.0021761, -0.0008259, -0.0021843, -0.0008657, -0.0007865, 0.0007992
5: 0.0048949, 0.0061727, 0.0049326, 0.0061805, -0.0007563, 0.0007442
6: -0.0028788, 0.0021910, -0.0027294, 0.0022219, -0.0030009, 0.0029530
7: -0.0055406, 0.0013640, -0.0055828, 0.0011605, -0.0040217, 0.0040870
8: 0.9853109, 0.9901747, 0.9852812, 0.9900313, -0.0028329, 0.0028790
9: -0.0069685, -0.0025535, -0.0068384, -0.0025266, -0.0026133, 0.0025716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033155, 0.0054826, 0.0033030, 0.0054644, -0.0012777, 0.0012806
1: 0.0018013, 0.0021144, 0.0017995, 0.0021117, -0.0001846, 0.0001850
2: 0.0113287, 0.0125268, 0.0113387, 0.0125337, -0.0007080, 0.0007064
3: -0.0029638, -0.0017247, -0.0029534, -0.0017175, -0.0007322, 0.0007306
4: -0.0021699, -0.0008285, -0.0021776, -0.0008397, -0.0007909, 0.0007927
5: 0.0048974, 0.0061668, 0.0049080, 0.0061741, -0.0007502, 0.0007485
6: -0.0028690, 0.0021678, -0.0028267, 0.0021968, -0.0029764, 0.0029698
7: -0.0055090, 0.0013506, -0.0055486, 0.0012930, -0.0040446, 0.0040536
8: 0.9853333, 0.9901652, 0.9853053, 0.9901247, -0.0028491, 0.0028554
9: -0.0069600, -0.0025737, -0.0069232, -0.0025484, -0.0025920, 0.0025862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
time: 0.70 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.15 seconds
IS_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
IS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014110
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
IS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016357, upper bound: 0.0014111
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0015727, upper bound: 0.0013833
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.15
Output dim: 8, lower bound: -0.0016172, upper bound: 0.0013833

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.29 + 397.16 = 400.45 seconds
