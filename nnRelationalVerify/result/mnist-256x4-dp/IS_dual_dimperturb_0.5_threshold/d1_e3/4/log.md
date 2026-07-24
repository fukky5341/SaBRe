## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00399952


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0153260, 0.0180630, 0.0153260, 0.0180630, -0.0022573, 0.0022573)
1: (-0.0017530, 0.0002114, -0.0017530, 0.0002114, -0.0016417, 0.0016417)
2: (0.0036297, 0.0045125, 0.0036297, 0.0045125, -0.0007236, 0.0007236)
3: (0.0013791, 0.0027532, 0.0013791, 0.0027532, -0.0010482, 0.0010482)
4: (-0.0045921, -0.0026830, -0.0045921, -0.0026830, -0.0013828, 0.0013828)
5: (-0.0002769, 0.0009117, -0.0002769, 0.0009117, -0.0009980, 0.0009980)
6: (-0.0049177, -0.0016370, -0.0049177, -0.0016370, -0.0023369, 0.0023369)
7: (-0.0225698, -0.0115603, -0.0225698, -0.0115603, -0.0080160, 0.0080161)
8: (0.9748465, 0.9850336, 0.9748465, 0.9850336, -0.0076866, 0.0076866)
9: (-0.0001642, 0.0070871, -0.0001642, 0.0070871, -0.0053149, 0.0053149)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.71 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0052624, upper bound: 0.0052624

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049737, upper bound: 0.0050584
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050583, upper bound: 0.0050583
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 8, lower bound: -0.0049737, upper bound: 0.0050584
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 8, lower bound: -0.0050583, upper bound: 0.0050583

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0153520, 0.0180323, 0.0153330, 0.0180545, -0.0022245, 0.0022047
1: -0.0017310, 0.0001987, -0.0017471, 0.0002079, -0.0016183, 0.0016084
2: 0.0036411, 0.0045045, 0.0036328, 0.0045103, -0.0007062, 0.0007131
3: 0.0013821, 0.0027172, 0.0013800, 0.0027430, -0.0010340, 0.0010109
4: -0.0045337, -0.0026927, -0.0045760, -0.0026855, -0.0013198, 0.0013574
5: -0.0002718, 0.0008979, -0.0002755, 0.0009080, -0.0009785, 0.0009833
6: -0.0049089, -0.0017590, -0.0049153, -0.0016710, -0.0022908, 0.0022062
7: -0.0222416, -0.0116188, -0.0224795, -0.0115759, -0.0076595, 0.0078707
8: 0.9751063, 0.9849699, 0.9749182, 0.9850168, -0.0073853, 0.0075578
9: -0.0001246, 0.0068782, -0.0001536, 0.0070294, -0.0052198, 0.0050844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048990
time: 0.70 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048361, upper bound: 0.0049361
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0151335, 0.0180402, 0.0153321, 0.0180548, -0.0024609, 0.0022109
1: -0.0018773, 0.0002023, -0.0017480, 0.0002078, -0.0017818, 0.0016148
2: 0.0036376, 0.0045752, 0.0036325, 0.0045106, -0.0007084, 0.0007898
3: 0.0013571, 0.0027399, 0.0013801, 0.0027431, -0.0010683, 0.0010339
4: -0.0045452, -0.0025788, -0.0045761, -0.0026851, -0.0013307, 0.0015030
5: -0.0002736, 0.0009845, -0.0002754, 0.0009085, -0.0009834, 0.0010800
6: -0.0050048, -0.0017317, -0.0049148, -0.0016714, -0.0024285, 0.0022313
7: -0.0223068, -0.0109398, -0.0224800, -0.0115730, -0.0077225, 0.0087274
8: 0.9750465, 0.9856889, 0.9749146, 0.9850197, -0.0074447, 0.0084135
9: -0.0005895, 0.0069212, -0.0001556, 0.0070303, -0.0057966, 0.0051254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0048990
time: 0.74 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049361, upper bound: 0.0049361
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.01 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048990
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 8, lower bound: -0.0048361, upper bound: 0.0049361
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0048990
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 8, lower bound: -0.0049361, upper bound: 0.0049361

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0153480, 0.0180499, -0.0022745, 0.0021685
1: -0.0017754, 0.0002050, -0.0017345, 0.0002074, -0.0016562, 0.0016000
2: 0.0036480, 0.0045326, 0.0036352, 0.0045058, -0.0006921, 0.0007289
3: 0.0013484, 0.0026366, 0.0013804, 0.0027091, -0.0010038, 0.0009241
4: -0.0044353, -0.0026111, -0.0045431, -0.0026882, -0.0012014, 0.0013664
5: -0.0002789, 0.0009215, -0.0002752, 0.0009000, -0.0009773, 0.0010066
6: -0.0050114, -0.0020146, -0.0049137, -0.0017611, -0.0022228, 0.0019143
7: -0.0216981, -0.0111475, -0.0222963, -0.0115936, -0.0070014, 0.0079329
8: 0.9754941, 0.9854095, 0.9750478, 0.9849914, -0.0068968, 0.0076572
9: -0.0004374, 0.0065373, -0.0001407, 0.0069143, -0.0052690, 0.0046700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048506
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048990
time: 0.76 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0153331, 0.0180545, -0.0022147, 0.0021705
1: -0.0017211, 0.0001983, -0.0017470, 0.0002079, -0.0016106, 0.0015973
2: 0.0036433, 0.0045010, 0.0036329, 0.0045102, -0.0006934, 0.0007101
3: 0.0013826, 0.0026914, 0.0013800, 0.0027428, -0.0010331, 0.0009279
4: -0.0045041, -0.0026951, -0.0045757, -0.0026856, -0.0012244, 0.0013548
5: -0.0002715, 0.0008917, -0.0002755, 0.0009079, -0.0009749, 0.0009784
6: -0.0049073, -0.0018278, -0.0049153, -0.0016717, -0.0022881, 0.0019859
7: -0.0220766, -0.0116342, -0.0224777, -0.0115760, -0.0071252, 0.0078544
8: 0.9752277, 0.9849489, 0.9749196, 0.9850165, -0.0069733, 0.0075371
9: -0.0001135, 0.0067747, -0.0001535, 0.0070283, -0.0052082, 0.0047456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048621
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0049361
time: 0.87 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0153471, 0.0180502, -0.0025245, 0.0021746
1: -0.0019284, 0.0002083, -0.0017354, 0.0002073, -0.0018273, 0.0016061
2: 0.0036444, 0.0046066, 0.0036349, 0.0045060, -0.0006943, 0.0008101
3: 0.0013196, 0.0026613, 0.0013806, 0.0027091, -0.0010422, 0.0009484
4: -0.0044509, -0.0024907, -0.0045432, -0.0026878, -0.0012161, 0.0015228
5: -0.0002807, 0.0010107, -0.0002751, 0.0009005, -0.0009824, 0.0011073
6: -0.0051098, -0.0019891, -0.0049133, -0.0017617, -0.0023736, 0.0019411
7: -0.0217854, -0.0104309, -0.0222972, -0.0115906, -0.0070832, 0.0088537
8: 0.9754301, 0.9861599, 0.9750438, 0.9849946, -0.0069621, 0.0085800
9: -0.0009249, 0.0065934, -0.0001427, 0.0069158, -0.0058890, 0.0047219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
time: 0.73 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0153322, 0.0180548, -0.0024514, 0.0021758
1: -0.0018679, 0.0002019, -0.0017479, 0.0002078, -0.0017742, 0.0016035
2: 0.0036398, 0.0045718, 0.0036325, 0.0045105, -0.0006951, 0.0007869
3: 0.0013576, 0.0027145, 0.0013801, 0.0027428, -0.0010675, 0.0009517
4: -0.0045170, -0.0025811, -0.0045758, -0.0026851, -0.0012390, 0.0015004
5: -0.0002733, 0.0009785, -0.0002754, 0.0009085, -0.0009793, 0.0010751
6: -0.0050033, -0.0018011, -0.0049148, -0.0016722, -0.0024260, 0.0020161
7: -0.0221476, -0.0109538, -0.0224783, -0.0115732, -0.0072088, 0.0087112
8: 0.9751647, 0.9856695, 0.9749159, 0.9850196, -0.0070461, 0.0083929
9: -0.0005793, 0.0068189, -0.0001555, 0.0070293, -0.0057850, 0.0047997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0048621
time: 0.92 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0049361
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.10 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048506
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048990
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048621
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0049361
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0048621
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0049361

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0153675, 0.0180277, -0.0022401, 0.0021536
1: -0.0017754, 0.0002050, -0.0017178, 0.0001982, -0.0016350, 0.0015885
2: 0.0036480, 0.0045326, 0.0036434, 0.0044998, -0.0006874, 0.0007175
3: 0.0013484, 0.0026366, 0.0013826, 0.0026836, -0.0009784, 0.0009213
4: -0.0044353, -0.0026111, -0.0045008, -0.0026956, -0.0011957, 0.0013218
5: -0.0002789, 0.0009215, -0.0002715, 0.0008896, -0.0009699, 0.0009944
6: -0.0050114, -0.0020146, -0.0049073, -0.0018486, -0.0021298, 0.0019067
7: -0.0216981, -0.0111475, -0.0220588, -0.0116383, -0.0069678, 0.0076804
8: 0.9754941, 0.9854095, 0.9752364, 0.9849425, -0.0068611, 0.0074461
9: -0.0004374, 0.0065373, -0.0001104, 0.0067633, -0.0051063, 0.0046474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048108
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048506
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0151484, 0.0180356, -0.0022539, 0.0023867
1: -0.0017754, 0.0002050, -0.0018647, 0.0002018, -0.0016386, 0.0017507
2: 0.0036480, 0.0045326, 0.0036399, 0.0045706, -0.0007631, 0.0007226
3: 0.0013484, 0.0026366, 0.0013576, 0.0027065, -0.0010011, 0.0009554
4: -0.0044353, -0.0026111, -0.0045128, -0.0025816, -0.0013365, 0.0013514
5: -0.0002789, 0.0009215, -0.0002733, 0.0009764, -0.0010664, 0.0009960
6: -0.0050114, -0.0020146, -0.0050033, -0.0018219, -0.0021824, 0.0020394
7: -0.0216981, -0.0111475, -0.0221279, -0.0109572, -0.0077958, 0.0078486
8: 0.9754941, 0.9854095, 0.9751766, 0.9856637, -0.0076882, 0.0075883
9: -0.0004374, 0.0065373, -0.0005767, 0.0068079, -0.0052155, 0.0052044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048621
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048990
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0152469, 0.0180409, -0.0021907, 0.0022600
1: -0.0017211, 0.0001983, -0.0017932, 0.0002141, -0.0016124, 0.0016485
2: 0.0036433, 0.0045010, 0.0036398, 0.0045393, -0.0007241, 0.0006997
3: 0.0013826, 0.0026914, 0.0013461, 0.0026608, -0.0009445, 0.0009999
4: -0.0045041, -0.0026951, -0.0044785, -0.0026030, -0.0013493, 0.0012425
5: -0.0002715, 0.0008917, -0.0002826, 0.0009327, -0.0010024, 0.0009834
6: -0.0049073, -0.0018278, -0.0050180, -0.0019284, -0.0019980, 0.0021906
7: -0.0220766, -0.0116342, -0.0219398, -0.0110996, -0.0078341, 0.0072318
8: 0.9752277, 0.9849489, 0.9753064, 0.9854641, -0.0075704, 0.0070791
9: -0.0001135, 0.0067747, -0.0004703, 0.0066919, -0.0048171, 0.0052044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048108
time: 1.02 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048621
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0153442, 0.0180500, -0.0021810, 0.0021612
1: -0.0017211, 0.0001983, -0.0017377, 0.0002075, -0.0015990, 0.0015897
2: 0.0036433, 0.0045010, 0.0036351, 0.0045069, -0.0006905, 0.0006974
3: 0.0013826, 0.0026914, 0.0013804, 0.0027173, -0.0009495, 0.0009273
4: -0.0045041, -0.0026951, -0.0045465, -0.0026877, -0.0012221, 0.0012628
5: -0.0002715, 0.0008917, -0.0002752, 0.0009020, -0.0009701, 0.0009746
6: -0.0049073, -0.0018278, -0.0049138, -0.0017407, -0.0020699, 0.0019840
7: -0.0220766, -0.0116342, -0.0223154, -0.0115900, -0.0071108, 0.0073379
8: 0.9752277, 0.9849489, 0.9750378, 0.9849972, -0.0069543, 0.0071315
9: -0.0001135, 0.0067747, -0.0001434, 0.0069268, -0.0048796, 0.0047351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048731
time: 0.96 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0049173
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0153675, 0.0180277, -0.0024860, 0.0021668
1: -0.0019284, 0.0002083, -0.0017178, 0.0001982, -0.0018049, 0.0015925
2: 0.0036444, 0.0046066, 0.0036434, 0.0044998, -0.0006925, 0.0007973
3: 0.0013196, 0.0026613, 0.0013826, 0.0026836, -0.0010168, 0.0009446
4: -0.0044509, -0.0024907, -0.0045008, -0.0026956, -0.0012274, 0.0014722
5: -0.0002807, 0.0010107, -0.0002715, 0.0008896, -0.0009714, 0.0010950
6: -0.0051098, -0.0019891, -0.0049073, -0.0018486, -0.0022745, 0.0019638
7: -0.0217854, -0.0104309, -0.0220588, -0.0116383, -0.0071469, 0.0085658
8: 0.9754301, 0.9861599, 0.9752364, 0.9849425, -0.0070066, 0.0083394
9: -0.0009249, 0.0065934, -0.0001104, 0.0067633, -0.0057031, 0.0047626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047536
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0151484, 0.0180356, -0.0024384, 0.0023423
1: -0.0019284, 0.0002083, -0.0018647, 0.0002018, -0.0017781, 0.0017247
2: 0.0036444, 0.0046066, 0.0036399, 0.0045706, -0.0007485, 0.0007818
3: 0.0013196, 0.0026613, 0.0013576, 0.0027065, -0.0010310, 0.0009721
4: -0.0044509, -0.0024907, -0.0045128, -0.0025816, -0.0013066, 0.0014365
5: -0.0002807, 0.0010107, -0.0002733, 0.0009764, -0.0010535, 0.0010809
6: -0.0051098, -0.0019891, -0.0050033, -0.0018219, -0.0022379, 0.0020080
7: -0.0217854, -0.0104309, -0.0221279, -0.0109572, -0.0076212, 0.0083602
8: 0.9754301, 0.9861599, 0.9751766, 0.9856637, -0.0075238, 0.0081532
9: -0.0009249, 0.0065934, -0.0005767, 0.0068079, -0.0055678, 0.0050884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047536
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0152456, 0.0180414, -0.0024274, 0.0022665
1: -0.0018679, 0.0002019, -0.0017943, 0.0002139, -0.0017763, 0.0016551
2: 0.0036398, 0.0045718, 0.0036395, 0.0045398, -0.0007264, 0.0007767
3: 0.0013576, 0.0027145, 0.0013463, 0.0026611, -0.0009786, 0.0010224
4: -0.0045170, -0.0025811, -0.0044798, -0.0026023, -0.0013599, 0.0013900
5: -0.0002733, 0.0009785, -0.0002825, 0.0009334, -0.0010075, 0.0010805
6: -0.0050033, -0.0018011, -0.0050176, -0.0019301, -0.0021399, 0.0022164
7: -0.0221476, -0.0109538, -0.0219481, -0.0110959, -0.0078956, 0.0080984
8: 0.9751647, 0.9856695, 0.9752996, 0.9854684, -0.0076307, 0.0079364
9: -0.0005793, 0.0068189, -0.0004728, 0.0066967, -0.0053987, 0.0052446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0047536
time: 0.77 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0047536
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0153433, 0.0180503, -0.0024188, 0.0021665
1: -0.0018679, 0.0002019, -0.0017386, 0.0002074, -0.0017628, 0.0015960
2: 0.0036398, 0.0045718, 0.0036348, 0.0045072, -0.0006922, 0.0007747
3: 0.0013576, 0.0027145, 0.0013806, 0.0027174, -0.0009881, 0.0009511
4: -0.0045170, -0.0025811, -0.0045469, -0.0026873, -0.0012368, 0.0014160
5: -0.0002733, 0.0009785, -0.0002751, 0.0009026, -0.0009745, 0.0010714
6: -0.0050033, -0.0018011, -0.0049133, -0.0017413, -0.0022172, 0.0020142
7: -0.0221476, -0.0109538, -0.0223179, -0.0115870, -0.0071945, 0.0082371
8: 0.9751647, 0.9856695, 0.9750339, 0.9850004, -0.0070271, 0.0080222
9: -0.0005793, 0.0068189, -0.0001454, 0.0069286, -0.0054834, 0.0047894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0048195
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0048195
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.01 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048108
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048506
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048621
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047536, upper bound: 0.0048990
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048108
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048621
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0048731
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0047908, upper bound: 0.0049173
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047536
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047536
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048621, upper bound: 0.0047908
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0047536
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0047536
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0048195
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 8, lower bound: -0.0048990, upper bound: 0.0048195

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0152687, 0.0180188, -0.0022320, 0.0022320
1: -0.0017754, 0.0002050, -0.0017754, 0.0002050, -0.0016422, 0.0016422
2: 0.0036480, 0.0045326, 0.0036480, 0.0045326, -0.0007130, 0.0007130
3: 0.0013484, 0.0026366, 0.0013484, 0.0026366, -0.0009298, 0.0009298
4: -0.0044353, -0.0026111, -0.0044353, -0.0026111, -0.0012465, 0.0012465
5: -0.0002789, 0.0009215, -0.0002789, 0.0009215, -0.0010017, 0.0010017
6: -0.0050114, -0.0020146, -0.0050114, -0.0020146, -0.0019444, 0.0019444
7: -0.0216981, -0.0111475, -0.0216981, -0.0111475, -0.0072643, 0.0072643
8: 0.9754941, 0.9854095, 0.9754941, 0.9854095, -0.0071492, 0.0071492
9: -0.0004374, 0.0065373, -0.0004374, 0.0065373, -0.0048455, 0.0048455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044713, upper bound: 0.0040183
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0039848
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0153636, 0.0180278, -0.0022431, 0.0021564
1: -0.0017754, 0.0002050, -0.0017211, 0.0001983, -0.0016351, 0.0015908
2: 0.0036480, 0.0045326, 0.0036433, 0.0045010, -0.0006883, 0.0007187
3: 0.0013484, 0.0026366, 0.0013826, 0.0026914, -0.0009971, 0.0009213
4: -0.0044353, -0.0026111, -0.0045041, -0.0026951, -0.0011960, 0.0013427
5: -0.0002789, 0.0009215, -0.0002715, 0.0008917, -0.0009714, 0.0009941
6: -0.0050114, -0.0020146, -0.0049073, -0.0018278, -0.0021832, 0.0019069
7: -0.0216981, -0.0111475, -0.0220766, -0.0116342, -0.0069704, 0.0077951
8: 0.9754941, 0.9854095, 0.9752277, 0.9849489, -0.0068656, 0.0075284
9: -0.0004374, 0.0065373, -0.0001135, 0.0067747, -0.0051781, 0.0046494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044713, upper bound: 0.0043189
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0042960
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0150412, 0.0180267, -0.0022452, 0.0024780
1: -0.0017754, 0.0002050, -0.0019284, 0.0002083, -0.0016462, 0.0018121
2: 0.0036480, 0.0045326, 0.0036444, 0.0046066, -0.0007929, 0.0007180
3: 0.0013484, 0.0026366, 0.0013196, 0.0026613, -0.0009531, 0.0009682
4: -0.0044353, -0.0026111, -0.0044509, -0.0024907, -0.0013969, 0.0012783
5: -0.0002789, 0.0009215, -0.0002807, 0.0010107, -0.0011022, 0.0010031
6: -0.0050114, -0.0020146, -0.0051098, -0.0019891, -0.0020015, 0.0020891
7: -0.0216981, -0.0111475, -0.0217854, -0.0104309, -0.0081497, 0.0074434
8: 0.9754941, 0.9854095, 0.9754301, 0.9861599, -0.0080425, 0.0072947
9: -0.0004374, 0.0065373, -0.0009249, 0.0065934, -0.0049607, 0.0054423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0037065, upper bound: 0.0043648
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0036946
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0151446, 0.0180356, -0.0022563, 0.0023894
1: -0.0017754, 0.0002050, -0.0018679, 0.0002019, -0.0016386, 0.0017529
2: 0.0036480, 0.0045326, 0.0036398, 0.0045718, -0.0007639, 0.0007236
3: 0.0013484, 0.0026366, 0.0013576, 0.0027145, -0.0010196, 0.0009555
4: -0.0044353, -0.0026111, -0.0045170, -0.0025811, -0.0013368, 0.0013689
5: -0.0002789, 0.0009215, -0.0002733, 0.0009785, -0.0010679, 0.0009956
6: -0.0050114, -0.0020146, -0.0050033, -0.0018011, -0.0022341, 0.0020395
7: -0.0216981, -0.0111475, -0.0221476, -0.0109538, -0.0077980, 0.0079446
8: 0.9754941, 0.9854095, 0.9751647, 0.9856695, -0.0076924, 0.0076609
9: -0.0004374, 0.0065373, -0.0005793, 0.0068189, -0.0052750, 0.0052062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041467, upper bound: 0.0037916
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0041078
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0152687, 0.0180188, -0.0021564, 0.0022431
1: -0.0017211, 0.0001983, -0.0017754, 0.0002050, -0.0015908, 0.0016351
2: 0.0036433, 0.0045010, 0.0036480, 0.0045326, -0.0007187, 0.0006883
3: 0.0013826, 0.0026914, 0.0013484, 0.0026366, -0.0009213, 0.0009971
4: -0.0045041, -0.0026951, -0.0044353, -0.0026111, -0.0013427, 0.0011960
5: -0.0002715, 0.0008917, -0.0002789, 0.0009215, -0.0009941, 0.0009714
6: -0.0049073, -0.0018278, -0.0050114, -0.0020146, -0.0019069, 0.0021832
7: -0.0220766, -0.0116342, -0.0216981, -0.0111475, -0.0077951, 0.0069704
8: 0.9752277, 0.9849489, 0.9754941, 0.9854095, -0.0075284, 0.0068656
9: -0.0001135, 0.0067747, -0.0004374, 0.0065373, -0.0046494, 0.0051781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040788, upper bound: 0.0044166
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0150412, 0.0180267, -0.0021696, 0.0024891
1: -0.0017211, 0.0001983, -0.0019284, 0.0002083, -0.0015948, 0.0018050
2: 0.0036433, 0.0045010, 0.0036444, 0.0046066, -0.0007986, 0.0006933
3: 0.0013826, 0.0026914, 0.0013196, 0.0026613, -0.0009446, 0.0010355
4: -0.0045041, -0.0026951, -0.0044509, -0.0024907, -0.0014930, 0.0012277
5: -0.0002715, 0.0008917, -0.0002807, 0.0010107, -0.0010947, 0.0009728
6: -0.0049073, -0.0018278, -0.0051098, -0.0019891, -0.0019639, 0.0023279
7: -0.0220766, -0.0116342, -0.0217854, -0.0104309, -0.0086805, 0.0071495
8: 0.9752277, 0.9849489, 0.9754301, 0.9861599, -0.0084217, 0.0070111
9: -0.0001135, 0.0067747, -0.0009249, 0.0065934, -0.0047646, 0.0057749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040788, upper bound: 0.0044166
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0153636, 0.0180278, -0.0021464, 0.0021464
1: -0.0017211, 0.0001983, -0.0017211, 0.0001983, -0.0015779, 0.0015779
2: 0.0036433, 0.0045010, 0.0036433, 0.0045010, -0.0006859, 0.0006859
3: 0.0013826, 0.0026914, 0.0013826, 0.0026914, -0.0009246, 0.0009246
4: -0.0045041, -0.0026951, -0.0045041, -0.0026951, -0.0012162, 0.0012162
5: -0.0002715, 0.0008917, -0.0002715, 0.0008917, -0.0009625, 0.0009625
6: -0.0049073, -0.0018278, -0.0049073, -0.0018278, -0.0019771, 0.0019771
7: -0.0220766, -0.0116342, -0.0220766, -0.0116342, -0.0070759, 0.0070759
8: 0.9752277, 0.9849489, 0.9752277, 0.9849489, -0.0069183, 0.0069183
9: -0.0001135, 0.0067747, -0.0001135, 0.0067747, -0.0047117, 0.0047117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046088, upper bound: 0.0044421
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044877
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0151446, 0.0180356, -0.0021601, 0.0023805
1: -0.0017211, 0.0001983, -0.0018679, 0.0002019, -0.0015813, 0.0017408
2: 0.0036433, 0.0045010, 0.0036398, 0.0045718, -0.0007618, 0.0006911
3: 0.0013826, 0.0026914, 0.0013576, 0.0027145, -0.0009483, 0.0009632
4: -0.0045041, -0.0026951, -0.0045170, -0.0025811, -0.0013629, 0.0012478
5: -0.0002715, 0.0008917, -0.0002733, 0.0009785, -0.0010591, 0.0009641
6: -0.0049073, -0.0018278, -0.0050033, -0.0018011, -0.0020336, 0.0021163
7: -0.0220766, -0.0116342, -0.0221476, -0.0109538, -0.0079368, 0.0072550
8: 0.9752277, 0.9849489, 0.9751647, 0.9856695, -0.0077785, 0.0070631
9: -0.0001135, 0.0067747, -0.0005793, 0.0068189, -0.0048270, 0.0052909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044154, upper bound: 0.0046840
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044877
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0152687, 0.0180188, -0.0024780, 0.0022452
1: -0.0019284, 0.0002083, -0.0017754, 0.0002050, -0.0018121, 0.0016462
2: 0.0036444, 0.0046066, 0.0036480, 0.0045326, -0.0007180, 0.0007929
3: 0.0013196, 0.0026613, 0.0013484, 0.0026366, -0.0009682, 0.0009531
4: -0.0044509, -0.0024907, -0.0044353, -0.0026111, -0.0012782, 0.0013969
5: -0.0002807, 0.0010107, -0.0002789, 0.0009215, -0.0010031, 0.0011022
6: -0.0051098, -0.0019891, -0.0050114, -0.0020146, -0.0020891, 0.0020015
7: -0.0217854, -0.0104309, -0.0216981, -0.0111475, -0.0074434, 0.0081497
8: 0.9754301, 0.9861599, 0.9754941, 0.9854095, -0.0072947, 0.0080425
9: -0.0009249, 0.0065934, -0.0004374, 0.0065373, -0.0054423, 0.0049607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043648, upper bound: 0.0037065
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0034984
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0153636, 0.0180278, -0.0024891, 0.0021696
1: -0.0019284, 0.0002083, -0.0017211, 0.0001983, -0.0018050, 0.0015948
2: 0.0036444, 0.0046066, 0.0036433, 0.0045010, -0.0006933, 0.0007986
3: 0.0013196, 0.0026613, 0.0013826, 0.0026914, -0.0010355, 0.0009446
4: -0.0044509, -0.0024907, -0.0045041, -0.0026951, -0.0012277, 0.0014930
5: -0.0002807, 0.0010107, -0.0002715, 0.0008917, -0.0009728, 0.0010947
6: -0.0051098, -0.0019891, -0.0049073, -0.0018278, -0.0023279, 0.0019639
7: -0.0217854, -0.0104309, -0.0220766, -0.0116342, -0.0071495, 0.0086805
8: 0.9754301, 0.9861599, 0.9752277, 0.9849489, -0.0070111, 0.0084217
9: -0.0009249, 0.0065934, -0.0001135, 0.0067747, -0.0057749, 0.0047646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043648, upper bound: 0.0040788
time: 0.86 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0038840
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0150412, 0.0180267, -0.0024306, 0.0024306
1: -0.0019284, 0.0002083, -0.0019284, 0.0002083, -0.0017850, 0.0017850
2: 0.0036444, 0.0046066, 0.0036444, 0.0046066, -0.0007773, 0.0007773
3: 0.0013196, 0.0026613, 0.0013196, 0.0026613, -0.0009839, 0.0009839
4: -0.0044509, -0.0024907, -0.0044509, -0.0024907, -0.0013634, 0.0013634
5: -0.0002807, 0.0010107, -0.0002807, 0.0010107, -0.0010885, 0.0010885
6: -0.0051098, -0.0019891, -0.0051098, -0.0019891, -0.0020524, 0.0020524
7: -0.0217854, -0.0104309, -0.0217854, -0.0104309, -0.0079538, 0.0079538
8: 0.9754301, 0.9861599, 0.9754301, 0.9861599, -0.0078583, 0.0078583
9: -0.0009249, 0.0065934, -0.0009249, 0.0065934, -0.0053121, 0.0053121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041938, upper bound: 0.0036994
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0150412, 0.0180267, 0.0151446, 0.0180356, -0.0024417, 0.0023450
1: -0.0019284, 0.0002083, -0.0018679, 0.0002019, -0.0017781, 0.0017270
2: 0.0036444, 0.0046066, 0.0036398, 0.0045718, -0.0007494, 0.0007831
3: 0.0013196, 0.0026613, 0.0013576, 0.0027145, -0.0010493, 0.0009721
4: -0.0044509, -0.0024907, -0.0045170, -0.0025811, -0.0013070, 0.0014552
5: -0.0002807, 0.0010107, -0.0002733, 0.0009785, -0.0010550, 0.0010810
6: -0.0051098, -0.0019891, -0.0050033, -0.0018011, -0.0022903, 0.0020081
7: -0.0217854, -0.0104309, -0.0221476, -0.0109538, -0.0076235, 0.0084631
8: 0.9754301, 0.9861599, 0.9751647, 0.9856695, -0.0075280, 0.0082311
9: -0.0009249, 0.0065934, -0.0005793, 0.0068189, -0.0056322, 0.0050902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041938, upper bound: 0.0040772
time: 0.86 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0152687, 0.0180188, -0.0023894, 0.0022563
1: -0.0018679, 0.0002019, -0.0017754, 0.0002050, -0.0017529, 0.0016386
2: 0.0036398, 0.0045718, 0.0036480, 0.0045326, -0.0007236, 0.0007639
3: 0.0013576, 0.0027145, 0.0013484, 0.0026366, -0.0009555, 0.0010196
4: -0.0045170, -0.0025811, -0.0044353, -0.0026111, -0.0013689, 0.0013368
5: -0.0002733, 0.0009785, -0.0002789, 0.0009215, -0.0009956, 0.0010679
6: -0.0050033, -0.0018011, -0.0050114, -0.0020146, -0.0020395, 0.0022341
7: -0.0221476, -0.0109538, -0.0216981, -0.0111475, -0.0079446, 0.0077980
8: 0.9751647, 0.9856695, 0.9754941, 0.9854095, -0.0076609, 0.0076924
9: -0.0005793, 0.0068189, -0.0004374, 0.0065373, -0.0052062, 0.0052750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040839, upper bound: 0.0042665
time: 0.87 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
time: 0.67 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0150412, 0.0180267, -0.0023450, 0.0024417
1: -0.0018679, 0.0002019, -0.0019284, 0.0002083, -0.0017270, 0.0017781
2: 0.0036398, 0.0045718, 0.0036444, 0.0046066, -0.0007831, 0.0007494
3: 0.0013576, 0.0027145, 0.0013196, 0.0026613, -0.0009721, 0.0010493
4: -0.0045170, -0.0025811, -0.0044509, -0.0024907, -0.0014552, 0.0013070
5: -0.0002733, 0.0009785, -0.0002807, 0.0010107, -0.0010810, 0.0010550
6: -0.0050033, -0.0018011, -0.0051098, -0.0019891, -0.0020081, 0.0022903
7: -0.0221476, -0.0109538, -0.0217854, -0.0104309, -0.0084631, 0.0076235
8: 0.9751647, 0.9856695, 0.9754301, 0.9861599, -0.0082311, 0.0075280
9: -0.0005793, 0.0068189, -0.0009249, 0.0065934, -0.0050902, 0.0056322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040839, upper bound: 0.0042665
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0153636, 0.0180278, -0.0023805, 0.0021601
1: -0.0018679, 0.0002019, -0.0017211, 0.0001983, -0.0017408, 0.0015813
2: 0.0036398, 0.0045718, 0.0036433, 0.0045010, -0.0006911, 0.0007618
3: 0.0013576, 0.0027145, 0.0013826, 0.0026914, -0.0009632, 0.0009483
4: -0.0045170, -0.0025811, -0.0045041, -0.0026951, -0.0012478, 0.0013629
5: -0.0002733, 0.0009785, -0.0002715, 0.0008917, -0.0009641, 0.0010591
6: -0.0050033, -0.0018011, -0.0049073, -0.0018278, -0.0021163, 0.0020336
7: -0.0221476, -0.0109538, -0.0220766, -0.0116342, -0.0072550, 0.0079368
8: 0.9751647, 0.9856695, 0.9752277, 0.9849489, -0.0070631, 0.0077785
9: -0.0005793, 0.0068189, -0.0001135, 0.0067747, -0.0052909, 0.0048270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046874, upper bound: 0.0043819
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044541, upper bound: 0.0043952
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0151446, 0.0180356, -0.0023354, 0.0023354
1: -0.0018679, 0.0002019, -0.0018679, 0.0002019, -0.0017152, 0.0017152
2: 0.0036398, 0.0045718, 0.0036398, 0.0045718, -0.0007468, 0.0007468
3: 0.0013576, 0.0027145, 0.0013576, 0.0027145, -0.0009784, 0.0009784
4: -0.0045170, -0.0025811, -0.0045170, -0.0025811, -0.0013299, 0.0013299
5: -0.0002733, 0.0009785, -0.0002733, 0.0009785, -0.0010456, 0.0010456
6: -0.0050033, -0.0018011, -0.0050033, -0.0018011, -0.0020845, 0.0020845
7: -0.0221476, -0.0109538, -0.0221476, -0.0109538, -0.0077469, 0.0077469
8: 0.9751647, 0.9856695, 0.9751647, 0.9856695, -0.0076042, 0.0076042
9: -0.0005793, 0.0068189, -0.0005793, 0.0068189, -0.0051655, 0.0051655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046874, upper bound: 0.0043819
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044541, upper bound: 0.0043952
time: 0.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.01 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044713, upper bound: 0.0040183
IS_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0039848
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044713, upper bound: 0.0043189
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0042960
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0037065, upper bound: 0.0043648
IS_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0036946
IS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0041467, upper bound: 0.0037916
IS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0041078
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0040788, upper bound: 0.0044166
IS_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0040788, upper bound: 0.0044166
IS_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0046088, upper bound: 0.0044421
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044877
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044154, upper bound: 0.0046840
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044877
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0043648, upper bound: 0.0037065
IS_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0034984
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0043648, upper bound: 0.0040788
IS_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0038840
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0041938, upper bound: 0.0036994
IS_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0041938, upper bound: 0.0040772
IS_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0040839, upper bound: 0.0042665
IS_A2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0040839, upper bound: 0.0042665
IS_A2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0046874, upper bound: 0.0043819
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044541, upper bound: 0.0043952
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0046874, upper bound: 0.0043819
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.01
Output dim: 8, lower bound: -0.0044541, upper bound: 0.0043952

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0152687, 0.0180188, -0.0021637, 0.0022303
1: -0.0017270, 0.0002047, -0.0017754, 0.0002050, -0.0015911, 0.0016404
2: 0.0036482, 0.0045111, 0.0036480, 0.0045326, -0.0007125, 0.0006914
3: 0.0013488, 0.0026240, 0.0013484, 0.0026366, -0.0009266, 0.0009157
4: -0.0044353, -0.0026347, -0.0044353, -0.0026111, -0.0012465, 0.0012225
5: -0.0002787, 0.0008921, -0.0002789, 0.0009215, -0.0010004, 0.0009703
6: -0.0050098, -0.0020174, -0.0050114, -0.0020146, -0.0019362, 0.0019381
7: -0.0216981, -0.0112933, -0.0216981, -0.0111475, -0.0072642, 0.0071177
8: 0.9754941, 0.9852356, 0.9754941, 0.9854095, -0.0071492, 0.0069750
9: -0.0003342, 0.0065373, -0.0004374, 0.0065373, -0.0047428, 0.0048455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0039848
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0039848
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0153636, 0.0180278, -0.0021748, 0.0021547
1: -0.0017270, 0.0002047, -0.0017211, 0.0001983, -0.0015840, 0.0015890
2: 0.0036482, 0.0045111, 0.0036433, 0.0045010, -0.0006878, 0.0006971
3: 0.0013488, 0.0026240, 0.0013826, 0.0026914, -0.0009939, 0.0009072
4: -0.0044353, -0.0026347, -0.0045041, -0.0026951, -0.0011960, 0.0013186
5: -0.0002787, 0.0008921, -0.0002715, 0.0008917, -0.0009701, 0.0009627
6: -0.0050098, -0.0020174, -0.0049073, -0.0018278, -0.0021750, 0.0019006
7: -0.0216981, -0.0112933, -0.0220766, -0.0116342, -0.0069704, 0.0076485
8: 0.9754941, 0.9852356, 0.9752277, 0.9849489, -0.0068656, 0.0073542
9: -0.0003342, 0.0065373, -0.0001135, 0.0067747, -0.0050754, 0.0046494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0153922, 0.0181438, 0.0154084, 0.0180271, -0.0021702, 0.0022499
1: -0.0016849, 0.0002951, -0.0016871, 0.0001976, -0.0015793, 0.0016544
2: 0.0036075, 0.0044937, 0.0036435, 0.0044871, -0.0007185, 0.0006960
3: 0.0013296, 0.0026276, 0.0013831, 0.0026772, -0.0010053, 0.0009459
4: -0.0044820, -0.0026568, -0.0045040, -0.0027106, -0.0012239, 0.0013117
5: -0.0003318, 0.0008672, -0.0002711, 0.0008708, -0.0010080, 0.0009603
6: -0.0050155, -0.0019585, -0.0049055, -0.0018301, -0.0021688, 0.0019896
7: -0.0219863, -0.0114240, -0.0220766, -0.0117287, -0.0071450, 0.0076123
8: 0.9751433, 0.9850889, 0.9752277, 0.9848412, -0.0070899, 0.0073307
9: -0.0002416, 0.0067398, -0.0000482, 0.0067747, -0.0050525, 0.0047733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0152687, 0.0180188, 0.0151112, 0.0180263, -0.0022436, 0.0024075
1: -0.0017754, 0.0002050, -0.0018771, 0.0002079, -0.0016445, 0.0017596
2: 0.0036480, 0.0045326, 0.0036446, 0.0045845, -0.0007705, 0.0007176
3: 0.0013484, 0.0026366, 0.0013199, 0.0026486, -0.0009392, 0.0009648
4: -0.0044353, -0.0026111, -0.0044509, -0.0025157, -0.0013724, 0.0012782
5: -0.0002789, 0.0009215, -0.0002804, 0.0009800, -0.0010702, 0.0010019
6: -0.0050114, -0.0020146, -0.0051082, -0.0019922, -0.0019950, 0.0020809
7: -0.0216981, -0.0111475, -0.0217854, -0.0105834, -0.0080009, 0.0074434
8: 0.9754941, 0.9854095, 0.9754301, 0.9859780, -0.0078658, 0.0072947
9: -0.0004374, 0.0065373, -0.0008174, 0.0065934, -0.0049607, 0.0053381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0036946
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0036946
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0151446, 0.0180356, -0.0021880, 0.0023877
1: -0.0017270, 0.0002047, -0.0018679, 0.0002019, -0.0015875, 0.0017511
2: 0.0036482, 0.0045111, 0.0036398, 0.0045718, -0.0007635, 0.0007020
3: 0.0013488, 0.0026240, 0.0013576, 0.0027145, -0.0010164, 0.0009413
4: -0.0044353, -0.0026347, -0.0045170, -0.0025811, -0.0013368, 0.0013449
5: -0.0002787, 0.0008921, -0.0002733, 0.0009785, -0.0010666, 0.0009643
6: -0.0050098, -0.0020174, -0.0050033, -0.0018011, -0.0022258, 0.0020332
7: -0.0216981, -0.0112933, -0.0221476, -0.0109538, -0.0077979, 0.0077980
8: 0.9754941, 0.9852356, 0.9751647, 0.9856695, -0.0076924, 0.0074867
9: -0.0003342, 0.0065373, -0.0005793, 0.0068189, -0.0051723, 0.0052062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0153922, 0.0181438, 0.0151884, 0.0180349, -0.0021833, 0.0024824
1: -0.0016849, 0.0002951, -0.0018351, 0.0002012, -0.0015827, 0.0018164
2: 0.0036075, 0.0044937, 0.0036400, 0.0045579, -0.0007940, 0.0007009
3: 0.0013296, 0.0026276, 0.0013581, 0.0026997, -0.0010277, 0.0009803
4: -0.0044820, -0.0026568, -0.0045170, -0.0025957, -0.0013647, 0.0013380
5: -0.0003318, 0.0008672, -0.0002729, 0.0009574, -0.0011045, 0.0009618
6: -0.0050155, -0.0019585, -0.0050017, -0.0018032, -0.0022197, 0.0021225
7: -0.0219863, -0.0114240, -0.0221475, -0.0110428, -0.0079727, 0.0077618
8: 0.9751433, 0.9850889, 0.9751647, 0.9855638, -0.0079184, 0.0074632
9: -0.0002416, 0.0067398, -0.0005171, 0.0068189, -0.0051494, 0.0053306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0153362, 0.0180184, -0.0021547, 0.0021748
1: -0.0017211, 0.0001983, -0.0017270, 0.0002047, -0.0015890, 0.0015840
2: 0.0036433, 0.0045010, 0.0036482, 0.0045111, -0.0006971, 0.0006878
3: 0.0013826, 0.0026914, 0.0013488, 0.0026240, -0.0009072, 0.0009939
4: -0.0045041, -0.0026951, -0.0044353, -0.0026347, -0.0013186, 0.0011960
5: -0.0002715, 0.0008917, -0.0002787, 0.0008921, -0.0009627, 0.0009701
6: -0.0049073, -0.0018278, -0.0050098, -0.0020174, -0.0019006, 0.0021750
7: -0.0220766, -0.0116342, -0.0216981, -0.0112933, -0.0076485, 0.0069704
8: 0.9752277, 0.9849489, 0.9754941, 0.9852356, -0.0073542, 0.0068656
9: -0.0001135, 0.0067747, -0.0003342, 0.0065373, -0.0046494, 0.0050754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042960, upper bound: 0.0040473
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042960, upper bound: 0.0040482
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0151112, 0.0180263, -0.0021679, 0.0024185
1: -0.0017211, 0.0001983, -0.0018771, 0.0002079, -0.0015930, 0.0017525
2: 0.0036433, 0.0045010, 0.0036446, 0.0045845, -0.0007763, 0.0006929
3: 0.0013826, 0.0026914, 0.0013199, 0.0026486, -0.0009307, 0.0010321
4: -0.0045041, -0.0026951, -0.0044509, -0.0025157, -0.0014686, 0.0012277
5: -0.0002715, 0.0008917, -0.0002804, 0.0009800, -0.0010627, 0.0009716
6: -0.0049073, -0.0018278, -0.0051082, -0.0019922, -0.0019575, 0.0023197
7: -0.0220766, -0.0116342, -0.0217854, -0.0105834, -0.0085317, 0.0071495
8: 0.9752277, 0.9849489, 0.9754301, 0.9859780, -0.0082449, 0.0070111
9: -0.0001135, 0.0067747, -0.0008174, 0.0065934, -0.0047646, 0.0056707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0154290, 0.0180274, 0.0153636, 0.0180278, -0.0020781, 0.0021447
1: -0.0016738, 0.0001979, -0.0017211, 0.0001983, -0.0015275, 0.0015761
2: 0.0036435, 0.0044800, 0.0036433, 0.0045010, -0.0006854, 0.0006643
3: 0.0013829, 0.0026800, 0.0013826, 0.0026914, -0.0009214, 0.0009114
4: -0.0045040, -0.0027187, -0.0045041, -0.0026951, -0.0012162, 0.0011926
5: -0.0002713, 0.0008631, -0.0002715, 0.0008917, -0.0009613, 0.0009320
6: -0.0049056, -0.0018310, -0.0049073, -0.0018278, -0.0019689, 0.0019706
7: -0.0220766, -0.0117779, -0.0220766, -0.0116342, -0.0070759, 0.0069319
8: 0.9752277, 0.9847767, 0.9752277, 0.9849489, -0.0069183, 0.0067488
9: -0.0000130, 0.0067747, -0.0001135, 0.0067747, -0.0046108, 0.0047117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0045381
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0045381
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0154943, 0.0181521, 0.0154084, 0.0180271, -0.0020648, 0.0022418
1: -0.0016245, 0.0002897, -0.0016871, 0.0001976, -0.0015201, 0.0016431
2: 0.0036032, 0.0044591, 0.0036435, 0.0044871, -0.0007166, 0.0006601
3: 0.0013621, 0.0026841, 0.0013831, 0.0026772, -0.0009329, 0.0009537
4: -0.0045507, -0.0027440, -0.0045040, -0.0027106, -0.0012445, 0.0011829
5: -0.0003257, 0.0008341, -0.0002711, 0.0008708, -0.0010009, 0.0009287
6: -0.0049122, -0.0017752, -0.0049055, -0.0018301, -0.0019628, 0.0020587
7: -0.0223662, -0.0119301, -0.0220766, -0.0117287, -0.0072531, 0.0068797
8: 0.9748762, 0.9846075, 0.9752277, 0.9848412, -0.0071465, 0.0067050
9: 0.0000921, 0.0069784, -0.0000482, 0.0067747, -0.0045770, 0.0048374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0046083
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0046083
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0153636, 0.0180278, 0.0152116, 0.0180352, -0.0021584, 0.0023122
1: -0.0017211, 0.0001983, -0.0018193, 0.0002015, -0.0015796, 0.0016901
2: 0.0036433, 0.0045010, 0.0036399, 0.0045504, -0.0007402, 0.0006907
3: 0.0013826, 0.0026914, 0.0013580, 0.0027031, -0.0009349, 0.0009599
4: -0.0045041, -0.0026951, -0.0045170, -0.0026057, -0.0013389, 0.0012478
5: -0.0002715, 0.0008917, -0.0002731, 0.0009492, -0.0010282, 0.0009629
6: -0.0049073, -0.0018278, -0.0050017, -0.0018042, -0.0020270, 0.0021079
7: -0.0220766, -0.0116342, -0.0221476, -0.0111046, -0.0077906, 0.0072550
8: 0.9752277, 0.9849489, 0.9751647, 0.9854926, -0.0076046, 0.0070631
9: -0.0001135, 0.0067747, -0.0004736, 0.0068189, -0.0048270, 0.0051881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044154, upper bound: 0.0044439
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044154, upper bound: 0.0044877
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0154084, 0.0180271, 0.0152727, 0.0181633, -0.0022543, 0.0022992
1: -0.0016871, 0.0001976, -0.0017739, 0.0002930, -0.0016472, 0.0016808
2: 0.0036435, 0.0044871, 0.0035992, 0.0045309, -0.0007362, 0.0007211
3: 0.0013831, 0.0026772, 0.0013382, 0.0027067, -0.0009766, 0.0009673
4: -0.0045040, -0.0027106, -0.0045657, -0.0026282, -0.0013286, 0.0012783
5: -0.0002711, 0.0008708, -0.0003274, 0.0009201, -0.0010231, 0.0010026
6: -0.0049055, -0.0018301, -0.0050083, -0.0017481, -0.0021160, 0.0021016
7: -0.0220766, -0.0117287, -0.0224520, -0.0112394, -0.0077351, 0.0074451
8: 0.9752277, 0.9848412, 0.9748068, 0.9853299, -0.0075590, 0.0073029
9: -0.0000482, 0.0067747, -0.0003790, 0.0070340, -0.0049616, 0.0051524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044439
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044877
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0151112, 0.0180263, 0.0152687, 0.0180188, -0.0024075, 0.0022436
1: -0.0018771, 0.0002079, -0.0017754, 0.0002050, -0.0017596, 0.0016445
2: 0.0036446, 0.0045845, 0.0036480, 0.0045326, -0.0007176, 0.0007705
3: 0.0013199, 0.0026486, 0.0013484, 0.0026366, -0.0009648, 0.0009392
4: -0.0044509, -0.0025157, -0.0044353, -0.0026111, -0.0012782, 0.0013724
5: -0.0002804, 0.0009800, -0.0002789, 0.0009215, -0.0010019, 0.0010702
6: -0.0051082, -0.0019922, -0.0050114, -0.0020146, -0.0020809, 0.0019950
7: -0.0217854, -0.0105834, -0.0216981, -0.0111475, -0.0074434, 0.0080009
8: 0.9754301, 0.9859780, 0.9754941, 0.9854095, -0.0072947, 0.0078658
9: -0.0008174, 0.0065934, -0.0004374, 0.0065373, -0.0053381, 0.0049607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0034984
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0034984
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0151112, 0.0180263, 0.0153636, 0.0180278, -0.0024185, 0.0021679
1: -0.0018771, 0.0002079, -0.0017211, 0.0001983, -0.0017526, 0.0015930
2: 0.0036446, 0.0045845, 0.0036433, 0.0045010, -0.0006929, 0.0007763
3: 0.0013199, 0.0026486, 0.0013826, 0.0026914, -0.0010321, 0.0009307
4: -0.0044509, -0.0025157, -0.0045041, -0.0026951, -0.0012277, 0.0014686
5: -0.0002804, 0.0009800, -0.0002715, 0.0008917, -0.0009716, 0.0010627
6: -0.0051082, -0.0019922, -0.0049073, -0.0018278, -0.0023197, 0.0019575
7: -0.0217854, -0.0105834, -0.0220766, -0.0116342, -0.0071495, 0.0085317
8: 0.9754301, 0.9859780, 0.9752277, 0.9849489, -0.0070111, 0.0082449
9: -0.0008174, 0.0065934, -0.0001135, 0.0067747, -0.0056707, 0.0047646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038108, upper bound: 0.0038840
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038108, upper bound: 0.0038840
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0151112, 0.0180263, 0.0150412, 0.0180267, -0.0023610, 0.0024288
1: -0.0018771, 0.0002079, -0.0019284, 0.0002083, -0.0017327, 0.0017832
2: 0.0036446, 0.0045845, 0.0036444, 0.0046066, -0.0007768, 0.0007552
3: 0.0013199, 0.0026486, 0.0013196, 0.0026613, -0.0009806, 0.0009698
4: -0.0044509, -0.0025157, -0.0044509, -0.0024907, -0.0013633, 0.0013391
5: -0.0002804, 0.0009800, -0.0002807, 0.0010107, -0.0010873, 0.0010568
6: -0.0051082, -0.0019922, -0.0051098, -0.0019891, -0.0020443, 0.0020459
7: -0.0217854, -0.0105834, -0.0217854, -0.0104309, -0.0079537, 0.0078059
8: 0.9754301, 0.9859780, 0.9754301, 0.9861599, -0.0078583, 0.0076809
9: -0.0008174, 0.0065934, -0.0009249, 0.0065934, -0.0052079, 0.0053121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
time: 0.64 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
time: 0.77 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0151112, 0.0180263, 0.0151446, 0.0180356, -0.0023721, 0.0023432
1: -0.0018771, 0.0002079, -0.0018679, 0.0002019, -0.0017258, 0.0017252
2: 0.0036446, 0.0045845, 0.0036398, 0.0045718, -0.0007489, 0.0007610
3: 0.0013199, 0.0026486, 0.0013576, 0.0027145, -0.0010460, 0.0009580
4: -0.0044509, -0.0025157, -0.0045170, -0.0025811, -0.0013069, 0.0014310
5: -0.0002804, 0.0009800, -0.0002733, 0.0009785, -0.0010538, 0.0010492
6: -0.0051082, -0.0019922, -0.0050033, -0.0018011, -0.0022821, 0.0020015
7: -0.0217854, -0.0105834, -0.0221476, -0.0109538, -0.0076235, 0.0083152
8: 0.9754301, 0.9859780, 0.9751647, 0.9856695, -0.0075280, 0.0080537
9: -0.0008174, 0.0065934, -0.0005793, 0.0068189, -0.0055280, 0.0050902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036757, upper bound: 0.0038829
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036757, upper bound: 0.0038829
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0153362, 0.0180184, -0.0023877, 0.0021880
1: -0.0018679, 0.0002019, -0.0017270, 0.0002047, -0.0017511, 0.0015875
2: 0.0036398, 0.0045718, 0.0036482, 0.0045111, -0.0007020, 0.0007635
3: 0.0013576, 0.0027145, 0.0013488, 0.0026240, -0.0009413, 0.0010164
4: -0.0045170, -0.0025811, -0.0044353, -0.0026347, -0.0013449, 0.0013368
5: -0.0002733, 0.0009785, -0.0002787, 0.0008921, -0.0009643, 0.0010666
6: -0.0050033, -0.0018011, -0.0050098, -0.0020174, -0.0020332, 0.0022258
7: -0.0221476, -0.0109538, -0.0216981, -0.0112933, -0.0077980, 0.0077979
8: 0.9751647, 0.9856695, 0.9754941, 0.9852356, -0.0074867, 0.0076924
9: -0.0005793, 0.0068189, -0.0003342, 0.0065373, -0.0052062, 0.0051723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041078, upper bound: 0.0036907
time: 0.84 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041078, upper bound: 0.0036907
time: 0.69 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0151446, 0.0180356, 0.0151112, 0.0180263, -0.0023432, 0.0023721
1: -0.0018679, 0.0002019, -0.0018771, 0.0002079, -0.0017252, 0.0017258
2: 0.0036398, 0.0045718, 0.0036446, 0.0045845, -0.0007610, 0.0007489
3: 0.0013576, 0.0027145, 0.0013199, 0.0026486, -0.0009580, 0.0010460
4: -0.0045170, -0.0025811, -0.0044509, -0.0025157, -0.0014310, 0.0013069
5: -0.0002733, 0.0009785, -0.0002804, 0.0009800, -0.0010492, 0.0010538
6: -0.0050033, -0.0018011, -0.0051082, -0.0019922, -0.0020015, 0.0022821
7: -0.0221476, -0.0109538, -0.0217854, -0.0105834, -0.0083152, 0.0076235
8: 0.9751647, 0.9856695, 0.9754301, 0.9859780, -0.0080537, 0.0075280
9: -0.0005793, 0.0068189, -0.0008174, 0.0065934, -0.0050902, 0.0055280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0153636, 0.0180278, -0.0023122, 0.0021584
1: -0.0018193, 0.0002015, -0.0017211, 0.0001983, -0.0016901, 0.0015796
2: 0.0036399, 0.0045504, 0.0036433, 0.0045010, -0.0006907, 0.0007402
3: 0.0013580, 0.0027031, 0.0013826, 0.0026914, -0.0009599, 0.0009349
4: -0.0045170, -0.0026057, -0.0045041, -0.0026951, -0.0012478, 0.0013389
5: -0.0002731, 0.0009492, -0.0002715, 0.0008917, -0.0009629, 0.0010282
6: -0.0050017, -0.0018042, -0.0049073, -0.0018278, -0.0021079, 0.0020270
7: -0.0221476, -0.0111046, -0.0220766, -0.0116342, -0.0072550, 0.0077906
8: 0.9751647, 0.9854926, 0.9752277, 0.9849489, -0.0070631, 0.0076046
9: -0.0004736, 0.0068189, -0.0001135, 0.0067747, -0.0051881, 0.0048270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043819
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043819
time: 0.74 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0154084, 0.0180271, -0.0022992, 0.0022543
1: -0.0017739, 0.0002930, -0.0016871, 0.0001976, -0.0016808, 0.0016472
2: 0.0035992, 0.0045309, 0.0036435, 0.0044871, -0.0007211, 0.0007362
3: 0.0013382, 0.0027067, 0.0013831, 0.0026772, -0.0009673, 0.0009766
4: -0.0045657, -0.0026282, -0.0045040, -0.0027106, -0.0012783, 0.0013286
5: -0.0003274, 0.0009201, -0.0002711, 0.0008708, -0.0010026, 0.0010231
6: -0.0050083, -0.0017481, -0.0049055, -0.0018301, -0.0021016, 0.0021160
7: -0.0224520, -0.0112394, -0.0220766, -0.0117287, -0.0074451, 0.0077351
8: 0.9748068, 0.9853299, 0.9752277, 0.9848412, -0.0073029, 0.0075590
9: -0.0003790, 0.0070340, -0.0000482, 0.0067747, -0.0051524, 0.0049616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043952
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043952
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0151446, 0.0180356, -0.0022666, 0.0023336
1: -0.0018193, 0.0002015, -0.0018679, 0.0002019, -0.0016644, 0.0017133
2: 0.0036399, 0.0045504, 0.0036398, 0.0045718, -0.0007463, 0.0007249
3: 0.0013580, 0.0027031, 0.0013576, 0.0027145, -0.0009751, 0.0009648
4: -0.0045170, -0.0026057, -0.0045170, -0.0025811, -0.0013299, 0.0013061
5: -0.0002731, 0.0009492, -0.0002733, 0.0009785, -0.0010443, 0.0010147
6: -0.0050017, -0.0018042, -0.0050033, -0.0018011, -0.0020763, 0.0020780
7: -0.0221476, -0.0111046, -0.0221476, -0.0109538, -0.0077469, 0.0076020
8: 0.9751647, 0.9854926, 0.9751647, 0.9856695, -0.0076042, 0.0074314
9: -0.0004736, 0.0068189, -0.0005793, 0.0068189, -0.0050635, 0.0051655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043819
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043819
time: 0.91 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0151884, 0.0180349, -0.0022566, 0.0024307
1: -0.0017739, 0.0002930, -0.0018351, 0.0002012, -0.0016566, 0.0017797
2: 0.0035992, 0.0045309, 0.0036400, 0.0045579, -0.0007776, 0.0007217
3: 0.0013382, 0.0027067, 0.0013581, 0.0026997, -0.0009832, 0.0010061
4: -0.0045657, -0.0026282, -0.0045170, -0.0025957, -0.0013590, 0.0012960
5: -0.0003274, 0.0009201, -0.0002729, 0.0009574, -0.0010836, 0.0010102
6: -0.0050083, -0.0017481, -0.0050017, -0.0018032, -0.0020704, 0.0021675
7: -0.0224520, -0.0112394, -0.0221475, -0.0110428, -0.0079294, 0.0075453
8: 0.9748068, 0.9853299, 0.9751647, 0.9855638, -0.0078371, 0.0073866
9: -0.0003790, 0.0070340, -0.0005171, 0.0068189, -0.0050265, 0.0052948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043952
time: 0.88 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043952
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.08 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0039848
IS_A1_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0039848, upper bound: 0.0039848
IS_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
IS_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
IS_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
IS_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0040473, upper bound: 0.0042960
IS_A1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0036946
IS_A1_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0034984, upper bound: 0.0036946
IS_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
IS_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
IS_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
IS_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036907, upper bound: 0.0041078
IS_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0042960, upper bound: 0.0040473
IS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0042960, upper bound: 0.0040482
IS_A1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
IS_A1_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0038840, upper bound: 0.0038108
IS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0045381
IS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0045381
IS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0046083
IS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0045840, upper bound: 0.0046083
IS_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044154, upper bound: 0.0044439
IS_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044154, upper bound: 0.0044877
IS_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044439
IS_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044255, upper bound: 0.0044877
IS_A2_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0034984
IS_A2_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036946, upper bound: 0.0034984
IS_A2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0038108, upper bound: 0.0038840
IS_A2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0038108, upper bound: 0.0038840
IS_A2_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
IS_A2_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0034902, upper bound: 0.0034890
IS_A2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036757, upper bound: 0.0038829
IS_A2_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0036757, upper bound: 0.0038829
IS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0041078, upper bound: 0.0036907
IS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0041078, upper bound: 0.0036907
IS_A2_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
IS_A2_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0038856, upper bound: 0.0036752
IS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043819
IS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043819
IS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043952
IS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044851, upper bound: 0.0043952
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043819
IS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043819
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043952
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 8, lower bound: -0.0044461, upper bound: 0.0043952

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0154290, 0.0180274, -0.0021730, 0.0020871
1: -0.0017270, 0.0002047, -0.0016738, 0.0001979, -0.0015822, 0.0015381
2: 0.0036482, 0.0045111, 0.0036435, 0.0044800, -0.0006664, 0.0006966
3: 0.0013488, 0.0026240, 0.0013829, 0.0026800, -0.0009801, 0.0009040
4: -0.0044353, -0.0026347, -0.0045040, -0.0027187, -0.0011728, 0.0013186
5: -0.0002787, 0.0008921, -0.0002713, 0.0008631, -0.0009396, 0.0009615
6: -0.0050098, -0.0020174, -0.0049056, -0.0018310, -0.0021671, 0.0018931
7: -0.0216981, -0.0112933, -0.0220766, -0.0117779, -0.0068296, 0.0076485
8: 0.9754941, 0.9852356, 0.9752277, 0.9847767, -0.0066980, 0.0073542
9: -0.0003342, 0.0065373, -0.0000130, 0.0067747, -0.0050754, 0.0045508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042448, upper bound: 0.0039919
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0154943, 0.0181521, -0.0023101, 0.0020189
1: -0.0017270, 0.0002047, -0.0016245, 0.0002897, -0.0016809, 0.0014890
2: 0.0036482, 0.0045111, 0.0036032, 0.0044591, -0.0006448, 0.0007403
3: 0.0013488, 0.0026240, 0.0013621, 0.0026841, -0.0009883, 0.0009303
4: -0.0044353, -0.0026347, -0.0045507, -0.0027440, -0.0011426, 0.0013609
5: -0.0002787, 0.0008921, -0.0003257, 0.0008341, -0.0009105, 0.0010205
6: -0.0050098, -0.0020174, -0.0049122, -0.0017752, -0.0022211, 0.0019003
7: -0.0216981, -0.0112933, -0.0223662, -0.0119301, -0.0066469, 0.0079135
8: 0.9754941, 0.9852356, 0.9748762, 0.9846075, -0.0064925, 0.0076900
9: -0.0003342, 0.0065373, 0.0000921, 0.0069784, -0.0052641, 0.0044238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042448, upper bound: 0.0039919
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0153922, 0.0181438, 0.0154290, 0.0180274, -0.0021161, 0.0022255
1: -0.0016849, 0.0002951, -0.0016738, 0.0001979, -0.0015421, 0.0016363
2: 0.0036075, 0.0044937, 0.0036435, 0.0044800, -0.0007107, 0.0006783
3: 0.0013296, 0.0026276, 0.0013829, 0.0026800, -0.0010054, 0.0009088
4: -0.0044820, -0.0026568, -0.0045040, -0.0027187, -0.0012171, 0.0012953
5: -0.0003318, 0.0008672, -0.0002713, 0.0008631, -0.0009975, 0.0009378
6: -0.0050155, -0.0019585, -0.0049056, -0.0018310, -0.0021723, 0.0019540
7: -0.0219863, -0.0114240, -0.0220766, -0.0117779, -0.0071038, 0.0075070
8: 0.9751433, 0.9850889, 0.9752277, 0.9847767, -0.0070380, 0.0071940
9: -0.0002416, 0.0067398, -0.0000130, 0.0067747, -0.0049759, 0.0047445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034705, upper bound: 0.0037869
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029881, upper bound: 0.0034206
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0153922, 0.0181438, 0.0154943, 0.0181521, -0.0021593, 0.0020688
1: -0.0016849, 0.0002951, -0.0016245, 0.0002897, -0.0015691, 0.0015235
2: 0.0036075, 0.0044937, 0.0036032, 0.0044591, -0.0006613, 0.0006928
3: 0.0013296, 0.0026276, 0.0013621, 0.0026841, -0.0010147, 0.0009357
4: -0.0044820, -0.0026568, -0.0045507, -0.0027440, -0.0011686, 0.0013174
5: -0.0003318, 0.0008672, -0.0003257, 0.0008341, -0.0009315, 0.0009535
6: -0.0050155, -0.0019585, -0.0049122, -0.0017752, -0.0022437, 0.0019745
7: -0.0219863, -0.0114240, -0.0223662, -0.0119301, -0.0068060, 0.0076430
8: 0.9751433, 0.9850889, 0.9748762, 0.9846075, -0.0066715, 0.0073448
9: -0.0002416, 0.0067398, 0.0000921, 0.0069784, -0.0050706, 0.0045340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031090, upper bound: 0.0035234
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029881, upper bound: 0.0034206
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0152116, 0.0180352, -0.0021863, 0.0023201
1: -0.0017270, 0.0002047, -0.0018193, 0.0002015, -0.0015857, 0.0017005
2: 0.0036482, 0.0045111, 0.0036399, 0.0045504, -0.0007420, 0.0007015
3: 0.0013488, 0.0026240, 0.0013580, 0.0027031, -0.0010028, 0.0009382
4: -0.0044353, -0.0026347, -0.0045170, -0.0026057, -0.0013135, 0.0013448
5: -0.0002787, 0.0008921, -0.0002731, 0.0009492, -0.0010359, 0.0009631
6: -0.0050098, -0.0020174, -0.0050017, -0.0018042, -0.0022179, 0.0020256
7: -0.0216981, -0.0112933, -0.0221476, -0.0111046, -0.0076561, 0.0077980
8: 0.9754941, 0.9852356, 0.9751647, 0.9854926, -0.0075235, 0.0074867
9: -0.0003342, 0.0065373, -0.0004736, 0.0068189, -0.0051723, 0.0051068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039511, upper bound: 0.0038167
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0153362, 0.0180184, 0.0152727, 0.0181633, -0.0023218, 0.0022543
1: -0.0017270, 0.0002047, -0.0017739, 0.0002930, -0.0016852, 0.0016534
2: 0.0036482, 0.0045111, 0.0035992, 0.0045309, -0.0007209, 0.0007448
3: 0.0013488, 0.0026240, 0.0013382, 0.0027067, -0.0010120, 0.0009633
4: -0.0044353, -0.0026347, -0.0045657, -0.0026282, -0.0012837, 0.0013905
5: -0.0002787, 0.0008921, -0.0003274, 0.0009201, -0.0010073, 0.0010223
6: -0.0050098, -0.0020174, -0.0050083, -0.0017481, -0.0022725, 0.0020325
7: -0.0216981, -0.0112933, -0.0224520, -0.0112394, -0.0074782, 0.0080809
8: 0.9754941, 0.9852356, 0.9748068, 0.9853299, -0.0073301, 0.0078222
9: -0.0003342, 0.0065373, -0.0003790, 0.0070340, -0.0053714, 0.0049849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040013, upper bound: 0.0038618
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0153922, 0.0181438, 0.0152116, 0.0180352, -0.0021294, 0.0024585
1: -0.0016849, 0.0002951, -0.0018193, 0.0002015, -0.0015456, 0.0017987
2: 0.0036075, 0.0044937, 0.0036399, 0.0045504, -0.0007864, 0.0006832
3: 0.0013296, 0.0026276, 0.0013580, 0.0027031, -0.0010281, 0.0009430
4: -0.0044820, -0.0026568, -0.0045170, -0.0026057, -0.0013578, 0.0013215
5: -0.0003318, 0.0008672, -0.0002731, 0.0009492, -0.0010938, 0.0009394
6: -0.0050155, -0.0019585, -0.0050017, -0.0018042, -0.0022231, 0.0020865
7: -0.0219863, -0.0114240, -0.0221476, -0.0111046, -0.0079302, 0.0076565
8: 0.9751433, 0.9850889, 0.9751647, 0.9854926, -0.0078635, 0.0073265
9: -0.0002416, 0.0067398, -0.0004736, 0.0068189, -0.0050727, 0.0053004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029535, upper bound: 0.0034145
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: A, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: A, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 81
type: B, layer: 3, pos: 225
type: A, layer: 3, pos: 115
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 122
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: A, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: A, layer: 3, pos: 99
type: B, layer: 3, pos: 118

Time for candidate selection: 8.32 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029196, upper bound: 0.0023957
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029193, upper bound: 0.0033405
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0153922, 0.0181438, 0.0152727, 0.0181633, -0.0021721, 0.0023011
1: -0.0016849, 0.0002951, -0.0017739, 0.0002930, -0.0015721, 0.0016844
2: 0.0036075, 0.0044937, 0.0035992, 0.0045309, -0.0007365, 0.0006977
3: 0.0013296, 0.0026276, 0.0013382, 0.0027067, -0.0010377, 0.0009712
4: -0.0044820, -0.0026568, -0.0045657, -0.0026282, -0.0013087, 0.0013443
5: -0.0003318, 0.0008672, -0.0003274, 0.0009201, -0.0010259, 0.0009548
6: -0.0050155, -0.0019585, -0.0050083, -0.0017481, -0.0022958, 0.0021079
7: -0.0219863, -0.0114240, -0.0224520, -0.0112394, -0.0076301, 0.0077953
8: 0.9751433, 0.9850889, 0.9748068, 0.9853299, -0.0074981, 0.0074772
9: -0.0002416, 0.0067398, -0.0003790, 0.0070340, -0.0051687, 0.0050890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029535, upper bound: 0.0034145
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: A, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: A, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 225
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 81
type: B, layer: 3, pos: 115
type: A, layer: 3, pos: 115
type: B, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 122
type: A, layer: 3, pos: 122
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: A, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: A, layer: 3, pos: 99

Time for candidate selection: 8.37 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029196, upper bound: 0.0023957
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029193, upper bound: 0.0033405
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0154290, 0.0180274, 0.0153362, 0.0180184, -0.0020871, 0.0021730
1: -0.0016738, 0.0001979, -0.0017270, 0.0002047, -0.0015381, 0.0015822
2: 0.0036435, 0.0044800, 0.0036482, 0.0045111, -0.0006966, 0.0006664
3: 0.0013829, 0.0026800, 0.0013488, 0.0026240, -0.0009040, 0.0009801
4: -0.0045040, -0.0027187, -0.0044353, -0.0026347, -0.0013186, 0.0011728
5: -0.0002713, 0.0008631, -0.0002787, 0.0008921, -0.0009615, 0.0009396
6: -0.0049056, -0.0018310, -0.0050098, -0.0020174, -0.0018931, 0.0021671
7: -0.0220766, -0.0117779, -0.0216981, -0.0112933, -0.0076485, 0.0068296
8: 0.9752277, 0.9847767, 0.9754941, 0.9852356, -0.0073542, 0.0066980
9: -0.0000130, 0.0067747, -0.0003342, 0.0065373, -0.0045508, 0.0050754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039919, upper bound: 0.0042448
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0154943, 0.0181521, 0.0153362, 0.0180184, -0.0020189, 0.0023101
1: -0.0016245, 0.0002897, -0.0017270, 0.0002047, -0.0014890, 0.0016809
2: 0.0036032, 0.0044591, 0.0036482, 0.0045111, -0.0007403, 0.0006448
3: 0.0013621, 0.0026841, 0.0013488, 0.0026240, -0.0009303, 0.0009883
4: -0.0045507, -0.0027440, -0.0044353, -0.0026347, -0.0013609, 0.0011426
5: -0.0003257, 0.0008341, -0.0002787, 0.0008921, -0.0010205, 0.0009105
6: -0.0049122, -0.0017752, -0.0050098, -0.0020174, -0.0019003, 0.0022211
7: -0.0223662, -0.0119301, -0.0216981, -0.0112933, -0.0079135, 0.0066469
8: 0.9748762, 0.9846075, 0.9754941, 0.9852356, -0.0076900, 0.0064924
9: 0.0000921, 0.0069784, -0.0003342, 0.0065373, -0.0044238, 0.0052641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039919, upper bound: 0.0042448
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0154290, 0.0180274, 0.0154290, 0.0180274, -0.0020764, 0.0020764
1: -0.0016738, 0.0001979, -0.0016738, 0.0001979, -0.0015257, 0.0015257
2: 0.0036435, 0.0044800, 0.0036435, 0.0044800, -0.0006638, 0.0006638
3: 0.0013829, 0.0026800, 0.0013829, 0.0026800, -0.0009082, 0.0009082
4: -0.0045040, -0.0027187, -0.0045040, -0.0027187, -0.0011926, 0.0011926
5: -0.0002713, 0.0008631, -0.0002713, 0.0008631, -0.0009308, 0.0009308
6: -0.0049056, -0.0018310, -0.0049056, -0.0018310, -0.0019624, 0.0019624
7: -0.0220766, -0.0117779, -0.0220766, -0.0117779, -0.0069319, 0.0069319
8: 0.9752277, 0.9847767, 0.9752277, 0.9847767, -0.0067488, 0.0067488
9: -0.0000130, 0.0067747, -0.0000130, 0.0067747, -0.0046108, 0.0046108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024455, upper bound: 0.0024499
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047325, upper bound: 0.0045029
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0154290, 0.0180274, 0.0154943, 0.0181521, -0.0022168, 0.0020103
1: -0.0016738, 0.0001979, -0.0016245, 0.0002897, -0.0016251, 0.0014785
2: 0.0036435, 0.0044800, 0.0036032, 0.0044591, -0.0006427, 0.0007087
3: 0.0013829, 0.0026800, 0.0013621, 0.0026841, -0.0009122, 0.0009348
4: -0.0045040, -0.0027187, -0.0045507, -0.0027440, -0.0011655, 0.0012367
5: -0.0002713, 0.0008631, -0.0003257, 0.0008341, -0.0009024, 0.0009904
6: -0.0049056, -0.0018310, -0.0049122, -0.0017752, -0.0020210, 0.0019684
7: -0.0220766, -0.0117779, -0.0223662, -0.0119301, -0.0067671, 0.0072055
8: 0.9752277, 0.9847767, 0.9748762, 0.9846075, -0.0065564, 0.0070899
9: -0.0000130, 0.0067747, 0.0000921, 0.0069784, -0.0048042, 0.0044950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024455, upper bound: 0.0024499
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047325, upper bound: 0.0045029
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0154943, 0.0181521, 0.0154290, 0.0180274, -0.0020103, 0.0022168
1: -0.0016245, 0.0002897, -0.0016738, 0.0001979, -0.0014785, 0.0016251
2: 0.0036032, 0.0044591, 0.0036435, 0.0044800, -0.0007087, 0.0006427
3: 0.0013621, 0.0026841, 0.0013829, 0.0026800, -0.0009348, 0.0009122
4: -0.0045507, -0.0027440, -0.0045040, -0.0027187, -0.0012367, 0.0011655
5: -0.0003257, 0.0008341, -0.0002713, 0.0008631, -0.0009904, 0.0009024
6: -0.0049122, -0.0017752, -0.0049056, -0.0018310, -0.0019684, 0.0020210
7: -0.0223662, -0.0119301, -0.0220766, -0.0117779, -0.0072055, 0.0067671
8: 0.9748762, 0.9846075, 0.9752277, 0.9847767, -0.0070899, 0.0065564
9: 0.0000921, 0.0069784, -0.0000130, 0.0067747, -0.0044950, 0.0048042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044216, upper bound: 0.0043719
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0154943, 0.0181521, 0.0154943, 0.0181521, -0.0020539, 0.0020539
1: -0.0016245, 0.0002897, -0.0016245, 0.0002897, -0.0015099, 0.0015099
2: 0.0036032, 0.0044591, 0.0036032, 0.0044591, -0.0006570, 0.0006570
3: 0.0013621, 0.0026841, 0.0013621, 0.0026841, -0.0009444, 0.0009444
4: -0.0045507, -0.0027440, -0.0045507, -0.0027440, -0.0011890, 0.0011890
5: -0.0003257, 0.0008341, -0.0003257, 0.0008341, -0.0009218, 0.0009218
6: -0.0049122, -0.0017752, -0.0049122, -0.0017752, -0.0020465, 0.0020465
7: -0.0223662, -0.0119301, -0.0223662, -0.0119301, -0.0069118, 0.0069119
8: 0.9748762, 0.9846075, 0.9748762, 0.9846075, -0.0067192, 0.0067192
9: 0.0000921, 0.0069784, 0.0000921, 0.0069784, -0.0045956, 0.0045956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044216, upper bound: 0.0043719
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0154290, 0.0180274, 0.0152116, 0.0180352, -0.0020901, 0.0023105
1: -0.0016738, 0.0001979, -0.0018193, 0.0002015, -0.0015291, 0.0016884
2: 0.0036435, 0.0044800, 0.0036399, 0.0045504, -0.0007397, 0.0006691
3: 0.0013829, 0.0026800, 0.0013580, 0.0027031, -0.0009318, 0.0009466
4: -0.0045040, -0.0027187, -0.0045170, -0.0026057, -0.0013388, 0.0012242
5: -0.0002713, 0.0008631, -0.0002731, 0.0009492, -0.0010270, 0.0009324
6: -0.0049056, -0.0018310, -0.0050017, -0.0018042, -0.0020188, 0.0021015
7: -0.0220766, -0.0117779, -0.0221476, -0.0111046, -0.0077905, 0.0071109
8: 0.9752277, 0.9847767, 0.9751647, 0.9854926, -0.0076046, 0.0068936
9: -0.0000130, 0.0067747, -0.0004736, 0.0068189, -0.0047261, 0.0051881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042437, upper bound: 0.0045765
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0154943, 0.0181521, 0.0152116, 0.0180352, -0.0020241, 0.0024509
1: -0.0016245, 0.0002897, -0.0018193, 0.0002015, -0.0014819, 0.0017877
2: 0.0036032, 0.0044591, 0.0036399, 0.0045504, -0.0007846, 0.0006480
3: 0.0013621, 0.0026841, 0.0013580, 0.0027031, -0.0009584, 0.0009506
4: -0.0045507, -0.0027440, -0.0045170, -0.0026057, -0.0013829, 0.0011971
5: -0.0003257, 0.0008341, -0.0002731, 0.0009492, -0.0010867, 0.0009040
6: -0.0049122, -0.0017752, -0.0050017, -0.0018042, -0.0020248, 0.0021601
7: -0.0223662, -0.0119301, -0.0221476, -0.0111046, -0.0080641, 0.0069462
8: 0.9748762, 0.9846075, 0.9751647, 0.9854926, -0.0079457, 0.0067012
9: 0.0000921, 0.0069784, -0.0004736, 0.0068189, -0.0046103, 0.0053815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042437, upper bound: 0.0045765
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0154290, 0.0180274, 0.0152727, 0.0181633, -0.0022293, 0.0022463
1: -0.0016738, 0.0001979, -0.0017739, 0.0002930, -0.0016292, 0.0016415
2: 0.0036435, 0.0044800, 0.0035992, 0.0045309, -0.0007192, 0.0007132
3: 0.0013829, 0.0026800, 0.0013382, 0.0027067, -0.0009362, 0.0009692
4: -0.0045040, -0.0027187, -0.0045657, -0.0026282, -0.0013122, 0.0012706
5: -0.0002713, 0.0008631, -0.0003274, 0.0009201, -0.0009985, 0.0009921
6: -0.0049056, -0.0018310, -0.0050083, -0.0017481, -0.0020766, 0.0021072
7: -0.0220766, -0.0117779, -0.0224520, -0.0112394, -0.0076294, 0.0073974
8: 0.9752277, 0.9847767, 0.9748068, 0.9853299, -0.0074177, 0.0072463
9: -0.0000130, 0.0067747, -0.0003790, 0.0070340, -0.0049284, 0.0050756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042042, upper bound: 0.0042601
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042132
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0154943, 0.0181521, 0.0152727, 0.0181633, -0.0020667, 0.0022883
1: -0.0016245, 0.0002897, -0.0017739, 0.0002930, -0.0015130, 0.0016706
2: 0.0036032, 0.0044591, 0.0035992, 0.0045309, -0.0007330, 0.0006620
3: 0.0013621, 0.0026841, 0.0013382, 0.0027067, -0.0009673, 0.0009836
4: -0.0045507, -0.0027440, -0.0045657, -0.0026282, -0.0013347, 0.0012210
5: -0.0003257, 0.0008341, -0.0003274, 0.0009201, -0.0010162, 0.0009232
6: -0.0049122, -0.0017752, -0.0050083, -0.0017481, -0.0021038, 0.0021872
7: -0.0223662, -0.0119301, -0.0224520, -0.0112394, -0.0077673, 0.0070930
8: 0.9748762, 0.9846075, 0.9748068, 0.9853299, -0.0075732, 0.0068646
9: 0.0000921, 0.0069784, -0.0003790, 0.0070340, -0.0047121, 0.0051710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042185, upper bound: 0.0042500
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0153362, 0.0180184, -0.0023201, 0.0021863
1: -0.0018193, 0.0002015, -0.0017270, 0.0002047, -0.0017005, 0.0015857
2: 0.0036399, 0.0045504, 0.0036482, 0.0045111, -0.0007015, 0.0007420
3: 0.0013580, 0.0027031, 0.0013488, 0.0026240, -0.0009382, 0.0010028
4: -0.0045170, -0.0026057, -0.0044353, -0.0026347, -0.0013448, 0.0013135
5: -0.0002731, 0.0009492, -0.0002787, 0.0008921, -0.0009631, 0.0010359
6: -0.0050017, -0.0018042, -0.0050098, -0.0020174, -0.0020256, 0.0022179
7: -0.0221476, -0.0111046, -0.0216981, -0.0112933, -0.0077980, 0.0076561
8: 0.9751647, 0.9854926, 0.9754941, 0.9852356, -0.0074867, 0.0075235
9: -0.0004736, 0.0068189, -0.0003342, 0.0065373, -0.0051068, 0.0051723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038167, upper bound: 0.0039511
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0153362, 0.0180184, -0.0022543, 0.0023218
1: -0.0017739, 0.0002930, -0.0017270, 0.0002047, -0.0016534, 0.0016852
2: 0.0035992, 0.0045309, 0.0036482, 0.0045111, -0.0007448, 0.0007209
3: 0.0013382, 0.0027067, 0.0013488, 0.0026240, -0.0009633, 0.0010120
4: -0.0045657, -0.0026282, -0.0044353, -0.0026347, -0.0013905, 0.0012837
5: -0.0003274, 0.0009201, -0.0002787, 0.0008921, -0.0010223, 0.0010073
6: -0.0050083, -0.0017481, -0.0050098, -0.0020174, -0.0020325, 0.0022725
7: -0.0224520, -0.0112394, -0.0216981, -0.0112933, -0.0080809, 0.0074782
8: 0.9748068, 0.9853299, 0.9754941, 0.9852356, -0.0078222, 0.0073301
9: -0.0003790, 0.0070340, -0.0003342, 0.0065373, -0.0049849, 0.0053714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038618, upper bound: 0.0040013
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0154290, 0.0180274, -0.0023105, 0.0020901
1: -0.0018193, 0.0002015, -0.0016738, 0.0001979, -0.0016884, 0.0015291
2: 0.0036399, 0.0045504, 0.0036435, 0.0044800, -0.0006691, 0.0007397
3: 0.0013580, 0.0027031, 0.0013829, 0.0026800, -0.0009466, 0.0009318
4: -0.0045170, -0.0026057, -0.0045040, -0.0027187, -0.0012242, 0.0013388
5: -0.0002731, 0.0009492, -0.0002713, 0.0008631, -0.0009324, 0.0010270
6: -0.0050017, -0.0018042, -0.0049056, -0.0018310, -0.0021015, 0.0020188
7: -0.0221476, -0.0111046, -0.0220766, -0.0117779, -0.0071109, 0.0077905
8: 0.9751647, 0.9854926, 0.9752277, 0.9847767, -0.0068936, 0.0076046
9: -0.0004736, 0.0068189, -0.0000130, 0.0067747, -0.0051881, 0.0047261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045903, upper bound: 0.0042056
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0154943, 0.0181521, -0.0024509, 0.0020241
1: -0.0018193, 0.0002015, -0.0016245, 0.0002897, -0.0017877, 0.0014819
2: 0.0036399, 0.0045504, 0.0036032, 0.0044591, -0.0006480, 0.0007846
3: 0.0013580, 0.0027031, 0.0013621, 0.0026841, -0.0009506, 0.0009584
4: -0.0045170, -0.0026057, -0.0045507, -0.0027440, -0.0011971, 0.0013829
5: -0.0002731, 0.0009492, -0.0003257, 0.0008341, -0.0009040, 0.0010867
6: -0.0050017, -0.0018042, -0.0049122, -0.0017752, -0.0021601, 0.0020248
7: -0.0221476, -0.0111046, -0.0223662, -0.0119301, -0.0069462, 0.0080641
8: 0.9751647, 0.9854926, 0.9748762, 0.9846075, -0.0067012, 0.0079457
9: -0.0004736, 0.0068189, 0.0000921, 0.0069784, -0.0053815, 0.0046103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045903, upper bound: 0.0042056
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0154290, 0.0180274, -0.0022463, 0.0022293
1: -0.0017739, 0.0002930, -0.0016738, 0.0001979, -0.0016415, 0.0016292
2: 0.0035992, 0.0045309, 0.0036435, 0.0044800, -0.0007132, 0.0007192
3: 0.0013382, 0.0027067, 0.0013829, 0.0026800, -0.0009692, 0.0009362
4: -0.0045657, -0.0026282, -0.0045040, -0.0027187, -0.0012706, 0.0013122
5: -0.0003274, 0.0009201, -0.0002713, 0.0008631, -0.0009921, 0.0009985
6: -0.0050083, -0.0017481, -0.0049056, -0.0018310, -0.0021072, 0.0020766
7: -0.0224520, -0.0112394, -0.0220766, -0.0117779, -0.0073974, 0.0076294
8: 0.9748068, 0.9853299, 0.9752277, 0.9847767, -0.0072463, 0.0074177
9: -0.0003790, 0.0070340, -0.0000130, 0.0067747, -0.0050756, 0.0049284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043068, upper bound: 0.0041549
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
time: 0.71 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0154943, 0.0181521, -0.0022883, 0.0020667
1: -0.0017739, 0.0002930, -0.0016245, 0.0002897, -0.0016706, 0.0015130
2: 0.0035992, 0.0045309, 0.0036032, 0.0044591, -0.0006620, 0.0007330
3: 0.0013382, 0.0027067, 0.0013621, 0.0026841, -0.0009836, 0.0009673
4: -0.0045657, -0.0026282, -0.0045507, -0.0027440, -0.0012210, 0.0013347
5: -0.0003274, 0.0009201, -0.0003257, 0.0008341, -0.0009232, 0.0010162
6: -0.0050083, -0.0017481, -0.0049122, -0.0017752, -0.0021872, 0.0021038
7: -0.0224520, -0.0112394, -0.0223662, -0.0119301, -0.0070930, 0.0077673
8: 0.9748068, 0.9853299, 0.9748762, 0.9846075, -0.0068646, 0.0075732
9: -0.0003790, 0.0070340, 0.0000921, 0.0069784, -0.0051710, 0.0047121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042915, upper bound: 0.0041899
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0152116, 0.0180352, -0.0022648, 0.0022648
1: -0.0018193, 0.0002015, -0.0018193, 0.0002015, -0.0016626, 0.0016626
2: 0.0036399, 0.0045504, 0.0036399, 0.0045504, -0.0007244, 0.0007244
3: 0.0013580, 0.0027031, 0.0013580, 0.0027031, -0.0009615, 0.0009615
4: -0.0045170, -0.0026057, -0.0045170, -0.0026057, -0.0013061, 0.0013061
5: -0.0002731, 0.0009492, -0.0002731, 0.0009492, -0.0010134, 0.0010134
6: -0.0050017, -0.0018042, -0.0050017, -0.0018042, -0.0020698, 0.0020698
7: -0.0221476, -0.0111046, -0.0221476, -0.0111046, -0.0076019, 0.0076019
8: 0.9751647, 0.9854926, 0.9751647, 0.9854926, -0.0074314, 0.0074314
9: -0.0004736, 0.0068189, -0.0004736, 0.0068189, -0.0050635, 0.0050635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045357, upper bound: 0.0041608
time: 0.92 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0152116, 0.0180352, 0.0152727, 0.0181633, -0.0024049, 0.0022007
1: -0.0018193, 0.0002015, -0.0017739, 0.0002930, -0.0017614, 0.0016155
2: 0.0036399, 0.0045504, 0.0035992, 0.0045309, -0.0007039, 0.0007694
3: 0.0013580, 0.0027031, 0.0013382, 0.0027067, -0.0009660, 0.0009846
4: -0.0045170, -0.0026057, -0.0045657, -0.0026282, -0.0012794, 0.0013511
5: -0.0002731, 0.0009492, -0.0003274, 0.0009201, -0.0009846, 0.0010727
6: -0.0050017, -0.0018042, -0.0050083, -0.0017481, -0.0021286, 0.0020759
7: -0.0221476, -0.0111046, -0.0224520, -0.0112394, -0.0074400, 0.0078812
8: 0.9751647, 0.9854926, 0.9748068, 0.9853299, -0.0072467, 0.0077742
9: -0.0004736, 0.0068189, -0.0003790, 0.0070340, -0.0052602, 0.0049512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045710, upper bound: 0.0042056
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0152116, 0.0180352, -0.0022007, 0.0024049
1: -0.0017739, 0.0002930, -0.0018193, 0.0002015, -0.0016155, 0.0017614
2: 0.0035992, 0.0045309, 0.0036399, 0.0045504, -0.0007694, 0.0007039
3: 0.0013382, 0.0027067, 0.0013580, 0.0027031, -0.0009846, 0.0009660
4: -0.0045657, -0.0026282, -0.0045170, -0.0026057, -0.0013511, 0.0012794
5: -0.0003274, 0.0009201, -0.0002731, 0.0009492, -0.0010727, 0.0009846
6: -0.0050083, -0.0017481, -0.0050017, -0.0018042, -0.0020759, 0.0021286
7: -0.0224520, -0.0112394, -0.0221476, -0.0111046, -0.0078812, 0.0074400
8: 0.9748068, 0.9853299, 0.9751647, 0.9854926, -0.0077742, 0.0072467
9: -0.0003790, 0.0070340, -0.0004736, 0.0068189, -0.0049512, 0.0052602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042348, upper bound: 0.0041549
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
time: 0.71 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152727, 0.0181633, 0.0152727, 0.0181633, -0.0022462, 0.0022462
1: -0.0017739, 0.0002930, -0.0017739, 0.0002930, -0.0016471, 0.0016471
2: 0.0035992, 0.0045309, 0.0035992, 0.0045309, -0.0007186, 0.0007186
3: 0.0013382, 0.0027067, 0.0013382, 0.0027067, -0.0009970, 0.0009970
4: -0.0045657, -0.0026282, -0.0045657, -0.0026282, -0.0013023, 0.0013023
5: -0.0003274, 0.0009201, -0.0003274, 0.0009201, -0.0010040, 0.0010040
6: -0.0050083, -0.0017481, -0.0050083, -0.0017481, -0.0021557, 0.0021557
7: -0.0224520, -0.0112394, -0.0224520, -0.0112394, -0.0075788, 0.0075788
8: 0.9748068, 0.9853299, 0.9748068, 0.9853299, -0.0074014, 0.0074014
9: -0.0003790, 0.0070340, -0.0003790, 0.0070340, -0.0050459, 0.0050459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042348, upper bound: 0.0041549
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.74 seconds
IS_A1_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042448, upper bound: 0.0039919
IS_A1_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
IS_A1_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042448, upper bound: 0.0039919
IS_A1_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
IS_A1_A1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0034705, upper bound: 0.0037869
IS_A1_A1_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0029881, upper bound: 0.0034206
IS_A1_A1_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0031090, upper bound: 0.0035234
IS_A1_A1_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0029881, upper bound: 0.0034206
IS_A1_A1_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039511, upper bound: 0.0038167
IS_A1_A1_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
IS_A1_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0040013, upper bound: 0.0038618
IS_A1_A1_B2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
IS_A1_A1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0029196, upper bound: 0.0023957
IS_A1_A1_B2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0029193, upper bound: 0.0033405
IS_A1_A1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0029196, upper bound: 0.0023957
IS_A1_A1_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0029193, upper bound: 0.0033405
IS_A1_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039919, upper bound: 0.0042448
IS_A1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
IS_A1_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039919, upper bound: 0.0042448
IS_A1_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
IS_A1_A2_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0024455, upper bound: 0.0024499
IS_A1_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0047325, upper bound: 0.0045029
IS_A1_A2_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0024455, upper bound: 0.0024499
IS_A1_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0047325, upper bound: 0.0045029
IS_A1_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0044216, upper bound: 0.0043719
IS_A1_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
IS_A1_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0044216, upper bound: 0.0043719
IS_A1_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
IS_A1_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042437, upper bound: 0.0045765
IS_A1_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
IS_A1_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042437, upper bound: 0.0045765
IS_A1_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
IS_A1_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042042, upper bound: 0.0042601
IS_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042132
IS_A1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042185, upper bound: 0.0042500
IS_A1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
IS_A2_A2_B1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0038167, upper bound: 0.0039511
IS_A2_A2_B1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
IS_A2_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0038618, upper bound: 0.0040013
IS_A2_A2_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
IS_A2_A2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045903, upper bound: 0.0042056
IS_A2_A2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
IS_A2_A2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045903, upper bound: 0.0042056
IS_A2_A2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
IS_A2_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0043068, upper bound: 0.0041549
IS_A2_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
IS_A2_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042915, upper bound: 0.0041899
IS_A2_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
IS_A2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045357, upper bound: 0.0041608
IS_A2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
IS_A2_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045710, upper bound: 0.0042056
IS_A2_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
IS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042348, upper bound: 0.0041549
IS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
IS_A2_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0042348, upper bound: 0.0041549
IS_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.74
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0153398, 0.0180183, 0.0154573, 0.0180264, -0.0021196, 0.0020043
1: -0.0017241, 0.0002045, -0.0016508, 0.0001968, -0.0015362, 0.0014723
2: 0.0036482, 0.0045100, 0.0036437, 0.0044713, -0.0006407, 0.0006803
3: 0.0013492, 0.0026221, 0.0013861, 0.0026643, -0.0009339, 0.0008682
4: -0.0044353, -0.0026358, -0.0045036, -0.0027270, -0.0011589, 0.0013164
5: -0.0002786, 0.0008903, -0.0002704, 0.0008484, -0.0008983, 0.0009325
6: -0.0050074, -0.0020191, -0.0048878, -0.0018447, -0.0021177, 0.0018408
7: -0.0216980, -0.0112996, -0.0220752, -0.0118275, -0.0067393, 0.0076355
8: 0.9754941, 0.9852275, 0.9752277, 0.9847166, -0.0065537, 0.0073002
9: -0.0003298, 0.0065373, 0.0000220, 0.0067741, -0.0050653, 0.0044825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031649, upper bound: 0.0024678
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046688, upper bound: 0.0047116
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0153512, 0.0180179, 0.0153737, 0.0180570, -0.0021082, 0.0022550
1: -0.0017151, 0.0002041, -0.0017137, 0.0002190, -0.0015268, 0.0016725
2: 0.0036483, 0.0045065, 0.0036338, 0.0044976, -0.0007185, 0.0006767
3: 0.0013508, 0.0026171, 0.0013861, 0.0026919, -0.0010448, 0.0008563
4: -0.0044351, -0.0026389, -0.0045196, -0.0026983, -0.0012100, 0.0013250
5: -0.0002782, 0.0008846, -0.0002835, 0.0008872, -0.0010245, 0.0009265
6: -0.0049962, -0.0020237, -0.0048726, -0.0018119, -0.0021896, 0.0018152
7: -0.0216975, -0.0113173, -0.0221706, -0.0116502, -0.0070727, 0.0076778
8: 0.9754941, 0.9852049, 0.9751150, 0.9849264, -0.0070131, 0.0073021
9: -0.0003171, 0.0065371, -0.0001011, 0.0068398, -0.0050861, 0.0047263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046956, upper bound: 0.0047410
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046956, upper bound: 0.0047410
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0153398, 0.0180183, 0.0155208, 0.0181512, -0.0022560, 0.0019399
1: -0.0017241, 0.0002045, -0.0016031, 0.0002886, -0.0016351, 0.0014268
2: 0.0036482, 0.0045100, 0.0036035, 0.0044509, -0.0006202, 0.0007240
3: 0.0013492, 0.0026221, 0.0013652, 0.0026698, -0.0009430, 0.0008944
4: -0.0044353, -0.0026358, -0.0045503, -0.0027532, -0.0011296, 0.0013588
5: -0.0002786, 0.0008903, -0.0003249, 0.0008205, -0.0008716, 0.0009918
6: -0.0050074, -0.0020191, -0.0048946, -0.0017878, -0.0021717, 0.0018477
7: -0.0216980, -0.0112996, -0.0223648, -0.0119837, -0.0065626, 0.0079005
8: 0.9754941, 0.9852275, 0.9748762, 0.9845445, -0.0063554, 0.0076352
9: -0.0003298, 0.0065373, 0.0001294, 0.0069778, -0.0052539, 0.0043602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0153512, 0.0180179, 0.0154402, 0.0181811, -0.0022392, 0.0021861
1: -0.0017151, 0.0002041, -0.0016637, 0.0003072, -0.0016219, 0.0016242
2: 0.0036483, 0.0045065, 0.0035939, 0.0044765, -0.0006960, 0.0007187
3: 0.0013508, 0.0026171, 0.0013654, 0.0026933, -0.0010480, 0.0008824
4: -0.0044351, -0.0026389, -0.0045662, -0.0027235, -0.0011799, 0.0013667
5: -0.0002782, 0.0008846, -0.0003355, 0.0008569, -0.0009953, 0.0009832
6: -0.0049962, -0.0020237, -0.0048790, -0.0017593, -0.0022372, 0.0018220
7: -0.0216975, -0.0113173, -0.0224586, -0.0118008, -0.0068890, 0.0079363
8: 0.9754941, 0.9852049, 0.9747749, 0.9847559, -0.0068015, 0.0076188
9: -0.0003171, 0.0065371, 0.0000027, 0.0070414, -0.0052680, 0.0045979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0153398, 0.0180183, 0.0152997, 0.0181626, -0.0022680, 0.0021749
1: -0.0017241, 0.0002045, -0.0017521, 0.0002922, -0.0016398, 0.0015900
2: 0.0036482, 0.0045100, 0.0035994, 0.0045226, -0.0006961, 0.0007285
3: 0.0013492, 0.0026221, 0.0013414, 0.0026917, -0.0009661, 0.0009268
4: -0.0044353, -0.0026358, -0.0045653, -0.0026360, -0.0012695, 0.0013884
5: -0.0002786, 0.0008903, -0.0003268, 0.0009063, -0.0009673, 0.0009935
6: -0.0050074, -0.0020191, -0.0049893, -0.0017603, -0.0022240, 0.0019767
7: -0.0216980, -0.0112996, -0.0224507, -0.0112852, -0.0073858, 0.0080681
8: 0.9754941, 0.9852275, 0.9748068, 0.9852751, -0.0071879, 0.0077678
9: -0.0003298, 0.0065373, -0.0003469, 0.0070335, -0.0053613, 0.0049152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0154573, 0.0180264, 0.0153398, 0.0180183, -0.0020043, 0.0021196
1: -0.0016508, 0.0001968, -0.0017241, 0.0002045, -0.0014723, 0.0015362
2: 0.0036437, 0.0044713, 0.0036482, 0.0045100, -0.0006803, 0.0006407
3: 0.0013861, 0.0026643, 0.0013492, 0.0026221, -0.0008682, 0.0009339
4: -0.0045036, -0.0027270, -0.0044353, -0.0026358, -0.0013164, 0.0011589
5: -0.0002704, 0.0008484, -0.0002786, 0.0008903, -0.0009325, 0.0008983
6: -0.0048878, -0.0018447, -0.0050074, -0.0020191, -0.0018408, 0.0021177
7: -0.0220752, -0.0118275, -0.0216980, -0.0112996, -0.0076355, 0.0067393
8: 0.9752277, 0.9847166, 0.9754941, 0.9852275, -0.0073002, 0.0065537
9: 0.0000220, 0.0067741, -0.0003298, 0.0065373, -0.0044825, 0.0050653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024678, upper bound: 0.0031649
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047116, upper bound: 0.0046688
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0153737, 0.0180570, 0.0153512, 0.0180179, -0.0022550, 0.0021082
1: -0.0017137, 0.0002190, -0.0017151, 0.0002041, -0.0016725, 0.0015268
2: 0.0036338, 0.0044976, 0.0036483, 0.0045065, -0.0006767, 0.0007185
3: 0.0013861, 0.0026919, 0.0013508, 0.0026171, -0.0008563, 0.0010448
4: -0.0045196, -0.0026983, -0.0044351, -0.0026389, -0.0013250, 0.0012100
5: -0.0002835, 0.0008872, -0.0002782, 0.0008846, -0.0009265, 0.0010245
6: -0.0048726, -0.0018119, -0.0049962, -0.0020237, -0.0018152, 0.0021896
7: -0.0221706, -0.0116502, -0.0216975, -0.0113173, -0.0076778, 0.0070727
8: 0.9751150, 0.9849264, 0.9754941, 0.9852049, -0.0073021, 0.0070131
9: -0.0001011, 0.0068398, -0.0003171, 0.0065371, -0.0047263, 0.0050861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047410, upper bound: 0.0046956
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047410, upper bound: 0.0046956
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0155208, 0.0181512, 0.0153398, 0.0180183, -0.0019399, 0.0022560
1: -0.0016031, 0.0002886, -0.0017241, 0.0002045, -0.0014268, 0.0016351
2: 0.0036035, 0.0044509, 0.0036482, 0.0045100, -0.0007240, 0.0006202
3: 0.0013652, 0.0026698, 0.0013492, 0.0026221, -0.0008944, 0.0009430
4: -0.0045503, -0.0027532, -0.0044353, -0.0026358, -0.0013588, 0.0011296
5: -0.0003249, 0.0008205, -0.0002786, 0.0008903, -0.0009918, 0.0008716
6: -0.0048946, -0.0017878, -0.0050074, -0.0020191, -0.0018477, 0.0021717
7: -0.0223648, -0.0119837, -0.0216980, -0.0112996, -0.0079005, 0.0065626
8: 0.9748762, 0.9845445, 0.9754941, 0.9852275, -0.0076352, 0.0063554
9: 0.0001294, 0.0069778, -0.0003298, 0.0065373, -0.0043602, 0.0052539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0154402, 0.0181811, 0.0153512, 0.0180179, -0.0021861, 0.0022392
1: -0.0016637, 0.0003072, -0.0017151, 0.0002041, -0.0016242, 0.0016219
2: 0.0035939, 0.0044765, 0.0036483, 0.0045065, -0.0007187, 0.0006960
3: 0.0013654, 0.0026933, 0.0013508, 0.0026171, -0.0008824, 0.0010480
4: -0.0045662, -0.0027235, -0.0044351, -0.0026389, -0.0013667, 0.0011799
5: -0.0003355, 0.0008569, -0.0002782, 0.0008846, -0.0009832, 0.0009953
6: -0.0048790, -0.0017593, -0.0049962, -0.0020237, -0.0018220, 0.0022372
7: -0.0224586, -0.0118008, -0.0216975, -0.0113173, -0.0079363, 0.0068890
8: 0.9747749, 0.9847559, 0.9754941, 0.9852049, -0.0076188, 0.0068015
9: 0.0000027, 0.0070414, -0.0003171, 0.0065371, -0.0045979, 0.0052680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0154298, 0.0180224, 0.0154290, 0.0180274, -0.0020758, 0.0019683
1: -0.0016731, 0.0001942, -0.0016738, 0.0001979, -0.0015253, 0.0014674
2: 0.0036450, 0.0044798, 0.0036435, 0.0044800, -0.0006266, 0.0006636
3: 0.0013830, 0.0026789, 0.0013829, 0.0026800, -0.0009078, 0.0008954
4: -0.0045008, -0.0027189, -0.0045040, -0.0027187, -0.0010541, 0.0011924
5: -0.0002689, 0.0008626, -0.0002713, 0.0008631, -0.0009007, 0.0009305
6: -0.0049054, -0.0018363, -0.0049056, -0.0018310, -0.0019621, 0.0017763
7: -0.0220579, -0.0117795, -0.0220766, -0.0117779, -0.0061442, 0.0069306
8: 0.9752457, 0.9847751, 0.9752277, 0.9847767, -0.0060732, 0.0067475
9: -0.0000119, 0.0067623, -0.0000130, 0.0067747, -0.0046099, 0.0040991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042784, upper bound: 0.0036010
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042784, upper bound: 0.0047864
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0154298, 0.0180224, 0.0154943, 0.0181521, -0.0022163, 0.0019072
1: -0.0016731, 0.0001942, -0.0016245, 0.0002897, -0.0016246, 0.0014211
2: 0.0036450, 0.0044798, 0.0036032, 0.0044591, -0.0006072, 0.0007085
3: 0.0013830, 0.0026789, 0.0013621, 0.0026841, -0.0009118, 0.0009232
4: -0.0045008, -0.0027189, -0.0045507, -0.0027440, -0.0010275, 0.0012365
5: -0.0002689, 0.0008626, -0.0003257, 0.0008341, -0.0008726, 0.0009901
6: -0.0049054, -0.0018363, -0.0049122, -0.0017752, -0.0020208, 0.0017838
7: -0.0220579, -0.0117795, -0.0223662, -0.0119301, -0.0059854, 0.0072042
8: 0.9752457, 0.9847751, 0.9748762, 0.9846075, -0.0058961, 0.0070886
9: -0.0000119, 0.0067623, 0.0000921, 0.0069784, -0.0048033, 0.0039900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046340, upper bound: 0.0043386
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046123, upper bound: 0.0042946
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0155208, 0.0181512, 0.0154326, 0.0180273, -0.0019319, 0.0021624
1: -0.0016031, 0.0002886, -0.0016709, 0.0001978, -0.0014163, 0.0015798
2: 0.0036035, 0.0044509, 0.0036435, 0.0044789, -0.0006921, 0.0006183
3: 0.0013652, 0.0026698, 0.0013833, 0.0026781, -0.0008965, 0.0008676
4: -0.0045503, -0.0027532, -0.0045040, -0.0027198, -0.0012346, 0.0011522
5: -0.0003249, 0.0008205, -0.0002712, 0.0008612, -0.0009615, 0.0008635
6: -0.0048946, -0.0017878, -0.0049035, -0.0018327, -0.0019181, 0.0019687
7: -0.0223648, -0.0119837, -0.0220764, -0.0117844, -0.0071927, 0.0066814
8: 0.9748762, 0.9845445, 0.9752277, 0.9847686, -0.0070361, 0.0064211
9: 0.0001294, 0.0069778, -0.0000083, 0.0067746, -0.0044306, 0.0047941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0154402, 0.0181811, 0.0154451, 0.0180268, -0.0021785, 0.0021438
1: -0.0016637, 0.0003072, -0.0016611, 0.0001973, -0.0016130, 0.0015662
2: 0.0035939, 0.0044765, 0.0036436, 0.0044751, -0.0006863, 0.0006948
3: 0.0013654, 0.0026933, 0.0013848, 0.0026731, -0.0008850, 0.0009777
4: -0.0045662, -0.0027235, -0.0045038, -0.0027235, -0.0012403, 0.0012024
5: -0.0003355, 0.0008569, -0.0002708, 0.0008550, -0.0009528, 0.0009867
6: -0.0048790, -0.0017593, -0.0048913, -0.0018377, -0.0018903, 0.0020298
7: -0.0224586, -0.0118008, -0.0220760, -0.0118060, -0.0072190, 0.0070117
8: 0.9747749, 0.9847559, 0.9752277, 0.9847420, -0.0070227, 0.0068794
9: 0.0000027, 0.0070414, 0.0000071, 0.0067744, -0.0046729, 0.0048050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0155208, 0.0181512, 0.0154976, 0.0181520, -0.0019702, 0.0019998
1: -0.0016031, 0.0002886, -0.0016218, 0.0002896, -0.0014428, 0.0014642
2: 0.0036035, 0.0044509, 0.0036033, 0.0044581, -0.0006405, 0.0006312
3: 0.0013652, 0.0026698, 0.0013625, 0.0026823, -0.0009076, 0.0008962
4: -0.0045503, -0.0027532, -0.0045507, -0.0027452, -0.0011869, 0.0011757
5: -0.0003249, 0.0008205, -0.0003256, 0.0008324, -0.0008927, 0.0008801
6: -0.0048946, -0.0017878, -0.0049100, -0.0017767, -0.0019976, 0.0019941
7: -0.0223648, -0.0119837, -0.0223660, -0.0119371, -0.0068993, 0.0068236
8: 0.9748762, 0.9845445, 0.9748762, 0.9845989, -0.0066638, 0.0065777
9: 0.0001294, 0.0069778, 0.0000970, 0.0069783, -0.0045279, 0.0045856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0154402, 0.0181811, 0.0155130, 0.0181516, -0.0022144, 0.0019879
1: -0.0016637, 0.0003072, -0.0016099, 0.0002891, -0.0016418, 0.0014548
2: 0.0035939, 0.0044765, 0.0036034, 0.0044533, -0.0006370, 0.0007070
3: 0.0013654, 0.0026933, 0.0013641, 0.0026769, -0.0008980, 0.0010014
4: -0.0045662, -0.0027235, -0.0045505, -0.0027499, -0.0011939, 0.0012257
5: -0.0003355, 0.0008569, -0.0003252, 0.0008250, -0.0008869, 0.0010053
6: -0.0048790, -0.0017593, -0.0048981, -0.0017820, -0.0019692, 0.0020542
7: -0.0224586, -0.0118008, -0.0223656, -0.0119647, -0.0069329, 0.0071519
8: 0.9747749, 0.9847559, 0.9748762, 0.9845641, -0.0066592, 0.0070361
9: 0.0000027, 0.0070414, 0.0001170, 0.0069782, -0.0047697, 0.0046013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0154573, 0.0180264, 0.0152152, 0.0180351, -0.0020078, 0.0022575
1: -0.0016508, 0.0001968, -0.0018164, 0.0002014, -0.0014632, 0.0016428
2: 0.0036437, 0.0044713, 0.0036400, 0.0045493, -0.0007235, 0.0006435
3: 0.0013861, 0.0026643, 0.0013583, 0.0027011, -0.0008944, 0.0009009
4: -0.0045036, -0.0027270, -0.0045170, -0.0026067, -0.0013367, 0.0012101
5: -0.0002704, 0.0008484, -0.0002730, 0.0009474, -0.0009980, 0.0008910
6: -0.0048878, -0.0018447, -0.0049994, -0.0018058, -0.0019690, 0.0020486
7: -0.0220752, -0.0118275, -0.0221474, -0.0111100, -0.0077779, 0.0070186
8: 0.9752277, 0.9847166, 0.9751647, 0.9854856, -0.0075487, 0.0067496
9: 0.0000220, 0.0067741, -0.0004696, 0.0068188, -0.0046560, 0.0051781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024204, upper bound: 0.0035290
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046775, upper bound: 0.0047517
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0153737, 0.0180570, 0.0152281, 0.0180347, -0.0022579, 0.0022435
1: -0.0017137, 0.0002190, -0.0018061, 0.0002009, -0.0016635, 0.0016316
2: 0.0036338, 0.0044976, 0.0036401, 0.0045454, -0.0007192, 0.0007213
3: 0.0013861, 0.0026919, 0.0013602, 0.0026958, -0.0008837, 0.0010156
4: -0.0045196, -0.0026983, -0.0045168, -0.0026102, -0.0013441, 0.0012616
5: -0.0002835, 0.0008872, -0.0002726, 0.0009409, -0.0009906, 0.0010170
6: -0.0048726, -0.0018119, -0.0049868, -0.0018104, -0.0019417, 0.0021153
7: -0.0221706, -0.0116502, -0.0221470, -0.0111298, -0.0078126, 0.0073571
8: 0.9751150, 0.9849264, 0.9751647, 0.9854617, -0.0075437, 0.0072203
9: -0.0001011, 0.0068398, -0.0004556, 0.0068187, -0.0049049, 0.0051942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047026, upper bound: 0.0047754
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047026, upper bound: 0.0047754
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0155208, 0.0181512, 0.0152152, 0.0180351, -0.0019458, 0.0023969
1: -0.0016031, 0.0002886, -0.0018164, 0.0002014, -0.0014198, 0.0017427
2: 0.0036035, 0.0044509, 0.0036400, 0.0045493, -0.0007681, 0.0006237
3: 0.0013652, 0.0026698, 0.0013583, 0.0027011, -0.0009198, 0.0009061
4: -0.0045503, -0.0027532, -0.0045170, -0.0026067, -0.0013808, 0.0011839
5: -0.0003249, 0.0008205, -0.0002730, 0.0009474, -0.0010579, 0.0008651
6: -0.0048946, -0.0017878, -0.0049994, -0.0018058, -0.0019747, 0.0021077
7: -0.0223648, -0.0119837, -0.0221474, -0.0111100, -0.0080514, 0.0068605
8: 0.9748762, 0.9845445, 0.9751647, 0.9854856, -0.0078917, 0.0065660
9: 0.0001294, 0.0069778, -0.0004696, 0.0068188, -0.0045459, 0.0053715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0154402, 0.0181811, 0.0152281, 0.0180347, -0.0021919, 0.0023755
1: -0.0016637, 0.0003072, -0.0018061, 0.0002009, -0.0016167, 0.0017267
2: 0.0035939, 0.0044765, 0.0036401, 0.0045454, -0.0007614, 0.0006999
3: 0.0013654, 0.0026933, 0.0013602, 0.0026958, -0.0009091, 0.0010151
4: -0.0045662, -0.0027235, -0.0045168, -0.0026102, -0.0013866, 0.0012341
5: -0.0003355, 0.0008569, -0.0002726, 0.0009409, -0.0010475, 0.0009882
6: -0.0048790, -0.0017593, -0.0049868, -0.0018104, -0.0019474, 0.0021679
7: -0.0224586, -0.0118008, -0.0221470, -0.0111298, -0.0080770, 0.0071909
8: 0.9747749, 0.9847559, 0.9751647, 0.9854617, -0.0078747, 0.0070233
9: 0.0000027, 0.0070414, -0.0004556, 0.0068187, -0.0047882, 0.0053815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0154326, 0.0180273, 0.0152997, 0.0181626, -0.0021750, 0.0021673
1: -0.0016709, 0.0001978, -0.0017521, 0.0002922, -0.0015844, 0.0015782
2: 0.0036435, 0.0044789, 0.0035994, 0.0045226, -0.0006946, 0.0006967
3: 0.0013833, 0.0026781, 0.0013414, 0.0026917, -0.0008899, 0.0009315
4: -0.0045040, -0.0027198, -0.0045653, -0.0026360, -0.0012985, 0.0012685
5: -0.0002712, 0.0008612, -0.0003268, 0.0009063, -0.0009587, 0.0009631
6: -0.0049035, -0.0018327, -0.0049893, -0.0017603, -0.0020256, 0.0020556
7: -0.0220764, -0.0117844, -0.0224507, -0.0112852, -0.0075400, 0.0073849
8: 0.9752277, 0.9847686, 0.9748068, 0.9852751, -0.0072753, 0.0071921
9: -0.0000083, 0.0067746, -0.0003469, 0.0070335, -0.0049183, 0.0050081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0154451, 0.0180268, 0.0152255, 0.0181923, -0.0021563, 0.0023993
1: -0.0016611, 0.0001973, -0.0018085, 0.0003104, -0.0015700, 0.0017639
2: 0.0036436, 0.0044751, 0.0035896, 0.0045462, -0.0007667, 0.0006909
3: 0.0013848, 0.0026731, 0.0013428, 0.0027151, -0.0009974, 0.0009186
4: -0.0045038, -0.0027235, -0.0045809, -0.0026088, -0.0013452, 0.0012752
5: -0.0002708, 0.0008550, -0.0003371, 0.0009412, -0.0010754, 0.0009542
6: -0.0048913, -0.0018377, -0.0049720, -0.0017318, -0.0020879, 0.0020249
7: -0.0220760, -0.0118060, -0.0225420, -0.0111150, -0.0078498, 0.0074162
8: 0.9752277, 0.9847420, 0.9747007, 0.9854748, -0.0077136, 0.0071815
9: 0.0000071, 0.0067744, -0.0004654, 0.0070965, -0.0049316, 0.0052369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0155208, 0.0181512, 0.0152761, 0.0181633, -0.0019832, 0.0022345
1: -0.0016031, 0.0002886, -0.0017711, 0.0002929, -0.0014459, 0.0016251
2: 0.0036035, 0.0044509, 0.0035992, 0.0045299, -0.0007167, 0.0006363
3: 0.0013652, 0.0026698, 0.0013386, 0.0027049, -0.0009304, 0.0009354
4: -0.0045503, -0.0027532, -0.0045656, -0.0026292, -0.0013326, 0.0012077
5: -0.0003249, 0.0008205, -0.0003273, 0.0009184, -0.0009872, 0.0008814
6: -0.0048946, -0.0017878, -0.0050060, -0.0017495, -0.0020549, 0.0021347
7: -0.0223648, -0.0119837, -0.0224519, -0.0112453, -0.0077552, 0.0070047
8: 0.9748762, 0.9845445, 0.9748068, 0.9853226, -0.0075176, 0.0067233
9: 0.0001294, 0.0069778, -0.0003748, 0.0070339, -0.0046444, 0.0051614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0154402, 0.0181811, 0.0152909, 0.0181629, -0.0022267, 0.0022201
1: -0.0016637, 0.0003072, -0.0017593, 0.0002925, -0.0016452, 0.0016130
2: 0.0035939, 0.0044765, 0.0035993, 0.0045253, -0.0007123, 0.0007119
3: 0.0013654, 0.0026933, 0.0013406, 0.0026994, -0.0009206, 0.0010391
4: -0.0045662, -0.0027235, -0.0045655, -0.0026335, -0.0013394, 0.0012578
5: -0.0003355, 0.0008569, -0.0003270, 0.0009110, -0.0009793, 0.0010066
6: -0.0048790, -0.0017593, -0.0049935, -0.0017543, -0.0020274, 0.0021938
7: -0.0224586, -0.0118008, -0.0224515, -0.0112704, -0.0077870, 0.0073331
8: 0.9747749, 0.9847559, 0.9748068, 0.9852917, -0.0075091, 0.0071803
9: 0.0000027, 0.0070414, -0.0003569, 0.0070338, -0.0048861, 0.0051760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0152997, 0.0181626, 0.0153398, 0.0180183, -0.0021749, 0.0022680
1: -0.0017521, 0.0002922, -0.0017241, 0.0002045, -0.0015900, 0.0016398
2: 0.0035994, 0.0045226, 0.0036482, 0.0045100, -0.0007285, 0.0006961
3: 0.0013414, 0.0026917, 0.0013492, 0.0026221, -0.0009268, 0.0009661
4: -0.0045653, -0.0026360, -0.0044353, -0.0026358, -0.0013884, 0.0012695
5: -0.0003268, 0.0009063, -0.0002786, 0.0008903, -0.0009935, 0.0009673
6: -0.0049893, -0.0017603, -0.0050074, -0.0020191, -0.0019767, 0.0022240
7: -0.0224507, -0.0112852, -0.0216980, -0.0112996, -0.0080681, 0.0073858
8: 0.9748068, 0.9852751, 0.9754941, 0.9852275, -0.0077678, 0.0071879
9: -0.0003469, 0.0070335, -0.0003298, 0.0065373, -0.0049151, 0.0053613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
time: 0.80 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0152152, 0.0180351, 0.0154573, 0.0180264, -0.0022575, 0.0020078
1: -0.0018164, 0.0002014, -0.0016508, 0.0001968, -0.0016428, 0.0014632
2: 0.0036400, 0.0045493, 0.0036437, 0.0044713, -0.0006435, 0.0007235
3: 0.0013583, 0.0027011, 0.0013861, 0.0026643, -0.0009009, 0.0008944
4: -0.0045170, -0.0026067, -0.0045036, -0.0027270, -0.0012101, 0.0013367
5: -0.0002730, 0.0009474, -0.0002704, 0.0008484, -0.0008910, 0.0009980
6: -0.0049994, -0.0018058, -0.0048878, -0.0018447, -0.0020486, 0.0019690
7: -0.0221474, -0.0111100, -0.0220752, -0.0118275, -0.0070186, 0.0077779
8: 0.9751647, 0.9854856, 0.9752277, 0.9847166, -0.0067496, 0.0075487
9: -0.0004696, 0.0068188, 0.0000220, 0.0067741, -0.0051781, 0.0046560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036278, upper bound: 0.0024198
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047689, upper bound: 0.0046600
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0152281, 0.0180347, 0.0153737, 0.0180570, -0.0022435, 0.0022579
1: -0.0018061, 0.0002009, -0.0017137, 0.0002190, -0.0016316, 0.0016635
2: 0.0036401, 0.0045454, 0.0036338, 0.0044976, -0.0007213, 0.0007192
3: 0.0013602, 0.0026958, 0.0013861, 0.0026919, -0.0010156, 0.0008837
4: -0.0045168, -0.0026102, -0.0045196, -0.0026983, -0.0012616, 0.0013441
5: -0.0002726, 0.0009409, -0.0002835, 0.0008872, -0.0010170, 0.0009906
6: -0.0049868, -0.0018104, -0.0048726, -0.0018119, -0.0021153, 0.0019417
7: -0.0221470, -0.0111298, -0.0221706, -0.0116502, -0.0073571, 0.0078126
8: 0.9751647, 0.9854617, 0.9751150, 0.9849264, -0.0072203, 0.0075437
9: -0.0004556, 0.0068187, -0.0001011, 0.0068398, -0.0051942, 0.0049049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0152152, 0.0180351, 0.0155208, 0.0181512, -0.0023969, 0.0019458
1: -0.0018164, 0.0002014, -0.0016031, 0.0002886, -0.0017427, 0.0014198
2: 0.0036400, 0.0045493, 0.0036035, 0.0044509, -0.0006237, 0.0007681
3: 0.0013583, 0.0027011, 0.0013652, 0.0026698, -0.0009061, 0.0009198
4: -0.0045170, -0.0026067, -0.0045503, -0.0027532, -0.0011839, 0.0013808
5: -0.0002730, 0.0009474, -0.0003249, 0.0008205, -0.0008651, 0.0010579
6: -0.0049994, -0.0018058, -0.0048946, -0.0017878, -0.0021077, 0.0019747
7: -0.0221474, -0.0111100, -0.0223648, -0.0119837, -0.0068605, 0.0080514
8: 0.9751647, 0.9854856, 0.9748762, 0.9845445, -0.0065660, 0.0078917
9: -0.0004696, 0.0068188, 0.0001294, 0.0069778, -0.0053715, 0.0045459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0152281, 0.0180347, 0.0154402, 0.0181811, -0.0023755, 0.0021919
1: -0.0018061, 0.0002009, -0.0016637, 0.0003072, -0.0017267, 0.0016167
2: 0.0036401, 0.0045454, 0.0035939, 0.0044765, -0.0006999, 0.0007614
3: 0.0013602, 0.0026958, 0.0013654, 0.0026933, -0.0010151, 0.0009091
4: -0.0045168, -0.0026102, -0.0045662, -0.0027235, -0.0012341, 0.0013866
5: -0.0002726, 0.0009409, -0.0003355, 0.0008569, -0.0009882, 0.0010475
6: -0.0049868, -0.0018104, -0.0048790, -0.0017593, -0.0021679, 0.0019474
7: -0.0221470, -0.0111298, -0.0224586, -0.0118008, -0.0071909, 0.0080769
8: 0.9751647, 0.9854617, 0.9747749, 0.9847559, -0.0070233, 0.0078747
9: -0.0004556, 0.0068187, 0.0000027, 0.0070414, -0.0053815, 0.0047882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
time: 0.80 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0152997, 0.0181626, 0.0154326, 0.0180273, -0.0021673, 0.0021750
1: -0.0017521, 0.0002922, -0.0016709, 0.0001978, -0.0015782, 0.0015844
2: 0.0035994, 0.0045226, 0.0036435, 0.0044789, -0.0006967, 0.0006946
3: 0.0013414, 0.0026917, 0.0013833, 0.0026781, -0.0009315, 0.0008899
4: -0.0045653, -0.0026360, -0.0045040, -0.0027198, -0.0012685, 0.0012985
5: -0.0003268, 0.0009063, -0.0002712, 0.0008612, -0.0009631, 0.0009587
6: -0.0049893, -0.0017603, -0.0049035, -0.0018327, -0.0020556, 0.0020256
7: -0.0224507, -0.0112852, -0.0220764, -0.0117844, -0.0073849, 0.0075400
8: 0.9748068, 0.9852751, 0.9752277, 0.9847686, -0.0071921, 0.0072753
9: -0.0003469, 0.0070335, -0.0000083, 0.0067746, -0.0050081, 0.0049183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152255, 0.0181923, 0.0154451, 0.0180268, -0.0023993, 0.0021563
1: -0.0018085, 0.0003104, -0.0016611, 0.0001973, -0.0017639, 0.0015700
2: 0.0035896, 0.0045462, 0.0036436, 0.0044751, -0.0006909, 0.0007667
3: 0.0013428, 0.0027151, 0.0013848, 0.0026731, -0.0009186, 0.0009974
4: -0.0045809, -0.0026088, -0.0045038, -0.0027235, -0.0012752, 0.0013452
5: -0.0003371, 0.0009412, -0.0002708, 0.0008550, -0.0009542, 0.0010754
6: -0.0049720, -0.0017318, -0.0048913, -0.0018377, -0.0020249, 0.0020879
7: -0.0225420, -0.0111150, -0.0220760, -0.0118060, -0.0074162, 0.0078498
8: 0.9747007, 0.9854748, 0.9752277, 0.9847420, -0.0071815, 0.0077136
9: -0.0004654, 0.0070965, 0.0000071, 0.0067744, -0.0052369, 0.0049316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0152761, 0.0181633, 0.0155208, 0.0181512, -0.0022345, 0.0019832
1: -0.0017711, 0.0002929, -0.0016031, 0.0002886, -0.0016251, 0.0014459
2: 0.0035992, 0.0045299, 0.0036035, 0.0044509, -0.0006363, 0.0007167
3: 0.0013386, 0.0027049, 0.0013652, 0.0026698, -0.0009354, 0.0009304
4: -0.0045656, -0.0026292, -0.0045503, -0.0027532, -0.0012077, 0.0013326
5: -0.0003273, 0.0009184, -0.0003249, 0.0008205, -0.0008814, 0.0009872
6: -0.0050060, -0.0017495, -0.0048946, -0.0017878, -0.0021347, 0.0020549
7: -0.0224519, -0.0112453, -0.0223648, -0.0119837, -0.0070047, 0.0077552
8: 0.9748068, 0.9853226, 0.9748762, 0.9845445, -0.0067233, 0.0075176
9: -0.0003748, 0.0070339, 0.0001294, 0.0069778, -0.0051614, 0.0046444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
time: 0.95 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0152909, 0.0181629, 0.0154402, 0.0181811, -0.0022201, 0.0022267
1: -0.0017593, 0.0002925, -0.0016637, 0.0003072, -0.0016130, 0.0016452
2: 0.0035993, 0.0045253, 0.0035939, 0.0044765, -0.0007119, 0.0007123
3: 0.0013406, 0.0026994, 0.0013654, 0.0026933, -0.0010391, 0.0009206
4: -0.0045655, -0.0026335, -0.0045662, -0.0027235, -0.0012578, 0.0013394
5: -0.0003270, 0.0009110, -0.0003355, 0.0008569, -0.0010066, 0.0009793
6: -0.0049935, -0.0017543, -0.0048790, -0.0017593, -0.0021938, 0.0020274
7: -0.0224515, -0.0112704, -0.0224586, -0.0118008, -0.0073330, 0.0077870
8: 0.9748068, 0.9852917, 0.9747749, 0.9847559, -0.0071803, 0.0075091
9: -0.0003569, 0.0070338, 0.0000027, 0.0070414, -0.0051760, 0.0048861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0152400, 0.0180345, 0.0152152, 0.0180351, -0.0021826, 0.0022121
1: -0.0017962, 0.0002005, -0.0018164, 0.0002014, -0.0015961, 0.0016180
2: 0.0036401, 0.0045417, 0.0036400, 0.0045493, -0.0007084, 0.0006990
3: 0.0013611, 0.0026871, 0.0013583, 0.0027011, -0.0009242, 0.0009152
4: -0.0045166, -0.0026127, -0.0045170, -0.0026067, -0.0013041, 0.0012920
5: -0.0002724, 0.0009346, -0.0002730, 0.0009474, -0.0009853, 0.0009713
6: -0.0049826, -0.0018173, -0.0049994, -0.0018058, -0.0020185, 0.0020185
7: -0.0221463, -0.0111453, -0.0221474, -0.0111100, -0.0075896, 0.0075104
8: 0.9751647, 0.9854413, 0.9751647, 0.9854856, -0.0073757, 0.0072877
9: -0.0004444, 0.0068184, -0.0004696, 0.0068188, -0.0049944, 0.0050536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
time: 0.84 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151634, 0.0180647, 0.0152281, 0.0180347, -0.0024204, 0.0021994
1: -0.0018545, 0.0002224, -0.0018061, 0.0002009, -0.0017842, 0.0016056
2: 0.0036301, 0.0045660, 0.0036401, 0.0045454, -0.0007044, 0.0007728
3: 0.0013624, 0.0027137, 0.0013602, 0.0026958, -0.0009114, 0.0010282
4: -0.0045330, -0.0025865, -0.0045168, -0.0026102, -0.0013105, 0.0013399
5: -0.0002851, 0.0009708, -0.0002726, 0.0009409, -0.0009768, 0.0010895
6: -0.0049656, -0.0017845, -0.0049868, -0.0018104, -0.0019891, 0.0020857
7: -0.0222455, -0.0109817, -0.0221470, -0.0111298, -0.0076171, 0.0078255
8: 0.9750581, 0.9856368, 0.9751647, 0.9854617, -0.0073666, 0.0077255
9: -0.0005587, 0.0068881, -0.0004556, 0.0068187, -0.0052269, 0.0050642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0152152, 0.0180351, 0.0152997, 0.0181626, -0.0023512, 0.0021216
1: -0.0018164, 0.0002014, -0.0017521, 0.0002922, -0.0017156, 0.0015522
2: 0.0036400, 0.0045493, 0.0035994, 0.0045226, -0.0006794, 0.0007531
3: 0.0013583, 0.0027011, 0.0013414, 0.0026917, -0.0009202, 0.0009467
4: -0.0045170, -0.0026067, -0.0045653, -0.0026360, -0.0012657, 0.0013491
5: -0.0002730, 0.0009474, -0.0003268, 0.0009063, -0.0009448, 0.0010435
6: -0.0049994, -0.0018058, -0.0049893, -0.0017603, -0.0020775, 0.0020241
7: -0.0221474, -0.0111100, -0.0224507, -0.0112852, -0.0073499, 0.0078689
8: 0.9751647, 0.9854856, 0.9748068, 0.9852751, -0.0071031, 0.0077200
9: -0.0004696, 0.0068188, -0.0003469, 0.0070335, -0.0052503, 0.0048824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0152281, 0.0180347, 0.0152255, 0.0181923, -0.0023307, 0.0023560
1: -0.0018061, 0.0002009, -0.0018085, 0.0003104, -0.0016997, 0.0017393
2: 0.0036401, 0.0045454, 0.0035896, 0.0045462, -0.0007520, 0.0007464
3: 0.0013602, 0.0026958, 0.0013428, 0.0027151, -0.0010282, 0.0009341
4: -0.0045168, -0.0026102, -0.0045809, -0.0026088, -0.0013128, 0.0013540
5: -0.0002726, 0.0009409, -0.0003371, 0.0009412, -0.0010626, 0.0010329
6: -0.0049868, -0.0018104, -0.0049720, -0.0017318, -0.0021385, 0.0019950
7: -0.0221470, -0.0111298, -0.0225420, -0.0111150, -0.0076606, 0.0078888
8: 0.9751647, 0.9854617, 0.9747007, 0.9854748, -0.0075348, 0.0077025
9: -0.0004556, 0.0068187, -0.0004654, 0.0070965, -0.0052582, 0.0051117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
time: 0.94 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0152997, 0.0181626, 0.0152152, 0.0180351, -0.0021216, 0.0023512
1: -0.0017521, 0.0002922, -0.0018164, 0.0002014, -0.0015522, 0.0017156
2: 0.0035994, 0.0045226, 0.0036400, 0.0045493, -0.0007531, 0.0006794
3: 0.0013414, 0.0026917, 0.0013583, 0.0027011, -0.0009467, 0.0009202
4: -0.0045653, -0.0026360, -0.0045170, -0.0026067, -0.0013491, 0.0012657
5: -0.0003268, 0.0009063, -0.0002730, 0.0009474, -0.0010435, 0.0009448
6: -0.0049893, -0.0017603, -0.0049994, -0.0018058, -0.0020241, 0.0020775
7: -0.0224507, -0.0112852, -0.0221474, -0.0111100, -0.0078689, 0.0073499
8: 0.9748068, 0.9852751, 0.9751647, 0.9854856, -0.0077200, 0.0071031
9: -0.0003469, 0.0070335, -0.0004696, 0.0068188, -0.0048824, 0.0052503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042165, upper bound: 0.0044555
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042165, upper bound: 0.0044555
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152255, 0.0181923, 0.0152281, 0.0180347, -0.0023560, 0.0023307
1: -0.0018085, 0.0003104, -0.0018061, 0.0002009, -0.0017393, 0.0016997
2: 0.0035896, 0.0045462, 0.0036401, 0.0045454, -0.0007464, 0.0007521
3: 0.0013428, 0.0027151, 0.0013602, 0.0026958, -0.0009341, 0.0010282
4: -0.0045809, -0.0026088, -0.0045168, -0.0026102, -0.0013540, 0.0013128
5: -0.0003371, 0.0009412, -0.0002726, 0.0009409, -0.0010329, 0.0010626
6: -0.0049720, -0.0017318, -0.0049868, -0.0018104, -0.0019950, 0.0021385
7: -0.0225420, -0.0111150, -0.0221470, -0.0111298, -0.0078888, 0.0076605
8: 0.9747007, 0.9854748, 0.9751647, 0.9854617, -0.0077025, 0.0075348
9: -0.0004654, 0.0070965, -0.0004556, 0.0068187, -0.0051117, 0.0052582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042164, upper bound: 0.0044555
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042164, upper bound: 0.0044555
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0152997, 0.0181626, 0.0152761, 0.0181633, -0.0021630, 0.0021928
1: -0.0017521, 0.0002922, -0.0017711, 0.0002929, -0.0015798, 0.0016024
2: 0.0035994, 0.0045226, 0.0035992, 0.0045299, -0.0007025, 0.0006930
3: 0.0013414, 0.0026917, 0.0013386, 0.0027049, -0.0009598, 0.0009484
4: -0.0045653, -0.0026360, -0.0045656, -0.0026292, -0.0013002, 0.0012884
5: -0.0003268, 0.0009063, -0.0003273, 0.0009184, -0.0009757, 0.0009616
6: -0.0049893, -0.0017603, -0.0050060, -0.0017495, -0.0021051, 0.0021045
7: -0.0224507, -0.0112852, -0.0224519, -0.0112453, -0.0075666, 0.0074884
8: 0.9748068, 0.9852751, 0.9748068, 0.9853226, -0.0073459, 0.0072578
9: -0.0003469, 0.0070335, -0.0003748, 0.0070339, -0.0049775, 0.0050360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152255, 0.0181923, 0.0152909, 0.0181629, -0.0023942, 0.0021795
1: -0.0018085, 0.0003104, -0.0017593, 0.0002925, -0.0017655, 0.0015898
2: 0.0035896, 0.0045462, 0.0035993, 0.0045253, -0.0006984, 0.0007650
3: 0.0013428, 0.0027151, 0.0013406, 0.0026994, -0.0009485, 0.0010513
4: -0.0045809, -0.0026088, -0.0045655, -0.0026335, -0.0013064, 0.0013356
5: -0.0003371, 0.0009412, -0.0003270, 0.0009110, -0.0009670, 0.0010789
6: -0.0049720, -0.0017318, -0.0049935, -0.0017543, -0.0020749, 0.0021649
7: -0.0225420, -0.0111150, -0.0224515, -0.0112704, -0.0075931, 0.0077990
8: 0.9747007, 0.9854748, 0.9748068, 0.9852917, -0.0073336, 0.0076951
9: -0.0004654, 0.0070965, -0.0003569, 0.0070338, -0.0052074, 0.0050460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
time: 0.85 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
time: 0.75 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.73 seconds
IS_A1_A1_B1_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0031649, upper bound: 0.0024678
IS_A1_A1_B1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0046688, upper bound: 0.0047116
IS_A1_A1_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0046956, upper bound: 0.0047410
IS_A1_A1_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0046956, upper bound: 0.0047410
IS_A1_A1_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
IS_A1_A1_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
IS_A1_A1_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
IS_A1_A1_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041534, upper bound: 0.0039263
IS_A1_A1_B2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
IS_A1_A1_B2_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0039183, upper bound: 0.0037888
IS_A1_A2_B1_B1_B1_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0024678, upper bound: 0.0031649
IS_A1_A2_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047116, upper bound: 0.0046688
IS_A1_A2_B1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047410, upper bound: 0.0046956
IS_A1_A2_B1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047410, upper bound: 0.0046956
IS_A1_A2_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
IS_A1_A2_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
IS_A1_A2_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
IS_A1_A2_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0039263, upper bound: 0.0041534
IS_A1_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042784, upper bound: 0.0036010
IS_A1_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042784, upper bound: 0.0047864
IS_A1_A2_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0046340, upper bound: 0.0043386
IS_A1_A2_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0046123, upper bound: 0.0042946
IS_A1_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
IS_A1_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
IS_A1_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
IS_A1_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043857, upper bound: 0.0046271
IS_A1_A2_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
IS_A1_A2_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
IS_A1_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
IS_A1_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0043856, upper bound: 0.0043407
IS_A1_A2_B2_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0024204, upper bound: 0.0035290
IS_A1_A2_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0046775, upper bound: 0.0047517
IS_A1_A2_B2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047026, upper bound: 0.0047754
IS_A1_A2_B2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047026, upper bound: 0.0047754
IS_A1_A2_B2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
IS_A1_A2_B2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
IS_A1_A2_B2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
IS_A1_A2_B2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041968, upper bound: 0.0045541
IS_A1_A2_B2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
IS_A1_A2_B2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
IS_A1_A2_B2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
IS_A1_A2_B2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0044622, upper bound: 0.0042132
IS_A1_A2_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
IS_A1_A2_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
IS_A1_A2_B2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
IS_A1_A2_B2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041709, upper bound: 0.0042174
IS_A2_A2_B1_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
IS_A2_A2_B1_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0037888, upper bound: 0.0039183
IS_A2_A2_B2_B1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0036278, upper bound: 0.0024198
IS_A2_A2_B2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047689, upper bound: 0.0046600
IS_A2_A2_B2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
IS_A2_A2_B2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
IS_A2_A2_B2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
IS_A2_A2_B2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
IS_A2_A2_B2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
IS_A2_A2_B2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045640, upper bound: 0.0041498
IS_A2_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
IS_A2_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
IS_A2_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
IS_A2_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042701, upper bound: 0.0044555
IS_A2_A2_B2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
IS_A2_A2_B2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
IS_A2_A2_B2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
IS_A2_A2_B2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042664, upper bound: 0.0041354
IS_A2_A2_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
IS_A2_A2_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
IS_A2_A2_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
IS_A2_A2_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0047930, upper bound: 0.0046851
IS_A2_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
IS_A2_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
IS_A2_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
IS_A2_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0045278, upper bound: 0.0041498
IS_A2_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042165, upper bound: 0.0044555
IS_A2_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042165, upper bound: 0.0044555
IS_A2_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042164, upper bound: 0.0044555
IS_A2_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0042164, upper bound: 0.0044555
IS_A2_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
IS_A2_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
IS_A2_A2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354
IS_A2_A2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.73
Output dim: 8, lower bound: -0.0041845, upper bound: 0.0041354

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0153398, 0.0180183, 0.0154581, 0.0180215, -0.0020093, 0.0020037
1: -0.0017241, 0.0002045, -0.0016500, 0.0001930, -0.0014775, 0.0014719
2: 0.0036482, 0.0045100, 0.0036452, 0.0044710, -0.0006406, 0.0006428
3: 0.0013492, 0.0026221, 0.0013861, 0.0026632, -0.0009194, 0.0008679
4: -0.0044353, -0.0026358, -0.0045004, -0.0027272, -0.0011587, 0.0011869
5: -0.0002786, 0.0008903, -0.0002681, 0.0008479, -0.0008980, 0.0009019
6: -0.0050074, -0.0020191, -0.0048876, -0.0018501, -0.0019455, 0.0018405
7: -0.0216980, -0.0112996, -0.0220565, -0.0118289, -0.0067380, 0.0068939
8: 0.9754941, 0.9852275, 0.9752457, 0.9847150, -0.0065524, 0.0066465
9: -0.0003298, 0.0065373, 0.0000230, 0.0067618, -0.0045804, 0.0044816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: B, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: B, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: B, layer: 3, pos: 225
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 70
type: B, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 122
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: A, layer: 3, pos: 72
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 99
type: B, layer: 3, pos: 99

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043104, upper bound: 0.0026772
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043041, upper bound: 0.0043697
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0153644, 0.0180175, 0.0153737, 0.0180570, -0.0020773, 0.0020635
1: -0.0017043, 0.0002035, -0.0017137, 0.0002190, -0.0015017, 0.0015156
2: 0.0036484, 0.0045024, 0.0036338, 0.0044976, -0.0006598, 0.0006673
3: 0.0013519, 0.0026082, 0.0013861, 0.0026919, -0.0009462, 0.0008339
4: -0.0044349, -0.0026426, -0.0045196, -0.0026983, -0.0011888, 0.0013207
5: -0.0002778, 0.0008778, -0.0002835, 0.0008872, -0.0009251, 0.0009109
6: -0.0049906, -0.0020307, -0.0048726, -0.0018119, -0.0021197, 0.0017890
7: -0.0216967, -0.0113397, -0.0221706, -0.0116502, -0.0069254, 0.0076493
8: 0.9754941, 0.9851775, 0.9751150, 0.9849264, -0.0067441, 0.0072569
9: -0.0003016, 0.0065368, -0.0001011, 0.0068398, -0.0050646, 0.0046119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.10 + 597.62 = 600.72 seconds
