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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9906129, 0.9926358, 0.9906129, 0.9926358, -0.0010327, 0.0010327)
1: (-0.0036030, -0.0030989, -0.0036030, -0.0030989, -0.0002573, 0.0002573)
2: (0.0063687, 0.0090399, 0.0063687, 0.0090399, -0.0013637, 0.0013637)
3: (-0.0053877, -0.0041719, -0.0053877, -0.0041719, -0.0006207, 0.0006207)
4: (0.0017605, 0.0022775, 0.0017605, 0.0022775, -0.0002639, 0.0002639)
5: (0.0069696, 0.0103293, 0.0069696, 0.0103293, -0.0017151, 0.0017151)
6: (-0.0010809, -0.0002281, -0.0010809, -0.0002281, -0.0004353, 0.0004353)
7: (-0.0059341, -0.0037278, -0.0059341, -0.0037278, -0.0011263, 0.0011263)
8: (-0.0026848, -0.0015246, -0.0026848, -0.0015246, -0.0005923, 0.0005923)
9: (-0.0000960, 0.0012494, -0.0000960, 0.0012494, -0.0006868, 0.0006868)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.48 = 3.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0007083, upper bound: 0.0007083

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006871, upper bound: 0.0006512
time: 0.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006870, upper bound: 0.0006871
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.39 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 0, lower bound: -0.0006871, upper bound: 0.0006512
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 0, lower bound: -0.0006870, upper bound: 0.0006871

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9906319, 0.9925524, 0.9906129, 0.9926099, -0.0009651, 0.0009521
1: -0.0035982, -0.0031197, -0.0036030, -0.0031054, -0.0002405, 0.0002372
2: 0.0064788, 0.0090147, 0.0064028, 0.0090398, -0.0012572, 0.0012745
3: -0.0053762, -0.0042220, -0.0053876, -0.0041874, -0.0005801, 0.0005722
4: 0.0017818, 0.0022727, 0.0017671, 0.0022775, -0.0002433, 0.0002467
5: 0.0071081, 0.0102976, 0.0070124, 0.0103291, -0.0015813, 0.0016029
6: -0.0010728, -0.0002633, -0.0010808, -0.0002390, -0.0004068, 0.0004013
7: -0.0059133, -0.0038188, -0.0059340, -0.0037560, -0.0010526, 0.0010384
8: -0.0026739, -0.0015724, -0.0026848, -0.0015394, -0.0005536, 0.0005461
9: -0.0000406, 0.0012366, -0.0000789, 0.0012493, -0.0006332, 0.0006419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006213
time: 0.62 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006172
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9906131, 0.9925992, 0.9906129, 0.9926266, -0.0010203, 0.0009381
1: -0.0036029, -0.0031080, -0.0036030, -0.0031012, -0.0002542, 0.0002337
2: 0.0064169, 0.0090396, 0.0063809, 0.0090398, -0.0012387, 0.0013473
3: -0.0053875, -0.0041938, -0.0053877, -0.0041774, -0.0006132, 0.0005638
4: 0.0017699, 0.0022775, 0.0017629, 0.0022775, -0.0002398, 0.0002608
5: 0.0070302, 0.0103289, 0.0069849, 0.0103292, -0.0015580, 0.0016945
6: -0.0010807, -0.0002435, -0.0010808, -0.0002320, -0.0004301, 0.0003954
7: -0.0059338, -0.0037677, -0.0059340, -0.0037379, -0.0011127, 0.0010231
8: -0.0026847, -0.0015455, -0.0026848, -0.0015299, -0.0005852, 0.0005380
9: -0.0000717, 0.0012492, -0.0000899, 0.0012493, -0.0006239, 0.0006785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006533
time: 0.59 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006527
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.66 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006213
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006172
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006533
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0006527, upper bound: 0.0006527

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9906129, 0.9926099, -0.0009641, 0.0009206
1: -0.0035980, -0.0031250, -0.0036030, -0.0031054, -0.0002402, 0.0002294
2: 0.0065068, 0.0090137, 0.0064028, 0.0090398, -0.0012157, 0.0012730
3: -0.0053757, -0.0042347, -0.0053876, -0.0041874, -0.0005794, 0.0005533
4: 0.0017873, 0.0022725, 0.0017671, 0.0022775, -0.0002353, 0.0002464
5: 0.0071433, 0.0102962, 0.0070124, 0.0103291, -0.0015290, 0.0016011
6: -0.0010725, -0.0002722, -0.0010808, -0.0002390, -0.0004064, 0.0003881
7: -0.0059124, -0.0038419, -0.0059340, -0.0037560, -0.0010514, 0.0010041
8: -0.0026734, -0.0015846, -0.0026848, -0.0015394, -0.0005529, 0.0005280
9: -0.0000264, 0.0012361, -0.0000789, 0.0012493, -0.0006123, 0.0006412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006522, upper bound: 0.0006168
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006522, upper bound: 0.0006168
time: 0.61 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.9905773, 0.9924693, 0.9906136, 0.9925830, -0.0010643, 0.0009222
1: -0.0036118, -0.0031404, -0.0036028, -0.0031121, -0.0002652, 0.0002298
2: 0.0065885, 0.0090868, 0.0064383, 0.0090390, -0.0012177, 0.0014054
3: -0.0054091, -0.0042719, -0.0053873, -0.0042036, -0.0006397, 0.0005542
4: 0.0018031, 0.0022866, 0.0017740, 0.0022774, -0.0002357, 0.0002720
5: 0.0072461, 0.0103883, 0.0070571, 0.0103281, -0.0015315, 0.0017676
6: -0.0010958, -0.0002983, -0.0010805, -0.0002503, -0.0004486, 0.0003887
7: -0.0059729, -0.0039094, -0.0059333, -0.0037853, -0.0011607, 0.0010057
8: -0.0027052, -0.0016201, -0.0026844, -0.0015548, -0.0006104, 0.0005289
9: 0.0000147, 0.0012730, -0.0000610, 0.0012489, -0.0006133, 0.0007078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006311, upper bound: 0.0006172
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006311, upper bound: 0.0006172
time: 0.67 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9906129, 0.9926266, -0.0010193, 0.0009069
1: -0.0036027, -0.0031139, -0.0036030, -0.0031012, -0.0002540, 0.0002260
2: 0.0064483, 0.0090385, 0.0063809, 0.0090398, -0.0011975, 0.0013459
3: -0.0053871, -0.0042081, -0.0053877, -0.0041774, -0.0006126, 0.0005451
4: 0.0017759, 0.0022773, 0.0017629, 0.0022775, -0.0002318, 0.0002605
5: 0.0070697, 0.0103275, 0.0069849, 0.0103292, -0.0015061, 0.0016928
6: -0.0010804, -0.0002535, -0.0010808, -0.0002320, -0.0004297, 0.0003823
7: -0.0059330, -0.0037936, -0.0059340, -0.0037379, -0.0011117, 0.0009891
8: -0.0026842, -0.0015592, -0.0026848, -0.0015299, -0.0005846, 0.0005201
9: -0.0000559, 0.0012487, -0.0000899, 0.0012493, -0.0006031, 0.0006779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006534
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006534
time: 0.64 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.9905506, 0.9925197, 0.9906136, 0.9925996, -0.0011029, 0.0009147
1: -0.0036185, -0.0031278, -0.0036028, -0.0031079, -0.0002748, 0.0002279
2: 0.0065219, 0.0091221, 0.0064164, 0.0090390, -0.0012079, 0.0014563
3: -0.0054251, -0.0042416, -0.0053873, -0.0041936, -0.0006629, 0.0005498
4: 0.0017902, 0.0022935, 0.0017698, 0.0022774, -0.0002338, 0.0002819
5: 0.0071623, 0.0104327, 0.0070295, 0.0103281, -0.0015192, 0.0018317
6: -0.0011071, -0.0002770, -0.0010806, -0.0002433, -0.0004649, 0.0003856
7: -0.0060020, -0.0038544, -0.0059333, -0.0037672, -0.0012028, 0.0009977
8: -0.0027205, -0.0015911, -0.0026844, -0.0015453, -0.0006326, 0.0005247
9: -0.0000188, 0.0012908, -0.0000720, 0.0012489, -0.0006084, 0.0007335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006527
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006526
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.73 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006522, upper bound: 0.0006168
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006522, upper bound: 0.0006168
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006311, upper bound: 0.0006172
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006311, upper bound: 0.0006172
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006534
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006534
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006527
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006172, upper bound: 0.0006526

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9906138, 0.9925877, -0.0009326, 0.0009196
1: -0.0035980, -0.0031250, -0.0036028, -0.0031109, -0.0002324, 0.0002291
2: 0.0065068, 0.0090137, 0.0064321, 0.0090388, -0.0012144, 0.0012315
3: -0.0053757, -0.0042347, -0.0053872, -0.0042007, -0.0005605, 0.0005527
4: 0.0017873, 0.0022725, 0.0017728, 0.0022773, -0.0002350, 0.0002384
5: 0.0071433, 0.0102962, 0.0070494, 0.0103278, -0.0015274, 0.0015489
6: -0.0010725, -0.0002722, -0.0010805, -0.0002484, -0.0003931, 0.0003877
7: -0.0059124, -0.0038419, -0.0059332, -0.0037802, -0.0010171, 0.0010030
8: -0.0026734, -0.0015846, -0.0026843, -0.0015521, -0.0005349, 0.0005275
9: -0.0000264, 0.0012361, -0.0000641, 0.0012488, -0.0006116, 0.0006202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9905505, 0.9925281, -0.0009201, 0.0010223
1: -0.0035980, -0.0031250, -0.0036185, -0.0031258, -0.0002293, 0.0002547
2: 0.0065068, 0.0090137, 0.0065109, 0.0091223, -0.0013499, 0.0012150
3: -0.0053757, -0.0042347, -0.0054252, -0.0042366, -0.0005530, 0.0006144
4: 0.0017873, 0.0022725, 0.0017881, 0.0022935, -0.0002613, 0.0002352
5: 0.0071433, 0.0102962, 0.0071485, 0.0104330, -0.0016978, 0.0015282
6: -0.0010725, -0.0002722, -0.0011072, -0.0002735, -0.0003879, 0.0004309
7: -0.0059124, -0.0038419, -0.0060022, -0.0038453, -0.0010035, 0.0011149
8: -0.0026734, -0.0015846, -0.0027206, -0.0015864, -0.0005277, 0.0005863
9: -0.0000264, 0.0012361, -0.0000244, 0.0012909, -0.0006799, 0.0006119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905773, 0.9924693, 0.9906326, 0.9925251, -0.0010146, 0.0008845
1: -0.0036118, -0.0031404, -0.0035981, -0.0031265, -0.0002528, 0.0002204
2: 0.0065885, 0.0090868, 0.0065147, 0.0090138, -0.0011679, 0.0013398
3: -0.0054091, -0.0042719, -0.0053758, -0.0042383, -0.0006098, 0.0005316
4: 0.0018031, 0.0022866, 0.0017888, 0.0022725, -0.0002261, 0.0002593
5: 0.0072461, 0.0103883, 0.0071532, 0.0102965, -0.0014690, 0.0016851
6: -0.0010958, -0.0002983, -0.0010725, -0.0002747, -0.0004277, 0.0003728
7: -0.0059729, -0.0039094, -0.0059126, -0.0038484, -0.0011066, 0.0009646
8: -0.0027052, -0.0016201, -0.0026735, -0.0015880, -0.0005819, 0.0005073
9: 0.0000147, 0.0012730, -0.0000225, 0.0012362, -0.0005882, 0.0006748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0005726
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006063
time: 0.61 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905773, 0.9924693, 0.9906138, 0.9925739, -0.0010637, 0.0009172
1: -0.0036118, -0.0031404, -0.0036028, -0.0031143, -0.0002651, 0.0002285
2: 0.0065885, 0.0090868, 0.0064504, 0.0090387, -0.0012112, 0.0014046
3: -0.0054091, -0.0042719, -0.0053872, -0.0042090, -0.0006393, 0.0005513
4: 0.0018031, 0.0022866, 0.0017763, 0.0022773, -0.0002344, 0.0002719
5: 0.0072461, 0.0103883, 0.0070723, 0.0103278, -0.0015233, 0.0017667
6: -0.0010958, -0.0002983, -0.0010805, -0.0002542, -0.0004484, 0.0003866
7: -0.0059729, -0.0039094, -0.0059331, -0.0037953, -0.0011601, 0.0010004
8: -0.0027052, -0.0016201, -0.0026843, -0.0015601, -0.0006101, 0.0005261
9: 0.0000147, 0.0012730, -0.0000549, 0.0012488, -0.0006100, 0.0007074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0005726
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006063
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9906319, 0.9925524, -0.0009462, 0.0009395
1: -0.0036027, -0.0031139, -0.0035982, -0.0031197, -0.0002358, 0.0002341
2: 0.0064483, 0.0090385, 0.0064788, 0.0090147, -0.0012406, 0.0012494
3: -0.0053871, -0.0042081, -0.0053762, -0.0042220, -0.0005687, 0.0005647
4: 0.0017759, 0.0022773, 0.0017818, 0.0022727, -0.0002401, 0.0002418
5: 0.0070697, 0.0103275, 0.0071081, 0.0102976, -0.0015604, 0.0015714
6: -0.0010804, -0.0002535, -0.0010728, -0.0002633, -0.0003988, 0.0003960
7: -0.0059330, -0.0037936, -0.0059133, -0.0038188, -0.0010319, 0.0010247
8: -0.0026842, -0.0015592, -0.0026739, -0.0015724, -0.0005427, 0.0005389
9: -0.0000559, 0.0012487, -0.0000406, 0.0012366, -0.0006248, 0.0006293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006533
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006533
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9906131, 0.9925992, -0.0009352, 0.0009051
1: -0.0036027, -0.0031139, -0.0036029, -0.0031080, -0.0002330, 0.0002255
2: 0.0064483, 0.0090385, 0.0064169, 0.0090396, -0.0011951, 0.0012349
3: -0.0053871, -0.0042081, -0.0053875, -0.0041938, -0.0005621, 0.0005440
4: 0.0017759, 0.0022773, 0.0017699, 0.0022775, -0.0002313, 0.0002390
5: 0.0070697, 0.0103275, 0.0070302, 0.0103289, -0.0015032, 0.0015532
6: -0.0010804, -0.0002535, -0.0010807, -0.0002435, -0.0003942, 0.0003815
7: -0.0059330, -0.0037936, -0.0059338, -0.0037677, -0.0010200, 0.0009871
8: -0.0026842, -0.0015592, -0.0026847, -0.0015455, -0.0005364, 0.0005191
9: -0.0000559, 0.0012487, -0.0000717, 0.0012492, -0.0006019, 0.0006220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006534
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006534
time: 0.62 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905506, 0.9925197, 0.9906326, 0.9925251, -0.0010335, 0.0009417
1: -0.0036185, -0.0031278, -0.0035981, -0.0031265, -0.0002575, 0.0002346
2: 0.0065219, 0.0091221, 0.0065147, 0.0090138, -0.0012435, 0.0013647
3: -0.0054251, -0.0042416, -0.0053758, -0.0042383, -0.0006212, 0.0005660
4: 0.0017902, 0.0022935, 0.0017888, 0.0022725, -0.0002407, 0.0002641
5: 0.0071623, 0.0104327, 0.0071532, 0.0102965, -0.0015640, 0.0017164
6: -0.0011071, -0.0002770, -0.0010725, -0.0002747, -0.0004356, 0.0003969
7: -0.0060020, -0.0038544, -0.0059126, -0.0038484, -0.0011272, 0.0010270
8: -0.0027205, -0.0015911, -0.0026735, -0.0015880, -0.0005928, 0.0005401
9: -0.0000188, 0.0012908, -0.0000225, 0.0012362, -0.0006263, 0.0006873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005725, upper bound: 0.0006428
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006063, upper bound: 0.0006428
time: 0.62 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905506, 0.9925197, 0.9906138, 0.9925739, -0.0010362, 0.0009130
1: -0.0036185, -0.0031278, -0.0036028, -0.0031143, -0.0002582, 0.0002275
2: 0.0065219, 0.0091221, 0.0064504, 0.0090387, -0.0012055, 0.0013682
3: -0.0054251, -0.0042416, -0.0053872, -0.0042090, -0.0006228, 0.0005487
4: 0.0017902, 0.0022935, 0.0017763, 0.0022773, -0.0002333, 0.0002648
5: 0.0071623, 0.0104327, 0.0070723, 0.0103278, -0.0015163, 0.0017209
6: -0.0011071, -0.0002770, -0.0010805, -0.0002542, -0.0004368, 0.0003848
7: -0.0060020, -0.0038544, -0.0059331, -0.0037953, -0.0011301, 0.0009957
8: -0.0027205, -0.0015911, -0.0026843, -0.0015601, -0.0005943, 0.0005236
9: -0.0000188, 0.0012908, -0.0000549, 0.0012488, -0.0006072, 0.0006891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006063, upper bound: 0.0006040
time: 0.63 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006063, upper bound: 0.0006428
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.74 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006310, upper bound: 0.0006213
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0005726
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006063
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0005726
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006063
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006533
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006533
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006534
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006534
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0005725, upper bound: 0.0006428
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006063, upper bound: 0.0006428
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006063, upper bound: 0.0006040
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0006063, upper bound: 0.0006428

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9906328, 0.9925311, -0.0008819, 0.0008819
1: -0.0035980, -0.0031250, -0.0035980, -0.0031250, -0.0002198, 0.0002198
2: 0.0065068, 0.0090137, 0.0065068, 0.0090137, -0.0011646, 0.0011646
3: -0.0053757, -0.0042347, -0.0053757, -0.0042347, -0.0005301, 0.0005301
4: 0.0017873, 0.0022725, 0.0017873, 0.0022725, -0.0002254, 0.0002254
5: 0.0071433, 0.0102962, 0.0071433, 0.0102962, -0.0014647, 0.0014647
6: -0.0010725, -0.0002722, -0.0010725, -0.0002722, -0.0003718, 0.0003718
7: -0.0059124, -0.0038419, -0.0059124, -0.0038419, -0.0009619, 0.0009619
8: -0.0026734, -0.0015846, -0.0026734, -0.0015846, -0.0005058, 0.0005058
9: -0.0000264, 0.0012361, -0.0000264, 0.0012361, -0.0005865, 0.0005865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005860, upper bound: 0.0006116
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006117
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9906139, 0.9925754, -0.0009384, 0.0009147
1: -0.0035980, -0.0031250, -0.0036027, -0.0031139, -0.0002338, 0.0002279
2: 0.0065068, 0.0090137, 0.0064483, 0.0090385, -0.0012078, 0.0012392
3: -0.0053757, -0.0042347, -0.0053871, -0.0042081, -0.0005640, 0.0005498
4: 0.0017873, 0.0022725, 0.0017759, 0.0022773, -0.0002338, 0.0002398
5: 0.0071433, 0.0102962, 0.0070697, 0.0103275, -0.0015191, 0.0015585
6: -0.0010725, -0.0002722, -0.0010804, -0.0002535, -0.0003956, 0.0003856
7: -0.0059124, -0.0038419, -0.0059330, -0.0037936, -0.0010235, 0.0009976
8: -0.0026734, -0.0015846, -0.0026842, -0.0015592, -0.0005382, 0.0005246
9: -0.0000264, 0.0012361, -0.0000559, 0.0012487, -0.0006083, 0.0006241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0005753
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006116
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9905773, 0.9924693, -0.0008730, 0.0009991
1: -0.0035980, -0.0031250, -0.0036118, -0.0031404, -0.0002175, 0.0002489
2: 0.0065068, 0.0090137, 0.0065885, 0.0090868, -0.0013193, 0.0011528
3: -0.0053757, -0.0042347, -0.0054091, -0.0042719, -0.0005247, 0.0006005
4: 0.0017873, 0.0022725, 0.0018031, 0.0022866, -0.0002553, 0.0002231
5: 0.0071433, 0.0102962, 0.0072461, 0.0103883, -0.0016593, 0.0014499
6: -0.0010725, -0.0002722, -0.0010958, -0.0002983, -0.0003680, 0.0004212
7: -0.0059124, -0.0038419, -0.0059729, -0.0039094, -0.0009521, 0.0010897
8: -0.0026734, -0.0015846, -0.0027052, -0.0016201, -0.0005007, 0.0005730
9: -0.0000264, 0.0012361, 0.0000147, 0.0012730, -0.0006645, 0.0005806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005804, upper bound: 0.0006107
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006108
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906328, 0.9925311, 0.9905506, 0.9925197, -0.0009139, 0.0010180
1: -0.0035980, -0.0031250, -0.0036185, -0.0031278, -0.0002277, 0.0002536
2: 0.0065068, 0.0090137, 0.0065219, 0.0091221, -0.0013442, 0.0012068
3: -0.0053757, -0.0042347, -0.0054251, -0.0042416, -0.0005493, 0.0006118
4: 0.0017873, 0.0022725, 0.0017902, 0.0022935, -0.0002602, 0.0002336
5: 0.0071433, 0.0102962, 0.0071623, 0.0104327, -0.0016906, 0.0015179
6: -0.0010725, -0.0002722, -0.0011071, -0.0002770, -0.0003853, 0.0004291
7: -0.0059124, -0.0038419, -0.0060020, -0.0038544, -0.0009968, 0.0011102
8: -0.0026734, -0.0015846, -0.0027205, -0.0015911, -0.0005242, 0.0005839
9: -0.0000264, 0.0012361, -0.0000188, 0.0012908, -0.0006770, 0.0006078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0005745
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006108
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905937, 0.9924691, 0.9906825, 0.9925067, -0.0010083, 0.0008475
1: -0.0036078, -0.0031404, -0.0035856, -0.0031311, -0.0002512, 0.0002112
2: 0.0065887, 0.0090652, 0.0065392, 0.0089478, -0.0011191, 0.0013315
3: -0.0053992, -0.0042720, -0.0053458, -0.0042495, -0.0006060, 0.0005093
4: 0.0018031, 0.0022824, 0.0017935, 0.0022597, -0.0002166, 0.0002577
5: 0.0072463, 0.0103611, 0.0071840, 0.0102135, -0.0014075, 0.0016746
6: -0.0010889, -0.0002984, -0.0010515, -0.0002825, -0.0004250, 0.0003572
7: -0.0059550, -0.0039096, -0.0058581, -0.0038686, -0.0010997, 0.0009243
8: -0.0026958, -0.0016202, -0.0026448, -0.0015986, -0.0005783, 0.0004861
9: 0.0000148, 0.0012621, -0.0000102, 0.0012030, -0.0005636, 0.0006706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006007, upper bound: 0.0005688
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006091, upper bound: 0.0005691
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905813, 0.9924693, 0.9906496, 0.9925251, -0.0010080, 0.0008564
1: -0.0036109, -0.0031404, -0.0035938, -0.0031265, -0.0002512, 0.0002134
2: 0.0065886, 0.0090816, 0.0065148, 0.0089914, -0.0011308, 0.0013311
3: -0.0054067, -0.0042719, -0.0053656, -0.0042384, -0.0006058, 0.0005147
4: 0.0018031, 0.0022856, 0.0017888, 0.0022681, -0.0002189, 0.0002576
5: 0.0072461, 0.0103818, 0.0071534, 0.0102682, -0.0014223, 0.0016741
6: -0.0010942, -0.0002983, -0.0010653, -0.0002748, -0.0004249, 0.0003610
7: -0.0059686, -0.0039095, -0.0058940, -0.0038486, -0.0010994, 0.0009340
8: -0.0027030, -0.0016201, -0.0026637, -0.0015881, -0.0005782, 0.0004912
9: 0.0000147, 0.0012704, -0.0000224, 0.0012249, -0.0005695, 0.0006704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006007, upper bound: 0.0006089
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006091, upper bound: 0.0006091
time: 0.60 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905937, 0.9924691, 0.9906640, 0.9925556, -0.0010525, 0.0008738
1: -0.0036078, -0.0031404, -0.0035902, -0.0031189, -0.0002623, 0.0002177
2: 0.0065887, 0.0090652, 0.0064747, 0.0089724, -0.0011539, 0.0013899
3: -0.0053992, -0.0042720, -0.0053570, -0.0042201, -0.0006326, 0.0005252
4: 0.0018031, 0.0022824, 0.0017810, 0.0022645, -0.0002233, 0.0002690
5: 0.0072463, 0.0103611, 0.0071029, 0.0102444, -0.0014513, 0.0017481
6: -0.0010889, -0.0002984, -0.0010593, -0.0002620, -0.0004437, 0.0003684
7: -0.0059550, -0.0039096, -0.0058784, -0.0038154, -0.0011479, 0.0009530
8: -0.0026958, -0.0016202, -0.0026555, -0.0015706, -0.0006037, 0.0005012
9: 0.0000148, 0.0012621, -0.0000426, 0.0012154, -0.0005812, 0.0007000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006208, upper bound: 0.0005604
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005609
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905813, 0.9924693, 0.9906313, 0.9925737, -0.0010571, 0.0008847
1: -0.0036109, -0.0031404, -0.0035984, -0.0031144, -0.0002634, 0.0002204
2: 0.0065886, 0.0090816, 0.0064505, 0.0090156, -0.0011683, 0.0013959
3: -0.0054067, -0.0042719, -0.0053766, -0.0042091, -0.0006354, 0.0005317
4: 0.0018031, 0.0022856, 0.0017764, 0.0022728, -0.0002261, 0.0002702
5: 0.0072461, 0.0103818, 0.0070725, 0.0102987, -0.0014694, 0.0017557
6: -0.0010942, -0.0002983, -0.0010731, -0.0002542, -0.0004456, 0.0003729
7: -0.0059686, -0.0039095, -0.0059140, -0.0037954, -0.0011529, 0.0009649
8: -0.0027030, -0.0016201, -0.0026743, -0.0015601, -0.0006063, 0.0005074
9: 0.0000147, 0.0012704, -0.0000548, 0.0012371, -0.0005884, 0.0007030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006208, upper bound: 0.0005938
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005942
time: 0.60 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9906328, 0.9925311, -0.0009147, 0.0009384
1: -0.0036027, -0.0031139, -0.0035980, -0.0031250, -0.0002279, 0.0002338
2: 0.0064483, 0.0090385, 0.0065068, 0.0090137, -0.0012392, 0.0012078
3: -0.0053871, -0.0042081, -0.0053757, -0.0042347, -0.0005498, 0.0005640
4: 0.0017759, 0.0022773, 0.0017873, 0.0022725, -0.0002398, 0.0002338
5: 0.0070697, 0.0103275, 0.0071433, 0.0102962, -0.0015585, 0.0015191
6: -0.0010804, -0.0002535, -0.0010725, -0.0002722, -0.0003856, 0.0003956
7: -0.0059330, -0.0037936, -0.0059124, -0.0038419, -0.0009976, 0.0010235
8: -0.0026842, -0.0015592, -0.0026734, -0.0015846, -0.0005246, 0.0005382
9: -0.0000559, 0.0012487, -0.0000264, 0.0012361, -0.0006241, 0.0006083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006434
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9905773, 0.9924693, -0.0009057, 0.0010556
1: -0.0036027, -0.0031139, -0.0036118, -0.0031404, -0.0002257, 0.0002630
2: 0.0064483, 0.0090385, 0.0065885, 0.0090868, -0.0013939, 0.0011960
3: -0.0053871, -0.0042081, -0.0054091, -0.0042719, -0.0005444, 0.0006344
4: 0.0017759, 0.0022773, 0.0018031, 0.0022866, -0.0002698, 0.0002315
5: 0.0070697, 0.0103275, 0.0072461, 0.0103883, -0.0017531, 0.0015043
6: -0.0010804, -0.0002535, -0.0010958, -0.0002983, -0.0003818, 0.0004450
7: -0.0059330, -0.0037936, -0.0059729, -0.0039094, -0.0009878, 0.0011512
8: -0.0026842, -0.0015592, -0.0027052, -0.0016201, -0.0005195, 0.0006054
9: -0.0000559, 0.0012487, 0.0000147, 0.0012730, -0.0007020, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006435
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9906139, 0.9925754, -0.0009040, 0.0009040
1: -0.0036027, -0.0031139, -0.0036027, -0.0031139, -0.0002253, 0.0002253
2: 0.0064483, 0.0090385, 0.0064483, 0.0090385, -0.0011937, 0.0011937
3: -0.0053871, -0.0042081, -0.0053871, -0.0042081, -0.0005433, 0.0005433
4: 0.0017759, 0.0022773, 0.0017759, 0.0022773, -0.0002310, 0.0002310
5: 0.0070697, 0.0103275, 0.0070697, 0.0103275, -0.0015014, 0.0015014
6: -0.0010804, -0.0002535, -0.0010804, -0.0002535, -0.0003811, 0.0003811
7: -0.0059330, -0.0037936, -0.0059330, -0.0037936, -0.0009859, 0.0009859
8: -0.0026842, -0.0015592, -0.0026842, -0.0015592, -0.0005185, 0.0005185
9: -0.0000559, 0.0012487, -0.0000559, 0.0012487, -0.0006012, 0.0006012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006435
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906139, 0.9925754, 0.9905506, 0.9925197, -0.0008945, 0.0010208
1: -0.0036027, -0.0031139, -0.0036185, -0.0031278, -0.0002229, 0.0002544
2: 0.0064483, 0.0090385, 0.0065219, 0.0091221, -0.0013480, 0.0011812
3: -0.0053871, -0.0042081, -0.0054251, -0.0042416, -0.0005376, 0.0006136
4: 0.0017759, 0.0022773, 0.0017902, 0.0022935, -0.0002609, 0.0002286
5: 0.0070697, 0.0103275, 0.0071623, 0.0104327, -0.0016954, 0.0014856
6: -0.0010804, -0.0002535, -0.0011071, -0.0002770, -0.0003771, 0.0004303
7: -0.0059330, -0.0037936, -0.0060020, -0.0038544, -0.0009756, 0.0011134
8: -0.0026842, -0.0015592, -0.0027205, -0.0015911, -0.0005131, 0.0005855
9: -0.0000559, 0.0012487, -0.0000188, 0.0012908, -0.0006789, 0.0005949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006435
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906007, 0.9925011, 0.9906487, 0.9925251, -0.0009984, 0.0009181
1: -0.0036060, -0.0031325, -0.0035940, -0.0031265, -0.0002488, 0.0002288
2: 0.0065464, 0.0090560, 0.0065148, 0.0089925, -0.0012124, 0.0013184
3: -0.0053950, -0.0042527, -0.0053661, -0.0042384, -0.0006001, 0.0005518
4: 0.0017949, 0.0022807, 0.0017888, 0.0022684, -0.0002346, 0.0002552
5: 0.0071931, 0.0103495, 0.0071534, 0.0102697, -0.0015248, 0.0016582
6: -0.0010860, -0.0002848, -0.0010657, -0.0002748, -0.0004209, 0.0003870
7: -0.0059474, -0.0038746, -0.0058950, -0.0038485, -0.0010889, 0.0010013
8: -0.0026918, -0.0016018, -0.0026643, -0.0015881, -0.0005727, 0.0005266
9: -0.0000065, 0.0012574, -0.0000224, 0.0012255, -0.0006106, 0.0006640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005507, upper bound: 0.0006299
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005609, upper bound: 0.0006303
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9905682, 0.9925196, 0.9906365, 0.9925251, -0.0010100, 0.0009356
1: -0.0036141, -0.0031279, -0.0035971, -0.0031265, -0.0002517, 0.0002331
2: 0.0065222, 0.0090989, 0.0065147, 0.0090086, -0.0012354, 0.0013337
3: -0.0054145, -0.0042417, -0.0053734, -0.0042384, -0.0006071, 0.0005623
4: 0.0017902, 0.0022890, 0.0017888, 0.0022715, -0.0002391, 0.0002581
5: 0.0071627, 0.0104035, 0.0071533, 0.0102898, -0.0015538, 0.0016775
6: -0.0010997, -0.0002771, -0.0010708, -0.0002747, -0.0004258, 0.0003944
7: -0.0059828, -0.0038546, -0.0059082, -0.0038485, -0.0011016, 0.0010204
8: -0.0027105, -0.0015913, -0.0026712, -0.0015880, -0.0005793, 0.0005366
9: -0.0000187, 0.0012791, -0.0000225, 0.0012336, -0.0006222, 0.0006717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0006299
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0006303
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905663, 0.9925196, 0.9906640, 0.9925556, -0.0010300, 0.0008762
1: -0.0036146, -0.0031279, -0.0035902, -0.0031189, -0.0002567, 0.0002183
2: 0.0065222, 0.0091014, 0.0064747, 0.0089724, -0.0011571, 0.0013601
3: -0.0054157, -0.0042417, -0.0053570, -0.0042201, -0.0006191, 0.0005266
4: 0.0017902, 0.0022894, 0.0017810, 0.0022645, -0.0002239, 0.0002632
5: 0.0071626, 0.0104066, 0.0071029, 0.0102444, -0.0014553, 0.0017107
6: -0.0011005, -0.0002771, -0.0010593, -0.0002620, -0.0004342, 0.0003694
7: -0.0059849, -0.0038546, -0.0058784, -0.0038154, -0.0011234, 0.0009557
8: -0.0027115, -0.0015912, -0.0026555, -0.0015706, -0.0005908, 0.0005026
9: -0.0000187, 0.0012803, -0.0000426, 0.0012154, -0.0005828, 0.0006850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0005922
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0005924
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905547, 0.9925196, 0.9906313, 0.9925737, -0.0010296, 0.0008849
1: -0.0036175, -0.0031279, -0.0035984, -0.0031144, -0.0002565, 0.0002205
2: 0.0065220, 0.0091167, 0.0064505, 0.0090156, -0.0011685, 0.0013596
3: -0.0054226, -0.0042416, -0.0053766, -0.0042091, -0.0006188, 0.0005318
4: 0.0017902, 0.0022924, 0.0017764, 0.0022728, -0.0002262, 0.0002631
5: 0.0071624, 0.0104258, 0.0070725, 0.0102987, -0.0014696, 0.0017100
6: -0.0011054, -0.0002771, -0.0010731, -0.0002542, -0.0004340, 0.0003730
7: -0.0059975, -0.0038545, -0.0059140, -0.0037954, -0.0011229, 0.0009651
8: -0.0027182, -0.0015912, -0.0026743, -0.0015601, -0.0005905, 0.0005075
9: -0.0000188, 0.0012880, -0.0000548, 0.0012371, -0.0005885, 0.0006847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0006299
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0006303
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.72 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005860, upper bound: 0.0006116
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006117
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0005753
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006116
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005804, upper bound: 0.0006107
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006108
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0005745
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006215, upper bound: 0.0006108
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006007, upper bound: 0.0005688
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006091, upper bound: 0.0005691
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006007, upper bound: 0.0006089
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006091, upper bound: 0.0006091
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006208, upper bound: 0.0005604
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005609
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006208, upper bound: 0.0005938
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005942
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006434
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006435
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006435
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0006435
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0006060, upper bound: 0.0006435
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005507, upper bound: 0.0006299
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005609, upper bound: 0.0006303
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0006299
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0006303
IS_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0005922
IS_A2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0005924
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0006299
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.72
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0006303

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906827, 0.9925140, 0.9906489, 0.9925310, -0.0008451, 0.0008703
1: -0.0035856, -0.0031293, -0.0035940, -0.0031250, -0.0002106, 0.0002169
2: 0.0065295, 0.0089477, 0.0065069, 0.0089923, -0.0011492, 0.0011159
3: -0.0053457, -0.0042451, -0.0053660, -0.0042348, -0.0005079, 0.0005231
4: 0.0017917, 0.0022597, 0.0017873, 0.0022683, -0.0002224, 0.0002160
5: 0.0071718, 0.0102133, 0.0071435, 0.0102694, -0.0014454, 0.0014035
6: -0.0010514, -0.0002795, -0.0010657, -0.0002723, -0.0003562, 0.0003669
7: -0.0058579, -0.0038607, -0.0058948, -0.0038420, -0.0009217, 0.0009492
8: -0.0026448, -0.0015944, -0.0026642, -0.0015846, -0.0004847, 0.0004992
9: -0.0000150, 0.0012029, -0.0000264, 0.0012254, -0.0005788, 0.0005620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005619, upper bound: 0.0006155
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005744, upper bound: 0.0006155
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906497, 0.9925311, 0.9906367, 0.9925311, -0.0008546, 0.0008760
1: -0.0035938, -0.0031250, -0.0035970, -0.0031250, -0.0002130, 0.0002183
2: 0.0065070, 0.0089912, 0.0065069, 0.0090084, -0.0011567, 0.0011285
3: -0.0053655, -0.0042348, -0.0053733, -0.0042348, -0.0005137, 0.0005265
4: 0.0017873, 0.0022681, 0.0017873, 0.0022714, -0.0002239, 0.0002184
5: 0.0071435, 0.0102680, 0.0071434, 0.0102896, -0.0014548, 0.0014194
6: -0.0010653, -0.0002723, -0.0010708, -0.0002722, -0.0003603, 0.0003693
7: -0.0058939, -0.0038420, -0.0059081, -0.0038420, -0.0009321, 0.0009554
8: -0.0026637, -0.0015846, -0.0026711, -0.0015846, -0.0004902, 0.0005024
9: -0.0000264, 0.0012248, -0.0000264, 0.0012335, -0.0005826, 0.0005684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006114, upper bound: 0.0006155
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006155, upper bound: 0.0006155
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906489, 0.9925310, 0.9906641, 0.9925576, -0.0009167, 0.0008714
1: -0.0035940, -0.0031250, -0.0035902, -0.0031184, -0.0002284, 0.0002171
2: 0.0065069, 0.0089923, 0.0064720, 0.0089722, -0.0011507, 0.0012104
3: -0.0053660, -0.0042348, -0.0053569, -0.0042189, -0.0005509, 0.0005238
4: 0.0017873, 0.0022683, 0.0017805, 0.0022644, -0.0002227, 0.0002343
5: 0.0071435, 0.0102694, 0.0070995, 0.0102442, -0.0014473, 0.0015224
6: -0.0010657, -0.0002723, -0.0010592, -0.0002611, -0.0003864, 0.0003673
7: -0.0058948, -0.0038420, -0.0058782, -0.0038131, -0.0009998, 0.0009504
8: -0.0026642, -0.0015846, -0.0026554, -0.0015694, -0.0005258, 0.0004998
9: -0.0000264, 0.0012254, -0.0000440, 0.0012153, -0.0005796, 0.0006096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005547
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005627
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906367, 0.9925311, 0.9906314, 0.9925753, -0.0009324, 0.0008830
1: -0.0035970, -0.0031250, -0.0035984, -0.0031140, -0.0002323, 0.0002200
2: 0.0065069, 0.0090084, 0.0064485, 0.0090155, -0.0011660, 0.0012313
3: -0.0053733, -0.0042348, -0.0053766, -0.0042082, -0.0005604, 0.0005307
4: 0.0017873, 0.0022714, 0.0017760, 0.0022728, -0.0002257, 0.0002383
5: 0.0071434, 0.0102896, 0.0070699, 0.0102985, -0.0014665, 0.0015486
6: -0.0010708, -0.0002722, -0.0010730, -0.0002536, -0.0003931, 0.0003722
7: -0.0059081, -0.0038420, -0.0059139, -0.0037937, -0.0010169, 0.0009631
8: -0.0026711, -0.0015846, -0.0026742, -0.0015592, -0.0005348, 0.0005065
9: -0.0000264, 0.0012335, -0.0000558, 0.0012370, -0.0005873, 0.0006201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005948
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005992
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906827, 0.9925140, 0.9905937, 0.9924691, -0.0008360, 0.0009893
1: -0.0035856, -0.0031293, -0.0036078, -0.0031404, -0.0002083, 0.0002465
2: 0.0065295, 0.0089477, 0.0065887, 0.0090652, -0.0013064, 0.0011039
3: -0.0053457, -0.0042451, -0.0053992, -0.0042720, -0.0005024, 0.0005946
4: 0.0017917, 0.0022597, 0.0018031, 0.0022824, -0.0002528, 0.0002137
5: 0.0071718, 0.0102133, 0.0072463, 0.0103611, -0.0016431, 0.0013884
6: -0.0010514, -0.0002795, -0.0010889, -0.0002984, -0.0003524, 0.0004170
7: -0.0058579, -0.0038607, -0.0059550, -0.0039096, -0.0009117, 0.0010790
8: -0.0026448, -0.0015944, -0.0026958, -0.0016202, -0.0004795, 0.0005674
9: -0.0000150, 0.0012029, 0.0000148, 0.0012621, -0.0006580, 0.0005560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005688, upper bound: 0.0006096
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005690, upper bound: 0.0006137
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906497, 0.9925311, 0.9905813, 0.9924693, -0.0008518, 0.0009925
1: -0.0035938, -0.0031250, -0.0036109, -0.0031404, -0.0002122, 0.0002473
2: 0.0065070, 0.0089912, 0.0065886, 0.0090816, -0.0013106, 0.0011248
3: -0.0053655, -0.0042348, -0.0054067, -0.0042719, -0.0005120, 0.0005965
4: 0.0017873, 0.0022681, 0.0018031, 0.0022856, -0.0002537, 0.0002177
5: 0.0071435, 0.0102680, 0.0072461, 0.0103818, -0.0016484, 0.0014147
6: -0.0010653, -0.0002723, -0.0010942, -0.0002983, -0.0003591, 0.0004184
7: -0.0058939, -0.0038420, -0.0059686, -0.0039095, -0.0009290, 0.0010825
8: -0.0026637, -0.0015846, -0.0027030, -0.0016201, -0.0004886, 0.0005693
9: -0.0000264, 0.0012248, 0.0000147, 0.0012704, -0.0006601, 0.0005665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006089, upper bound: 0.0006096
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006091, upper bound: 0.0006137
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906489, 0.9925310, 0.9906007, 0.9925011, -0.0009071, 0.0009829
1: -0.0035940, -0.0031250, -0.0036060, -0.0031325, -0.0002260, 0.0002449
2: 0.0065069, 0.0089923, 0.0065464, 0.0090560, -0.0012979, 0.0011979
3: -0.0053660, -0.0042348, -0.0053950, -0.0042527, -0.0005452, 0.0005908
4: 0.0017873, 0.0022683, 0.0017949, 0.0022807, -0.0002512, 0.0002318
5: 0.0071435, 0.0102694, 0.0071931, 0.0103495, -0.0016324, 0.0015066
6: -0.0010657, -0.0002723, -0.0010860, -0.0002848, -0.0003824, 0.0004143
7: -0.0058948, -0.0038420, -0.0059474, -0.0038746, -0.0009894, 0.0010720
8: -0.0026642, -0.0015846, -0.0026918, -0.0016018, -0.0005203, 0.0005638
9: -0.0000264, 0.0012254, -0.0000065, 0.0012574, -0.0006537, 0.0006033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006298, upper bound: 0.0005534
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005617
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906367, 0.9925311, 0.9905682, 0.9925196, -0.0009078, 0.0009924
1: -0.0035970, -0.0031250, -0.0036141, -0.0031279, -0.0002262, 0.0002473
2: 0.0065069, 0.0090084, 0.0065222, 0.0090989, -0.0013104, 0.0011987
3: -0.0053733, -0.0042348, -0.0054145, -0.0042417, -0.0005456, 0.0005964
4: 0.0017873, 0.0022714, 0.0017902, 0.0022890, -0.0002536, 0.0002320
5: 0.0071434, 0.0102896, 0.0071627, 0.0104035, -0.0016481, 0.0015077
6: -0.0010708, -0.0002722, -0.0010997, -0.0002771, -0.0003827, 0.0004183
7: -0.0059081, -0.0038420, -0.0059828, -0.0038546, -0.0009901, 0.0010823
8: -0.0026711, -0.0015846, -0.0027105, -0.0015913, -0.0005207, 0.0005692
9: -0.0000264, 0.0012335, -0.0000187, 0.0012791, -0.0006600, 0.0006037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006299, upper bound: 0.0005933
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005982
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905945, 0.9924596, 0.9906864, 0.9924722, -0.0009678, 0.0008369
1: -0.0036076, -0.0031428, -0.0035847, -0.0031397, -0.0002411, 0.0002085
2: 0.0066013, 0.0090641, 0.0065847, 0.0089428, -0.0011052, 0.0012779
3: -0.0053987, -0.0042778, -0.0053435, -0.0042702, -0.0005817, 0.0005030
4: 0.0018056, 0.0022822, 0.0018023, 0.0022587, -0.0002139, 0.0002473
5: 0.0072622, 0.0103597, 0.0072413, 0.0102071, -0.0013900, 0.0016073
6: -0.0010886, -0.0003024, -0.0010498, -0.0002971, -0.0004080, 0.0003528
7: -0.0059541, -0.0039200, -0.0058539, -0.0039063, -0.0010555, 0.0009128
8: -0.0026954, -0.0016256, -0.0026426, -0.0016184, -0.0005551, 0.0004800
9: 0.0000211, 0.0012615, 0.0000128, 0.0012004, -0.0005566, 0.0006436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005860, upper bound: 0.0005507
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005794, upper bound: 0.0005511
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905944, 0.9924608, 0.9906694, 0.9924831, -0.0009787, 0.0008623
1: -0.0036076, -0.0031425, -0.0035889, -0.0031370, -0.0002439, 0.0002149
2: 0.0065998, 0.0090643, 0.0065702, 0.0089653, -0.0011387, 0.0012924
3: -0.0053988, -0.0042770, -0.0053538, -0.0042636, -0.0005882, 0.0005183
4: 0.0018053, 0.0022823, 0.0017995, 0.0022631, -0.0002204, 0.0002501
5: 0.0072602, 0.0103599, 0.0072231, 0.0102355, -0.0014322, 0.0016255
6: -0.0010886, -0.0003019, -0.0010570, -0.0002925, -0.0004126, 0.0003635
7: -0.0059542, -0.0039187, -0.0058725, -0.0038943, -0.0010674, 0.0009405
8: -0.0026954, -0.0016249, -0.0026524, -0.0016121, -0.0005614, 0.0004946
9: 0.0000204, 0.0012616, 0.0000055, 0.0012118, -0.0005735, 0.0006509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005949, upper bound: 0.0005511
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005895, upper bound: 0.0005516
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905821, 0.9924597, 0.9906530, 0.9924914, -0.0009716, 0.0008452
1: -0.0036107, -0.0031428, -0.0035930, -0.0031349, -0.0002421, 0.0002106
2: 0.0066012, 0.0090806, 0.0065594, 0.0089869, -0.0011160, 0.0012830
3: -0.0054062, -0.0042777, -0.0053636, -0.0042587, -0.0005840, 0.0005080
4: 0.0018055, 0.0022854, 0.0017974, 0.0022673, -0.0002160, 0.0002483
5: 0.0072620, 0.0103804, 0.0072095, 0.0102626, -0.0014037, 0.0016137
6: -0.0010938, -0.0003023, -0.0010639, -0.0002890, -0.0004096, 0.0003563
7: -0.0059677, -0.0039199, -0.0058903, -0.0038854, -0.0010597, 0.0009218
8: -0.0027025, -0.0016256, -0.0026618, -0.0016074, -0.0005573, 0.0004847
9: 0.0000211, 0.0012698, 0.0000000, 0.0012227, -0.0005621, 0.0006462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005860, upper bound: 0.0005788
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005889
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905819, 0.9924610, 0.9906371, 0.9925014, -0.0009793, 0.0008705
1: -0.0036107, -0.0031425, -0.0035969, -0.0031324, -0.0002440, 0.0002169
2: 0.0065996, 0.0090807, 0.0065461, 0.0090079, -0.0011494, 0.0012932
3: -0.0054063, -0.0042770, -0.0053731, -0.0042526, -0.0005886, 0.0005232
4: 0.0018052, 0.0022854, 0.0017949, 0.0022713, -0.0002225, 0.0002503
5: 0.0072600, 0.0103806, 0.0071928, 0.0102890, -0.0014457, 0.0016265
6: -0.0010939, -0.0003018, -0.0010706, -0.0002848, -0.0004128, 0.0003669
7: -0.0059678, -0.0039186, -0.0059077, -0.0038744, -0.0010681, 0.0009494
8: -0.0027026, -0.0016249, -0.0026709, -0.0016017, -0.0005617, 0.0004993
9: 0.0000203, 0.0012699, -0.0000067, 0.0012332, -0.0005789, 0.0006513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005949, upper bound: 0.0005793
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005895, upper bound: 0.0005895
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905945, 0.9924596, 0.9906676, 0.9925198, -0.0010147, 0.0008629
1: -0.0036076, -0.0031428, -0.0035893, -0.0031278, -0.0002528, 0.0002150
2: 0.0066013, 0.0090641, 0.0065219, 0.0089675, -0.0011395, 0.0013399
3: -0.0053987, -0.0042778, -0.0053547, -0.0042416, -0.0006099, 0.0005187
4: 0.0018056, 0.0022822, 0.0017902, 0.0022635, -0.0002205, 0.0002593
5: 0.0072622, 0.0103597, 0.0071622, 0.0102382, -0.0014332, 0.0016853
6: -0.0010886, -0.0003024, -0.0010577, -0.0002770, -0.0004277, 0.0003638
7: -0.0059541, -0.0039200, -0.0058743, -0.0038543, -0.0011067, 0.0009412
8: -0.0026954, -0.0016256, -0.0026534, -0.0015911, -0.0005820, 0.0004949
9: 0.0000211, 0.0012615, -0.0000189, 0.0012129, -0.0005739, 0.0006749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005413
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005422
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905944, 0.9924608, 0.9906505, 0.9925325, -0.0010231, 0.0008860
1: -0.0036076, -0.0031425, -0.0035936, -0.0031247, -0.0002549, 0.0002208
2: 0.0065998, 0.0090643, 0.0065050, 0.0089902, -0.0011700, 0.0013510
3: -0.0053988, -0.0042770, -0.0053651, -0.0042339, -0.0006149, 0.0005325
4: 0.0018053, 0.0022823, 0.0017869, 0.0022679, -0.0002265, 0.0002615
5: 0.0072602, 0.0103599, 0.0071411, 0.0102667, -0.0014716, 0.0016992
6: -0.0010886, -0.0003019, -0.0010650, -0.0002716, -0.0004313, 0.0003735
7: -0.0059542, -0.0039187, -0.0058930, -0.0038405, -0.0011159, 0.0009663
8: -0.0026954, -0.0016249, -0.0026632, -0.0015838, -0.0005868, 0.0005082
9: 0.0000204, 0.0012616, -0.0000273, 0.0012243, -0.0005893, 0.0006804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006162, upper bound: 0.0005421
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006110, upper bound: 0.0005430
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905821, 0.9924597, 0.9906345, 0.9925388, -0.0010226, 0.0008729
1: -0.0036107, -0.0031428, -0.0035976, -0.0031231, -0.0002548, 0.0002175
2: 0.0066012, 0.0090806, 0.0064967, 0.0090114, -0.0011527, 0.0013504
3: -0.0054062, -0.0042777, -0.0053747, -0.0042302, -0.0006146, 0.0005247
4: 0.0018055, 0.0022854, 0.0017853, 0.0022720, -0.0002231, 0.0002614
5: 0.0072620, 0.0103804, 0.0071306, 0.0102934, -0.0014498, 0.0016984
6: -0.0010938, -0.0003023, -0.0010717, -0.0002690, -0.0004311, 0.0003680
7: -0.0059677, -0.0039199, -0.0059105, -0.0038336, -0.0011153, 0.0009521
8: -0.0027025, -0.0016256, -0.0026724, -0.0015802, -0.0005865, 0.0005007
9: 0.0000211, 0.0012698, -0.0000315, 0.0012350, -0.0005806, 0.0006801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005676
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005741
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905819, 0.9924610, 0.9906178, 0.9925500, -0.0010290, 0.0008979
1: -0.0036107, -0.0031425, -0.0036018, -0.0031203, -0.0002564, 0.0002237
2: 0.0065996, 0.0090807, 0.0064819, 0.0090334, -0.0011856, 0.0013587
3: -0.0054063, -0.0042770, -0.0053847, -0.0042234, -0.0006184, 0.0005396
4: 0.0018052, 0.0022854, 0.0017824, 0.0022763, -0.0002295, 0.0002630
5: 0.0072600, 0.0103806, 0.0071119, 0.0103211, -0.0014912, 0.0017089
6: -0.0010939, -0.0003018, -0.0010788, -0.0002643, -0.0004337, 0.0003785
7: -0.0059678, -0.0039186, -0.0059287, -0.0038213, -0.0011222, 0.0009793
8: -0.0027026, -0.0016249, -0.0026820, -0.0015737, -0.0005902, 0.0005150
9: 0.0000203, 0.0012699, -0.0000390, 0.0012461, -0.0005971, 0.0006843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006163, upper bound: 0.0005684
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006110, upper bound: 0.0005751
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925576, 0.9906489, 0.9925310, -0.0008714, 0.0009167
1: -0.0035902, -0.0031184, -0.0035940, -0.0031250, -0.0002171, 0.0002284
2: 0.0064720, 0.0089722, 0.0065069, 0.0089923, -0.0012104, 0.0011507
3: -0.0053569, -0.0042189, -0.0053660, -0.0042348, -0.0005238, 0.0005509
4: 0.0017805, 0.0022644, 0.0017873, 0.0022683, -0.0002343, 0.0002227
5: 0.0070995, 0.0102442, 0.0071435, 0.0102694, -0.0015224, 0.0014473
6: -0.0010592, -0.0002611, -0.0010657, -0.0002723, -0.0003673, 0.0003864
7: -0.0058782, -0.0038131, -0.0058948, -0.0038420, -0.0009504, 0.0009998
8: -0.0026554, -0.0015694, -0.0026642, -0.0015846, -0.0004998, 0.0005258
9: -0.0000440, 0.0012153, -0.0000264, 0.0012254, -0.0006096, 0.0005796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006332
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005627, upper bound: 0.0006332
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906314, 0.9925753, 0.9906367, 0.9925311, -0.0008830, 0.0009324
1: -0.0035984, -0.0031140, -0.0035970, -0.0031250, -0.0002200, 0.0002323
2: 0.0064485, 0.0090155, 0.0065069, 0.0090084, -0.0012313, 0.0011660
3: -0.0053766, -0.0042082, -0.0053733, -0.0042348, -0.0005307, 0.0005604
4: 0.0017760, 0.0022728, 0.0017873, 0.0022714, -0.0002383, 0.0002257
5: 0.0070699, 0.0102985, 0.0071434, 0.0102896, -0.0015486, 0.0014665
6: -0.0010730, -0.0002536, -0.0010708, -0.0002722, -0.0003722, 0.0003931
7: -0.0059139, -0.0037937, -0.0059081, -0.0038420, -0.0009631, 0.0010169
8: -0.0026742, -0.0015592, -0.0026711, -0.0015846, -0.0005065, 0.0005348
9: -0.0000558, 0.0012370, -0.0000264, 0.0012335, -0.0006201, 0.0005873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005948, upper bound: 0.0006333
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005991, upper bound: 0.0006332
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925576, 0.9905937, 0.9924691, -0.0008623, 0.0010357
1: -0.0035902, -0.0031184, -0.0036078, -0.0031404, -0.0002149, 0.0002581
2: 0.0064720, 0.0089722, 0.0065887, 0.0090652, -0.0013676, 0.0011387
3: -0.0053569, -0.0042189, -0.0053992, -0.0042720, -0.0005183, 0.0006225
4: 0.0017805, 0.0022644, 0.0018031, 0.0022824, -0.0002647, 0.0002204
5: 0.0070995, 0.0102442, 0.0072463, 0.0103611, -0.0017201, 0.0014322
6: -0.0010592, -0.0002611, -0.0010889, -0.0002984, -0.0003635, 0.0004366
7: -0.0058782, -0.0038131, -0.0059550, -0.0039096, -0.0009405, 0.0011296
8: -0.0026554, -0.0015694, -0.0026958, -0.0016202, -0.0004946, 0.0005940
9: -0.0000440, 0.0012153, 0.0000148, 0.0012621, -0.0006888, 0.0005735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005604, upper bound: 0.0006260
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005609, upper bound: 0.0006307
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906314, 0.9925753, 0.9905813, 0.9924693, -0.0008802, 0.0010490
1: -0.0035984, -0.0031140, -0.0036109, -0.0031404, -0.0002193, 0.0002614
2: 0.0064485, 0.0090155, 0.0065886, 0.0090816, -0.0013851, 0.0011623
3: -0.0053766, -0.0042082, -0.0054067, -0.0042719, -0.0005290, 0.0006305
4: 0.0017760, 0.0022728, 0.0018031, 0.0022856, -0.0002681, 0.0002250
5: 0.0070699, 0.0102985, 0.0072461, 0.0103818, -0.0017421, 0.0014618
6: -0.0010730, -0.0002536, -0.0010942, -0.0002983, -0.0003710, 0.0004422
7: -0.0059139, -0.0037937, -0.0059686, -0.0039095, -0.0009600, 0.0011440
8: -0.0026742, -0.0015592, -0.0027030, -0.0016201, -0.0005048, 0.0006016
9: -0.0000558, 0.0012370, 0.0000147, 0.0012704, -0.0006976, 0.0005854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005938, upper bound: 0.0006260
time: 0.64 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005941, upper bound: 0.0006307
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925576, 0.9906299, 0.9925753, -0.0008674, 0.0008928
1: -0.0035902, -0.0031184, -0.0035988, -0.0031140, -0.0002161, 0.0002225
2: 0.0064720, 0.0089722, 0.0064484, 0.0090175, -0.0011790, 0.0011454
3: -0.0053569, -0.0042189, -0.0053775, -0.0042082, -0.0005213, 0.0005366
4: 0.0017805, 0.0022644, 0.0017760, 0.0022732, -0.0002282, 0.0002217
5: 0.0070995, 0.0102442, 0.0070699, 0.0103011, -0.0014829, 0.0014406
6: -0.0010592, -0.0002611, -0.0010737, -0.0002536, -0.0003656, 0.0003764
7: -0.0058782, -0.0038131, -0.0059156, -0.0037937, -0.0009460, 0.0009738
8: -0.0026554, -0.0015694, -0.0026751, -0.0015592, -0.0004975, 0.0005121
9: -0.0000440, 0.0012153, -0.0000559, 0.0012381, -0.0005938, 0.0005769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006332
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005627, upper bound: 0.0006332
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906314, 0.9925753, 0.9906180, 0.9925754, -0.0008774, 0.0008980
1: -0.0035984, -0.0031140, -0.0036017, -0.0031140, -0.0002186, 0.0002238
2: 0.0064485, 0.0090155, 0.0064483, 0.0090331, -0.0011858, 0.0011587
3: -0.0053766, -0.0042082, -0.0053846, -0.0042081, -0.0005274, 0.0005397
4: 0.0017760, 0.0022728, 0.0017760, 0.0022762, -0.0002295, 0.0002243
5: 0.0070699, 0.0102985, 0.0070698, 0.0103207, -0.0014914, 0.0014573
6: -0.0010730, -0.0002536, -0.0010787, -0.0002536, -0.0003699, 0.0003785
7: -0.0059139, -0.0037937, -0.0059285, -0.0037936, -0.0009570, 0.0009794
8: -0.0026742, -0.0015592, -0.0026819, -0.0015592, -0.0005033, 0.0005151
9: -0.0000558, 0.0012370, -0.0000559, 0.0012459, -0.0005972, 0.0005836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005947, upper bound: 0.0006332
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005991, upper bound: 0.0006332
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906641, 0.9925576, 0.9905663, 0.9925196, -0.0008577, 0.0010112
1: -0.0035902, -0.0031184, -0.0036146, -0.0031279, -0.0002137, 0.0002520
2: 0.0064720, 0.0089722, 0.0065222, 0.0091014, -0.0013353, 0.0011326
3: -0.0053569, -0.0042189, -0.0054157, -0.0042417, -0.0005155, 0.0006078
4: 0.0017805, 0.0022644, 0.0017902, 0.0022894, -0.0002584, 0.0002192
5: 0.0070995, 0.0102442, 0.0071626, 0.0104066, -0.0016795, 0.0014246
6: -0.0010592, -0.0002611, -0.0011005, -0.0002771, -0.0003616, 0.0004263
7: -0.0058782, -0.0038131, -0.0059849, -0.0038546, -0.0009355, 0.0011029
8: -0.0026554, -0.0015694, -0.0027115, -0.0015912, -0.0004920, 0.0005800
9: -0.0000440, 0.0012153, -0.0000187, 0.0012803, -0.0006725, 0.0005705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005604, upper bound: 0.0006260
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005609, upper bound: 0.0006307
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906314, 0.9925753, 0.9905547, 0.9925196, -0.0008738, 0.0010143
1: -0.0035984, -0.0031140, -0.0036175, -0.0031279, -0.0002177, 0.0002527
2: 0.0064485, 0.0090155, 0.0065220, 0.0091167, -0.0013393, 0.0011539
3: -0.0053766, -0.0042082, -0.0054226, -0.0042416, -0.0005252, 0.0006096
4: 0.0017760, 0.0022728, 0.0017902, 0.0022924, -0.0002592, 0.0002233
5: 0.0070699, 0.0102985, 0.0071624, 0.0104258, -0.0016845, 0.0014513
6: -0.0010730, -0.0002536, -0.0011054, -0.0002771, -0.0003684, 0.0004276
7: -0.0059139, -0.0037937, -0.0059975, -0.0038545, -0.0009531, 0.0011062
8: -0.0026742, -0.0015592, -0.0027182, -0.0015912, -0.0005012, 0.0005817
9: -0.0000558, 0.0012370, -0.0000188, 0.0012880, -0.0006746, 0.0005812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005938, upper bound: 0.0006260
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0006307
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906017, 0.9924911, 0.9906522, 0.9924914, -0.0009619, 0.0009065
1: -0.0036058, -0.0031350, -0.0035932, -0.0031349, -0.0002397, 0.0002259
2: 0.0065598, 0.0090547, 0.0065594, 0.0089879, -0.0011970, 0.0012702
3: -0.0053944, -0.0042588, -0.0053640, -0.0042587, -0.0005782, 0.0005448
4: 0.0017975, 0.0022804, 0.0017974, 0.0022675, -0.0002317, 0.0002458
5: 0.0072099, 0.0103479, 0.0072094, 0.0102639, -0.0015056, 0.0015976
6: -0.0010856, -0.0002891, -0.0010643, -0.0002890, -0.0004055, 0.0003821
7: -0.0059463, -0.0038857, -0.0058912, -0.0038853, -0.0010491, 0.0009887
8: -0.0026913, -0.0016076, -0.0026623, -0.0016074, -0.0005517, 0.0005199
9: 0.0000002, 0.0012568, 0.0000000, 0.0012232, -0.0006029, 0.0006398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005348, upper bound: 0.0005995
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005307, upper bound: 0.0006100
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906015, 0.9924933, 0.9906348, 0.9925014, -0.0009697, 0.0009329
1: -0.0036058, -0.0031344, -0.0035975, -0.0031324, -0.0002416, 0.0002325
2: 0.0065569, 0.0090549, 0.0065461, 0.0090110, -0.0012319, 0.0012805
3: -0.0053945, -0.0042575, -0.0053745, -0.0042526, -0.0005828, 0.0005607
4: 0.0017970, 0.0022804, 0.0017949, 0.0022719, -0.0002384, 0.0002478
5: 0.0072063, 0.0103482, 0.0071927, 0.0102929, -0.0015494, 0.0016105
6: -0.0010856, -0.0002882, -0.0010716, -0.0002848, -0.0004088, 0.0003933
7: -0.0059465, -0.0038833, -0.0059102, -0.0038744, -0.0010576, 0.0010175
8: -0.0026914, -0.0016063, -0.0026723, -0.0016016, -0.0005562, 0.0005351
9: -0.0000012, 0.0012569, -0.0000067, 0.0012348, -0.0006204, 0.0006449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005441, upper bound: 0.0006001
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0006110
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9905691, 0.9925099, 0.9906399, 0.9924914, -0.0009700, 0.0009244
1: -0.0036139, -0.0031303, -0.0035962, -0.0031349, -0.0002417, 0.0002303
2: 0.0065349, 0.0090978, 0.0065593, 0.0090041, -0.0012206, 0.0012808
3: -0.0054140, -0.0042475, -0.0053714, -0.0042586, -0.0005830, 0.0005556
4: 0.0017927, 0.0022887, 0.0017974, 0.0022706, -0.0002363, 0.0002479
5: 0.0071786, 0.0104021, 0.0072093, 0.0102843, -0.0015352, 0.0016109
6: -0.0010993, -0.0002812, -0.0010694, -0.0002890, -0.0004089, 0.0003897
7: -0.0059819, -0.0038651, -0.0059046, -0.0038853, -0.0010579, 0.0010082
8: -0.0027100, -0.0015968, -0.0026693, -0.0016074, -0.0005563, 0.0005302
9: -0.0000123, 0.0012785, -0.0000000, 0.0012313, -0.0006148, 0.0006451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9905689, 0.9925113, 0.9906232, 0.9925015, -0.0009803, 0.0009485
1: -0.0036139, -0.0031299, -0.0036004, -0.0031324, -0.0002443, 0.0002363
2: 0.0065330, 0.0090979, 0.0065460, 0.0090262, -0.0012525, 0.0012944
3: -0.0054141, -0.0042467, -0.0053815, -0.0042526, -0.0005892, 0.0005701
4: 0.0017923, 0.0022888, 0.0017948, 0.0022749, -0.0002424, 0.0002505
5: 0.0071763, 0.0104022, 0.0071925, 0.0103120, -0.0015753, 0.0016280
6: -0.0010994, -0.0002806, -0.0010765, -0.0002847, -0.0004132, 0.0003998
7: -0.0059820, -0.0038636, -0.0059228, -0.0038743, -0.0010691, 0.0010345
8: -0.0027100, -0.0015960, -0.0026789, -0.0016016, -0.0005622, 0.0005440
9: -0.0000133, 0.0012786, -0.0000067, 0.0012424, -0.0006308, 0.0006519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006110
time: 0.62 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905556, 0.9925100, 0.9906345, 0.9925388, -0.0009930, 0.0008736
1: -0.0036172, -0.0031302, -0.0035976, -0.0031231, -0.0002474, 0.0002177
2: 0.0065347, 0.0091155, 0.0064967, 0.0090114, -0.0011536, 0.0013112
3: -0.0054221, -0.0042474, -0.0053747, -0.0042302, -0.0005968, 0.0005251
4: 0.0017927, 0.0022922, 0.0017853, 0.0022720, -0.0002233, 0.0002538
5: 0.0071784, 0.0104244, 0.0071306, 0.0102934, -0.0014509, 0.0016492
6: -0.0011050, -0.0002811, -0.0010717, -0.0002690, -0.0004186, 0.0003683
7: -0.0059966, -0.0038649, -0.0059105, -0.0038336, -0.0010830, 0.0009528
8: -0.0027177, -0.0015967, -0.0026724, -0.0015802, -0.0005695, 0.0005011
9: -0.0000124, 0.0012874, -0.0000315, 0.0012350, -0.0005810, 0.0006604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
time: 0.67 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905555, 0.9925114, 0.9906178, 0.9925500, -0.0010016, 0.0008987
1: -0.0036173, -0.0031299, -0.0036018, -0.0031203, -0.0002496, 0.0002239
2: 0.0065328, 0.0091157, 0.0064819, 0.0090334, -0.0011867, 0.0013225
3: -0.0054222, -0.0042466, -0.0053847, -0.0042234, -0.0006020, 0.0005401
4: 0.0017923, 0.0022922, 0.0017824, 0.0022763, -0.0002297, 0.0002560
5: 0.0071760, 0.0104245, 0.0071119, 0.0103211, -0.0014926, 0.0016634
6: -0.0011050, -0.0002805, -0.0010788, -0.0002643, -0.0004222, 0.0003788
7: -0.0059967, -0.0038634, -0.0059287, -0.0038213, -0.0010923, 0.0009802
8: -0.0027177, -0.0015959, -0.0026820, -0.0015737, -0.0005745, 0.0005155
9: -0.0000134, 0.0012875, -0.0000390, 0.0012461, -0.0005977, 0.0006661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006110
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.76 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005619, upper bound: 0.0006155
IS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005744, upper bound: 0.0006155
IS_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006114, upper bound: 0.0006155
IS_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006155, upper bound: 0.0006155
IS_A1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005547
IS_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005627
IS_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005948
IS_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006332, upper bound: 0.0005992
IS_A1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005688, upper bound: 0.0006096
IS_A1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005690, upper bound: 0.0006137
IS_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006089, upper bound: 0.0006096
IS_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006091, upper bound: 0.0006137
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006298, upper bound: 0.0005534
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005617
IS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006299, upper bound: 0.0005933
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0005982
IS_A1_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005860, upper bound: 0.0005507
IS_A1_A2_B1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005794, upper bound: 0.0005511
IS_A1_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005949, upper bound: 0.0005511
IS_A1_A2_B1_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005895, upper bound: 0.0005516
IS_A1_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005860, upper bound: 0.0005788
IS_A1_A2_B1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005889
IS_A1_A2_B1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005949, upper bound: 0.0005793
IS_A1_A2_B1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005895, upper bound: 0.0005895
IS_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005413
IS_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005422
IS_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006162, upper bound: 0.0005421
IS_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006110, upper bound: 0.0005430
IS_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005676
IS_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005741
IS_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006163, upper bound: 0.0005684
IS_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0006110, upper bound: 0.0005751
IS_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006332
IS_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005627, upper bound: 0.0006332
IS_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005948, upper bound: 0.0006333
IS_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005991, upper bound: 0.0006332
IS_A2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005604, upper bound: 0.0006260
IS_A2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005609, upper bound: 0.0006307
IS_A2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005938, upper bound: 0.0006260
IS_A2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005941, upper bound: 0.0006307
IS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005547, upper bound: 0.0006332
IS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005627, upper bound: 0.0006332
IS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005947, upper bound: 0.0006332
IS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005991, upper bound: 0.0006332
IS_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005604, upper bound: 0.0006260
IS_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005609, upper bound: 0.0006307
IS_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005938, upper bound: 0.0006260
IS_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005942, upper bound: 0.0006307
IS_A2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005348, upper bound: 0.0005995
IS_A2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005307, upper bound: 0.0006100
IS_A2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005441, upper bound: 0.0006001
IS_A2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0006110
IS_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
IS_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
IS_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
IS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006110
IS_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
IS_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
IS_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
IS_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.76
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006110

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906837, 0.9925045, 0.9906523, 0.9924982, -0.0008140, 0.0008587
1: -0.0035853, -0.0031316, -0.0035931, -0.0031332, -0.0002028, 0.0002140
2: 0.0065421, 0.0089463, 0.0065502, 0.0089878, -0.0011339, 0.0010749
3: -0.0053451, -0.0042508, -0.0053640, -0.0042545, -0.0004892, 0.0005161
4: 0.0017941, 0.0022594, 0.0017957, 0.0022675, -0.0002195, 0.0002080
5: 0.0071876, 0.0102115, 0.0071979, 0.0102637, -0.0014261, 0.0013519
6: -0.0010510, -0.0002835, -0.0010642, -0.0002861, -0.0003431, 0.0003620
7: -0.0058568, -0.0038710, -0.0058910, -0.0038778, -0.0008878, 0.0009365
8: -0.0026442, -0.0015999, -0.0026622, -0.0016034, -0.0004669, 0.0004925
9: -0.0000087, 0.0012022, -0.0000046, 0.0012231, -0.0005711, 0.0005414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005492, upper bound: 0.0005861
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005454, upper bound: 0.0005968
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906834, 0.9925053, 0.9906349, 0.9925060, -0.0008185, 0.0008858
1: -0.0035854, -0.0031314, -0.0035975, -0.0031313, -0.0002039, 0.0002207
2: 0.0065410, 0.0089466, 0.0065401, 0.0090108, -0.0011697, 0.0010808
3: -0.0053452, -0.0042503, -0.0053745, -0.0042499, -0.0004919, 0.0005324
4: 0.0017939, 0.0022595, 0.0017937, 0.0022719, -0.0002264, 0.0002092
5: 0.0071863, 0.0102119, 0.0071851, 0.0102927, -0.0014711, 0.0013593
6: -0.0010511, -0.0002831, -0.0010716, -0.0002828, -0.0003450, 0.0003734
7: -0.0058571, -0.0038702, -0.0059101, -0.0038694, -0.0008926, 0.0009661
8: -0.0026443, -0.0015994, -0.0026722, -0.0015990, -0.0004694, 0.0005081
9: -0.0000092, 0.0012024, -0.0000097, 0.0012347, -0.0005891, 0.0005443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005594, upper bound: 0.0005862
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005968
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906507, 0.9925218, 0.9906400, 0.9924983, -0.0008214, 0.0008657
1: -0.0035935, -0.0031273, -0.0035962, -0.0031332, -0.0002047, 0.0002157
2: 0.0065193, 0.0089899, 0.0065502, 0.0090040, -0.0011432, 0.0010847
3: -0.0053649, -0.0042404, -0.0053713, -0.0042545, -0.0004937, 0.0005203
4: 0.0017897, 0.0022679, 0.0017957, 0.0022706, -0.0002213, 0.0002099
5: 0.0071590, 0.0102664, 0.0071978, 0.0102841, -0.0014378, 0.0013642
6: -0.0010649, -0.0002762, -0.0010694, -0.0002861, -0.0003463, 0.0003649
7: -0.0058928, -0.0038523, -0.0059044, -0.0038777, -0.0008959, 0.0009442
8: -0.0026631, -0.0015900, -0.0026692, -0.0016034, -0.0004711, 0.0004965
9: -0.0000202, 0.0012242, -0.0000046, 0.0012313, -0.0005758, 0.0005463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005994, upper bound: 0.0005861
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005925, upper bound: 0.0005968
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906505, 0.9925224, 0.9906232, 0.9925060, -0.0008268, 0.0008882
1: -0.0035936, -0.0031272, -0.0036004, -0.0031312, -0.0002060, 0.0002213
2: 0.0065185, 0.0089902, 0.0065400, 0.0090261, -0.0011728, 0.0010918
3: -0.0053651, -0.0042401, -0.0053814, -0.0042498, -0.0004970, 0.0005338
4: 0.0017895, 0.0022679, 0.0017937, 0.0022749, -0.0002270, 0.0002113
5: 0.0071580, 0.0102668, 0.0071850, 0.0103119, -0.0014751, 0.0013732
6: -0.0010650, -0.0002760, -0.0010765, -0.0002828, -0.0003485, 0.0003744
7: -0.0058931, -0.0038516, -0.0059227, -0.0038693, -0.0009018, 0.0009687
8: -0.0026632, -0.0015897, -0.0026789, -0.0015990, -0.0004742, 0.0005094
9: -0.0000206, 0.0012243, -0.0000098, 0.0012424, -0.0005907, 0.0005499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006028, upper bound: 0.0005861
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005968, upper bound: 0.0005968
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906523, 0.9924982, 0.9906651, 0.9925478, -0.0009048, 0.0008403
1: -0.0035931, -0.0031332, -0.0035900, -0.0031208, -0.0002254, 0.0002094
2: 0.0065502, 0.0089878, 0.0064848, 0.0089709, -0.0011096, 0.0011947
3: -0.0053640, -0.0042545, -0.0053563, -0.0042247, -0.0005438, 0.0005050
4: 0.0017957, 0.0022675, 0.0017830, 0.0022642, -0.0002148, 0.0002312
5: 0.0071979, 0.0102637, 0.0071156, 0.0102424, -0.0013956, 0.0015027
6: -0.0010642, -0.0002861, -0.0010588, -0.0002652, -0.0003814, 0.0003542
7: -0.0058910, -0.0038778, -0.0058771, -0.0038237, -0.0009868, 0.0009164
8: -0.0026622, -0.0016034, -0.0026548, -0.0015750, -0.0005189, 0.0004819
9: -0.0000046, 0.0012231, -0.0000375, 0.0012146, -0.0005588, 0.0006017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005415
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005386
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906349, 0.9925060, 0.9906650, 0.9925492, -0.0009317, 0.0008447
1: -0.0035975, -0.0031313, -0.0035900, -0.0031205, -0.0002322, 0.0002105
2: 0.0065401, 0.0090108, 0.0064831, 0.0089712, -0.0011155, 0.0012303
3: -0.0053745, -0.0042499, -0.0053564, -0.0042239, -0.0005600, 0.0005077
4: 0.0017937, 0.0022719, 0.0017827, 0.0022642, -0.0002159, 0.0002381
5: 0.0071851, 0.0102927, 0.0071134, 0.0102428, -0.0014030, 0.0015474
6: -0.0010716, -0.0002828, -0.0010589, -0.0002646, -0.0003928, 0.0003561
7: -0.0059101, -0.0038694, -0.0058773, -0.0038223, -0.0010162, 0.0009213
8: -0.0026722, -0.0015990, -0.0026550, -0.0015743, -0.0005344, 0.0004845
9: -0.0000097, 0.0012347, -0.0000384, 0.0012147, -0.0005618, 0.0006197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005492
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005470
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906400, 0.9924983, 0.9906324, 0.9925655, -0.0009211, 0.0008496
1: -0.0035962, -0.0031332, -0.0035981, -0.0031164, -0.0002295, 0.0002117
2: 0.0065502, 0.0090040, 0.0064615, 0.0090142, -0.0011219, 0.0012163
3: -0.0053713, -0.0042545, -0.0053760, -0.0042141, -0.0005536, 0.0005107
4: 0.0017957, 0.0022706, 0.0017785, 0.0022726, -0.0002172, 0.0002354
5: 0.0071978, 0.0102841, 0.0070863, 0.0102970, -0.0014111, 0.0015298
6: -0.0010694, -0.0002861, -0.0010726, -0.0002578, -0.0003883, 0.0003582
7: -0.0059044, -0.0038777, -0.0059129, -0.0038045, -0.0010046, 0.0009267
8: -0.0026692, -0.0016034, -0.0026737, -0.0015649, -0.0005283, 0.0004873
9: -0.0000046, 0.0012313, -0.0000493, 0.0012364, -0.0005651, 0.0006126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005836
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005766
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906232, 0.9925060, 0.9906321, 0.9925671, -0.0009460, 0.0008551
1: -0.0036004, -0.0031312, -0.0035982, -0.0031160, -0.0002357, 0.0002131
2: 0.0065400, 0.0090261, 0.0064594, 0.0090145, -0.0011291, 0.0012491
3: -0.0053814, -0.0042498, -0.0053761, -0.0042132, -0.0005686, 0.0005139
4: 0.0017937, 0.0022749, 0.0017781, 0.0022726, -0.0002185, 0.0002418
5: 0.0071850, 0.0103119, 0.0070837, 0.0102973, -0.0014202, 0.0015711
6: -0.0010765, -0.0002828, -0.0010727, -0.0002571, -0.0003988, 0.0003605
7: -0.0059227, -0.0038693, -0.0059131, -0.0038028, -0.0010317, 0.0009326
8: -0.0026789, -0.0015990, -0.0026738, -0.0015640, -0.0005426, 0.0004904
9: -0.0000098, 0.0012424, -0.0000503, 0.0012365, -0.0005687, 0.0006291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005878
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005813
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906865, 0.9924803, 0.9905945, 0.9924596, -0.0008234, 0.0009545
1: -0.0035846, -0.0031377, -0.0036076, -0.0031428, -0.0002052, 0.0002378
2: 0.0065740, 0.0089426, 0.0066013, 0.0090641, -0.0012604, 0.0010873
3: -0.0053434, -0.0042653, -0.0053987, -0.0042778, -0.0004949, 0.0005737
4: 0.0018003, 0.0022587, 0.0018056, 0.0022822, -0.0002439, 0.0002104
5: 0.0072278, 0.0102069, 0.0072622, 0.0103597, -0.0015852, 0.0013675
6: -0.0010498, -0.0002937, -0.0010886, -0.0003024, -0.0003471, 0.0004023
7: -0.0058538, -0.0038974, -0.0059541, -0.0039200, -0.0008980, 0.0010410
8: -0.0026426, -0.0016137, -0.0026954, -0.0016256, -0.0004723, 0.0005475
9: 0.0000074, 0.0012004, 0.0000211, 0.0012615, -0.0006348, 0.0005476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005506, upper bound: 0.0005952
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005511, upper bound: 0.0005893
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906694, 0.9924889, 0.9905944, 0.9924608, -0.0008530, 0.0009610
1: -0.0035889, -0.0031355, -0.0036076, -0.0031425, -0.0002126, 0.0002395
2: 0.0065626, 0.0089652, 0.0065998, 0.0090643, -0.0012690, 0.0011264
3: -0.0053537, -0.0042601, -0.0053988, -0.0042770, -0.0005127, 0.0005776
4: 0.0017981, 0.0022631, 0.0018053, 0.0022823, -0.0002456, 0.0002180
5: 0.0072134, 0.0102353, 0.0072602, 0.0103599, -0.0015961, 0.0014168
6: -0.0010570, -0.0002900, -0.0010886, -0.0003019, -0.0003596, 0.0004051
7: -0.0058724, -0.0038880, -0.0059542, -0.0039187, -0.0009304, 0.0010481
8: -0.0026524, -0.0016088, -0.0026954, -0.0016249, -0.0004893, 0.0005512
9: 0.0000016, 0.0012117, 0.0000204, 0.0012616, -0.0006391, 0.0005673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0005997
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005516, upper bound: 0.0005945
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906530, 0.9924982, 0.9905821, 0.9924597, -0.0008391, 0.0009612
1: -0.0035930, -0.0031332, -0.0036107, -0.0031428, -0.0002091, 0.0002395
2: 0.0065503, 0.0089868, 0.0066012, 0.0090806, -0.0012693, 0.0011080
3: -0.0053635, -0.0042545, -0.0054062, -0.0042777, -0.0005043, 0.0005777
4: 0.0017957, 0.0022673, 0.0018055, 0.0022854, -0.0002457, 0.0002145
5: 0.0071980, 0.0102625, 0.0072620, 0.0103804, -0.0015964, 0.0013936
6: -0.0010639, -0.0002861, -0.0010938, -0.0003023, -0.0003537, 0.0004052
7: -0.0058902, -0.0038778, -0.0059677, -0.0039199, -0.0009152, 0.0010483
8: -0.0026618, -0.0016035, -0.0027025, -0.0016256, -0.0004813, 0.0005513
9: -0.0000046, 0.0012226, 0.0000211, 0.0012698, -0.0006393, 0.0005581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005788, upper bound: 0.0005952
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005889, upper bound: 0.0005894
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906371, 0.9925060, 0.9905819, 0.9924610, -0.0008677, 0.0009656
1: -0.0035969, -0.0031313, -0.0036107, -0.0031425, -0.0002162, 0.0002406
2: 0.0065401, 0.0090078, 0.0065996, 0.0090807, -0.0012751, 0.0011458
3: -0.0053731, -0.0042499, -0.0054063, -0.0042770, -0.0005215, 0.0005804
4: 0.0017937, 0.0022713, 0.0018052, 0.0022854, -0.0002468, 0.0002218
5: 0.0071852, 0.0102889, 0.0072600, 0.0103806, -0.0016037, 0.0014412
6: -0.0010706, -0.0002829, -0.0010939, -0.0003018, -0.0003658, 0.0004070
7: -0.0059076, -0.0038694, -0.0059678, -0.0039186, -0.0009464, 0.0010531
8: -0.0026709, -0.0015990, -0.0027026, -0.0016249, -0.0004977, 0.0005538
9: -0.0000097, 0.0012332, 0.0000203, 0.0012699, -0.0006422, 0.0005771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005997
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005895, upper bound: 0.0005945
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9906523, 0.9924982, 0.9906017, 0.9924911, -0.0008944, 0.0009515
1: -0.0035931, -0.0031332, -0.0036058, -0.0031350, -0.0002228, 0.0002371
2: 0.0065502, 0.0089878, 0.0065598, 0.0090547, -0.0012565, 0.0011810
3: -0.0053640, -0.0042545, -0.0053944, -0.0042588, -0.0005375, 0.0005719
4: 0.0017957, 0.0022675, 0.0017975, 0.0022804, -0.0002432, 0.0002286
5: 0.0071979, 0.0102637, 0.0072099, 0.0103479, -0.0015803, 0.0014854
6: -0.0010642, -0.0002861, -0.0010856, -0.0002891, -0.0003770, 0.0004011
7: -0.0058910, -0.0038778, -0.0059463, -0.0038857, -0.0009754, 0.0010378
8: -0.0026622, -0.0016034, -0.0026913, -0.0016076, -0.0005130, 0.0005458
9: -0.0000046, 0.0012231, 0.0000002, 0.0012568, -0.0006328, 0.0005948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005995, upper bound: 0.0005370
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006100, upper bound: 0.0005343
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906349, 0.9925060, 0.9906015, 0.9924933, -0.0009227, 0.0009560
1: -0.0035975, -0.0031313, -0.0036058, -0.0031344, -0.0002299, 0.0002382
2: 0.0065401, 0.0090108, 0.0065569, 0.0090549, -0.0012624, 0.0012184
3: -0.0053745, -0.0042499, -0.0053945, -0.0042575, -0.0005546, 0.0005746
4: 0.0017937, 0.0022719, 0.0017970, 0.0022804, -0.0002443, 0.0002358
5: 0.0071851, 0.0102927, 0.0072063, 0.0103482, -0.0015877, 0.0015325
6: -0.0010716, -0.0002828, -0.0010856, -0.0002882, -0.0003890, 0.0004030
7: -0.0059101, -0.0038694, -0.0059465, -0.0038833, -0.0010064, 0.0010426
8: -0.0026722, -0.0015990, -0.0026914, -0.0016063, -0.0005292, 0.0005483
9: -0.0000097, 0.0012347, -0.0000012, 0.0012569, -0.0006358, 0.0006137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0005460
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006110, upper bound: 0.0005443
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9906400, 0.9924983, 0.9905691, 0.9925099, -0.0008962, 0.0009587
1: -0.0035962, -0.0031332, -0.0036139, -0.0031303, -0.0002233, 0.0002389
2: 0.0065502, 0.0090040, 0.0065349, 0.0090978, -0.0012660, 0.0011834
3: -0.0053713, -0.0042545, -0.0054140, -0.0042475, -0.0005386, 0.0005762
4: 0.0017957, 0.0022706, 0.0017927, 0.0022887, -0.0002450, 0.0002290
5: 0.0071978, 0.0102841, 0.0071786, 0.0104021, -0.0015923, 0.0014884
6: -0.0010694, -0.0002861, -0.0010993, -0.0002812, -0.0003778, 0.0004041
7: -0.0059044, -0.0038777, -0.0059819, -0.0038651, -0.0009774, 0.0010456
8: -0.0026692, -0.0016034, -0.0027100, -0.0015968, -0.0005140, 0.0005499
9: -0.0000046, 0.0012313, -0.0000123, 0.0012785, -0.0006376, 0.0005960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005994, upper bound: 0.0005789
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006100, upper bound: 0.0005726
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906232, 0.9925060, 0.9905689, 0.9925113, -0.0009215, 0.0009642
1: -0.0036004, -0.0031312, -0.0036139, -0.0031299, -0.0002296, 0.0002402
2: 0.0065400, 0.0090261, 0.0065330, 0.0090979, -0.0012732, 0.0012168
3: -0.0053814, -0.0042498, -0.0054141, -0.0042467, -0.0005539, 0.0005795
4: 0.0017937, 0.0022749, 0.0017923, 0.0022888, -0.0002464, 0.0002355
5: 0.0071850, 0.0103119, 0.0071763, 0.0104022, -0.0016013, 0.0015305
6: -0.0010765, -0.0002828, -0.0010994, -0.0002806, -0.0003884, 0.0004064
7: -0.0059227, -0.0038693, -0.0059820, -0.0038636, -0.0010050, 0.0010515
8: -0.0026789, -0.0015990, -0.0027100, -0.0015960, -0.0005285, 0.0005530
9: -0.0000098, 0.0012424, -0.0000133, 0.0012786, -0.0006412, 0.0006129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0005850
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006109, upper bound: 0.0005791
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905959, 0.9924515, 0.9906613, 0.9924908, -0.0009758, 0.0008350
1: -0.0036072, -0.0031448, -0.0035909, -0.0031351, -0.0002431, 0.0002081
2: 0.0066120, 0.0090624, 0.0065601, 0.0089760, -0.0011026, 0.0012885
3: -0.0053979, -0.0042826, -0.0053586, -0.0042590, -0.0005865, 0.0005018
4: 0.0018076, 0.0022819, 0.0017976, 0.0022652, -0.0002134, 0.0002494
5: 0.0072756, 0.0103575, 0.0072104, 0.0102489, -0.0013868, 0.0016206
6: -0.0010880, -0.0003058, -0.0010605, -0.0002892, -0.0004113, 0.0003520
7: -0.0059527, -0.0039288, -0.0058813, -0.0038860, -0.0010642, 0.0009107
8: -0.0026946, -0.0016303, -0.0026571, -0.0016077, -0.0005597, 0.0004789
9: 0.0000265, 0.0012607, 0.0000004, 0.0012172, -0.0005553, 0.0006489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005413
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005413
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905951, 0.9924505, 0.9906704, 0.9924881, -0.0009758, 0.0008564
1: -0.0036074, -0.0031451, -0.0035886, -0.0031357, -0.0002432, 0.0002134
2: 0.0066133, 0.0090634, 0.0065637, 0.0089639, -0.0011309, 0.0012886
3: -0.0053984, -0.0042832, -0.0053531, -0.0042606, -0.0005865, 0.0005147
4: 0.0018079, 0.0022821, 0.0017983, 0.0022628, -0.0002189, 0.0002494
5: 0.0072772, 0.0103588, 0.0072148, 0.0102337, -0.0014224, 0.0016207
6: -0.0010883, -0.0003062, -0.0010566, -0.0002904, -0.0004113, 0.0003610
7: -0.0059535, -0.0039299, -0.0058713, -0.0038889, -0.0010643, 0.0009341
8: -0.0026950, -0.0016308, -0.0026518, -0.0016093, -0.0005597, 0.0004912
9: 0.0000272, 0.0012612, 0.0000022, 0.0012111, -0.0005696, 0.0006490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005422
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005422
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905958, 0.9924526, 0.9906446, 0.9925026, -0.0009854, 0.0008620
1: -0.0036072, -0.0031446, -0.0035951, -0.0031321, -0.0002455, 0.0002148
2: 0.0066106, 0.0090625, 0.0065446, 0.0089979, -0.0011383, 0.0013012
3: -0.0053980, -0.0042820, -0.0053686, -0.0042519, -0.0005922, 0.0005181
4: 0.0018074, 0.0022819, 0.0017946, 0.0022694, -0.0002203, 0.0002518
5: 0.0072739, 0.0103577, 0.0071909, 0.0102765, -0.0014316, 0.0016365
6: -0.0010881, -0.0003054, -0.0010675, -0.0002843, -0.0004154, 0.0003634
7: -0.0059528, -0.0039277, -0.0058995, -0.0038732, -0.0010747, 0.0009401
8: -0.0026946, -0.0016297, -0.0026666, -0.0016010, -0.0005652, 0.0004944
9: 0.0000258, 0.0012607, -0.0000074, 0.0012282, -0.0005733, 0.0006553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005421
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005421
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905950, 0.9924518, 0.9906533, 0.9925009, -0.0009842, 0.0008798
1: -0.0036074, -0.0031448, -0.0035929, -0.0031325, -0.0002452, 0.0002192
2: 0.0066115, 0.0090635, 0.0065467, 0.0089865, -0.0011618, 0.0012996
3: -0.0053984, -0.0042824, -0.0053634, -0.0042529, -0.0005915, 0.0005288
4: 0.0018075, 0.0022821, 0.0017950, 0.0022672, -0.0002249, 0.0002515
5: 0.0072750, 0.0103589, 0.0071935, 0.0102621, -0.0014613, 0.0016346
6: -0.0010884, -0.0003056, -0.0010638, -0.0002849, -0.0004149, 0.0003709
7: -0.0059536, -0.0039284, -0.0058900, -0.0038749, -0.0010734, 0.0009596
8: -0.0026951, -0.0016301, -0.0026616, -0.0016019, -0.0005645, 0.0005046
9: 0.0000263, 0.0012612, -0.0000064, 0.0012224, -0.0005852, 0.0006546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005430
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005430
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905834, 0.9924517, 0.9906258, 0.9925107, -0.0009868, 0.0008466
1: -0.0036103, -0.0031448, -0.0035998, -0.0031301, -0.0002459, 0.0002109
2: 0.0066118, 0.0090788, 0.0065338, 0.0090228, -0.0011179, 0.0013030
3: -0.0054054, -0.0042825, -0.0053799, -0.0042470, -0.0005931, 0.0005088
4: 0.0018076, 0.0022851, 0.0017925, 0.0022742, -0.0002164, 0.0002522
5: 0.0072754, 0.0103782, 0.0071772, 0.0103078, -0.0014060, 0.0016389
6: -0.0010933, -0.0003057, -0.0010754, -0.0002808, -0.0004160, 0.0003569
7: -0.0059663, -0.0039287, -0.0059200, -0.0038642, -0.0010762, 0.0009233
8: -0.0027017, -0.0016302, -0.0026774, -0.0015963, -0.0005660, 0.0004855
9: 0.0000264, 0.0012690, -0.0000129, 0.0012407, -0.0005630, 0.0006563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005676
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005676
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905826, 0.9924507, 0.9906366, 0.9925062, -0.0009856, 0.0008671
1: -0.0036105, -0.0031451, -0.0035971, -0.0031312, -0.0002456, 0.0002161
2: 0.0066131, 0.0090798, 0.0065397, 0.0090085, -0.0011450, 0.0013015
3: -0.0054059, -0.0042831, -0.0053734, -0.0042497, -0.0005924, 0.0005211
4: 0.0018078, 0.0022853, 0.0017936, 0.0022715, -0.0002216, 0.0002519
5: 0.0072770, 0.0103794, 0.0071847, 0.0102898, -0.0014401, 0.0016369
6: -0.0010936, -0.0003062, -0.0010708, -0.0002827, -0.0004155, 0.0003655
7: -0.0059671, -0.0039297, -0.0059082, -0.0038691, -0.0010749, 0.0009457
8: -0.0027022, -0.0016308, -0.0026712, -0.0015989, -0.0005653, 0.0004973
9: 0.0000271, 0.0012694, -0.0000099, 0.0012335, -0.0005767, 0.0006555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005741
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005741
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905832, 0.9924527, 0.9906107, 0.9925209, -0.0009937, 0.0008738
1: -0.0036104, -0.0031446, -0.0036035, -0.0031275, -0.0002476, 0.0002177
2: 0.0066105, 0.0090790, 0.0065204, 0.0090428, -0.0011538, 0.0013122
3: -0.0054055, -0.0042819, -0.0053890, -0.0042409, -0.0005972, 0.0005252
4: 0.0018073, 0.0022851, 0.0017899, 0.0022781, -0.0002233, 0.0002540
5: 0.0072737, 0.0103784, 0.0071603, 0.0103329, -0.0014512, 0.0016504
6: -0.0010933, -0.0003053, -0.0010818, -0.0002765, -0.0004189, 0.0003683
7: -0.0059664, -0.0039275, -0.0059365, -0.0038531, -0.0010838, 0.0009530
8: -0.0027018, -0.0016296, -0.0026861, -0.0015905, -0.0005699, 0.0005012
9: 0.0000258, 0.0012690, -0.0000196, 0.0012508, -0.0005811, 0.0006609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005684
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005685
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905825, 0.9924520, 0.9906201, 0.9925179, -0.0009920, 0.0008922
1: -0.0036105, -0.0031447, -0.0036012, -0.0031283, -0.0002472, 0.0002223
2: 0.0066114, 0.0090799, 0.0065243, 0.0090302, -0.0011781, 0.0013099
3: -0.0054059, -0.0042823, -0.0053833, -0.0042427, -0.0005962, 0.0005362
4: 0.0018075, 0.0022853, 0.0017907, 0.0022757, -0.0002280, 0.0002535
5: 0.0072748, 0.0103796, 0.0071654, 0.0103171, -0.0014818, 0.0016475
6: -0.0010936, -0.0003056, -0.0010778, -0.0002778, -0.0004181, 0.0003761
7: -0.0059672, -0.0039283, -0.0059261, -0.0038564, -0.0010819, 0.0009731
8: -0.0027022, -0.0016300, -0.0026806, -0.0015922, -0.0005689, 0.0005117
9: 0.0000262, 0.0012695, -0.0000176, 0.0012445, -0.0005934, 0.0006597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A2_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005751
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005751
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906651, 0.9925478, 0.9906523, 0.9924982, -0.0008403, 0.0009048
1: -0.0035900, -0.0031208, -0.0035931, -0.0031332, -0.0002094, 0.0002254
2: 0.0064848, 0.0089709, 0.0065502, 0.0089878, -0.0011947, 0.0011096
3: -0.0053563, -0.0042247, -0.0053640, -0.0042545, -0.0005050, 0.0005438
4: 0.0017830, 0.0022642, 0.0017957, 0.0022675, -0.0002312, 0.0002148
5: 0.0071156, 0.0102424, 0.0071979, 0.0102637, -0.0015027, 0.0013956
6: -0.0010588, -0.0002652, -0.0010642, -0.0002861, -0.0003542, 0.0003814
7: -0.0058771, -0.0038237, -0.0058910, -0.0038778, -0.0009164, 0.0009868
8: -0.0026548, -0.0015750, -0.0026622, -0.0016034, -0.0004819, 0.0005189
9: -0.0000375, 0.0012146, -0.0000046, 0.0012231, -0.0006017, 0.0005588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0006050
time: 0.64 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005386, upper bound: 0.0006146
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906650, 0.9925492, 0.9906349, 0.9925060, -0.0008447, 0.0009317
1: -0.0035900, -0.0031205, -0.0035975, -0.0031313, -0.0002105, 0.0002322
2: 0.0064831, 0.0089712, 0.0065401, 0.0090108, -0.0012303, 0.0011155
3: -0.0053564, -0.0042239, -0.0053745, -0.0042499, -0.0005077, 0.0005600
4: 0.0017827, 0.0022642, 0.0017937, 0.0022719, -0.0002381, 0.0002159
5: 0.0071134, 0.0102428, 0.0071851, 0.0102927, -0.0015474, 0.0014030
6: -0.0010589, -0.0002646, -0.0010716, -0.0002828, -0.0003561, 0.0003928
7: -0.0058773, -0.0038223, -0.0059101, -0.0038694, -0.0009213, 0.0010162
8: -0.0026550, -0.0015743, -0.0026722, -0.0015990, -0.0004845, 0.0005344
9: -0.0000384, 0.0012147, -0.0000097, 0.0012347, -0.0006197, 0.0005618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005492, upper bound: 0.0006050
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005470, upper bound: 0.0006146
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906324, 0.9925655, 0.9906400, 0.9924983, -0.0008496, 0.0009211
1: -0.0035981, -0.0031164, -0.0035962, -0.0031332, -0.0002117, 0.0002295
2: 0.0064615, 0.0090142, 0.0065502, 0.0090040, -0.0012163, 0.0011219
3: -0.0053760, -0.0042141, -0.0053713, -0.0042545, -0.0005107, 0.0005536
4: 0.0017785, 0.0022726, 0.0017957, 0.0022706, -0.0002354, 0.0002172
5: 0.0070863, 0.0102970, 0.0071978, 0.0102841, -0.0015298, 0.0014111
6: -0.0010726, -0.0002578, -0.0010694, -0.0002861, -0.0003582, 0.0003883
7: -0.0059129, -0.0038045, -0.0059044, -0.0038777, -0.0009267, 0.0010046
8: -0.0026737, -0.0015649, -0.0026692, -0.0016034, -0.0004873, 0.0005283
9: -0.0000493, 0.0012364, -0.0000046, 0.0012313, -0.0006126, 0.0005651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005836, upper bound: 0.0006050
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005766, upper bound: 0.0006145
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906321, 0.9925671, 0.9906232, 0.9925060, -0.0008551, 0.0009460
1: -0.0035982, -0.0031160, -0.0036004, -0.0031312, -0.0002131, 0.0002357
2: 0.0064594, 0.0090145, 0.0065400, 0.0090261, -0.0012491, 0.0011291
3: -0.0053761, -0.0042132, -0.0053814, -0.0042498, -0.0005139, 0.0005686
4: 0.0017781, 0.0022726, 0.0017937, 0.0022749, -0.0002418, 0.0002185
5: 0.0070837, 0.0102973, 0.0071850, 0.0103119, -0.0015711, 0.0014202
6: -0.0010727, -0.0002571, -0.0010765, -0.0002828, -0.0003605, 0.0003988
7: -0.0059131, -0.0038028, -0.0059227, -0.0038693, -0.0009326, 0.0010317
8: -0.0026738, -0.0015640, -0.0026789, -0.0015990, -0.0004904, 0.0005426
9: -0.0000503, 0.0012365, -0.0000098, 0.0012424, -0.0006291, 0.0005687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005878, upper bound: 0.0006050
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005813, upper bound: 0.0006146
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906679, 0.9925244, 0.9905945, 0.9924596, -0.0008494, 0.0010009
1: -0.0035893, -0.0031267, -0.0036076, -0.0031428, -0.0002116, 0.0002494
2: 0.0065157, 0.0089673, 0.0066013, 0.0090641, -0.0013217, 0.0011216
3: -0.0053546, -0.0042388, -0.0053987, -0.0042778, -0.0005105, 0.0006016
4: 0.0017890, 0.0022635, 0.0018056, 0.0022822, -0.0002558, 0.0002171
5: 0.0071545, 0.0102380, 0.0072622, 0.0103597, -0.0016623, 0.0014106
6: -0.0010577, -0.0002751, -0.0010886, -0.0003024, -0.0003580, 0.0004219
7: -0.0058741, -0.0038493, -0.0059541, -0.0039200, -0.0009264, 0.0010916
8: -0.0026533, -0.0015885, -0.0026954, -0.0016256, -0.0004872, 0.0005741
9: -0.0000220, 0.0012128, 0.0000211, 0.0012615, -0.0006657, 0.0005649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005413, upper bound: 0.0006117
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0006055
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906506, 0.9925340, 0.9905944, 0.9924608, -0.0008767, 0.0010070
1: -0.0035936, -0.0031243, -0.0036076, -0.0031425, -0.0002185, 0.0002509
2: 0.0065031, 0.0089900, 0.0065998, 0.0090643, -0.0013297, 0.0011577
3: -0.0053650, -0.0042331, -0.0053988, -0.0042770, -0.0005269, 0.0006052
4: 0.0017866, 0.0022679, 0.0018053, 0.0022823, -0.0002574, 0.0002241
5: 0.0071387, 0.0102666, 0.0072602, 0.0103599, -0.0016724, 0.0014561
6: -0.0010649, -0.0002710, -0.0010886, -0.0003019, -0.0003696, 0.0004245
7: -0.0058929, -0.0038389, -0.0059542, -0.0039187, -0.0009562, 0.0010983
8: -0.0026632, -0.0015830, -0.0026954, -0.0016249, -0.0005029, 0.0005776
9: -0.0000283, 0.0012242, 0.0000204, 0.0012616, -0.0006697, 0.0005831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005421, upper bound: 0.0006170
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0006117
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906346, 0.9925426, 0.9905821, 0.9924597, -0.0008669, 0.0010159
1: -0.0035976, -0.0031221, -0.0036107, -0.0031428, -0.0002160, 0.0002531
2: 0.0064917, 0.0090113, 0.0066012, 0.0090806, -0.0013414, 0.0011448
3: -0.0053747, -0.0042279, -0.0054062, -0.0042777, -0.0005210, 0.0006106
4: 0.0017843, 0.0022720, 0.0018055, 0.0022854, -0.0002596, 0.0002216
5: 0.0071243, 0.0102932, 0.0072620, 0.0103804, -0.0016872, 0.0014398
6: -0.0010717, -0.0002674, -0.0010938, -0.0003023, -0.0003654, 0.0004282
7: -0.0059104, -0.0038294, -0.0059677, -0.0039199, -0.0009455, 0.0011080
8: -0.0026724, -0.0015780, -0.0027025, -0.0016256, -0.0004972, 0.0005827
9: -0.0000341, 0.0012349, 0.0000211, 0.0012698, -0.0006756, 0.0005766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005676, upper bound: 0.0006117
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005741, upper bound: 0.0006055
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906178, 0.9925513, 0.9905819, 0.9924610, -0.0008952, 0.0010209
1: -0.0036017, -0.0031200, -0.0036107, -0.0031425, -0.0002230, 0.0002544
2: 0.0064802, 0.0090334, 0.0065996, 0.0090807, -0.0013481, 0.0011820
3: -0.0053847, -0.0042226, -0.0054063, -0.0042770, -0.0005380, 0.0006136
4: 0.0017821, 0.0022763, 0.0018052, 0.0022854, -0.0002609, 0.0002288
5: 0.0071098, 0.0103210, 0.0072600, 0.0103806, -0.0016956, 0.0014867
6: -0.0010788, -0.0002637, -0.0010939, -0.0003018, -0.0003773, 0.0004304
7: -0.0059287, -0.0038199, -0.0059678, -0.0039186, -0.0009763, 0.0011135
8: -0.0026820, -0.0015730, -0.0027026, -0.0016249, -0.0005134, 0.0005856
9: -0.0000399, 0.0012460, 0.0000203, 0.0012699, -0.0006790, 0.0005953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005685, upper bound: 0.0006170
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006117
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9906651, 0.9925478, 0.9906331, 0.9925426, -0.0008365, 0.0008812
1: -0.0035900, -0.0031208, -0.0035979, -0.0031221, -0.0002084, 0.0002196
2: 0.0064848, 0.0089709, 0.0064917, 0.0090132, -0.0011637, 0.0011046
3: -0.0053563, -0.0042247, -0.0053756, -0.0042278, -0.0005028, 0.0005296
4: 0.0017830, 0.0022642, 0.0017843, 0.0022724, -0.0002252, 0.0002138
5: 0.0071156, 0.0102424, 0.0071243, 0.0102957, -0.0014636, 0.0013893
6: -0.0010588, -0.0002652, -0.0010723, -0.0002674, -0.0003526, 0.0003715
7: -0.0058771, -0.0038237, -0.0059121, -0.0038294, -0.0009123, 0.0009611
8: -0.0026548, -0.0015750, -0.0026732, -0.0015780, -0.0004798, 0.0005054
9: -0.0000375, 0.0012146, -0.0000341, 0.0012359, -0.0005861, 0.0005563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0006050
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005386, upper bound: 0.0006146
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9906650, 0.9925492, 0.9906155, 0.9925514, -0.0008412, 0.0009080
1: -0.0035900, -0.0031205, -0.0036023, -0.0031200, -0.0002096, 0.0002262
2: 0.0064831, 0.0089712, 0.0064801, 0.0090364, -0.0011990, 0.0011109
3: -0.0053564, -0.0042239, -0.0053861, -0.0042226, -0.0005056, 0.0005457
4: 0.0017827, 0.0022642, 0.0017821, 0.0022769, -0.0002321, 0.0002150
5: 0.0071134, 0.0102428, 0.0071097, 0.0103248, -0.0015080, 0.0013972
6: -0.0010589, -0.0002646, -0.0010797, -0.0002637, -0.0003546, 0.0003828
7: -0.0058773, -0.0038223, -0.0059312, -0.0038199, -0.0009175, 0.0009903
8: -0.0026550, -0.0015743, -0.0026833, -0.0015730, -0.0004825, 0.0005208
9: -0.0000384, 0.0012147, -0.0000399, 0.0012476, -0.0006039, 0.0005595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005491, upper bound: 0.0006050
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005470, upper bound: 0.0006145
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9906324, 0.9925655, 0.9906212, 0.9925427, -0.0008443, 0.0008878
1: -0.0035981, -0.0031164, -0.0036009, -0.0031221, -0.0002104, 0.0002212
2: 0.0064615, 0.0090142, 0.0064916, 0.0090288, -0.0011723, 0.0011150
3: -0.0053760, -0.0042141, -0.0053826, -0.0042278, -0.0005075, 0.0005336
4: 0.0017785, 0.0022726, 0.0017843, 0.0022754, -0.0002269, 0.0002158
5: 0.0070863, 0.0102970, 0.0071242, 0.0103153, -0.0014745, 0.0014023
6: -0.0010726, -0.0002578, -0.0010773, -0.0002674, -0.0003559, 0.0003742
7: -0.0059129, -0.0038045, -0.0059249, -0.0038294, -0.0009209, 0.0009682
8: -0.0026737, -0.0015649, -0.0026800, -0.0015780, -0.0004843, 0.0005092
9: -0.0000493, 0.0012364, -0.0000341, 0.0012438, -0.0005904, 0.0005615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005836, upper bound: 0.0006050
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005766, upper bound: 0.0006146
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9906321, 0.9925671, 0.9906040, 0.9925516, -0.0008505, 0.0009098
1: -0.0035982, -0.0031160, -0.0036052, -0.0031199, -0.0002119, 0.0002267
2: 0.0064594, 0.0090145, 0.0064800, 0.0090517, -0.0012013, 0.0011231
3: -0.0053761, -0.0042132, -0.0053931, -0.0042225, -0.0005112, 0.0005468
4: 0.0017781, 0.0022726, 0.0017821, 0.0022798, -0.0002325, 0.0002174
5: 0.0070837, 0.0102973, 0.0071095, 0.0103441, -0.0015110, 0.0014126
6: -0.0010727, -0.0002571, -0.0010846, -0.0002636, -0.0003585, 0.0003835
7: -0.0059131, -0.0038028, -0.0059439, -0.0038197, -0.0009276, 0.0009922
8: -0.0026738, -0.0015640, -0.0026900, -0.0015729, -0.0004878, 0.0005218
9: -0.0000503, 0.0012365, -0.0000400, 0.0012553, -0.0006051, 0.0005657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005878, upper bound: 0.0006051
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005813, upper bound: 0.0006145
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906679, 0.9925244, 0.9905672, 0.9925100, -0.0008452, 0.0009766
1: -0.0035893, -0.0031267, -0.0036144, -0.0031303, -0.0002106, 0.0002433
2: 0.0065157, 0.0089673, 0.0065349, 0.0091002, -0.0012896, 0.0011161
3: -0.0053546, -0.0042388, -0.0054151, -0.0042475, -0.0005080, 0.0005870
4: 0.0017890, 0.0022635, 0.0017927, 0.0022892, -0.0002496, 0.0002160
5: 0.0071545, 0.0102380, 0.0071786, 0.0104051, -0.0016220, 0.0014037
6: -0.0010577, -0.0002751, -0.0011001, -0.0002812, -0.0003563, 0.0004117
7: -0.0058741, -0.0038493, -0.0059839, -0.0038651, -0.0009218, 0.0010651
8: -0.0026533, -0.0015885, -0.0027110, -0.0015968, -0.0004848, 0.0005601
9: -0.0000220, 0.0012128, -0.0000123, 0.0012797, -0.0006495, 0.0005621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005413, upper bound: 0.0006117
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0006055
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906506, 0.9925340, 0.9905671, 0.9925113, -0.0008747, 0.0009836
1: -0.0035936, -0.0031243, -0.0036144, -0.0031299, -0.0002179, 0.0002451
2: 0.0065031, 0.0089900, 0.0065330, 0.0091003, -0.0012988, 0.0011550
3: -0.0053650, -0.0042331, -0.0054152, -0.0042467, -0.0005257, 0.0005912
4: 0.0017866, 0.0022679, 0.0017923, 0.0022892, -0.0002514, 0.0002235
5: 0.0071387, 0.0102666, 0.0071762, 0.0104053, -0.0016336, 0.0014526
6: -0.0010649, -0.0002710, -0.0011001, -0.0002806, -0.0003687, 0.0004146
7: -0.0058929, -0.0038389, -0.0059840, -0.0038635, -0.0009539, 0.0010728
8: -0.0026632, -0.0015830, -0.0027111, -0.0015959, -0.0005017, 0.0005642
9: -0.0000283, 0.0012242, -0.0000133, 0.0012798, -0.0006542, 0.0005817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005421, upper bound: 0.0006171
time: 0.68 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0006117
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906346, 0.9925426, 0.9905556, 0.9925100, -0.0008611, 0.0009832
1: -0.0035976, -0.0031221, -0.0036172, -0.0031302, -0.0002146, 0.0002450
2: 0.0064917, 0.0090113, 0.0065347, 0.0091155, -0.0012982, 0.0011371
3: -0.0053747, -0.0042279, -0.0054221, -0.0042474, -0.0005176, 0.0005909
4: 0.0017843, 0.0022720, 0.0017927, 0.0022922, -0.0002513, 0.0002201
5: 0.0071243, 0.0102932, 0.0071784, 0.0104244, -0.0016328, 0.0014302
6: -0.0010717, -0.0002674, -0.0011050, -0.0002811, -0.0003630, 0.0004144
7: -0.0059104, -0.0038294, -0.0059966, -0.0038649, -0.0009392, 0.0010723
8: -0.0026724, -0.0015780, -0.0027177, -0.0015967, -0.0004939, 0.0005639
9: -0.0000341, 0.0012349, -0.0000124, 0.0012874, -0.0006539, 0.0005727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005676, upper bound: 0.0006117
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005741, upper bound: 0.0006055
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906178, 0.9925513, 0.9905555, 0.9925114, -0.0008893, 0.0009878
1: -0.0036017, -0.0031200, -0.0036173, -0.0031299, -0.0002216, 0.0002461
2: 0.0064802, 0.0090334, 0.0065328, 0.0091157, -0.0013044, 0.0011743
3: -0.0053847, -0.0042226, -0.0054222, -0.0042466, -0.0005345, 0.0005937
4: 0.0017821, 0.0022763, 0.0017923, 0.0022922, -0.0002525, 0.0002273
5: 0.0071098, 0.0103210, 0.0071760, 0.0104245, -0.0016406, 0.0014770
6: -0.0010788, -0.0002637, -0.0011050, -0.0002805, -0.0003749, 0.0004164
7: -0.0059287, -0.0038199, -0.0059967, -0.0038634, -0.0009699, 0.0010774
8: -0.0026820, -0.0015730, -0.0027177, -0.0015959, -0.0005101, 0.0005666
9: -0.0000399, 0.0012460, -0.0000134, 0.0012875, -0.0006570, 0.0005915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005684, upper bound: 0.0006170
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006117
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906032, 0.9924821, 0.9906440, 0.9924635, -0.0009092, 0.0008726
1: -0.0036054, -0.0031372, -0.0035952, -0.0031418, -0.0002265, 0.0002174
2: 0.0065715, 0.0090526, 0.0065962, 0.0089987, -0.0011523, 0.0012006
3: -0.0053935, -0.0042642, -0.0053690, -0.0042754, -0.0005465, 0.0005245
4: 0.0017998, 0.0022800, 0.0018046, 0.0022696, -0.0002230, 0.0002324
5: 0.0072247, 0.0103453, 0.0072557, 0.0102775, -0.0014493, 0.0015100
6: -0.0010849, -0.0002929, -0.0010677, -0.0003007, -0.0003833, 0.0003678
7: -0.0059446, -0.0038954, -0.0059001, -0.0039157, -0.0009916, 0.0009517
8: -0.0026904, -0.0016127, -0.0026670, -0.0016234, -0.0005215, 0.0005005
9: 0.0000061, 0.0012557, 0.0000185, 0.0012286, -0.0005804, 0.0006047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005344, upper bound: 0.0005995
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005344, upper bound: 0.0005995
time: 0.71 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906023, 0.9924827, 0.9906544, 0.9924576, -0.0009259, 0.0009008
1: -0.0036056, -0.0031371, -0.0035926, -0.0031433, -0.0002307, 0.0002245
2: 0.0065708, 0.0090539, 0.0066039, 0.0089851, -0.0011895, 0.0012227
3: -0.0053941, -0.0042639, -0.0053627, -0.0042789, -0.0005565, 0.0005414
4: 0.0017997, 0.0022802, 0.0018061, 0.0022669, -0.0002302, 0.0002366
5: 0.0072238, 0.0103469, 0.0072654, 0.0102603, -0.0014961, 0.0015378
6: -0.0010853, -0.0002926, -0.0010633, -0.0003032, -0.0003903, 0.0003797
7: -0.0059457, -0.0038948, -0.0058888, -0.0039221, -0.0010098, 0.0009825
8: -0.0026909, -0.0016124, -0.0026610, -0.0016268, -0.0005311, 0.0005167
9: 0.0000058, 0.0012564, 0.0000224, 0.0012217, -0.0005991, 0.0006158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005307, upper bound: 0.0006100
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005307, upper bound: 0.0006100
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906030, 0.9924843, 0.9906273, 0.9924722, -0.0009160, 0.0009046
1: -0.0036054, -0.0031367, -0.0035994, -0.0031397, -0.0002282, 0.0002254
2: 0.0065687, 0.0090529, 0.0065847, 0.0090209, -0.0011945, 0.0012096
3: -0.0053936, -0.0042629, -0.0053790, -0.0042702, -0.0005506, 0.0005437
4: 0.0017992, 0.0022801, 0.0018023, 0.0022739, -0.0002312, 0.0002341
5: 0.0072211, 0.0103456, 0.0072412, 0.0103053, -0.0015024, 0.0015213
6: -0.0010850, -0.0002920, -0.0010748, -0.0002971, -0.0003861, 0.0003813
7: -0.0059448, -0.0038930, -0.0059184, -0.0039062, -0.0009990, 0.0009866
8: -0.0026905, -0.0016114, -0.0026766, -0.0016184, -0.0005254, 0.0005188
9: 0.0000047, 0.0012559, 0.0000128, 0.0012398, -0.0006016, 0.0006092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005434, upper bound: 0.0006001
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005434, upper bound: 0.0006001
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906021, 0.9924849, 0.9906372, 0.9924685, -0.0009337, 0.0009271
1: -0.0036056, -0.0031365, -0.0035969, -0.0031406, -0.0002327, 0.0002310
2: 0.0065679, 0.0090541, 0.0065895, 0.0090078, -0.0012242, 0.0012330
3: -0.0053941, -0.0042626, -0.0053731, -0.0042724, -0.0005612, 0.0005572
4: 0.0017991, 0.0022803, 0.0018033, 0.0022713, -0.0002369, 0.0002386
5: 0.0072202, 0.0103471, 0.0072473, 0.0102889, -0.0015398, 0.0015508
6: -0.0010854, -0.0002917, -0.0010706, -0.0002986, -0.0003936, 0.0003908
7: -0.0059458, -0.0038924, -0.0059076, -0.0039102, -0.0010184, 0.0010111
8: -0.0026910, -0.0016111, -0.0026709, -0.0016205, -0.0005356, 0.0005318
9: 0.0000043, 0.0012565, 0.0000152, 0.0012332, -0.0006166, 0.0006210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006110
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006110
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905704, 0.9925019, 0.9906319, 0.9924636, -0.0009151, 0.0008963
1: -0.0036136, -0.0031323, -0.0035982, -0.0031418, -0.0002280, 0.0002233
2: 0.0065454, 0.0090960, 0.0065961, 0.0090147, -0.0011835, 0.0012084
3: -0.0054132, -0.0042523, -0.0053762, -0.0042754, -0.0005500, 0.0005387
4: 0.0017947, 0.0022884, 0.0018045, 0.0022727, -0.0002291, 0.0002339
5: 0.0071919, 0.0103998, 0.0072556, 0.0102975, -0.0014886, 0.0015199
6: -0.0010988, -0.0002845, -0.0010728, -0.0003007, -0.0003858, 0.0003778
7: -0.0059804, -0.0038738, -0.0059133, -0.0039157, -0.0009981, 0.0009775
8: -0.0027092, -0.0016013, -0.0026739, -0.0016234, -0.0005249, 0.0005141
9: -0.0000070, 0.0012776, 0.0000185, 0.0012366, -0.0005961, 0.0006086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905696, 0.9925011, 0.9906422, 0.9924577, -0.0009349, 0.0009178
1: -0.0036138, -0.0031325, -0.0035957, -0.0031433, -0.0002329, 0.0002287
2: 0.0065464, 0.0090970, 0.0066039, 0.0090012, -0.0012120, 0.0012345
3: -0.0054137, -0.0042528, -0.0053701, -0.0042789, -0.0005619, 0.0005516
4: 0.0017949, 0.0022886, 0.0018061, 0.0022701, -0.0002346, 0.0002389
5: 0.0071931, 0.0104011, 0.0072654, 0.0102806, -0.0015243, 0.0015527
6: -0.0010991, -0.0002849, -0.0010685, -0.0003032, -0.0003941, 0.0003869
7: -0.0059813, -0.0038746, -0.0059022, -0.0039221, -0.0010196, 0.0010010
8: -0.0027096, -0.0016018, -0.0026680, -0.0016267, -0.0005362, 0.0005264
9: -0.0000065, 0.0012781, 0.0000224, 0.0012299, -0.0006104, 0.0006218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905703, 0.9925030, 0.9906156, 0.9924722, -0.0009246, 0.0009234
1: -0.0036136, -0.0031320, -0.0036023, -0.0031397, -0.0002304, 0.0002301
2: 0.0065440, 0.0090961, 0.0065846, 0.0090363, -0.0012194, 0.0012209
3: -0.0054133, -0.0042517, -0.0053861, -0.0042701, -0.0005557, 0.0005550
4: 0.0017945, 0.0022884, 0.0018023, 0.0022768, -0.0002360, 0.0002363
5: 0.0071901, 0.0104000, 0.0072411, 0.0103247, -0.0015337, 0.0015356
6: -0.0010988, -0.0002841, -0.0010797, -0.0002970, -0.0003898, 0.0003893
7: -0.0059806, -0.0038726, -0.0059311, -0.0039061, -0.0010084, 0.0010071
8: -0.0027093, -0.0016007, -0.0026833, -0.0016184, -0.0005303, 0.0005296
9: -0.0000077, 0.0012777, 0.0000127, 0.0012475, -0.0006142, 0.0006149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905695, 0.9925028, 0.9906256, 0.9924686, -0.0009451, 0.0009415
1: -0.0036138, -0.0031321, -0.0035998, -0.0031406, -0.0002355, 0.0002346
2: 0.0065443, 0.0090971, 0.0065894, 0.0090231, -0.0012433, 0.0012480
3: -0.0054137, -0.0042518, -0.0053800, -0.0042723, -0.0005681, 0.0005659
4: 0.0017945, 0.0022886, 0.0018032, 0.0022743, -0.0002406, 0.0002416
5: 0.0071904, 0.0104012, 0.0072471, 0.0103081, -0.0015637, 0.0015697
6: -0.0010991, -0.0002842, -0.0010755, -0.0002986, -0.0003984, 0.0003969
7: -0.0059814, -0.0038729, -0.0059202, -0.0039101, -0.0010308, 0.0010269
8: -0.0027097, -0.0016008, -0.0026775, -0.0016204, -0.0005421, 0.0005400
9: -0.0000076, 0.0012782, 0.0000151, 0.0012409, -0.0006262, 0.0006286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905570, 0.9925020, 0.9906258, 0.9925107, -0.0009558, 0.0008369
1: -0.0036169, -0.0031322, -0.0035998, -0.0031301, -0.0002382, 0.0002085
2: 0.0065452, 0.0091138, 0.0065338, 0.0090228, -0.0011051, 0.0012621
3: -0.0054213, -0.0042522, -0.0053799, -0.0042470, -0.0005744, 0.0005030
4: 0.0017947, 0.0022918, 0.0017925, 0.0022742, -0.0002139, 0.0002443
5: 0.0071916, 0.0104222, 0.0071772, 0.0103078, -0.0013900, 0.0015874
6: -0.0011044, -0.0002845, -0.0010754, -0.0002808, -0.0004029, 0.0003528
7: -0.0059951, -0.0038737, -0.0059200, -0.0038642, -0.0010424, 0.0009128
8: -0.0027169, -0.0016013, -0.0026774, -0.0015963, -0.0005482, 0.0004800
9: -0.0000071, 0.0012865, -0.0000129, 0.0012407, -0.0005566, 0.0006356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905562, 0.9925013, 0.9906366, 0.9925062, -0.0009558, 0.0008681
1: -0.0036171, -0.0031324, -0.0035971, -0.0031312, -0.0002382, 0.0002163
2: 0.0065462, 0.0091148, 0.0065397, 0.0090085, -0.0011463, 0.0012621
3: -0.0054218, -0.0042527, -0.0053734, -0.0042497, -0.0005745, 0.0005217
4: 0.0017949, 0.0022920, 0.0017936, 0.0022715, -0.0002219, 0.0002443
5: 0.0071929, 0.0104234, 0.0071847, 0.0102898, -0.0014417, 0.0015874
6: -0.0011047, -0.0002848, -0.0010708, -0.0002827, -0.0004029, 0.0003659
7: -0.0059959, -0.0038745, -0.0059082, -0.0038691, -0.0010424, 0.0009468
8: -0.0027174, -0.0016017, -0.0026712, -0.0015989, -0.0005482, 0.0004979
9: -0.0000066, 0.0012871, -0.0000099, 0.0012335, -0.0005773, 0.0006357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
time: 0.69 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905569, 0.9925031, 0.9906107, 0.9925209, -0.0009637, 0.0008670
1: -0.0036169, -0.0031320, -0.0036035, -0.0031275, -0.0002401, 0.0002160
2: 0.0065438, 0.0091139, 0.0065204, 0.0090428, -0.0011449, 0.0012725
3: -0.0054214, -0.0042516, -0.0053890, -0.0042409, -0.0005792, 0.0005211
4: 0.0017944, 0.0022919, 0.0017899, 0.0022781, -0.0002216, 0.0002463
5: 0.0071898, 0.0104223, 0.0071603, 0.0103329, -0.0014400, 0.0016005
6: -0.0011045, -0.0002840, -0.0010818, -0.0002765, -0.0004062, 0.0003655
7: -0.0059952, -0.0038725, -0.0059365, -0.0038531, -0.0010510, 0.0009456
8: -0.0027170, -0.0016007, -0.0026861, -0.0015905, -0.0005527, 0.0004973
9: -0.0000078, 0.0012866, -0.0000196, 0.0012508, -0.0005766, 0.0006409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005796, upper bound: 0.0006001
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005796, upper bound: 0.0006001
time: 0.67 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905561, 0.9925030, 0.9906201, 0.9925179, -0.0009641, 0.0008931
1: -0.0036171, -0.0031320, -0.0036012, -0.0031283, -0.0002402, 0.0002225
2: 0.0065441, 0.0091149, 0.0065243, 0.0090302, -0.0011794, 0.0012731
3: -0.0054218, -0.0042517, -0.0053833, -0.0042427, -0.0005795, 0.0005368
4: 0.0017945, 0.0022921, 0.0017907, 0.0022757, -0.0002283, 0.0002464
5: 0.0071902, 0.0104236, 0.0071654, 0.0103171, -0.0014834, 0.0016013
6: -0.0011048, -0.0002841, -0.0010778, -0.0002778, -0.0004064, 0.0003765
7: -0.0059960, -0.0038727, -0.0059261, -0.0038564, -0.0010515, 0.0009741
8: -0.0027174, -0.0016008, -0.0026806, -0.0015922, -0.0005530, 0.0005123
9: -0.0000077, 0.0012871, -0.0000176, 0.0012445, -0.0005940, 0.0006412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
time: 0.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.92 seconds
IS_A1_A1_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005492, upper bound: 0.0005861
IS_A1_A1_B1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005454, upper bound: 0.0005968
IS_A1_A1_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005594, upper bound: 0.0005862
IS_A1_A1_B1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005968
IS_A1_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005994, upper bound: 0.0005861
IS_A1_A1_B1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005925, upper bound: 0.0005968
IS_A1_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006028, upper bound: 0.0005861
IS_A1_A1_B1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005968, upper bound: 0.0005968
IS_A1_A1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005415
IS_A1_A1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005386
IS_A1_A1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005492
IS_A1_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005470
IS_A1_A1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005836
IS_A1_A1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005766
IS_A1_A1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006050, upper bound: 0.0005878
IS_A1_A1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006145, upper bound: 0.0005813
IS_A1_A1_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005506, upper bound: 0.0005952
IS_A1_A1_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005511, upper bound: 0.0005893
IS_A1_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0005997
IS_A1_A1_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005516, upper bound: 0.0005945
IS_A1_A1_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005788, upper bound: 0.0005952
IS_A1_A1_B2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005889, upper bound: 0.0005894
IS_A1_A1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005793, upper bound: 0.0005997
IS_A1_A1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005895, upper bound: 0.0005945
IS_A1_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005995, upper bound: 0.0005370
IS_A1_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006100, upper bound: 0.0005343
IS_A1_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0005460
IS_A1_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006110, upper bound: 0.0005443
IS_A1_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005994, upper bound: 0.0005789
IS_A1_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006100, upper bound: 0.0005726
IS_A1_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006001, upper bound: 0.0005850
IS_A1_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006109, upper bound: 0.0005791
IS_A1_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005413
IS_A1_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005413
IS_A1_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005422
IS_A1_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005422
IS_A1_A2_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005421
IS_A1_A2_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005421
IS_A1_A2_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005430
IS_A1_A2_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005430
IS_A1_A2_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005676
IS_A1_A2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005676
IS_A1_A2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005741
IS_A1_A2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005990, upper bound: 0.0005741
IS_A1_A2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005684
IS_A1_A2_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006159, upper bound: 0.0005685
IS_A1_A2_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005751
IS_A1_A2_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0005751
IS_A2_A1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0006050
IS_A2_A1_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005386, upper bound: 0.0006146
IS_A2_A1_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005492, upper bound: 0.0006050
IS_A2_A1_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005470, upper bound: 0.0006146
IS_A2_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005836, upper bound: 0.0006050
IS_A2_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005766, upper bound: 0.0006145
IS_A2_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005878, upper bound: 0.0006050
IS_A2_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005813, upper bound: 0.0006146
IS_A2_A1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005413, upper bound: 0.0006117
IS_A2_A1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0006055
IS_A2_A1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005421, upper bound: 0.0006170
IS_A2_A1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0006117
IS_A2_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005676, upper bound: 0.0006117
IS_A2_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005741, upper bound: 0.0006055
IS_A2_A1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005685, upper bound: 0.0006170
IS_A2_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006117
IS_A2_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0006050
IS_A2_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005386, upper bound: 0.0006146
IS_A2_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005491, upper bound: 0.0006050
IS_A2_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005470, upper bound: 0.0006145
IS_A2_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005836, upper bound: 0.0006050
IS_A2_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005766, upper bound: 0.0006146
IS_A2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005878, upper bound: 0.0006051
IS_A2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005813, upper bound: 0.0006145
IS_A2_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005413, upper bound: 0.0006117
IS_A2_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0006055
IS_A2_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005421, upper bound: 0.0006171
IS_A2_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005430, upper bound: 0.0006117
IS_A2_A1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005676, upper bound: 0.0006117
IS_A2_A1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005741, upper bound: 0.0006055
IS_A2_A1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005684, upper bound: 0.0006170
IS_A2_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005751, upper bound: 0.0006117
IS_A2_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005344, upper bound: 0.0005995
IS_A2_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005344, upper bound: 0.0005995
IS_A2_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005307, upper bound: 0.0006100
IS_A2_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005307, upper bound: 0.0006100
IS_A2_A2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005434, upper bound: 0.0006001
IS_A2_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005434, upper bound: 0.0006001
IS_A2_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006110
IS_A2_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005425, upper bound: 0.0006110
IS_A2_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
IS_A2_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
IS_A2_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
IS_A2_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
IS_A2_A2_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
IS_A2_A2_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005795, upper bound: 0.0006001
IS_A2_A2_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
IS_A2_A2_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
IS_A2_A2_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
IS_A2_A2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005718, upper bound: 0.0005995
IS_A2_A2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
IS_A2_A2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0006100
IS_A2_A2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005796, upper bound: 0.0006001
IS_A2_A2_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005796, upper bound: 0.0006001
IS_A2_A2_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110
IS_A2_A2_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006110

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906523, 0.9925131, 0.9906321, 0.9924694, -0.0007553, 0.0008363
1: -0.0035932, -0.0031295, -0.0035982, -0.0031404, -0.0001882, 0.0002084
2: 0.0065306, 0.0089879, 0.0065883, 0.0090145, -0.0011043, 0.0009973
3: -0.0053640, -0.0042456, -0.0053761, -0.0042718, -0.0004539, 0.0005026
4: 0.0017919, 0.0022675, 0.0018030, 0.0022726, -0.0002137, 0.0001930
5: 0.0071732, 0.0102639, 0.0072459, 0.0102973, -0.0013889, 0.0012544
6: -0.0010643, -0.0002798, -0.0010727, -0.0002982, -0.0003184, 0.0003525
7: -0.0058912, -0.0038616, -0.0059131, -0.0039093, -0.0008237, 0.0009121
8: -0.0026623, -0.0015949, -0.0026738, -0.0016200, -0.0004332, 0.0004796
9: -0.0000145, 0.0012232, 0.0000146, 0.0012366, -0.0005562, 0.0005023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005767, upper bound: 0.0005491
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0005673
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906520, 0.9925137, 0.9906156, 0.9924769, -0.0007599, 0.0008618
1: -0.0035932, -0.0031293, -0.0036023, -0.0031385, -0.0001893, 0.0002147
2: 0.0065298, 0.0089883, 0.0065784, 0.0090363, -0.0011380, 0.0010034
3: -0.0053642, -0.0042452, -0.0053861, -0.0042673, -0.0004567, 0.0005180
4: 0.0017917, 0.0022676, 0.0018011, 0.0022769, -0.0002203, 0.0001942
5: 0.0071722, 0.0102644, 0.0072334, 0.0103248, -0.0014313, 0.0012621
6: -0.0010644, -0.0002796, -0.0010797, -0.0002951, -0.0003203, 0.0003633
7: -0.0058915, -0.0038609, -0.0059312, -0.0039011, -0.0008288, 0.0009399
8: -0.0026624, -0.0015946, -0.0026833, -0.0016157, -0.0004358, 0.0004943
9: -0.0000149, 0.0012233, 0.0000096, 0.0012475, -0.0005732, 0.0005054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005801, upper bound: 0.0005491
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005846, upper bound: 0.0005673
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906441, 0.9924694, 0.9906667, 0.9925388, -0.0008712, 0.0007876
1: -0.0035952, -0.0031404, -0.0035896, -0.0031231, -0.0002171, 0.0001963
2: 0.0065884, 0.0089986, 0.0064968, 0.0089689, -0.0010400, 0.0011504
3: -0.0053689, -0.0042719, -0.0053554, -0.0042302, -0.0005236, 0.0004734
4: 0.0018031, 0.0022696, 0.0017853, 0.0022638, -0.0002013, 0.0002226
5: 0.0072459, 0.0102774, 0.0071307, 0.0102399, -0.0013081, 0.0014469
6: -0.0010677, -0.0002983, -0.0010582, -0.0002690, -0.0003672, 0.0003320
7: -0.0059000, -0.0039093, -0.0058754, -0.0038336, -0.0009501, 0.0008590
8: -0.0026669, -0.0016200, -0.0026540, -0.0015802, -0.0004997, 0.0004517
9: 0.0000146, 0.0012286, -0.0000315, 0.0012136, -0.0005238, 0.0005794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005156
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005866, upper bound: 0.0005253
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906546, 0.9924650, 0.9906659, 0.9925395, -0.0008991, 0.0007976
1: -0.0035926, -0.0031415, -0.0035898, -0.0031229, -0.0002240, 0.0001987
2: 0.0065942, 0.0089849, 0.0064958, 0.0089699, -0.0010532, 0.0011872
3: -0.0053627, -0.0042745, -0.0053558, -0.0042297, -0.0005404, 0.0004794
4: 0.0018042, 0.0022669, 0.0017851, 0.0022640, -0.0002038, 0.0002298
5: 0.0072532, 0.0102601, 0.0071294, 0.0102412, -0.0013246, 0.0014932
6: -0.0010633, -0.0003001, -0.0010585, -0.0002687, -0.0003790, 0.0003362
7: -0.0058887, -0.0039141, -0.0058763, -0.0038328, -0.0009806, 0.0008699
8: -0.0026609, -0.0016225, -0.0026544, -0.0015798, -0.0005157, 0.0004575
9: 0.0000176, 0.0012216, -0.0000320, 0.0012141, -0.0005304, 0.0005979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0005125
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005221
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906272, 0.9924768, 0.9906665, 0.9925401, -0.0009038, 0.0007903
1: -0.0035994, -0.0031385, -0.0035896, -0.0031228, -0.0002252, 0.0001969
2: 0.0065785, 0.0090210, 0.0064950, 0.0089691, -0.0010435, 0.0011934
3: -0.0053791, -0.0042674, -0.0053555, -0.0042294, -0.0005432, 0.0004750
4: 0.0018011, 0.0022739, 0.0017850, 0.0022638, -0.0002020, 0.0002310
5: 0.0072335, 0.0103055, 0.0071284, 0.0102403, -0.0013125, 0.0015010
6: -0.0010748, -0.0002951, -0.0010583, -0.0002684, -0.0003810, 0.0003331
7: -0.0059185, -0.0039012, -0.0058757, -0.0038322, -0.0009857, 0.0008619
8: -0.0026766, -0.0016157, -0.0026541, -0.0015794, -0.0005184, 0.0004533
9: 0.0000097, 0.0012398, -0.0000324, 0.0012137, -0.0005256, 0.0006011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005251
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005865, upper bound: 0.0005322
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906373, 0.9924735, 0.9906657, 0.9925408, -0.0009259, 0.0008020
1: -0.0035969, -0.0031393, -0.0035898, -0.0031226, -0.0002307, 0.0001998
2: 0.0065828, 0.0090077, 0.0064940, 0.0089702, -0.0010591, 0.0012226
3: -0.0053730, -0.0042693, -0.0053560, -0.0042289, -0.0005565, 0.0004820
4: 0.0018020, 0.0022713, 0.0017848, 0.0022641, -0.0002050, 0.0002366
5: 0.0072389, 0.0102887, 0.0071272, 0.0102416, -0.0013320, 0.0015377
6: -0.0010706, -0.0002965, -0.0010586, -0.0002681, -0.0003903, 0.0003381
7: -0.0059075, -0.0039047, -0.0058765, -0.0038314, -0.0010098, 0.0008747
8: -0.0026708, -0.0016176, -0.0026546, -0.0015790, -0.0005310, 0.0004600
9: 0.0000118, 0.0012331, -0.0000329, 0.0012142, -0.0005334, 0.0006158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0005230
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005302
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906321, 0.9924694, 0.9906338, 0.9925571, -0.0008929, 0.0007915
1: -0.0035982, -0.0031404, -0.0035978, -0.0031185, -0.0002225, 0.0001972
2: 0.0065883, 0.0090145, 0.0064725, 0.0090123, -0.0010452, 0.0011791
3: -0.0053761, -0.0042718, -0.0053751, -0.0042191, -0.0005367, 0.0004757
4: 0.0018030, 0.0022726, 0.0017806, 0.0022722, -0.0002023, 0.0002282
5: 0.0072459, 0.0102973, 0.0071001, 0.0102946, -0.0013146, 0.0014830
6: -0.0010727, -0.0002982, -0.0010720, -0.0002613, -0.0003764, 0.0003337
7: -0.0059131, -0.0039093, -0.0059113, -0.0038136, -0.0009739, 0.0008633
8: -0.0026738, -0.0016200, -0.0026728, -0.0015697, -0.0005121, 0.0004540
9: 0.0000146, 0.0012366, -0.0000437, 0.0012354, -0.0005264, 0.0005939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005592
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005866, upper bound: 0.0005664
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906422, 0.9924650, 0.9906329, 0.9925569, -0.0009143, 0.0008083
1: -0.0035957, -0.0031415, -0.0035980, -0.0031186, -0.0002278, 0.0002014
2: 0.0065941, 0.0090011, 0.0064729, 0.0090134, -0.0010673, 0.0012073
3: -0.0053700, -0.0042745, -0.0053756, -0.0042193, -0.0005495, 0.0004858
4: 0.0018042, 0.0022700, 0.0017807, 0.0022724, -0.0002066, 0.0002337
5: 0.0072532, 0.0102805, 0.0071006, 0.0102960, -0.0013424, 0.0015185
6: -0.0010685, -0.0003001, -0.0010724, -0.0002614, -0.0003854, 0.0003407
7: -0.0059021, -0.0039141, -0.0059123, -0.0038139, -0.0009972, 0.0008816
8: -0.0026680, -0.0016225, -0.0026733, -0.0015698, -0.0005244, 0.0004636
9: 0.0000175, 0.0012298, -0.0000436, 0.0012360, -0.0005376, 0.0006081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0005522
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005588
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906156, 0.9924769, 0.9906335, 0.9925585, -0.0009209, 0.0007962
1: -0.0036023, -0.0031385, -0.0035978, -0.0031182, -0.0002295, 0.0001984
2: 0.0065784, 0.0090363, 0.0064706, 0.0090126, -0.0010514, 0.0012160
3: -0.0053861, -0.0042673, -0.0053753, -0.0042182, -0.0005535, 0.0004785
4: 0.0018011, 0.0022769, 0.0017803, 0.0022723, -0.0002035, 0.0002354
5: 0.0072334, 0.0103248, 0.0070977, 0.0102950, -0.0013224, 0.0015294
6: -0.0010797, -0.0002951, -0.0010721, -0.0002607, -0.0003882, 0.0003356
7: -0.0059312, -0.0039011, -0.0059116, -0.0038120, -0.0010043, 0.0008684
8: -0.0026833, -0.0016157, -0.0026730, -0.0015688, -0.0005282, 0.0004567
9: 0.0000096, 0.0012475, -0.0000447, 0.0012356, -0.0005295, 0.0006124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005640
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005865, upper bound: 0.0005697
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906256, 0.9924737, 0.9906327, 0.9925586, -0.0009388, 0.0008137
1: -0.0035998, -0.0031393, -0.0035980, -0.0031182, -0.0002339, 0.0002028
2: 0.0065827, 0.0090230, 0.0064706, 0.0090137, -0.0010745, 0.0012397
3: -0.0053800, -0.0042693, -0.0053757, -0.0042183, -0.0005643, 0.0004891
4: 0.0018020, 0.0022743, 0.0017803, 0.0022725, -0.0002080, 0.0002399
5: 0.0072388, 0.0103081, 0.0070977, 0.0102963, -0.0013514, 0.0015592
6: -0.0010755, -0.0002965, -0.0010725, -0.0002607, -0.0003958, 0.0003430
7: -0.0059202, -0.0039046, -0.0059124, -0.0038120, -0.0010239, 0.0008875
8: -0.0026775, -0.0016176, -0.0026734, -0.0015688, -0.0005385, 0.0004667
9: 0.0000118, 0.0012408, -0.0000447, 0.0012361, -0.0005412, 0.0006244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0005579
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005633
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906617, 0.9924605, 0.9905958, 0.9924526, -0.0008218, 0.0008956
1: -0.0035908, -0.0031426, -0.0036072, -0.0031446, -0.0002048, 0.0002232
2: 0.0066002, 0.0089754, 0.0066106, 0.0090625, -0.0011826, 0.0010852
3: -0.0053583, -0.0042772, -0.0053980, -0.0042820, -0.0004939, 0.0005383
4: 0.0018053, 0.0022651, 0.0018074, 0.0022819, -0.0002289, 0.0002100
5: 0.0072607, 0.0102481, 0.0072739, 0.0103577, -0.0014874, 0.0013649
6: -0.0010603, -0.0003020, -0.0010881, -0.0003054, -0.0003464, 0.0003775
7: -0.0058808, -0.0039190, -0.0059528, -0.0039277, -0.0008963, 0.0009767
8: -0.0026568, -0.0016251, -0.0026946, -0.0016297, -0.0004714, 0.0005137
9: 0.0000206, 0.0012168, 0.0000258, 0.0012607, -0.0005956, 0.0005466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005002, upper bound: 0.0005714
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005364, upper bound: 0.0005839
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906297, 0.9924768, 0.9905832, 0.9924527, -0.0008384, 0.0009076
1: -0.0035988, -0.0031385, -0.0036104, -0.0031446, -0.0002089, 0.0002262
2: 0.0065786, 0.0090177, 0.0066105, 0.0090790, -0.0011985, 0.0011071
3: -0.0053776, -0.0042674, -0.0054055, -0.0042819, -0.0005039, 0.0005455
4: 0.0018012, 0.0022732, 0.0018073, 0.0022851, -0.0002320, 0.0002143
5: 0.0072335, 0.0103013, 0.0072737, 0.0103784, -0.0015074, 0.0013924
6: -0.0010738, -0.0002951, -0.0010933, -0.0003053, -0.0003534, 0.0003826
7: -0.0059158, -0.0039012, -0.0059664, -0.0039275, -0.0009144, 0.0009899
8: -0.0026752, -0.0016157, -0.0027018, -0.0016296, -0.0004809, 0.0005206
9: 0.0000097, 0.0012382, 0.0000258, 0.0012690, -0.0006036, 0.0005576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005341, upper bound: 0.0005718
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005634, upper bound: 0.0005839
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906441, 0.9924694, 0.9906032, 0.9924821, -0.0008622, 0.0008961
1: -0.0035952, -0.0031404, -0.0036054, -0.0031372, -0.0002148, 0.0002233
2: 0.0065884, 0.0089986, 0.0065715, 0.0090526, -0.0011832, 0.0011386
3: -0.0053689, -0.0042719, -0.0053935, -0.0042642, -0.0005182, 0.0005386
4: 0.0018031, 0.0022696, 0.0017998, 0.0022800, -0.0002290, 0.0002204
5: 0.0072459, 0.0102774, 0.0072247, 0.0103453, -0.0014882, 0.0014320
6: -0.0010677, -0.0002983, -0.0010849, -0.0002929, -0.0003635, 0.0003777
7: -0.0059000, -0.0039093, -0.0059446, -0.0038954, -0.0009404, 0.0009773
8: -0.0026669, -0.0016200, -0.0026904, -0.0016127, -0.0004945, 0.0005139
9: 0.0000146, 0.0012286, 0.0000061, 0.0012557, -0.0005959, 0.0005734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0005017
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005852, upper bound: 0.0005237
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906546, 0.9924650, 0.9906023, 0.9924827, -0.0008873, 0.0009155
1: -0.0035926, -0.0031415, -0.0036056, -0.0031371, -0.0002211, 0.0002281
2: 0.0065942, 0.0089849, 0.0065708, 0.0090539, -0.0012089, 0.0011717
3: -0.0053627, -0.0042745, -0.0053941, -0.0042639, -0.0005333, 0.0005502
4: 0.0018042, 0.0022669, 0.0017997, 0.0022802, -0.0002340, 0.0002268
5: 0.0072532, 0.0102601, 0.0072238, 0.0103469, -0.0015205, 0.0014737
6: -0.0010633, -0.0003001, -0.0010853, -0.0002926, -0.0003740, 0.0003859
7: -0.0058887, -0.0039141, -0.0059457, -0.0038948, -0.0009677, 0.0009985
8: -0.0026609, -0.0016225, -0.0026909, -0.0016124, -0.0005089, 0.0005251
9: 0.0000176, 0.0012216, 0.0000058, 0.0012564, -0.0006089, 0.0005901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0004977
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0005208
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906272, 0.9924768, 0.9906030, 0.9924843, -0.0008955, 0.0008987
1: -0.0035994, -0.0031385, -0.0036054, -0.0031367, -0.0002231, 0.0002239
2: 0.0065785, 0.0090210, 0.0065687, 0.0090529, -0.0011868, 0.0011825
3: -0.0053791, -0.0042674, -0.0053936, -0.0042629, -0.0005382, 0.0005402
4: 0.0018011, 0.0022739, 0.0017992, 0.0022801, -0.0002297, 0.0002289
5: 0.0072335, 0.0103055, 0.0072211, 0.0103456, -0.0014927, 0.0014872
6: -0.0010748, -0.0002951, -0.0010850, -0.0002920, -0.0003775, 0.0003789
7: -0.0059185, -0.0039012, -0.0059448, -0.0038930, -0.0009766, 0.0009802
8: -0.0026766, -0.0016157, -0.0026905, -0.0016114, -0.0005136, 0.0005155
9: 0.0000097, 0.0012398, 0.0000047, 0.0012559, -0.0005977, 0.0005956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0005110
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005853, upper bound: 0.0005313
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906373, 0.9924735, 0.9906021, 0.9924849, -0.0009159, 0.0009199
1: -0.0035969, -0.0031393, -0.0036056, -0.0031365, -0.0002282, 0.0002292
2: 0.0065828, 0.0090077, 0.0065679, 0.0090541, -0.0012148, 0.0012094
3: -0.0053730, -0.0042693, -0.0053941, -0.0042626, -0.0005505, 0.0005529
4: 0.0018020, 0.0022713, 0.0017991, 0.0022803, -0.0002351, 0.0002341
5: 0.0072389, 0.0102887, 0.0072202, 0.0103471, -0.0015278, 0.0015211
6: -0.0010706, -0.0002965, -0.0010854, -0.0002917, -0.0003861, 0.0003878
7: -0.0059075, -0.0039047, -0.0059458, -0.0038924, -0.0009989, 0.0010033
8: -0.0026708, -0.0016176, -0.0026910, -0.0016111, -0.0005253, 0.0005276
9: 0.0000118, 0.0012331, 0.0000043, 0.0012565, -0.0006118, 0.0006091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0005071
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005957, upper bound: 0.0005293
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906321, 0.9924694, 0.9905704, 0.9925019, -0.0008699, 0.0008962
1: -0.0035982, -0.0031404, -0.0036136, -0.0031323, -0.0002168, 0.0002233
2: 0.0065883, 0.0090145, 0.0065454, 0.0090960, -0.0011834, 0.0011487
3: -0.0053761, -0.0042718, -0.0054132, -0.0042523, -0.0005229, 0.0005386
4: 0.0018030, 0.0022726, 0.0017947, 0.0022884, -0.0002290, 0.0002223
5: 0.0072459, 0.0102973, 0.0071919, 0.0103998, -0.0014884, 0.0014448
6: -0.0010727, -0.0002982, -0.0010988, -0.0002845, -0.0003667, 0.0003778
7: -0.0059131, -0.0039093, -0.0059804, -0.0038738, -0.0009488, 0.0009774
8: -0.0026738, -0.0016200, -0.0027092, -0.0016013, -0.0004990, 0.0005140
9: 0.0000146, 0.0012366, -0.0000070, 0.0012776, -0.0005960, 0.0005786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005451, upper bound: 0.0005499
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005853, upper bound: 0.0005651
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906422, 0.9924650, 0.9905696, 0.9925011, -0.0008882, 0.0009237
1: -0.0035957, -0.0031415, -0.0036138, -0.0031325, -0.0002213, 0.0002302
2: 0.0065941, 0.0090011, 0.0065464, 0.0090970, -0.0012197, 0.0011729
3: -0.0053700, -0.0042745, -0.0054137, -0.0042528, -0.0005338, 0.0005552
4: 0.0018042, 0.0022700, 0.0017949, 0.0022886, -0.0002361, 0.0002270
5: 0.0072532, 0.0102805, 0.0071931, 0.0104011, -0.0015341, 0.0014752
6: -0.0010685, -0.0003001, -0.0010991, -0.0002849, -0.0003744, 0.0003894
7: -0.0059021, -0.0039141, -0.0059813, -0.0038746, -0.0009687, 0.0010074
8: -0.0026680, -0.0016225, -0.0027096, -0.0016018, -0.0005094, 0.0005298
9: 0.0000175, 0.0012298, -0.0000065, 0.0012781, -0.0006143, 0.0005907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005542, upper bound: 0.0005413
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0005579
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906156, 0.9924769, 0.9905703, 0.9925030, -0.0008973, 0.0009008
1: -0.0036023, -0.0031385, -0.0036136, -0.0031320, -0.0002236, 0.0002245
2: 0.0065784, 0.0090363, 0.0065440, 0.0090961, -0.0011895, 0.0011849
3: -0.0053861, -0.0042673, -0.0054133, -0.0042517, -0.0005393, 0.0005414
4: 0.0018011, 0.0022769, 0.0017945, 0.0022884, -0.0002302, 0.0002293
5: 0.0072334, 0.0103248, 0.0071901, 0.0104000, -0.0014961, 0.0014903
6: -0.0010797, -0.0002951, -0.0010988, -0.0002841, -0.0003783, 0.0003797
7: -0.0059312, -0.0039011, -0.0059806, -0.0038726, -0.0009786, 0.0009825
8: -0.0026833, -0.0016157, -0.0027093, -0.0016007, -0.0005147, 0.0005167
9: 0.0000096, 0.0012475, -0.0000077, 0.0012777, -0.0005991, 0.0005968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005451, upper bound: 0.0005551
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005853, upper bound: 0.0005690
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906256, 0.9924737, 0.9905695, 0.9925028, -0.0009141, 0.0009290
1: -0.0035998, -0.0031393, -0.0036138, -0.0031321, -0.0002278, 0.0002315
2: 0.0065827, 0.0090230, 0.0065443, 0.0090971, -0.0012268, 0.0012070
3: -0.0053800, -0.0042693, -0.0054137, -0.0042518, -0.0005494, 0.0005584
4: 0.0018020, 0.0022743, 0.0017945, 0.0022886, -0.0002374, 0.0002336
5: 0.0072388, 0.0103081, 0.0071904, 0.0104012, -0.0015430, 0.0015181
6: -0.0010755, -0.0002965, -0.0010991, -0.0002842, -0.0003853, 0.0003916
7: -0.0059202, -0.0039046, -0.0059814, -0.0038729, -0.0009969, 0.0010132
8: -0.0026775, -0.0016176, -0.0027097, -0.0016008, -0.0005243, 0.0005329
9: 0.0000118, 0.0012408, -0.0000076, 0.0012782, -0.0006179, 0.0006079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005542, upper bound: 0.0005484
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0005626
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905959, 0.9924515, 0.9906613, 0.9924955, -0.0009592, 0.0008237
1: -0.0036072, -0.0031448, -0.0035909, -0.0031339, -0.0002390, 0.0002052
2: 0.0066120, 0.0090624, 0.0065540, 0.0089761, -0.0010877, 0.0012666
3: -0.0053979, -0.0042826, -0.0053586, -0.0042562, -0.0005765, 0.0004951
4: 0.0018076, 0.0022819, 0.0017964, 0.0022652, -0.0002105, 0.0002451
5: 0.0072756, 0.0103575, 0.0072026, 0.0102490, -0.0013680, 0.0015930
6: -0.0010880, -0.0003058, -0.0010605, -0.0002873, -0.0004043, 0.0003472
7: -0.0059527, -0.0039288, -0.0058814, -0.0038809, -0.0010461, 0.0008984
8: -0.0026946, -0.0016303, -0.0026571, -0.0016051, -0.0005501, 0.0004724
9: 0.0000265, 0.0012607, -0.0000027, 0.0012172, -0.0005478, 0.0006379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005756, upper bound: 0.0004932
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005278
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905959, 0.9924515, 0.9905974, 0.9924384, -0.0008428, 0.0008355
1: -0.0036072, -0.0031448, -0.0036068, -0.0031481, -0.0002100, 0.0002082
2: 0.0066120, 0.0090624, 0.0066292, 0.0090603, -0.0011033, 0.0011129
3: -0.0053979, -0.0042826, -0.0053970, -0.0042905, -0.0005065, 0.0005022
4: 0.0018076, 0.0022819, 0.0018110, 0.0022815, -0.0002135, 0.0002154
5: 0.0072756, 0.0103575, 0.0072973, 0.0103549, -0.0013877, 0.0013997
6: -0.0010880, -0.0003058, -0.0010874, -0.0003113, -0.0003553, 0.0003522
7: -0.0059527, -0.0039288, -0.0059509, -0.0039430, -0.0009192, 0.0009113
8: -0.0026946, -0.0016303, -0.0026937, -0.0016378, -0.0004834, 0.0004792
9: 0.0000265, 0.0012607, 0.0000352, 0.0012596, -0.0005557, 0.0005605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005756, upper bound: 0.0004932
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005278
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905951, 0.9924505, 0.9906704, 0.9924922, -0.0009619, 0.0008414
1: -0.0036074, -0.0031451, -0.0035886, -0.0031347, -0.0002397, 0.0002096
2: 0.0066133, 0.0090634, 0.0065583, 0.0089638, -0.0011110, 0.0012702
3: -0.0053984, -0.0042832, -0.0053530, -0.0042582, -0.0005782, 0.0005057
4: 0.0018079, 0.0022821, 0.0017972, 0.0022628, -0.0002150, 0.0002459
5: 0.0072772, 0.0103588, 0.0072080, 0.0102335, -0.0013974, 0.0015976
6: -0.0010883, -0.0003062, -0.0010565, -0.0002886, -0.0004055, 0.0003547
7: -0.0059535, -0.0039299, -0.0058712, -0.0038844, -0.0010491, 0.0009176
8: -0.0026950, -0.0016308, -0.0026518, -0.0016069, -0.0005517, 0.0004826
9: 0.0000272, 0.0012612, -0.0000005, 0.0012110, -0.0005596, 0.0006398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0004943
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005287
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905951, 0.9924505, 0.9906067, 0.9924340, -0.0008368, 0.0008568
1: -0.0036074, -0.0031451, -0.0036045, -0.0031492, -0.0002085, 0.0002135
2: 0.0066133, 0.0090634, 0.0066351, 0.0090480, -0.0011314, 0.0011050
3: -0.0053984, -0.0042832, -0.0053914, -0.0042931, -0.0005029, 0.0005150
4: 0.0018079, 0.0022821, 0.0018121, 0.0022791, -0.0002190, 0.0002139
5: 0.0072772, 0.0103588, 0.0073047, 0.0103395, -0.0014230, 0.0013898
6: -0.0010883, -0.0003062, -0.0010834, -0.0003132, -0.0003527, 0.0003612
7: -0.0059535, -0.0039299, -0.0059408, -0.0039479, -0.0009126, 0.0009345
8: -0.0026950, -0.0016308, -0.0026884, -0.0016403, -0.0004799, 0.0004914
9: 0.0000272, 0.0012612, 0.0000382, 0.0012534, -0.0005698, 0.0005565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0004943
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005287
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905958, 0.9924526, 0.9906445, 0.9925039, -0.0009683, 0.0008547
1: -0.0036072, -0.0031446, -0.0035951, -0.0031318, -0.0002413, 0.0002130
2: 0.0066106, 0.0090625, 0.0065428, 0.0089981, -0.0011287, 0.0012786
3: -0.0053980, -0.0042820, -0.0053687, -0.0042511, -0.0005820, 0.0005137
4: 0.0018074, 0.0022819, 0.0017942, 0.0022695, -0.0002184, 0.0002475
5: 0.0072739, 0.0103577, 0.0071886, 0.0102767, -0.0014195, 0.0016082
6: -0.0010881, -0.0003054, -0.0010675, -0.0002837, -0.0004082, 0.0003603
7: -0.0059528, -0.0039277, -0.0058996, -0.0038717, -0.0010561, 0.0009322
8: -0.0026946, -0.0016297, -0.0026667, -0.0016002, -0.0005554, 0.0004902
9: 0.0000258, 0.0012607, -0.0000083, 0.0012283, -0.0005684, 0.0006440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005823, upper bound: 0.0004932
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005281
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905958, 0.9924526, 0.9905801, 0.9924491, -0.0008515, 0.0008618
1: -0.0036072, -0.0031446, -0.0036111, -0.0031454, -0.0002122, 0.0002147
2: 0.0066106, 0.0090625, 0.0066152, 0.0090832, -0.0011380, 0.0011244
3: -0.0053980, -0.0042820, -0.0054074, -0.0042841, -0.0005118, 0.0005180
4: 0.0018074, 0.0022819, 0.0018082, 0.0022859, -0.0002203, 0.0002176
5: 0.0072739, 0.0103577, 0.0072796, 0.0103837, -0.0014313, 0.0014142
6: -0.0010881, -0.0003054, -0.0010947, -0.0003068, -0.0003590, 0.0003633
7: -0.0059528, -0.0039277, -0.0059699, -0.0039315, -0.0009287, 0.0009399
8: -0.0026946, -0.0016297, -0.0027036, -0.0016317, -0.0004884, 0.0004943
9: 0.0000258, 0.0012607, 0.0000281, 0.0012711, -0.0005731, 0.0005663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005823, upper bound: 0.0004932
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005281
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905950, 0.9924518, 0.9906534, 0.9925030, -0.0009680, 0.0008689
1: -0.0036074, -0.0031448, -0.0035929, -0.0031320, -0.0002412, 0.0002165
2: 0.0066115, 0.0090635, 0.0065440, 0.0089864, -0.0011473, 0.0012783
3: -0.0053984, -0.0042824, -0.0053633, -0.0042517, -0.0005818, 0.0005222
4: 0.0018075, 0.0022821, 0.0017945, 0.0022672, -0.0002221, 0.0002474
5: 0.0072750, 0.0103589, 0.0071901, 0.0102620, -0.0014430, 0.0016078
6: -0.0010884, -0.0003056, -0.0010638, -0.0002841, -0.0004081, 0.0003663
7: -0.0059536, -0.0039284, -0.0058899, -0.0038727, -0.0010558, 0.0009476
8: -0.0026951, -0.0016301, -0.0026616, -0.0016007, -0.0005552, 0.0004983
9: 0.0000263, 0.0012612, -0.0000077, 0.0012224, -0.0005779, 0.0006438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005746, upper bound: 0.0004943
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005291
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905950, 0.9924518, 0.9905874, 0.9924477, -0.0008437, 0.0008797
1: -0.0036074, -0.0031448, -0.0036093, -0.0031458, -0.0002102, 0.0002192
2: 0.0066115, 0.0090635, 0.0066170, 0.0090736, -0.0011617, 0.0011141
3: -0.0053984, -0.0042824, -0.0054030, -0.0042849, -0.0005071, 0.0005287
4: 0.0018075, 0.0022821, 0.0018086, 0.0022841, -0.0002248, 0.0002156
5: 0.0072750, 0.0103589, 0.0072819, 0.0103717, -0.0014611, 0.0014012
6: -0.0010884, -0.0003056, -0.0010916, -0.0003074, -0.0003556, 0.0003708
7: -0.0059536, -0.0039284, -0.0059619, -0.0039329, -0.0009201, 0.0009595
8: -0.0026951, -0.0016301, -0.0026995, -0.0016324, -0.0004839, 0.0005046
9: 0.0000263, 0.0012612, 0.0000290, 0.0012663, -0.0005851, 0.0005611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005746, upper bound: 0.0004943
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005291
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905834, 0.9924517, 0.9906259, 0.9925148, -0.0009784, 0.0008425
1: -0.0036103, -0.0031448, -0.0035997, -0.0031291, -0.0002438, 0.0002099
2: 0.0066118, 0.0090788, 0.0065284, 0.0090226, -0.0011125, 0.0012919
3: -0.0054054, -0.0042825, -0.0053798, -0.0042446, -0.0005880, 0.0005064
4: 0.0018076, 0.0022851, 0.0017915, 0.0022742, -0.0002153, 0.0002501
5: 0.0072754, 0.0103782, 0.0071705, 0.0103075, -0.0013993, 0.0016249
6: -0.0010933, -0.0003057, -0.0010753, -0.0002791, -0.0004124, 0.0003552
7: -0.0059663, -0.0039287, -0.0059198, -0.0038598, -0.0010671, 0.0009189
8: -0.0027017, -0.0016302, -0.0026773, -0.0015940, -0.0005612, 0.0004832
9: 0.0000264, 0.0012690, -0.0000156, 0.0012406, -0.0005603, 0.0006507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005760, upper bound: 0.0005281
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005527
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905834, 0.9924517, 0.9905627, 0.9924586, -0.0008655, 0.0008477
1: -0.0036103, -0.0031448, -0.0036155, -0.0031431, -0.0002157, 0.0002112
2: 0.0066118, 0.0090788, 0.0066025, 0.0091061, -0.0011194, 0.0011429
3: -0.0054054, -0.0042825, -0.0054178, -0.0042783, -0.0005202, 0.0005095
4: 0.0018076, 0.0022851, 0.0018058, 0.0022903, -0.0002166, 0.0002212
5: 0.0072754, 0.0103782, 0.0072637, 0.0104125, -0.0014079, 0.0014375
6: -0.0010933, -0.0003057, -0.0011020, -0.0003028, -0.0003649, 0.0003573
7: -0.0059663, -0.0039287, -0.0059888, -0.0039210, -0.0009440, 0.0009245
8: -0.0027017, -0.0016302, -0.0027136, -0.0016262, -0.0004964, 0.0004862
9: 0.0000264, 0.0012690, 0.0000218, 0.0012827, -0.0005638, 0.0005756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005760, upper bound: 0.0005281
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005526
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905826, 0.9924507, 0.9906368, 0.9925102, -0.0009788, 0.0008595
1: -0.0036105, -0.0031451, -0.0035970, -0.0031302, -0.0002439, 0.0002142
2: 0.0066131, 0.0090798, 0.0065345, 0.0090084, -0.0011350, 0.0012925
3: -0.0054059, -0.0042831, -0.0053734, -0.0042474, -0.0005883, 0.0005166
4: 0.0018078, 0.0022853, 0.0017926, 0.0022714, -0.0002197, 0.0002502
5: 0.0072770, 0.0103794, 0.0071782, 0.0102897, -0.0014275, 0.0016257
6: -0.0010936, -0.0003062, -0.0010708, -0.0002811, -0.0004126, 0.0003623
7: -0.0059671, -0.0039297, -0.0059081, -0.0038648, -0.0010676, 0.0009374
8: -0.0027022, -0.0016308, -0.0026712, -0.0015966, -0.0005614, 0.0004930
9: 0.0000271, 0.0012694, -0.0000125, 0.0012335, -0.0005716, 0.0006510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0005345
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005595
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905826, 0.9924507, 0.9905733, 0.9924529, -0.0008576, 0.0008676
1: -0.0036105, -0.0031451, -0.0036128, -0.0031445, -0.0002137, 0.0002162
2: 0.0066131, 0.0090798, 0.0066102, 0.0090921, -0.0011456, 0.0011325
3: -0.0054059, -0.0042831, -0.0054115, -0.0042818, -0.0005155, 0.0005214
4: 0.0018078, 0.0022853, 0.0018073, 0.0022876, -0.0002217, 0.0002192
5: 0.0072770, 0.0103794, 0.0072734, 0.0103949, -0.0014409, 0.0014244
6: -0.0010936, -0.0003062, -0.0010975, -0.0003052, -0.0003615, 0.0003657
7: -0.0059671, -0.0039297, -0.0059772, -0.0039273, -0.0009354, 0.0009462
8: -0.0027022, -0.0016308, -0.0027075, -0.0016295, -0.0004919, 0.0004976
9: 0.0000271, 0.0012694, 0.0000256, 0.0012756, -0.0005770, 0.0005704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0005345
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005595
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905832, 0.9924527, 0.9906107, 0.9925222, -0.0009837, 0.0008731
1: -0.0036104, -0.0031446, -0.0036035, -0.0031272, -0.0002451, 0.0002176
2: 0.0066105, 0.0090790, 0.0065186, 0.0090428, -0.0011529, 0.0012990
3: -0.0054055, -0.0042819, -0.0053890, -0.0042401, -0.0005913, 0.0005248
4: 0.0018073, 0.0022851, 0.0017896, 0.0022781, -0.0002231, 0.0002514
5: 0.0072737, 0.0103784, 0.0071581, 0.0103329, -0.0014501, 0.0016338
6: -0.0010933, -0.0003053, -0.0010818, -0.0002760, -0.0004147, 0.0003680
7: -0.0059664, -0.0039275, -0.0059365, -0.0038517, -0.0010729, 0.0009522
8: -0.0027018, -0.0016296, -0.0026861, -0.0015897, -0.0005642, 0.0005008
9: 0.0000258, 0.0012690, -0.0000205, 0.0012508, -0.0005807, 0.0006543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005832, upper bound: 0.0005281
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005527
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905832, 0.9924527, 0.9905467, 0.9924678, -0.0008710, 0.0008739
1: -0.0036104, -0.0031446, -0.0036195, -0.0031408, -0.0002170, 0.0002177
2: 0.0066105, 0.0090790, 0.0065905, 0.0091273, -0.0011539, 0.0011501
3: -0.0054055, -0.0042819, -0.0054275, -0.0042728, -0.0005235, 0.0005252
4: 0.0018073, 0.0022851, 0.0018035, 0.0022945, -0.0002233, 0.0002226
5: 0.0072737, 0.0103784, 0.0072486, 0.0104392, -0.0014513, 0.0014465
6: -0.0010933, -0.0003053, -0.0011088, -0.0002989, -0.0003671, 0.0003684
7: -0.0059664, -0.0039275, -0.0060063, -0.0039111, -0.0009499, 0.0009531
8: -0.0027018, -0.0016296, -0.0027228, -0.0016209, -0.0004995, 0.0005012
9: 0.0000258, 0.0012690, 0.0000157, 0.0012934, -0.0005812, 0.0005792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005832, upper bound: 0.0005281
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005527
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905825, 0.9924520, 0.9906202, 0.9925196, -0.0009839, 0.0008878
1: -0.0036105, -0.0031447, -0.0036011, -0.0031279, -0.0002452, 0.0002212
2: 0.0066114, 0.0090799, 0.0065222, 0.0090302, -0.0011724, 0.0012992
3: -0.0054059, -0.0042823, -0.0053833, -0.0042417, -0.0005914, 0.0005336
4: 0.0018075, 0.0022853, 0.0017902, 0.0022757, -0.0002269, 0.0002515
5: 0.0072748, 0.0103796, 0.0071626, 0.0103171, -0.0014745, 0.0016341
6: -0.0010936, -0.0003056, -0.0010778, -0.0002771, -0.0004148, 0.0003742
7: -0.0059672, -0.0039283, -0.0059261, -0.0038546, -0.0010731, 0.0009683
8: -0.0027022, -0.0016300, -0.0026806, -0.0015913, -0.0005643, 0.0005092
9: 0.0000262, 0.0012695, -0.0000187, 0.0012445, -0.0005905, 0.0006544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005770, upper bound: 0.0005345
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005596
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905825, 0.9924520, 0.9905558, 0.9924642, -0.0008636, 0.0008923
1: -0.0036105, -0.0031447, -0.0036172, -0.0031417, -0.0002152, 0.0002223
2: 0.0066114, 0.0090799, 0.0065952, 0.0091154, -0.0011782, 0.0011404
3: -0.0054059, -0.0042823, -0.0054221, -0.0042750, -0.0005190, 0.0005363
4: 0.0018075, 0.0022853, 0.0018044, 0.0022922, -0.0002280, 0.0002207
5: 0.0072748, 0.0103796, 0.0072544, 0.0104242, -0.0014819, 0.0014343
6: -0.0010936, -0.0003056, -0.0011049, -0.0003004, -0.0003640, 0.0003761
7: -0.0059672, -0.0039283, -0.0059965, -0.0039149, -0.0009419, 0.0009732
8: -0.0027022, -0.0016300, -0.0027176, -0.0016230, -0.0004953, 0.0005118
9: 0.0000262, 0.0012695, 0.0000180, 0.0012874, -0.0005934, 0.0005743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005770, upper bound: 0.0005345
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005596
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906667, 0.9925388, 0.9906441, 0.9924694, -0.0007876, 0.0008712
1: -0.0035896, -0.0031231, -0.0035952, -0.0031404, -0.0001963, 0.0002171
2: 0.0064968, 0.0089689, 0.0065884, 0.0089986, -0.0011504, 0.0010400
3: -0.0053554, -0.0042302, -0.0053689, -0.0042719, -0.0004734, 0.0005236
4: 0.0017853, 0.0022638, 0.0018031, 0.0022696, -0.0002226, 0.0002013
5: 0.0071307, 0.0102399, 0.0072459, 0.0102774, -0.0014469, 0.0013081
6: -0.0010582, -0.0002690, -0.0010677, -0.0002983, -0.0003320, 0.0003672
7: -0.0058754, -0.0038336, -0.0059000, -0.0039093, -0.0008590, 0.0009501
8: -0.0026540, -0.0015802, -0.0026669, -0.0016200, -0.0004517, 0.0004997
9: -0.0000315, 0.0012136, 0.0000146, 0.0012286, -0.0005794, 0.0005238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005610
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005253, upper bound: 0.0005866
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906659, 0.9925395, 0.9906546, 0.9924650, -0.0007976, 0.0008991
1: -0.0035898, -0.0031229, -0.0035926, -0.0031415, -0.0001987, 0.0002240
2: 0.0064958, 0.0089699, 0.0065942, 0.0089849, -0.0011872, 0.0010532
3: -0.0053558, -0.0042297, -0.0053627, -0.0042745, -0.0004794, 0.0005404
4: 0.0017851, 0.0022640, 0.0018042, 0.0022669, -0.0002298, 0.0002038
5: 0.0071294, 0.0102412, 0.0072532, 0.0102601, -0.0014932, 0.0013246
6: -0.0010585, -0.0002687, -0.0010633, -0.0003001, -0.0003362, 0.0003790
7: -0.0058763, -0.0038328, -0.0058887, -0.0039141, -0.0008699, 0.0009806
8: -0.0026544, -0.0015798, -0.0026609, -0.0016225, -0.0004575, 0.0005157
9: -0.0000320, 0.0012141, 0.0000176, 0.0012216, -0.0005979, 0.0005304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005695
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005221, upper bound: 0.0005964
time: 0.67 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906665, 0.9925401, 0.9906272, 0.9924768, -0.0007903, 0.0009038
1: -0.0035896, -0.0031228, -0.0035994, -0.0031385, -0.0001969, 0.0002252
2: 0.0064950, 0.0089691, 0.0065785, 0.0090210, -0.0011934, 0.0010435
3: -0.0053555, -0.0042294, -0.0053791, -0.0042674, -0.0004750, 0.0005432
4: 0.0017850, 0.0022638, 0.0018011, 0.0022739, -0.0002310, 0.0002020
5: 0.0071284, 0.0102403, 0.0072335, 0.0103055, -0.0015010, 0.0013125
6: -0.0010583, -0.0002684, -0.0010748, -0.0002951, -0.0003331, 0.0003810
7: -0.0058757, -0.0038322, -0.0059185, -0.0039012, -0.0008619, 0.0009857
8: -0.0026541, -0.0015794, -0.0026766, -0.0016157, -0.0004533, 0.0005184
9: -0.0000324, 0.0012137, 0.0000097, 0.0012398, -0.0006011, 0.0005256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005251, upper bound: 0.0005610
time: 0.64 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005322, upper bound: 0.0005865
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906657, 0.9925408, 0.9906373, 0.9924735, -0.0008020, 0.0009259
1: -0.0035898, -0.0031226, -0.0035969, -0.0031393, -0.0001998, 0.0002307
2: 0.0064940, 0.0089702, 0.0065828, 0.0090077, -0.0012226, 0.0010591
3: -0.0053560, -0.0042289, -0.0053730, -0.0042693, -0.0004820, 0.0005565
4: 0.0017848, 0.0022641, 0.0018020, 0.0022713, -0.0002366, 0.0002050
5: 0.0071272, 0.0102416, 0.0072389, 0.0102887, -0.0015377, 0.0013320
6: -0.0010586, -0.0002681, -0.0010706, -0.0002965, -0.0003381, 0.0003903
7: -0.0058765, -0.0038314, -0.0059075, -0.0039047, -0.0008747, 0.0010098
8: -0.0026546, -0.0015790, -0.0026708, -0.0016176, -0.0004600, 0.0005310
9: -0.0000329, 0.0012142, 0.0000118, 0.0012331, -0.0006158, 0.0005334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005230, upper bound: 0.0005695
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0005965
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906338, 0.9925571, 0.9906321, 0.9924694, -0.0007915, 0.0008929
1: -0.0035978, -0.0031185, -0.0035982, -0.0031404, -0.0001972, 0.0002225
2: 0.0064725, 0.0090123, 0.0065883, 0.0090145, -0.0011791, 0.0010452
3: -0.0053751, -0.0042191, -0.0053761, -0.0042718, -0.0004757, 0.0005367
4: 0.0017806, 0.0022722, 0.0018030, 0.0022726, -0.0002282, 0.0002023
5: 0.0071001, 0.0102946, 0.0072459, 0.0102973, -0.0014830, 0.0013146
6: -0.0010720, -0.0002613, -0.0010727, -0.0002982, -0.0003337, 0.0003764
7: -0.0059113, -0.0038136, -0.0059131, -0.0039093, -0.0008633, 0.0009739
8: -0.0026728, -0.0015697, -0.0026738, -0.0016200, -0.0004540, 0.0005121
9: -0.0000437, 0.0012354, 0.0000146, 0.0012366, -0.0005939, 0.0005264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005592, upper bound: 0.0005610
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005664, upper bound: 0.0005866
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906329, 0.9925569, 0.9906422, 0.9924650, -0.0008083, 0.0009143
1: -0.0035980, -0.0031186, -0.0035957, -0.0031415, -0.0002014, 0.0002278
2: 0.0064729, 0.0090134, 0.0065941, 0.0090011, -0.0012073, 0.0010673
3: -0.0053756, -0.0042193, -0.0053700, -0.0042745, -0.0004858, 0.0005495
4: 0.0017807, 0.0022724, 0.0018042, 0.0022700, -0.0002337, 0.0002066
5: 0.0071006, 0.0102960, 0.0072532, 0.0102805, -0.0015185, 0.0013424
6: -0.0010724, -0.0002614, -0.0010685, -0.0003001, -0.0003407, 0.0003854
7: -0.0059123, -0.0038139, -0.0059021, -0.0039141, -0.0008816, 0.0009972
8: -0.0026733, -0.0015698, -0.0026680, -0.0016225, -0.0004636, 0.0005244
9: -0.0000436, 0.0012360, 0.0000175, 0.0012298, -0.0006081, 0.0005376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005522, upper bound: 0.0005695
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005964
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906335, 0.9925585, 0.9906156, 0.9924769, -0.0007962, 0.0009209
1: -0.0035978, -0.0031182, -0.0036023, -0.0031385, -0.0001984, 0.0002295
2: 0.0064706, 0.0090126, 0.0065784, 0.0090363, -0.0012160, 0.0010514
3: -0.0053753, -0.0042182, -0.0053861, -0.0042673, -0.0004785, 0.0005535
4: 0.0017803, 0.0022723, 0.0018011, 0.0022769, -0.0002354, 0.0002035
5: 0.0070977, 0.0102950, 0.0072334, 0.0103248, -0.0015294, 0.0013224
6: -0.0010721, -0.0002607, -0.0010797, -0.0002951, -0.0003356, 0.0003882
7: -0.0059116, -0.0038120, -0.0059312, -0.0039011, -0.0008684, 0.0010043
8: -0.0026730, -0.0015688, -0.0026833, -0.0016157, -0.0004567, 0.0005282
9: -0.0000447, 0.0012356, 0.0000096, 0.0012475, -0.0006124, 0.0005295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005610
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005697, upper bound: 0.0005866
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906327, 0.9925586, 0.9906256, 0.9924737, -0.0008137, 0.0009388
1: -0.0035980, -0.0031182, -0.0035998, -0.0031393, -0.0002028, 0.0002339
2: 0.0064706, 0.0090137, 0.0065827, 0.0090230, -0.0012397, 0.0010745
3: -0.0053757, -0.0042183, -0.0053800, -0.0042693, -0.0004891, 0.0005643
4: 0.0017803, 0.0022725, 0.0018020, 0.0022743, -0.0002399, 0.0002080
5: 0.0070977, 0.0102963, 0.0072388, 0.0103081, -0.0015592, 0.0013514
6: -0.0010725, -0.0002607, -0.0010755, -0.0002965, -0.0003430, 0.0003958
7: -0.0059124, -0.0038120, -0.0059202, -0.0039046, -0.0008875, 0.0010239
8: -0.0026734, -0.0015688, -0.0026775, -0.0016176, -0.0004667, 0.0005385
9: -0.0000447, 0.0012361, 0.0000118, 0.0012408, -0.0006244, 0.0005412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005695
time: 0.68 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005633, upper bound: 0.0005965
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906613, 0.9924955, 0.9905959, 0.9924515, -0.0008237, 0.0009592
1: -0.0035909, -0.0031339, -0.0036072, -0.0031448, -0.0002052, 0.0002390
2: 0.0065540, 0.0089761, 0.0066120, 0.0090624, -0.0012666, 0.0010877
3: -0.0053586, -0.0042562, -0.0053979, -0.0042826, -0.0004951, 0.0005765
4: 0.0017964, 0.0022652, 0.0018076, 0.0022819, -0.0002451, 0.0002105
5: 0.0072026, 0.0102490, 0.0072756, 0.0103575, -0.0015930, 0.0013680
6: -0.0010605, -0.0002873, -0.0010880, -0.0003058, -0.0003472, 0.0004043
7: -0.0058814, -0.0038809, -0.0059527, -0.0039288, -0.0008984, 0.0010461
8: -0.0026571, -0.0016051, -0.0026946, -0.0016303, -0.0004724, 0.0005501
9: -0.0000027, 0.0012172, 0.0000265, 0.0012607, -0.0006379, 0.0005478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005828
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005278, upper bound: 0.0005980
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906704, 0.9924922, 0.9905951, 0.9924505, -0.0008414, 0.0009619
1: -0.0035886, -0.0031347, -0.0036074, -0.0031451, -0.0002096, 0.0002397
2: 0.0065583, 0.0089638, 0.0066133, 0.0090634, -0.0012702, 0.0011110
3: -0.0053530, -0.0042582, -0.0053984, -0.0042832, -0.0005057, 0.0005782
4: 0.0017972, 0.0022628, 0.0018079, 0.0022821, -0.0002459, 0.0002150
5: 0.0072080, 0.0102335, 0.0072772, 0.0103588, -0.0015976, 0.0013974
6: -0.0010565, -0.0002886, -0.0010883, -0.0003062, -0.0003547, 0.0004055
7: -0.0058712, -0.0038844, -0.0059535, -0.0039299, -0.0009176, 0.0010491
8: -0.0026518, -0.0016069, -0.0026950, -0.0016308, -0.0004826, 0.0005517
9: -0.0000005, 0.0012110, 0.0000272, 0.0012612, -0.0006398, 0.0005596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005738
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005287, upper bound: 0.0005909
time: 0.67 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906445, 0.9925039, 0.9905958, 0.9924526, -0.0008547, 0.0009683
1: -0.0035951, -0.0031318, -0.0036072, -0.0031446, -0.0002130, 0.0002413
2: 0.0065428, 0.0089981, 0.0066106, 0.0090625, -0.0012786, 0.0011287
3: -0.0053687, -0.0042511, -0.0053980, -0.0042820, -0.0005137, 0.0005820
4: 0.0017942, 0.0022695, 0.0018074, 0.0022819, -0.0002475, 0.0002184
5: 0.0071886, 0.0102767, 0.0072739, 0.0103577, -0.0016082, 0.0014195
6: -0.0010675, -0.0002837, -0.0010881, -0.0003054, -0.0003603, 0.0004082
7: -0.0058996, -0.0038717, -0.0059528, -0.0039277, -0.0009322, 0.0010561
8: -0.0026667, -0.0016002, -0.0026946, -0.0016297, -0.0004902, 0.0005554
9: -0.0000083, 0.0012283, 0.0000258, 0.0012607, -0.0006440, 0.0005684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005863
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0006014
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906534, 0.9925030, 0.9905950, 0.9924518, -0.0008689, 0.0009680
1: -0.0035929, -0.0031320, -0.0036074, -0.0031448, -0.0002165, 0.0002412
2: 0.0065440, 0.0089864, 0.0066115, 0.0090635, -0.0012783, 0.0011473
3: -0.0053633, -0.0042517, -0.0053984, -0.0042824, -0.0005222, 0.0005818
4: 0.0017945, 0.0022672, 0.0018075, 0.0022821, -0.0002474, 0.0002221
5: 0.0071901, 0.0102620, 0.0072750, 0.0103589, -0.0016078, 0.0014430
6: -0.0010638, -0.0002841, -0.0010884, -0.0003056, -0.0003663, 0.0004081
7: -0.0058899, -0.0038727, -0.0059536, -0.0039284, -0.0009476, 0.0010558
8: -0.0026616, -0.0016007, -0.0026951, -0.0016301, -0.0004983, 0.0005552
9: -0.0000077, 0.0012224, 0.0000263, 0.0012612, -0.0006438, 0.0005779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005797
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005956
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906259, 0.9925148, 0.9905834, 0.9924517, -0.0008425, 0.0009784
1: -0.0035997, -0.0031291, -0.0036103, -0.0031448, -0.0002099, 0.0002438
2: 0.0065284, 0.0090226, 0.0066118, 0.0090788, -0.0012919, 0.0011125
3: -0.0053798, -0.0042446, -0.0054054, -0.0042825, -0.0005064, 0.0005880
4: 0.0017915, 0.0022742, 0.0018076, 0.0022851, -0.0002501, 0.0002153
5: 0.0071705, 0.0103075, 0.0072754, 0.0103782, -0.0016249, 0.0013993
6: -0.0010753, -0.0002791, -0.0010933, -0.0003057, -0.0003552, 0.0004124
7: -0.0059198, -0.0038598, -0.0059663, -0.0039287, -0.0009189, 0.0010671
8: -0.0026773, -0.0015940, -0.0027017, -0.0016302, -0.0004832, 0.0005612
9: -0.0000156, 0.0012406, 0.0000264, 0.0012690, -0.0006507, 0.0005603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005828
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005527, upper bound: 0.0005979
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906368, 0.9925102, 0.9905826, 0.9924507, -0.0008595, 0.0009788
1: -0.0035970, -0.0031302, -0.0036105, -0.0031451, -0.0002142, 0.0002439
2: 0.0065345, 0.0090084, 0.0066131, 0.0090798, -0.0012925, 0.0011350
3: -0.0053734, -0.0042474, -0.0054059, -0.0042831, -0.0005166, 0.0005883
4: 0.0017926, 0.0022714, 0.0018078, 0.0022853, -0.0002502, 0.0002197
5: 0.0071782, 0.0102897, 0.0072770, 0.0103794, -0.0016257, 0.0014275
6: -0.0010708, -0.0002811, -0.0010936, -0.0003062, -0.0003623, 0.0004126
7: -0.0059081, -0.0038648, -0.0059671, -0.0039297, -0.0009374, 0.0010676
8: -0.0026712, -0.0015966, -0.0027022, -0.0016308, -0.0004930, 0.0005614
9: -0.0000125, 0.0012335, 0.0000271, 0.0012694, -0.0006510, 0.0005716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005744
time: 0.68 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0005910
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906107, 0.9925222, 0.9905832, 0.9924527, -0.0008731, 0.0009837
1: -0.0036035, -0.0031272, -0.0036104, -0.0031446, -0.0002176, 0.0002451
2: 0.0065186, 0.0090428, 0.0066105, 0.0090790, -0.0012990, 0.0011529
3: -0.0053890, -0.0042401, -0.0054055, -0.0042819, -0.0005248, 0.0005913
4: 0.0017896, 0.0022781, 0.0018073, 0.0022851, -0.0002514, 0.0002231
5: 0.0071581, 0.0103329, 0.0072737, 0.0103784, -0.0016338, 0.0014501
6: -0.0010818, -0.0002760, -0.0010933, -0.0003053, -0.0003680, 0.0004147
7: -0.0059365, -0.0038517, -0.0059664, -0.0039275, -0.0009522, 0.0010729
8: -0.0026861, -0.0015897, -0.0027018, -0.0016296, -0.0005008, 0.0005642
9: -0.0000205, 0.0012508, 0.0000258, 0.0012690, -0.0006543, 0.0005807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005864
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005527, upper bound: 0.0006015
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906202, 0.9925196, 0.9905825, 0.9924520, -0.0008878, 0.0009839
1: -0.0036011, -0.0031279, -0.0036105, -0.0031447, -0.0002212, 0.0002452
2: 0.0065222, 0.0090302, 0.0066114, 0.0090799, -0.0012992, 0.0011724
3: -0.0053833, -0.0042417, -0.0054059, -0.0042823, -0.0005336, 0.0005914
4: 0.0017902, 0.0022757, 0.0018075, 0.0022853, -0.0002515, 0.0002269
5: 0.0071626, 0.0103171, 0.0072748, 0.0103796, -0.0016341, 0.0014745
6: -0.0010778, -0.0002771, -0.0010936, -0.0003056, -0.0003742, 0.0004148
7: -0.0059261, -0.0038546, -0.0059672, -0.0039283, -0.0009683, 0.0010731
8: -0.0026806, -0.0015913, -0.0027022, -0.0016300, -0.0005092, 0.0005643
9: -0.0000187, 0.0012445, 0.0000262, 0.0012695, -0.0006544, 0.0005905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005804
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005596, upper bound: 0.0005956
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906667, 0.9925388, 0.9906254, 0.9925148, -0.0007963, 0.0008466
1: -0.0035896, -0.0031231, -0.0035999, -0.0031291, -0.0001984, 0.0002109
2: 0.0064968, 0.0089689, 0.0065284, 0.0090234, -0.0011179, 0.0010514
3: -0.0053554, -0.0042302, -0.0053802, -0.0042446, -0.0004786, 0.0005088
4: 0.0017853, 0.0022638, 0.0017914, 0.0022743, -0.0002164, 0.0002035
5: 0.0071307, 0.0102399, 0.0071705, 0.0103085, -0.0014060, 0.0013224
6: -0.0010582, -0.0002690, -0.0010756, -0.0002791, -0.0003356, 0.0003569
7: -0.0058754, -0.0038336, -0.0059205, -0.0038598, -0.0008684, 0.0009233
8: -0.0026540, -0.0015802, -0.0026777, -0.0015940, -0.0004567, 0.0004856
9: -0.0000315, 0.0012136, -0.0000156, 0.0012410, -0.0005630, 0.0005296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005610
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005866
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906659, 0.9925395, 0.9906353, 0.9925101, -0.0007886, 0.0008760
1: -0.0035898, -0.0031229, -0.0035974, -0.0031302, -0.0001965, 0.0002183
2: 0.0064958, 0.0089699, 0.0065345, 0.0090103, -0.0011567, 0.0010413
3: -0.0053558, -0.0042297, -0.0053742, -0.0042473, -0.0004739, 0.0005265
4: 0.0017851, 0.0022640, 0.0017926, 0.0022718, -0.0002239, 0.0002015
5: 0.0071294, 0.0102412, 0.0071782, 0.0102921, -0.0014549, 0.0013096
6: -0.0010585, -0.0002687, -0.0010714, -0.0002811, -0.0003324, 0.0003693
7: -0.0058763, -0.0038328, -0.0059097, -0.0038648, -0.0008600, 0.0009554
8: -0.0026544, -0.0015798, -0.0026720, -0.0015966, -0.0004523, 0.0005024
9: -0.0000320, 0.0012141, -0.0000125, 0.0012344, -0.0005826, 0.0005244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005695
time: 0.68 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005223, upper bound: 0.0005965
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906665, 0.9925401, 0.9906094, 0.9925222, -0.0008002, 0.0008788
1: -0.0035896, -0.0031228, -0.0036038, -0.0031272, -0.0001994, 0.0002190
2: 0.0064950, 0.0089691, 0.0065186, 0.0090445, -0.0011605, 0.0010567
3: -0.0053555, -0.0042294, -0.0053898, -0.0042401, -0.0004809, 0.0005282
4: 0.0017850, 0.0022638, 0.0017895, 0.0022784, -0.0002246, 0.0002045
5: 0.0071284, 0.0102403, 0.0071581, 0.0103351, -0.0014596, 0.0013290
6: -0.0010583, -0.0002684, -0.0010823, -0.0002760, -0.0003373, 0.0003705
7: -0.0058757, -0.0038322, -0.0059379, -0.0038516, -0.0008727, 0.0009585
8: -0.0026541, -0.0015794, -0.0026868, -0.0015897, -0.0004590, 0.0005040
9: -0.0000324, 0.0012137, -0.0000205, 0.0012517, -0.0005845, 0.0005322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005251, upper bound: 0.0005610
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005324, upper bound: 0.0005866
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906657, 0.9925408, 0.9906179, 0.9925196, -0.0007930, 0.0009029
1: -0.0035898, -0.0031226, -0.0036017, -0.0031279, -0.0001976, 0.0002250
2: 0.0064940, 0.0089702, 0.0065221, 0.0090332, -0.0011922, 0.0010472
3: -0.0053560, -0.0042289, -0.0053847, -0.0042417, -0.0004766, 0.0005426
4: 0.0017848, 0.0022641, 0.0017902, 0.0022763, -0.0002307, 0.0002027
5: 0.0071272, 0.0102416, 0.0071626, 0.0103209, -0.0014995, 0.0013171
6: -0.0010586, -0.0002681, -0.0010787, -0.0002771, -0.0003343, 0.0003806
7: -0.0058765, -0.0038314, -0.0059286, -0.0038546, -0.0008649, 0.0009847
8: -0.0026546, -0.0015790, -0.0026819, -0.0015912, -0.0004548, 0.0005178
9: -0.0000329, 0.0012142, -0.0000187, 0.0012460, -0.0006005, 0.0005274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005230, upper bound: 0.0005695
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005304, upper bound: 0.0005965
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906338, 0.9925571, 0.9906130, 0.9925148, -0.0007998, 0.0008574
1: -0.0035978, -0.0031185, -0.0036029, -0.0031291, -0.0001993, 0.0002136
2: 0.0064725, 0.0090123, 0.0065284, 0.0090397, -0.0011321, 0.0010561
3: -0.0053751, -0.0042191, -0.0053876, -0.0042445, -0.0004807, 0.0005153
4: 0.0017806, 0.0022722, 0.0017914, 0.0022775, -0.0002191, 0.0002044
5: 0.0071001, 0.0102946, 0.0071704, 0.0103290, -0.0014239, 0.0013283
6: -0.0010720, -0.0002613, -0.0010808, -0.0002791, -0.0003371, 0.0003614
7: -0.0059113, -0.0038136, -0.0059339, -0.0038597, -0.0008723, 0.0009351
8: -0.0026728, -0.0015697, -0.0026847, -0.0015939, -0.0004587, 0.0004917
9: -0.0000437, 0.0012354, -0.0000156, 0.0012492, -0.0005702, 0.0005319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005592, upper bound: 0.0005610
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005664, upper bound: 0.0005866
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906329, 0.9925569, 0.9906234, 0.9925102, -0.0007971, 0.0008817
1: -0.0035980, -0.0031186, -0.0036004, -0.0031302, -0.0001986, 0.0002197
2: 0.0064729, 0.0090134, 0.0065344, 0.0090260, -0.0011642, 0.0010526
3: -0.0053756, -0.0042193, -0.0053814, -0.0042473, -0.0004791, 0.0005299
4: 0.0017807, 0.0022724, 0.0017926, 0.0022748, -0.0002253, 0.0002037
5: 0.0071006, 0.0102960, 0.0071781, 0.0103118, -0.0014643, 0.0013238
6: -0.0010724, -0.0002614, -0.0010764, -0.0002810, -0.0003360, 0.0003717
7: -0.0059123, -0.0038139, -0.0059226, -0.0038647, -0.0008694, 0.0009616
8: -0.0026733, -0.0015698, -0.0026788, -0.0015966, -0.0004572, 0.0005057
9: -0.0000436, 0.0012360, -0.0000125, 0.0012423, -0.0005864, 0.0005301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005522, upper bound: 0.0005694
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005965
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906335, 0.9925585, 0.9905972, 0.9925224, -0.0008064, 0.0008831
1: -0.0035978, -0.0031182, -0.0036069, -0.0031272, -0.0002009, 0.0002201
2: 0.0064706, 0.0090126, 0.0065184, 0.0090605, -0.0011662, 0.0010649
3: -0.0053753, -0.0042182, -0.0053971, -0.0042400, -0.0004847, 0.0005308
4: 0.0017803, 0.0022723, 0.0017895, 0.0022815, -0.0002257, 0.0002061
5: 0.0070977, 0.0102950, 0.0071579, 0.0103552, -0.0014667, 0.0013393
6: -0.0010721, -0.0002607, -0.0010874, -0.0002759, -0.0003399, 0.0003723
7: -0.0059116, -0.0038120, -0.0059511, -0.0038515, -0.0008795, 0.0009632
8: -0.0026730, -0.0015688, -0.0026938, -0.0015896, -0.0004625, 0.0005065
9: -0.0000447, 0.0012356, -0.0000206, 0.0012597, -0.0005874, 0.0005363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005610
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005698, upper bound: 0.0005866
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906327, 0.9925586, 0.9906062, 0.9925196, -0.0008032, 0.0009040
1: -0.0035980, -0.0031182, -0.0036046, -0.0031279, -0.0002001, 0.0002253
2: 0.0064706, 0.0090137, 0.0065220, 0.0090487, -0.0011938, 0.0010607
3: -0.0053757, -0.0042183, -0.0053917, -0.0042417, -0.0004828, 0.0005434
4: 0.0017803, 0.0022725, 0.0017902, 0.0022792, -0.0002311, 0.0002053
5: 0.0070977, 0.0102963, 0.0071624, 0.0103403, -0.0015015, 0.0013340
6: -0.0010725, -0.0002607, -0.0010836, -0.0002771, -0.0003386, 0.0003811
7: -0.0059124, -0.0038120, -0.0059413, -0.0038545, -0.0008760, 0.0009860
8: -0.0026734, -0.0015688, -0.0026886, -0.0015912, -0.0004607, 0.0005185
9: -0.0000447, 0.0012361, -0.0000188, 0.0012538, -0.0006013, 0.0005342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005695
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005634, upper bound: 0.0005965
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906613, 0.9924955, 0.9905686, 0.9925020, -0.0008107, 0.0009328
1: -0.0035909, -0.0031339, -0.0036140, -0.0031323, -0.0002020, 0.0002324
2: 0.0065540, 0.0089761, 0.0065454, 0.0090984, -0.0012317, 0.0010706
3: -0.0053586, -0.0042562, -0.0054143, -0.0042523, -0.0004873, 0.0005606
4: 0.0017964, 0.0022652, 0.0017947, 0.0022889, -0.0002384, 0.0002072
5: 0.0072026, 0.0102490, 0.0071918, 0.0104028, -0.0015492, 0.0013465
6: -0.0010605, -0.0002873, -0.0010995, -0.0002845, -0.0003418, 0.0003932
7: -0.0058814, -0.0038809, -0.0059824, -0.0038738, -0.0008842, 0.0010173
8: -0.0026571, -0.0016051, -0.0027102, -0.0016013, -0.0004650, 0.0005350
9: -0.0000027, 0.0012172, -0.0000070, 0.0012788, -0.0006204, 0.0005392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005828
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005279, upper bound: 0.0005979
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906704, 0.9924922, 0.9905677, 0.9925012, -0.0008374, 0.0009378
1: -0.0035886, -0.0031347, -0.0036142, -0.0031325, -0.0002087, 0.0002337
2: 0.0065583, 0.0089638, 0.0065464, 0.0090994, -0.0012384, 0.0011058
3: -0.0053530, -0.0042582, -0.0054148, -0.0042527, -0.0005033, 0.0005637
4: 0.0017972, 0.0022628, 0.0017949, 0.0022891, -0.0002397, 0.0002140
5: 0.0072080, 0.0102335, 0.0071931, 0.0104041, -0.0015576, 0.0013908
6: -0.0010565, -0.0002886, -0.0010998, -0.0002848, -0.0003530, 0.0003953
7: -0.0058712, -0.0038844, -0.0059833, -0.0038746, -0.0009133, 0.0010228
8: -0.0026518, -0.0016069, -0.0027107, -0.0016018, -0.0004803, 0.0005379
9: -0.0000005, 0.0012110, -0.0000065, 0.0012793, -0.0006237, 0.0005569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005738
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005289, upper bound: 0.0005909
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906445, 0.9925039, 0.9905685, 0.9925030, -0.0008431, 0.0009405
1: -0.0035951, -0.0031318, -0.0036140, -0.0031320, -0.0002101, 0.0002343
2: 0.0065428, 0.0089981, 0.0065440, 0.0090985, -0.0012419, 0.0011132
3: -0.0053687, -0.0042511, -0.0054144, -0.0042517, -0.0005067, 0.0005652
4: 0.0017942, 0.0022695, 0.0017945, 0.0022889, -0.0002404, 0.0002155
5: 0.0071886, 0.0102767, 0.0071900, 0.0104030, -0.0015619, 0.0014002
6: -0.0010675, -0.0002837, -0.0010996, -0.0002841, -0.0003554, 0.0003964
7: -0.0058996, -0.0038717, -0.0059825, -0.0038726, -0.0009195, 0.0010257
8: -0.0026667, -0.0016002, -0.0027103, -0.0016007, -0.0004835, 0.0005394
9: -0.0000083, 0.0012283, -0.0000077, 0.0012789, -0.0006255, 0.0005607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005863
time: 0.73 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005282, upper bound: 0.0006015
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906534, 0.9925030, 0.9905677, 0.9925028, -0.0008669, 0.0009446
1: -0.0035929, -0.0031320, -0.0036142, -0.0031320, -0.0002160, 0.0002354
2: 0.0065440, 0.0089864, 0.0065442, 0.0090995, -0.0012473, 0.0011447
3: -0.0053633, -0.0042517, -0.0054148, -0.0042518, -0.0005210, 0.0005677
4: 0.0017945, 0.0022672, 0.0017945, 0.0022891, -0.0002414, 0.0002216
5: 0.0071901, 0.0102620, 0.0071904, 0.0104042, -0.0015687, 0.0014397
6: -0.0010638, -0.0002841, -0.0010999, -0.0002842, -0.0003654, 0.0003982
7: -0.0058899, -0.0038727, -0.0059833, -0.0038728, -0.0009455, 0.0010302
8: -0.0026616, -0.0016007, -0.0027107, -0.0016008, -0.0004972, 0.0005418
9: -0.0000077, 0.0012224, -0.0000076, 0.0012794, -0.0006282, 0.0005765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005797
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005292, upper bound: 0.0005956
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9906259, 0.9925148, 0.9905570, 0.9925020, -0.0008265, 0.0009425
1: -0.0035997, -0.0031291, -0.0036169, -0.0031322, -0.0002059, 0.0002348
2: 0.0065284, 0.0090226, 0.0065452, 0.0091138, -0.0012445, 0.0010913
3: -0.0053798, -0.0042446, -0.0054213, -0.0042522, -0.0004967, 0.0005665
4: 0.0017915, 0.0022742, 0.0017947, 0.0022918, -0.0002409, 0.0002112
5: 0.0071705, 0.0103075, 0.0071916, 0.0104222, -0.0015653, 0.0013726
6: -0.0010753, -0.0002791, -0.0011044, -0.0002845, -0.0003484, 0.0003973
7: -0.0059198, -0.0038598, -0.0059951, -0.0038737, -0.0009014, 0.0010279
8: -0.0026773, -0.0015940, -0.0027169, -0.0016013, -0.0004740, 0.0005406
9: -0.0000156, 0.0012406, -0.0000071, 0.0012865, -0.0006268, 0.0005497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005828
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005527, upper bound: 0.0005980
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9906368, 0.9925102, 0.9905562, 0.9925013, -0.0008540, 0.0009460
1: -0.0035970, -0.0031302, -0.0036171, -0.0031324, -0.0002128, 0.0002357
2: 0.0065345, 0.0090084, 0.0065462, 0.0091148, -0.0012491, 0.0011277
3: -0.0053734, -0.0042474, -0.0054218, -0.0042527, -0.0005133, 0.0005685
4: 0.0017926, 0.0022714, 0.0017949, 0.0022920, -0.0002418, 0.0002183
5: 0.0071782, 0.0102897, 0.0071929, 0.0104234, -0.0015711, 0.0014183
6: -0.0010708, -0.0002811, -0.0011047, -0.0002848, -0.0003600, 0.0003988
7: -0.0059081, -0.0038648, -0.0059959, -0.0038745, -0.0009314, 0.0010317
8: -0.0026712, -0.0015966, -0.0027174, -0.0016017, -0.0004898, 0.0005426
9: -0.0000125, 0.0012335, -0.0000066, 0.0012871, -0.0006291, 0.0005680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005744
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0005909
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9906107, 0.9925222, 0.9905569, 0.9925031, -0.0008598, 0.0009464
1: -0.0036035, -0.0031272, -0.0036169, -0.0031320, -0.0002142, 0.0002358
2: 0.0065186, 0.0090428, 0.0065438, 0.0091139, -0.0012497, 0.0011353
3: -0.0053890, -0.0042401, -0.0054214, -0.0042516, -0.0005167, 0.0005688
4: 0.0017896, 0.0022781, 0.0017944, 0.0022919, -0.0002419, 0.0002197
5: 0.0071581, 0.0103329, 0.0071898, 0.0104223, -0.0015718, 0.0014279
6: -0.0010818, -0.0002760, -0.0011045, -0.0002840, -0.0003624, 0.0003989
7: -0.0059365, -0.0038517, -0.0059952, -0.0038725, -0.0009377, 0.0010322
8: -0.0026861, -0.0015897, -0.0027170, -0.0016007, -0.0004931, 0.0005428
9: -0.0000205, 0.0012508, -0.0000078, 0.0012866, -0.0006294, 0.0005718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005863
time: 0.68 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005528, upper bound: 0.0006015
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9906202, 0.9925196, 0.9905561, 0.9925030, -0.0008821, 0.0009504
1: -0.0036011, -0.0031279, -0.0036171, -0.0031320, -0.0002198, 0.0002368
2: 0.0065222, 0.0090302, 0.0065441, 0.0091149, -0.0012549, 0.0011648
3: -0.0053833, -0.0042417, -0.0054218, -0.0042517, -0.0005302, 0.0005712
4: 0.0017902, 0.0022757, 0.0017945, 0.0022921, -0.0002429, 0.0002254
5: 0.0071626, 0.0103171, 0.0071902, 0.0104236, -0.0015784, 0.0014650
6: -0.0010778, -0.0002771, -0.0011048, -0.0002841, -0.0003718, 0.0004006
7: -0.0059261, -0.0038546, -0.0059960, -0.0038727, -0.0009620, 0.0010365
8: -0.0026806, -0.0015913, -0.0027174, -0.0016008, -0.0005059, 0.0005451
9: -0.0000187, 0.0012445, -0.0000077, 0.0012871, -0.0006321, 0.0005867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005804
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005596, upper bound: 0.0005956
time: 0.67 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906032, 0.9924821, 0.9906441, 0.9924694, -0.0008961, 0.0008622
1: -0.0036054, -0.0031372, -0.0035952, -0.0031404, -0.0002233, 0.0002148
2: 0.0065715, 0.0090526, 0.0065884, 0.0089986, -0.0011386, 0.0011832
3: -0.0053935, -0.0042642, -0.0053689, -0.0042719, -0.0005386, 0.0005182
4: 0.0017998, 0.0022800, 0.0018031, 0.0022696, -0.0002204, 0.0002290
5: 0.0072247, 0.0103453, 0.0072459, 0.0102774, -0.0014320, 0.0014882
6: -0.0010849, -0.0002929, -0.0010677, -0.0002983, -0.0003777, 0.0003635
7: -0.0059446, -0.0038954, -0.0059000, -0.0039093, -0.0009773, 0.0009404
8: -0.0026904, -0.0016127, -0.0026669, -0.0016200, -0.0005139, 0.0004945
9: 0.0000061, 0.0012557, 0.0000146, 0.0012286, -0.0005734, 0.0005959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004921, upper bound: 0.0005376
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0005853
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906032, 0.9924821, 0.9905868, 0.9924004, -0.0007890, 0.0008726
1: -0.0036054, -0.0031372, -0.0036095, -0.0031576, -0.0001966, 0.0002174
2: 0.0065715, 0.0090526, 0.0066795, 0.0090742, -0.0011522, 0.0010419
3: -0.0053935, -0.0042642, -0.0054033, -0.0043133, -0.0004742, 0.0005245
4: 0.0017998, 0.0022800, 0.0018207, 0.0022842, -0.0002230, 0.0002017
5: 0.0072247, 0.0103453, 0.0073604, 0.0103725, -0.0014492, 0.0013104
6: -0.0010849, -0.0002929, -0.0010918, -0.0003273, -0.0003326, 0.0003678
7: -0.0059446, -0.0038954, -0.0059625, -0.0039845, -0.0008605, 0.0009517
8: -0.0026904, -0.0016127, -0.0026997, -0.0016596, -0.0004525, 0.0005005
9: 0.0000061, 0.0012557, 0.0000605, 0.0012666, -0.0005803, 0.0005247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004921, upper bound: 0.0005376
time: 0.66 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0005853
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906023, 0.9924827, 0.9906546, 0.9924650, -0.0009155, 0.0008873
1: -0.0036056, -0.0031371, -0.0035926, -0.0031415, -0.0002281, 0.0002211
2: 0.0065708, 0.0090539, 0.0065942, 0.0089849, -0.0011717, 0.0012089
3: -0.0053941, -0.0042639, -0.0053627, -0.0042745, -0.0005502, 0.0005333
4: 0.0017997, 0.0022802, 0.0018042, 0.0022669, -0.0002268, 0.0002340
5: 0.0072238, 0.0103469, 0.0072532, 0.0102601, -0.0014737, 0.0015205
6: -0.0010853, -0.0002926, -0.0010633, -0.0003001, -0.0003859, 0.0003740
7: -0.0059457, -0.0038948, -0.0058887, -0.0039141, -0.0009985, 0.0009677
8: -0.0026909, -0.0016124, -0.0026609, -0.0016225, -0.0005251, 0.0005089
9: 0.0000058, 0.0012564, 0.0000176, 0.0012216, -0.0005901, 0.0006089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004847, upper bound: 0.0005477
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005176, upper bound: 0.0005956
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906023, 0.9924827, 0.9905989, 0.9924021, -0.0008003, 0.0009002
1: -0.0036056, -0.0031371, -0.0036065, -0.0031572, -0.0001994, 0.0002243
2: 0.0065708, 0.0090539, 0.0066773, 0.0090584, -0.0011888, 0.0010568
3: -0.0053941, -0.0042639, -0.0053961, -0.0043123, -0.0004810, 0.0005411
4: 0.0017997, 0.0022802, 0.0018203, 0.0022811, -0.0002301, 0.0002045
5: 0.0072238, 0.0103469, 0.0073577, 0.0103526, -0.0014952, 0.0013292
6: -0.0010853, -0.0002926, -0.0010868, -0.0003266, -0.0003374, 0.0003795
7: -0.0059457, -0.0038948, -0.0059494, -0.0039827, -0.0008728, 0.0009818
8: -0.0026909, -0.0016124, -0.0026929, -0.0016586, -0.0004590, 0.0005163
9: 0.0000058, 0.0012564, 0.0000594, 0.0012587, -0.0005987, 0.0005323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004847, upper bound: 0.0005477
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005176, upper bound: 0.0005956
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9906030, 0.9924843, 0.9906272, 0.9924768, -0.0008987, 0.0008955
1: -0.0036054, -0.0031367, -0.0035994, -0.0031385, -0.0002239, 0.0002231
2: 0.0065687, 0.0090529, 0.0065785, 0.0090210, -0.0011825, 0.0011868
3: -0.0053936, -0.0042629, -0.0053791, -0.0042674, -0.0005402, 0.0005382
4: 0.0017992, 0.0022801, 0.0018011, 0.0022739, -0.0002289, 0.0002297
5: 0.0072211, 0.0103456, 0.0072335, 0.0103055, -0.0014872, 0.0014927
6: -0.0010850, -0.0002920, -0.0010748, -0.0002951, -0.0003789, 0.0003775
7: -0.0059448, -0.0038930, -0.0059185, -0.0039012, -0.0009802, 0.0009766
8: -0.0026905, -0.0016114, -0.0026766, -0.0016157, -0.0005155, 0.0005136
9: 0.0000047, 0.0012559, 0.0000097, 0.0012398, -0.0005956, 0.0005977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005014, upper bound: 0.0005376
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005295, upper bound: 0.0005853
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9906030, 0.9924843, 0.9905689, 0.9924107, -0.0007918, 0.0009041
1: -0.0036054, -0.0031367, -0.0036139, -0.0031550, -0.0001973, 0.0002253
2: 0.0065687, 0.0090529, 0.0066660, 0.0090980, -0.0011938, 0.0010455
3: -0.0053936, -0.0042629, -0.0054141, -0.0043072, -0.0004759, 0.0005434
4: 0.0017992, 0.0022801, 0.0018181, 0.0022888, -0.0002311, 0.0002024
5: 0.0072211, 0.0103456, 0.0073435, 0.0104024, -0.0015015, 0.0013150
6: -0.0010850, -0.0002920, -0.0010994, -0.0003230, -0.0003338, 0.0003811
7: -0.0059448, -0.0038930, -0.0059821, -0.0039734, -0.0008635, 0.0009860
8: -0.0026905, -0.0016114, -0.0027101, -0.0016537, -0.0004541, 0.0005185
9: 0.0000047, 0.0012559, 0.0000537, 0.0012786, -0.0006013, 0.0005266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005014, upper bound: 0.0005376
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005295, upper bound: 0.0005853
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9906021, 0.9924849, 0.9906373, 0.9924735, -0.0009199, 0.0009159
1: -0.0036056, -0.0031365, -0.0035969, -0.0031393, -0.0002292, 0.0002282
2: 0.0065679, 0.0090541, 0.0065828, 0.0090077, -0.0012094, 0.0012148
3: -0.0053941, -0.0042626, -0.0053730, -0.0042693, -0.0005529, 0.0005505
4: 0.0017991, 0.0022803, 0.0018020, 0.0022713, -0.0002341, 0.0002351
5: 0.0072202, 0.0103471, 0.0072389, 0.0102887, -0.0015211, 0.0015278
6: -0.0010854, -0.0002917, -0.0010706, -0.0002965, -0.0003878, 0.0003861
7: -0.0059458, -0.0038924, -0.0059075, -0.0039047, -0.0010033, 0.0009989
8: -0.0026910, -0.0016111, -0.0026708, -0.0016176, -0.0005276, 0.0005253
9: 0.0000043, 0.0012565, 0.0000118, 0.0012331, -0.0006091, 0.0006118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004978, upper bound: 0.0005477
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005279, upper bound: 0.0005957
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9906021, 0.9924849, 0.9905788, 0.9924135, -0.0008054, 0.0009262
1: -0.0036056, -0.0031365, -0.0036115, -0.0031543, -0.0002007, 0.0002308
2: 0.0065679, 0.0090541, 0.0066622, 0.0090848, -0.0012231, 0.0010635
3: -0.0053941, -0.0042626, -0.0054081, -0.0043055, -0.0004841, 0.0005567
4: 0.0017991, 0.0022803, 0.0018173, 0.0022862, -0.0002367, 0.0002058
5: 0.0072202, 0.0103471, 0.0073388, 0.0103857, -0.0015383, 0.0013376
6: -0.0010854, -0.0002917, -0.0010952, -0.0003218, -0.0003395, 0.0003904
7: -0.0059458, -0.0038924, -0.0059712, -0.0039703, -0.0008784, 0.0010102
8: -0.0026910, -0.0016111, -0.0027043, -0.0016521, -0.0004619, 0.0005313
9: 0.0000043, 0.0012565, 0.0000518, 0.0012720, -0.0006160, 0.0005356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004978, upper bound: 0.0005477
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005279, upper bound: 0.0005957
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905704, 0.9925019, 0.9906321, 0.9924694, -0.0008962, 0.0008699
1: -0.0036136, -0.0031323, -0.0035982, -0.0031404, -0.0002233, 0.0002168
2: 0.0065454, 0.0090960, 0.0065883, 0.0090145, -0.0011487, 0.0011834
3: -0.0054132, -0.0042523, -0.0053761, -0.0042718, -0.0005386, 0.0005229
4: 0.0017947, 0.0022884, 0.0018030, 0.0022726, -0.0002223, 0.0002290
5: 0.0071919, 0.0103998, 0.0072459, 0.0102973, -0.0014448, 0.0014884
6: -0.0010988, -0.0002845, -0.0010727, -0.0002982, -0.0003778, 0.0003667
7: -0.0059804, -0.0038738, -0.0059131, -0.0039093, -0.0009774, 0.0009488
8: -0.0027092, -0.0016013, -0.0026738, -0.0016200, -0.0005140, 0.0004990
9: -0.0000070, 0.0012776, 0.0000146, 0.0012366, -0.0005786, 0.0005960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005582, upper bound: 0.0005853
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905704, 0.9925019, 0.9905745, 0.9924005, -0.0007932, 0.0008963
1: -0.0036136, -0.0031323, -0.0036125, -0.0031575, -0.0001976, 0.0002233
2: 0.0065454, 0.0090960, 0.0066794, 0.0090906, -0.0011835, 0.0010474
3: -0.0054132, -0.0042523, -0.0054108, -0.0043133, -0.0004767, 0.0005387
4: 0.0017947, 0.0022884, 0.0018207, 0.0022874, -0.0002291, 0.0002027
5: 0.0071919, 0.0103998, 0.0073603, 0.0103930, -0.0014886, 0.0013173
6: -0.0010988, -0.0002845, -0.0010970, -0.0003273, -0.0003344, 0.0003778
7: -0.0059804, -0.0038738, -0.0059760, -0.0039844, -0.0008651, 0.0009775
8: -0.0027092, -0.0016013, -0.0027068, -0.0016595, -0.0004549, 0.0005141
9: -0.0000070, 0.0012776, 0.0000604, 0.0012749, -0.0005961, 0.0005275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005582, upper bound: 0.0005853
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905696, 0.9925011, 0.9906422, 0.9924650, -0.0009237, 0.0008882
1: -0.0036138, -0.0031325, -0.0035957, -0.0031415, -0.0002302, 0.0002213
2: 0.0065464, 0.0090970, 0.0065941, 0.0090011, -0.0011729, 0.0012197
3: -0.0054137, -0.0042528, -0.0053700, -0.0042745, -0.0005552, 0.0005338
4: 0.0017949, 0.0022886, 0.0018042, 0.0022700, -0.0002270, 0.0002361
5: 0.0071931, 0.0104011, 0.0072532, 0.0102805, -0.0014752, 0.0015341
6: -0.0010991, -0.0002849, -0.0010685, -0.0003001, -0.0003894, 0.0003744
7: -0.0059813, -0.0038746, -0.0059021, -0.0039141, -0.0010074, 0.0009687
8: -0.0027096, -0.0016018, -0.0026680, -0.0016225, -0.0005298, 0.0005094
9: -0.0000065, 0.0012781, 0.0000175, 0.0012298, -0.0005907, 0.0006143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905696, 0.9925011, 0.9905862, 0.9924022, -0.0008101, 0.0009171
1: -0.0036138, -0.0031325, -0.0036096, -0.0031571, -0.0002019, 0.0002285
2: 0.0065464, 0.0090970, 0.0066771, 0.0090750, -0.0012110, 0.0010697
3: -0.0054137, -0.0042528, -0.0054037, -0.0043123, -0.0004869, 0.0005512
4: 0.0017949, 0.0022886, 0.0018202, 0.0022843, -0.0002344, 0.0002070
5: 0.0071931, 0.0104011, 0.0073575, 0.0103734, -0.0015231, 0.0013454
6: -0.0010991, -0.0002849, -0.0010921, -0.0003266, -0.0003415, 0.0003866
7: -0.0059813, -0.0038746, -0.0059631, -0.0039826, -0.0008835, 0.0010002
8: -0.0027096, -0.0016018, -0.0027001, -0.0016586, -0.0004646, 0.0005260
9: -0.0000065, 0.0012781, 0.0000593, 0.0012670, -0.0006099, 0.0005388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905703, 0.9925030, 0.9906156, 0.9924769, -0.0009008, 0.0008973
1: -0.0036136, -0.0031320, -0.0036023, -0.0031385, -0.0002245, 0.0002236
2: 0.0065440, 0.0090961, 0.0065784, 0.0090363, -0.0011849, 0.0011895
3: -0.0054133, -0.0042517, -0.0053861, -0.0042673, -0.0005414, 0.0005393
4: 0.0017945, 0.0022884, 0.0018011, 0.0022769, -0.0002293, 0.0002302
5: 0.0071901, 0.0104000, 0.0072334, 0.0103248, -0.0014903, 0.0014961
6: -0.0010988, -0.0002841, -0.0010797, -0.0002951, -0.0003797, 0.0003783
7: -0.0059806, -0.0038726, -0.0059312, -0.0039011, -0.0009825, 0.0009786
8: -0.0027093, -0.0016007, -0.0026833, -0.0016157, -0.0005167, 0.0005147
9: -0.0000077, 0.0012777, 0.0000096, 0.0012475, -0.0005968, 0.0005991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
time: 0.66 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0005853
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905703, 0.9925030, 0.9905572, 0.9924107, -0.0007960, 0.0009228
1: -0.0036136, -0.0031320, -0.0036169, -0.0031550, -0.0001983, 0.0002299
2: 0.0065440, 0.0090961, 0.0066658, 0.0091134, -0.0012186, 0.0010511
3: -0.0054133, -0.0042517, -0.0054212, -0.0043071, -0.0004784, 0.0005547
4: 0.0017945, 0.0022884, 0.0018180, 0.0022918, -0.0002359, 0.0002034
5: 0.0071901, 0.0104000, 0.0073433, 0.0104217, -0.0015327, 0.0013220
6: -0.0010988, -0.0002841, -0.0011043, -0.0003230, -0.0003355, 0.0003890
7: -0.0059806, -0.0038726, -0.0059948, -0.0039733, -0.0008681, 0.0010065
8: -0.0027093, -0.0016007, -0.0027168, -0.0016536, -0.0004565, 0.0005293
9: -0.0000077, 0.0012777, 0.0000536, 0.0012864, -0.0006138, 0.0005294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0005853
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905695, 0.9925028, 0.9906256, 0.9924737, -0.0009290, 0.0009141
1: -0.0036138, -0.0031321, -0.0035998, -0.0031393, -0.0002315, 0.0002278
2: 0.0065443, 0.0090971, 0.0065827, 0.0090230, -0.0012070, 0.0012268
3: -0.0054137, -0.0042518, -0.0053800, -0.0042693, -0.0005584, 0.0005494
4: 0.0017945, 0.0022886, 0.0018020, 0.0022743, -0.0002336, 0.0002374
5: 0.0071904, 0.0104012, 0.0072388, 0.0103081, -0.0015181, 0.0015430
6: -0.0010991, -0.0002842, -0.0010755, -0.0002965, -0.0003916, 0.0003853
7: -0.0059814, -0.0038729, -0.0059202, -0.0039046, -0.0010132, 0.0009969
8: -0.0027097, -0.0016008, -0.0026775, -0.0016176, -0.0005329, 0.0005243
9: -0.0000076, 0.0012782, 0.0000118, 0.0012408, -0.0006079, 0.0006179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005957
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905695, 0.9925028, 0.9905673, 0.9924136, -0.0008151, 0.0009405
1: -0.0036138, -0.0031321, -0.0036143, -0.0031543, -0.0002031, 0.0002343
2: 0.0065443, 0.0090971, 0.0066620, 0.0091001, -0.0012419, 0.0010764
3: -0.0054137, -0.0042518, -0.0054151, -0.0043054, -0.0004899, 0.0005652
4: 0.0017945, 0.0022886, 0.0018173, 0.0022892, -0.0002404, 0.0002083
5: 0.0071904, 0.0104012, 0.0073385, 0.0104050, -0.0015619, 0.0013538
6: -0.0010991, -0.0002842, -0.0011001, -0.0003218, -0.0003436, 0.0003964
7: -0.0059814, -0.0038729, -0.0059838, -0.0039701, -0.0008890, 0.0010257
8: -0.0027097, -0.0016008, -0.0027110, -0.0016520, -0.0004675, 0.0005394
9: -0.0000076, 0.0012782, 0.0000517, 0.0012797, -0.0006255, 0.0005421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005956
time: 0.67 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905570, 0.9925020, 0.9906259, 0.9925148, -0.0009425, 0.0008265
1: -0.0036169, -0.0031322, -0.0035997, -0.0031291, -0.0002348, 0.0002059
2: 0.0065452, 0.0091138, 0.0065284, 0.0090226, -0.0010913, 0.0012445
3: -0.0054213, -0.0042522, -0.0053798, -0.0042446, -0.0005665, 0.0004967
4: 0.0017947, 0.0022918, 0.0017915, 0.0022742, -0.0002112, 0.0002409
5: 0.0071916, 0.0104222, 0.0071705, 0.0103075, -0.0013726, 0.0015653
6: -0.0011044, -0.0002845, -0.0010753, -0.0002791, -0.0003973, 0.0003484
7: -0.0059951, -0.0038737, -0.0059198, -0.0038598, -0.0010279, 0.0009014
8: -0.0027169, -0.0016013, -0.0026773, -0.0015940, -0.0005406, 0.0004740
9: -0.0000071, 0.0012865, -0.0000156, 0.0012406, -0.0005497, 0.0006268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005852
time: 0.69 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905570, 0.9925020, 0.9905627, 0.9924586, -0.0008362, 0.0008372
1: -0.0036169, -0.0031322, -0.0036155, -0.0031431, -0.0002084, 0.0002086
2: 0.0065452, 0.0091138, 0.0066025, 0.0091061, -0.0011055, 0.0011042
3: -0.0054213, -0.0042522, -0.0054178, -0.0042783, -0.0005026, 0.0005032
4: 0.0017947, 0.0022918, 0.0018058, 0.0022903, -0.0002140, 0.0002137
5: 0.0071916, 0.0104222, 0.0072637, 0.0104125, -0.0013905, 0.0013888
6: -0.0011044, -0.0002845, -0.0011020, -0.0003028, -0.0003525, 0.0003529
7: -0.0059951, -0.0038737, -0.0059888, -0.0039210, -0.0009120, 0.0009131
8: -0.0027169, -0.0016013, -0.0027136, -0.0016262, -0.0004796, 0.0004802
9: -0.0000071, 0.0012865, 0.0000218, 0.0012827, -0.0005568, 0.0005561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005853
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905562, 0.9925013, 0.9906368, 0.9925102, -0.0009460, 0.0008540
1: -0.0036171, -0.0031324, -0.0035970, -0.0031302, -0.0002357, 0.0002128
2: 0.0065462, 0.0091148, 0.0065345, 0.0090084, -0.0011277, 0.0012491
3: -0.0054218, -0.0042527, -0.0053734, -0.0042474, -0.0005685, 0.0005133
4: 0.0017949, 0.0022920, 0.0017926, 0.0022714, -0.0002183, 0.0002418
5: 0.0071929, 0.0104234, 0.0071782, 0.0102897, -0.0014183, 0.0015711
6: -0.0011047, -0.0002848, -0.0010708, -0.0002811, -0.0003988, 0.0003600
7: -0.0059959, -0.0038745, -0.0059081, -0.0038648, -0.0010317, 0.0009314
8: -0.0027174, -0.0016017, -0.0026712, -0.0015966, -0.0005426, 0.0004898
9: -0.0000066, 0.0012871, -0.0000125, 0.0012335, -0.0005680, 0.0006291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
time: 0.69 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905562, 0.9925013, 0.9905733, 0.9924529, -0.0008307, 0.0008679
1: -0.0036171, -0.0031324, -0.0036128, -0.0031445, -0.0002070, 0.0002163
2: 0.0065462, 0.0091148, 0.0066102, 0.0090921, -0.0011461, 0.0010969
3: -0.0054218, -0.0042527, -0.0054115, -0.0042818, -0.0004993, 0.0005216
4: 0.0017949, 0.0022920, 0.0018073, 0.0022876, -0.0002218, 0.0002123
5: 0.0071929, 0.0104234, 0.0072734, 0.0103949, -0.0014415, 0.0013796
6: -0.0011047, -0.0002848, -0.0010975, -0.0003052, -0.0003502, 0.0003659
7: -0.0059959, -0.0038745, -0.0059772, -0.0039273, -0.0009060, 0.0009466
8: -0.0027174, -0.0016017, -0.0027075, -0.0016295, -0.0004765, 0.0004978
9: -0.0000066, 0.0012871, 0.0000256, 0.0012756, -0.0005772, 0.0005525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
time: 0.69 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9905569, 0.9925031, 0.9906107, 0.9925222, -0.0009464, 0.0008598
1: -0.0036169, -0.0031320, -0.0036035, -0.0031272, -0.0002358, 0.0002142
2: 0.0065438, 0.0091139, 0.0065186, 0.0090428, -0.0011353, 0.0012497
3: -0.0054214, -0.0042516, -0.0053890, -0.0042401, -0.0005688, 0.0005167
4: 0.0017944, 0.0022919, 0.0017896, 0.0022781, -0.0002197, 0.0002419
5: 0.0071898, 0.0104223, 0.0071581, 0.0103329, -0.0014279, 0.0015718
6: -0.0011045, -0.0002840, -0.0010818, -0.0002760, -0.0003989, 0.0003624
7: -0.0059952, -0.0038725, -0.0059365, -0.0038517, -0.0010322, 0.0009377
8: -0.0027170, -0.0016007, -0.0026861, -0.0015897, -0.0005428, 0.0004931
9: -0.0000078, 0.0012866, -0.0000205, 0.0012508, -0.0005718, 0.0006294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0005853
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9905569, 0.9925031, 0.9905467, 0.9924678, -0.0008391, 0.0008668
1: -0.0036169, -0.0031320, -0.0036195, -0.0031408, -0.0002091, 0.0002160
2: 0.0065438, 0.0091139, 0.0065905, 0.0091273, -0.0011447, 0.0011081
3: -0.0054214, -0.0042516, -0.0054275, -0.0042728, -0.0005044, 0.0005210
4: 0.0017944, 0.0022919, 0.0018035, 0.0022945, -0.0002215, 0.0002145
5: 0.0071898, 0.0104223, 0.0072486, 0.0104392, -0.0014397, 0.0013937
6: -0.0011045, -0.0002840, -0.0011088, -0.0002989, -0.0003537, 0.0003654
7: -0.0059952, -0.0038725, -0.0060063, -0.0039111, -0.0009152, 0.0009454
8: -0.0027170, -0.0016007, -0.0027228, -0.0016209, -0.0004813, 0.0004972
9: -0.0000078, 0.0012866, 0.0000157, 0.0012934, -0.0005765, 0.0005581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0005852
time: 0.67 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9905561, 0.9925030, 0.9906202, 0.9925196, -0.0009504, 0.0008821
1: -0.0036171, -0.0031320, -0.0036011, -0.0031279, -0.0002368, 0.0002198
2: 0.0065441, 0.0091149, 0.0065222, 0.0090302, -0.0011648, 0.0012549
3: -0.0054218, -0.0042517, -0.0053833, -0.0042417, -0.0005712, 0.0005302
4: 0.0017945, 0.0022921, 0.0017902, 0.0022757, -0.0002254, 0.0002429
5: 0.0071902, 0.0104236, 0.0071626, 0.0103171, -0.0014650, 0.0015784
6: -0.0011048, -0.0002841, -0.0010778, -0.0002771, -0.0004006, 0.0003718
7: -0.0059960, -0.0038727, -0.0059261, -0.0038546, -0.0010365, 0.0009620
8: -0.0027174, -0.0016008, -0.0026806, -0.0015913, -0.0005451, 0.0005059
9: -0.0000077, 0.0012871, -0.0000187, 0.0012445, -0.0005867, 0.0006321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005957
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9905561, 0.9925030, 0.9905558, 0.9924642, -0.0008354, 0.0008930
1: -0.0036171, -0.0031320, -0.0036172, -0.0031417, -0.0002082, 0.0002225
2: 0.0065441, 0.0091149, 0.0065952, 0.0091154, -0.0011792, 0.0011032
3: -0.0054218, -0.0042517, -0.0054221, -0.0042750, -0.0005021, 0.0005367
4: 0.0017945, 0.0022921, 0.0018044, 0.0022922, -0.0002282, 0.0002135
5: 0.0071902, 0.0104236, 0.0072544, 0.0104242, -0.0014832, 0.0013875
6: -0.0011048, -0.0002841, -0.0011049, -0.0003004, -0.0003522, 0.0003764
7: -0.0059960, -0.0038727, -0.0059965, -0.0039149, -0.0009112, 0.0009740
8: -0.0027174, -0.0016008, -0.0027176, -0.0016230, -0.0004792, 0.0005122
9: -0.0000077, 0.0012871, 0.0000180, 0.0012874, -0.0005939, 0.0005556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005956
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.06 seconds
IS_A1_A1_B1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005767, upper bound: 0.0005491
IS_A1_A1_B1_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0005673
IS_A1_A1_B1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005801, upper bound: 0.0005491
IS_A1_A1_B1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005846, upper bound: 0.0005673
IS_A1_A1_B1_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005156
IS_A1_A1_B1_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005866, upper bound: 0.0005253
IS_A1_A1_B1_B2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0005125
IS_A1_A1_B1_B2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005221
IS_A1_A1_B1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005251
IS_A1_A1_B1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005865, upper bound: 0.0005322
IS_A1_A1_B1_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0005230
IS_A1_A1_B1_B2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005302
IS_A1_A1_B1_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005592
IS_A1_A1_B1_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005866, upper bound: 0.0005664
IS_A1_A1_B1_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005694, upper bound: 0.0005522
IS_A1_A1_B1_B2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005588
IS_A1_A1_B1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005610, upper bound: 0.0005640
IS_A1_A1_B1_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005865, upper bound: 0.0005697
IS_A1_A1_B1_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005695, upper bound: 0.0005579
IS_A1_A1_B1_B2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005964, upper bound: 0.0005633
IS_A1_A1_B2_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005002, upper bound: 0.0005714
IS_A1_A1_B2_B1_A1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005364, upper bound: 0.0005839
IS_A1_A1_B2_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005341, upper bound: 0.0005718
IS_A1_A1_B2_B1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005634, upper bound: 0.0005839
IS_A1_A1_B2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0005017
IS_A1_A1_B2_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005852, upper bound: 0.0005237
IS_A1_A1_B2_B2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0004977
IS_A1_A1_B2_B2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0005208
IS_A1_A1_B2_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005378, upper bound: 0.0005110
IS_A1_A1_B2_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005853, upper bound: 0.0005313
IS_A1_A1_B2_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0005071
IS_A1_A1_B2_B2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005957, upper bound: 0.0005293
IS_A1_A1_B2_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005451, upper bound: 0.0005499
IS_A1_A1_B2_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005853, upper bound: 0.0005651
IS_A1_A1_B2_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005542, upper bound: 0.0005413
IS_A1_A1_B2_B2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0005579
IS_A1_A1_B2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005451, upper bound: 0.0005551
IS_A1_A1_B2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005853, upper bound: 0.0005690
IS_A1_A1_B2_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005542, upper bound: 0.0005484
IS_A1_A1_B2_B2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005956, upper bound: 0.0005626
IS_A1_A2_B2_B1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005756, upper bound: 0.0004932
IS_A1_A2_B2_B1_B1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005278
IS_A1_A2_B2_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005756, upper bound: 0.0004932
IS_A1_A2_B2_B1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005278
IS_A1_A2_B2_B1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0004943
IS_A1_A2_B2_B1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005287
IS_A1_A2_B2_B1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0004943
IS_A1_A2_B2_B1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005287
IS_A1_A2_B2_B1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005823, upper bound: 0.0004932
IS_A1_A2_B2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005281
IS_A1_A2_B2_B1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005823, upper bound: 0.0004932
IS_A1_A2_B2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005281
IS_A1_A2_B2_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005746, upper bound: 0.0004943
IS_A1_A2_B2_B1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005291
IS_A1_A2_B2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005746, upper bound: 0.0004943
IS_A1_A2_B2_B1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005291
IS_A1_A2_B2_B2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005760, upper bound: 0.0005281
IS_A1_A2_B2_B2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005527
IS_A1_A2_B2_B2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005760, upper bound: 0.0005281
IS_A1_A2_B2_B2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0005526
IS_A1_A2_B2_B2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0005345
IS_A1_A2_B2_B2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005595
IS_A1_A2_B2_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005665, upper bound: 0.0005345
IS_A1_A2_B2_B2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005857, upper bound: 0.0005595
IS_A1_A2_B2_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005832, upper bound: 0.0005281
IS_A1_A2_B2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005527
IS_A1_A2_B2_B2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005832, upper bound: 0.0005281
IS_A1_A2_B2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0006011, upper bound: 0.0005527
IS_A1_A2_B2_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005770, upper bound: 0.0005345
IS_A1_A2_B2_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005596
IS_A1_A2_B2_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005770, upper bound: 0.0005345
IS_A1_A2_B2_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005951, upper bound: 0.0005596
IS_A2_A1_B1_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005610
IS_A2_A1_B1_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005253, upper bound: 0.0005866
IS_A2_A1_B1_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005695
IS_A2_A1_B1_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005221, upper bound: 0.0005964
IS_A2_A1_B1_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005251, upper bound: 0.0005610
IS_A2_A1_B1_B1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005322, upper bound: 0.0005865
IS_A2_A1_B1_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005230, upper bound: 0.0005695
IS_A2_A1_B1_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005302, upper bound: 0.0005965
IS_A2_A1_B1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005592, upper bound: 0.0005610
IS_A2_A1_B1_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005664, upper bound: 0.0005866
IS_A2_A1_B1_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005522, upper bound: 0.0005695
IS_A2_A1_B1_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005964
IS_A2_A1_B1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005610
IS_A2_A1_B1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005697, upper bound: 0.0005866
IS_A2_A1_B1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005695
IS_A2_A1_B1_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005633, upper bound: 0.0005965
IS_A2_A1_B1_B2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005828
IS_A2_A1_B1_B2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005278, upper bound: 0.0005980
IS_A2_A1_B1_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005738
IS_A2_A1_B1_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005287, upper bound: 0.0005909
IS_A2_A1_B1_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005863
IS_A2_A1_B1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0006014
IS_A2_A1_B1_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005797
IS_A2_A1_B1_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005956
IS_A2_A1_B1_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005828
IS_A2_A1_B1_B2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005527, upper bound: 0.0005979
IS_A2_A1_B1_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005744
IS_A2_A1_B1_B2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0005910
IS_A2_A1_B1_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005864
IS_A2_A1_B1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005527, upper bound: 0.0006015
IS_A2_A1_B1_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005804
IS_A2_A1_B1_B2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005596, upper bound: 0.0005956
IS_A2_A1_B2_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005610
IS_A2_A1_B2_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005255, upper bound: 0.0005866
IS_A2_A1_B2_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005695
IS_A2_A1_B2_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005223, upper bound: 0.0005965
IS_A2_A1_B2_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005251, upper bound: 0.0005610
IS_A2_A1_B2_B1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005324, upper bound: 0.0005866
IS_A2_A1_B2_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005230, upper bound: 0.0005695
IS_A2_A1_B2_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005304, upper bound: 0.0005965
IS_A2_A1_B2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005592, upper bound: 0.0005610
IS_A2_A1_B2_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005664, upper bound: 0.0005866
IS_A2_A1_B2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005522, upper bound: 0.0005694
IS_A2_A1_B2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005589, upper bound: 0.0005965
IS_A2_A1_B2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005640, upper bound: 0.0005610
IS_A2_A1_B2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005698, upper bound: 0.0005866
IS_A2_A1_B2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005579, upper bound: 0.0005695
IS_A2_A1_B2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005634, upper bound: 0.0005965
IS_A2_A1_B2_B2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005828
IS_A2_A1_B2_B2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005279, upper bound: 0.0005979
IS_A2_A1_B2_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005738
IS_A2_A1_B2_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005289, upper bound: 0.0005909
IS_A2_A1_B2_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004950, upper bound: 0.0005863
IS_A2_A1_B2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005282, upper bound: 0.0006015
IS_A2_A1_B2_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004965, upper bound: 0.0005797
IS_A2_A1_B2_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005292, upper bound: 0.0005956
IS_A2_A1_B2_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005828
IS_A2_A1_B2_B2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005527, upper bound: 0.0005980
IS_A2_A1_B2_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005744
IS_A2_A1_B2_B2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0005909
IS_A2_A1_B2_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005281, upper bound: 0.0005863
IS_A2_A1_B2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005528, upper bound: 0.0006015
IS_A2_A1_B2_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005345, upper bound: 0.0005804
IS_A2_A1_B2_B2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005596, upper bound: 0.0005956
IS_A2_A2_B1_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004921, upper bound: 0.0005376
IS_A2_A2_B1_A1_B1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0005853
IS_A2_A2_B1_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004921, upper bound: 0.0005376
IS_A2_A2_B1_A1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0005853
IS_A2_A2_B1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004847, upper bound: 0.0005477
IS_A2_A2_B1_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005176, upper bound: 0.0005956
IS_A2_A2_B1_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004847, upper bound: 0.0005477
IS_A2_A2_B1_A1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005176, upper bound: 0.0005956
IS_A2_A2_B1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005014, upper bound: 0.0005376
IS_A2_A2_B1_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005295, upper bound: 0.0005853
IS_A2_A2_B1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005014, upper bound: 0.0005376
IS_A2_A2_B1_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005295, upper bound: 0.0005853
IS_A2_A2_B1_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004978, upper bound: 0.0005477
IS_A2_A2_B1_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005279, upper bound: 0.0005957
IS_A2_A2_B1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0004978, upper bound: 0.0005477
IS_A2_A2_B1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005279, upper bound: 0.0005957
IS_A2_A2_B1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
IS_A2_A2_B1_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005582, upper bound: 0.0005853
IS_A2_A2_B1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
IS_A2_A2_B1_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005582, upper bound: 0.0005853
IS_A2_A2_B1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
IS_A2_A2_B1_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
IS_A2_A2_B1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
IS_A2_A2_B1_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
IS_A2_A2_B1_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
IS_A2_A2_B1_A2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0005853
IS_A2_A2_B1_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
IS_A2_A2_B1_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005645, upper bound: 0.0005853
IS_A2_A2_B1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
IS_A2_A2_B1_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005957
IS_A2_A2_B1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
IS_A2_A2_B1_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005956
IS_A2_A2_B2_B2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
IS_A2_A2_B2_B2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005852
IS_A2_A2_B2_B2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005412, upper bound: 0.0005451
IS_A2_A2_B2_B2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005583, upper bound: 0.0005853
IS_A2_A2_B2_B2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
IS_A2_A2_B2_B2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
IS_A2_A2_B2_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005343, upper bound: 0.0005542
IS_A2_A2_B2_B2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005526, upper bound: 0.0005956
IS_A2_A2_B2_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
IS_A2_A2_B2_B2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0005853
IS_A2_A2_B2_B2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005475, upper bound: 0.0005451
IS_A2_A2_B2_B2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0005852
IS_A2_A2_B2_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
IS_A2_A2_B2_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005957
IS_A2_A2_B2_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005542
IS_A2_A2_B2_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 0, lower bound: -0.0005588, upper bound: 0.0005956

## BFS IS instance: IS_A1_A2_B2_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906140, 0.9924515, 0.9906445, 0.9925039, -0.0009573, 0.0008515
1: -0.0036027, -0.0031448, -0.0035951, -0.0031318, -0.0002385, 0.0002122
2: 0.0066119, 0.0090384, 0.0065428, 0.0089981, -0.0011244, 0.0012642
3: -0.0053870, -0.0042826, -0.0053687, -0.0042511, -0.0005754, 0.0005118
4: 0.0018076, 0.0022773, 0.0017942, 0.0022695, -0.0002176, 0.0002447
5: 0.0072755, 0.0103274, 0.0071886, 0.0102767, -0.0014142, 0.0015900
6: -0.0010804, -0.0003058, -0.0010675, -0.0002837, -0.0004036, 0.0003589
7: -0.0059329, -0.0039287, -0.0058996, -0.0038717, -0.0010441, 0.0009287
8: -0.0026842, -0.0016302, -0.0026667, -0.0016002, -0.0005491, 0.0004884
9: 0.0000265, 0.0012486, -0.0000083, 0.0012283, -0.0005663, 0.0006367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005796, upper bound: 0.0005045
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0004998
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906140, 0.9924515, 0.9905801, 0.9924491, -0.0008399, 0.0008583
1: -0.0036027, -0.0031448, -0.0036111, -0.0031454, -0.0002093, 0.0002139
2: 0.0066119, 0.0090384, 0.0066152, 0.0090832, -0.0011333, 0.0011091
3: -0.0053870, -0.0042826, -0.0054074, -0.0042841, -0.0005048, 0.0005158
4: 0.0018076, 0.0022773, 0.0018082, 0.0022859, -0.0002194, 0.0002147
5: 0.0072755, 0.0103274, 0.0072796, 0.0103837, -0.0014254, 0.0013949
6: -0.0010804, -0.0003058, -0.0010947, -0.0003068, -0.0003540, 0.0003618
7: -0.0059329, -0.0039287, -0.0059699, -0.0039315, -0.0009160, 0.0009361
8: -0.0026842, -0.0016302, -0.0027036, -0.0016317, -0.0004817, 0.0004923
9: 0.0000265, 0.0012486, 0.0000281, 0.0012711, -0.0005708, 0.0005586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005771, upper bound: 0.0005045
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005720, upper bound: 0.0004997
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9906019, 0.9924517, 0.9906107, 0.9925222, -0.0009808, 0.0008699
1: -0.0036057, -0.0031448, -0.0036035, -0.0031272, -0.0002444, 0.0002168
2: 0.0066118, 0.0090544, 0.0065186, 0.0090428, -0.0011487, 0.0012952
3: -0.0053943, -0.0042825, -0.0053890, -0.0042401, -0.0005895, 0.0005228
4: 0.0018076, 0.0022803, 0.0017896, 0.0022781, -0.0002223, 0.0002507
5: 0.0072753, 0.0103475, 0.0071581, 0.0103329, -0.0014447, 0.0016290
6: -0.0010855, -0.0003057, -0.0010818, -0.0002760, -0.0004135, 0.0003667
7: -0.0059460, -0.0039286, -0.0059365, -0.0038517, -0.0010697, 0.0009487
8: -0.0026911, -0.0016302, -0.0026861, -0.0015897, -0.0005626, 0.0004989
9: 0.0000264, 0.0012566, -0.0000205, 0.0012508, -0.0005785, 0.0006523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005796, upper bound: 0.0005278
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005288
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9906019, 0.9924517, 0.9905467, 0.9924678, -0.0008702, 0.0008702
1: -0.0036057, -0.0031448, -0.0036195, -0.0031408, -0.0002168, 0.0002168
2: 0.0066118, 0.0090544, 0.0065905, 0.0091273, -0.0011491, 0.0011491
3: -0.0053943, -0.0042825, -0.0054275, -0.0042728, -0.0005230, 0.0005230
4: 0.0018076, 0.0022803, 0.0018035, 0.0022945, -0.0002224, 0.0002224
5: 0.0072753, 0.0103475, 0.0072486, 0.0104392, -0.0014453, 0.0014453
6: -0.0010855, -0.0003057, -0.0011088, -0.0002989, -0.0003668, 0.0003668
7: -0.0059460, -0.0039286, -0.0060063, -0.0039111, -0.0009491, 0.0009491
8: -0.0026911, -0.0016302, -0.0027228, -0.0016209, -0.0004991, 0.0004991
9: 0.0000264, 0.0012566, 0.0000157, 0.0012934, -0.0005787, 0.0005788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.14 + 596.93 = 600.08 seconds
