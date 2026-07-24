## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00049488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9876423, 0.9897519, 0.9876423, 0.9897519, -0.0015473, 0.0015473)
1: (-0.0043432, -0.0038175, -0.0043432, -0.0038175, -0.0003856, 0.0003856)
2: (0.0101767, 0.0129625, 0.0101767, 0.0129625, -0.0020432, 0.0020432)
3: (-0.0071731, -0.0059051, -0.0071731, -0.0059051, -0.0009300, 0.0009300)
4: (0.0024976, 0.0030368, 0.0024976, 0.0030368, -0.0003955, 0.0003955)
5: (0.0117591, 0.0152629, 0.0117591, 0.0152629, -0.0025699, 0.0025699)
6: (-0.0023330, -0.0014437, -0.0023330, -0.0014437, -0.0006523, 0.0006523)
7: (-0.0091739, -0.0068730, -0.0091739, -0.0068730, -0.0016876, 0.0016876)
8: (-0.0043886, -0.0031786, -0.0043886, -0.0031786, -0.0008875, 0.0008875)
9: (0.0018219, 0.0032250, 0.0018219, 0.0032250, -0.0010291, 0.0010291)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 1.39 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006537, upper bound: 0.0006538

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006037, upper bound: 0.0006124
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006056, upper bound: 0.0006057
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 0, lower bound: -0.0006037, upper bound: 0.0006124
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 0, lower bound: -0.0006056, upper bound: 0.0006057

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9875428, 0.9896365, 0.9876428, 0.9897015, -0.0015147, 0.0013960
1: -0.0043680, -0.0038463, -0.0043431, -0.0038301, -0.0003774, 0.0003478
2: 0.0103291, 0.0130939, 0.0102433, 0.0129619, -0.0018433, 0.0020002
3: -0.0072329, -0.0059745, -0.0071728, -0.0059354, -0.0009104, 0.0008390
4: 0.0025271, 0.0030622, 0.0025105, 0.0030366, -0.0003568, 0.0003871
5: 0.0119508, 0.0154281, 0.0118428, 0.0152621, -0.0023184, 0.0025157
6: -0.0023750, -0.0014924, -0.0023329, -0.0014650, -0.0006385, 0.0005884
7: -0.0092824, -0.0069989, -0.0091734, -0.0069281, -0.0016520, 0.0015225
8: -0.0044457, -0.0032448, -0.0043884, -0.0032075, -0.0008688, 0.0008007
9: 0.0018987, 0.0032911, 0.0018554, 0.0032247, -0.0009284, 0.0010074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005436, upper bound: 0.0005883
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005782, upper bound: 0.0005883
time: 0.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9876431, 0.9896742, 0.9876425, 0.9897310, -0.0015324, 0.0014238
1: -0.0043430, -0.0038369, -0.0043431, -0.0038227, -0.0003818, 0.0003548
2: 0.0102794, 0.0129615, 0.0102044, 0.0129622, -0.0018802, 0.0020235
3: -0.0071726, -0.0059519, -0.0071730, -0.0059177, -0.0009210, 0.0008558
4: 0.0025174, 0.0030366, 0.0025029, 0.0030367, -0.0003639, 0.0003916
5: 0.0118882, 0.0152616, 0.0117939, 0.0152625, -0.0023648, 0.0025450
6: -0.0023327, -0.0014765, -0.0023330, -0.0014526, -0.0006459, 0.0006002
7: -0.0091731, -0.0069579, -0.0091737, -0.0068959, -0.0016713, 0.0015529
8: -0.0043882, -0.0032232, -0.0043885, -0.0031906, -0.0008789, 0.0008167
9: 0.0018736, 0.0032245, 0.0018359, 0.0032248, -0.0009470, 0.0010191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005451, upper bound: 0.0005814
time: 0.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005814, upper bound: 0.0005814
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.0005436, upper bound: 0.0005883
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.0005782, upper bound: 0.0005883
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.0005451, upper bound: 0.0005814
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.0005814, upper bound: 0.0005814

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9875431, 0.9895858, 0.9876739, 0.9895785, -0.0013927, 0.0012585
1: -0.0043679, -0.0038589, -0.0043353, -0.0038607, -0.0003470, 0.0003136
2: 0.0103962, 0.0130935, 0.0104058, 0.0129207, -0.0016618, 0.0018390
3: -0.0072327, -0.0060050, -0.0071541, -0.0060094, -0.0008370, 0.0007564
4: 0.0025401, 0.0030621, 0.0025419, 0.0030287, -0.0003216, 0.0003559
5: 0.0120351, 0.0154276, 0.0120472, 0.0152103, -0.0020902, 0.0023130
6: -0.0023749, -0.0015138, -0.0023197, -0.0015169, -0.0005871, 0.0005305
7: -0.0092821, -0.0070543, -0.0091394, -0.0070622, -0.0015189, 0.0013726
8: -0.0044455, -0.0032739, -0.0043705, -0.0032781, -0.0007988, 0.0007218
9: 0.0019325, 0.0032910, 0.0019373, 0.0032039, -0.0008370, 0.0009262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005040, upper bound: 0.0005353
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005040, upper bound: 0.0005502
time: 0.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875430, 0.9896158, 0.9876434, 0.9896475, -0.0013695, 0.0013724
1: -0.0043679, -0.0038514, -0.0043429, -0.0038435, -0.0003412, 0.0003420
2: 0.0103565, 0.0130936, 0.0103146, 0.0129611, -0.0018122, 0.0018084
3: -0.0072328, -0.0059870, -0.0071725, -0.0059679, -0.0008231, 0.0008249
4: 0.0025324, 0.0030621, 0.0025243, 0.0030365, -0.0003508, 0.0003500
5: 0.0119852, 0.0154278, 0.0119325, 0.0152611, -0.0022793, 0.0022745
6: -0.0023749, -0.0015012, -0.0023326, -0.0014878, -0.0005773, 0.0005785
7: -0.0092822, -0.0070216, -0.0091728, -0.0069869, -0.0014936, 0.0014968
8: -0.0044456, -0.0032567, -0.0043880, -0.0032385, -0.0007855, 0.0007871
9: 0.0019125, 0.0032910, 0.0018913, 0.0032243, -0.0009127, 0.0009108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005403, upper bound: 0.0005353
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005403, upper bound: 0.0005501
time: 0.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876433, 0.9896260, 0.9876738, 0.9896113, -0.0014086, 0.0013064
1: -0.0043429, -0.0038489, -0.0043353, -0.0038525, -0.0003510, 0.0003255
2: 0.0103430, 0.0129611, 0.0103624, 0.0129210, -0.0017251, 0.0018600
3: -0.0071725, -0.0059808, -0.0071542, -0.0059896, -0.0008466, 0.0007852
4: 0.0025298, 0.0030365, 0.0025335, 0.0030287, -0.0003339, 0.0003600
5: 0.0119683, 0.0152611, 0.0119926, 0.0152107, -0.0021697, 0.0023394
6: -0.0023326, -0.0014968, -0.0023198, -0.0015030, -0.0005938, 0.0005507
7: -0.0091728, -0.0070104, -0.0091397, -0.0070264, -0.0015363, 0.0014248
8: -0.0043880, -0.0032509, -0.0043706, -0.0032593, -0.0008079, 0.0007493
9: 0.0019057, 0.0032243, 0.0019154, 0.0032041, -0.0008688, 0.0009368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005498
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005111, upper bound: 0.0005498
time: 0.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876433, 0.9896529, 0.9876431, 0.9896753, -0.0013577, 0.0014008
1: -0.0043429, -0.0038422, -0.0043430, -0.0038366, -0.0003383, 0.0003490
2: 0.0103075, 0.0129612, 0.0102779, 0.0129615, -0.0018498, 0.0017929
3: -0.0071725, -0.0059647, -0.0071726, -0.0059512, -0.0008160, 0.0008419
4: 0.0025229, 0.0030365, 0.0025172, 0.0030366, -0.0003580, 0.0003470
5: 0.0119236, 0.0152613, 0.0118863, 0.0152616, -0.0023265, 0.0022549
6: -0.0023326, -0.0014855, -0.0023327, -0.0014760, -0.0005723, 0.0005905
7: -0.0091729, -0.0069811, -0.0091731, -0.0069566, -0.0014808, 0.0015278
8: -0.0043881, -0.0032354, -0.0043882, -0.0032226, -0.0007787, 0.0008035
9: 0.0018878, 0.0032243, 0.0018729, 0.0032245, -0.0009316, 0.0009030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005361, upper bound: 0.0005498
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005497, upper bound: 0.0005498
time: 0.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.15 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005040, upper bound: 0.0005353
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005040, upper bound: 0.0005502
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005403, upper bound: 0.0005353
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005403, upper bound: 0.0005501
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005113, upper bound: 0.0005498
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005111, upper bound: 0.0005498
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005361, upper bound: 0.0005498
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -0.0005497, upper bound: 0.0005498

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9875596, 0.9895844, 0.9877285, 0.9895744, -0.0013732, 0.0012116
1: -0.0043638, -0.0038592, -0.0043217, -0.0038617, -0.0003422, 0.0003019
2: 0.0103979, 0.0130717, 0.0104112, 0.0128487, -0.0015999, 0.0018133
3: -0.0072228, -0.0060058, -0.0071213, -0.0060118, -0.0008253, 0.0007282
4: 0.0025404, 0.0030579, 0.0025430, 0.0030147, -0.0003097, 0.0003510
5: 0.0120373, 0.0154002, 0.0120540, 0.0151198, -0.0020123, 0.0022807
6: -0.0023679, -0.0015144, -0.0022967, -0.0015186, -0.0005789, 0.0005107
7: -0.0092641, -0.0070557, -0.0090800, -0.0070667, -0.0014977, 0.0013214
8: -0.0044361, -0.0032747, -0.0043392, -0.0032805, -0.0007876, 0.0006949
9: 0.0019333, 0.0032800, 0.0019400, 0.0031677, -0.0008058, 0.0009133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004882, upper bound: 0.0004641
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004930, upper bound: 0.0005205
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875738, 0.9895834, 0.9877502, 0.9896594, -0.0014170, 0.0012176
1: -0.0043602, -0.0038595, -0.0043163, -0.0038406, -0.0003531, 0.0003034
2: 0.0103993, 0.0130529, 0.0102989, 0.0128200, -0.0016079, 0.0018712
3: -0.0072142, -0.0060064, -0.0071082, -0.0059607, -0.0008517, 0.0007318
4: 0.0025407, 0.0030543, 0.0025212, 0.0030092, -0.0003112, 0.0003622
5: 0.0120390, 0.0153766, 0.0119128, 0.0150837, -0.0020223, 0.0023534
6: -0.0023619, -0.0015148, -0.0022876, -0.0014828, -0.0005973, 0.0005133
7: -0.0092486, -0.0070569, -0.0090562, -0.0069740, -0.0015455, 0.0013280
8: -0.0044279, -0.0032753, -0.0043267, -0.0032317, -0.0008127, 0.0006984
9: 0.0019340, 0.0032705, 0.0018835, 0.0031532, -0.0008098, 0.0009424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004882, upper bound: 0.0004966
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004930, upper bound: 0.0005370
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875596, 0.9896145, 0.9877002, 0.9896435, -0.0013506, 0.0013218
1: -0.0043638, -0.0038517, -0.0043288, -0.0038445, -0.0003365, 0.0003294
2: 0.0103582, 0.0130718, 0.0103199, 0.0128861, -0.0017454, 0.0017834
3: -0.0072228, -0.0059877, -0.0071383, -0.0059703, -0.0008117, 0.0007944
4: 0.0025327, 0.0030579, 0.0025253, 0.0030220, -0.0003378, 0.0003452
5: 0.0119874, 0.0154004, 0.0119392, 0.0151668, -0.0021953, 0.0022431
6: -0.0023679, -0.0015017, -0.0023087, -0.0014895, -0.0005693, 0.0005572
7: -0.0092642, -0.0070229, -0.0091109, -0.0069913, -0.0014730, 0.0014416
8: -0.0044361, -0.0032574, -0.0043555, -0.0032408, -0.0007746, 0.0007581
9: 0.0019133, 0.0032800, 0.0018940, 0.0031865, -0.0008791, 0.0008982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0004641
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005205
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875738, 0.9896135, 0.9877133, 0.9897314, -0.0013872, 0.0013294
1: -0.0043602, -0.0038520, -0.0043255, -0.0038226, -0.0003456, 0.0003312
2: 0.0103596, 0.0130530, 0.0102040, 0.0128687, -0.0017554, 0.0018318
3: -0.0072143, -0.0059884, -0.0071304, -0.0059175, -0.0008337, 0.0007990
4: 0.0025330, 0.0030543, 0.0025028, 0.0030186, -0.0003398, 0.0003545
5: 0.0119891, 0.0153767, 0.0117933, 0.0151448, -0.0022079, 0.0023039
6: -0.0023619, -0.0015021, -0.0023031, -0.0014524, -0.0005847, 0.0005604
7: -0.0092487, -0.0070241, -0.0090964, -0.0068955, -0.0015129, 0.0014499
8: -0.0044279, -0.0032581, -0.0043479, -0.0031904, -0.0007956, 0.0007625
9: 0.0019140, 0.0032706, 0.0018356, 0.0031777, -0.0008841, 0.0009226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0004966
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005370
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9877002, 0.9896221, 0.9876893, 0.9896101, -0.0013580, 0.0012876
1: -0.0043287, -0.0038498, -0.0043315, -0.0038528, -0.0003384, 0.0003208
2: 0.0103482, 0.0128861, 0.0103640, 0.0129004, -0.0017003, 0.0017933
3: -0.0071383, -0.0059832, -0.0071448, -0.0059904, -0.0008162, 0.0007739
4: 0.0025308, 0.0030220, 0.0025338, 0.0030247, -0.0003291, 0.0003471
5: 0.0119747, 0.0151667, 0.0119947, 0.0151848, -0.0021385, 0.0022555
6: -0.0023086, -0.0014985, -0.0023132, -0.0015035, -0.0005725, 0.0005428
7: -0.0091108, -0.0070147, -0.0091227, -0.0070278, -0.0014811, 0.0014043
8: -0.0043554, -0.0032531, -0.0043617, -0.0032600, -0.0007789, 0.0007385
9: 0.0019083, 0.0031865, 0.0019162, 0.0031937, -0.0008563, 0.0009032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004372, upper bound: 0.0005332
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004995, upper bound: 0.0005357
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877134, 0.9897041, 0.9877073, 0.9896088, -0.0013653, 0.0013390
1: -0.0043255, -0.0038294, -0.0043270, -0.0038532, -0.0003402, 0.0003336
2: 0.0102400, 0.0128687, 0.0103658, 0.0128766, -0.0017681, 0.0018028
3: -0.0071304, -0.0059339, -0.0071340, -0.0059912, -0.0008206, 0.0008048
4: 0.0025098, 0.0030186, 0.0025342, 0.0030201, -0.0003422, 0.0003489
5: 0.0118386, 0.0151449, 0.0119969, 0.0151548, -0.0022238, 0.0022675
6: -0.0023031, -0.0014639, -0.0023056, -0.0015041, -0.0005755, 0.0005644
7: -0.0090965, -0.0069253, -0.0091030, -0.0070292, -0.0014890, 0.0014604
8: -0.0043479, -0.0032061, -0.0043513, -0.0032607, -0.0007831, 0.0007680
9: 0.0018538, 0.0031777, 0.0019171, 0.0031817, -0.0008905, 0.0009080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004579, upper bound: 0.0005332
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004990, upper bound: 0.0005357
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9877001, 0.9896488, 0.9876586, 0.9896741, -0.0013109, 0.0013816
1: -0.0043288, -0.0038432, -0.0043391, -0.0038369, -0.0003266, 0.0003443
2: 0.0103129, 0.0128862, 0.0102796, 0.0129409, -0.0018244, 0.0017311
3: -0.0071384, -0.0059671, -0.0071633, -0.0059519, -0.0007879, 0.0008304
4: 0.0025239, 0.0030220, 0.0025175, 0.0030326, -0.0003531, 0.0003350
5: 0.0119304, 0.0151669, 0.0118884, 0.0152358, -0.0022946, 0.0021772
6: -0.0023087, -0.0014872, -0.0023262, -0.0014766, -0.0005526, 0.0005824
7: -0.0091109, -0.0069855, -0.0091561, -0.0069580, -0.0014298, 0.0015069
8: -0.0043555, -0.0032378, -0.0043793, -0.0032233, -0.0007519, 0.0007924
9: 0.0018905, 0.0031865, 0.0018737, 0.0032141, -0.0009189, 0.0008719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004663, upper bound: 0.0005332
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005233, upper bound: 0.0005357
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877133, 0.9897305, 0.9876733, 0.9896727, -0.0013171, 0.0014386
1: -0.0043255, -0.0038228, -0.0043354, -0.0038372, -0.0003282, 0.0003585
2: 0.0102050, 0.0128688, 0.0102813, 0.0129216, -0.0018997, 0.0017392
3: -0.0071304, -0.0059180, -0.0071545, -0.0059527, -0.0007916, 0.0008646
4: 0.0025030, 0.0030186, 0.0025178, 0.0030288, -0.0003677, 0.0003366
5: 0.0117947, 0.0151450, 0.0118907, 0.0152114, -0.0023893, 0.0021875
6: -0.0023031, -0.0014528, -0.0023200, -0.0014771, -0.0005552, 0.0006064
7: -0.0090965, -0.0068964, -0.0091401, -0.0069595, -0.0014365, 0.0015690
8: -0.0043479, -0.0031909, -0.0043708, -0.0032241, -0.0007554, 0.0008251
9: 0.0018362, 0.0031778, 0.0018746, 0.0032044, -0.0009568, 0.0008760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004925, upper bound: 0.0005332
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005357, upper bound: 0.0005357
time: 0.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.95 seconds
IS_A1_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004882, upper bound: 0.0004641
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004930, upper bound: 0.0005205
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004882, upper bound: 0.0004966
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004930, upper bound: 0.0005370
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0004641
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005205
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005212, upper bound: 0.0004966
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005370
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004372, upper bound: 0.0005332
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004995, upper bound: 0.0005357
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004579, upper bound: 0.0005332
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004990, upper bound: 0.0005357
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004663, upper bound: 0.0005332
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005233, upper bound: 0.0005357
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0004925, upper bound: 0.0005332
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -0.0005357, upper bound: 0.0005357

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875688, 0.9895840, 0.9877498, 0.9895729, -0.0013459, 0.0010891
1: -0.0043615, -0.0038593, -0.0043164, -0.0038621, -0.0003354, 0.0002714
2: 0.0103985, 0.0130595, 0.0104131, 0.0128206, -0.0014382, 0.0017772
3: -0.0072173, -0.0060060, -0.0071085, -0.0060127, -0.0008089, 0.0006546
4: 0.0025405, 0.0030555, 0.0025433, 0.0030093, -0.0002784, 0.0003440
5: 0.0120380, 0.0153849, 0.0120564, 0.0150844, -0.0018089, 0.0022353
6: -0.0023640, -0.0015145, -0.0022878, -0.0015192, -0.0005673, 0.0004591
7: -0.0092541, -0.0070562, -0.0090568, -0.0070683, -0.0014679, 0.0011879
8: -0.0044308, -0.0032749, -0.0043270, -0.0032813, -0.0007719, 0.0006247
9: 0.0019336, 0.0032738, 0.0019410, 0.0031535, -0.0007244, 0.0008951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0004937
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0005031
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9876208, 0.9895818, 0.9878623, 0.9896486, -0.0013441, 0.0010725
1: -0.0043485, -0.0038599, -0.0042883, -0.0038432, -0.0003349, 0.0002672
2: 0.0104013, 0.0129909, 0.0103132, 0.0126720, -0.0014163, 0.0017749
3: -0.0071860, -0.0060074, -0.0070409, -0.0059672, -0.0008078, 0.0006446
4: 0.0025410, 0.0030422, 0.0025240, 0.0029805, -0.0002741, 0.0003435
5: 0.0120416, 0.0152986, 0.0119307, 0.0148975, -0.0017813, 0.0022323
6: -0.0023421, -0.0015154, -0.0022403, -0.0014873, -0.0005666, 0.0004521
7: -0.0091974, -0.0070586, -0.0089340, -0.0069858, -0.0014659, 0.0011698
8: -0.0044010, -0.0032762, -0.0042624, -0.0032379, -0.0007709, 0.0006152
9: 0.0019350, 0.0032393, 0.0018906, 0.0030787, -0.0007133, 0.0008939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004611, upper bound: 0.0004800
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004701, upper bound: 0.0004800
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875830, 0.9895830, 0.9877746, 0.9896591, -0.0013931, 0.0010900
1: -0.0043579, -0.0038596, -0.0043102, -0.0038406, -0.0003471, 0.0002716
2: 0.0103999, 0.0130408, 0.0102993, 0.0127879, -0.0014393, 0.0018395
3: -0.0072087, -0.0060067, -0.0070936, -0.0059609, -0.0008373, 0.0006551
4: 0.0025408, 0.0030519, 0.0025213, 0.0030030, -0.0002786, 0.0003560
5: 0.0120398, 0.0153613, 0.0119132, 0.0150433, -0.0018103, 0.0023136
6: -0.0023580, -0.0015150, -0.0022773, -0.0014829, -0.0005872, 0.0004595
7: -0.0092386, -0.0070574, -0.0090297, -0.0069743, -0.0015193, 0.0011888
8: -0.0044226, -0.0032755, -0.0043128, -0.0032318, -0.0007990, 0.0006252
9: 0.0019343, 0.0032644, 0.0018836, 0.0031370, -0.0007249, 0.0009265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004653, upper bound: 0.0005187
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0005187
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9876062, 0.9896131, 0.9878099, 0.9896294, -0.0012691, 0.0011731
1: -0.0043522, -0.0038521, -0.0043014, -0.0038480, -0.0003162, 0.0002923
2: 0.0103602, 0.0130101, 0.0103386, 0.0127412, -0.0015490, 0.0016759
3: -0.0071948, -0.0059886, -0.0070724, -0.0059788, -0.0007628, 0.0007050
4: 0.0025331, 0.0030460, 0.0025289, 0.0029939, -0.0002998, 0.0003244
5: 0.0119898, 0.0153228, 0.0119627, 0.0149845, -0.0019482, 0.0021078
6: -0.0023483, -0.0015023, -0.0022624, -0.0014954, -0.0005350, 0.0004945
7: -0.0092133, -0.0070246, -0.0089911, -0.0070067, -0.0013842, 0.0012794
8: -0.0044093, -0.0032583, -0.0042925, -0.0032489, -0.0007279, 0.0006728
9: 0.0019143, 0.0032490, 0.0019034, 0.0031135, -0.0007802, 0.0008441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004919, upper bound: 0.0004474
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005044, upper bound: 0.0004475
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875688, 0.9896141, 0.9877263, 0.9896421, -0.0013256, 0.0012081
1: -0.0043615, -0.0038518, -0.0043222, -0.0038449, -0.0003303, 0.0003010
2: 0.0103588, 0.0130596, 0.0103217, 0.0128516, -0.0015953, 0.0017504
3: -0.0072173, -0.0059880, -0.0071226, -0.0059711, -0.0007967, 0.0007261
4: 0.0025328, 0.0030556, 0.0025256, 0.0030153, -0.0003088, 0.0003388
5: 0.0119880, 0.0153850, 0.0119415, 0.0151234, -0.0020064, 0.0022016
6: -0.0023641, -0.0015019, -0.0022977, -0.0014900, -0.0005588, 0.0005092
7: -0.0092542, -0.0070234, -0.0090824, -0.0069928, -0.0014458, 0.0013176
8: -0.0044308, -0.0032577, -0.0043405, -0.0032416, -0.0007603, 0.0006929
9: 0.0019136, 0.0032739, 0.0018949, 0.0031691, -0.0008035, 0.0008816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0004937
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005031
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9876208, 0.9896119, 0.9878228, 0.9897176, -0.0013115, 0.0011857
1: -0.0043485, -0.0038524, -0.0042982, -0.0038261, -0.0003268, 0.0002955
2: 0.0103616, 0.0129909, 0.0102221, 0.0127242, -0.0015658, 0.0017319
3: -0.0071860, -0.0059893, -0.0070646, -0.0059258, -0.0007883, 0.0007127
4: 0.0025334, 0.0030423, 0.0025064, 0.0029906, -0.0003031, 0.0003352
5: 0.0119916, 0.0152986, 0.0118162, 0.0149632, -0.0019693, 0.0021783
6: -0.0023421, -0.0015028, -0.0022570, -0.0014582, -0.0005529, 0.0004998
7: -0.0091974, -0.0070258, -0.0089771, -0.0069105, -0.0014304, 0.0012932
8: -0.0044010, -0.0032589, -0.0042851, -0.0031983, -0.0007522, 0.0006801
9: 0.0019150, 0.0032393, 0.0018448, 0.0031050, -0.0007886, 0.0008723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004919, upper bound: 0.0004800
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005044, upper bound: 0.0004800
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875830, 0.9896130, 0.9877404, 0.9897311, -0.0013684, 0.0012099
1: -0.0043579, -0.0038521, -0.0043187, -0.0038227, -0.0003410, 0.0003015
2: 0.0103602, 0.0130408, 0.0102043, 0.0128329, -0.0015977, 0.0018070
3: -0.0072087, -0.0059886, -0.0071141, -0.0059177, -0.0008225, 0.0007272
4: 0.0025331, 0.0030519, 0.0025029, 0.0030117, -0.0003092, 0.0003497
5: 0.0119898, 0.0153614, 0.0117938, 0.0150999, -0.0020094, 0.0022727
6: -0.0023580, -0.0015023, -0.0022917, -0.0014526, -0.0005768, 0.0005100
7: -0.0092386, -0.0070246, -0.0090669, -0.0068958, -0.0014925, 0.0013196
8: -0.0044226, -0.0032583, -0.0043323, -0.0031906, -0.0007849, 0.0006939
9: 0.0019143, 0.0032644, 0.0018358, 0.0031597, -0.0008047, 0.0009101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004968, upper bound: 0.0005187
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005187
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9878098, 0.9895994, 0.9877390, 0.9896085, -0.0012016, 0.0012245
1: -0.0043014, -0.0038555, -0.0043191, -0.0038532, -0.0002994, 0.0003051
2: 0.0103782, 0.0127414, 0.0103661, 0.0128347, -0.0016170, 0.0015867
3: -0.0070724, -0.0059968, -0.0071149, -0.0059913, -0.0007222, 0.0007360
4: 0.0025366, 0.0029940, 0.0025342, 0.0030120, -0.0003130, 0.0003071
5: 0.0120125, 0.0149848, 0.0119973, 0.0151022, -0.0020337, 0.0019956
6: -0.0022625, -0.0015081, -0.0022923, -0.0015042, -0.0005065, 0.0005162
7: -0.0089913, -0.0070395, -0.0090684, -0.0070295, -0.0013105, 0.0013355
8: -0.0042926, -0.0032661, -0.0043331, -0.0032609, -0.0006892, 0.0007023
9: 0.0019234, 0.0031136, 0.0019173, 0.0031606, -0.0008144, 0.0007991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004269, upper bound: 0.0005212
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004269, upper bound: 0.0005324
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896207, 0.9876963, 0.9896097, -0.0012259, 0.0012624
1: -0.0043222, -0.0038502, -0.0043297, -0.0038529, -0.0003055, 0.0003146
2: 0.0103500, 0.0128516, 0.0103646, 0.0128912, -0.0016670, 0.0016188
3: -0.0071226, -0.0059840, -0.0071406, -0.0059906, -0.0007368, 0.0007588
4: 0.0025311, 0.0030153, 0.0025339, 0.0030230, -0.0003226, 0.0003133
5: 0.0119770, 0.0151233, 0.0119953, 0.0151732, -0.0020967, 0.0020360
6: -0.0022976, -0.0014991, -0.0023103, -0.0015037, -0.0005168, 0.0005322
7: -0.0090823, -0.0070162, -0.0091150, -0.0070282, -0.0013370, 0.0013768
8: -0.0043404, -0.0032539, -0.0043577, -0.0032602, -0.0007031, 0.0007241
9: 0.0019092, 0.0031691, 0.0019165, 0.0031891, -0.0008396, 0.0008153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004904, upper bound: 0.0005264
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004904, upper bound: 0.0005348
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9878225, 0.9896852, 0.9877570, 0.9896072, -0.0012139, 0.0012788
1: -0.0042983, -0.0038341, -0.0043146, -0.0038536, -0.0003025, 0.0003186
2: 0.0102648, 0.0127245, 0.0103680, 0.0128110, -0.0016886, 0.0016030
3: -0.0070648, -0.0059452, -0.0071041, -0.0059922, -0.0007296, 0.0007686
4: 0.0025146, 0.0029907, 0.0025346, 0.0030074, -0.0003268, 0.0003103
5: 0.0118698, 0.0149636, 0.0119997, 0.0150723, -0.0021238, 0.0020162
6: -0.0022571, -0.0014719, -0.0022847, -0.0015048, -0.0005117, 0.0005390
7: -0.0089774, -0.0069458, -0.0090488, -0.0070310, -0.0013240, 0.0013947
8: -0.0042853, -0.0032169, -0.0043228, -0.0032617, -0.0006963, 0.0007334
9: 0.0018663, 0.0031051, 0.0019182, 0.0031486, -0.0008505, 0.0008074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004467, upper bound: 0.0005212
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004467, upper bound: 0.0005324
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9877405, 0.9897038, 0.9877148, 0.9896083, -0.0012271, 0.0013326
1: -0.0043187, -0.0038295, -0.0043251, -0.0038533, -0.0003058, 0.0003321
2: 0.0102403, 0.0128328, 0.0103664, 0.0128668, -0.0017597, 0.0016204
3: -0.0071141, -0.0059341, -0.0071295, -0.0059915, -0.0007375, 0.0008009
4: 0.0025099, 0.0030117, 0.0025343, 0.0030182, -0.0003406, 0.0003136
5: 0.0118391, 0.0150998, 0.0119976, 0.0151425, -0.0022133, 0.0020380
6: -0.0022917, -0.0014641, -0.0023025, -0.0015043, -0.0005173, 0.0005617
7: -0.0090668, -0.0069256, -0.0090949, -0.0070297, -0.0013383, 0.0014534
8: -0.0043323, -0.0032063, -0.0043470, -0.0032610, -0.0007038, 0.0007643
9: 0.0018540, 0.0031597, 0.0019174, 0.0031768, -0.0008863, 0.0008161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004901, upper bound: 0.0005264
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004901, upper bound: 0.0005348
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9878097, 0.9896289, 0.9877069, 0.9896724, -0.0011569, 0.0013149
1: -0.0043014, -0.0038482, -0.0043271, -0.0038373, -0.0002883, 0.0003276
2: 0.0103392, 0.0127414, 0.0102817, 0.0128772, -0.0017362, 0.0015277
3: -0.0070725, -0.0059791, -0.0071343, -0.0059529, -0.0006953, 0.0007903
4: 0.0025290, 0.0029940, 0.0025179, 0.0030202, -0.0003360, 0.0002957
5: 0.0119635, 0.0149848, 0.0118911, 0.0151556, -0.0021837, 0.0019214
6: -0.0022625, -0.0014956, -0.0023058, -0.0014773, -0.0004877, 0.0005543
7: -0.0089913, -0.0070073, -0.0091035, -0.0069598, -0.0012618, 0.0014340
8: -0.0042926, -0.0032492, -0.0043516, -0.0032242, -0.0006636, 0.0007541
9: 0.0019037, 0.0031136, 0.0018748, 0.0031820, -0.0008745, 0.0007694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004554, upper bound: 0.0005212
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004554, upper bound: 0.0005324
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896474, 0.9876670, 0.9896737, -0.0011704, 0.0013551
1: -0.0043222, -0.0038435, -0.0043370, -0.0038370, -0.0002916, 0.0003377
2: 0.0103148, 0.0128517, 0.0102801, 0.0129298, -0.0017894, 0.0015455
3: -0.0071226, -0.0059680, -0.0071582, -0.0059522, -0.0007034, 0.0008144
4: 0.0025243, 0.0030153, 0.0025176, 0.0030304, -0.0003463, 0.0002991
5: 0.0119327, 0.0151235, 0.0118891, 0.0152218, -0.0022506, 0.0019438
6: -0.0022977, -0.0014878, -0.0023226, -0.0014767, -0.0004934, 0.0005712
7: -0.0090824, -0.0069871, -0.0091469, -0.0069584, -0.0012765, 0.0014779
8: -0.0043405, -0.0032386, -0.0043744, -0.0032235, -0.0006713, 0.0007772
9: 0.0018914, 0.0031691, 0.0018740, 0.0032085, -0.0009012, 0.0007784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005168, upper bound: 0.0005264
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005168, upper bound: 0.0005347
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9878225, 0.9897150, 0.9877211, 0.9896711, -0.0011684, 0.0013752
1: -0.0042983, -0.0038267, -0.0043235, -0.0038377, -0.0002911, 0.0003427
2: 0.0102256, 0.0127245, 0.0102836, 0.0128584, -0.0018159, 0.0015429
3: -0.0070648, -0.0059274, -0.0071257, -0.0059538, -0.0007022, 0.0008265
4: 0.0025070, 0.0029907, 0.0025183, 0.0030166, -0.0003515, 0.0002986
5: 0.0118206, 0.0149636, 0.0118935, 0.0151320, -0.0022840, 0.0019405
6: -0.0022571, -0.0014594, -0.0022998, -0.0014779, -0.0004925, 0.0005797
7: -0.0089774, -0.0069135, -0.0090880, -0.0069613, -0.0012743, 0.0014999
8: -0.0042853, -0.0031999, -0.0043434, -0.0032250, -0.0006701, 0.0007888
9: 0.0018465, 0.0031051, 0.0018757, 0.0031725, -0.0009146, 0.0007771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004804, upper bound: 0.0005212
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004804, upper bound: 0.0005324
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9877405, 0.9897303, 0.9876820, 0.9896724, -0.0011739, 0.0014233
1: -0.0043187, -0.0038229, -0.0043333, -0.0038374, -0.0002925, 0.0003547
2: 0.0102054, 0.0128329, 0.0102819, 0.0129101, -0.0018795, 0.0015501
3: -0.0071141, -0.0059182, -0.0071493, -0.0059530, -0.0007056, 0.0008555
4: 0.0025031, 0.0030117, 0.0025179, 0.0030266, -0.0003638, 0.0003000
5: 0.0117952, 0.0150999, 0.0118914, 0.0151970, -0.0023639, 0.0019497
6: -0.0022917, -0.0014529, -0.0023163, -0.0014773, -0.0004948, 0.0006000
7: -0.0090669, -0.0068967, -0.0091307, -0.0069600, -0.0012803, 0.0015523
8: -0.0043323, -0.0031911, -0.0043659, -0.0032243, -0.0006733, 0.0008164
9: 0.0018364, 0.0031597, 0.0018749, 0.0031986, -0.0009466, 0.0007807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005274, upper bound: 0.0005264
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005274, upper bound: 0.0005347
time: 0.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.03 seconds
IS_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0004937
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0005031
IS_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004611, upper bound: 0.0004800
IS_A1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004701, upper bound: 0.0004800
IS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004653, upper bound: 0.0005187
IS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0005187
IS_A1_B2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004919, upper bound: 0.0004474
IS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005044, upper bound: 0.0004475
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0004937
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005031
IS_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004919, upper bound: 0.0004800
IS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005044, upper bound: 0.0004800
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004968, upper bound: 0.0005187
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005187
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004269, upper bound: 0.0005212
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004269, upper bound: 0.0005324
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004904, upper bound: 0.0005264
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004904, upper bound: 0.0005348
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004467, upper bound: 0.0005212
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004467, upper bound: 0.0005324
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004901, upper bound: 0.0005264
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004901, upper bound: 0.0005348
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004554, upper bound: 0.0005212
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004554, upper bound: 0.0005324
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005168, upper bound: 0.0005264
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005168, upper bound: 0.0005347
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004804, upper bound: 0.0005212
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0004804, upper bound: 0.0005324
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005274, upper bound: 0.0005264
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -0.0005274, upper bound: 0.0005347

## BFS IS instance: IS_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9875571, 0.9895427, 0.9877504, 0.9895539, -0.0013297, 0.0010493
1: -0.0043644, -0.0038696, -0.0043163, -0.0038668, -0.0003313, 0.0002615
2: 0.0104531, 0.0130751, 0.0104382, 0.0128199, -0.0013856, 0.0017558
3: -0.0072243, -0.0060309, -0.0071082, -0.0060241, -0.0007992, 0.0006307
4: 0.0025511, 0.0030585, 0.0025482, 0.0030091, -0.0002682, 0.0003398
5: 0.0121067, 0.0154044, 0.0120879, 0.0150835, -0.0017428, 0.0022083
6: -0.0023690, -0.0015320, -0.0022875, -0.0015272, -0.0005605, 0.0004423
7: -0.0092669, -0.0071013, -0.0090561, -0.0070890, -0.0014502, 0.0011444
8: -0.0044375, -0.0032987, -0.0043267, -0.0032922, -0.0007626, 0.0006019
9: 0.0019611, 0.0032817, 0.0019536, 0.0031531, -0.0006979, 0.0008843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004274, upper bound: 0.0004902
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004624, upper bound: 0.0004902
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875838, 0.9895615, 0.9877762, 0.9896105, -0.0013437, 0.0010653
1: -0.0043578, -0.0038650, -0.0043098, -0.0038527, -0.0003348, 0.0002654
2: 0.0104283, 0.0130399, 0.0103635, 0.0127857, -0.0014067, 0.0017743
3: -0.0072083, -0.0060196, -0.0070926, -0.0059901, -0.0008076, 0.0006403
4: 0.0025463, 0.0030517, 0.0025337, 0.0030025, -0.0002723, 0.0003434
5: 0.0120755, 0.0153602, 0.0119940, 0.0150405, -0.0017693, 0.0022317
6: -0.0023578, -0.0015241, -0.0022766, -0.0015034, -0.0005664, 0.0004491
7: -0.0092379, -0.0070808, -0.0090279, -0.0070273, -0.0014655, 0.0011619
8: -0.0044222, -0.0032879, -0.0043118, -0.0032598, -0.0007707, 0.0006110
9: 0.0019486, 0.0032640, 0.0019160, 0.0031359, -0.0007085, 0.0008937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004542, upper bound: 0.0005038
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875836, 0.9895656, 0.9877623, 0.9896161, -0.0013494, 0.0010727
1: -0.0043578, -0.0038639, -0.0043133, -0.0038514, -0.0003362, 0.0002673
2: 0.0104229, 0.0130400, 0.0103562, 0.0128041, -0.0014165, 0.0017818
3: -0.0072084, -0.0060172, -0.0071010, -0.0059868, -0.0008110, 0.0006447
4: 0.0025452, 0.0030518, 0.0025323, 0.0030061, -0.0002742, 0.0003449
5: 0.0120687, 0.0153603, 0.0119848, 0.0150637, -0.0017816, 0.0022411
6: -0.0023578, -0.0015223, -0.0022825, -0.0015010, -0.0005688, 0.0004522
7: -0.0092379, -0.0070764, -0.0090431, -0.0070213, -0.0014717, 0.0011700
8: -0.0044223, -0.0032855, -0.0043198, -0.0032566, -0.0007739, 0.0006153
9: 0.0019459, 0.0032640, 0.0019123, 0.0031452, -0.0007134, 0.0008974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004274, upper bound: 0.0005036
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004624, upper bound: 0.0005038
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9876068, 0.9895954, 0.9877959, 0.9895884, -0.0012241, 0.0011563
1: -0.0043520, -0.0038565, -0.0043049, -0.0038583, -0.0003050, 0.0002881
2: 0.0103835, 0.0130094, 0.0103927, 0.0127597, -0.0015269, 0.0016165
3: -0.0071945, -0.0059992, -0.0070808, -0.0060034, -0.0007358, 0.0006950
4: 0.0025376, 0.0030458, 0.0025394, 0.0029975, -0.0002955, 0.0003129
5: 0.0120191, 0.0153219, 0.0120307, 0.0150078, -0.0019204, 0.0020331
6: -0.0023480, -0.0015098, -0.0022683, -0.0015127, -0.0005160, 0.0004874
7: -0.0092127, -0.0070438, -0.0090064, -0.0070514, -0.0013351, 0.0012611
8: -0.0044090, -0.0032684, -0.0043005, -0.0032724, -0.0007021, 0.0006632
9: 0.0019260, 0.0032486, 0.0019307, 0.0031228, -0.0007690, 0.0008141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004547, upper bound: 0.0004302
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004903, upper bound: 0.0004323
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9875701, 0.9895625, 0.9877268, 0.9896204, -0.0012950, 0.0011607
1: -0.0043612, -0.0038647, -0.0043221, -0.0038502, -0.0003227, 0.0002892
2: 0.0104270, 0.0130579, 0.0103503, 0.0128510, -0.0015326, 0.0017101
3: -0.0072165, -0.0060190, -0.0071223, -0.0059841, -0.0007783, 0.0006976
4: 0.0025460, 0.0030552, 0.0025312, 0.0030152, -0.0002966, 0.0003310
5: 0.0120739, 0.0153828, 0.0119774, 0.0151226, -0.0019277, 0.0021508
6: -0.0023635, -0.0015236, -0.0022974, -0.0014992, -0.0005459, 0.0004893
7: -0.0092527, -0.0070798, -0.0090818, -0.0070164, -0.0014124, 0.0012659
8: -0.0044300, -0.0032873, -0.0043402, -0.0032540, -0.0007428, 0.0006657
9: 0.0019480, 0.0032730, 0.0019093, 0.0031688, -0.0007719, 0.0008613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0004801
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004955, upper bound: 0.0004802
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9875570, 0.9895722, 0.9877269, 0.9896244, -0.0013005, 0.0011677
1: -0.0043644, -0.0038623, -0.0043221, -0.0038493, -0.0003240, 0.0002910
2: 0.0104141, 0.0130752, 0.0103451, 0.0128509, -0.0015419, 0.0017172
3: -0.0072244, -0.0060132, -0.0071223, -0.0059818, -0.0007816, 0.0007018
4: 0.0025435, 0.0030586, 0.0025302, 0.0030151, -0.0002984, 0.0003324
5: 0.0120577, 0.0154046, 0.0119709, 0.0151225, -0.0019394, 0.0021598
6: -0.0023690, -0.0015195, -0.0022974, -0.0014975, -0.0005482, 0.0004922
7: -0.0092670, -0.0070691, -0.0090817, -0.0070122, -0.0014183, 0.0012735
8: -0.0044376, -0.0032817, -0.0043401, -0.0032518, -0.0007459, 0.0006697
9: 0.0019415, 0.0032817, 0.0019067, 0.0031687, -0.0007766, 0.0008649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0004902
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004955, upper bound: 0.0004902
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9876213, 0.9895943, 0.9878080, 0.9896755, -0.0012650, 0.0011685
1: -0.0043484, -0.0038568, -0.0043018, -0.0038365, -0.0003152, 0.0002912
2: 0.0103849, 0.0129902, 0.0102776, 0.0127435, -0.0015430, 0.0016704
3: -0.0071857, -0.0059999, -0.0070734, -0.0059510, -0.0007603, 0.0007023
4: 0.0025379, 0.0030421, 0.0025171, 0.0029944, -0.0002986, 0.0003233
5: 0.0120209, 0.0152978, 0.0118860, 0.0149875, -0.0019407, 0.0021010
6: -0.0023419, -0.0015102, -0.0022631, -0.0014760, -0.0005332, 0.0004926
7: -0.0091968, -0.0070450, -0.0089931, -0.0069564, -0.0013797, 0.0012744
8: -0.0044007, -0.0032690, -0.0042935, -0.0032224, -0.0007256, 0.0006702
9: 0.0019268, 0.0032389, 0.0018727, 0.0031147, -0.0007771, 0.0008413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004547, upper bound: 0.0004583
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004903, upper bound: 0.0004636
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875836, 0.9895917, 0.9877421, 0.9896808, -0.0013198, 0.0011856
1: -0.0043578, -0.0038574, -0.0043183, -0.0038352, -0.0003289, 0.0002954
2: 0.0103883, 0.0130400, 0.0102707, 0.0128306, -0.0015656, 0.0017428
3: -0.0072084, -0.0060014, -0.0071131, -0.0059479, -0.0007932, 0.0007126
4: 0.0025385, 0.0030518, 0.0025158, 0.0030112, -0.0003030, 0.0003373
5: 0.0120252, 0.0153603, 0.0118773, 0.0150970, -0.0019691, 0.0021920
6: -0.0023578, -0.0015113, -0.0022909, -0.0014737, -0.0005563, 0.0004998
7: -0.0092379, -0.0070478, -0.0090650, -0.0069507, -0.0014394, 0.0012931
8: -0.0044223, -0.0032705, -0.0043313, -0.0032194, -0.0007570, 0.0006800
9: 0.0019285, 0.0032640, 0.0018692, 0.0031586, -0.0007885, 0.0008778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004827, upper bound: 0.0005038
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875836, 0.9895954, 0.9877261, 0.9896885, -0.0013238, 0.0011847
1: -0.0043578, -0.0038565, -0.0043223, -0.0038333, -0.0003299, 0.0002952
2: 0.0103835, 0.0130401, 0.0102606, 0.0128518, -0.0015644, 0.0017481
3: -0.0072084, -0.0059992, -0.0071227, -0.0059433, -0.0007956, 0.0007121
4: 0.0025376, 0.0030518, 0.0025138, 0.0030153, -0.0003028, 0.0003383
5: 0.0120192, 0.0153604, 0.0118646, 0.0151237, -0.0019676, 0.0021986
6: -0.0023578, -0.0015098, -0.0022977, -0.0014705, -0.0005580, 0.0004994
7: -0.0092380, -0.0070438, -0.0090825, -0.0069423, -0.0014438, 0.0012921
8: -0.0044223, -0.0032684, -0.0043406, -0.0032150, -0.0007593, 0.0006795
9: 0.0019260, 0.0032640, 0.0018642, 0.0031692, -0.0007879, 0.0008804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004955, upper bound: 0.0005038
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9878098, 0.9895994, 0.9876447, 0.9895108, -0.0010711, 0.0013066
1: -0.0043014, -0.0038555, -0.0043426, -0.0038776, -0.0002669, 0.0003256
2: 0.0103782, 0.0127414, 0.0104952, 0.0129593, -0.0017253, 0.0014144
3: -0.0070724, -0.0059968, -0.0071716, -0.0060501, -0.0006438, 0.0007853
4: 0.0025366, 0.0029940, 0.0025592, 0.0030361, -0.0003339, 0.0002738
5: 0.0120125, 0.0149848, 0.0121596, 0.0152588, -0.0021700, 0.0017790
6: -0.0022625, -0.0015081, -0.0023320, -0.0015454, -0.0004515, 0.0005508
7: -0.0089913, -0.0070395, -0.0091713, -0.0071361, -0.0011682, 0.0014250
8: -0.0042926, -0.0032661, -0.0043872, -0.0033169, -0.0006144, 0.0007494
9: 0.0019234, 0.0031136, 0.0019823, 0.0032234, -0.0008690, 0.0007124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0004919
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0005043
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9878098, 0.9895994, 0.9877396, 0.9895579, -0.0010916, 0.0012241
1: -0.0043014, -0.0038555, -0.0043189, -0.0038658, -0.0002720, 0.0003050
2: 0.0103782, 0.0127414, 0.0104329, 0.0128341, -0.0016164, 0.0014414
3: -0.0070724, -0.0059968, -0.0071146, -0.0060217, -0.0006561, 0.0007357
4: 0.0025366, 0.0029940, 0.0025472, 0.0030119, -0.0003128, 0.0002790
5: 0.0120125, 0.0149848, 0.0120813, 0.0151014, -0.0020330, 0.0018130
6: -0.0022625, -0.0015081, -0.0022921, -0.0015255, -0.0004601, 0.0005160
7: -0.0089913, -0.0070395, -0.0090679, -0.0070847, -0.0011905, 0.0013350
8: -0.0042926, -0.0032661, -0.0043329, -0.0032899, -0.0006261, 0.0007021
9: 0.0019234, 0.0031136, 0.0019509, 0.0031603, -0.0008141, 0.0007260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0005020
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0005157
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896207, 0.9876089, 0.9895119, -0.0011069, 0.0013761
1: -0.0043222, -0.0038502, -0.0043515, -0.0038773, -0.0002758, 0.0003429
2: 0.0103500, 0.0128516, 0.0104937, 0.0130065, -0.0018171, 0.0014616
3: -0.0071226, -0.0059840, -0.0071931, -0.0060494, -0.0006653, 0.0008271
4: 0.0025311, 0.0030153, 0.0025589, 0.0030453, -0.0003517, 0.0002829
5: 0.0119770, 0.0151233, 0.0121578, 0.0153182, -0.0022854, 0.0018383
6: -0.0022976, -0.0014991, -0.0023471, -0.0015450, -0.0004666, 0.0005801
7: -0.0090823, -0.0070162, -0.0092103, -0.0071349, -0.0012072, 0.0015008
8: -0.0043404, -0.0032539, -0.0044077, -0.0033163, -0.0006348, 0.0007893
9: 0.0019092, 0.0031691, 0.0019816, 0.0032471, -0.0009152, 0.0007361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004622, upper bound: 0.0005096
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004720, upper bound: 0.0005096
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896207, 0.9876968, 0.9895591, -0.0011428, 0.0012619
1: -0.0043222, -0.0038502, -0.0043296, -0.0038655, -0.0002848, 0.0003144
2: 0.0103500, 0.0128516, 0.0104314, 0.0128906, -0.0016664, 0.0015091
3: -0.0071226, -0.0059840, -0.0071403, -0.0060210, -0.0006869, 0.0007585
4: 0.0025311, 0.0030153, 0.0025469, 0.0030228, -0.0003225, 0.0002921
5: 0.0119770, 0.0151233, 0.0120794, 0.0151724, -0.0020959, 0.0018981
6: -0.0022976, -0.0014991, -0.0023101, -0.0015250, -0.0004818, 0.0005320
7: -0.0090823, -0.0070162, -0.0091145, -0.0070834, -0.0012464, 0.0013763
8: -0.0043404, -0.0032539, -0.0043574, -0.0032892, -0.0006555, 0.0007238
9: 0.0019092, 0.0031691, 0.0019502, 0.0031887, -0.0008393, 0.0007601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004622, upper bound: 0.0005182
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004720, upper bound: 0.0005183
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9878225, 0.9896852, 0.9876594, 0.9895096, -0.0010839, 0.0013451
1: -0.0042983, -0.0038341, -0.0043389, -0.0038779, -0.0002701, 0.0003352
2: 0.0102648, 0.0127245, 0.0104967, 0.0129400, -0.0017761, 0.0014312
3: -0.0070648, -0.0059452, -0.0071628, -0.0060508, -0.0006514, 0.0008084
4: 0.0025146, 0.0029907, 0.0025595, 0.0030324, -0.0003438, 0.0002770
5: 0.0118698, 0.0149636, 0.0121615, 0.0152346, -0.0022339, 0.0018001
6: -0.0022571, -0.0014719, -0.0023259, -0.0015459, -0.0004569, 0.0005670
7: -0.0089774, -0.0069458, -0.0091553, -0.0071373, -0.0011821, 0.0014670
8: -0.0042853, -0.0032169, -0.0043788, -0.0033176, -0.0006217, 0.0007715
9: 0.0018663, 0.0031051, 0.0019831, 0.0032136, -0.0008946, 0.0007208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0004919
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0005044
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9878225, 0.9896852, 0.9877576, 0.9895566, -0.0010973, 0.0012783
1: -0.0042983, -0.0038341, -0.0043145, -0.0038662, -0.0002734, 0.0003185
2: 0.0102648, 0.0127245, 0.0104347, 0.0128103, -0.0016880, 0.0014490
3: -0.0070648, -0.0059452, -0.0071038, -0.0060225, -0.0006595, 0.0007683
4: 0.0025146, 0.0029907, 0.0025475, 0.0030073, -0.0003267, 0.0002804
5: 0.0118698, 0.0149636, 0.0120836, 0.0150715, -0.0021231, 0.0018224
6: -0.0022571, -0.0014719, -0.0022845, -0.0015261, -0.0004626, 0.0005389
7: -0.0089774, -0.0069458, -0.0090482, -0.0070861, -0.0011968, 0.0013942
8: -0.0042853, -0.0032169, -0.0043225, -0.0032907, -0.0006294, 0.0007332
9: 0.0018663, 0.0031051, 0.0019518, 0.0031483, -0.0008502, 0.0007298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0005020
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0005157
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9877405, 0.9897038, 0.9876245, 0.9895108, -0.0011083, 0.0014138
1: -0.0043187, -0.0038295, -0.0043476, -0.0038776, -0.0002762, 0.0003523
2: 0.0102403, 0.0128328, 0.0104952, 0.0129861, -0.0018669, 0.0014635
3: -0.0071141, -0.0059341, -0.0071838, -0.0060501, -0.0006661, 0.0008497
4: 0.0025099, 0.0030117, 0.0025592, 0.0030413, -0.0003613, 0.0002833
5: 0.0118391, 0.0150998, 0.0121597, 0.0152926, -0.0023481, 0.0018407
6: -0.0022917, -0.0014641, -0.0023406, -0.0015454, -0.0004672, 0.0005960
7: -0.0090668, -0.0069256, -0.0091934, -0.0071361, -0.0012087, 0.0015420
8: -0.0043323, -0.0032063, -0.0043989, -0.0033170, -0.0006357, 0.0008109
9: 0.0018540, 0.0031597, 0.0019823, 0.0032369, -0.0009403, 0.0007371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0004968
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0005096
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9877405, 0.9897038, 0.9877153, 0.9895578, -0.0011416, 0.0013322
1: -0.0043187, -0.0038295, -0.0043250, -0.0038659, -0.0002845, 0.0003319
2: 0.0102403, 0.0128328, 0.0104332, 0.0128661, -0.0017591, 0.0015075
3: -0.0071141, -0.0059341, -0.0071292, -0.0060218, -0.0006861, 0.0008007
4: 0.0025099, 0.0030117, 0.0025472, 0.0030181, -0.0003405, 0.0002918
5: 0.0118391, 0.0150998, 0.0120816, 0.0151417, -0.0022125, 0.0018960
6: -0.0022917, -0.0014641, -0.0023023, -0.0015256, -0.0004812, 0.0005616
7: -0.0090668, -0.0069256, -0.0090943, -0.0070849, -0.0012451, 0.0014529
8: -0.0043323, -0.0032063, -0.0043468, -0.0032900, -0.0006548, 0.0007641
9: 0.0018540, 0.0031597, 0.0019511, 0.0031764, -0.0008860, 0.0007592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0005037
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0005182
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9878097, 0.9896289, 0.9876065, 0.9895796, -0.0010259, 0.0014024
1: -0.0043014, -0.0038482, -0.0043521, -0.0038605, -0.0002556, 0.0003494
2: 0.0103392, 0.0127414, 0.0104044, 0.0130098, -0.0018519, 0.0013547
3: -0.0070725, -0.0059791, -0.0071946, -0.0060087, -0.0006166, 0.0008429
4: 0.0025290, 0.0029940, 0.0025416, 0.0030459, -0.0003584, 0.0002622
5: 0.0119635, 0.0149848, 0.0120454, 0.0153224, -0.0023292, 0.0017039
6: -0.0022625, -0.0014956, -0.0023482, -0.0015164, -0.0004325, 0.0005912
7: -0.0089913, -0.0070073, -0.0092130, -0.0070611, -0.0011189, 0.0015295
8: -0.0042926, -0.0032492, -0.0044092, -0.0032775, -0.0005884, 0.0008044
9: 0.0019037, 0.0031136, 0.0019366, 0.0032488, -0.0009327, 0.0006823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0004919
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0005044
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9878097, 0.9896289, 0.9877074, 0.9896151, -0.0010588, 0.0013144
1: -0.0043014, -0.0038482, -0.0043269, -0.0038516, -0.0002638, 0.0003275
2: 0.0103392, 0.0127414, 0.0103575, 0.0128765, -0.0017356, 0.0013981
3: -0.0070725, -0.0059791, -0.0071339, -0.0059874, -0.0006364, 0.0007900
4: 0.0025290, 0.0029940, 0.0025326, 0.0030201, -0.0003359, 0.0002706
5: 0.0119635, 0.0149848, 0.0119864, 0.0151547, -0.0021829, 0.0017585
6: -0.0022625, -0.0014956, -0.0023056, -0.0015015, -0.0004463, 0.0005541
7: -0.0089913, -0.0070073, -0.0091029, -0.0070223, -0.0011548, 0.0014335
8: -0.0042926, -0.0032492, -0.0043513, -0.0032571, -0.0006073, 0.0007539
9: 0.0019037, 0.0031136, 0.0019129, 0.0031816, -0.0008741, 0.0007042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004306, upper bound: 0.0005157
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0005157
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896474, 0.9875691, 0.9895806, -0.0010530, 0.0014623
1: -0.0043222, -0.0038435, -0.0043614, -0.0038602, -0.0002624, 0.0003644
2: 0.0103148, 0.0128517, 0.0104030, 0.0130592, -0.0019309, 0.0013905
3: -0.0071226, -0.0059680, -0.0072171, -0.0060081, -0.0006329, 0.0008789
4: 0.0025243, 0.0030153, 0.0025414, 0.0030555, -0.0003737, 0.0002691
5: 0.0119327, 0.0151235, 0.0120437, 0.0153845, -0.0024285, 0.0017489
6: -0.0022977, -0.0014878, -0.0023639, -0.0015160, -0.0004439, 0.0006164
7: -0.0090824, -0.0069871, -0.0092538, -0.0070599, -0.0011485, 0.0015948
8: -0.0043405, -0.0032386, -0.0044306, -0.0032769, -0.0006040, 0.0008387
9: 0.0018914, 0.0031691, 0.0019359, 0.0032737, -0.0009725, 0.0007003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004887, upper bound: 0.0005096
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004994, upper bound: 0.0005096
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896474, 0.9876676, 0.9896163, -0.0011051, 0.0013546
1: -0.0043222, -0.0038435, -0.0043369, -0.0038513, -0.0002754, 0.0003375
2: 0.0103148, 0.0128517, 0.0103559, 0.0129291, -0.0017887, 0.0014593
3: -0.0071226, -0.0059680, -0.0071579, -0.0059867, -0.0006642, 0.0008141
4: 0.0025243, 0.0030153, 0.0025322, 0.0030303, -0.0003462, 0.0002824
5: 0.0119327, 0.0151235, 0.0119844, 0.0152209, -0.0022497, 0.0018354
6: -0.0022977, -0.0014878, -0.0023224, -0.0015009, -0.0004658, 0.0005710
7: -0.0090824, -0.0069871, -0.0091464, -0.0070210, -0.0012053, 0.0014774
8: -0.0043405, -0.0032386, -0.0043741, -0.0032564, -0.0006338, 0.0007769
9: 0.0018914, 0.0031691, 0.0019121, 0.0032082, -0.0009009, 0.0007350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004887, upper bound: 0.0005182
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004994, upper bound: 0.0005183
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9878225, 0.9897150, 0.9876210, 0.9895785, -0.0010378, 0.0014519
1: -0.0042983, -0.0038267, -0.0043485, -0.0038607, -0.0002586, 0.0003618
2: 0.0102256, 0.0127245, 0.0104058, 0.0129906, -0.0019172, 0.0013704
3: -0.0070648, -0.0059274, -0.0071859, -0.0060094, -0.0006237, 0.0008726
4: 0.0025070, 0.0029907, 0.0025419, 0.0030422, -0.0003711, 0.0002652
5: 0.0118206, 0.0149636, 0.0120472, 0.0152982, -0.0024113, 0.0017235
6: -0.0022571, -0.0014594, -0.0023420, -0.0015169, -0.0004375, 0.0006120
7: -0.0089774, -0.0069135, -0.0091971, -0.0070623, -0.0011318, 0.0015835
8: -0.0042853, -0.0031999, -0.0044008, -0.0032781, -0.0005952, 0.0008327
9: 0.0018465, 0.0031051, 0.0019373, 0.0032391, -0.0009656, 0.0006902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0004919
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0005044
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9878225, 0.9897150, 0.9877216, 0.9896137, -0.0010615, 0.0013747
1: -0.0042983, -0.0038267, -0.0043234, -0.0038520, -0.0002645, 0.0003425
2: 0.0102256, 0.0127245, 0.0103594, 0.0128577, -0.0018153, 0.0014017
3: -0.0070648, -0.0059274, -0.0071254, -0.0059883, -0.0006380, 0.0008263
4: 0.0025070, 0.0029907, 0.0025329, 0.0030165, -0.0003513, 0.0002713
5: 0.0118206, 0.0149636, 0.0119888, 0.0151311, -0.0022832, 0.0017630
6: -0.0022571, -0.0014594, -0.0022996, -0.0015021, -0.0004475, 0.0005795
7: -0.0089774, -0.0069135, -0.0090874, -0.0070239, -0.0011577, 0.0014993
8: -0.0042853, -0.0031999, -0.0043431, -0.0032579, -0.0006088, 0.0007885
9: 0.0018465, 0.0031051, 0.0019139, 0.0031722, -0.0009143, 0.0007060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0005020
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0005157
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9877405, 0.9897303, 0.9875833, 0.9895796, -0.0010567, 0.0015084
1: -0.0043187, -0.0038229, -0.0043579, -0.0038605, -0.0002633, 0.0003759
2: 0.0102054, 0.0128329, 0.0104044, 0.0130404, -0.0019919, 0.0013954
3: -0.0071141, -0.0059182, -0.0072085, -0.0060087, -0.0006351, 0.0009066
4: 0.0025031, 0.0030117, 0.0025416, 0.0030518, -0.0003855, 0.0002701
5: 0.0117952, 0.0150999, 0.0120454, 0.0153608, -0.0025052, 0.0017550
6: -0.0022917, -0.0014529, -0.0023579, -0.0015164, -0.0004454, 0.0006359
7: -0.0090669, -0.0068967, -0.0092382, -0.0070611, -0.0011525, 0.0016451
8: -0.0043323, -0.0031911, -0.0044225, -0.0032775, -0.0006061, 0.0008652
9: 0.0018364, 0.0031597, 0.0019366, 0.0032642, -0.0010032, 0.0007028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0004968
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005096
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9877405, 0.9897303, 0.9876825, 0.9896149, -0.0011005, 0.0014228
1: -0.0043187, -0.0038229, -0.0043332, -0.0038516, -0.0002742, 0.0003545
2: 0.0102054, 0.0128329, 0.0103577, 0.0129094, -0.0018788, 0.0014532
3: -0.0071141, -0.0059182, -0.0071489, -0.0059875, -0.0006614, 0.0008552
4: 0.0025031, 0.0030117, 0.0025326, 0.0030265, -0.0003636, 0.0002813
5: 0.0117952, 0.0150999, 0.0119867, 0.0151961, -0.0023631, 0.0018277
6: -0.0022917, -0.0014529, -0.0023161, -0.0015015, -0.0004639, 0.0005998
7: -0.0090669, -0.0068967, -0.0091301, -0.0070225, -0.0012002, 0.0015518
8: -0.0043323, -0.0031911, -0.0043656, -0.0032572, -0.0006312, 0.0008161
9: 0.0018364, 0.0031597, 0.0019131, 0.0031982, -0.0009463, 0.0007319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005036
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005183
time: 0.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.02 seconds
IS_A1_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004274, upper bound: 0.0004902
IS_A1_B1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004624, upper bound: 0.0004902
IS_A1_B1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
IS_A1_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004542, upper bound: 0.0005038
IS_A1_B1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004274, upper bound: 0.0005036
IS_A1_B1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004624, upper bound: 0.0005038
IS_A1_B2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004547, upper bound: 0.0004302
IS_A1_B2_B1_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004903, upper bound: 0.0004323
IS_A1_B2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0004801
IS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004955, upper bound: 0.0004802
IS_A1_B2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0004902
IS_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004955, upper bound: 0.0004902
IS_A1_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004547, upper bound: 0.0004583
IS_A1_B2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004903, upper bound: 0.0004636
IS_A1_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
IS_A1_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004827, upper bound: 0.0005038
IS_A1_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
IS_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004955, upper bound: 0.0005038
IS_A2_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0004919
IS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0005043
IS_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0005020
IS_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004102, upper bound: 0.0005157
IS_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004622, upper bound: 0.0005096
IS_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004720, upper bound: 0.0005096
IS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004622, upper bound: 0.0005182
IS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004720, upper bound: 0.0005183
IS_A2_B1_A2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0004919
IS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0005044
IS_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0005020
IS_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004320, upper bound: 0.0005157
IS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0004968
IS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0005096
IS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0005037
IS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004714, upper bound: 0.0005182
IS_A2_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0004919
IS_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0005044
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004306, upper bound: 0.0005157
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004383, upper bound: 0.0005157
IS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004887, upper bound: 0.0005096
IS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004994, upper bound: 0.0005096
IS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004887, upper bound: 0.0005182
IS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004994, upper bound: 0.0005183
IS_A2_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0004919
IS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0005044
IS_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0005020
IS_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0004650, upper bound: 0.0005157
IS_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0004968
IS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005096
IS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005036
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005183

## BFS IS instance: IS_A1_B1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9875853, 0.9895276, 0.9877893, 0.9895267, -0.0012673, 0.0010333
1: -0.0043574, -0.0038734, -0.0043065, -0.0038736, -0.0003158, 0.0002575
2: 0.0104729, 0.0130378, 0.0104741, 0.0127684, -0.0013645, 0.0016734
3: -0.0072073, -0.0060399, -0.0070848, -0.0060405, -0.0007617, 0.0006211
4: 0.0025549, 0.0030513, 0.0025551, 0.0029992, -0.0002641, 0.0003239
5: 0.0121316, 0.0153575, 0.0121331, 0.0150188, -0.0017162, 0.0021047
6: -0.0023571, -0.0015383, -0.0022711, -0.0015387, -0.0005342, 0.0004356
7: -0.0092361, -0.0071177, -0.0090136, -0.0071187, -0.0013821, 0.0011270
8: -0.0044213, -0.0033073, -0.0043043, -0.0033078, -0.0007269, 0.0005927
9: 0.0019711, 0.0032629, 0.0019717, 0.0031272, -0.0006872, 0.0008428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875845, 0.9895546, 0.9877784, 0.9895886, -0.0012894, 0.0010553
1: -0.0043576, -0.0038667, -0.0043093, -0.0038582, -0.0003213, 0.0002629
2: 0.0104372, 0.0130389, 0.0103925, 0.0127828, -0.0013934, 0.0017026
3: -0.0072079, -0.0060237, -0.0070913, -0.0060033, -0.0007749, 0.0006342
4: 0.0025480, 0.0030515, 0.0025393, 0.0030020, -0.0002697, 0.0003295
5: 0.0120868, 0.0153589, 0.0120305, 0.0150369, -0.0017526, 0.0021414
6: -0.0023574, -0.0015269, -0.0022757, -0.0015126, -0.0005435, 0.0004448
7: -0.0092370, -0.0070882, -0.0090255, -0.0070513, -0.0014062, 0.0011509
8: -0.0044218, -0.0032918, -0.0043106, -0.0032723, -0.0007395, 0.0006052
9: 0.0019531, 0.0032634, 0.0019306, 0.0031345, -0.0007018, 0.0008575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875852, 0.9895318, 0.9877746, 0.9895334, -0.0012729, 0.0010567
1: -0.0043574, -0.0038724, -0.0043102, -0.0038720, -0.0003172, 0.0002633
2: 0.0104675, 0.0130378, 0.0104653, 0.0127878, -0.0013954, 0.0016809
3: -0.0072074, -0.0060375, -0.0070936, -0.0060365, -0.0007651, 0.0006351
4: 0.0025538, 0.0030513, 0.0025534, 0.0030029, -0.0002701, 0.0003253
5: 0.0121248, 0.0153576, 0.0121221, 0.0150432, -0.0017550, 0.0021141
6: -0.0023571, -0.0015366, -0.0022773, -0.0015359, -0.0005366, 0.0004454
7: -0.0092362, -0.0071132, -0.0090296, -0.0071114, -0.0013883, 0.0011525
8: -0.0044214, -0.0033049, -0.0043127, -0.0033040, -0.0007301, 0.0006061
9: 0.0019683, 0.0032629, 0.0019673, 0.0031370, -0.0007028, 0.0008466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875845, 0.9895588, 0.9877642, 0.9895952, -0.0012950, 0.0010626
1: -0.0043576, -0.0038656, -0.0043128, -0.0038565, -0.0003227, 0.0002648
2: 0.0104318, 0.0130390, 0.0103836, 0.0128014, -0.0014031, 0.0017100
3: -0.0072079, -0.0060212, -0.0070998, -0.0059993, -0.0007783, 0.0006386
4: 0.0025469, 0.0030516, 0.0025376, 0.0030056, -0.0002716, 0.0003310
5: 0.0120800, 0.0153590, 0.0120193, 0.0150603, -0.0017647, 0.0021508
6: -0.0023575, -0.0015252, -0.0022816, -0.0015098, -0.0005459, 0.0004479
7: -0.0092371, -0.0070838, -0.0090409, -0.0070439, -0.0014124, 0.0011589
8: -0.0044218, -0.0032894, -0.0043187, -0.0032685, -0.0007428, 0.0006094
9: 0.0019504, 0.0032635, 0.0019261, 0.0031439, -0.0007067, 0.0008613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004603, upper bound: 0.0005038
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004603, upper bound: 0.0005038
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875709, 0.9895554, 0.9877291, 0.9895970, -0.0012380, 0.0011510
1: -0.0043610, -0.0038665, -0.0043216, -0.0038561, -0.0003085, 0.0002868
2: 0.0104363, 0.0130569, 0.0103814, 0.0128480, -0.0015199, 0.0016347
3: -0.0072160, -0.0060232, -0.0071210, -0.0059983, -0.0007441, 0.0006918
4: 0.0025478, 0.0030550, 0.0025372, 0.0030146, -0.0002942, 0.0003164
5: 0.0120855, 0.0153815, 0.0120165, 0.0151188, -0.0019116, 0.0020561
6: -0.0023632, -0.0015266, -0.0022965, -0.0015091, -0.0005219, 0.0004852
7: -0.0092519, -0.0070874, -0.0090793, -0.0070421, -0.0013502, 0.0012553
8: -0.0044296, -0.0032913, -0.0043389, -0.0032675, -0.0007101, 0.0006602
9: 0.0019526, 0.0032725, 0.0019250, 0.0031673, -0.0007655, 0.0008233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004802
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004802
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9875576, 0.9895653, 0.9877291, 0.9896011, -0.0012518, 0.0011581
1: -0.0043643, -0.0038640, -0.0043215, -0.0038551, -0.0003119, 0.0002886
2: 0.0104233, 0.0130743, 0.0103760, 0.0128478, -0.0015292, 0.0016530
3: -0.0072240, -0.0060173, -0.0071209, -0.0059958, -0.0007524, 0.0006960
4: 0.0025453, 0.0030584, 0.0025361, 0.0030146, -0.0002960, 0.0003199
5: 0.0120692, 0.0154034, 0.0120097, 0.0151187, -0.0019234, 0.0020790
6: -0.0023687, -0.0015225, -0.0022964, -0.0015074, -0.0005277, 0.0004882
7: -0.0092662, -0.0070767, -0.0090792, -0.0070376, -0.0013653, 0.0012631
8: -0.0044372, -0.0032857, -0.0043388, -0.0032652, -0.0007180, 0.0006642
9: 0.0019461, 0.0032813, 0.0019223, 0.0031672, -0.0007702, 0.0008325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004902
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004902
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9875853, 0.9895588, 0.9877498, 0.9895988, -0.0012430, 0.0011473
1: -0.0043574, -0.0038656, -0.0043164, -0.0038557, -0.0003097, 0.0002859
2: 0.0104318, 0.0130378, 0.0103790, 0.0128206, -0.0015150, 0.0016414
3: -0.0072074, -0.0060212, -0.0071085, -0.0059972, -0.0007471, 0.0006895
4: 0.0025469, 0.0030513, 0.0025367, 0.0030093, -0.0002932, 0.0003177
5: 0.0120799, 0.0153576, 0.0120135, 0.0150844, -0.0019054, 0.0020644
6: -0.0023571, -0.0015252, -0.0022878, -0.0015083, -0.0005240, 0.0004836
7: -0.0092361, -0.0070837, -0.0090567, -0.0070401, -0.0013557, 0.0012513
8: -0.0044213, -0.0032894, -0.0043270, -0.0032665, -0.0007129, 0.0006580
9: 0.0019504, 0.0032629, 0.0019238, 0.0031535, -0.0007630, 0.0008267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875844, 0.9895848, 0.9877443, 0.9896570, -0.0012620, 0.0011759
1: -0.0043576, -0.0038592, -0.0043177, -0.0038411, -0.0003145, 0.0002930
2: 0.0103975, 0.0130390, 0.0103020, 0.0128277, -0.0015528, 0.0016665
3: -0.0072079, -0.0060056, -0.0071117, -0.0059622, -0.0007585, 0.0007068
4: 0.0025403, 0.0030516, 0.0025218, 0.0030107, -0.0003005, 0.0003225
5: 0.0120367, 0.0153591, 0.0119167, 0.0150933, -0.0019530, 0.0020960
6: -0.0023575, -0.0015142, -0.0022900, -0.0014838, -0.0005320, 0.0004957
7: -0.0092371, -0.0070554, -0.0090626, -0.0069765, -0.0013764, 0.0012825
8: -0.0044218, -0.0032745, -0.0043301, -0.0032330, -0.0007238, 0.0006745
9: 0.0019331, 0.0032635, 0.0018850, 0.0031571, -0.0007821, 0.0008393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004828, upper bound: 0.0005038
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004828, upper bound: 0.0005038
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875851, 0.9895623, 0.9877360, 0.9896058, -0.0012470, 0.0011512
1: -0.0043574, -0.0038647, -0.0043198, -0.0038539, -0.0003107, 0.0002868
2: 0.0104271, 0.0130379, 0.0103696, 0.0128388, -0.0015201, 0.0016467
3: -0.0072074, -0.0060191, -0.0071168, -0.0059929, -0.0007495, 0.0006919
4: 0.0025460, 0.0030513, 0.0025349, 0.0030128, -0.0002942, 0.0003187
5: 0.0120740, 0.0153577, 0.0120017, 0.0151073, -0.0019119, 0.0020711
6: -0.0023571, -0.0015237, -0.0022936, -0.0015053, -0.0005257, 0.0004853
7: -0.0092362, -0.0070799, -0.0090718, -0.0070323, -0.0013600, 0.0012555
8: -0.0044214, -0.0032874, -0.0043349, -0.0032624, -0.0007152, 0.0006603
9: 0.0019480, 0.0032629, 0.0019190, 0.0031627, -0.0007656, 0.0008294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875844, 0.9895885, 0.9877284, 0.9896649, -0.0012646, 0.0011747
1: -0.0043576, -0.0038582, -0.0043217, -0.0038392, -0.0003151, 0.0002927
2: 0.0103927, 0.0130390, 0.0102916, 0.0128489, -0.0015512, 0.0016699
3: -0.0072079, -0.0060034, -0.0071214, -0.0059574, -0.0007601, 0.0007060
4: 0.0025394, 0.0030516, 0.0025198, 0.0030148, -0.0003002, 0.0003232
5: 0.0120307, 0.0153591, 0.0119036, 0.0151200, -0.0019510, 0.0021004
6: -0.0023575, -0.0015127, -0.0022968, -0.0014804, -0.0005331, 0.0004952
7: -0.0092371, -0.0070514, -0.0090801, -0.0069679, -0.0013793, 0.0012812
8: -0.0044219, -0.0032724, -0.0043393, -0.0032285, -0.0007253, 0.0006738
9: 0.0019307, 0.0032635, 0.0018798, 0.0031678, -0.0007813, 0.0008411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005038
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005038
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877957, 0.9895604, 0.9876453, 0.9894928, -0.0010547, 0.0012614
1: -0.0043049, -0.0038652, -0.0043424, -0.0038821, -0.0002628, 0.0003143
2: 0.0104296, 0.0127599, 0.0105189, 0.0129586, -0.0016657, 0.0013927
3: -0.0070809, -0.0060202, -0.0071713, -0.0060609, -0.0006339, 0.0007581
4: 0.0025465, 0.0029975, 0.0025638, 0.0030360, -0.0003224, 0.0002696
5: 0.0120772, 0.0150081, 0.0121895, 0.0152580, -0.0020950, 0.0017517
6: -0.0022684, -0.0015245, -0.0023318, -0.0015530, -0.0004446, 0.0005317
7: -0.0090066, -0.0070819, -0.0091707, -0.0071557, -0.0011503, 0.0013757
8: -0.0043006, -0.0032885, -0.0043870, -0.0033272, -0.0006049, 0.0007235
9: 0.0019493, 0.0031229, 0.0019942, 0.0032230, -0.0008389, 0.0007014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004067, upper bound: 0.0004547
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004090, upper bound: 0.0004903
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9878108, 0.9895462, 0.9877400, 0.9895364, -0.0010711, 0.0011745
1: -0.0043012, -0.0038688, -0.0043188, -0.0038712, -0.0002669, 0.0002926
2: 0.0104485, 0.0127400, 0.0104613, 0.0128334, -0.0015509, 0.0014143
3: -0.0070718, -0.0060288, -0.0071143, -0.0060347, -0.0006437, 0.0007059
4: 0.0025502, 0.0029937, 0.0025527, 0.0030118, -0.0003002, 0.0002737
5: 0.0121009, 0.0149830, 0.0121170, 0.0151005, -0.0019506, 0.0017789
6: -0.0022620, -0.0015305, -0.0022918, -0.0015346, -0.0004515, 0.0004951
7: -0.0089902, -0.0070975, -0.0090673, -0.0071081, -0.0011681, 0.0012809
8: -0.0042920, -0.0032966, -0.0043326, -0.0033022, -0.0006143, 0.0006736
9: 0.0019588, 0.0031129, 0.0019652, 0.0031600, -0.0007811, 0.0007123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002758, upper bound: 0.0004347
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004659
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877957, 0.9895604, 0.9877401, 0.9895398, -0.0010731, 0.0011828
1: -0.0043049, -0.0038652, -0.0043188, -0.0038704, -0.0002674, 0.0002947
2: 0.0104296, 0.0127599, 0.0104569, 0.0128334, -0.0015618, 0.0014170
3: -0.0070809, -0.0060202, -0.0071143, -0.0060327, -0.0006450, 0.0007109
4: 0.0025465, 0.0029975, 0.0025518, 0.0030118, -0.0003023, 0.0002743
5: 0.0120772, 0.0150081, 0.0121115, 0.0151005, -0.0019644, 0.0017822
6: -0.0022684, -0.0015245, -0.0022918, -0.0015332, -0.0004523, 0.0004986
7: -0.0090066, -0.0070819, -0.0090673, -0.0071045, -0.0011703, 0.0012900
8: -0.0043006, -0.0032885, -0.0043326, -0.0033003, -0.0006155, 0.0006784
9: 0.0019493, 0.0031229, 0.0019630, 0.0031600, -0.0007866, 0.0007137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002758, upper bound: 0.0004436
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004796
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9877269, 0.9895985, 0.9876102, 0.9894627, -0.0010587, 0.0013446
1: -0.0043221, -0.0038557, -0.0043512, -0.0038896, -0.0002638, 0.0003350
2: 0.0103793, 0.0128509, 0.0105587, 0.0130049, -0.0017756, 0.0013980
3: -0.0071223, -0.0059973, -0.0071924, -0.0060790, -0.0006363, 0.0008082
4: 0.0025368, 0.0030152, 0.0025715, 0.0030450, -0.0003437, 0.0002706
5: 0.0120139, 0.0151225, 0.0122396, 0.0153161, -0.0022332, 0.0017583
6: -0.0022974, -0.0015084, -0.0023466, -0.0015657, -0.0004463, 0.0005668
7: -0.0090817, -0.0070404, -0.0092089, -0.0071886, -0.0011546, 0.0014665
8: -0.0043401, -0.0032666, -0.0044070, -0.0033445, -0.0006072, 0.0007712
9: 0.0019239, 0.0031688, 0.0020143, 0.0032463, -0.0008943, 0.0007041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004546, upper bound: 0.0004631
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004546, upper bound: 0.0004955
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9877269, 0.9896038, 0.9875972, 0.9894700, -0.0010664, 0.0013547
1: -0.0043221, -0.0038544, -0.0043544, -0.0038878, -0.0002657, 0.0003375
2: 0.0103723, 0.0128508, 0.0105491, 0.0130220, -0.0017888, 0.0014082
3: -0.0071222, -0.0059941, -0.0072002, -0.0060746, -0.0006409, 0.0008142
4: 0.0025354, 0.0030151, 0.0025696, 0.0030483, -0.0003462, 0.0002725
5: 0.0120051, 0.0151224, 0.0122274, 0.0153377, -0.0022499, 0.0017711
6: -0.0022974, -0.0015062, -0.0023520, -0.0015626, -0.0004495, 0.0005710
7: -0.0090817, -0.0070346, -0.0092231, -0.0071806, -0.0011631, 0.0014775
8: -0.0043401, -0.0032636, -0.0044145, -0.0033403, -0.0006116, 0.0007770
9: 0.0019204, 0.0031687, 0.0020094, 0.0032550, -0.0009010, 0.0007092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004626, upper bound: 0.0004631
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004626, upper bound: 0.0004955
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9877269, 0.9895985, 0.9876981, 0.9895071, -0.0010939, 0.0012345
1: -0.0043221, -0.0038557, -0.0043293, -0.0038785, -0.0002726, 0.0003076
2: 0.0103793, 0.0128509, 0.0105000, 0.0128888, -0.0016301, 0.0014445
3: -0.0071223, -0.0059973, -0.0071395, -0.0060523, -0.0006575, 0.0007420
4: 0.0025368, 0.0030152, 0.0025601, 0.0030225, -0.0003155, 0.0002796
5: 0.0120139, 0.0151225, 0.0121657, 0.0151702, -0.0020502, 0.0018168
6: -0.0022974, -0.0015084, -0.0023095, -0.0015470, -0.0004611, 0.0005204
7: -0.0090817, -0.0070404, -0.0091131, -0.0071401, -0.0011931, 0.0013464
8: -0.0043401, -0.0032666, -0.0043566, -0.0033190, -0.0006274, 0.0007080
9: 0.0019239, 0.0031688, 0.0019847, 0.0031879, -0.0008210, 0.0007275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003579, upper bound: 0.0004535
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004323, upper bound: 0.0004811
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9877269, 0.9896038, 0.9876871, 0.9895155, -0.0011036, 0.0012373
1: -0.0043221, -0.0038544, -0.0043320, -0.0038764, -0.0002750, 0.0003083
2: 0.0103723, 0.0128508, 0.0104890, 0.0129033, -0.0016339, 0.0014573
3: -0.0071222, -0.0059941, -0.0071461, -0.0060473, -0.0006633, 0.0007437
4: 0.0025354, 0.0030151, 0.0025580, 0.0030253, -0.0003162, 0.0002821
5: 0.0120051, 0.0151224, 0.0121518, 0.0151884, -0.0020550, 0.0018330
6: -0.0022974, -0.0015062, -0.0023141, -0.0015434, -0.0004652, 0.0005216
7: -0.0090817, -0.0070346, -0.0091250, -0.0071310, -0.0012037, 0.0013495
8: -0.0043401, -0.0032636, -0.0043629, -0.0033142, -0.0006330, 0.0007097
9: 0.0019204, 0.0031687, 0.0019792, 0.0031951, -0.0008229, 0.0007340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003680, upper bound: 0.0004535
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004418, upper bound: 0.0004811
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9878079, 0.9896451, 0.9876599, 0.9894918, -0.0010670, 0.0013022
1: -0.0043019, -0.0038441, -0.0043388, -0.0038823, -0.0002659, 0.0003245
2: 0.0103178, 0.0127439, 0.0105204, 0.0129393, -0.0017196, 0.0014089
3: -0.0070736, -0.0059693, -0.0071625, -0.0060615, -0.0006413, 0.0007827
4: 0.0025249, 0.0029944, 0.0025641, 0.0030323, -0.0003328, 0.0002727
5: 0.0119365, 0.0149879, 0.0121913, 0.0152337, -0.0021627, 0.0017720
6: -0.0022633, -0.0014888, -0.0023257, -0.0015535, -0.0004498, 0.0005489
7: -0.0089934, -0.0069895, -0.0091548, -0.0071569, -0.0011637, 0.0014202
8: -0.0042937, -0.0032399, -0.0043786, -0.0033279, -0.0006120, 0.0007469
9: 0.0018929, 0.0031149, 0.0019950, 0.0032133, -0.0008661, 0.0007096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004270, upper bound: 0.0004547
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004310, upper bound: 0.0004903
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9878241, 0.9896323, 0.9877582, 0.9895351, -0.0010765, 0.0012304
1: -0.0042979, -0.0038473, -0.0043143, -0.0038715, -0.0002682, 0.0003066
2: 0.0103348, 0.0127225, 0.0104631, 0.0128096, -0.0016247, 0.0014215
3: -0.0070638, -0.0059771, -0.0071035, -0.0060355, -0.0006470, 0.0007395
4: 0.0025282, 0.0029903, 0.0025530, 0.0030072, -0.0003145, 0.0002751
5: 0.0119579, 0.0149610, 0.0121193, 0.0150706, -0.0020434, 0.0017879
6: -0.0022564, -0.0014942, -0.0022842, -0.0015352, -0.0004538, 0.0005186
7: -0.0089757, -0.0070036, -0.0090476, -0.0071096, -0.0011741, 0.0013419
8: -0.0042844, -0.0032473, -0.0043222, -0.0033030, -0.0006174, 0.0007057
9: 0.0019015, 0.0031041, 0.0019662, 0.0031480, -0.0008183, 0.0007159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003524, upper bound: 0.0003323
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004081, upper bound: 0.0004659
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9878079, 0.9896451, 0.9877580, 0.9895384, -0.0010786, 0.0012386
1: -0.0043019, -0.0038441, -0.0043143, -0.0038707, -0.0002688, 0.0003086
2: 0.0103178, 0.0127439, 0.0104586, 0.0128097, -0.0016355, 0.0014243
3: -0.0070736, -0.0059693, -0.0071035, -0.0060334, -0.0006483, 0.0007444
4: 0.0025249, 0.0029944, 0.0025521, 0.0030072, -0.0003166, 0.0002757
5: 0.0119365, 0.0149879, 0.0121137, 0.0150706, -0.0020571, 0.0017913
6: -0.0022633, -0.0014888, -0.0022843, -0.0015337, -0.0004547, 0.0005221
7: -0.0089934, -0.0069895, -0.0090477, -0.0071059, -0.0011763, 0.0013509
8: -0.0042937, -0.0032399, -0.0043222, -0.0033011, -0.0006186, 0.0007104
9: 0.0018929, 0.0031149, 0.0019639, 0.0031480, -0.0008237, 0.0007173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003524, upper bound: 0.0003400
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004081, upper bound: 0.0004796
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9877422, 0.9896514, 0.9876250, 0.9894905, -0.0010823, 0.0013614
1: -0.0043183, -0.0038426, -0.0043475, -0.0038826, -0.0002697, 0.0003392
2: 0.0103096, 0.0128306, 0.0105220, 0.0129853, -0.0017978, 0.0014292
3: -0.0071130, -0.0059656, -0.0071835, -0.0060623, -0.0006505, 0.0008183
4: 0.0025233, 0.0030112, 0.0025644, 0.0030412, -0.0003480, 0.0002766
5: 0.0119262, 0.0150969, 0.0121933, 0.0152915, -0.0022611, 0.0017975
6: -0.0022909, -0.0014862, -0.0023403, -0.0015540, -0.0004562, 0.0005739
7: -0.0090650, -0.0069828, -0.0091927, -0.0071582, -0.0011804, 0.0014848
8: -0.0043313, -0.0032363, -0.0043985, -0.0033286, -0.0006208, 0.0007809
9: 0.0018888, 0.0031585, 0.0019958, 0.0032364, -0.0009055, 0.0007198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0004362
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004671, upper bound: 0.0004827
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896629, 0.9876249, 0.9894928, -0.0010853, 0.0013676
1: -0.0043223, -0.0038397, -0.0043475, -0.0038821, -0.0002704, 0.0003408
2: 0.0102944, 0.0128518, 0.0105189, 0.0129854, -0.0018059, 0.0014331
3: -0.0071227, -0.0059587, -0.0071835, -0.0060609, -0.0006523, 0.0008220
4: 0.0025203, 0.0030153, 0.0025638, 0.0030412, -0.0003495, 0.0002774
5: 0.0119070, 0.0151236, 0.0121895, 0.0152916, -0.0022714, 0.0018025
6: -0.0022977, -0.0014813, -0.0023403, -0.0015530, -0.0004575, 0.0005765
7: -0.0090825, -0.0069702, -0.0091928, -0.0071557, -0.0011837, 0.0014916
8: -0.0043405, -0.0032297, -0.0043986, -0.0033272, -0.0006225, 0.0007844
9: 0.0018812, 0.0031692, 0.0019943, 0.0032365, -0.0009096, 0.0007218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0004631
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004671, upper bound: 0.0004955
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9877422, 0.9896514, 0.9877159, 0.9895362, -0.0011150, 0.0012847
1: -0.0043183, -0.0038426, -0.0043248, -0.0038713, -0.0002778, 0.0003201
2: 0.0103096, 0.0128306, 0.0104616, 0.0128653, -0.0016964, 0.0014723
3: -0.0071130, -0.0059656, -0.0071288, -0.0060348, -0.0006701, 0.0007721
4: 0.0025233, 0.0030112, 0.0025527, 0.0030179, -0.0003283, 0.0002850
5: 0.0119262, 0.0150969, 0.0121174, 0.0151406, -0.0021336, 0.0018518
6: -0.0022909, -0.0014862, -0.0023020, -0.0015347, -0.0004700, 0.0005415
7: -0.0090650, -0.0069828, -0.0090936, -0.0071084, -0.0012161, 0.0014011
8: -0.0043313, -0.0032363, -0.0043464, -0.0033024, -0.0006395, 0.0007368
9: 0.0018888, 0.0031585, 0.0019654, 0.0031760, -0.0008544, 0.0007415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003677, upper bound: 0.0004450
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004410, upper bound: 0.0004674
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877262, 0.9896629, 0.9877158, 0.9895397, -0.0011148, 0.0012924
1: -0.0043223, -0.0038397, -0.0043248, -0.0038704, -0.0002778, 0.0003220
2: 0.0102944, 0.0128518, 0.0104571, 0.0128654, -0.0017066, 0.0014721
3: -0.0071227, -0.0059587, -0.0071289, -0.0060327, -0.0006700, 0.0007768
4: 0.0025203, 0.0030153, 0.0025518, 0.0030180, -0.0003303, 0.0002849
5: 0.0119070, 0.0151236, 0.0121117, 0.0151407, -0.0021465, 0.0018515
6: -0.0022977, -0.0014813, -0.0023020, -0.0015333, -0.0004699, 0.0005448
7: -0.0090825, -0.0069702, -0.0090937, -0.0071046, -0.0012159, 0.0014096
8: -0.0043405, -0.0032297, -0.0043464, -0.0033004, -0.0006394, 0.0007413
9: 0.0018812, 0.0031692, 0.0019631, 0.0031761, -0.0008596, 0.0007414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003677, upper bound: 0.0004535
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004412, upper bound: 0.0004811
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877958, 0.9895883, 0.9876071, 0.9895619, -0.0010199, 0.0013632
1: -0.0043049, -0.0038583, -0.0043520, -0.0038648, -0.0002541, 0.0003397
2: 0.0103928, 0.0127599, 0.0104277, 0.0130091, -0.0018001, 0.0013467
3: -0.0070809, -0.0060035, -0.0071943, -0.0060193, -0.0006130, 0.0008193
4: 0.0025394, 0.0029975, 0.0025461, 0.0030458, -0.0003484, 0.0002607
5: 0.0120309, 0.0150081, 0.0120747, 0.0153215, -0.0022641, 0.0016938
6: -0.0022684, -0.0015127, -0.0023479, -0.0015239, -0.0004299, 0.0005746
7: -0.0090066, -0.0070515, -0.0092124, -0.0070803, -0.0011123, 0.0014868
8: -0.0043006, -0.0032725, -0.0044089, -0.0032876, -0.0005850, 0.0007819
9: 0.0019307, 0.0031229, 0.0019483, 0.0032484, -0.0009066, 0.0006783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004301, upper bound: 0.0004547
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004323, upper bound: 0.0004903
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9878102, 0.9896075, 0.9877087, 0.9895607, -0.0010136, 0.0012942
1: -0.0043013, -0.0038535, -0.0043266, -0.0038652, -0.0002526, 0.0003225
2: 0.0103675, 0.0127408, 0.0104293, 0.0128748, -0.0017090, 0.0013385
3: -0.0070722, -0.0059919, -0.0071332, -0.0060201, -0.0006092, 0.0007778
4: 0.0025345, 0.0029939, 0.0025465, 0.0030198, -0.0003308, 0.0002591
5: 0.0119990, 0.0149841, 0.0120767, 0.0151526, -0.0021494, 0.0016835
6: -0.0022623, -0.0015046, -0.0023051, -0.0015244, -0.0004273, 0.0005455
7: -0.0089909, -0.0070306, -0.0091015, -0.0070816, -0.0011055, 0.0014115
8: -0.0042923, -0.0032615, -0.0043505, -0.0032883, -0.0005814, 0.0007423
9: 0.0019180, 0.0031133, 0.0019491, 0.0031808, -0.0008607, 0.0006741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003600, upper bound: 0.0003466
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004071, upper bound: 0.0004795
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9878101, 0.9896122, 0.9876927, 0.9895735, -0.0010206, 0.0012961
1: -0.0043013, -0.0038523, -0.0043306, -0.0038620, -0.0002543, 0.0003229
2: 0.0103613, 0.0127408, 0.0104124, 0.0128960, -0.0017114, 0.0013477
3: -0.0070722, -0.0059892, -0.0071428, -0.0060124, -0.0006134, 0.0007790
4: 0.0025333, 0.0029938, 0.0025432, 0.0030239, -0.0003312, 0.0002608
5: 0.0119913, 0.0149840, 0.0120556, 0.0151792, -0.0021525, 0.0016951
6: -0.0022623, -0.0015027, -0.0023118, -0.0015190, -0.0004302, 0.0005463
7: -0.0089908, -0.0070255, -0.0091190, -0.0070677, -0.0011131, 0.0014135
8: -0.0042923, -0.0032588, -0.0043597, -0.0032810, -0.0005854, 0.0007434
9: 0.0019149, 0.0031133, 0.0019406, 0.0031915, -0.0008620, 0.0006788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002834, upper bound: 0.0004436
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004132, upper bound: 0.0004796
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9877267, 0.9896252, 0.9875705, 0.9895292, -0.0010041, 0.0014344
1: -0.0043221, -0.0038491, -0.0043611, -0.0038730, -0.0002502, 0.0003574
2: 0.0103441, 0.0128510, 0.0104710, 0.0130574, -0.0018941, 0.0013259
3: -0.0071223, -0.0059813, -0.0072163, -0.0060390, -0.0006035, 0.0008621
4: 0.0025300, 0.0030152, 0.0025545, 0.0030551, -0.0003666, 0.0002566
5: 0.0119695, 0.0151226, 0.0121292, 0.0153823, -0.0023823, 0.0016676
6: -0.0022975, -0.0014972, -0.0023634, -0.0015377, -0.0004233, 0.0006047
7: -0.0090818, -0.0070112, -0.0092523, -0.0071161, -0.0010951, 0.0015644
8: -0.0043402, -0.0032513, -0.0044299, -0.0033064, -0.0005759, 0.0008227
9: 0.0019062, 0.0031688, 0.0019701, 0.0032728, -0.0009540, 0.0006678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004801, upper bound: 0.0004631
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004802, upper bound: 0.0004955
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9877268, 0.9896301, 0.9875573, 0.9895387, -0.0010111, 0.0014467
1: -0.0043221, -0.0038479, -0.0043643, -0.0038706, -0.0002519, 0.0003605
2: 0.0103377, 0.0128509, 0.0104584, 0.0130747, -0.0019104, 0.0013352
3: -0.0071223, -0.0059784, -0.0072242, -0.0060333, -0.0006077, 0.0008695
4: 0.0025287, 0.0030152, 0.0025521, 0.0030585, -0.0003697, 0.0002584
5: 0.0119615, 0.0151225, 0.0121133, 0.0154040, -0.0024027, 0.0016793
6: -0.0022974, -0.0014951, -0.0023689, -0.0015337, -0.0004262, 0.0006098
7: -0.0090817, -0.0070060, -0.0092666, -0.0071057, -0.0011028, 0.0015778
8: -0.0043401, -0.0032485, -0.0044374, -0.0033009, -0.0005799, 0.0008298
9: 0.0019030, 0.0031688, 0.0019638, 0.0032815, -0.0009622, 0.0006725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004902, upper bound: 0.0004631
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004902, upper bound: 0.0004955
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9877267, 0.9896252, 0.9876689, 0.9895619, -0.0010552, 0.0013264
1: -0.0043221, -0.0038491, -0.0043365, -0.0038649, -0.0002629, 0.0003305
2: 0.0103441, 0.0128510, 0.0104277, 0.0129273, -0.0017515, 0.0013934
3: -0.0071223, -0.0059813, -0.0071571, -0.0060194, -0.0006342, 0.0007972
4: 0.0025300, 0.0030152, 0.0025461, 0.0030299, -0.0003390, 0.0002697
5: 0.0119695, 0.0151226, 0.0120747, 0.0152186, -0.0022029, 0.0017526
6: -0.0022975, -0.0014972, -0.0023218, -0.0015239, -0.0004448, 0.0005591
7: -0.0090818, -0.0070112, -0.0091448, -0.0070803, -0.0011509, 0.0014466
8: -0.0043402, -0.0032513, -0.0043733, -0.0032876, -0.0006052, 0.0007608
9: 0.0019062, 0.0031688, 0.0019483, 0.0032072, -0.0008821, 0.0007018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003791, upper bound: 0.0004535
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004598, upper bound: 0.0004811
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9877268, 0.9896301, 0.9876534, 0.9895746, -0.0010657, 0.0013364
1: -0.0043221, -0.0038479, -0.0043404, -0.0038617, -0.0002655, 0.0003330
2: 0.0103377, 0.0128509, 0.0104108, 0.0129479, -0.0017647, 0.0014072
3: -0.0071223, -0.0059784, -0.0071664, -0.0060117, -0.0006405, 0.0008032
4: 0.0025287, 0.0030152, 0.0025429, 0.0030339, -0.0003416, 0.0002724
5: 0.0119615, 0.0151225, 0.0120536, 0.0152445, -0.0022195, 0.0017699
6: -0.0022974, -0.0014951, -0.0023284, -0.0015185, -0.0004492, 0.0005633
7: -0.0090817, -0.0070060, -0.0091618, -0.0070664, -0.0011623, 0.0014575
8: -0.0043401, -0.0032485, -0.0043823, -0.0032803, -0.0006112, 0.0007665
9: 0.0019030, 0.0031688, 0.0019398, 0.0032176, -0.0008888, 0.0007088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003868, upper bound: 0.0004535
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004713, upper bound: 0.0004811
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9878079, 0.9896736, 0.9876215, 0.9895608, -0.0010296, 0.0014120
1: -0.0043019, -0.0038370, -0.0043483, -0.0038651, -0.0002565, 0.0003518
2: 0.0102802, 0.0127439, 0.0104291, 0.0129899, -0.0018646, 0.0013596
3: -0.0070736, -0.0059522, -0.0071856, -0.0060200, -0.0006188, 0.0008487
4: 0.0025176, 0.0029944, 0.0025464, 0.0030421, -0.0003609, 0.0002631
5: 0.0118892, 0.0149879, 0.0120765, 0.0152973, -0.0023451, 0.0017100
6: -0.0022633, -0.0014768, -0.0023418, -0.0015243, -0.0004340, 0.0005952
7: -0.0089933, -0.0069585, -0.0091966, -0.0070815, -0.0011229, 0.0015400
8: -0.0042937, -0.0032236, -0.0044005, -0.0032882, -0.0005905, 0.0008099
9: 0.0018740, 0.0031149, 0.0019490, 0.0032388, -0.0009391, 0.0006847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004583, upper bound: 0.0004547
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004635, upper bound: 0.0004903
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9878241, 0.9896627, 0.9877223, 0.9895912, -0.0010408, 0.0013303
1: -0.0042979, -0.0038397, -0.0043232, -0.0038575, -0.0002593, 0.0003315
2: 0.0102947, 0.0127225, 0.0103889, 0.0128569, -0.0017566, 0.0013744
3: -0.0070638, -0.0059588, -0.0071250, -0.0060017, -0.0006256, 0.0007995
4: 0.0025204, 0.0029903, 0.0025386, 0.0030163, -0.0003400, 0.0002660
5: 0.0119074, 0.0149610, 0.0120259, 0.0151300, -0.0022093, 0.0017286
6: -0.0022564, -0.0014814, -0.0022993, -0.0015115, -0.0004387, 0.0005608
7: -0.0089757, -0.0069705, -0.0090867, -0.0070483, -0.0011352, 0.0014508
8: -0.0042844, -0.0032298, -0.0043428, -0.0032708, -0.0005970, 0.0007630
9: 0.0018813, 0.0031041, 0.0019288, 0.0031718, -0.0008847, 0.0006922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003719, upper bound: 0.0003323
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004420, upper bound: 0.0004659
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9878079, 0.9896736, 0.9877222, 0.9895965, -0.0010501, 0.0013386
1: -0.0043019, -0.0038370, -0.0043233, -0.0038562, -0.0002616, 0.0003335
2: 0.0102802, 0.0127439, 0.0103821, 0.0128570, -0.0017676, 0.0013866
3: -0.0070736, -0.0059522, -0.0071251, -0.0059986, -0.0006311, 0.0008045
4: 0.0025176, 0.0029944, 0.0025373, 0.0030163, -0.0003421, 0.0002684
5: 0.0118892, 0.0149879, 0.0120174, 0.0151301, -0.0022232, 0.0017440
6: -0.0022633, -0.0014768, -0.0022994, -0.0015093, -0.0004426, 0.0005643
7: -0.0089933, -0.0069585, -0.0090868, -0.0070427, -0.0011452, 0.0014600
8: -0.0042937, -0.0032236, -0.0043428, -0.0032678, -0.0006023, 0.0007678
9: 0.0018740, 0.0031149, 0.0019253, 0.0031718, -0.0008903, 0.0006984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003719, upper bound: 0.0003400
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004421, upper bound: 0.0004796
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9877421, 0.9896787, 0.9875839, 0.9895584, -0.0010296, 0.0014586
1: -0.0043183, -0.0038357, -0.0043577, -0.0038657, -0.0002566, 0.0003634
2: 0.0102734, 0.0128307, 0.0104323, 0.0130396, -0.0019260, 0.0013596
3: -0.0071131, -0.0059491, -0.0072082, -0.0060215, -0.0006188, 0.0008766
4: 0.0025163, 0.0030112, 0.0025470, 0.0030517, -0.0003728, 0.0002632
5: 0.0118807, 0.0150971, 0.0120806, 0.0153598, -0.0024224, 0.0017100
6: -0.0022910, -0.0014746, -0.0023576, -0.0015253, -0.0004340, 0.0006148
7: -0.0090650, -0.0069529, -0.0092376, -0.0070841, -0.0011230, 0.0015908
8: -0.0043314, -0.0032206, -0.0044221, -0.0032896, -0.0005906, 0.0008366
9: 0.0018706, 0.0031586, 0.0019506, 0.0032638, -0.0009700, 0.0006848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005036, upper bound: 0.0004363
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005038, upper bound: 0.0004828
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877260, 0.9896894, 0.9875839, 0.9895619, -0.0010403, 0.0014648
1: -0.0043223, -0.0038331, -0.0043577, -0.0038648, -0.0002592, 0.0003650
2: 0.0102594, 0.0128519, 0.0104277, 0.0130396, -0.0019342, 0.0013737
3: -0.0071227, -0.0059428, -0.0072082, -0.0060193, -0.0006252, 0.0008804
4: 0.0025136, 0.0030153, 0.0025461, 0.0030517, -0.0003744, 0.0002659
5: 0.0118631, 0.0151237, 0.0120747, 0.0153599, -0.0024327, 0.0017277
6: -0.0022977, -0.0014701, -0.0023577, -0.0015239, -0.0004385, 0.0006174
7: -0.0090825, -0.0069413, -0.0092376, -0.0070803, -0.0011346, 0.0015975
8: -0.0043406, -0.0032145, -0.0044221, -0.0032876, -0.0005967, 0.0008401
9: 0.0018636, 0.0031692, 0.0019483, 0.0032638, -0.0009742, 0.0006919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005036, upper bound: 0.0004631
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005038, upper bound: 0.0004955
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9877421, 0.9896787, 0.9876831, 0.9895924, -0.0010725, 0.0013749
1: -0.0043183, -0.0038357, -0.0043330, -0.0038572, -0.0002672, 0.0003426
2: 0.0102734, 0.0128307, 0.0103873, 0.0129086, -0.0018155, 0.0014162
3: -0.0071131, -0.0059491, -0.0071485, -0.0060010, -0.0006446, 0.0008264
4: 0.0025163, 0.0030112, 0.0025383, 0.0030263, -0.0003514, 0.0002741
5: 0.0118807, 0.0150971, 0.0120239, 0.0151950, -0.0022835, 0.0017812
6: -0.0022910, -0.0014746, -0.0023158, -0.0015110, -0.0004521, 0.0005796
7: -0.0090650, -0.0069529, -0.0091294, -0.0070470, -0.0011697, 0.0014995
8: -0.0043314, -0.0032206, -0.0043652, -0.0032701, -0.0006151, 0.0007886
9: 0.0018706, 0.0031586, 0.0019280, 0.0031978, -0.0009144, 0.0007133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004450
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004815, upper bound: 0.0004674
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877260, 0.9896894, 0.9876831, 0.9895976, -0.0010835, 0.0013826
1: -0.0043223, -0.0038331, -0.0043330, -0.0038559, -0.0002700, 0.0003445
2: 0.0102594, 0.0128519, 0.0103805, 0.0129086, -0.0018257, 0.0014307
3: -0.0071227, -0.0059428, -0.0071486, -0.0059979, -0.0006512, 0.0008310
4: 0.0025136, 0.0030153, 0.0025370, 0.0030263, -0.0003534, 0.0002769
5: 0.0118631, 0.0151237, 0.0120154, 0.0151951, -0.0022963, 0.0017995
6: -0.0022977, -0.0014701, -0.0023159, -0.0015088, -0.0004567, 0.0005828
7: -0.0090825, -0.0069413, -0.0091294, -0.0070413, -0.0011817, 0.0015079
8: -0.0043406, -0.0032145, -0.0043652, -0.0032671, -0.0006214, 0.0007930
9: 0.0018636, 0.0031692, 0.0019245, 0.0031978, -0.0009195, 0.0007206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004535
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004815, upper bound: 0.0004811
time: 0.60 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.33 seconds
IS_A1_B1_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
IS_A1_B1_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
IS_A1_B1_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
IS_A1_B1_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
IS_A1_B1_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
IS_A1_B1_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
IS_A1_B1_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004603, upper bound: 0.0005038
IS_A1_B1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004603, upper bound: 0.0005038
IS_A1_B2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004802
IS_A1_B2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004802
IS_A1_B2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004902
IS_A1_B2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0004902
IS_A1_B2_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
IS_A1_B2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
IS_A1_B2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004828, upper bound: 0.0005038
IS_A1_B2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004828, upper bound: 0.0005038
IS_A1_B2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
IS_A1_B2_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
IS_A1_B2_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005038
IS_A1_B2_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005038
IS_A2_B1_A1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004067, upper bound: 0.0004547
IS_A2_B1_A1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004090, upper bound: 0.0004903
IS_A2_B1_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0002758, upper bound: 0.0004347
IS_A2_B1_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004659
IS_A2_B1_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0002758, upper bound: 0.0004436
IS_A2_B1_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004796
IS_A2_B1_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004546, upper bound: 0.0004631
IS_A2_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004546, upper bound: 0.0004955
IS_A2_B1_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004626, upper bound: 0.0004631
IS_A2_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004626, upper bound: 0.0004955
IS_A2_B1_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003579, upper bound: 0.0004535
IS_A2_B1_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004323, upper bound: 0.0004811
IS_A2_B1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003680, upper bound: 0.0004535
IS_A2_B1_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004418, upper bound: 0.0004811
IS_A2_B1_A2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004270, upper bound: 0.0004547
IS_A2_B1_A2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004310, upper bound: 0.0004903
IS_A2_B1_A2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003524, upper bound: 0.0003323
IS_A2_B1_A2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004081, upper bound: 0.0004659
IS_A2_B1_A2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003524, upper bound: 0.0003400
IS_A2_B1_A2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004081, upper bound: 0.0004796
IS_A2_B1_A2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0004362
IS_A2_B1_A2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004671, upper bound: 0.0004827
IS_A2_B1_A2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0004631
IS_A2_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004671, upper bound: 0.0004955
IS_A2_B1_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003677, upper bound: 0.0004450
IS_A2_B1_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004410, upper bound: 0.0004674
IS_A2_B1_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003677, upper bound: 0.0004535
IS_A2_B1_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004412, upper bound: 0.0004811
IS_A2_B2_A1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004301, upper bound: 0.0004547
IS_A2_B2_A1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004323, upper bound: 0.0004903
IS_A2_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003600, upper bound: 0.0003466
IS_A2_B2_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004071, upper bound: 0.0004795
IS_A2_B2_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0002834, upper bound: 0.0004436
IS_A2_B2_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004132, upper bound: 0.0004796
IS_A2_B2_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004801, upper bound: 0.0004631
IS_A2_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004802, upper bound: 0.0004955
IS_A2_B2_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004902, upper bound: 0.0004631
IS_A2_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004902, upper bound: 0.0004955
IS_A2_B2_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003791, upper bound: 0.0004535
IS_A2_B2_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004598, upper bound: 0.0004811
IS_A2_B2_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003868, upper bound: 0.0004535
IS_A2_B2_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004713, upper bound: 0.0004811
IS_A2_B2_A2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004583, upper bound: 0.0004547
IS_A2_B2_A2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004635, upper bound: 0.0004903
IS_A2_B2_A2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003719, upper bound: 0.0003323
IS_A2_B2_A2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004420, upper bound: 0.0004659
IS_A2_B2_A2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003719, upper bound: 0.0003400
IS_A2_B2_A2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004421, upper bound: 0.0004796
IS_A2_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0005036, upper bound: 0.0004363
IS_A2_B2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0005038, upper bound: 0.0004828
IS_A2_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0005036, upper bound: 0.0004631
IS_A2_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0005038, upper bound: 0.0004955
IS_A2_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004450
IS_A2_B2_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004815, upper bound: 0.0004674
IS_A2_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0003860, upper bound: 0.0004535
IS_A2_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.33
Output dim: 0, lower bound: -0.0004815, upper bound: 0.0004811

## BFS IS instance: IS_A1_B1_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9875853, 0.9895276, 0.9876896, 0.9894649, -0.0012004, 0.0010950
1: -0.0043574, -0.0038734, -0.0043314, -0.0038890, -0.0002991, 0.0002728
2: 0.0104729, 0.0130378, 0.0105557, 0.0129001, -0.0014460, 0.0015851
3: -0.0072073, -0.0060399, -0.0071447, -0.0060776, -0.0007215, 0.0006581
4: 0.0025549, 0.0030513, 0.0025709, 0.0030247, -0.0002799, 0.0003068
5: 0.0121316, 0.0153575, 0.0122358, 0.0151843, -0.0018186, 0.0019936
6: -0.0023571, -0.0015383, -0.0023131, -0.0015647, -0.0005060, 0.0004616
7: -0.0092361, -0.0071177, -0.0091223, -0.0071861, -0.0013092, 0.0011943
8: -0.0044213, -0.0033073, -0.0043615, -0.0033432, -0.0006885, 0.0006281
9: 0.0019711, 0.0032629, 0.0020128, 0.0031935, -0.0007283, 0.0007983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0004772
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875853, 0.9895276, 0.9877896, 0.9894959, -0.0012686, 0.0010330
1: -0.0043574, -0.0038734, -0.0043065, -0.0038813, -0.0003161, 0.0002574
2: 0.0104729, 0.0130378, 0.0105148, 0.0127680, -0.0013641, 0.0016752
3: -0.0072073, -0.0060399, -0.0070845, -0.0060590, -0.0007625, 0.0006209
4: 0.0025549, 0.0030513, 0.0025630, 0.0029991, -0.0002640, 0.0003242
5: 0.0121316, 0.0153575, 0.0121843, 0.0150182, -0.0017157, 0.0021069
6: -0.0023571, -0.0015383, -0.0022709, -0.0015517, -0.0005348, 0.0004355
7: -0.0092361, -0.0071177, -0.0090133, -0.0071523, -0.0013836, 0.0011267
8: -0.0044213, -0.0033073, -0.0043041, -0.0033255, -0.0007276, 0.0005925
9: 0.0019711, 0.0032629, 0.0019922, 0.0031270, -0.0006870, 0.0008437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0004772
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875845, 0.9895546, 0.9876823, 0.9895261, -0.0012217, 0.0011136
1: -0.0043576, -0.0038667, -0.0043332, -0.0038738, -0.0003044, 0.0002775
2: 0.0104372, 0.0130389, 0.0104750, 0.0129096, -0.0014705, 0.0016132
3: -0.0072079, -0.0060237, -0.0071490, -0.0060409, -0.0007343, 0.0006693
4: 0.0025480, 0.0030515, 0.0025553, 0.0030265, -0.0002846, 0.0003122
5: 0.0120868, 0.0153589, 0.0121343, 0.0151963, -0.0018496, 0.0020290
6: -0.0023574, -0.0015269, -0.0023162, -0.0015390, -0.0005150, 0.0004694
7: -0.0092370, -0.0070882, -0.0091302, -0.0071194, -0.0013324, 0.0012146
8: -0.0044218, -0.0032918, -0.0043657, -0.0033082, -0.0007007, 0.0006387
9: 0.0019531, 0.0032634, 0.0019721, 0.0031983, -0.0007406, 0.0008125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0004779
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875845, 0.9895546, 0.9877787, 0.9895605, -0.0012893, 0.0010549
1: -0.0043576, -0.0038667, -0.0043092, -0.0038652, -0.0003213, 0.0002629
2: 0.0104372, 0.0130389, 0.0104294, 0.0127824, -0.0013930, 0.0017025
3: -0.0072079, -0.0060237, -0.0070911, -0.0060201, -0.0007749, 0.0006340
4: 0.0025480, 0.0030515, 0.0025465, 0.0030019, -0.0002696, 0.0003295
5: 0.0120868, 0.0153589, 0.0120769, 0.0150364, -0.0017520, 0.0021413
6: -0.0023574, -0.0015269, -0.0022756, -0.0015244, -0.0005435, 0.0004447
7: -0.0092370, -0.0070882, -0.0090252, -0.0070818, -0.0014062, 0.0011505
8: -0.0044218, -0.0032918, -0.0043104, -0.0032884, -0.0007395, 0.0006051
9: 0.0019531, 0.0032634, 0.0019492, 0.0031343, -0.0007016, 0.0008575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0004779
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9875852, 0.9895318, 0.9876772, 0.9894725, -0.0012059, 0.0011130
1: -0.0043574, -0.0038724, -0.0043345, -0.0038871, -0.0003005, 0.0002773
2: 0.0104675, 0.0130378, 0.0105458, 0.0129164, -0.0014697, 0.0015924
3: -0.0072074, -0.0060375, -0.0071521, -0.0060731, -0.0007248, 0.0006689
4: 0.0025538, 0.0030513, 0.0025690, 0.0030278, -0.0002845, 0.0003082
5: 0.0121248, 0.0153576, 0.0122232, 0.0152049, -0.0018485, 0.0020028
6: -0.0023571, -0.0015366, -0.0023183, -0.0015616, -0.0005083, 0.0004692
7: -0.0092362, -0.0071132, -0.0091359, -0.0071778, -0.0013152, 0.0012139
8: -0.0044214, -0.0033049, -0.0043686, -0.0033389, -0.0006917, 0.0006384
9: 0.0019683, 0.0032629, 0.0020078, 0.0032018, -0.0007402, 0.0008020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0004772
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9875852, 0.9895318, 0.9877750, 0.9895062, -0.0012745, 0.0010563
1: -0.0043574, -0.0038724, -0.0043101, -0.0038787, -0.0003176, 0.0002632
2: 0.0104675, 0.0130378, 0.0105013, 0.0127873, -0.0013948, 0.0016830
3: -0.0072074, -0.0060375, -0.0070934, -0.0060528, -0.0007660, 0.0006349
4: 0.0025538, 0.0030513, 0.0025604, 0.0030028, -0.0002700, 0.0003257
5: 0.0121248, 0.0153576, 0.0121673, 0.0150425, -0.0017543, 0.0021168
6: -0.0023571, -0.0015366, -0.0022771, -0.0015473, -0.0005373, 0.0004453
7: -0.0092362, -0.0071132, -0.0090292, -0.0071411, -0.0013901, 0.0011520
8: -0.0044214, -0.0033049, -0.0043125, -0.0033196, -0.0007310, 0.0006058
9: 0.0019683, 0.0032629, 0.0019854, 0.0031367, -0.0007025, 0.0008477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0004772
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875845, 0.9895588, 0.9876713, 0.9895321, -0.0012273, 0.0011192
1: -0.0043576, -0.0038656, -0.0043360, -0.0038723, -0.0003058, 0.0002789
2: 0.0104318, 0.0130390, 0.0104671, 0.0129243, -0.0014778, 0.0016207
3: -0.0072079, -0.0060212, -0.0071557, -0.0060373, -0.0007377, 0.0006726
4: 0.0025469, 0.0030516, 0.0025538, 0.0030294, -0.0002860, 0.0003137
5: 0.0120800, 0.0153590, 0.0121243, 0.0152148, -0.0018587, 0.0020384
6: -0.0023575, -0.0015252, -0.0023209, -0.0015364, -0.0005174, 0.0004718
7: -0.0092371, -0.0070838, -0.0091424, -0.0071129, -0.0013386, 0.0012206
8: -0.0044218, -0.0032894, -0.0043720, -0.0033047, -0.0007039, 0.0006419
9: 0.0019504, 0.0032635, 0.0019682, 0.0032057, -0.0007443, 0.0008163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004779
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0005038
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875845, 0.9895588, 0.9877647, 0.9895698, -0.0012953, 0.0010622
1: -0.0043576, -0.0038656, -0.0043127, -0.0038629, -0.0003228, 0.0002647
2: 0.0104318, 0.0130390, 0.0104174, 0.0128010, -0.0014026, 0.0017105
3: -0.0072079, -0.0060212, -0.0070996, -0.0060147, -0.0007785, 0.0006384
4: 0.0025469, 0.0030516, 0.0025441, 0.0030055, -0.0002715, 0.0003311
5: 0.0120800, 0.0153590, 0.0120618, 0.0150598, -0.0017641, 0.0021513
6: -0.0023575, -0.0015252, -0.0022815, -0.0015206, -0.0005460, 0.0004478
7: -0.0092371, -0.0070838, -0.0090406, -0.0070718, -0.0014127, 0.0011585
8: -0.0044218, -0.0032894, -0.0043185, -0.0032831, -0.0007429, 0.0006092
9: 0.0019504, 0.0032635, 0.0019431, 0.0031436, -0.0007064, 0.0008615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004779
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0005038
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875709, 0.9895554, 0.9876354, 0.9895309, -0.0011595, 0.0011890
1: -0.0043610, -0.0038665, -0.0043449, -0.0038726, -0.0002889, 0.0002963
2: 0.0104363, 0.0130569, 0.0104686, 0.0129716, -0.0015701, 0.0015311
3: -0.0072160, -0.0060232, -0.0071772, -0.0060380, -0.0006969, 0.0007146
4: 0.0025478, 0.0030550, 0.0025541, 0.0030385, -0.0003039, 0.0002963
5: 0.0120855, 0.0153815, 0.0121262, 0.0152743, -0.0019748, 0.0019257
6: -0.0023632, -0.0015266, -0.0023360, -0.0015369, -0.0004888, 0.0005012
7: -0.0092519, -0.0070874, -0.0091815, -0.0071141, -0.0012646, 0.0012968
8: -0.0044296, -0.0032913, -0.0043926, -0.0033054, -0.0006650, 0.0006820
9: 0.0019526, 0.0032725, 0.0019689, 0.0032296, -0.0007908, 0.0007711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004546
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004802
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875709, 0.9895554, 0.9877293, 0.9895687, -0.0012529, 0.0011507
1: -0.0043610, -0.0038665, -0.0043215, -0.0038631, -0.0003122, 0.0002867
2: 0.0104363, 0.0130569, 0.0104187, 0.0128476, -0.0015195, 0.0016545
3: -0.0072160, -0.0060232, -0.0071208, -0.0060152, -0.0007530, 0.0006916
4: 0.0025478, 0.0030550, 0.0025444, 0.0030145, -0.0002941, 0.0003202
5: 0.0120855, 0.0153815, 0.0120634, 0.0151183, -0.0019112, 0.0020809
6: -0.0023632, -0.0015266, -0.0022964, -0.0015210, -0.0005281, 0.0004851
7: -0.0092519, -0.0070874, -0.0090790, -0.0070729, -0.0013665, 0.0012550
8: -0.0044296, -0.0032913, -0.0043387, -0.0032837, -0.0007186, 0.0006600
9: 0.0019526, 0.0032725, 0.0019438, 0.0031671, -0.0007653, 0.0008333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004546
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004802
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9875576, 0.9895653, 0.9876356, 0.9895344, -0.0011693, 0.0011963
1: -0.0043643, -0.0038640, -0.0043449, -0.0038717, -0.0002914, 0.0002981
2: 0.0104233, 0.0130743, 0.0104640, 0.0129715, -0.0015796, 0.0015441
3: -0.0072240, -0.0060173, -0.0071772, -0.0060359, -0.0007028, 0.0007190
4: 0.0025453, 0.0030584, 0.0025532, 0.0030385, -0.0003057, 0.0002988
5: 0.0120692, 0.0154034, 0.0121204, 0.0152742, -0.0019868, 0.0019420
6: -0.0023687, -0.0015225, -0.0023359, -0.0015355, -0.0004929, 0.0005043
7: -0.0092662, -0.0070767, -0.0091814, -0.0071103, -0.0012753, 0.0013047
8: -0.0044372, -0.0032857, -0.0043925, -0.0033034, -0.0006707, 0.0006861
9: 0.0019461, 0.0032813, 0.0019666, 0.0032295, -0.0007956, 0.0007777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004626
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004902
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9875576, 0.9895653, 0.9877295, 0.9895738, -0.0012744, 0.0011578
1: -0.0043643, -0.0038640, -0.0043215, -0.0038619, -0.0003175, 0.0002885
2: 0.0104233, 0.0130743, 0.0104119, 0.0128475, -0.0015289, 0.0016828
3: -0.0072240, -0.0060173, -0.0071207, -0.0060122, -0.0007659, 0.0006959
4: 0.0025453, 0.0030584, 0.0025431, 0.0030145, -0.0002959, 0.0003257
5: 0.0120692, 0.0154034, 0.0120549, 0.0151182, -0.0019229, 0.0021165
6: -0.0023687, -0.0015225, -0.0022963, -0.0015188, -0.0005372, 0.0004881
7: -0.0092662, -0.0070767, -0.0090789, -0.0070673, -0.0013899, 0.0012627
8: -0.0044372, -0.0032857, -0.0043387, -0.0032808, -0.0007309, 0.0006641
9: 0.0019461, 0.0032813, 0.0019404, 0.0031670, -0.0007700, 0.0008475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004626
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004902
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9875853, 0.9895588, 0.9876432, 0.9895371, -0.0011776, 0.0011954
1: -0.0043574, -0.0038656, -0.0043429, -0.0038710, -0.0002934, 0.0002979
2: 0.0104318, 0.0130378, 0.0104604, 0.0129613, -0.0015785, 0.0015550
3: -0.0072074, -0.0060212, -0.0071725, -0.0060342, -0.0007078, 0.0007185
4: 0.0025469, 0.0030513, 0.0025525, 0.0030365, -0.0003055, 0.0003010
5: 0.0120799, 0.0153576, 0.0121159, 0.0152613, -0.0019854, 0.0019557
6: -0.0023571, -0.0015252, -0.0023327, -0.0015343, -0.0004964, 0.0005039
7: -0.0092361, -0.0070837, -0.0091729, -0.0071074, -0.0012843, 0.0013038
8: -0.0044213, -0.0032894, -0.0043881, -0.0033018, -0.0006754, 0.0006856
9: 0.0019504, 0.0032629, 0.0019648, 0.0032244, -0.0007950, 0.0007832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0004670
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.20 seconds
IS_A1_B1_B2_B2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0004772
IS_A1_B1_B2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
IS_A1_B1_B2_B2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0004772
IS_A1_B1_B2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
IS_A1_B1_B2_B2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0004779
IS_A1_B1_B2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
IS_A1_B1_B2_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0004779
IS_A1_B1_B2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004536, upper bound: 0.0005038
IS_A1_B1_B2_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0004772
IS_A1_B1_B2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
IS_A1_B1_B2_B2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0004772
IS_A1_B1_B2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004252, upper bound: 0.0005036
IS_A1_B1_B2_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004779
IS_A1_B1_B2_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0005038
IS_A1_B1_B2_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004779
IS_A1_B1_B2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0005038
IS_A1_B2_B1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004546
IS_A1_B2_B1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004802
IS_A1_B2_B1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004546
IS_A1_B2_B1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004802
IS_A1_B2_B1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004626
IS_A1_B2_B1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004902
IS_A1_B2_B1_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004626
IS_A1_B2_B1_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004604, upper bound: 0.0004902
IS_A1_B2_B2_B2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0004670
IS_A1_B2_B2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 7.20
Output dim: 0, lower bound: -0.0004011, upper bound: 0.0005036
IS_A1_B2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004362, upper bound: 0.0005036
IS_A1_B2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004828, upper bound: 0.0005038
IS_A1_B2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004828, upper bound: 0.0005038
IS_A1_B2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
IS_A1_B2_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004631, upper bound: 0.0005036
IS_A1_B2_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005038
IS_A1_B2_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005038
IS_A2_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004546, upper bound: 0.0004955
IS_A2_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004626, upper bound: 0.0004955
IS_A2_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004671, upper bound: 0.0004955
IS_A2_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004802, upper bound: 0.0004955
IS_A2_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0004902, upper bound: 0.0004955
IS_A2_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0005036, upper bound: 0.0004363
IS_A2_B2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0005038, upper bound: 0.0004828
IS_A2_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0005036, upper bound: 0.0004631
IS_A2_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.20
Output dim: 0, lower bound: -0.0005038, upper bound: 0.0004955

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.19 + 597.78 = 600.96 seconds
