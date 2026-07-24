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
Threshold: 0.0061821


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9812924, 0.9898738, 0.9812924, 0.9898738, -0.0085814, 0.0085814)
1: (-0.0045244, -0.0037872, -0.0045244, -0.0037872, -0.0007372, 0.0007372)
2: (0.0100159, 0.0139227, 0.0100159, 0.0139227, -0.0039068, 0.0039068)
3: (-0.0077033, -0.0058319, -0.0077033, -0.0058319, -0.0018714, 0.0018714)
4: (0.0024664, 0.0036257, 0.0024664, 0.0036257, -0.0011593, 0.0011593)
5: (0.0115568, 0.0206471, 0.0115568, 0.0206471, -0.0090904, 0.0090904)
6: (-0.0026396, -0.0013924, -0.0026396, -0.0013924, -0.0012472, 0.0012472)
7: (-0.0099670, -0.0067402, -0.0099670, -0.0067402, -0.0032268, 0.0032268)
8: (-0.0048057, -0.0026828, -0.0048057, -0.0026828, -0.0021229, 0.0021229)
9: (0.0017409, 0.0037086, 0.0017409, 0.0037086, -0.0019677, 0.0019677)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.82 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0069447, upper bound: 0.0069447

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067527, upper bound: 0.0067624
time: 1.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067527, upper bound: 0.0067527
time: 0.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 0, lower bound: -0.0067527, upper bound: 0.0067624
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 0, lower bound: -0.0067527, upper bound: 0.0067527

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9813740, 0.9897160, 0.9812924, 0.9898738, -0.0084997, 0.0084236
1: -0.0045232, -0.0038265, -0.0045244, -0.0037872, -0.0007360, 0.0006979
2: 0.0102242, 0.0139164, 0.0100159, 0.0139227, -0.0036985, 0.0039005
3: -0.0076992, -0.0059268, -0.0077033, -0.0058319, -0.0018673, 0.0017766
4: 0.0025068, 0.0036190, 0.0024664, 0.0036257, -0.0011189, 0.0011525
5: 0.0118189, 0.0205822, 0.0115568, 0.0206471, -0.0088283, 0.0090254
6: -0.0026376, -0.0014589, -0.0026396, -0.0013924, -0.0012452, 0.0011807
7: -0.0099618, -0.0069123, -0.0099670, -0.0067402, -0.0032216, 0.0030547
8: -0.0048030, -0.0027091, -0.0048057, -0.0026828, -0.0021202, 0.0020966
9: 0.0018458, 0.0037054, 0.0017409, 0.0037086, -0.0018627, 0.0019645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067405, upper bound: 0.0067405
time: 0.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067405, upper bound: 0.0067405
time: 0.91 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9782707, 0.9896300, 0.9813101, 0.9898199, -0.0115492, 0.0083199
1: -0.0045683, -0.0038479, -0.0045241, -0.0038006, -0.0007677, 0.0006762
2: 0.0103379, 0.0141555, 0.0100869, 0.0139213, -0.0035835, 0.0040686
3: -0.0078565, -0.0059785, -0.0077024, -0.0058643, -0.0019922, 0.0017240
4: 0.0025288, 0.0038748, 0.0024802, 0.0036243, -0.0010955, 0.0013946
5: 0.0119618, 0.0230536, 0.0116462, 0.0206331, -0.0086713, 0.0114074
6: -0.0027139, -0.0014952, -0.0026391, -0.0014151, -0.0012988, 0.0011439
7: -0.0101593, -0.0070061, -0.0099659, -0.0067989, -0.0033604, 0.0029597
8: -0.0049068, -0.0017096, -0.0048051, -0.0026885, -0.0022183, 0.0030955
9: 0.0019031, 0.0038258, 0.0017767, 0.0037079, -0.0018048, 0.0020491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065985, upper bound: 0.0066288
time: 0.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066288, upper bound: 0.0066288
time: 0.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -0.0067405, upper bound: 0.0067405
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -0.0067405, upper bound: 0.0067405
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -0.0065985, upper bound: 0.0066288
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -0.0066288, upper bound: 0.0066288

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9813740, 0.9897160, 0.9813740, 0.9897160, -0.0083420, 0.0083420
1: -0.0045232, -0.0038265, -0.0045232, -0.0038265, -0.0006967, 0.0006967
2: 0.0102242, 0.0139164, 0.0102242, 0.0139164, -0.0036922, 0.0036922
3: -0.0076992, -0.0059268, -0.0076992, -0.0059268, -0.0017724, 0.0017724
4: 0.0025068, 0.0036190, 0.0025068, 0.0036190, -0.0011122, 0.0011122
5: 0.0118189, 0.0205822, 0.0118189, 0.0205822, -0.0087633, 0.0087633
6: -0.0026376, -0.0014589, -0.0026376, -0.0014589, -0.0011786, 0.0011786
7: -0.0099618, -0.0069123, -0.0099618, -0.0069123, -0.0030495, 0.0030495
8: -0.0048030, -0.0027091, -0.0048030, -0.0027091, -0.0020939, 0.0020939
9: 0.0018458, 0.0037054, 0.0018458, 0.0037054, -0.0018596, 0.0018596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066039
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066461
time: 0.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9813740, 0.9897160, 0.9782707, 0.9896300, -0.0082560, 0.0114453
1: -0.0045232, -0.0038265, -0.0045683, -0.0038479, -0.0006753, 0.0007418
2: 0.0102242, 0.0139164, 0.0103379, 0.0141555, -0.0039313, 0.0035785
3: -0.0076992, -0.0059268, -0.0078565, -0.0059785, -0.0017207, 0.0019297
4: 0.0025068, 0.0036190, 0.0025288, 0.0038748, -0.0013680, 0.0010902
5: 0.0118189, 0.0205822, 0.0119618, 0.0230536, -0.0112347, 0.0086204
6: -0.0026376, -0.0014589, -0.0027139, -0.0014952, -0.0011424, 0.0012550
7: -0.0099618, -0.0069123, -0.0101593, -0.0070061, -0.0029557, 0.0032470
8: -0.0048030, -0.0027091, -0.0049068, -0.0017096, -0.0030933, 0.0021977
9: 0.0018458, 0.0037054, 0.0019031, 0.0038258, -0.0019800, 0.0018023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066039
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066461
time: 1.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9782745, 0.9896047, 0.9827266, 0.9896133, -0.0113388, 0.0068780
1: -0.0045682, -0.0038542, -0.0045035, -0.0038521, -0.0007162, 0.0006493
2: 0.0103712, 0.0141552, 0.0103599, 0.0138122, -0.0034410, 0.0037953
3: -0.0078563, -0.0059937, -0.0076307, -0.0059885, -0.0018678, 0.0016370
4: 0.0025352, 0.0038745, 0.0025330, 0.0035075, -0.0009723, 0.0013414
5: 0.0120037, 0.0230506, 0.0119895, 0.0195049, -0.0075012, 0.0110610
6: -0.0027138, -0.0015059, -0.0026043, -0.0015022, -0.0012116, 0.0010984
7: -0.0101590, -0.0070337, -0.0098757, -0.0070244, -0.0031347, 0.0028420
8: -0.0049067, -0.0017108, -0.0047577, -0.0031447, -0.0017620, 0.0030469
9: 0.0019199, 0.0038257, 0.0019142, 0.0036529, -0.0017330, 0.0019115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064291, upper bound: 0.0065442
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065138, upper bound: 0.0065442
time: 0.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9782730, 0.9896179, 0.9813360, 0.9897016, -0.0114285, 0.0082819
1: -0.0045682, -0.0038509, -0.0045237, -0.0038300, -0.0007382, 0.0006728
2: 0.0103538, 0.0141553, 0.0102432, 0.0139193, -0.0035656, 0.0039121
3: -0.0078563, -0.0059857, -0.0077011, -0.0059354, -0.0019209, 0.0017154
4: 0.0025318, 0.0038746, 0.0025104, 0.0036221, -0.0010903, 0.0013641
5: 0.0119818, 0.0230517, 0.0118428, 0.0206125, -0.0086307, 0.0112090
6: -0.0027138, -0.0015003, -0.0026385, -0.0014650, -0.0012488, 0.0011382
7: -0.0101591, -0.0070193, -0.0099642, -0.0069280, -0.0032311, 0.0029449
8: -0.0049067, -0.0017104, -0.0048042, -0.0026968, -0.0022099, 0.0030939
9: 0.0019111, 0.0038257, 0.0018554, 0.0037069, -0.0017958, 0.0019703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065441
time: 1.04 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065442
time: 0.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.25 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066039
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066461
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066039
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0066210, upper bound: 0.0066461
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0064291, upper bound: 0.0065442
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0065138, upper bound: 0.0065442
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065441
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065442

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9827852, 0.9895000, 0.9813775, 0.9896898, -0.0069046, 0.0081224
1: -0.0045027, -0.0038803, -0.0045231, -0.0038330, -0.0006697, 0.0006428
2: 0.0105095, 0.0138077, 0.0102588, 0.0139162, -0.0034067, 0.0035489
3: -0.0076277, -0.0060566, -0.0076990, -0.0059425, -0.0016852, 0.0016424
4: 0.0025620, 0.0035027, 0.0025135, 0.0036187, -0.0010567, 0.0009892
5: 0.0121776, 0.0194584, 0.0118623, 0.0205794, -0.0084018, 0.0075961
6: -0.0026029, -0.0015500, -0.0026375, -0.0014699, -0.0011329, 0.0010875
7: -0.0098720, -0.0071479, -0.0099616, -0.0069408, -0.0029312, 0.0028137
8: -0.0047557, -0.0031636, -0.0048028, -0.0027102, -0.0020455, 0.0016393
9: 0.0019895, 0.0036507, 0.0018632, 0.0037053, -0.0017158, 0.0017874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064795, upper bound: 0.0065240
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0065240
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9813981, 0.9896014, 0.9813762, 0.9897045, -0.0083064, 0.0082253
1: -0.0045228, -0.0038550, -0.0045231, -0.0038293, -0.0006935, 0.0006681
2: 0.0103756, 0.0139146, 0.0102394, 0.0139163, -0.0035407, 0.0036752
3: -0.0076980, -0.0059956, -0.0076991, -0.0059336, -0.0017643, 0.0017035
4: 0.0025361, 0.0036170, 0.0025097, 0.0036188, -0.0010828, 0.0011073
5: 0.0120092, 0.0205630, 0.0118379, 0.0205805, -0.0085713, 0.0087251
6: -0.0026370, -0.0015072, -0.0026375, -0.0014638, -0.0011732, 0.0011303
7: -0.0099603, -0.0070373, -0.0099617, -0.0069248, -0.0030355, 0.0029244
8: -0.0048022, -0.0027168, -0.0048029, -0.0027098, -0.0020924, 0.0020861
9: 0.0019221, 0.0037045, 0.0018535, 0.0037053, -0.0017833, 0.0018510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064795, upper bound: 0.0065664
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0065664
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9827852, 0.9895000, 0.9782745, 0.9896047, -0.0068194, 0.0112255
1: -0.0045027, -0.0038803, -0.0045682, -0.0038542, -0.0006484, 0.0006879
2: 0.0105095, 0.0138077, 0.0103712, 0.0141552, -0.0036457, 0.0034365
3: -0.0076277, -0.0060566, -0.0078563, -0.0059937, -0.0016340, 0.0017997
4: 0.0025620, 0.0035027, 0.0025352, 0.0038745, -0.0013125, 0.0009675
5: 0.0121776, 0.0194584, 0.0120037, 0.0230506, -0.0108729, 0.0074546
6: -0.0026029, -0.0015500, -0.0027138, -0.0015059, -0.0010970, 0.0011638
7: -0.0098720, -0.0071479, -0.0101590, -0.0070337, -0.0028383, 0.0030111
8: -0.0047557, -0.0031636, -0.0049067, -0.0017108, -0.0030449, 0.0017431
9: 0.0019895, 0.0036507, 0.0019199, 0.0038257, -0.0018362, 0.0017308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0064297
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065182
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9813981, 0.9896014, 0.9782730, 0.9896179, -0.0082198, 0.0113284
1: -0.0045228, -0.0038550, -0.0045682, -0.0038509, -0.0006719, 0.0007132
2: 0.0103756, 0.0139146, 0.0103538, 0.0141553, -0.0037798, 0.0035608
3: -0.0076980, -0.0059956, -0.0078563, -0.0059857, -0.0017123, 0.0018607
4: 0.0025361, 0.0036170, 0.0025318, 0.0038746, -0.0013385, 0.0010852
5: 0.0120092, 0.0205630, 0.0119818, 0.0230517, -0.0110426, 0.0085813
6: -0.0026370, -0.0015072, -0.0027138, -0.0015003, -0.0011367, 0.0012066
7: -0.0099603, -0.0070373, -0.0101591, -0.0070193, -0.0029410, 0.0031218
8: -0.0048022, -0.0027168, -0.0049067, -0.0017104, -0.0030918, 0.0021899
9: 0.0019221, 0.0037045, 0.0019111, 0.0038257, -0.0019037, 0.0017934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065618
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065618
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9782895, 0.9895600, 0.9836808, 0.9893570, -0.0110675, 0.0058792
1: -0.0045680, -0.0038653, -0.0044896, -0.0039159, -0.0006521, 0.0006243
2: 0.0104302, 0.0141541, 0.0106982, 0.0137387, -0.0033085, 0.0034558
3: -0.0078555, -0.0060205, -0.0075823, -0.0061425, -0.0017130, 0.0015618
4: 0.0025466, 0.0038732, 0.0025985, 0.0034289, -0.0008822, 0.0012747
5: 0.0120779, 0.0230386, 0.0124150, 0.0187451, -0.0066673, 0.0106236
6: -0.0027134, -0.0015247, -0.0025808, -0.0016102, -0.0011032, 0.0010562
7: -0.0101581, -0.0070824, -0.0098150, -0.0073038, -0.0028543, 0.0027326
8: -0.0049062, -0.0017157, -0.0047258, -0.0034051, -0.0015011, 0.0030101
9: 0.0019496, 0.0038251, 0.0020846, 0.0036159, -0.0016663, 0.0017405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9782747, 0.9896041, 0.9827667, 0.9895390, -0.0112643, 0.0068374
1: -0.0045682, -0.0038543, -0.0045029, -0.0038706, -0.0006976, 0.0006486
2: 0.0103720, 0.0141552, 0.0104580, 0.0138091, -0.0034371, 0.0036972
3: -0.0078563, -0.0059940, -0.0076286, -0.0060332, -0.0018231, 0.0016346
4: 0.0025354, 0.0038744, 0.0025520, 0.0035042, -0.0009688, 0.0013224
5: 0.0120047, 0.0230504, 0.0121129, 0.0194730, -0.0074683, 0.0109375
6: -0.0027138, -0.0015061, -0.0026033, -0.0015335, -0.0011802, 0.0010972
7: -0.0101590, -0.0070343, -0.0098732, -0.0071054, -0.0030536, 0.0028388
8: -0.0049067, -0.0017109, -0.0047564, -0.0031577, -0.0017490, 0.0030454
9: 0.0019203, 0.0038257, 0.0019636, 0.0036514, -0.0017311, 0.0018621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065442
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065441
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9782879, 0.9895736, 0.9822548, 0.9894525, -0.0111646, 0.0073188
1: -0.0045680, -0.0038619, -0.0045104, -0.0038921, -0.0006759, 0.0006484
2: 0.0104123, 0.0141542, 0.0105722, 0.0138486, -0.0034363, 0.0035820
3: -0.0078556, -0.0060123, -0.0076546, -0.0060851, -0.0017705, 0.0016422
4: 0.0025432, 0.0038734, 0.0025741, 0.0035464, -0.0010032, 0.0012992
5: 0.0120553, 0.0230398, 0.0122565, 0.0198807, -0.0078254, 0.0107833
6: -0.0027135, -0.0015189, -0.0026159, -0.0015700, -0.0011435, 0.0010970
7: -0.0101582, -0.0070676, -0.0099058, -0.0071997, -0.0029585, 0.0028382
8: -0.0049062, -0.0017152, -0.0047735, -0.0029928, -0.0019135, 0.0030583
9: 0.0019405, 0.0038252, 0.0020211, 0.0036712, -0.0017307, 0.0018041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0065442
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0065442
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9782733, 0.9896173, 0.9813720, 0.9896321, -0.0113588, 0.0082453
1: -0.0045682, -0.0038511, -0.0045232, -0.0038473, -0.0007209, 0.0006721
2: 0.0103545, 0.0141553, 0.0103349, 0.0139166, -0.0035621, 0.0038204
3: -0.0078563, -0.0059860, -0.0076993, -0.0059771, -0.0018792, 0.0017133
4: 0.0025320, 0.0038746, 0.0025282, 0.0036192, -0.0010872, 0.0013464
5: 0.0119827, 0.0230516, 0.0119580, 0.0205839, -0.0086012, 0.0110935
6: -0.0027138, -0.0015005, -0.0026376, -0.0014942, -0.0012196, 0.0011371
7: -0.0101591, -0.0070199, -0.0099619, -0.0070037, -0.0031554, 0.0029421
8: -0.0049067, -0.0017104, -0.0048030, -0.0027084, -0.0021983, 0.0030926
9: 0.0019115, 0.0038257, 0.0019016, 0.0037055, -0.0017940, 0.0019242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065365, upper bound: 0.0065442
time: 1.03 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065365, upper bound: 0.0065441
time: 1.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.31 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064795, upper bound: 0.0065240
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0065240
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064795, upper bound: 0.0065664
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0065664
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0064297
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065182
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065618
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065618
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065442
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065441
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0065442
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0065442
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065365, upper bound: 0.0065442
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -0.0065365, upper bound: 0.0065441

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9828000, 0.9894577, 0.9822769, 0.9894480, -0.0066480, 0.0071808
1: -0.0045024, -0.0038908, -0.0045100, -0.0038932, -0.0006092, 0.0006192
2: 0.0105652, 0.0138066, 0.0105781, 0.0138469, -0.0032816, 0.0032285
3: -0.0076269, -0.0060820, -0.0076534, -0.0060878, -0.0015391, 0.0015715
4: 0.0025728, 0.0035015, 0.0025753, 0.0035446, -0.0009718, 0.0009262
5: 0.0122478, 0.0194466, 0.0122639, 0.0198631, -0.0076154, 0.0071827
6: -0.0026025, -0.0015678, -0.0026154, -0.0015719, -0.0010306, 0.0010476
7: -0.0098711, -0.0071939, -0.0099044, -0.0072045, -0.0026665, 0.0027104
8: -0.0047552, -0.0031683, -0.0047727, -0.0029999, -0.0017554, 0.0016044
9: 0.0020176, 0.0036501, 0.0020241, 0.0036704, -0.0016528, 0.0016260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9827855, 0.9894993, 0.9814098, 0.9896156, -0.0068301, 0.0080895
1: -0.0045026, -0.0038805, -0.0045226, -0.0038515, -0.0006512, 0.0006422
2: 0.0105104, 0.0138077, 0.0103568, 0.0139137, -0.0034033, 0.0034509
3: -0.0076277, -0.0060570, -0.0076974, -0.0059871, -0.0016406, 0.0016404
4: 0.0025621, 0.0035027, 0.0025324, 0.0036160, -0.0010539, 0.0009702
5: 0.0121787, 0.0194581, 0.0119855, 0.0205537, -0.0083750, 0.0074726
6: -0.0026029, -0.0015503, -0.0026367, -0.0015012, -0.0011016, 0.0010864
7: -0.0098720, -0.0071486, -0.0099595, -0.0070218, -0.0028502, 0.0028109
8: -0.0047557, -0.0031637, -0.0048018, -0.0027206, -0.0020351, 0.0016381
9: 0.0019899, 0.0036506, 0.0019126, 0.0037040, -0.0017141, 0.0017381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0064356
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0065240
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9814124, 0.9895587, 0.9822755, 0.9894654, -0.0080531, 0.0072832
1: -0.0045226, -0.0038656, -0.0045101, -0.0038889, -0.0006337, 0.0006444
2: 0.0104319, 0.0139135, 0.0105551, 0.0138470, -0.0034151, 0.0033583
3: -0.0076973, -0.0060213, -0.0076535, -0.0060774, -0.0016199, 0.0016323
4: 0.0025470, 0.0036158, 0.0025708, 0.0035447, -0.0009978, 0.0010450
5: 0.0120800, 0.0205516, 0.0122350, 0.0198644, -0.0077844, 0.0083166
6: -0.0026366, -0.0015252, -0.0026154, -0.0015646, -0.0010721, 0.0010902
7: -0.0099594, -0.0070838, -0.0099044, -0.0071856, -0.0027738, 0.0028207
8: -0.0048017, -0.0027214, -0.0047728, -0.0029994, -0.0018023, 0.0020514
9: 0.0019504, 0.0037039, 0.0020125, 0.0036704, -0.0017200, 0.0016914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9813983, 0.9896008, 0.9814083, 0.9896307, -0.0082324, 0.0081925
1: -0.0045228, -0.0038552, -0.0045227, -0.0038477, -0.0006751, 0.0006675
2: 0.0103764, 0.0139145, 0.0103369, 0.0139138, -0.0035374, 0.0035777
3: -0.0076980, -0.0059960, -0.0076975, -0.0059780, -0.0017200, 0.0017015
4: 0.0025362, 0.0036170, 0.0025286, 0.0036162, -0.0010799, 0.0010884
5: 0.0120102, 0.0205628, 0.0119605, 0.0205548, -0.0085446, 0.0086023
6: -0.0026370, -0.0015075, -0.0026367, -0.0014949, -0.0011421, 0.0011292
7: -0.0099603, -0.0070380, -0.0099596, -0.0070053, -0.0029549, 0.0029217
8: -0.0048022, -0.0027169, -0.0048018, -0.0027201, -0.0020820, 0.0020849
9: 0.0019225, 0.0037045, 0.0019026, 0.0037041, -0.0017816, 0.0018019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065240, upper bound: 0.0065664
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065240, upper bound: 0.0065664
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9782895, 0.9895600, -0.0058298, 0.0109648
1: -0.0044889, -0.0039415, -0.0045680, -0.0038653, -0.0006236, 0.0006265
2: 0.0108338, 0.0137349, 0.0104302, 0.0141541, -0.0033203, 0.0033047
3: -0.0075798, -0.0062042, -0.0078555, -0.0060205, -0.0015593, 0.0016513
4: 0.0026247, 0.0034248, 0.0025466, 0.0038732, -0.0012485, 0.0008782
5: 0.0125855, 0.0187057, 0.0120779, 0.0230386, -0.0104531, 0.0066278
6: -0.0025796, -0.0016535, -0.0027134, -0.0015247, -0.0010550, 0.0010599
7: -0.0098119, -0.0074157, -0.0101581, -0.0070824, -0.0027295, 0.0027423
8: -0.0047241, -0.0034640, -0.0049062, -0.0017157, -0.0030084, 0.0014422
9: 0.0021528, 0.0036140, 0.0019496, 0.0038251, -0.0016723, 0.0016644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065146, upper bound: 0.0064297
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065146, upper bound: 0.0064297
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9782747, 0.9896041, -0.0067818, 0.0111482
1: -0.0045021, -0.0038995, -0.0045682, -0.0038543, -0.0006478, 0.0006687
2: 0.0106112, 0.0138048, 0.0103720, 0.0141552, -0.0035440, 0.0034328
3: -0.0076258, -0.0061029, -0.0078563, -0.0059940, -0.0016318, 0.0017534
4: 0.0025817, 0.0034996, 0.0025354, 0.0038744, -0.0012928, 0.0009643
5: 0.0123056, 0.0194288, 0.0120047, 0.0230504, -0.0107448, 0.0074241
6: -0.0026019, -0.0015825, -0.0027138, -0.0015061, -0.0010959, 0.0011313
7: -0.0098696, -0.0072319, -0.0101590, -0.0070343, -0.0028353, 0.0029271
8: -0.0047545, -0.0031755, -0.0049067, -0.0017109, -0.0030436, 0.0017311
9: 0.0020408, 0.0036492, 0.0019203, 0.0038257, -0.0017849, 0.0017289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065182
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065182
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9814124, 0.9895587, 0.9794132, 0.9893691, -0.0079567, 0.0101455
1: -0.0045226, -0.0038656, -0.0045517, -0.0039129, -0.0006097, 0.0006860
2: 0.0104319, 0.0139135, 0.0106824, 0.0140675, -0.0036356, 0.0032311
3: -0.0076973, -0.0060213, -0.0077986, -0.0061353, -0.0015620, 0.0017773
4: 0.0025470, 0.0036158, 0.0025954, 0.0037806, -0.0012337, 0.0010204
5: 0.0120800, 0.0205516, 0.0123951, 0.0221438, -0.0100638, 0.0081565
6: -0.0026366, -0.0015252, -0.0026858, -0.0016052, -0.0010314, 0.0011606
7: -0.0099594, -0.0070838, -0.0100866, -0.0072907, -0.0026687, 0.0030028
8: -0.0048017, -0.0027214, -0.0048686, -0.0020776, -0.0027241, 0.0021471
9: 0.0019504, 0.0037039, 0.0020766, 0.0037815, -0.0018311, 0.0016273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9813983, 0.9896008, 0.9783083, 0.9895449, -0.0081466, 0.0112925
1: -0.0045228, -0.0038552, -0.0045677, -0.0038691, -0.0006537, 0.0007126
2: 0.0103764, 0.0139145, 0.0104502, 0.0141526, -0.0037762, 0.0034643
3: -0.0076980, -0.0059960, -0.0078546, -0.0060296, -0.0016684, 0.0018586
4: 0.0025362, 0.0036170, 0.0025505, 0.0038717, -0.0013355, 0.0010665
5: 0.0120102, 0.0205628, 0.0121031, 0.0230236, -0.0110134, 0.0084597
6: -0.0026370, -0.0015075, -0.0027130, -0.0015311, -0.0011059, 0.0012055
7: -0.0099603, -0.0070380, -0.0101569, -0.0070989, -0.0028613, 0.0031189
8: -0.0048022, -0.0027169, -0.0049056, -0.0017217, -0.0030804, 0.0021886
9: 0.0019225, 0.0037045, 0.0019597, 0.0038244, -0.0019019, 0.0017448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0064721
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065618
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9782895, 0.9895600, 0.9837303, 0.9892544, -0.0109648, 0.0058298
1: -0.0045680, -0.0038653, -0.0044889, -0.0039415, -0.0006265, 0.0006236
2: 0.0104302, 0.0141541, 0.0108338, 0.0137349, -0.0033047, 0.0033203
3: -0.0078555, -0.0060205, -0.0075798, -0.0062042, -0.0016513, 0.0015593
4: 0.0025466, 0.0038732, 0.0026247, 0.0034248, -0.0008782, 0.0012485
5: 0.0120779, 0.0230386, 0.0125855, 0.0187057, -0.0066278, 0.0104531
6: -0.0027134, -0.0015247, -0.0025796, -0.0016535, -0.0010599, 0.0010550
7: -0.0101581, -0.0070824, -0.0098119, -0.0074157, -0.0027423, 0.0027295
8: -0.0049062, -0.0017157, -0.0047241, -0.0034640, -0.0014422, 0.0030084
9: 0.0019496, 0.0038251, 0.0021528, 0.0036140, -0.0016644, 0.0016723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9782895, 0.9895600, 0.9804042, 0.9891771, -0.0108876, 0.0091558
1: -0.0045680, -0.0038653, -0.0045373, -0.0039607, -0.0006073, 0.0006719
2: 0.0104302, 0.0141541, 0.0109359, 0.0139911, -0.0035609, 0.0032182
3: -0.0078555, -0.0060205, -0.0077483, -0.0062507, -0.0016049, 0.0017279
4: 0.0025466, 0.0038732, 0.0026445, 0.0036989, -0.0011523, 0.0012287
5: 0.0120779, 0.0230386, 0.0127139, 0.0213545, -0.0092766, 0.0103247
6: -0.0027134, -0.0015247, -0.0026614, -0.0016861, -0.0010273, 0.0011368
7: -0.0101581, -0.0070824, -0.0100235, -0.0075001, -0.0026580, 0.0029411
8: -0.0049062, -0.0017157, -0.0048354, -0.0023968, -0.0025094, 0.0031197
9: 0.0019496, 0.0038251, 0.0022043, 0.0037430, -0.0017935, 0.0016209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9782747, 0.9896041, 0.9828223, 0.9894229, -0.0111482, 0.0067818
1: -0.0045682, -0.0038543, -0.0045021, -0.0038995, -0.0006687, 0.0006478
2: 0.0103720, 0.0141552, 0.0106112, 0.0138048, -0.0034328, 0.0035440
3: -0.0078563, -0.0059940, -0.0076258, -0.0061029, -0.0017534, 0.0016318
4: 0.0025354, 0.0038744, 0.0025817, 0.0034996, -0.0009643, 0.0012928
5: 0.0120047, 0.0230504, 0.0123056, 0.0194288, -0.0074241, 0.0107448
6: -0.0027138, -0.0015061, -0.0026019, -0.0015825, -0.0011313, 0.0010959
7: -0.0101590, -0.0070343, -0.0098696, -0.0072319, -0.0029271, 0.0028353
8: -0.0049067, -0.0017109, -0.0047545, -0.0031755, -0.0017311, 0.0030436
9: 0.0019203, 0.0038257, 0.0020408, 0.0036492, -0.0017289, 0.0017849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9782747, 0.9896041, 0.9794893, 0.9893448, -0.0110701, 0.0101148
1: -0.0045682, -0.0038543, -0.0045506, -0.0039190, -0.0006493, 0.0006962
2: 0.0103720, 0.0141552, 0.0107144, 0.0140616, -0.0036896, 0.0034408
3: -0.0078563, -0.0059940, -0.0077947, -0.0061498, -0.0017064, 0.0018007
4: 0.0025354, 0.0038744, 0.0026016, 0.0037743, -0.0012390, 0.0012728
5: 0.0120047, 0.0230504, 0.0124353, 0.0220831, -0.0100784, 0.0106151
6: -0.0027138, -0.0015061, -0.0026839, -0.0016154, -0.0010984, 0.0011778
7: -0.0101590, -0.0070343, -0.0100817, -0.0073171, -0.0028419, 0.0030474
8: -0.0049067, -0.0017109, -0.0048660, -0.0021021, -0.0028046, 0.0031551
9: 0.0019203, 0.0038257, 0.0020927, 0.0037786, -0.0018583, 0.0017330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9782879, 0.9895736, 0.9823049, 0.9893634, -0.0110756, 0.0072687
1: -0.0045680, -0.0038619, -0.0045096, -0.0039143, -0.0006537, 0.0006477
2: 0.0104123, 0.0141542, 0.0106898, 0.0138447, -0.0034324, 0.0034643
3: -0.0078556, -0.0060123, -0.0076520, -0.0061387, -0.0017169, 0.0016397
4: 0.0025432, 0.0038734, 0.0025969, 0.0035423, -0.0009991, 0.0012765
5: 0.0120553, 0.0230398, 0.0124045, 0.0198408, -0.0077855, 0.0106354
6: -0.0027135, -0.0015189, -0.0026147, -0.0016076, -0.0011059, 0.0010957
7: -0.0101582, -0.0070676, -0.0099026, -0.0072969, -0.0028613, 0.0028350
8: -0.0049062, -0.0017152, -0.0047718, -0.0030089, -0.0018973, 0.0030566
9: 0.0019405, 0.0038252, 0.0020803, 0.0036693, -0.0017288, 0.0017448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9782879, 0.9895736, 0.9794433, 0.9892499, -0.0109621, 0.0101303
1: -0.0045680, -0.0038619, -0.0045512, -0.0039426, -0.0006255, 0.0006893
2: 0.0104123, 0.0141542, 0.0108396, 0.0140652, -0.0036529, 0.0033146
3: -0.0078556, -0.0060123, -0.0077970, -0.0062068, -0.0016488, 0.0017847
4: 0.0025432, 0.0038734, 0.0026259, 0.0037781, -0.0012350, 0.0012475
5: 0.0120553, 0.0230398, 0.0125928, 0.0221198, -0.0100645, 0.0104471
6: -0.0027135, -0.0015189, -0.0026850, -0.0016554, -0.0010581, 0.0011661
7: -0.0101582, -0.0070676, -0.0100847, -0.0074205, -0.0027377, 0.0030171
8: -0.0049062, -0.0017152, -0.0048676, -0.0020873, -0.0028190, 0.0031524
9: 0.0019405, 0.0038252, 0.0021558, 0.0037803, -0.0018398, 0.0016694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9782733, 0.9896173, 0.9814312, 0.9895265, -0.0112531, 0.0081861
1: -0.0045682, -0.0038511, -0.0045223, -0.0038737, -0.0006946, 0.0006713
2: 0.0103545, 0.0141553, 0.0104745, 0.0139120, -0.0035575, 0.0036809
3: -0.0078563, -0.0059860, -0.0076963, -0.0060406, -0.0018157, 0.0017103
4: 0.0025320, 0.0038746, 0.0025552, 0.0036143, -0.0010823, 0.0013194
5: 0.0119827, 0.0230516, 0.0121336, 0.0205366, -0.0085539, 0.0109180
6: -0.0027138, -0.0015005, -0.0026362, -0.0015388, -0.0011750, 0.0011356
7: -0.0101591, -0.0070199, -0.0099582, -0.0071190, -0.0030401, 0.0029383
8: -0.0049067, -0.0017104, -0.0048010, -0.0027275, -0.0021792, 0.0030906
9: 0.0019115, 0.0038257, 0.0019719, 0.0037032, -0.0017917, 0.0018539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065138
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065441
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9782733, 0.9896173, 0.9783329, 0.9894307, -0.0111574, 0.0112845
1: -0.0045682, -0.0038511, -0.0045674, -0.0038975, -0.0006707, 0.0007163
2: 0.0103545, 0.0141553, 0.0106009, 0.0141507, -0.0037962, 0.0035544
3: -0.0078563, -0.0059860, -0.0078533, -0.0060982, -0.0017581, 0.0018673
4: 0.0025320, 0.0038746, 0.0025797, 0.0038696, -0.0013377, 0.0012949
5: 0.0119827, 0.0230516, 0.0122926, 0.0230040, -0.0110213, 0.0107590
6: -0.0027138, -0.0015005, -0.0027124, -0.0015792, -0.0011347, 0.0012118
7: -0.0101591, -0.0070199, -0.0101553, -0.0072234, -0.0029357, 0.0031354
8: -0.0049067, -0.0017104, -0.0049047, -0.0017297, -0.0031771, 0.0031943
9: 0.0019115, 0.0038257, 0.0020355, 0.0038234, -0.0019120, 0.0017902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0064600
time: 1.06 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0065442
time: 0.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0064356
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065664, upper bound: 0.0065240
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065240, upper bound: 0.0065664
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065240, upper bound: 0.0065664
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065146, upper bound: 0.0064297
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065146, upper bound: 0.0064297
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065182
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064600, upper bound: 0.0065182
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0064721
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065442, upper bound: 0.0065618
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
IS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065138
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0065022, upper bound: 0.0065441
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0064600
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -0.0064501, upper bound: 0.0065442

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9828000, 0.9894577, 0.9837303, 0.9892544, -0.0064543, 0.0057274
1: -0.0045024, -0.0038908, -0.0044889, -0.0039415, -0.0005610, 0.0005981
2: 0.0105652, 0.0138066, 0.0108338, 0.0137349, -0.0031696, 0.0029728
3: -0.0076269, -0.0060820, -0.0075798, -0.0062042, -0.0014228, 0.0014978
4: 0.0025728, 0.0035015, 0.0026247, 0.0034248, -0.0008520, 0.0008767
5: 0.0122478, 0.0194466, 0.0125855, 0.0187057, -0.0064580, 0.0068611
6: -0.0026025, -0.0015678, -0.0025796, -0.0016535, -0.0009490, 0.0010118
7: -0.0098711, -0.0071939, -0.0098119, -0.0074157, -0.0024553, 0.0026179
8: -0.0047552, -0.0031683, -0.0047241, -0.0034640, -0.0012912, 0.0015558
9: 0.0020176, 0.0036501, 0.0021528, 0.0036140, -0.0015964, 0.0014972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9828000, 0.9894577, 0.9823049, 0.9893634, -0.0065634, 0.0071527
1: -0.0045024, -0.0038908, -0.0045096, -0.0039143, -0.0005881, 0.0006188
2: 0.0105652, 0.0138066, 0.0106898, 0.0138447, -0.0032794, 0.0031167
3: -0.0076269, -0.0060820, -0.0076520, -0.0061387, -0.0014883, 0.0015701
4: 0.0025728, 0.0035015, 0.0025969, 0.0035423, -0.0009695, 0.0009046
5: 0.0122478, 0.0194466, 0.0124045, 0.0198408, -0.0075930, 0.0070421
6: -0.0026025, -0.0015678, -0.0026147, -0.0016076, -0.0009949, 0.0010469
7: -0.0098711, -0.0071939, -0.0099026, -0.0072969, -0.0025742, 0.0027086
8: -0.0047552, -0.0031683, -0.0047718, -0.0030089, -0.0017463, 0.0016035
9: 0.0020176, 0.0036501, 0.0020803, 0.0036693, -0.0016517, 0.0015697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9814098, 0.9896156, -0.0058853, 0.0078445
1: -0.0044889, -0.0039415, -0.0045226, -0.0038515, -0.0006374, 0.0005812
2: 0.0108338, 0.0137349, 0.0103568, 0.0139137, -0.0030799, 0.0033781
3: -0.0075798, -0.0062042, -0.0076974, -0.0059871, -0.0015927, 0.0014932
4: 0.0026247, 0.0034248, 0.0025324, 0.0036160, -0.0009913, 0.0008924
5: 0.0125855, 0.0187057, 0.0119855, 0.0205537, -0.0079682, 0.0067202
6: -0.0025796, -0.0016535, -0.0026367, -0.0015012, -0.0010784, 0.0009832
7: -0.0098119, -0.0074157, -0.0099595, -0.0070218, -0.0027901, 0.0025438
8: -0.0047241, -0.0034640, -0.0048018, -0.0027206, -0.0020035, 0.0013377
9: 0.0021528, 0.0036140, 0.0019126, 0.0037040, -0.0015512, 0.0017014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9814098, 0.9896156, -0.0067933, 0.0080130
1: -0.0045021, -0.0038995, -0.0045226, -0.0038515, -0.0006506, 0.0006232
2: 0.0106112, 0.0138048, 0.0103568, 0.0139137, -0.0033024, 0.0034481
3: -0.0076258, -0.0061029, -0.0076974, -0.0059871, -0.0016387, 0.0015945
4: 0.0025817, 0.0034996, 0.0025324, 0.0036160, -0.0010344, 0.0009672
5: 0.0123056, 0.0194288, 0.0119855, 0.0205537, -0.0082481, 0.0074432
6: -0.0026019, -0.0015825, -0.0026367, -0.0015012, -0.0011007, 0.0010542
7: -0.0098696, -0.0072319, -0.0099595, -0.0070218, -0.0028479, 0.0027276
8: -0.0047545, -0.0031755, -0.0048018, -0.0027206, -0.0020339, 0.0016262
9: 0.0020408, 0.0036492, 0.0019126, 0.0037040, -0.0016633, 0.0017366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9814124, 0.9895587, 0.9837303, 0.9892544, -0.0078420, 0.0058284
1: -0.0045226, -0.0038656, -0.0044889, -0.0039415, -0.0005811, 0.0006233
2: 0.0104319, 0.0139135, 0.0108338, 0.0137349, -0.0033030, 0.0030797
3: -0.0076973, -0.0060213, -0.0075798, -0.0062042, -0.0014931, 0.0015585
4: 0.0025470, 0.0036158, 0.0026247, 0.0034248, -0.0008778, 0.0009911
5: 0.0120800, 0.0205516, 0.0125855, 0.0187057, -0.0066257, 0.0079661
6: -0.0026366, -0.0015252, -0.0025796, -0.0016535, -0.0009831, 0.0010544
7: -0.0099594, -0.0070838, -0.0098119, -0.0074157, -0.0025436, 0.0027281
8: -0.0048017, -0.0027214, -0.0047241, -0.0034640, -0.0013377, 0.0020027
9: 0.0019504, 0.0037039, 0.0021528, 0.0036140, -0.0016636, 0.0015511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064797
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9814124, 0.9895587, 0.9823049, 0.9893634, -0.0079511, 0.0072538
1: -0.0045226, -0.0038656, -0.0045096, -0.0039143, -0.0006083, 0.0006440
2: 0.0104319, 0.0139135, 0.0106898, 0.0138447, -0.0034128, 0.0032236
3: -0.0076973, -0.0060213, -0.0076520, -0.0061387, -0.0015586, 0.0016308
4: 0.0025470, 0.0036158, 0.0025969, 0.0035423, -0.0009953, 0.0010189
5: 0.0120800, 0.0205516, 0.0124045, 0.0198408, -0.0077608, 0.0081471
6: -0.0026366, -0.0015252, -0.0026147, -0.0016076, -0.0010291, 0.0010895
7: -0.0099594, -0.0070838, -0.0099026, -0.0072969, -0.0026625, 0.0028188
8: -0.0048017, -0.0027214, -0.0047718, -0.0030089, -0.0017928, 0.0020504
9: 0.0019504, 0.0037039, 0.0020803, 0.0036693, -0.0017189, 0.0016236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064796
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9813983, 0.9896008, 0.9828223, 0.9894229, -0.0080245, 0.0067785
1: -0.0045228, -0.0038552, -0.0045021, -0.0038995, -0.0006233, 0.0006469
2: 0.0103764, 0.0139145, 0.0106112, 0.0138048, -0.0034285, 0.0033033
3: -0.0076980, -0.0059960, -0.0076258, -0.0061029, -0.0015951, 0.0016298
4: 0.0025362, 0.0036170, 0.0025817, 0.0034996, -0.0009634, 0.0010353
5: 0.0120102, 0.0205628, 0.0123056, 0.0194288, -0.0074186, 0.0082572
6: -0.0026370, -0.0015075, -0.0026019, -0.0015825, -0.0010545, 0.0010945
7: -0.0099603, -0.0070380, -0.0098696, -0.0072319, -0.0027283, 0.0028317
8: -0.0048022, -0.0027169, -0.0047545, -0.0031755, -0.0016266, 0.0020376
9: 0.0019225, 0.0037045, 0.0020408, 0.0036492, -0.0017267, 0.0016637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064795
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9813983, 0.9896008, 0.9814312, 0.9895265, -0.0081281, 0.0081696
1: -0.0045228, -0.0038552, -0.0045223, -0.0038737, -0.0006491, 0.0006672
2: 0.0103764, 0.0139145, 0.0104745, 0.0139120, -0.0035356, 0.0034401
3: -0.0076980, -0.0059960, -0.0076963, -0.0060406, -0.0016573, 0.0017003
4: 0.0025362, 0.0036170, 0.0025552, 0.0036143, -0.0010781, 0.0010618
5: 0.0120102, 0.0205628, 0.0121336, 0.0205366, -0.0085264, 0.0084292
6: -0.0026370, -0.0015075, -0.0026362, -0.0015388, -0.0010982, 0.0011287
7: -0.0099603, -0.0070380, -0.0099582, -0.0071190, -0.0028413, 0.0029202
8: -0.0048022, -0.0027169, -0.0048010, -0.0027275, -0.0020746, 0.0020841
9: 0.0019225, 0.0037045, 0.0019719, 0.0037032, -0.0017807, 0.0017326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064795
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9794665, 0.9893759, -0.0056457, 0.0097879
1: -0.0044889, -0.0039415, -0.0045509, -0.0039112, -0.0005777, 0.0006094
2: 0.0108338, 0.0137349, 0.0106733, 0.0140634, -0.0032296, 0.0030616
3: -0.0075798, -0.0062042, -0.0077959, -0.0061311, -0.0014487, 0.0015917
4: 0.0026247, 0.0034248, 0.0025937, 0.0037762, -0.0011515, 0.0008311
5: 0.0125855, 0.0187057, 0.0123836, 0.0221013, -0.0095158, 0.0063221
6: -0.0025796, -0.0016535, -0.0026845, -0.0016023, -0.0009774, 0.0010310
7: -0.0098119, -0.0074157, -0.0100832, -0.0072832, -0.0025287, 0.0026674
8: -0.0047241, -0.0034640, -0.0048668, -0.0020947, -0.0026294, 0.0014028
9: 0.0021528, 0.0036140, 0.0020720, 0.0037794, -0.0016266, 0.0015420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9783124, 0.9894611, -0.0057309, 0.0109419
1: -0.0044889, -0.0039415, -0.0045677, -0.0038900, -0.0005989, 0.0006262
2: 0.0108338, 0.0137349, 0.0105608, 0.0141523, -0.0033185, 0.0031741
3: -0.0075798, -0.0062042, -0.0078543, -0.0060799, -0.0014998, 0.0016502
4: 0.0026247, 0.0034248, 0.0025719, 0.0038713, -0.0012466, 0.0008529
5: 0.0125855, 0.0187057, 0.0122422, 0.0230203, -0.0104348, 0.0064635
6: -0.0025796, -0.0016535, -0.0027129, -0.0015664, -0.0010132, 0.0010594
7: -0.0098119, -0.0074157, -0.0101566, -0.0071903, -0.0026216, 0.0027409
8: -0.0047241, -0.0034640, -0.0049054, -0.0017231, -0.0030011, 0.0014414
9: 0.0021528, 0.0036140, 0.0020154, 0.0038242, -0.0016714, 0.0015986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9794148, 0.9893565, -0.0065342, 0.0100080
1: -0.0045021, -0.0038995, -0.0045517, -0.0039160, -0.0005861, 0.0006522
2: 0.0106112, 0.0138048, 0.0106990, 0.0140674, -0.0034561, 0.0031059
3: -0.0076258, -0.0061029, -0.0077985, -0.0061428, -0.0014830, 0.0016956
4: 0.0025817, 0.0034996, 0.0025987, 0.0037805, -0.0011988, 0.0009010
5: 0.0123056, 0.0194288, 0.0124159, 0.0221424, -0.0098368, 0.0070129
6: -0.0026019, -0.0015825, -0.0026857, -0.0016105, -0.0009915, 0.0011033
7: -0.0098696, -0.0072319, -0.0100865, -0.0073044, -0.0025653, 0.0028545
8: -0.0047545, -0.0031755, -0.0048685, -0.0020781, -0.0026764, 0.0016930
9: 0.0020408, 0.0036492, 0.0020849, 0.0037814, -0.0017407, 0.0015643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9783099, 0.9895312, -0.0067089, 0.0111130
1: -0.0045021, -0.0038995, -0.0045677, -0.0038725, -0.0006296, 0.0006682
2: 0.0106112, 0.0138048, 0.0104682, 0.0141525, -0.0035413, 0.0033367
3: -0.0076258, -0.0061029, -0.0078545, -0.0060378, -0.0015880, 0.0017516
4: 0.0025817, 0.0034996, 0.0025540, 0.0038715, -0.0012899, 0.0009456
5: 0.0123056, 0.0194288, 0.0121257, 0.0230224, -0.0107168, 0.0073031
6: -0.0026019, -0.0015825, -0.0027129, -0.0015368, -0.0010652, 0.0011305
7: -0.0098696, -0.0072319, -0.0101568, -0.0071138, -0.0027559, 0.0029249
8: -0.0047545, -0.0031755, -0.0049055, -0.0017222, -0.0030323, 0.0017300
9: 0.0020408, 0.0036492, 0.0019687, 0.0038243, -0.0017836, 0.0016805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9814124, 0.9895587, 0.9804042, 0.9891771, -0.0077648, 0.0091545
1: -0.0045226, -0.0038656, -0.0045373, -0.0039607, -0.0005619, 0.0006716
2: 0.0104319, 0.0139135, 0.0109359, 0.0139911, -0.0035593, 0.0029776
3: -0.0076973, -0.0060213, -0.0077483, -0.0062507, -0.0014466, 0.0017271
4: 0.0025470, 0.0036158, 0.0026445, 0.0036989, -0.0011520, 0.0009713
5: 0.0120800, 0.0205516, 0.0127139, 0.0213545, -0.0092745, 0.0078377
6: -0.0026366, -0.0015252, -0.0026614, -0.0016861, -0.0009505, 0.0011362
7: -0.0099594, -0.0070838, -0.0100235, -0.0075001, -0.0024593, 0.0029397
8: -0.0048017, -0.0027214, -0.0048354, -0.0023968, -0.0024049, 0.0021140
9: 0.0019504, 0.0037039, 0.0022043, 0.0037430, -0.0017926, 0.0014997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064728
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9814124, 0.9895587, 0.9794433, 0.9892499, -0.0078376, 0.0101154
1: -0.0045226, -0.0038656, -0.0045512, -0.0039426, -0.0005800, 0.0006856
2: 0.0104319, 0.0139135, 0.0108396, 0.0140652, -0.0036333, 0.0030739
3: -0.0076973, -0.0060213, -0.0077970, -0.0062068, -0.0014904, 0.0017758
4: 0.0025470, 0.0036158, 0.0026259, 0.0037781, -0.0012312, 0.0009900
5: 0.0120800, 0.0205516, 0.0125928, 0.0221198, -0.0100398, 0.0079588
6: -0.0026366, -0.0015252, -0.0026850, -0.0016554, -0.0009813, 0.0011598
7: -0.0099594, -0.0070838, -0.0100847, -0.0074205, -0.0025388, 0.0030009
8: -0.0048017, -0.0027214, -0.0048676, -0.0020873, -0.0027144, 0.0021461
9: 0.0019504, 0.0037039, 0.0021558, 0.0037803, -0.0018299, 0.0015482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064728
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9783083, 0.9895449, -0.0072400, 0.0110551
1: -0.0045096, -0.0039143, -0.0045677, -0.0038691, -0.0006405, 0.0006534
2: 0.0106898, 0.0138447, 0.0104502, 0.0141526, -0.0034628, 0.0033945
3: -0.0076520, -0.0061387, -0.0078546, -0.0060296, -0.0016224, 0.0017159
4: 0.0025969, 0.0035423, 0.0025505, 0.0038717, -0.0012748, 0.0009918
5: 0.0124045, 0.0198408, 0.0121031, 0.0230236, -0.0106192, 0.0077377
6: -0.0026147, -0.0016076, -0.0027130, -0.0015311, -0.0010836, 0.0011054
7: -0.0099026, -0.0072969, -0.0101569, -0.0070989, -0.0028036, 0.0028600
8: -0.0047718, -0.0030089, -0.0049056, -0.0017217, -0.0030501, 0.0018967
9: 0.0020803, 0.0036693, 0.0019597, 0.0038244, -0.0017440, 0.0017096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064721
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064721
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9783083, 0.9895449, -0.0081137, 0.0112181
1: -0.0045223, -0.0038737, -0.0045677, -0.0038691, -0.0006532, 0.0006941
2: 0.0104745, 0.0139120, 0.0104502, 0.0141526, -0.0036782, 0.0034618
3: -0.0076963, -0.0060406, -0.0078546, -0.0060296, -0.0016667, 0.0018139
4: 0.0025552, 0.0036143, 0.0025505, 0.0038717, -0.0013165, 0.0010638
5: 0.0121336, 0.0205366, 0.0121031, 0.0230236, -0.0108901, 0.0084335
6: -0.0026362, -0.0015388, -0.0027130, -0.0015311, -0.0011051, 0.0011742
7: -0.0099582, -0.0071190, -0.0101569, -0.0070989, -0.0028592, 0.0030379
8: -0.0048010, -0.0027275, -0.0049056, -0.0017217, -0.0030793, 0.0021780
9: 0.0019719, 0.0037032, 0.0019597, 0.0038244, -0.0018525, 0.0017435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9794665, 0.9893759, 0.9837303, 0.9892544, -0.0097879, 0.0056457
1: -0.0045509, -0.0039112, -0.0044889, -0.0039415, -0.0006094, 0.0005777
2: 0.0106733, 0.0140634, 0.0108338, 0.0137349, -0.0030616, 0.0032296
3: -0.0077959, -0.0061311, -0.0075798, -0.0062042, -0.0015917, 0.0014487
4: 0.0025937, 0.0037762, 0.0026247, 0.0034248, -0.0008311, 0.0011515
5: 0.0123836, 0.0221013, 0.0125855, 0.0187057, -0.0063221, 0.0095158
6: -0.0026845, -0.0016023, -0.0025796, -0.0016535, -0.0010310, 0.0009774
7: -0.0100832, -0.0072832, -0.0098119, -0.0074157, -0.0026674, 0.0025287
8: -0.0048668, -0.0020947, -0.0047241, -0.0034640, -0.0014028, 0.0026294
9: 0.0020720, 0.0037794, 0.0021528, 0.0036140, -0.0015420, 0.0016266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064301
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065146
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9783124, 0.9894611, 0.9837303, 0.9892544, -0.0109419, 0.0057309
1: -0.0045677, -0.0038900, -0.0044889, -0.0039415, -0.0006262, 0.0005989
2: 0.0105608, 0.0141523, 0.0108338, 0.0137349, -0.0031741, 0.0033185
3: -0.0078543, -0.0060799, -0.0075798, -0.0062042, -0.0016502, 0.0014998
4: 0.0025719, 0.0038713, 0.0026247, 0.0034248, -0.0008529, 0.0012466
5: 0.0122422, 0.0230203, 0.0125855, 0.0187057, -0.0064635, 0.0104348
6: -0.0027129, -0.0015664, -0.0025796, -0.0016535, -0.0010594, 0.0010132
7: -0.0101566, -0.0071903, -0.0098119, -0.0074157, -0.0027409, 0.0026216
8: -0.0049054, -0.0017231, -0.0047241, -0.0034640, -0.0014414, 0.0030011
9: 0.0020154, 0.0038242, 0.0021528, 0.0036140, -0.0015986, 0.0016714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9794665, 0.9893759, 0.9804042, 0.9891771, -0.0097106, 0.0089718
1: -0.0045509, -0.0039112, -0.0045373, -0.0039607, -0.0005901, 0.0006261
2: 0.0106733, 0.0140634, 0.0109359, 0.0139911, -0.0033179, 0.0031275
3: -0.0077959, -0.0061311, -0.0077483, -0.0062507, -0.0015452, 0.0016172
4: 0.0025937, 0.0037762, 0.0026445, 0.0036989, -0.0011052, 0.0011317
5: 0.0123836, 0.0221013, 0.0127139, 0.0213545, -0.0089708, 0.0093874
6: -0.0026845, -0.0016023, -0.0026614, -0.0016861, -0.0009984, 0.0010592
7: -0.0100832, -0.0072832, -0.0100235, -0.0075001, -0.0025831, 0.0027404
8: -0.0048668, -0.0020947, -0.0048354, -0.0023968, -0.0024700, 0.0027407
9: 0.0020720, 0.0037794, 0.0022043, 0.0037430, -0.0016710, 0.0015752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064301
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9783124, 0.9894611, 0.9804042, 0.9891771, -0.0108647, 0.0090569
1: -0.0045677, -0.0038900, -0.0045373, -0.0039607, -0.0006069, 0.0006473
2: 0.0105608, 0.0141523, 0.0109359, 0.0139911, -0.0034303, 0.0032164
3: -0.0078543, -0.0060799, -0.0077483, -0.0062507, -0.0016037, 0.0016684
4: 0.0025719, 0.0038713, 0.0026445, 0.0036989, -0.0011270, 0.0012268
5: 0.0122422, 0.0230203, 0.0127139, 0.0213545, -0.0091123, 0.0103065
6: -0.0027129, -0.0015664, -0.0026614, -0.0016861, -0.0010268, 0.0010950
7: -0.0101566, -0.0071903, -0.0100235, -0.0075001, -0.0026566, 0.0028332
8: -0.0049054, -0.0017231, -0.0048354, -0.0023968, -0.0025087, 0.0031124
9: 0.0020154, 0.0038242, 0.0022043, 0.0037430, -0.0017277, 0.0016200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9794148, 0.9893565, 0.9828223, 0.9894229, -0.0100080, 0.0065342
1: -0.0045517, -0.0039160, -0.0045021, -0.0038995, -0.0006522, 0.0005861
2: 0.0106990, 0.0140674, 0.0106112, 0.0138048, -0.0031059, 0.0034561
3: -0.0077985, -0.0061428, -0.0076258, -0.0061029, -0.0016956, 0.0014830
4: 0.0025987, 0.0037805, 0.0025817, 0.0034996, -0.0009010, 0.0011988
5: 0.0124159, 0.0221424, 0.0123056, 0.0194288, -0.0070129, 0.0098368
6: -0.0026857, -0.0016105, -0.0026019, -0.0015825, -0.0011033, 0.0009915
7: -0.0100865, -0.0073044, -0.0098696, -0.0072319, -0.0028545, 0.0025653
8: -0.0048685, -0.0020781, -0.0047545, -0.0031755, -0.0016930, 0.0026764
9: 0.0020849, 0.0037814, 0.0020408, 0.0036492, -0.0015643, 0.0017407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064301
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9783099, 0.9895312, 0.9828223, 0.9894229, -0.0111130, 0.0067089
1: -0.0045677, -0.0038725, -0.0045021, -0.0038995, -0.0006682, 0.0006296
2: 0.0104682, 0.0141525, 0.0106112, 0.0138048, -0.0033367, 0.0035413
3: -0.0078545, -0.0060378, -0.0076258, -0.0061029, -0.0017516, 0.0015880
4: 0.0025540, 0.0038715, 0.0025817, 0.0034996, -0.0009456, 0.0012899
5: 0.0121257, 0.0230224, 0.0123056, 0.0194288, -0.0073031, 0.0107168
6: -0.0027129, -0.0015368, -0.0026019, -0.0015825, -0.0011305, 0.0010652
7: -0.0101568, -0.0071138, -0.0098696, -0.0072319, -0.0029249, 0.0027559
8: -0.0049055, -0.0017222, -0.0047545, -0.0031755, -0.0017300, 0.0030323
9: 0.0019687, 0.0038243, 0.0020408, 0.0036492, -0.0016805, 0.0017836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065146
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9794148, 0.9893565, 0.9794893, 0.9893448, -0.0099300, 0.0098673
1: -0.0045517, -0.0039160, -0.0045506, -0.0039190, -0.0006327, 0.0006345
2: 0.0106990, 0.0140674, 0.0107144, 0.0140616, -0.0033627, 0.0033530
3: -0.0077985, -0.0061428, -0.0077947, -0.0061498, -0.0016486, 0.0016519
4: 0.0025987, 0.0037805, 0.0026016, 0.0037743, -0.0011757, 0.0011788
5: 0.0124159, 0.0221424, 0.0124353, 0.0220831, -0.0096672, 0.0097071
6: -0.0026857, -0.0016105, -0.0026839, -0.0016154, -0.0010704, 0.0010735
7: -0.0100865, -0.0073044, -0.0100817, -0.0073171, -0.0027694, 0.0027773
8: -0.0048685, -0.0020781, -0.0048660, -0.0021021, -0.0027665, 0.0027879
9: 0.0020849, 0.0037814, 0.0020927, 0.0037786, -0.0016936, 0.0016887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064301
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9783099, 0.9895312, 0.9794893, 0.9893448, -0.0110350, 0.0100420
1: -0.0045677, -0.0038725, -0.0045506, -0.0039190, -0.0006488, 0.0006781
2: 0.0104682, 0.0141525, 0.0107144, 0.0140616, -0.0035935, 0.0034381
3: -0.0078545, -0.0060378, -0.0077947, -0.0061498, -0.0017046, 0.0017569
4: 0.0025540, 0.0038715, 0.0026016, 0.0037743, -0.0012204, 0.0012699
5: 0.0121257, 0.0230224, 0.0124353, 0.0220831, -0.0099575, 0.0105871
6: -0.0027129, -0.0015368, -0.0026839, -0.0016154, -0.0010975, 0.0011471
7: -0.0101568, -0.0071138, -0.0100817, -0.0073171, -0.0028397, 0.0029680
8: -0.0049055, -0.0017222, -0.0048660, -0.0021021, -0.0028034, 0.0031438
9: 0.0019687, 0.0038243, 0.0020927, 0.0037786, -0.0018099, 0.0017316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9794665, 0.9893759, 0.9823049, 0.9893634, -0.0098969, 0.0070710
1: -0.0045509, -0.0039112, -0.0045096, -0.0039143, -0.0006366, 0.0005984
2: 0.0106733, 0.0140634, 0.0106898, 0.0138447, -0.0031714, 0.0033735
3: -0.0077959, -0.0061311, -0.0076520, -0.0061387, -0.0016572, 0.0015209
4: 0.0025937, 0.0037762, 0.0025969, 0.0035423, -0.0009486, 0.0011793
5: 0.0123836, 0.0221013, 0.0124045, 0.0198408, -0.0074572, 0.0096968
6: -0.0026845, -0.0016023, -0.0026147, -0.0016076, -0.0010769, 0.0010124
7: -0.0100832, -0.0072832, -0.0099026, -0.0072969, -0.0027863, 0.0026194
8: -0.0048668, -0.0020947, -0.0047718, -0.0030089, -0.0018579, 0.0026771
9: 0.0020720, 0.0037794, 0.0020803, 0.0036693, -0.0015973, 0.0016991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064290
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065138
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9783124, 0.9894611, 0.9823049, 0.9893634, -0.0110510, 0.0071562
1: -0.0045677, -0.0038900, -0.0045096, -0.0039143, -0.0006534, 0.0006197
2: 0.0105608, 0.0141523, 0.0106898, 0.0138447, -0.0032839, 0.0034624
3: -0.0078543, -0.0060799, -0.0076520, -0.0061387, -0.0017157, 0.0015721
4: 0.0025719, 0.0038713, 0.0025969, 0.0035423, -0.0009704, 0.0012744
5: 0.0122422, 0.0230203, 0.0124045, 0.0198408, -0.0075986, 0.0106159
6: -0.0027129, -0.0015664, -0.0026147, -0.0016076, -0.0011053, 0.0010483
7: -0.0101566, -0.0071903, -0.0099026, -0.0072969, -0.0028598, 0.0027123
8: -0.0049054, -0.0017231, -0.0047718, -0.0030089, -0.0018965, 0.0030488
9: 0.0020154, 0.0038242, 0.0020803, 0.0036693, -0.0016539, 0.0017439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9794665, 0.9893759, 0.9794433, 0.9892499, -0.0097834, 0.0099326
1: -0.0045509, -0.0039112, -0.0045512, -0.0039426, -0.0006083, 0.0006400
2: 0.0106733, 0.0140634, 0.0108396, 0.0140652, -0.0033919, 0.0032238
3: -0.0077959, -0.0061311, -0.0077970, -0.0062068, -0.0015890, 0.0016659
4: 0.0025937, 0.0037762, 0.0026259, 0.0037781, -0.0011844, 0.0011503
5: 0.0123836, 0.0221013, 0.0125928, 0.0221198, -0.0097362, 0.0095085
6: -0.0026845, -0.0016023, -0.0026850, -0.0016554, -0.0010291, 0.0010828
7: -0.0100832, -0.0072832, -0.0100847, -0.0074205, -0.0026627, 0.0028015
8: -0.0048668, -0.0020947, -0.0048676, -0.0020873, -0.0027795, 0.0027728
9: 0.0020720, 0.0037794, 0.0021558, 0.0037803, -0.0017083, 0.0016237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064290
time: 1.29 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9783124, 0.9894611, 0.9794433, 0.9892499, -0.0109375, 0.0100178
1: -0.0045677, -0.0038900, -0.0045512, -0.0039426, -0.0006251, 0.0006613
2: 0.0105608, 0.0141523, 0.0108396, 0.0140652, -0.0035043, 0.0033127
3: -0.0078543, -0.0060799, -0.0077970, -0.0062068, -0.0016475, 0.0017171
4: 0.0025719, 0.0038713, 0.0026259, 0.0037781, -0.0012062, 0.0012455
5: 0.0122422, 0.0230203, 0.0125928, 0.0221198, -0.0098776, 0.0104276
6: -0.0027129, -0.0015664, -0.0026850, -0.0016554, -0.0010575, 0.0011187
7: -0.0101566, -0.0071903, -0.0100847, -0.0074205, -0.0027361, 0.0028944
8: -0.0049054, -0.0017231, -0.0048676, -0.0020873, -0.0028181, 0.0031445
9: 0.0020154, 0.0038242, 0.0021558, 0.0037803, -0.0017650, 0.0016685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9794502, 0.9894193, 0.9814312, 0.9895265, -0.0100762, 0.0079880
1: -0.0045511, -0.0039004, -0.0045223, -0.0038737, -0.0006775, 0.0006219
2: 0.0106160, 0.0140646, 0.0104745, 0.0139120, -0.0032960, 0.0035902
3: -0.0077967, -0.0061051, -0.0076963, -0.0060406, -0.0017561, 0.0015912
4: 0.0025826, 0.0037776, 0.0025552, 0.0036143, -0.0010317, 0.0012224
5: 0.0123116, 0.0221142, 0.0121336, 0.0205366, -0.0082249, 0.0099807
6: -0.0026849, -0.0015840, -0.0026362, -0.0015388, -0.0011461, 0.0010522
7: -0.0100842, -0.0072359, -0.0099582, -0.0071190, -0.0029653, 0.0027223
8: -0.0048673, -0.0020895, -0.0048010, -0.0027275, -0.0021398, 0.0027115
9: 0.0020432, 0.0037801, 0.0019719, 0.0037032, -0.0016600, 0.0018082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064290
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065138
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9782971, 0.9895055, 0.9814312, 0.9895265, -0.0112293, 0.0080743
1: -0.0045679, -0.0038789, -0.0045223, -0.0038737, -0.0006942, 0.0006434
2: 0.0105020, 0.0141535, 0.0104745, 0.0139120, -0.0034100, 0.0036790
3: -0.0078551, -0.0060532, -0.0076963, -0.0060406, -0.0018145, 0.0016431
4: 0.0025605, 0.0038726, 0.0025552, 0.0036143, -0.0010537, 0.0013174
5: 0.0121683, 0.0230325, 0.0121336, 0.0205366, -0.0083683, 0.0108990
6: -0.0027132, -0.0015476, -0.0026362, -0.0015388, -0.0011744, 0.0010886
7: -0.0101576, -0.0071418, -0.0099582, -0.0071190, -0.0030386, 0.0028164
8: -0.0049059, -0.0017181, -0.0048010, -0.0027275, -0.0021784, 0.0030829
9: 0.0019858, 0.0038248, 0.0019719, 0.0037032, -0.0017174, 0.0018529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
time: 0.84 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9794132, 0.9893691, 0.9783329, 0.9894307, -0.0100176, 0.0110362
1: -0.0045517, -0.0039129, -0.0045674, -0.0038975, -0.0006541, 0.0006545
2: 0.0106824, 0.0140675, 0.0106009, 0.0141507, -0.0034683, 0.0034666
3: -0.0077986, -0.0061353, -0.0078533, -0.0060982, -0.0017004, 0.0017180
4: 0.0025954, 0.0037806, 0.0025797, 0.0038696, -0.0012742, 0.0012010
5: 0.0123951, 0.0221438, 0.0122926, 0.0230040, -0.0106089, 0.0098512
6: -0.0026858, -0.0016052, -0.0027124, -0.0015792, -0.0011066, 0.0011072
7: -0.0100866, -0.0072907, -0.0101553, -0.0072234, -0.0028632, 0.0028646
8: -0.0048686, -0.0020776, -0.0049047, -0.0017297, -0.0031389, 0.0028272
9: 0.0020766, 0.0037815, 0.0020355, 0.0038234, -0.0017468, 0.0017460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064290
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9783083, 0.9895449, 0.9783329, 0.9894307, -0.0111224, 0.0112121
1: -0.0045677, -0.0038691, -0.0045674, -0.0038975, -0.0006702, 0.0006983
2: 0.0104502, 0.0141526, 0.0106009, 0.0141507, -0.0037005, 0.0035517
3: -0.0078546, -0.0060296, -0.0078533, -0.0060982, -0.0017564, 0.0018237
4: 0.0025505, 0.0038717, 0.0025797, 0.0038696, -0.0013191, 0.0012920
5: 0.0121031, 0.0230236, 0.0122926, 0.0230040, -0.0109009, 0.0107310
6: -0.0027130, -0.0015311, -0.0027124, -0.0015792, -0.0011338, 0.0011813
7: -0.0101569, -0.0070989, -0.0101553, -0.0072234, -0.0029335, 0.0030564
8: -0.0049056, -0.0017217, -0.0049047, -0.0017297, -0.0031759, 0.0031830
9: 0.0019597, 0.0038244, 0.0020355, 0.0038234, -0.0018638, 0.0017888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
time: 1.03 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
time: 0.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.40 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
IS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
IS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
IS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0064356
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064358, upper bound: 0.0065240
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064797
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064796
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064795
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0064795
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064356, upper bound: 0.0065664
IS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
IS_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
IS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0064297
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064301, upper bound: 0.0065182
IS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064728
IS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
IS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064728
IS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064721
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0064721
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064290, upper bound: 0.0065618
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064301
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065146
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064301
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064301
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065146
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064301
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065146
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064290
IS_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065138
IS_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
IS_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
IS_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064290
IS_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
IS_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
IS_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442
IS_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064290
IS_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065138
IS_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0064600
IS_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064297, upper bound: 0.0065442
IS_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064290
IS_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0064600
IS_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065138
IS_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.40
Output dim: 0, lower bound: -0.0064139, upper bound: 0.0065442

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9837303, 0.9892544, -0.0055241, 0.0055241
1: -0.0044889, -0.0039415, -0.0044889, -0.0039415, -0.0005474, 0.0005474
2: 0.0108338, 0.0137349, 0.0108338, 0.0137349, -0.0029011, 0.0029011
3: -0.0075798, -0.0062042, -0.0075798, -0.0062042, -0.0013756, 0.0013756
4: 0.0026247, 0.0034248, 0.0026247, 0.0034248, -0.0008000, 0.0008000
5: 0.0125855, 0.0187057, 0.0125855, 0.0187057, -0.0061202, 0.0061202
6: -0.0025796, -0.0016535, -0.0025796, -0.0016535, -0.0009261, 0.0009261
7: -0.0098119, -0.0074157, -0.0098119, -0.0074157, -0.0023961, 0.0023961
8: -0.0047241, -0.0034640, -0.0047241, -0.0034640, -0.0012601, 0.0012601
9: 0.0021528, 0.0036140, 0.0021528, 0.0036140, -0.0014611, 0.0014611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062524, upper bound: 0.0061950
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0061951
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9837303, 0.9892544, -0.0064321, 0.0056926
1: -0.0045021, -0.0038995, -0.0044889, -0.0039415, -0.0005606, 0.0005894
2: 0.0106112, 0.0138048, 0.0108338, 0.0137349, -0.0031236, 0.0029710
3: -0.0076258, -0.0061029, -0.0075798, -0.0062042, -0.0014216, 0.0014769
4: 0.0025817, 0.0034996, 0.0026247, 0.0034248, -0.0008431, 0.0008749
5: 0.0123056, 0.0194288, 0.0125855, 0.0187057, -0.0064001, 0.0068433
6: -0.0026019, -0.0015825, -0.0025796, -0.0016535, -0.0009484, 0.0009972
7: -0.0098696, -0.0072319, -0.0098119, -0.0074157, -0.0024539, 0.0025799
8: -0.0047545, -0.0031755, -0.0047241, -0.0034640, -0.0012905, 0.0015486
9: 0.0020408, 0.0036492, 0.0021528, 0.0036140, -0.0015732, 0.0014964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063461
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062835
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9823049, 0.9893634, -0.0056332, 0.0069494
1: -0.0044889, -0.0039415, -0.0045096, -0.0039143, -0.0005746, 0.0005681
2: 0.0108338, 0.0137349, 0.0106898, 0.0138447, -0.0030109, 0.0030450
3: -0.0075798, -0.0062042, -0.0076520, -0.0061387, -0.0014411, 0.0014478
4: 0.0026247, 0.0034248, 0.0025969, 0.0035423, -0.0009175, 0.0008279
5: 0.0125855, 0.0187057, 0.0124045, 0.0198408, -0.0072553, 0.0063013
6: -0.0025796, -0.0016535, -0.0026147, -0.0016076, -0.0009721, 0.0009612
7: -0.0098119, -0.0074157, -0.0099026, -0.0072969, -0.0025150, 0.0024868
8: -0.0047241, -0.0034640, -0.0047718, -0.0030089, -0.0017152, 0.0013078
9: 0.0021528, 0.0036140, 0.0020803, 0.0036693, -0.0015165, 0.0015336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0062507
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0061951
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9823049, 0.9893634, -0.0065411, 0.0071179
1: -0.0045021, -0.0038995, -0.0045096, -0.0039143, -0.0005878, 0.0006101
2: 0.0106112, 0.0138048, 0.0106898, 0.0138447, -0.0032335, 0.0031150
3: -0.0076258, -0.0061029, -0.0076520, -0.0061387, -0.0014871, 0.0015491
4: 0.0025817, 0.0034996, 0.0025969, 0.0035423, -0.0009606, 0.0009027
5: 0.0123056, 0.0194288, 0.0124045, 0.0198408, -0.0075352, 0.0070243
6: -0.0026019, -0.0015825, -0.0026147, -0.0016076, -0.0009944, 0.0010322
7: -0.0098696, -0.0072319, -0.0099026, -0.0072969, -0.0025728, 0.0026706
8: -0.0047545, -0.0031755, -0.0047718, -0.0030089, -0.0017456, 0.0015963
9: 0.0020408, 0.0036492, 0.0020803, 0.0036693, -0.0016285, 0.0015689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0063439
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0062832
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9828223, 0.9894229, -0.0056926, 0.0064321
1: -0.0044889, -0.0039415, -0.0045021, -0.0038995, -0.0005894, 0.0005606
2: 0.0108338, 0.0137349, 0.0106112, 0.0138048, -0.0029710, 0.0031236
3: -0.0075798, -0.0062042, -0.0076258, -0.0061029, -0.0014769, 0.0014216
4: 0.0026247, 0.0034248, 0.0025817, 0.0034996, -0.0008749, 0.0008431
5: 0.0125855, 0.0187057, 0.0123056, 0.0194288, -0.0068433, 0.0064001
6: -0.0025796, -0.0016535, -0.0026019, -0.0015825, -0.0009972, 0.0009484
7: -0.0098119, -0.0074157, -0.0098696, -0.0072319, -0.0025799, 0.0024539
8: -0.0047241, -0.0034640, -0.0047545, -0.0031755, -0.0015486, 0.0012905
9: 0.0021528, 0.0036140, 0.0020408, 0.0036492, -0.0014964, 0.0015732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063461, upper bound: 0.0061949
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062835, upper bound: 0.0061950
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9814312, 0.9895265, -0.0057962, 0.0078231
1: -0.0044889, -0.0039415, -0.0045223, -0.0038737, -0.0006152, 0.0005808
2: 0.0108338, 0.0137349, 0.0104745, 0.0139120, -0.0030782, 0.0032604
3: -0.0075798, -0.0062042, -0.0076963, -0.0060406, -0.0015392, 0.0014921
4: 0.0026247, 0.0034248, 0.0025552, 0.0036143, -0.0009895, 0.0008696
5: 0.0125855, 0.0187057, 0.0121336, 0.0205366, -0.0079511, 0.0065721
6: -0.0025796, -0.0016535, -0.0026362, -0.0015388, -0.0010408, 0.0009826
7: -0.0098119, -0.0074157, -0.0099582, -0.0071190, -0.0026929, 0.0025424
8: -0.0047241, -0.0034640, -0.0048010, -0.0027275, -0.0019966, 0.0013370
9: 0.0021528, 0.0036140, 0.0019719, 0.0037032, -0.0015504, 0.0016421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063461, upper bound: 0.0061949
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062835, upper bound: 0.0061950
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9828223, 0.9894229, -0.0066006, 0.0066006
1: -0.0045021, -0.0038995, -0.0045021, -0.0038995, -0.0006026, 0.0006026
2: 0.0106112, 0.0138048, 0.0106112, 0.0138048, -0.0031936, 0.0031936
3: -0.0076258, -0.0061029, -0.0076258, -0.0061029, -0.0015229, 0.0015229
4: 0.0025817, 0.0034996, 0.0025817, 0.0034996, -0.0009180, 0.0009180
5: 0.0123056, 0.0194288, 0.0123056, 0.0194288, -0.0071232, 0.0071232
6: -0.0026019, -0.0015825, -0.0026019, -0.0015825, -0.0010195, 0.0010195
7: -0.0098696, -0.0072319, -0.0098696, -0.0072319, -0.0026377, 0.0026377
8: -0.0047545, -0.0031755, -0.0047545, -0.0031755, -0.0015790, 0.0015790
9: 0.0020408, 0.0036492, 0.0020408, 0.0036492, -0.0016085, 0.0016085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062524, upper bound: 0.0062817
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062832
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9814312, 0.9895265, -0.0067042, 0.0079916
1: -0.0045021, -0.0038995, -0.0045223, -0.0038737, -0.0006284, 0.0006228
2: 0.0106112, 0.0138048, 0.0104745, 0.0139120, -0.0033008, 0.0033304
3: -0.0076258, -0.0061029, -0.0076963, -0.0060406, -0.0015852, 0.0015934
4: 0.0025817, 0.0034996, 0.0025552, 0.0036143, -0.0010326, 0.0009444
5: 0.0123056, 0.0194288, 0.0121336, 0.0205366, -0.0082310, 0.0072952
6: -0.0026019, -0.0015825, -0.0026362, -0.0015388, -0.0010632, 0.0010537
7: -0.0098696, -0.0072319, -0.0099582, -0.0071190, -0.0027507, 0.0027262
8: -0.0047545, -0.0031755, -0.0048010, -0.0027275, -0.0020270, 0.0016255
9: 0.0020408, 0.0036492, 0.0019719, 0.0037032, -0.0016624, 0.0016773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063439
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062832
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9837303, 0.9892544, -0.0069494, 0.0056332
1: -0.0045096, -0.0039143, -0.0044889, -0.0039415, -0.0005681, 0.0005746
2: 0.0106898, 0.0138447, 0.0108338, 0.0137349, -0.0030450, 0.0030109
3: -0.0076520, -0.0061387, -0.0075798, -0.0062042, -0.0014478, 0.0014411
4: 0.0025969, 0.0035423, 0.0026247, 0.0034248, -0.0008279, 0.0009175
5: 0.0124045, 0.0198408, 0.0125855, 0.0187057, -0.0063013, 0.0072553
6: -0.0026147, -0.0016076, -0.0025796, -0.0016535, -0.0009612, 0.0009721
7: -0.0099026, -0.0072969, -0.0098119, -0.0074157, -0.0024868, 0.0025150
8: -0.0047718, -0.0030089, -0.0047241, -0.0034640, -0.0013078, 0.0017152
9: 0.0020803, 0.0036693, 0.0021528, 0.0036140, -0.0015336, 0.0015165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0062420
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9837303, 0.9892544, -0.0078231, 0.0057962
1: -0.0045223, -0.0038737, -0.0044889, -0.0039415, -0.0005808, 0.0006152
2: 0.0104745, 0.0139120, 0.0108338, 0.0137349, -0.0032604, 0.0030782
3: -0.0076963, -0.0060406, -0.0075798, -0.0062042, -0.0014921, 0.0015392
4: 0.0025552, 0.0036143, 0.0026247, 0.0034248, -0.0008696, 0.0009895
5: 0.0121336, 0.0205366, 0.0125855, 0.0187057, -0.0065721, 0.0079511
6: -0.0026362, -0.0015388, -0.0025796, -0.0016535, -0.0009826, 0.0010408
7: -0.0099582, -0.0071190, -0.0098119, -0.0074157, -0.0025424, 0.0026929
8: -0.0048010, -0.0027275, -0.0047241, -0.0034640, -0.0013370, 0.0019966
9: 0.0019719, 0.0037032, 0.0021528, 0.0036140, -0.0016421, 0.0015504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063872
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9823049, 0.9893634, -0.0070585, 0.0070585
1: -0.0045096, -0.0039143, -0.0045096, -0.0039143, -0.0005953, 0.0005953
2: 0.0106898, 0.0138447, 0.0106898, 0.0138447, -0.0031549, 0.0031549
3: -0.0076520, -0.0061387, -0.0076520, -0.0061387, -0.0015133, 0.0015133
4: 0.0025969, 0.0035423, 0.0025969, 0.0035423, -0.0009454, 0.0009454
5: 0.0124045, 0.0198408, 0.0124045, 0.0198408, -0.0074363, 0.0074363
6: -0.0026147, -0.0016076, -0.0026147, -0.0016076, -0.0010071, 0.0010071
7: -0.0099026, -0.0072969, -0.0099026, -0.0072969, -0.0026057, 0.0026057
8: -0.0047718, -0.0030089, -0.0047718, -0.0030089, -0.0017629, 0.0017629
9: 0.0020803, 0.0036693, 0.0020803, 0.0036693, -0.0015889, 0.0015889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0062957
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9823049, 0.9893634, -0.0079322, 0.0072215
1: -0.0045223, -0.0038737, -0.0045096, -0.0039143, -0.0006080, 0.0006359
2: 0.0104745, 0.0139120, 0.0106898, 0.0138447, -0.0033702, 0.0032222
3: -0.0076963, -0.0060406, -0.0076520, -0.0061387, -0.0015576, 0.0016114
4: 0.0025552, 0.0036143, 0.0025969, 0.0035423, -0.0009871, 0.0010174
5: 0.0121336, 0.0205366, 0.0124045, 0.0198408, -0.0077072, 0.0081321
6: -0.0026362, -0.0015388, -0.0026147, -0.0016076, -0.0010286, 0.0010759
7: -0.0099582, -0.0071190, -0.0099026, -0.0072969, -0.0026613, 0.0027836
8: -0.0048010, -0.0027275, -0.0047718, -0.0030089, -0.0017921, 0.0020443
9: 0.0019719, 0.0037032, 0.0020803, 0.0036693, -0.0016974, 0.0016228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063872
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9828223, 0.9894229, -0.0071179, 0.0065411
1: -0.0045096, -0.0039143, -0.0045021, -0.0038995, -0.0006101, 0.0005878
2: 0.0106898, 0.0138447, 0.0106112, 0.0138048, -0.0031150, 0.0032335
3: -0.0076520, -0.0061387, -0.0076258, -0.0061029, -0.0015491, 0.0014871
4: 0.0025969, 0.0035423, 0.0025817, 0.0034996, -0.0009027, 0.0009606
5: 0.0124045, 0.0198408, 0.0123056, 0.0194288, -0.0070243, 0.0075352
6: -0.0026147, -0.0016076, -0.0026019, -0.0015825, -0.0010322, 0.0009944
7: -0.0099026, -0.0072969, -0.0098696, -0.0072319, -0.0026706, 0.0025728
8: -0.0047718, -0.0030089, -0.0047545, -0.0031755, -0.0015963, 0.0017456
9: 0.0020803, 0.0036693, 0.0020408, 0.0036492, -0.0015689, 0.0016285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0062419
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9828223, 0.9894229, -0.0079916, 0.0067042
1: -0.0045223, -0.0038737, -0.0045021, -0.0038995, -0.0006228, 0.0006284
2: 0.0104745, 0.0139120, 0.0106112, 0.0138048, -0.0033304, 0.0033008
3: -0.0076963, -0.0060406, -0.0076258, -0.0061029, -0.0015934, 0.0015852
4: 0.0025552, 0.0036143, 0.0025817, 0.0034996, -0.0009444, 0.0010326
5: 0.0121336, 0.0205366, 0.0123056, 0.0194288, -0.0072952, 0.0082310
6: -0.0026362, -0.0015388, -0.0026019, -0.0015825, -0.0010537, 0.0010632
7: -0.0099582, -0.0071190, -0.0098696, -0.0072319, -0.0027262, 0.0027507
8: -0.0048010, -0.0027275, -0.0047545, -0.0031755, -0.0016255, 0.0020270
9: 0.0019719, 0.0037032, 0.0020408, 0.0036492, -0.0016773, 0.0016624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0063216
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9814312, 0.9895265, -0.0072215, 0.0079322
1: -0.0045096, -0.0039143, -0.0045223, -0.0038737, -0.0006359, 0.0006080
2: 0.0106898, 0.0138447, 0.0104745, 0.0139120, -0.0032222, 0.0033702
3: -0.0076520, -0.0061387, -0.0076963, -0.0060406, -0.0016114, 0.0015576
4: 0.0025969, 0.0035423, 0.0025552, 0.0036143, -0.0010174, 0.0009871
5: 0.0124045, 0.0198408, 0.0121336, 0.0205366, -0.0081321, 0.0077072
6: -0.0026147, -0.0016076, -0.0026362, -0.0015388, -0.0010759, 0.0010286
7: -0.0099026, -0.0072969, -0.0099582, -0.0071190, -0.0027836, 0.0026613
8: -0.0047718, -0.0030089, -0.0048010, -0.0027275, -0.0020443, 0.0017921
9: 0.0020803, 0.0036693, 0.0019719, 0.0037032, -0.0016228, 0.0016974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0062419
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9814312, 0.9895265, -0.0080952, 0.0080952
1: -0.0045223, -0.0038737, -0.0045223, -0.0038737, -0.0006487, 0.0006487
2: 0.0104745, 0.0139120, 0.0104745, 0.0139120, -0.0034375, 0.0034375
3: -0.0076963, -0.0060406, -0.0076963, -0.0060406, -0.0016557, 0.0016557
4: 0.0025552, 0.0036143, 0.0025552, 0.0036143, -0.0010591, 0.0010591
5: 0.0121336, 0.0205366, 0.0121336, 0.0205366, -0.0084030, 0.0084030
6: -0.0026362, -0.0015388, -0.0026362, -0.0015388, -0.0010974, 0.0010974
7: -0.0099582, -0.0071190, -0.0099582, -0.0071190, -0.0028392, 0.0028392
8: -0.0048010, -0.0027275, -0.0048010, -0.0027275, -0.0020735, 0.0020735
9: 0.0019719, 0.0037032, 0.0019719, 0.0037032, -0.0017313, 0.0017313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0063216
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9804042, 0.9891771, -0.0054469, 0.0088502
1: -0.0044889, -0.0039415, -0.0045373, -0.0039607, -0.0005282, 0.0005958
2: 0.0108338, 0.0137349, 0.0109359, 0.0139911, -0.0031573, 0.0027990
3: -0.0075798, -0.0062042, -0.0077483, -0.0062507, -0.0013291, 0.0015442
4: 0.0026247, 0.0034248, 0.0026445, 0.0036989, -0.0010742, 0.0007803
5: 0.0125855, 0.0187057, 0.0127139, 0.0213545, -0.0087690, 0.0059918
6: -0.0025796, -0.0016535, -0.0026614, -0.0016861, -0.0008935, 0.0010079
7: -0.0098119, -0.0074157, -0.0100235, -0.0075001, -0.0023118, 0.0026078
8: -0.0047241, -0.0034640, -0.0048354, -0.0023968, -0.0023273, 0.0013714
9: 0.0021528, 0.0036140, 0.0022043, 0.0037430, -0.0015902, 0.0014097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061971, upper bound: 0.0062461
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061972, upper bound: 0.0061920
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9794893, 0.9893448, -0.0056146, 0.0097651
1: -0.0044889, -0.0039415, -0.0045506, -0.0039190, -0.0005700, 0.0006091
2: 0.0108338, 0.0137349, 0.0107144, 0.0140616, -0.0032278, 0.0030205
3: -0.0075798, -0.0062042, -0.0077947, -0.0061498, -0.0014300, 0.0015905
4: 0.0026247, 0.0034248, 0.0026016, 0.0037743, -0.0011496, 0.0008231
5: 0.0125855, 0.0187057, 0.0124353, 0.0220831, -0.0094976, 0.0062704
6: -0.0025796, -0.0016535, -0.0026839, -0.0016154, -0.0009642, 0.0010304
7: -0.0098119, -0.0074157, -0.0100817, -0.0073171, -0.0024948, 0.0026660
8: -0.0047241, -0.0034640, -0.0048660, -0.0021021, -0.0026220, 0.0014020
9: 0.0021528, 0.0036140, 0.0020927, 0.0037786, -0.0016257, 0.0015213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061971, upper bound: 0.0062461
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061972, upper bound: 0.0061920
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9794433, 0.9892499, -0.0055197, 0.0098110
1: -0.0044889, -0.0039415, -0.0045512, -0.0039426, -0.0005463, 0.0006098
2: 0.0108338, 0.0137349, 0.0108396, 0.0140652, -0.0032314, 0.0028953
3: -0.0075798, -0.0062042, -0.0077970, -0.0062068, -0.0013730, 0.0015929
4: 0.0026247, 0.0034248, 0.0026259, 0.0037781, -0.0011534, 0.0007989
5: 0.0125855, 0.0187057, 0.0125928, 0.0221198, -0.0095343, 0.0061129
6: -0.0025796, -0.0016535, -0.0026850, -0.0016554, -0.0009243, 0.0010315
7: -0.0098119, -0.0074157, -0.0100847, -0.0074205, -0.0023913, 0.0026689
8: -0.0047241, -0.0034640, -0.0048676, -0.0020873, -0.0026368, 0.0014036
9: 0.0021528, 0.0036140, 0.0021558, 0.0037803, -0.0016275, 0.0014582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062229, upper bound: 0.0062434
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062241, upper bound: 0.0061914
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9837303, 0.9892544, 0.9783329, 0.9894307, -0.0057005, 0.0109215
1: -0.0044889, -0.0039415, -0.0045674, -0.0038975, -0.0005914, 0.0006259
2: 0.0108338, 0.0137349, 0.0106009, 0.0141507, -0.0033169, 0.0031340
3: -0.0075798, -0.0062042, -0.0078533, -0.0060982, -0.0014816, 0.0016491
4: 0.0026247, 0.0034248, 0.0025797, 0.0038696, -0.0012449, 0.0008451
5: 0.0125855, 0.0187057, 0.0122926, 0.0230040, -0.0104185, 0.0064131
6: -0.0025796, -0.0016535, -0.0027124, -0.0015792, -0.0010005, 0.0010588
7: -0.0098119, -0.0074157, -0.0101553, -0.0072234, -0.0025885, 0.0027396
8: -0.0047241, -0.0034640, -0.0049047, -0.0017297, -0.0029945, 0.0014407
9: 0.0021528, 0.0036140, 0.0020355, 0.0038234, -0.0016706, 0.0015784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062229, upper bound: 0.0062434
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062241, upper bound: 0.0061914
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9804042, 0.9891771, -0.0063548, 0.0090187
1: -0.0045021, -0.0038995, -0.0045373, -0.0039607, -0.0005414, 0.0006378
2: 0.0106112, 0.0138048, 0.0109359, 0.0139911, -0.0033799, 0.0028690
3: -0.0076258, -0.0061029, -0.0077483, -0.0062507, -0.0013752, 0.0016454
4: 0.0025817, 0.0034996, 0.0026445, 0.0036989, -0.0011172, 0.0008551
5: 0.0123056, 0.0194288, 0.0127139, 0.0213545, -0.0090489, 0.0067149
6: -0.0026019, -0.0015825, -0.0026614, -0.0016861, -0.0009159, 0.0010790
7: -0.0098696, -0.0072319, -0.0100235, -0.0075001, -0.0023696, 0.0027916
8: -0.0047545, -0.0031755, -0.0048354, -0.0023968, -0.0023577, 0.0016599
9: 0.0020408, 0.0036492, 0.0022043, 0.0037430, -0.0017023, 0.0014450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9794433, 0.9892499, -0.0064276, 0.0099795
1: -0.0045021, -0.0038995, -0.0045512, -0.0039426, -0.0005595, 0.0006517
2: 0.0106112, 0.0138048, 0.0108396, 0.0140652, -0.0034539, 0.0029653
3: -0.0076258, -0.0061029, -0.0077970, -0.0062068, -0.0014190, 0.0016941
4: 0.0025817, 0.0034996, 0.0026259, 0.0037781, -0.0011965, 0.0008738
5: 0.0123056, 0.0194288, 0.0125928, 0.0221198, -0.0098142, 0.0068360
6: -0.0026019, -0.0015825, -0.0026850, -0.0016554, -0.0009466, 0.0011026
7: -0.0098696, -0.0072319, -0.0100847, -0.0074205, -0.0024491, 0.0028527
8: -0.0047545, -0.0031755, -0.0048676, -0.0020873, -0.0026672, 0.0016920
9: 0.0020408, 0.0036492, 0.0021558, 0.0037803, -0.0017396, 0.0014935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9794893, 0.9893448, -0.0065225, 0.0099336
1: -0.0045021, -0.0038995, -0.0045506, -0.0039190, -0.0005832, 0.0006511
2: 0.0106112, 0.0138048, 0.0107144, 0.0140616, -0.0034504, 0.0030905
3: -0.0076258, -0.0061029, -0.0077947, -0.0061498, -0.0014760, 0.0016918
4: 0.0025817, 0.0034996, 0.0026016, 0.0037743, -0.0011927, 0.0008980
5: 0.0123056, 0.0194288, 0.0124353, 0.0220831, -0.0097775, 0.0069935
6: -0.0026019, -0.0015825, -0.0026839, -0.0016154, -0.0009866, 0.0011015
7: -0.0098696, -0.0072319, -0.0100817, -0.0073171, -0.0025525, 0.0028498
8: -0.0047545, -0.0031755, -0.0048660, -0.0021021, -0.0026524, 0.0016905
9: 0.0020408, 0.0036492, 0.0020927, 0.0037786, -0.0017378, 0.0015565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9828223, 0.9894229, 0.9783329, 0.9894307, -0.0066084, 0.0110900
1: -0.0045021, -0.0038995, -0.0045674, -0.0038975, -0.0006046, 0.0006679
2: 0.0106112, 0.0138048, 0.0106009, 0.0141507, -0.0035395, 0.0032039
3: -0.0076258, -0.0061029, -0.0078533, -0.0060982, -0.0015276, 0.0017504
4: 0.0025817, 0.0034996, 0.0025797, 0.0038696, -0.0012880, 0.0009200
5: 0.0123056, 0.0194288, 0.0122926, 0.0230040, -0.0106984, 0.0071362
6: -0.0026019, -0.0015825, -0.0027124, -0.0015792, -0.0010228, 0.0011299
7: -0.0098696, -0.0072319, -0.0101553, -0.0072234, -0.0026463, 0.0029234
8: -0.0047545, -0.0031755, -0.0049047, -0.0017297, -0.0030248, 0.0017292
9: 0.0020408, 0.0036492, 0.0020355, 0.0038234, -0.0017827, 0.0016137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9804042, 0.9891771, -0.0068722, 0.0089592
1: -0.0045096, -0.0039143, -0.0045373, -0.0039607, -0.0005489, 0.0006229
2: 0.0106898, 0.0138447, 0.0109359, 0.0139911, -0.0033013, 0.0029088
3: -0.0076520, -0.0061387, -0.0077483, -0.0062507, -0.0014014, 0.0016097
4: 0.0025969, 0.0035423, 0.0026445, 0.0036989, -0.0011020, 0.0008978
5: 0.0124045, 0.0198408, 0.0127139, 0.0213545, -0.0089500, 0.0071269
6: -0.0026147, -0.0016076, -0.0026614, -0.0016861, -0.0009286, 0.0010539
7: -0.0099026, -0.0072969, -0.0100235, -0.0075001, -0.0024025, 0.0027267
8: -0.0047718, -0.0030089, -0.0048354, -0.0023968, -0.0023750, 0.0018265
9: 0.0020803, 0.0036693, 0.0022043, 0.0037430, -0.0016627, 0.0014650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0062889
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0062372
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9804042, 0.9891771, -0.0077459, 0.0091223
1: -0.0045223, -0.0038737, -0.0045373, -0.0039607, -0.0005616, 0.0006636
2: 0.0104745, 0.0139120, 0.0109359, 0.0139911, -0.0035167, 0.0029761
3: -0.0076963, -0.0060406, -0.0077483, -0.0062507, -0.0014456, 0.0017077
4: 0.0025552, 0.0036143, 0.0026445, 0.0036989, -0.0011437, 0.0009698
5: 0.0121336, 0.0205366, 0.0127139, 0.0213545, -0.0092209, 0.0078227
6: -0.0026362, -0.0015388, -0.0026614, -0.0016861, -0.0009501, 0.0011226
7: -0.0099582, -0.0071190, -0.0100235, -0.0075001, -0.0024581, 0.0029045
8: -0.0048010, -0.0027275, -0.0048354, -0.0023968, -0.0024043, 0.0021079
9: 0.0019719, 0.0037032, 0.0022043, 0.0037430, -0.0017712, 0.0014989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0063787
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0063219
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9794433, 0.9892499, -0.0069450, 0.0099201
1: -0.0045096, -0.0039143, -0.0045512, -0.0039426, -0.0005671, 0.0006369
2: 0.0106898, 0.0138447, 0.0108396, 0.0140652, -0.0033753, 0.0030051
3: -0.0076520, -0.0061387, -0.0077970, -0.0062068, -0.0014452, 0.0016584
4: 0.0025969, 0.0035423, 0.0026259, 0.0037781, -0.0011812, 0.0009164
5: 0.0124045, 0.0198408, 0.0125928, 0.0221198, -0.0097153, 0.0072480
6: -0.0026147, -0.0016076, -0.0026850, -0.0016554, -0.0009593, 0.0010775
7: -0.0099026, -0.0072969, -0.0100847, -0.0074205, -0.0024820, 0.0027878
8: -0.0047718, -0.0030089, -0.0048676, -0.0020873, -0.0026845, 0.0018587
9: 0.0020803, 0.0036693, 0.0021558, 0.0037803, -0.0017000, 0.0015135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0062889
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0062372
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9794433, 0.9892499, -0.0078187, 0.0100831
1: -0.0045223, -0.0038737, -0.0045512, -0.0039426, -0.0005798, 0.0006776
2: 0.0104745, 0.0139120, 0.0108396, 0.0140652, -0.0035907, 0.0030724
3: -0.0076963, -0.0060406, -0.0077970, -0.0062068, -0.0014895, 0.0017564
4: 0.0025552, 0.0036143, 0.0026259, 0.0037781, -0.0012229, 0.0009884
5: 0.0121336, 0.0205366, 0.0125928, 0.0221198, -0.0099862, 0.0079438
6: -0.0026362, -0.0015388, -0.0026850, -0.0016554, -0.0009808, 0.0011463
7: -0.0099582, -0.0071190, -0.0100847, -0.0074205, -0.0025376, 0.0029657
8: -0.0048010, -0.0027275, -0.0048676, -0.0020873, -0.0027138, 0.0021401
9: 0.0019719, 0.0037032, 0.0021558, 0.0037803, -0.0018085, 0.0015474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0063787
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0063219
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9794893, 0.9893448, -0.0070399, 0.0098742
1: -0.0045096, -0.0039143, -0.0045506, -0.0039190, -0.0005907, 0.0006362
2: 0.0106898, 0.0138447, 0.0107144, 0.0140616, -0.0033718, 0.0031303
3: -0.0076520, -0.0061387, -0.0077947, -0.0061498, -0.0015022, 0.0016560
4: 0.0025969, 0.0035423, 0.0026016, 0.0037743, -0.0011774, 0.0009406
5: 0.0124045, 0.0198408, 0.0124353, 0.0220831, -0.0096787, 0.0074055
6: -0.0026147, -0.0016076, -0.0026839, -0.0016154, -0.0009993, 0.0010764
7: -0.0099026, -0.0072969, -0.0100817, -0.0073171, -0.0025854, 0.0027849
8: -0.0047718, -0.0030089, -0.0048660, -0.0021021, -0.0026697, 0.0018571
9: 0.0020803, 0.0036693, 0.0020927, 0.0037786, -0.0016982, 0.0015766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062787, upper bound: 0.0062876
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062372
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9823049, 0.9893634, 0.9783329, 0.9894307, -0.0071258, 0.0110306
1: -0.0045096, -0.0039143, -0.0045674, -0.0038975, -0.0006121, 0.0006531
2: 0.0106898, 0.0138447, 0.0106009, 0.0141507, -0.0034609, 0.0032438
3: -0.0076520, -0.0061387, -0.0078533, -0.0060982, -0.0015538, 0.0017146
4: 0.0025969, 0.0035423, 0.0025797, 0.0038696, -0.0012728, 0.0009626
5: 0.0124045, 0.0198408, 0.0122926, 0.0230040, -0.0105995, 0.0075482
6: -0.0026147, -0.0016076, -0.0027124, -0.0015792, -0.0010355, 0.0011048
7: -0.0099026, -0.0072969, -0.0101553, -0.0072234, -0.0026792, 0.0028585
8: -0.0047718, -0.0030089, -0.0049047, -0.0017297, -0.0030421, 0.0018958
9: 0.0020803, 0.0036693, 0.0020355, 0.0038234, -0.0017431, 0.0016337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062787, upper bound: 0.0062876
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062372
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9794893, 0.9893448, -0.0079136, 0.0100372
1: -0.0045223, -0.0038737, -0.0045506, -0.0039190, -0.0006034, 0.0006769
2: 0.0104745, 0.0139120, 0.0107144, 0.0140616, -0.0035872, 0.0031976
3: -0.0076963, -0.0060406, -0.0077947, -0.0061498, -0.0015465, 0.0017541
4: 0.0025552, 0.0036143, 0.0026016, 0.0037743, -0.0012191, 0.0010126
5: 0.0121336, 0.0205366, 0.0124353, 0.0220831, -0.0099496, 0.0081013
6: -0.0026362, -0.0015388, -0.0026839, -0.0016154, -0.0010208, 0.0011451
7: -0.0099582, -0.0071190, -0.0100817, -0.0073171, -0.0026410, 0.0029628
8: -0.0048010, -0.0027275, -0.0048660, -0.0021021, -0.0026990, 0.0021385
9: 0.0019719, 0.0037032, 0.0020927, 0.0037786, -0.0018067, 0.0016105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0063787
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0063219
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9814312, 0.9895265, 0.9783329, 0.9894307, -0.0079995, 0.0111936
1: -0.0045223, -0.0038737, -0.0045674, -0.0038975, -0.0006248, 0.0006937
2: 0.0104745, 0.0139120, 0.0106009, 0.0141507, -0.0036763, 0.0033111
3: -0.0076963, -0.0060406, -0.0078533, -0.0060982, -0.0015981, 0.0018127
4: 0.0025552, 0.0036143, 0.0025797, 0.0038696, -0.0013144, 0.0010346
5: 0.0121336, 0.0205366, 0.0122926, 0.0230040, -0.0108704, 0.0082440
6: -0.0026362, -0.0015388, -0.0027124, -0.0015792, -0.0010570, 0.0011736
7: -0.0099582, -0.0071190, -0.0101553, -0.0072234, -0.0027348, 0.0030364
8: -0.0048010, -0.0027275, -0.0049047, -0.0017297, -0.0030714, 0.0021772
9: 0.0019719, 0.0037032, 0.0020355, 0.0038234, -0.0018516, 0.0016677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0063787
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0063219
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9837303, 0.9892544, -0.0088502, 0.0054469
1: -0.0045373, -0.0039607, -0.0044889, -0.0039415, -0.0005958, 0.0005282
2: 0.0109359, 0.0139911, 0.0108338, 0.0137349, -0.0027990, 0.0031573
3: -0.0077483, -0.0062507, -0.0075798, -0.0062042, -0.0015442, 0.0013291
4: 0.0026445, 0.0036989, 0.0026247, 0.0034248, -0.0007803, 0.0010742
5: 0.0127139, 0.0213545, 0.0125855, 0.0187057, -0.0059918, 0.0087690
6: -0.0026614, -0.0016861, -0.0025796, -0.0016535, -0.0010079, 0.0008935
7: -0.0100235, -0.0075001, -0.0098119, -0.0074157, -0.0026078, 0.0023118
8: -0.0048354, -0.0023968, -0.0047241, -0.0034640, -0.0013714, 0.0023273
9: 0.0022043, 0.0037430, 0.0021528, 0.0036140, -0.0014097, 0.0015902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062461, upper bound: 0.0061971
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061920, upper bound: 0.0061972
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9837303, 0.9892544, -0.0097651, 0.0056146
1: -0.0045506, -0.0039190, -0.0044889, -0.0039415, -0.0006091, 0.0005700
2: 0.0107144, 0.0140616, 0.0108338, 0.0137349, -0.0030205, 0.0032278
3: -0.0077947, -0.0061498, -0.0075798, -0.0062042, -0.0015905, 0.0014300
4: 0.0026016, 0.0037743, 0.0026247, 0.0034248, -0.0008231, 0.0011496
5: 0.0124353, 0.0220831, 0.0125855, 0.0187057, -0.0062704, 0.0094976
6: -0.0026839, -0.0016154, -0.0025796, -0.0016535, -0.0010304, 0.0009642
7: -0.0100817, -0.0073171, -0.0098119, -0.0074157, -0.0026660, 0.0024948
8: -0.0048660, -0.0021021, -0.0047241, -0.0034640, -0.0014020, 0.0026220
9: 0.0020927, 0.0037786, 0.0021528, 0.0036140, -0.0015213, 0.0016257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062461, upper bound: 0.0062790
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061920, upper bound: 0.0062810
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9837303, 0.9892544, -0.0098110, 0.0055197
1: -0.0045512, -0.0039426, -0.0044889, -0.0039415, -0.0006098, 0.0005463
2: 0.0108396, 0.0140652, 0.0108338, 0.0137349, -0.0028953, 0.0032314
3: -0.0077970, -0.0062068, -0.0075798, -0.0062042, -0.0015929, 0.0013730
4: 0.0026259, 0.0037781, 0.0026247, 0.0034248, -0.0007989, 0.0011534
5: 0.0125928, 0.0221198, 0.0125855, 0.0187057, -0.0061129, 0.0095343
6: -0.0026850, -0.0016554, -0.0025796, -0.0016535, -0.0010315, 0.0009243
7: -0.0100847, -0.0074205, -0.0098119, -0.0074157, -0.0026689, 0.0023913
8: -0.0048676, -0.0020873, -0.0047241, -0.0034640, -0.0014036, 0.0026368
9: 0.0021558, 0.0037803, 0.0021528, 0.0036140, -0.0014582, 0.0016275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062229
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062241
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9837303, 0.9892544, -0.0109215, 0.0057005
1: -0.0045674, -0.0038975, -0.0044889, -0.0039415, -0.0006259, 0.0005914
2: 0.0106009, 0.0141507, 0.0108338, 0.0137349, -0.0031340, 0.0033169
3: -0.0078533, -0.0060982, -0.0075798, -0.0062042, -0.0016491, 0.0014816
4: 0.0025797, 0.0038696, 0.0026247, 0.0034248, -0.0008451, 0.0012449
5: 0.0122926, 0.0230040, 0.0125855, 0.0187057, -0.0064131, 0.0104185
6: -0.0027124, -0.0015792, -0.0025796, -0.0016535, -0.0010588, 0.0010005
7: -0.0101553, -0.0072234, -0.0098119, -0.0074157, -0.0027396, 0.0025885
8: -0.0049047, -0.0017297, -0.0047241, -0.0034640, -0.0014407, 0.0029945
9: 0.0020355, 0.0038234, 0.0021528, 0.0036140, -0.0015784, 0.0016706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9804042, 0.9891771, -0.0087729, 0.0087729
1: -0.0045373, -0.0039607, -0.0045373, -0.0039607, -0.0005765, 0.0005765
2: 0.0109359, 0.0139911, 0.0109359, 0.0139911, -0.0030553, 0.0030553
3: -0.0077483, -0.0062507, -0.0077483, -0.0062507, -0.0014977, 0.0014977
4: 0.0026445, 0.0036989, 0.0026445, 0.0036989, -0.0010544, 0.0010544
5: 0.0127139, 0.0213545, 0.0127139, 0.0213545, -0.0086406, 0.0086406
6: -0.0026614, -0.0016861, -0.0026614, -0.0016861, -0.0009753, 0.0009753
7: -0.0100235, -0.0075001, -0.0100235, -0.0075001, -0.0025235, 0.0025235
8: -0.0048354, -0.0023968, -0.0048354, -0.0023968, -0.0024387, 0.0024387
9: 0.0022043, 0.0037430, 0.0022043, 0.0037430, -0.0015388, 0.0015388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062302, upper bound: 0.0061971
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061775, upper bound: 0.0061972
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9804042, 0.9891771, -0.0096878, 0.0089406
1: -0.0045506, -0.0039190, -0.0045373, -0.0039607, -0.0005898, 0.0006183
2: 0.0107144, 0.0140616, 0.0109359, 0.0139911, -0.0032768, 0.0031258
3: -0.0077947, -0.0061498, -0.0077483, -0.0062507, -0.0015441, 0.0015985
4: 0.0026016, 0.0037743, 0.0026445, 0.0036989, -0.0010973, 0.0011298
5: 0.0124353, 0.0220831, 0.0127139, 0.0213545, -0.0089191, 0.0093693
6: -0.0026839, -0.0016154, -0.0026614, -0.0016861, -0.0009978, 0.0010460
7: -0.0100817, -0.0073171, -0.0100235, -0.0075001, -0.0025817, 0.0027064
8: -0.0048660, -0.0021021, -0.0048354, -0.0023968, -0.0024693, 0.0027333
9: 0.0020927, 0.0037786, 0.0022043, 0.0037430, -0.0016503, 0.0015743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061772, upper bound: 0.0063327
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061775, upper bound: 0.0062810
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9804042, 0.9891771, -0.0097338, 0.0088457
1: -0.0045512, -0.0039426, -0.0045373, -0.0039607, -0.0005905, 0.0005947
2: 0.0108396, 0.0140652, 0.0109359, 0.0139911, -0.0031516, 0.0031293
3: -0.0077970, -0.0062068, -0.0077483, -0.0062507, -0.0015464, 0.0015415
4: 0.0026259, 0.0037781, 0.0026445, 0.0036989, -0.0010730, 0.0011336
5: 0.0125928, 0.0221198, 0.0127139, 0.0213545, -0.0087617, 0.0094059
6: -0.0026850, -0.0016554, -0.0026614, -0.0016861, -0.0009990, 0.0010061
7: -0.0100847, -0.0074205, -0.0100235, -0.0075001, -0.0025846, 0.0026030
8: -0.0048676, -0.0020873, -0.0048354, -0.0023968, -0.0024708, 0.0027481
9: 0.0021558, 0.0037803, 0.0022043, 0.0037430, -0.0015873, 0.0015761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0062229
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062241
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9804042, 0.9891771, -0.0108442, 0.0090265
1: -0.0045674, -0.0038975, -0.0045373, -0.0039607, -0.0006066, 0.0006397
2: 0.0106009, 0.0141507, 0.0109359, 0.0139911, -0.0033902, 0.0032149
3: -0.0078533, -0.0060982, -0.0077483, -0.0062507, -0.0016027, 0.0016502
4: 0.0025797, 0.0038696, 0.0026445, 0.0036989, -0.0011192, 0.0012251
5: 0.0122926, 0.0230040, 0.0127139, 0.0213545, -0.0090619, 0.0102901
6: -0.0027124, -0.0015792, -0.0026614, -0.0016861, -0.0010263, 0.0010823
7: -0.0101553, -0.0072234, -0.0100235, -0.0075001, -0.0026553, 0.0028001
8: -0.0049047, -0.0017297, -0.0048354, -0.0023968, -0.0025080, 0.0031058
9: 0.0020355, 0.0038234, 0.0022043, 0.0037430, -0.0017075, 0.0016192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061765, upper bound: 0.0063642
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063092
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9828223, 0.9894229, -0.0090187, 0.0063548
1: -0.0045373, -0.0039607, -0.0045021, -0.0038995, -0.0006378, 0.0005414
2: 0.0109359, 0.0139911, 0.0106112, 0.0138048, -0.0028690, 0.0033799
3: -0.0077483, -0.0062507, -0.0076258, -0.0061029, -0.0016454, 0.0013752
4: 0.0026445, 0.0036989, 0.0025817, 0.0034996, -0.0008551, 0.0011172
5: 0.0127139, 0.0213545, 0.0123056, 0.0194288, -0.0067149, 0.0090489
6: -0.0026614, -0.0016861, -0.0026019, -0.0015825, -0.0010790, 0.0009159
7: -0.0100235, -0.0075001, -0.0098696, -0.0072319, -0.0027916, 0.0023696
8: -0.0048354, -0.0023968, -0.0047545, -0.0031755, -0.0016599, 0.0023577
9: 0.0022043, 0.0037430, 0.0020408, 0.0036492, -0.0014450, 0.0017023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063353, upper bound: 0.0061961
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062798, upper bound: 0.0061966
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9828223, 0.9894229, -0.0099795, 0.0064276
1: -0.0045512, -0.0039426, -0.0045021, -0.0038995, -0.0006517, 0.0005595
2: 0.0108396, 0.0140652, 0.0106112, 0.0138048, -0.0029653, 0.0034539
3: -0.0077970, -0.0062068, -0.0076258, -0.0061029, -0.0016941, 0.0014190
4: 0.0026259, 0.0037781, 0.0025817, 0.0034996, -0.0008738, 0.0011965
5: 0.0125928, 0.0221198, 0.0123056, 0.0194288, -0.0068360, 0.0098142
6: -0.0026850, -0.0016554, -0.0026019, -0.0015825, -0.0011026, 0.0009466
7: -0.0100847, -0.0074205, -0.0098696, -0.0072319, -0.0028527, 0.0024491
8: -0.0048676, -0.0020873, -0.0047545, -0.0031755, -0.0016920, 0.0026672
9: 0.0021558, 0.0037803, 0.0020408, 0.0036492, -0.0014935, 0.0017396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063353, upper bound: 0.0062229
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062798, upper bound: 0.0062241
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9828223, 0.9894229, -0.0099336, 0.0065225
1: -0.0045506, -0.0039190, -0.0045021, -0.0038995, -0.0006511, 0.0005832
2: 0.0107144, 0.0140616, 0.0106112, 0.0138048, -0.0030905, 0.0034504
3: -0.0077947, -0.0061498, -0.0076258, -0.0061029, -0.0016918, 0.0014760
4: 0.0026016, 0.0037743, 0.0025817, 0.0034996, -0.0008980, 0.0011927
5: 0.0124353, 0.0220831, 0.0123056, 0.0194288, -0.0069935, 0.0097775
6: -0.0026839, -0.0016154, -0.0026019, -0.0015825, -0.0011015, 0.0009866
7: -0.0100817, -0.0073171, -0.0098696, -0.0072319, -0.0028498, 0.0025525
8: -0.0048660, -0.0021021, -0.0047545, -0.0031755, -0.0016905, 0.0026524
9: 0.0020927, 0.0037786, 0.0020408, 0.0036492, -0.0015565, 0.0017378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062790
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062810
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9828223, 0.9894229, -0.0110900, 0.0066084
1: -0.0045674, -0.0038975, -0.0045021, -0.0038995, -0.0006679, 0.0006046
2: 0.0106009, 0.0141507, 0.0106112, 0.0138048, -0.0032039, 0.0035395
3: -0.0078533, -0.0060982, -0.0076258, -0.0061029, -0.0017504, 0.0015276
4: 0.0025797, 0.0038696, 0.0025817, 0.0034996, -0.0009200, 0.0012880
5: 0.0122926, 0.0230040, 0.0123056, 0.0194288, -0.0071362, 0.0106984
6: -0.0027124, -0.0015792, -0.0026019, -0.0015825, -0.0011299, 0.0010228
7: -0.0101553, -0.0072234, -0.0098696, -0.0072319, -0.0029234, 0.0026463
8: -0.0049047, -0.0017297, -0.0047545, -0.0031755, -0.0017292, 0.0030248
9: 0.0020355, 0.0038234, 0.0020408, 0.0036492, -0.0016137, 0.0017827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9794893, 0.9893448, -0.0089406, 0.0096878
1: -0.0045373, -0.0039607, -0.0045506, -0.0039190, -0.0006183, 0.0005898
2: 0.0109359, 0.0139911, 0.0107144, 0.0140616, -0.0031258, 0.0032768
3: -0.0077483, -0.0062507, -0.0077947, -0.0061498, -0.0015985, 0.0015441
4: 0.0026445, 0.0036989, 0.0026016, 0.0037743, -0.0011298, 0.0010973
5: 0.0127139, 0.0213545, 0.0124353, 0.0220831, -0.0093693, 0.0089191
6: -0.0026614, -0.0016861, -0.0026839, -0.0016154, -0.0010460, 0.0009978
7: -0.0100235, -0.0075001, -0.0100817, -0.0073171, -0.0027064, 0.0025817
8: -0.0048354, -0.0023968, -0.0048660, -0.0021021, -0.0027333, 0.0024693
9: 0.0022043, 0.0037430, 0.0020927, 0.0037786, -0.0015743, 0.0016503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0061961
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0061966
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9794893, 0.9893448, -0.0099015, 0.0097607
1: -0.0045512, -0.0039426, -0.0045506, -0.0039190, -0.0006323, 0.0006080
2: 0.0108396, 0.0140652, 0.0107144, 0.0140616, -0.0032221, 0.0033508
3: -0.0077970, -0.0062068, -0.0077947, -0.0061498, -0.0016472, 0.0015879
4: 0.0026259, 0.0037781, 0.0026016, 0.0037743, -0.0011485, 0.0011765
5: 0.0125928, 0.0221198, 0.0124353, 0.0220831, -0.0094904, 0.0096845
6: -0.0026850, -0.0016554, -0.0026839, -0.0016154, -0.0010697, 0.0010286
7: -0.0100847, -0.0074205, -0.0100817, -0.0073171, -0.0027675, 0.0026612
8: -0.0048676, -0.0020873, -0.0048660, -0.0021021, -0.0027655, 0.0027788
9: 0.0021558, 0.0037803, 0.0020927, 0.0037786, -0.0016228, 0.0016876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0062229
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0062241
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9794893, 0.9893448, -0.0098556, 0.0098556
1: -0.0045506, -0.0039190, -0.0045506, -0.0039190, -0.0006316, 0.0006316
2: 0.0107144, 0.0140616, 0.0107144, 0.0140616, -0.0033473, 0.0033473
3: -0.0077947, -0.0061498, -0.0077947, -0.0061498, -0.0016449, 0.0016449
4: 0.0026016, 0.0037743, 0.0026016, 0.0037743, -0.0011727, 0.0011727
5: 0.0124353, 0.0220831, 0.0124353, 0.0220831, -0.0096478, 0.0096478
6: -0.0026839, -0.0016154, -0.0026839, -0.0016154, -0.0010685, 0.0010685
7: -0.0100817, -0.0073171, -0.0100817, -0.0073171, -0.0027646, 0.0027646
8: -0.0048660, -0.0021021, -0.0048660, -0.0021021, -0.0027640, 0.0027640
9: 0.0020927, 0.0037786, 0.0020927, 0.0037786, -0.0016859, 0.0016859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0062790
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062810
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9794893, 0.9893448, -0.0110120, 0.0099415
1: -0.0045674, -0.0038975, -0.0045506, -0.0039190, -0.0006484, 0.0006530
2: 0.0106009, 0.0141507, 0.0107144, 0.0140616, -0.0034607, 0.0034363
3: -0.0078533, -0.0060982, -0.0077947, -0.0061498, -0.0017035, 0.0016965
4: 0.0025797, 0.0038696, 0.0026016, 0.0037743, -0.0011947, 0.0012680
5: 0.0122926, 0.0230040, 0.0124353, 0.0220831, -0.0097906, 0.0105687
6: -0.0027124, -0.0015792, -0.0026839, -0.0016154, -0.0010970, 0.0011048
7: -0.0101553, -0.0072234, -0.0100817, -0.0073171, -0.0028382, 0.0028584
8: -0.0049047, -0.0017297, -0.0048660, -0.0021021, -0.0028027, 0.0031364
9: 0.0020355, 0.0038234, 0.0020927, 0.0037786, -0.0017430, 0.0017307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0063041
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063091
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9823049, 0.9893634, -0.0089592, 0.0068722
1: -0.0045373, -0.0039607, -0.0045096, -0.0039143, -0.0006229, 0.0005489
2: 0.0109359, 0.0139911, 0.0106898, 0.0138447, -0.0029088, 0.0033013
3: -0.0077483, -0.0062507, -0.0076520, -0.0061387, -0.0016097, 0.0014014
4: 0.0026445, 0.0036989, 0.0025969, 0.0035423, -0.0008978, 0.0011020
5: 0.0127139, 0.0213545, 0.0124045, 0.0198408, -0.0071269, 0.0089500
6: -0.0026614, -0.0016861, -0.0026147, -0.0016076, -0.0010539, 0.0009286
7: -0.0100235, -0.0075001, -0.0099026, -0.0072969, -0.0027267, 0.0024025
8: -0.0048354, -0.0023968, -0.0047718, -0.0030089, -0.0018265, 0.0023750
9: 0.0022043, 0.0037430, 0.0020803, 0.0036693, -0.0014650, 0.0016627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0061965
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0061965
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9823049, 0.9893634, -0.0098742, 0.0070399
1: -0.0045506, -0.0039190, -0.0045096, -0.0039143, -0.0006362, 0.0005907
2: 0.0107144, 0.0140616, 0.0106898, 0.0138447, -0.0031303, 0.0033718
3: -0.0077947, -0.0061498, -0.0076520, -0.0061387, -0.0016560, 0.0015022
4: 0.0026016, 0.0037743, 0.0025969, 0.0035423, -0.0009406, 0.0011774
5: 0.0124353, 0.0220831, 0.0124045, 0.0198408, -0.0074055, 0.0096787
6: -0.0026839, -0.0016154, -0.0026147, -0.0016076, -0.0010764, 0.0009993
7: -0.0100817, -0.0073171, -0.0099026, -0.0072969, -0.0027849, 0.0025854
8: -0.0048660, -0.0021021, -0.0047718, -0.0030089, -0.0018571, 0.0026697
9: 0.0020927, 0.0037786, 0.0020803, 0.0036693, -0.0015766, 0.0016982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0062787
time: 0.94 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0062803
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9823049, 0.9893634, -0.0099201, 0.0069450
1: -0.0045512, -0.0039426, -0.0045096, -0.0039143, -0.0006369, 0.0005671
2: 0.0108396, 0.0140652, 0.0106898, 0.0138447, -0.0030051, 0.0033753
3: -0.0077970, -0.0062068, -0.0076520, -0.0061387, -0.0016584, 0.0014452
4: 0.0026259, 0.0037781, 0.0025969, 0.0035423, -0.0009164, 0.0011812
5: 0.0125928, 0.0221198, 0.0124045, 0.0198408, -0.0072480, 0.0097153
6: -0.0026850, -0.0016554, -0.0026147, -0.0016076, -0.0010775, 0.0009593
7: -0.0100847, -0.0074205, -0.0099026, -0.0072969, -0.0027878, 0.0024820
8: -0.0048676, -0.0020873, -0.0047718, -0.0030089, -0.0018587, 0.0026845
9: 0.0021558, 0.0037803, 0.0020803, 0.0036693, -0.0015135, 0.0017000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062229
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062241
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9823049, 0.9893634, -0.0110306, 0.0071258
1: -0.0045674, -0.0038975, -0.0045096, -0.0039143, -0.0006531, 0.0006121
2: 0.0106009, 0.0141507, 0.0106898, 0.0138447, -0.0032438, 0.0034609
3: -0.0078533, -0.0060982, -0.0076520, -0.0061387, -0.0017146, 0.0015538
4: 0.0025797, 0.0038696, 0.0025969, 0.0035423, -0.0009626, 0.0012728
5: 0.0122926, 0.0230040, 0.0124045, 0.0198408, -0.0075482, 0.0105995
6: -0.0027124, -0.0015792, -0.0026147, -0.0016076, -0.0011048, 0.0010355
7: -0.0101553, -0.0072234, -0.0099026, -0.0072969, -0.0028585, 0.0026792
8: -0.0049047, -0.0017297, -0.0047718, -0.0030089, -0.0018958, 0.0030421
9: 0.0020355, 0.0038234, 0.0020803, 0.0036693, -0.0016337, 0.0017431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9794433, 0.9892499, -0.0088457, 0.0097338
1: -0.0045373, -0.0039607, -0.0045512, -0.0039426, -0.0005947, 0.0005905
2: 0.0109359, 0.0139911, 0.0108396, 0.0140652, -0.0031293, 0.0031516
3: -0.0077483, -0.0062507, -0.0077970, -0.0062068, -0.0015415, 0.0015464
4: 0.0026445, 0.0036989, 0.0026259, 0.0037781, -0.0011336, 0.0010730
5: 0.0127139, 0.0213545, 0.0125928, 0.0221198, -0.0094059, 0.0087617
6: -0.0026614, -0.0016861, -0.0026850, -0.0016554, -0.0010061, 0.0009990
7: -0.0100235, -0.0075001, -0.0100847, -0.0074205, -0.0026030, 0.0025846
8: -0.0048354, -0.0023968, -0.0048676, -0.0020873, -0.0027481, 0.0024708
9: 0.0022043, 0.0037430, 0.0021558, 0.0037803, -0.0015761, 0.0015873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062113, upper bound: 0.0062431
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062132, upper bound: 0.0061965
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9794433, 0.9892499, -0.0097607, 0.0099015
1: -0.0045506, -0.0039190, -0.0045512, -0.0039426, -0.0006080, 0.0006323
2: 0.0107144, 0.0140616, 0.0108396, 0.0140652, -0.0033508, 0.0032221
3: -0.0077947, -0.0061498, -0.0077970, -0.0062068, -0.0015879, 0.0016472
4: 0.0026016, 0.0037743, 0.0026259, 0.0037781, -0.0011765, 0.0011485
5: 0.0124353, 0.0220831, 0.0125928, 0.0221198, -0.0096845, 0.0094904
6: -0.0026839, -0.0016154, -0.0026850, -0.0016554, -0.0010286, 0.0010697
7: -0.0100817, -0.0073171, -0.0100847, -0.0074205, -0.0026612, 0.0027675
8: -0.0048660, -0.0021021, -0.0048676, -0.0020873, -0.0027788, 0.0027655
9: 0.0020927, 0.0037786, 0.0021558, 0.0037803, -0.0016876, 0.0016228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062113, upper bound: 0.0063314
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062132, upper bound: 0.0062803
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9794433, 0.9892499, -0.0098066, 0.0098066
1: -0.0045512, -0.0039426, -0.0045512, -0.0039426, -0.0006087, 0.0006087
2: 0.0108396, 0.0140652, 0.0108396, 0.0140652, -0.0032256, 0.0032256
3: -0.0077970, -0.0062068, -0.0077970, -0.0062068, -0.0015902, 0.0015902
4: 0.0026259, 0.0037781, 0.0026259, 0.0037781, -0.0011523, 0.0011523
5: 0.0125928, 0.0221198, 0.0125928, 0.0221198, -0.0095270, 0.0095270
6: -0.0026850, -0.0016554, -0.0026850, -0.0016554, -0.0010297, 0.0010297
7: -0.0100847, -0.0074205, -0.0100847, -0.0074205, -0.0026641, 0.0026641
8: -0.0048676, -0.0020873, -0.0048676, -0.0020873, -0.0027803, 0.0027803
9: 0.0021558, 0.0037803, 0.0021558, 0.0037803, -0.0016246, 0.0016246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0062229
time: 1.09 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062241
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9794433, 0.9892499, -0.0109171, 0.0099874
1: -0.0045674, -0.0038975, -0.0045512, -0.0039426, -0.0006248, 0.0006537
2: 0.0106009, 0.0141507, 0.0108396, 0.0140652, -0.0034643, 0.0033111
3: -0.0078533, -0.0060982, -0.0077970, -0.0062068, -0.0016465, 0.0016989
4: 0.0025797, 0.0038696, 0.0026259, 0.0037781, -0.0011985, 0.0012438
5: 0.0122926, 0.0230040, 0.0125928, 0.0221198, -0.0098272, 0.0104112
6: -0.0027124, -0.0015792, -0.0026850, -0.0016554, -0.0010570, 0.0011059
7: -0.0101553, -0.0072234, -0.0100847, -0.0074205, -0.0027348, 0.0028613
8: -0.0049047, -0.0017297, -0.0048676, -0.0020873, -0.0028175, 0.0031379
9: 0.0020355, 0.0038234, 0.0021558, 0.0037803, -0.0017448, 0.0016677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061765, upper bound: 0.0063642
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063091
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9814312, 0.9895265, -0.0091223, 0.0077459
1: -0.0045373, -0.0039607, -0.0045223, -0.0038737, -0.0006636, 0.0005616
2: 0.0109359, 0.0139911, 0.0104745, 0.0139120, -0.0029761, 0.0035167
3: -0.0077483, -0.0062507, -0.0076963, -0.0060406, -0.0017077, 0.0014456
4: 0.0026445, 0.0036989, 0.0025552, 0.0036143, -0.0009698, 0.0011437
5: 0.0127139, 0.0213545, 0.0121336, 0.0205366, -0.0078227, 0.0092209
6: -0.0026614, -0.0016861, -0.0026362, -0.0015388, -0.0011226, 0.0009501
7: -0.0100235, -0.0075001, -0.0099582, -0.0071190, -0.0029045, 0.0024581
8: -0.0048354, -0.0023968, -0.0048010, -0.0027275, -0.0021079, 0.0024043
9: 0.0022043, 0.0037430, 0.0019719, 0.0037032, -0.0014989, 0.0017712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0061965
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0061965
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9814312, 0.9895265, -0.0100372, 0.0079136
1: -0.0045506, -0.0039190, -0.0045223, -0.0038737, -0.0006769, 0.0006034
2: 0.0107144, 0.0140616, 0.0104745, 0.0139120, -0.0031976, 0.0035872
3: -0.0077947, -0.0061498, -0.0076963, -0.0060406, -0.0017541, 0.0015465
4: 0.0026016, 0.0037743, 0.0025552, 0.0036143, -0.0010126, 0.0012191
5: 0.0124353, 0.0220831, 0.0121336, 0.0205366, -0.0081013, 0.0099496
6: -0.0026839, -0.0016154, -0.0026362, -0.0015388, -0.0011451, 0.0010208
7: -0.0100817, -0.0073171, -0.0099582, -0.0071190, -0.0029628, 0.0026410
8: -0.0048660, -0.0021021, -0.0048010, -0.0027275, -0.0021385, 0.0026990
9: 0.0020927, 0.0037786, 0.0019719, 0.0037032, -0.0016105, 0.0018067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0062787
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0062803
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9814312, 0.9895265, -0.0100831, 0.0078187
1: -0.0045512, -0.0039426, -0.0045223, -0.0038737, -0.0006776, 0.0005798
2: 0.0108396, 0.0140652, 0.0104745, 0.0139120, -0.0030724, 0.0035907
3: -0.0077970, -0.0062068, -0.0076963, -0.0060406, -0.0017564, 0.0014895
4: 0.0026259, 0.0037781, 0.0025552, 0.0036143, -0.0009884, 0.0012229
5: 0.0125928, 0.0221198, 0.0121336, 0.0205366, -0.0079438, 0.0099862
6: -0.0026850, -0.0016554, -0.0026362, -0.0015388, -0.0011463, 0.0009808
7: -0.0100847, -0.0074205, -0.0099582, -0.0071190, -0.0029657, 0.0025376
8: -0.0048676, -0.0020873, -0.0048010, -0.0027275, -0.0021401, 0.0027138
9: 0.0021558, 0.0037803, 0.0019719, 0.0037032, -0.0015474, 0.0018085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062229
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062241
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9814312, 0.9895265, -0.0111936, 0.0079995
1: -0.0045674, -0.0038975, -0.0045223, -0.0038737, -0.0006937, 0.0006248
2: 0.0106009, 0.0141507, 0.0104745, 0.0139120, -0.0033111, 0.0036763
3: -0.0078533, -0.0060982, -0.0076963, -0.0060406, -0.0018127, 0.0015981
4: 0.0025797, 0.0038696, 0.0025552, 0.0036143, -0.0010346, 0.0013144
5: 0.0122926, 0.0230040, 0.0121336, 0.0205366, -0.0082440, 0.0108704
6: -0.0027124, -0.0015792, -0.0026362, -0.0015388, -0.0011736, 0.0010570
7: -0.0101553, -0.0072234, -0.0099582, -0.0071190, -0.0030364, 0.0027348
8: -0.0049047, -0.0017297, -0.0048010, -0.0027275, -0.0021772, 0.0030714
9: 0.0020355, 0.0038234, 0.0019719, 0.0037032, -0.0016677, 0.0018516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9804042, 0.9891771, 0.9783329, 0.9894307, -0.0090265, 0.0108442
1: -0.0045373, -0.0039607, -0.0045674, -0.0038975, -0.0006397, 0.0006066
2: 0.0109359, 0.0139911, 0.0106009, 0.0141507, -0.0032149, 0.0033902
3: -0.0077483, -0.0062507, -0.0078533, -0.0060982, -0.0016502, 0.0016027
4: 0.0026445, 0.0036989, 0.0025797, 0.0038696, -0.0012251, 0.0011192
5: 0.0127139, 0.0213545, 0.0122926, 0.0230040, -0.0102901, 0.0090619
6: -0.0026614, -0.0016861, -0.0027124, -0.0015792, -0.0010823, 0.0010263
7: -0.0100235, -0.0075001, -0.0101553, -0.0072234, -0.0028001, 0.0026553
8: -0.0048354, -0.0023968, -0.0049047, -0.0017297, -0.0031058, 0.0025080
9: 0.0022043, 0.0037430, 0.0020355, 0.0038234, -0.0016192, 0.0017075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0061959
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0061962
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9794433, 0.9892499, 0.9783329, 0.9894307, -0.0099874, 0.0109171
1: -0.0045512, -0.0039426, -0.0045674, -0.0038975, -0.0006537, 0.0006248
2: 0.0108396, 0.0140652, 0.0106009, 0.0141507, -0.0033111, 0.0034643
3: -0.0077970, -0.0062068, -0.0078533, -0.0060982, -0.0016989, 0.0016465
4: 0.0026259, 0.0037781, 0.0025797, 0.0038696, -0.0012438, 0.0011985
5: 0.0125928, 0.0221198, 0.0122926, 0.0230040, -0.0104112, 0.0098272
6: -0.0026850, -0.0016554, -0.0027124, -0.0015792, -0.0011059, 0.0010570
7: -0.0100847, -0.0074205, -0.0101553, -0.0072234, -0.0028613, 0.0027348
8: -0.0048676, -0.0020873, -0.0049047, -0.0017297, -0.0031379, 0.0028175
9: 0.0021558, 0.0037803, 0.0020355, 0.0038234, -0.0016677, 0.0017448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0062229
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0062241
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9794893, 0.9893448, 0.9783329, 0.9894307, -0.0099415, 0.0110120
1: -0.0045506, -0.0039190, -0.0045674, -0.0038975, -0.0006530, 0.0006484
2: 0.0107144, 0.0140616, 0.0106009, 0.0141507, -0.0034363, 0.0034607
3: -0.0077947, -0.0061498, -0.0078533, -0.0060982, -0.0016965, 0.0017035
4: 0.0026016, 0.0037743, 0.0025797, 0.0038696, -0.0012680, 0.0011947
5: 0.0124353, 0.0220831, 0.0122926, 0.0230040, -0.0105687, 0.0097906
6: -0.0026839, -0.0016154, -0.0027124, -0.0015792, -0.0011048, 0.0010970
7: -0.0100817, -0.0073171, -0.0101553, -0.0072234, -0.0028584, 0.0028382
8: -0.0048660, -0.0021021, -0.0049047, -0.0017297, -0.0031364, 0.0028027
9: 0.0020927, 0.0037786, 0.0020355, 0.0038234, -0.0017307, 0.0017430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061765, upper bound: 0.0063314
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062803
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9783329, 0.9894307, 0.9783329, 0.9894307, -0.0110978, 0.0110978
1: -0.0045674, -0.0038975, -0.0045674, -0.0038975, -0.0006698, 0.0006698
2: 0.0106009, 0.0141507, 0.0106009, 0.0141507, -0.0035498, 0.0035498
3: -0.0078533, -0.0060982, -0.0078533, -0.0060982, -0.0017551, 0.0017551
4: 0.0025797, 0.0038696, 0.0025797, 0.0038696, -0.0012900, 0.0012900
5: 0.0122926, 0.0230040, 0.0122926, 0.0230040, -0.0107114, 0.0107114
6: -0.0027124, -0.0015792, -0.0027124, -0.0015792, -0.0011332, 0.0011332
7: -0.0101553, -0.0072234, -0.0101553, -0.0072234, -0.0029319, 0.0029319
8: -0.0049047, -0.0017297, -0.0049047, -0.0017297, -0.0031751, 0.0031751
9: 0.0020355, 0.0038234, 0.0020355, 0.0038234, -0.0017879, 0.0017879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0063041
time: 0.81 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063091
time: 0.89 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.06 seconds
IS_A1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062524, upper bound: 0.0061950
IS_A1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0061951
IS_A1_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063461
IS_A1_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062835
IS_A1_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0062507
IS_A1_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0061951
IS_A1_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0063439
IS_A1_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0062832
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063461, upper bound: 0.0061949
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062835, upper bound: 0.0061950
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063461, upper bound: 0.0061949
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062835, upper bound: 0.0061950
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062524, upper bound: 0.0062817
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062832
IS_A1_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063439
IS_A1_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062832
IS_A1_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0062420
IS_A1_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
IS_A1_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063872
IS_A1_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
IS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0062957
IS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
IS_A1_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063872
IS_A1_B1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
IS_A1_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0062419
IS_A1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
IS_A1_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0063216
IS_A1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
IS_A1_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0062419
IS_A1_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
IS_A1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062503, upper bound: 0.0063216
IS_A1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
IS_A1_B2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061971, upper bound: 0.0062461
IS_A1_B2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061972, upper bound: 0.0061920
IS_A1_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061971, upper bound: 0.0062461
IS_A1_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061972, upper bound: 0.0061920
IS_A1_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062229, upper bound: 0.0062434
IS_A1_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062241, upper bound: 0.0061914
IS_A1_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062229, upper bound: 0.0062434
IS_A1_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062241, upper bound: 0.0061914
IS_A1_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
IS_A1_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
IS_A1_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
IS_A1_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
IS_A1_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
IS_A1_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
IS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061961, upper bound: 0.0063353
IS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061966, upper bound: 0.0062798
IS_A1_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0062889
IS_A1_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0062372
IS_A1_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0063787
IS_A1_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061965, upper bound: 0.0063219
IS_A1_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0062889
IS_A1_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0062372
IS_A1_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0063787
IS_A1_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0063219
IS_A1_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062787, upper bound: 0.0062876
IS_A1_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062372
IS_A1_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062787, upper bound: 0.0062876
IS_A1_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062372
IS_A1_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0063787
IS_A1_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0063219
IS_A1_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061959, upper bound: 0.0063787
IS_A1_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061962, upper bound: 0.0063219
IS_A2_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062461, upper bound: 0.0061971
IS_A2_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061920, upper bound: 0.0061972
IS_A2_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062461, upper bound: 0.0062790
IS_A2_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061920, upper bound: 0.0062810
IS_A2_B1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062229
IS_A2_B1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062241
IS_A2_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
IS_A2_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
IS_A2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062302, upper bound: 0.0061971
IS_A2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061775, upper bound: 0.0061972
IS_A2_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061772, upper bound: 0.0063327
IS_A2_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061775, upper bound: 0.0062810
IS_A2_B1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0062229
IS_A2_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062241
IS_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061765, upper bound: 0.0063642
IS_A2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063092
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063353, upper bound: 0.0061961
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062798, upper bound: 0.0061966
IS_A2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063353, upper bound: 0.0062229
IS_A2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062798, upper bound: 0.0062241
IS_A2_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062790
IS_A2_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062810
IS_A2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
IS_A2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
IS_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0061961
IS_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0061966
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0062229
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0062241
IS_A2_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0062790
IS_A2_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062810
IS_A2_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0063041
IS_A2_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063091
IS_A2_B2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0061965
IS_A2_B2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0061965
IS_A2_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0062787
IS_A2_B2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0062803
IS_A2_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062229
IS_A2_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062241
IS_A2_B2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
IS_A2_B2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
IS_A2_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062113, upper bound: 0.0062431
IS_A2_B2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062132, upper bound: 0.0061965
IS_A2_B2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062113, upper bound: 0.0063314
IS_A2_B2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062132, upper bound: 0.0062803
IS_A2_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0062229
IS_A2_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062241
IS_A2_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061765, upper bound: 0.0063642
IS_A2_B2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063091
IS_A2_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0061965
IS_A2_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0061965
IS_A2_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062876, upper bound: 0.0062787
IS_A2_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062372, upper bound: 0.0062803
IS_A2_B2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0062229
IS_A2_B2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0062241
IS_A2_B2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062434, upper bound: 0.0063041
IS_A2_B2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061914, upper bound: 0.0063091
IS_A2_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0061959
IS_A2_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0061962
IS_A2_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0063210, upper bound: 0.0062229
IS_A2_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062665, upper bound: 0.0062241
IS_A2_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061765, upper bound: 0.0063314
IS_A2_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0062803
IS_A2_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0062275, upper bound: 0.0063041
IS_A2_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0061766, upper bound: 0.0063091

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9849420, 0.9892707, 0.9838706, 0.9892539, -0.0043120, 0.0054001
1: -0.0044713, -0.0039374, -0.0044869, -0.0039416, -0.0005297, 0.0005495
2: 0.0108122, 0.0136415, 0.0108344, 0.0137241, -0.0029119, 0.0028072
3: -0.0075184, -0.0061943, -0.0075727, -0.0062045, -0.0013139, 0.0013783
4: 0.0026206, 0.0033249, 0.0026249, 0.0034132, -0.0007927, 0.0007001
5: 0.0125583, 0.0177408, 0.0125862, 0.0185940, -0.0060357, 0.0051546
6: -0.0025498, -0.0016466, -0.0025762, -0.0016537, -0.0008961, 0.0009296
7: -0.0097348, -0.0073979, -0.0098029, -0.0074162, -0.0023185, 0.0024051
8: -0.0046836, -0.0034546, -0.0047194, -0.0034643, -0.0012193, 0.0012648
9: 0.0021420, 0.0035670, 0.0021531, 0.0036085, -0.0014666, 0.0014138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0061950
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0061950
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9852858, 0.9892513, 0.9838991, 0.9892540, -0.0039682, 0.0053523
1: -0.0044663, -0.0039422, -0.0044865, -0.0039416, -0.0005247, 0.0005442
2: 0.0108378, 0.0136150, 0.0108342, 0.0137219, -0.0028841, 0.0027808
3: -0.0075010, -0.0062060, -0.0075712, -0.0062044, -0.0012966, 0.0013652
4: 0.0026255, 0.0032966, 0.0026248, 0.0034109, -0.0007854, 0.0006717
5: 0.0125905, 0.0174670, 0.0125861, 0.0185713, -0.0059807, 0.0048809
6: -0.0025414, -0.0016548, -0.0025755, -0.0016536, -0.0008877, 0.0009207
7: -0.0097129, -0.0074190, -0.0098011, -0.0074161, -0.0022968, 0.0023821
8: -0.0046721, -0.0034657, -0.0047185, -0.0034642, -0.0012079, 0.0012527
9: 0.0021549, 0.0035536, 0.0021531, 0.0036074, -0.0014526, 0.0014006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0061951
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0061951
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9849420, 0.9892707, -0.0063177, 0.0044805
1: -0.0045002, -0.0038996, -0.0044713, -0.0039374, -0.0005628, 0.0005717
2: 0.0106118, 0.0137948, 0.0108122, 0.0136415, -0.0030297, 0.0029826
3: -0.0076192, -0.0061032, -0.0075184, -0.0061943, -0.0014248, 0.0014152
4: 0.0025818, 0.0034888, 0.0026206, 0.0033249, -0.0007431, 0.0008683
5: 0.0123064, 0.0193247, 0.0125583, 0.0177408, -0.0054345, 0.0067664
6: -0.0025987, -0.0015827, -0.0025498, -0.0016466, -0.0009521, 0.0009672
7: -0.0098613, -0.0072324, -0.0097348, -0.0073979, -0.0024635, 0.0025023
8: -0.0047501, -0.0032176, -0.0046836, -0.0034546, -0.0012955, 0.0014659
9: 0.0020411, 0.0036441, 0.0021420, 0.0035670, -0.0015259, 0.0015022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062818
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062835
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9852858, 0.9892513, -0.0062510, 0.0041367
1: -0.0044995, -0.0038996, -0.0044663, -0.0039422, -0.0005573, 0.0005667
2: 0.0106117, 0.0137911, 0.0108378, 0.0136150, -0.0030033, 0.0029533
3: -0.0076168, -0.0061031, -0.0075010, -0.0062060, -0.0014108, 0.0013979
4: 0.0025818, 0.0034849, 0.0026255, 0.0032966, -0.0007148, 0.0008594
5: 0.0123062, 0.0192870, 0.0125905, 0.0174670, -0.0051608, 0.0066965
6: -0.0025976, -0.0015826, -0.0025414, -0.0016548, -0.0009428, 0.0009587
7: -0.0098583, -0.0072323, -0.0097129, -0.0074190, -0.0024393, 0.0024806
8: -0.0047485, -0.0032329, -0.0046721, -0.0034657, -0.0012828, 0.0014392
9: 0.0020410, 0.0036423, 0.0021549, 0.0035536, -0.0015126, 0.0014875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062818
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062835
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9838706, 0.9892539, 0.9834547, 0.9893827, -0.0055121, 0.0057992
1: -0.0044869, -0.0039416, -0.0044929, -0.0039095, -0.0005774, 0.0005513
2: 0.0108344, 0.0137241, 0.0106642, 0.0137561, -0.0029217, 0.0030598
3: -0.0075727, -0.0062045, -0.0075938, -0.0061270, -0.0014457, 0.0013893
4: 0.0026249, 0.0034132, 0.0025919, 0.0034475, -0.0008226, 0.0008213
5: 0.0125862, 0.0185940, 0.0123722, 0.0189251, -0.0063389, 0.0062218
6: -0.0025762, -0.0016537, -0.0025864, -0.0015994, -0.0009768, 0.0009327
7: -0.0098029, -0.0074162, -0.0098294, -0.0072757, -0.0025272, 0.0024132
8: -0.0047194, -0.0034643, -0.0047333, -0.0033792, -0.0013402, 0.0012691
9: 0.0021531, 0.0036085, 0.0020674, 0.0036247, -0.0014715, 0.0015411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062420, upper bound: 0.0061950
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062420, upper bound: 0.0061951
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9838991, 0.9892540, 0.9838901, 0.9893602, -0.0054612, 0.0053639
1: -0.0044865, -0.0039416, -0.0044866, -0.0039151, -0.0005713, 0.0005450
2: 0.0108342, 0.0137219, 0.0106941, 0.0137226, -0.0028884, 0.0030278
3: -0.0075712, -0.0062044, -0.0075717, -0.0061406, -0.0014306, 0.0013673
4: 0.0026248, 0.0034109, 0.0025977, 0.0034116, -0.0007868, 0.0008132
5: 0.0125861, 0.0185713, 0.0124098, 0.0185785, -0.0059925, 0.0061615
6: -0.0025755, -0.0016536, -0.0025757, -0.0016089, -0.0009666, 0.0009220
7: -0.0098011, -0.0074161, -0.0098017, -0.0073003, -0.0025008, 0.0023856
8: -0.0047185, -0.0034642, -0.0047188, -0.0034033, -0.0013151, 0.0012546
9: 0.0021531, 0.0036074, 0.0020825, 0.0036078, -0.0014547, 0.0015250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0061950
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0061951
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9834547, 0.9893827, -0.0064297, 0.0059677
1: -0.0045002, -0.0038996, -0.0044929, -0.0039095, -0.0005907, 0.0005933
2: 0.0106118, 0.0137948, 0.0106642, 0.0137561, -0.0031443, 0.0031305
3: -0.0076192, -0.0061032, -0.0075938, -0.0061270, -0.0014922, 0.0014906
4: 0.0025818, 0.0034888, 0.0025919, 0.0034475, -0.0008657, 0.0008969
5: 0.0123064, 0.0193247, 0.0123722, 0.0189251, -0.0066188, 0.0069524
6: -0.0025987, -0.0015827, -0.0025864, -0.0015994, -0.0009994, 0.0010037
7: -0.0098613, -0.0072324, -0.0098294, -0.0072757, -0.0025856, 0.0025970
8: -0.0047501, -0.0032176, -0.0047333, -0.0033792, -0.0013709, 0.0015157
9: 0.0020411, 0.0036441, 0.0020674, 0.0036247, -0.0015836, 0.0015767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0062817
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0062832
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9838901, 0.9893602, -0.0063599, 0.0055324
1: -0.0044995, -0.0038996, -0.0044866, -0.0039151, -0.0005844, 0.0005870
2: 0.0106117, 0.0137911, 0.0106941, 0.0137226, -0.0031109, 0.0030971
3: -0.0076168, -0.0061031, -0.0075717, -0.0061406, -0.0014762, 0.0014686
4: 0.0025818, 0.0034849, 0.0025977, 0.0034116, -0.0008299, 0.0008872
5: 0.0123062, 0.0192870, 0.0124098, 0.0185785, -0.0062724, 0.0068773
6: -0.0025976, -0.0015826, -0.0025757, -0.0016089, -0.0009887, 0.0009931
7: -0.0098583, -0.0072323, -0.0098017, -0.0073003, -0.0025580, 0.0025694
8: -0.0047485, -0.0032329, -0.0047188, -0.0034033, -0.0013452, 0.0014859
9: 0.0020410, 0.0036423, 0.0020825, 0.0036078, -0.0015668, 0.0015598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0062817
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0062832
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9849420, 0.9892707, 0.9829531, 0.9894224, -0.0044805, 0.0063177
1: -0.0044713, -0.0039374, -0.0045002, -0.0038996, -0.0005717, 0.0005628
2: 0.0108122, 0.0136415, 0.0106118, 0.0137948, -0.0029826, 0.0030297
3: -0.0075184, -0.0061943, -0.0076192, -0.0061032, -0.0014152, 0.0014248
4: 0.0026206, 0.0033249, 0.0025818, 0.0034888, -0.0008683, 0.0007431
5: 0.0125583, 0.0177408, 0.0123064, 0.0193247, -0.0067664, 0.0054345
6: -0.0025498, -0.0016466, -0.0025987, -0.0015827, -0.0009672, 0.0009521
7: -0.0097348, -0.0073979, -0.0098613, -0.0072324, -0.0025023, 0.0024635
8: -0.0046836, -0.0034546, -0.0047501, -0.0032176, -0.0014659, 0.0012955
9: 0.0021420, 0.0035670, 0.0020411, 0.0036441, -0.0015022, 0.0015259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062818, upper bound: 0.0061950
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062818, upper bound: 0.0061950
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9852858, 0.9892513, 0.9830003, 0.9894225, -0.0041367, 0.0062510
1: -0.0044663, -0.0039422, -0.0044995, -0.0038996, -0.0005667, 0.0005573
2: 0.0108378, 0.0136150, 0.0106117, 0.0137911, -0.0029533, 0.0030033
3: -0.0075010, -0.0062060, -0.0076168, -0.0061031, -0.0013979, 0.0014108
4: 0.0026255, 0.0032966, 0.0025818, 0.0034849, -0.0008594, 0.0007148
5: 0.0125905, 0.0174670, 0.0123062, 0.0192870, -0.0066965, 0.0051608
6: -0.0025414, -0.0016548, -0.0025976, -0.0015826, -0.0009587, 0.0009428
7: -0.0097129, -0.0074190, -0.0098583, -0.0072323, -0.0024806, 0.0024393
8: -0.0046721, -0.0034657, -0.0047485, -0.0032329, -0.0014392, 0.0012828
9: 0.0021549, 0.0035536, 0.0020410, 0.0036423, -0.0014875, 0.0015126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062818, upper bound: 0.0061950
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062818, upper bound: 0.0061950
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9849420, 0.9892707, 0.9815614, 0.9895260, -0.0045841, 0.0077093
1: -0.0044713, -0.0039374, -0.0045204, -0.0038738, -0.0005975, 0.0005830
2: 0.0108122, 0.0136415, 0.0104751, 0.0139020, -0.0030898, 0.0031665
3: -0.0075184, -0.0061943, -0.0076897, -0.0060409, -0.0014775, 0.0014954
4: 0.0026206, 0.0033249, 0.0025553, 0.0036035, -0.0009830, 0.0007696
5: 0.0125583, 0.0177408, 0.0121343, 0.0204330, -0.0078747, 0.0056065
6: -0.0025498, -0.0016466, -0.0026330, -0.0015390, -0.0010108, 0.0009864
7: -0.0097348, -0.0073979, -0.0099499, -0.0071195, -0.0026153, 0.0025520
8: -0.0046836, -0.0034546, -0.0047967, -0.0027694, -0.0019141, 0.0013421
9: 0.0021420, 0.0035670, 0.0019722, 0.0036981, -0.0015562, 0.0015948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063216, upper bound: 0.0061949
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063216, upper bound: 0.0061949
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9852858, 0.9892513, 0.9816096, 0.9895262, -0.0042403, 0.0076417
1: -0.0044663, -0.0039422, -0.0045197, -0.0038738, -0.0005925, 0.0005775
2: 0.0108378, 0.0136150, 0.0104749, 0.0138983, -0.0030605, 0.0031401
3: -0.0075010, -0.0062060, -0.0076873, -0.0060408, -0.0014601, 0.0014812
4: 0.0026255, 0.0032966, 0.0025553, 0.0035996, -0.0009741, 0.0007413
5: 0.0125905, 0.0174670, 0.0121341, 0.0203945, -0.0078040, 0.0053328
6: -0.0025414, -0.0016548, -0.0026318, -0.0015389, -0.0010024, 0.0009770
7: -0.0097129, -0.0074190, -0.0099468, -0.0071193, -0.0025936, 0.0025278
8: -0.0046721, -0.0034657, -0.0047951, -0.0027850, -0.0018871, 0.0013293
9: 0.0021549, 0.0035536, 0.0019721, 0.0036963, -0.0015414, 0.0015815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063216, upper bound: 0.0061950
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063216, upper bound: 0.0061950
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9839562, 0.9894458, 0.9829531, 0.9894224, -0.0054662, 0.0064927
1: -0.0044856, -0.0038938, -0.0045002, -0.0038996, -0.0005860, 0.0006064
2: 0.0105810, 0.0137175, 0.0106118, 0.0137948, -0.0032138, 0.0031056
3: -0.0075683, -0.0060891, -0.0076192, -0.0061032, -0.0014652, 0.0015301
4: 0.0025758, 0.0034062, 0.0025818, 0.0034888, -0.0009130, 0.0008244
5: 0.0122675, 0.0185258, 0.0123064, 0.0193247, -0.0070572, 0.0062195
6: -0.0025741, -0.0015728, -0.0025987, -0.0015827, -0.0009914, 0.0010259
7: -0.0097975, -0.0072069, -0.0098613, -0.0072324, -0.0025651, 0.0026544
8: -0.0047166, -0.0033542, -0.0047501, -0.0032176, -0.0014989, 0.0013959
9: 0.0020255, 0.0036052, 0.0020411, 0.0036441, -0.0016186, 0.0015642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062818
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062818
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9844353, 0.9894198, 0.9830003, 0.9894225, -0.0049872, 0.0064195
1: -0.0044787, -0.0039003, -0.0044995, -0.0038996, -0.0005791, 0.0005993
2: 0.0106153, 0.0136806, 0.0106117, 0.0137911, -0.0031758, 0.0030689
3: -0.0075441, -0.0061048, -0.0076168, -0.0061031, -0.0014410, 0.0015120
4: 0.0025825, 0.0033667, 0.0025818, 0.0034849, -0.0009025, 0.0007849
5: 0.0123107, 0.0181444, 0.0123062, 0.0192870, -0.0069763, 0.0058382
6: -0.0025623, -0.0015838, -0.0025976, -0.0015826, -0.0009797, 0.0010138
7: -0.0097670, -0.0072353, -0.0098583, -0.0072323, -0.0025347, 0.0026230
8: -0.0047005, -0.0033691, -0.0047485, -0.0032329, -0.0014677, 0.0013794
9: 0.0020428, 0.0035866, 0.0020410, 0.0036423, -0.0015995, 0.0015456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062835
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062835
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9825497, 0.9895535, -0.0066004, 0.0068727
1: -0.0045002, -0.0038996, -0.0045061, -0.0038670, -0.0006332, 0.0006065
2: 0.0106118, 0.0137948, 0.0104389, 0.0138258, -0.0032140, 0.0033559
3: -0.0076192, -0.0061032, -0.0076396, -0.0060245, -0.0015947, 0.0015365
4: 0.0025818, 0.0034888, 0.0025483, 0.0035221, -0.0009403, 0.0009405
5: 0.0123064, 0.0193247, 0.0120888, 0.0196459, -0.0073396, 0.0072359
6: -0.0025987, -0.0015827, -0.0026086, -0.0015275, -0.0010713, 0.0010260
7: -0.0098613, -0.0072324, -0.0098870, -0.0070896, -0.0027717, 0.0026546
8: -0.0047501, -0.0032176, -0.0047636, -0.0030877, -0.0016624, 0.0015460
9: 0.0020411, 0.0036441, 0.0019540, 0.0036598, -0.0016187, 0.0016902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0062817
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062419, upper bound: 0.0062832
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9830657, 0.9895235, -0.0065231, 0.0063568
1: -0.0044995, -0.0038996, -0.0044986, -0.0038745, -0.0006251, 0.0005990
2: 0.0106117, 0.0137911, 0.0104785, 0.0137861, -0.0031744, 0.0033126
3: -0.0076168, -0.0061031, -0.0076135, -0.0060425, -0.0015743, 0.0015104
4: 0.0025818, 0.0034849, 0.0025560, 0.0034796, -0.0008978, 0.0009290
5: 0.0123062, 0.0192870, 0.0121387, 0.0192350, -0.0069288, 0.0071484
6: -0.0025976, -0.0015826, -0.0025960, -0.0015401, -0.0010575, 0.0010133
7: -0.0098583, -0.0072323, -0.0098542, -0.0071223, -0.0027360, 0.0026219
8: -0.0047485, -0.0032329, -0.0047464, -0.0032539, -0.0014946, 0.0015135
9: 0.0020410, 0.0036423, 0.0019739, 0.0036398, -0.0015988, 0.0016684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0062817
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062436, upper bound: 0.0062832
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9834547, 0.9893827, 0.9838706, 0.9892539, -0.0057992, 0.0055121
1: -0.0044929, -0.0039095, -0.0044869, -0.0039416, -0.0005513, 0.0005774
2: 0.0106642, 0.0137561, 0.0108344, 0.0137241, -0.0030598, 0.0029217
3: -0.0075938, -0.0061270, -0.0075727, -0.0062045, -0.0013893, 0.0014457
4: 0.0025919, 0.0034475, 0.0026249, 0.0034132, -0.0008213, 0.0008226
5: 0.0123722, 0.0189251, 0.0125862, 0.0185940, -0.0062218, 0.0063389
6: -0.0025864, -0.0015994, -0.0025762, -0.0016537, -0.0009327, 0.0009768
7: -0.0098294, -0.0072757, -0.0098029, -0.0074162, -0.0024132, 0.0025272
8: -0.0047333, -0.0033792, -0.0047194, -0.0034643, -0.0012691, 0.0013402
9: 0.0020674, 0.0036247, 0.0021531, 0.0036085, -0.0015411, 0.0014715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062420
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062420
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9838901, 0.9893602, 0.9838991, 0.9892540, -0.0053639, 0.0054612
1: -0.0044866, -0.0039151, -0.0044865, -0.0039416, -0.0005450, 0.0005713
2: 0.0106941, 0.0137226, 0.0108342, 0.0137219, -0.0030278, 0.0028884
3: -0.0075717, -0.0061406, -0.0075712, -0.0062044, -0.0013673, 0.0014306
4: 0.0025977, 0.0034116, 0.0026248, 0.0034109, -0.0008132, 0.0007868
5: 0.0124098, 0.0185785, 0.0125861, 0.0185713, -0.0061615, 0.0059925
6: -0.0025757, -0.0016089, -0.0025755, -0.0016536, -0.0009220, 0.0009666
7: -0.0098017, -0.0073003, -0.0098011, -0.0074161, -0.0023856, 0.0025008
8: -0.0047188, -0.0034033, -0.0047185, -0.0034642, -0.0012546, 0.0013151
9: 0.0020825, 0.0036078, 0.0021531, 0.0036074, -0.0015250, 0.0014547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9815614, 0.9895260, 0.9849420, 0.9892707, -0.0077093, 0.0045841
1: -0.0045204, -0.0038738, -0.0044713, -0.0039374, -0.0005830, 0.0005975
2: 0.0104751, 0.0139020, 0.0108122, 0.0136415, -0.0031665, 0.0030898
3: -0.0076897, -0.0060409, -0.0075184, -0.0061943, -0.0014954, 0.0014775
4: 0.0025553, 0.0036035, 0.0026206, 0.0033249, -0.0007696, 0.0009830
5: 0.0121343, 0.0204330, 0.0125583, 0.0177408, -0.0056065, 0.0078747
6: -0.0026330, -0.0015390, -0.0025498, -0.0016466, -0.0009864, 0.0010108
7: -0.0099499, -0.0071195, -0.0097348, -0.0073979, -0.0025520, 0.0026153
8: -0.0047967, -0.0027694, -0.0046836, -0.0034546, -0.0013421, 0.0019141
9: 0.0019722, 0.0036981, 0.0021420, 0.0035670, -0.0015948, 0.0015562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063216
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063253
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9816096, 0.9895262, 0.9852858, 0.9892513, -0.0076417, 0.0042403
1: -0.0045197, -0.0038738, -0.0044663, -0.0039422, -0.0005775, 0.0005925
2: 0.0104749, 0.0138983, 0.0108378, 0.0136150, -0.0031401, 0.0030605
3: -0.0076873, -0.0060408, -0.0075010, -0.0062060, -0.0014812, 0.0014601
4: 0.0025553, 0.0035996, 0.0026255, 0.0032966, -0.0007413, 0.0009741
5: 0.0121341, 0.0203945, 0.0125905, 0.0174670, -0.0053328, 0.0078040
6: -0.0026318, -0.0015389, -0.0025414, -0.0016548, -0.0009770, 0.0010024
7: -0.0099468, -0.0071193, -0.0097129, -0.0074190, -0.0025278, 0.0025936
8: -0.0047951, -0.0027850, -0.0046721, -0.0034657, -0.0013293, 0.0018871
9: 0.0019721, 0.0036963, 0.0021549, 0.0035536, -0.0015815, 0.0015414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063216
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9824418, 0.9893629, 0.9834547, 0.9893827, -0.0069410, 0.0059082
1: -0.0045076, -0.0039144, -0.0044929, -0.0039095, -0.0005982, 0.0005785
2: 0.0106905, 0.0138342, 0.0106642, 0.0137561, -0.0030656, 0.0031699
3: -0.0076451, -0.0061390, -0.0075938, -0.0061270, -0.0015181, 0.0014548
4: 0.0025970, 0.0035310, 0.0025919, 0.0034475, -0.0008505, 0.0009390
5: 0.0124053, 0.0197318, 0.0123722, 0.0189251, -0.0065199, 0.0073596
6: -0.0026113, -0.0016078, -0.0025864, -0.0015994, -0.0010119, 0.0009786
7: -0.0098939, -0.0072974, -0.0098294, -0.0072757, -0.0026182, 0.0025320
8: -0.0047672, -0.0030530, -0.0047333, -0.0033792, -0.0013880, 0.0016804
9: 0.0020807, 0.0036640, 0.0020674, 0.0036247, -0.0015440, 0.0015965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062420
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0062436
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9824769, 0.9893630, 0.9838901, 0.9893602, -0.0068833, 0.0054730
1: -0.0045071, -0.0039144, -0.0044866, -0.0039151, -0.0005920, 0.0005722
2: 0.0106903, 0.0138314, 0.0106941, 0.0137226, -0.0030323, 0.0031374
3: -0.0076433, -0.0061389, -0.0075717, -0.0061406, -0.0015027, 0.0014328
4: 0.0025970, 0.0035281, 0.0025977, 0.0034116, -0.0008147, 0.0009304
5: 0.0124050, 0.0197038, 0.0124098, 0.0185785, -0.0061735, 0.0072940
6: -0.0026104, -0.0016077, -0.0025757, -0.0016089, -0.0010015, 0.0009680
7: -0.0098916, -0.0072972, -0.0098017, -0.0073003, -0.0025913, 0.0025045
8: -0.0047661, -0.0030643, -0.0047188, -0.0034033, -0.0013627, 0.0016545
9: 0.0020806, 0.0036626, 0.0020825, 0.0036078, -0.0015272, 0.0015802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061951, upper bound: 0.0062420
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061951, upper bound: 0.0062436
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9815614, 0.9895260, 0.9834547, 0.9893827, -0.0078213, 0.0060713
1: -0.0045204, -0.0038738, -0.0044929, -0.0039095, -0.0006110, 0.0006191
2: 0.0104751, 0.0139020, 0.0106642, 0.0137561, -0.0032811, 0.0032378
3: -0.0076897, -0.0060409, -0.0075938, -0.0061270, -0.0015627, 0.0015529
4: 0.0025553, 0.0036035, 0.0025919, 0.0034475, -0.0008922, 0.0010116
5: 0.0121343, 0.0204330, 0.0123722, 0.0189251, -0.0067908, 0.0080607
6: -0.0026330, -0.0015390, -0.0025864, -0.0015994, -0.0010336, 0.0010474
7: -0.0099499, -0.0071195, -0.0098294, -0.0072757, -0.0026742, 0.0027099
8: -0.0047967, -0.0027694, -0.0047333, -0.0033792, -0.0014175, 0.0019639
9: 0.0019722, 0.0036981, 0.0020674, 0.0036247, -0.0016525, 0.0016307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063216
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063253
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9816096, 0.9895262, 0.9838901, 0.9893602, -0.0077506, 0.0056361
1: -0.0045197, -0.0038738, -0.0044866, -0.0039151, -0.0006046, 0.0006128
2: 0.0104749, 0.0138983, 0.0106941, 0.0137226, -0.0032477, 0.0032042
3: -0.0076873, -0.0060408, -0.0075717, -0.0061406, -0.0015467, 0.0015309
4: 0.0025553, 0.0035996, 0.0025977, 0.0034116, -0.0008563, 0.0010019
5: 0.0121341, 0.0203945, 0.0124098, 0.0185785, -0.0064444, 0.0079848
6: -0.0026318, -0.0015389, -0.0025757, -0.0016089, -0.0010229, 0.0010367
7: -0.0099468, -0.0071193, -0.0098017, -0.0073003, -0.0026465, 0.0026824
8: -0.0047951, -0.0027850, -0.0047188, -0.0034033, -0.0013918, 0.0019338
9: 0.0019721, 0.0036963, 0.0020825, 0.0036078, -0.0016357, 0.0016138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063216
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061950, upper bound: 0.0063253
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9834547, 0.9893827, 0.9829531, 0.9894224, -0.0059677, 0.0064297
1: -0.0044929, -0.0039095, -0.0045002, -0.0038996, -0.0005933, 0.0005907
2: 0.0106642, 0.0137561, 0.0106118, 0.0137948, -0.0031305, 0.0031443
3: -0.0075938, -0.0061270, -0.0076192, -0.0061032, -0.0014906, 0.0014922
4: 0.0025919, 0.0034475, 0.0025818, 0.0034888, -0.0008969, 0.0008657
5: 0.0123722, 0.0189251, 0.0123064, 0.0193247, -0.0069524, 0.0066188
6: -0.0025864, -0.0015994, -0.0025987, -0.0015827, -0.0010037, 0.0009994
7: -0.0098294, -0.0072757, -0.0098613, -0.0072324, -0.0025970, 0.0025856
8: -0.0047333, -0.0033792, -0.0047501, -0.0032176, -0.0015157, 0.0013709
9: 0.0020674, 0.0036247, 0.0020411, 0.0036441, -0.0015767, 0.0015836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062419
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062419
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9838901, 0.9893602, 0.9830003, 0.9894225, -0.0055324, 0.0063599
1: -0.0044866, -0.0039151, -0.0044995, -0.0038996, -0.0005870, 0.0005844
2: 0.0106941, 0.0137226, 0.0106117, 0.0137911, -0.0030971, 0.0031109
3: -0.0075717, -0.0061406, -0.0076168, -0.0061031, -0.0014686, 0.0014762
4: 0.0025977, 0.0034116, 0.0025818, 0.0034849, -0.0008872, 0.0008299
5: 0.0124098, 0.0185785, 0.0123062, 0.0192870, -0.0068773, 0.0062724
6: -0.0025757, -0.0016089, -0.0025976, -0.0015826, -0.0009931, 0.0009887
7: -0.0098017, -0.0073003, -0.0098583, -0.0072323, -0.0025694, 0.0025580
8: -0.0047188, -0.0034033, -0.0047485, -0.0032329, -0.0014859, 0.0013452
9: 0.0020825, 0.0036078, 0.0020410, 0.0036423, -0.0015598, 0.0015668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062436
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062436
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9825497, 0.9895535, 0.9829531, 0.9894224, -0.0068727, 0.0066004
1: -0.0045061, -0.0038670, -0.0045002, -0.0038996, -0.0006065, 0.0006332
2: 0.0104389, 0.0138258, 0.0106118, 0.0137948, -0.0033559, 0.0032140
3: -0.0076396, -0.0060245, -0.0076192, -0.0061032, -0.0015365, 0.0015947
4: 0.0025483, 0.0035221, 0.0025818, 0.0034888, -0.0009405, 0.0009403
5: 0.0120888, 0.0196459, 0.0123064, 0.0193247, -0.0072359, 0.0073396
6: -0.0026086, -0.0015275, -0.0025987, -0.0015827, -0.0010260, 0.0010713
7: -0.0098870, -0.0070896, -0.0098613, -0.0072324, -0.0026546, 0.0027717
8: -0.0047636, -0.0030877, -0.0047501, -0.0032176, -0.0015460, 0.0016624
9: 0.0019540, 0.0036598, 0.0020411, 0.0036441, -0.0016902, 0.0016187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063216
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063216
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9830657, 0.9895235, 0.9830003, 0.9894225, -0.0063568, 0.0065231
1: -0.0044986, -0.0038745, -0.0044995, -0.0038996, -0.0005990, 0.0006251
2: 0.0104785, 0.0137861, 0.0106117, 0.0137911, -0.0033126, 0.0031744
3: -0.0076135, -0.0060425, -0.0076168, -0.0061031, -0.0015104, 0.0015743
4: 0.0025560, 0.0034796, 0.0025818, 0.0034849, -0.0009290, 0.0008978
5: 0.0121387, 0.0192350, 0.0123062, 0.0192870, -0.0071484, 0.0069288
6: -0.0025960, -0.0015401, -0.0025976, -0.0015826, -0.0010133, 0.0010575
7: -0.0098542, -0.0071223, -0.0098583, -0.0072323, -0.0026219, 0.0027360
8: -0.0047464, -0.0032539, -0.0047485, -0.0032329, -0.0015135, 0.0014946
9: 0.0019739, 0.0036398, 0.0020410, 0.0036423, -0.0016684, 0.0015988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063253
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063253
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9834547, 0.9893827, 0.9815614, 0.9895260, -0.0060713, 0.0078213
1: -0.0044929, -0.0039095, -0.0045204, -0.0038738, -0.0006191, 0.0006110
2: 0.0106642, 0.0137561, 0.0104751, 0.0139020, -0.0032378, 0.0032811
3: -0.0075938, -0.0061270, -0.0076897, -0.0060409, -0.0015529, 0.0015627
4: 0.0025919, 0.0034475, 0.0025553, 0.0036035, -0.0010116, 0.0008922
5: 0.0123722, 0.0189251, 0.0121343, 0.0204330, -0.0080607, 0.0067908
6: -0.0025864, -0.0015994, -0.0026330, -0.0015390, -0.0010474, 0.0010336
7: -0.0098294, -0.0072757, -0.0099499, -0.0071195, -0.0027099, 0.0026742
8: -0.0047333, -0.0033792, -0.0047967, -0.0027694, -0.0019639, 0.0014175
9: 0.0020674, 0.0036247, 0.0019722, 0.0036981, -0.0016307, 0.0016525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062419
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062419
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9838901, 0.9893602, 0.9816096, 0.9895262, -0.0056361, 0.0077506
1: -0.0044866, -0.0039151, -0.0045197, -0.0038738, -0.0006128, 0.0006046
2: 0.0106941, 0.0137226, 0.0104749, 0.0138983, -0.0032042, 0.0032477
3: -0.0075717, -0.0061406, -0.0076873, -0.0060408, -0.0015309, 0.0015467
4: 0.0025977, 0.0034116, 0.0025553, 0.0035996, -0.0010019, 0.0008563
5: 0.0124098, 0.0185785, 0.0121341, 0.0203945, -0.0079848, 0.0064444
6: -0.0025757, -0.0016089, -0.0026318, -0.0015389, -0.0010367, 0.0010229
7: -0.0098017, -0.0073003, -0.0099468, -0.0071193, -0.0026824, 0.0026465
8: -0.0047188, -0.0034033, -0.0047951, -0.0027850, -0.0019338, 0.0013918
9: 0.0020825, 0.0036078, 0.0019721, 0.0036963, -0.0016138, 0.0016357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062436
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062817, upper bound: 0.0062436
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9825497, 0.9895535, 0.9815614, 0.9895260, -0.0069763, 0.0079920
1: -0.0045061, -0.0038670, -0.0045204, -0.0038738, -0.0006323, 0.0006535
2: 0.0104389, 0.0138258, 0.0104751, 0.0139020, -0.0034631, 0.0033508
3: -0.0076396, -0.0060245, -0.0076897, -0.0060409, -0.0015987, 0.0016652
4: 0.0025483, 0.0035221, 0.0025553, 0.0036035, -0.0010552, 0.0009668
5: 0.0120888, 0.0196459, 0.0121343, 0.0204330, -0.0083441, 0.0075116
6: -0.0026086, -0.0015275, -0.0026330, -0.0015390, -0.0010697, 0.0011055
7: -0.0098870, -0.0070896, -0.0099499, -0.0071195, -0.0027675, 0.0028603
8: -0.0047636, -0.0030877, -0.0047967, -0.0027694, -0.0019942, 0.0017090
9: 0.0019540, 0.0036598, 0.0019722, 0.0036981, -0.0017442, 0.0016876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063216
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063216
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9830657, 0.9895235, 0.9816096, 0.9895262, -0.0064604, 0.0079138
1: -0.0044986, -0.0038745, -0.0045197, -0.0038738, -0.0006248, 0.0006453
2: 0.0104785, 0.0137861, 0.0104749, 0.0138983, -0.0034198, 0.0033112
3: -0.0076135, -0.0060425, -0.0076873, -0.0060408, -0.0015726, 0.0016448
4: 0.0025560, 0.0034796, 0.0025553, 0.0035996, -0.0010436, 0.0009243
5: 0.0121387, 0.0192350, 0.0121341, 0.0203945, -0.0082559, 0.0071008
6: -0.0025960, -0.0015401, -0.0026318, -0.0015389, -0.0010570, 0.0010917
7: -0.0098542, -0.0071223, -0.0099468, -0.0071193, -0.0027348, 0.0028245
8: -0.0047464, -0.0032539, -0.0047951, -0.0027850, -0.0019614, 0.0015412
9: 0.0019739, 0.0036398, 0.0019721, 0.0036963, -0.0017224, 0.0016677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063253
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061949, upper bound: 0.0063253
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9838706, 0.9892539, 0.9816198, 0.9891887, -0.0053180, 0.0076341
1: -0.0044869, -0.0039416, -0.0045196, -0.0039579, -0.0005290, 0.0005780
2: 0.0108344, 0.0137241, 0.0109206, 0.0138975, -0.0030631, 0.0028035
3: -0.0075727, -0.0062045, -0.0076867, -0.0062437, -0.0013290, 0.0014823
4: 0.0026249, 0.0034132, 0.0026415, 0.0035987, -0.0009739, 0.0007717
5: 0.0125862, 0.0185940, 0.0126946, 0.0203864, -0.0078002, 0.0058994
6: -0.0025762, -0.0016537, -0.0026315, -0.0016812, -0.0008950, 0.0009778
7: -0.0098029, -0.0074162, -0.0099462, -0.0074874, -0.0023155, 0.0025299
8: -0.0047194, -0.0034643, -0.0047947, -0.0027882, -0.0019312, 0.0013305
9: 0.0021531, 0.0036085, 0.0021965, 0.0036959, -0.0015427, 0.0014120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059435, upper bound: 0.0055862
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061222, upper bound: 0.0061700
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9838991, 0.9892540, 0.9819033, 0.9891739, -0.0052749, 0.0073507
1: -0.0044865, -0.0039416, -0.0045155, -0.0039615, -0.0005249, 0.0005739
2: 0.0108342, 0.0137219, 0.0109399, 0.0138756, -0.0030414, 0.0027819
3: -0.0075712, -0.0062044, -0.0076724, -0.0062525, -0.0013187, 0.0014680
4: 0.0026248, 0.0034109, 0.0026453, 0.0035754, -0.0009505, 0.0007656
5: 0.0125861, 0.0185713, 0.0127190, 0.0201606, -0.0075745, 0.0058523
6: -0.0025755, -0.0016536, -0.0026245, -0.0016874, -0.0008881, 0.0009709
7: -0.0098011, -0.0074161, -0.0099281, -0.0075034, -0.0022977, 0.0025120
8: -0.0047185, -0.0034642, -0.0047852, -0.0028796, -0.0018389, 0.0013210
9: 0.0021531, 0.0036074, 0.0022063, 0.0036849, -0.0015318, 0.0014011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059435, upper bound: 0.0054674
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061224, upper bound: 0.0061184
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9838706, 0.9892539, 0.9806578, 0.9893626, -0.0054920, 0.0085961
1: -0.0044869, -0.0039416, -0.0045336, -0.0039145, -0.0005724, 0.0005920
2: 0.0108344, 0.0137241, 0.0106908, 0.0139716, -0.0031372, 0.0030332
3: -0.0075727, -0.0062045, -0.0077355, -0.0061391, -0.0014336, 0.0015310
4: 0.0026249, 0.0034132, 0.0025971, 0.0036780, -0.0010532, 0.0008161
5: 0.0125862, 0.0185940, 0.0124057, 0.0211525, -0.0085663, 0.0061883
6: -0.0025762, -0.0016537, -0.0026552, -0.0016079, -0.0009683, 0.0010015
7: -0.0098029, -0.0074162, -0.0100074, -0.0072977, -0.0025053, 0.0025911
8: -0.0047194, -0.0034643, -0.0048269, -0.0024784, -0.0022410, 0.0013627
9: 0.0021531, 0.0036085, 0.0020808, 0.0037332, -0.0015801, 0.0015277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0060504, upper bound: 0.0055862
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062049, upper bound: 0.0061694
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9838991, 0.9892540, 0.9810504, 0.9893418, -0.0054427, 0.0082036
1: -0.0044865, -0.0039416, -0.0045279, -0.0039197, -0.0005668, 0.0005863
2: 0.0108342, 0.0137219, 0.0107184, 0.0139413, -0.0031071, 0.0030035
3: -0.0075712, -0.0062044, -0.0077156, -0.0061516, -0.0014196, 0.0015112
4: 0.0026248, 0.0034109, 0.0026024, 0.0036457, -0.0010208, 0.0008085
5: 0.0125861, 0.0185713, 0.0124403, 0.0208398, -0.0082538, 0.0061310
6: -0.0025755, -0.0016536, -0.0026455, -0.0016167, -0.0009588, 0.0009919
7: -0.0098011, -0.0074161, -0.0099824, -0.0073204, -0.0024807, 0.0025663
8: -0.0047185, -0.0034642, -0.0048138, -0.0026049, -0.0021136, 0.0013496
9: 0.0021531, 0.0036074, 0.0020947, 0.0037180, -0.0015649, 0.0015127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062810, upper bound: 0.0061920
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062810, upper bound: 0.0061920
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9838706, 0.9892539, 0.9805492, 0.9892747, -0.0054041, 0.0087047
1: -0.0044869, -0.0039416, -0.0045352, -0.0039364, -0.0005504, 0.0005936
2: 0.0108344, 0.0137241, 0.0108070, 0.0139800, -0.0031456, 0.0029171
3: -0.0075727, -0.0062045, -0.0077410, -0.0061920, -0.0013807, 0.0015365
4: 0.0026249, 0.0034132, 0.0026196, 0.0036870, -0.0010621, 0.0007937
5: 0.0125862, 0.0185940, 0.0125519, 0.0212391, -0.0086528, 0.0060422
6: -0.0025762, -0.0016537, -0.0026579, -0.0016450, -0.0009312, 0.0010042
7: -0.0098029, -0.0074162, -0.0100143, -0.0073936, -0.0024093, 0.0025981
8: -0.0047194, -0.0034643, -0.0048306, -0.0024434, -0.0022760, 0.0013663
9: 0.0021531, 0.0036085, 0.0021394, 0.0037374, -0.0015843, 0.0014692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059858, upper bound: 0.0055857
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061476, upper bound: 0.0061667
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9838991, 0.9892540, 0.9810654, 0.9892468, -0.0053478, 0.0081886
1: -0.0044865, -0.0039416, -0.0045277, -0.0039434, -0.0005431, 0.0005861
2: 0.0108342, 0.0137219, 0.0108437, 0.0139402, -0.0031060, 0.0028782
3: -0.0075712, -0.0062044, -0.0077148, -0.0062087, -0.0013625, 0.0015104
4: 0.0026248, 0.0034109, 0.0026267, 0.0036444, -0.0010196, 0.0007842
5: 0.0125861, 0.0185713, 0.0125980, 0.0208280, -0.0082419, 0.0059733
6: -0.0025755, -0.0016536, -0.0026452, -0.0016567, -0.0009188, 0.0009915
7: -0.0098011, -0.0074161, -0.0099814, -0.0074239, -0.0023772, 0.0025653
8: -0.0047185, -0.0034642, -0.0048133, -0.0026097, -0.0021088, 0.0013491
9: 0.0021531, 0.0036074, 0.0021578, 0.0037174, -0.0015643, 0.0014496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059858, upper bound: 0.0054674
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061488, upper bound: 0.0061170
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9838706, 0.9892539, 0.9794966, 0.9894539, -0.0055833, 0.0097573
1: -0.0044869, -0.0039416, -0.0045505, -0.0038918, -0.0005951, 0.0006089
2: 0.0108344, 0.0137241, 0.0105703, 0.0140611, -0.0032267, 0.0031538
3: -0.0075727, -0.0062045, -0.0077943, -0.0060843, -0.0014884, 0.0015899
4: 0.0026249, 0.0034132, 0.0025737, 0.0037737, -0.0011489, 0.0008395
5: 0.0125862, 0.0185940, 0.0122541, 0.0220772, -0.0094910, 0.0063399
6: -0.0025762, -0.0016537, -0.0026837, -0.0015694, -0.0010068, 0.0010300
7: -0.0098029, -0.0074162, -0.0100813, -0.0071981, -0.0026048, 0.0026650
8: -0.0047194, -0.0034643, -0.0048658, -0.0021045, -0.0026149, 0.0014015
9: 0.0021531, 0.0036085, 0.0020201, 0.0037783, -0.0016251, 0.0015884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0060896, upper bound: 0.0055857
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062294, upper bound: 0.0061664
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9838991, 0.9892540, 0.9799438, 0.9894277, -0.0055286, 0.0093102
1: -0.0044865, -0.0039416, -0.0045440, -0.0038983, -0.0005882, 0.0006024
2: 0.0108342, 0.0137219, 0.0106049, 0.0140266, -0.0031924, 0.0031170
3: -0.0075712, -0.0062044, -0.0077717, -0.0061000, -0.0014712, 0.0015673
4: 0.0026248, 0.0034109, 0.0025804, 0.0037369, -0.0011120, 0.0008304
5: 0.0125861, 0.0185713, 0.0122976, 0.0217212, -0.0091351, 0.0062737
6: -0.0025755, -0.0016536, -0.0026727, -0.0015804, -0.0009950, 0.0010191
7: -0.0098011, -0.0074161, -0.0100528, -0.0072267, -0.0025744, 0.0026367
8: -0.0047185, -0.0034642, -0.0048508, -0.0022485, -0.0024700, 0.0013866
9: 0.0021531, 0.0036074, 0.0020376, 0.0037609, -0.0016079, 0.0015699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063091, upper bound: 0.0061914
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063091, upper bound: 0.0061914
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9816198, 0.9891887, -0.0062356, 0.0078026
1: -0.0045002, -0.0038996, -0.0045196, -0.0039579, -0.0005424, 0.0006200
2: 0.0106118, 0.0137948, 0.0109206, 0.0138975, -0.0032856, 0.0028742
3: -0.0076192, -0.0061032, -0.0076867, -0.0062437, -0.0013755, 0.0015836
4: 0.0025818, 0.0034888, 0.0026415, 0.0035987, -0.0010169, 0.0008473
5: 0.0123064, 0.0193247, 0.0126946, 0.0203864, -0.0080801, 0.0066301
6: -0.0025987, -0.0015827, -0.0026315, -0.0016812, -0.0009175, 0.0010489
7: -0.0098613, -0.0072324, -0.0099462, -0.0074874, -0.0023739, 0.0027137
8: -0.0047501, -0.0032176, -0.0047947, -0.0027882, -0.0019619, 0.0015771
9: 0.0020411, 0.0036441, 0.0021965, 0.0036959, -0.0016548, 0.0014476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059435, upper bound: 0.0056715
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061222, upper bound: 0.0062621
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9819033, 0.9891739, -0.0061736, 0.0075192
1: -0.0044995, -0.0038996, -0.0045155, -0.0039615, -0.0005380, 0.0006159
2: 0.0106117, 0.0137911, 0.0109399, 0.0138756, -0.0032639, 0.0028512
3: -0.0076168, -0.0061031, -0.0076724, -0.0062525, -0.0013643, 0.0015693
4: 0.0025818, 0.0034849, 0.0026453, 0.0035754, -0.0009936, 0.0008397
5: 0.0123062, 0.0192870, 0.0127190, 0.0201606, -0.0078544, 0.0065680
6: -0.0025976, -0.0015826, -0.0026245, -0.0016874, -0.0009102, 0.0010419
7: -0.0098583, -0.0072323, -0.0099281, -0.0075034, -0.0023549, 0.0026958
8: -0.0047485, -0.0032329, -0.0047852, -0.0028796, -0.0018690, 0.0015524
9: 0.0020410, 0.0036423, 0.0022063, 0.0036849, -0.0016439, 0.0014360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059435, upper bound: 0.0055484
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061224, upper bound: 0.0062065
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9805492, 0.9892747, -0.0063216, 0.0088732
1: -0.0045002, -0.0038996, -0.0045352, -0.0039364, -0.0005638, 0.0006356
2: 0.0106118, 0.0137948, 0.0108070, 0.0139800, -0.0033681, 0.0029877
3: -0.0076192, -0.0061032, -0.0077410, -0.0061920, -0.0014272, 0.0016378
4: 0.0025818, 0.0034888, 0.0026196, 0.0036870, -0.0011052, 0.0008693
5: 0.0123064, 0.0193247, 0.0125519, 0.0212391, -0.0089327, 0.0067728
6: -0.0025987, -0.0015827, -0.0026579, -0.0016450, -0.0009538, 0.0010752
7: -0.0098613, -0.0072324, -0.0100143, -0.0073936, -0.0024677, 0.0027819
8: -0.0047501, -0.0032176, -0.0048306, -0.0024434, -0.0023067, 0.0016129
9: 0.0020411, 0.0036441, 0.0021394, 0.0037374, -0.0016964, 0.0015048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059858, upper bound: 0.0056711
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061476, upper bound: 0.0062578
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9810654, 0.9892468, -0.0062465, 0.0083571
1: -0.0044995, -0.0038996, -0.0045277, -0.0039434, -0.0005562, 0.0006281
2: 0.0106117, 0.0137911, 0.0108437, 0.0139402, -0.0033285, 0.0029474
3: -0.0076168, -0.0061031, -0.0077148, -0.0062087, -0.0014081, 0.0016117
4: 0.0025818, 0.0034849, 0.0026267, 0.0036444, -0.0010627, 0.0008583
5: 0.0123062, 0.0192870, 0.0125980, 0.0208280, -0.0085218, 0.0066891
6: -0.0025976, -0.0015826, -0.0026452, -0.0016567, -0.0009409, 0.0010625
7: -0.0098583, -0.0072323, -0.0099814, -0.0074239, -0.0024344, 0.0027491
8: -0.0047485, -0.0032329, -0.0048133, -0.0026097, -0.0021389, 0.0015804
9: 0.0020410, 0.0036423, 0.0021578, 0.0037174, -0.0016764, 0.0014845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059858, upper bound: 0.0055484
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061488, upper bound: 0.0062051
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9806578, 0.9893626, -0.0064095, 0.0087646
1: -0.0045002, -0.0038996, -0.0045336, -0.0039145, -0.0005857, 0.0006340
2: 0.0106118, 0.0137948, 0.0106908, 0.0139716, -0.0033598, 0.0031039
3: -0.0076192, -0.0061032, -0.0077355, -0.0061391, -0.0014801, 0.0016323
4: 0.0025818, 0.0034888, 0.0025971, 0.0036780, -0.0010962, 0.0008918
5: 0.0123064, 0.0193247, 0.0124057, 0.0211525, -0.0088462, 0.0069190
6: -0.0025987, -0.0015827, -0.0026552, -0.0016079, -0.0009909, 0.0010725
7: -0.0098613, -0.0072324, -0.0100074, -0.0072977, -0.0025637, 0.0027750
8: -0.0047501, -0.0032176, -0.0048269, -0.0024784, -0.0022717, 0.0016093
9: 0.0020411, 0.0036441, 0.0020808, 0.0037332, -0.0016921, 0.0015633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059435, upper bound: 0.0056715
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061222, upper bound: 0.0062621
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9810504, 0.9893418, -0.0063415, 0.0083721
1: -0.0044995, -0.0038996, -0.0045279, -0.0039197, -0.0005798, 0.0006283
2: 0.0106117, 0.0137911, 0.0107184, 0.0139413, -0.0033297, 0.0030728
3: -0.0076168, -0.0061031, -0.0077156, -0.0061516, -0.0014651, 0.0016125
4: 0.0025818, 0.0034849, 0.0026024, 0.0036457, -0.0010639, 0.0008825
5: 0.0123062, 0.0192870, 0.0124403, 0.0208398, -0.0085337, 0.0068467
6: -0.0025976, -0.0015826, -0.0026455, -0.0016167, -0.0009809, 0.0010629
7: -0.0098583, -0.0072323, -0.0099824, -0.0073204, -0.0025379, 0.0027501
8: -0.0047485, -0.0032329, -0.0048138, -0.0026049, -0.0021437, 0.0015809
9: 0.0020410, 0.0036423, 0.0020947, 0.0037180, -0.0016770, 0.0015476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059435, upper bound: 0.0055484
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061224, upper bound: 0.0062065
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9829531, 0.9894224, 0.9794966, 0.9894539, -0.0065008, 0.0099258
1: -0.0045002, -0.0038996, -0.0045505, -0.0038918, -0.0006084, 0.0006509
2: 0.0106118, 0.0137948, 0.0105703, 0.0140611, -0.0034492, 0.0032245
3: -0.0076192, -0.0061032, -0.0077943, -0.0060843, -0.0015349, 0.0016912
4: 0.0025818, 0.0034888, 0.0025737, 0.0037737, -0.0011919, 0.0009151
5: 0.0123064, 0.0193247, 0.0122541, 0.0220772, -0.0097709, 0.0070706
6: -0.0025987, -0.0015827, -0.0026837, -0.0015694, -0.0010293, 0.0011011
7: -0.0098613, -0.0072324, -0.0100813, -0.0071981, -0.0026632, 0.0028488
8: -0.0047501, -0.0032176, -0.0048658, -0.0021045, -0.0026457, 0.0016482
9: 0.0020411, 0.0036441, 0.0020201, 0.0037783, -0.0017372, 0.0016240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059858, upper bound: 0.0056711
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061476, upper bound: 0.0062578
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9830003, 0.9894225, 0.9799438, 0.9894277, -0.0064273, 0.0094787
1: -0.0044995, -0.0038996, -0.0045440, -0.0038983, -0.0006012, 0.0006444
2: 0.0106117, 0.0137911, 0.0106049, 0.0140266, -0.0034149, 0.0031862
3: -0.0076168, -0.0061031, -0.0077717, -0.0061000, -0.0015168, 0.0016686
4: 0.0025818, 0.0034849, 0.0025804, 0.0037369, -0.0011551, 0.0009045
5: 0.0123062, 0.0192870, 0.0122976, 0.0217212, -0.0094150, 0.0069894
6: -0.0025976, -0.0015826, -0.0026727, -0.0015804, -0.0010171, 0.0010901
7: -0.0098583, -0.0072323, -0.0100528, -0.0072267, -0.0026316, 0.0028205
8: -0.0047485, -0.0032329, -0.0048508, -0.0022485, -0.0025001, 0.0016180
9: 0.0020410, 0.0036423, 0.0020376, 0.0037609, -0.0017199, 0.0016047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059858, upper bound: 0.0055484
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061488, upper bound: 0.0062051
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9824418, 0.9893629, 0.9816198, 0.9891887, -0.0067469, 0.0077431
1: -0.0045076, -0.0039144, -0.0045196, -0.0039579, -0.0005498, 0.0006051
2: 0.0106905, 0.0138342, 0.0109206, 0.0138975, -0.0032070, 0.0029136
3: -0.0076451, -0.0061390, -0.0076867, -0.0062437, -0.0014014, 0.0015478
4: 0.0025970, 0.0035310, 0.0026415, 0.0035987, -0.0010017, 0.0008894
5: 0.0124053, 0.0197318, 0.0126946, 0.0203864, -0.0079812, 0.0070372
6: -0.0026113, -0.0016078, -0.0026315, -0.0016812, -0.0009301, 0.0010238
7: -0.0098939, -0.0072974, -0.0099462, -0.0074874, -0.0024065, 0.0026488
8: -0.0047672, -0.0030530, -0.0047947, -0.0027882, -0.0019790, 0.0017418
9: 0.0020807, 0.0036640, 0.0021965, 0.0036959, -0.0016152, 0.0014674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0056229
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061208, upper bound: 0.0062104
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9824769, 0.9893630, 0.9819033, 0.9891739, -0.0066970, 0.0074597
1: -0.0045071, -0.0039144, -0.0045155, -0.0039615, -0.0005456, 0.0006011
2: 0.0106903, 0.0138314, 0.0109399, 0.0138756, -0.0031853, 0.0028915
3: -0.0076433, -0.0061389, -0.0076724, -0.0062525, -0.0013908, 0.0015335
4: 0.0025970, 0.0035281, 0.0026453, 0.0035754, -0.0009784, 0.0008828
5: 0.0124050, 0.0197038, 0.0127190, 0.0201606, -0.0077556, 0.0069848
6: -0.0026104, -0.0016077, -0.0026245, -0.0016874, -0.0009230, 0.0010168
7: -0.0098916, -0.0072972, -0.0099281, -0.0075034, -0.0023882, 0.0026309
8: -0.0047661, -0.0030643, -0.0047852, -0.0028796, -0.0018865, 0.0017209
9: 0.0020806, 0.0036626, 0.0022063, 0.0036849, -0.0016043, 0.0014563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0055064
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061209, upper bound: 0.0061627
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9815614, 0.9895260, 0.9816198, 0.9891887, -0.0076272, 0.0079062
1: -0.0045204, -0.0038738, -0.0045196, -0.0039579, -0.0005626, 0.0006458
2: 0.0104751, 0.0139020, 0.0109206, 0.0138975, -0.0034224, 0.0029814
3: -0.0076897, -0.0060409, -0.0076867, -0.0062437, -0.0014460, 0.0016458
4: 0.0025553, 0.0036035, 0.0026415, 0.0035987, -0.0010434, 0.0009620
5: 0.0121343, 0.0204330, 0.0126946, 0.0203864, -0.0082521, 0.0077383
6: -0.0026330, -0.0015390, -0.0026315, -0.0016812, -0.0009517, 0.0010925
7: -0.0099499, -0.0071195, -0.0099462, -0.0074874, -0.0024625, 0.0028267
8: -0.0047967, -0.0027694, -0.0047947, -0.0027882, -0.0020085, 0.0020253
9: 0.0019722, 0.0036981, 0.0021965, 0.0036959, -0.0017237, 0.0015016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0057158
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061208, upper bound: 0.0063011
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9816096, 0.9895262, 0.9819033, 0.9891739, -0.0075643, 0.0076228
1: -0.0045197, -0.0038738, -0.0045155, -0.0039615, -0.0005582, 0.0006417
2: 0.0104749, 0.0138983, 0.0109399, 0.0138756, -0.0034007, 0.0029583
3: -0.0076873, -0.0060408, -0.0076724, -0.0062525, -0.0014348, 0.0016315
4: 0.0025553, 0.0035996, 0.0026453, 0.0035754, -0.0010201, 0.0009543
5: 0.0121341, 0.0203945, 0.0127190, 0.0201606, -0.0080265, 0.0076755
6: -0.0026318, -0.0015389, -0.0026245, -0.0016874, -0.0009444, 0.0010856
7: -0.0099468, -0.0071193, -0.0099281, -0.0075034, -0.0024434, 0.0028088
8: -0.0047951, -0.0027850, -0.0047852, -0.0028796, -0.0019155, 0.0020003
9: 0.0019721, 0.0036963, 0.0022063, 0.0036849, -0.0017128, 0.0014900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0055899
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061209, upper bound: 0.0062477
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9824418, 0.9893629, 0.9805492, 0.9892747, -0.0068329, 0.0088137
1: -0.0045076, -0.0039144, -0.0045352, -0.0039364, -0.0005712, 0.0006207
2: 0.0106905, 0.0138342, 0.0108070, 0.0139800, -0.0032895, 0.0030271
3: -0.0076451, -0.0061390, -0.0077410, -0.0061920, -0.0014531, 0.0016020
4: 0.0025970, 0.0035310, 0.0026196, 0.0036870, -0.0010900, 0.0009114
5: 0.0124053, 0.0197318, 0.0125519, 0.0212391, -0.0088338, 0.0071800
6: -0.0026113, -0.0016078, -0.0026579, -0.0016450, -0.0009663, 0.0010501
7: -0.0098939, -0.0072974, -0.0100143, -0.0073936, -0.0025002, 0.0027169
8: -0.0047672, -0.0030530, -0.0048306, -0.0024434, -0.0023238, 0.0017776
9: 0.0020807, 0.0036640, 0.0021394, 0.0037374, -0.0016568, 0.0015246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0056229
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061201, upper bound: 0.0062104
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9824769, 0.9893630, 0.9810654, 0.9892468, -0.0067699, 0.0082976
1: -0.0045071, -0.0039144, -0.0045277, -0.0039434, -0.0005638, 0.0006132
2: 0.0106903, 0.0138314, 0.0108437, 0.0139402, -0.0032499, 0.0029878
3: -0.0076433, -0.0061389, -0.0077148, -0.0062087, -0.0014346, 0.0015760
4: 0.0025970, 0.0035281, 0.0026267, 0.0036444, -0.0010475, 0.0009014
5: 0.0124050, 0.0197038, 0.0125980, 0.0208280, -0.0084229, 0.0071058
6: -0.0026104, -0.0016077, -0.0026452, -0.0016567, -0.0009538, 0.0010375
7: -0.0098916, -0.0072972, -0.0099814, -0.0074239, -0.0024677, 0.0026842
8: -0.0047661, -0.0030643, -0.0048133, -0.0026097, -0.0021564, 0.0017490
9: 0.0020806, 0.0036626, 0.0021578, 0.0037174, -0.0016368, 0.0015048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0055064
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0061204, upper bound: 0.0061627
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9815614, 0.9895260, 0.9805492, 0.9892747, -0.0077133, 0.0089768
1: -0.0045204, -0.0038738, -0.0045352, -0.0039364, -0.0005840, 0.0006614
2: 0.0104751, 0.0139020, 0.0108070, 0.0139800, -0.0035049, 0.0030950
3: -0.0076897, -0.0060409, -0.0077410, -0.0061920, -0.0014977, 0.0017001
4: 0.0025553, 0.0036035, 0.0026196, 0.0036870, -0.0011317, 0.0009840
5: 0.0121343, 0.0204330, 0.0125519, 0.0212391, -0.0091047, 0.0078811
6: -0.0026330, -0.0015390, -0.0026579, -0.0016450, -0.0009880, 0.0011189
7: -0.0099499, -0.0071195, -0.0100143, -0.0073936, -0.0025562, 0.0028948
8: -0.0047967, -0.0027694, -0.0048306, -0.0024434, -0.0023533, 0.0020611
9: 0.0019722, 0.0036981, 0.0021394, 0.0037374, -0.0017653, 0.0015588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0057158
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061201, upper bound: 0.0063011
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9816096, 0.9895262, 0.9810654, 0.9892468, -0.0076372, 0.0084608
1: -0.0045197, -0.0038738, -0.0045277, -0.0039434, -0.0005764, 0.0006539
2: 0.0104749, 0.0138983, 0.0108437, 0.0139402, -0.0034653, 0.0030546
3: -0.0076873, -0.0060408, -0.0077148, -0.0062087, -0.0014786, 0.0016740
4: 0.0025553, 0.0035996, 0.0026267, 0.0036444, -0.0010891, 0.0009729
5: 0.0121341, 0.0203945, 0.0125980, 0.0208280, -0.0086939, 0.0077966
6: -0.0026318, -0.0015389, -0.0026452, -0.0016567, -0.0009751, 0.0011062
7: -0.0099468, -0.0071193, -0.0099814, -0.0074239, -0.0025229, 0.0028621
8: -0.0047951, -0.0027850, -0.0048133, -0.0026097, -0.0021854, 0.0020283
9: 0.0019721, 0.0036963, 0.0021578, 0.0037174, -0.0017453, 0.0015384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0059432, upper bound: 0.0055899
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0061204, upper bound: 0.0062477
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9824418, 0.9893629, 0.9806578, 0.9893626, -0.0069208, 0.0087051
1: -0.0045076, -0.0039144, -0.0045336, -0.0039145, -0.0005931, 0.0006191
2: 0.0106905, 0.0138342, 0.0106908, 0.0139716, -0.0032811, 0.0031433
3: -0.0076451, -0.0061390, -0.0077355, -0.0061391, -0.0015060, 0.0015965
4: 0.0025970, 0.0035310, 0.0025971, 0.0036780, -0.0010810, 0.0009339
5: 0.0124053, 0.0197318, 0.0124057, 0.0211525, -0.0087473, 0.0073261
6: -0.0026113, -0.0016078, -0.0026552, -0.0016079, -0.0010034, 0.0010474
7: -0.0098939, -0.0072974, -0.0100074, -0.0072977, -0.0025962, 0.0027100
8: -0.0047672, -0.0030530, -0.0048269, -0.0024784, -0.0022888, 0.0017740
9: 0.0020807, 0.0036640, 0.0020808, 0.0037332, -0.0016525, 0.0015831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0060498, upper bound: 0.0056229
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062044, upper bound: 0.0062086
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9824769, 0.9893630, 0.9810504, 0.9893418, -0.0068648, 0.0083126
1: -0.0045071, -0.0039144, -0.0045279, -0.0039197, -0.0005874, 0.0006135
2: 0.0106903, 0.0138314, 0.0107184, 0.0139413, -0.0032510, 0.0031131
3: -0.0076433, -0.0061389, -0.0077156, -0.0061516, -0.0014917, 0.0015767
4: 0.0025970, 0.0035281, 0.0026024, 0.0036457, -0.0010487, 0.0009257
5: 0.0124050, 0.0197038, 0.0124403, 0.0208398, -0.0084348, 0.0072635
6: -0.0026104, -0.0016077, -0.0026455, -0.0016167, -0.0009938, 0.0010378
7: -0.0098916, -0.0072972, -0.0099824, -0.0073204, -0.0025712, 0.0026852
8: -0.0047661, -0.0030643, -0.0048138, -0.0026049, -0.0021612, 0.0017495
9: 0.0020806, 0.0036626, 0.0020947, 0.0037180, -0.0016374, 0.0015679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062367
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062372
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9824418, 0.9893629, 0.9794966, 0.9894539, -0.0070121, 0.0098663
1: -0.0045076, -0.0039144, -0.0045505, -0.0038918, -0.0006159, 0.0006360
2: 0.0106905, 0.0138342, 0.0105703, 0.0140611, -0.0033706, 0.0032638
3: -0.0076451, -0.0061390, -0.0077943, -0.0060843, -0.0015608, 0.0016554
4: 0.0025970, 0.0035310, 0.0025737, 0.0037737, -0.0011767, 0.0009572
5: 0.0124053, 0.0197318, 0.0122541, 0.0220772, -0.0096720, 0.0074777
6: -0.0026113, -0.0016078, -0.0026837, -0.0015694, -0.0010419, 0.0010760
7: -0.0098939, -0.0072974, -0.0100813, -0.0071981, -0.0026957, 0.0027839
8: -0.0047672, -0.0030530, -0.0048658, -0.0021045, -0.0026628, 0.0018128
9: 0.0020807, 0.0036640, 0.0020201, 0.0037783, -0.0016976, 0.0016438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0060498, upper bound: 0.0056229
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062044, upper bound: 0.0062086
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9824769, 0.9893630, 0.9799438, 0.9894277, -0.0069507, 0.0094193
1: -0.0045071, -0.0039144, -0.0045440, -0.0038983, -0.0006088, 0.0006296
2: 0.0106903, 0.0138314, 0.0106049, 0.0140266, -0.0033363, 0.0032265
3: -0.0076433, -0.0061389, -0.0077717, -0.0061000, -0.0015433, 0.0016328
4: 0.0025970, 0.0035281, 0.0025804, 0.0037369, -0.0011399, 0.0009476
5: 0.0124050, 0.0197038, 0.0122976, 0.0217212, -0.0093161, 0.0074062
6: -0.0026104, -0.0016077, -0.0026727, -0.0015804, -0.0010300, 0.0010650
7: -0.0098916, -0.0072972, -0.0100528, -0.0072267, -0.0026649, 0.0027556
8: -0.0047661, -0.0030643, -0.0048508, -0.0022485, -0.0025176, 0.0017865
9: 0.0020806, 0.0036626, 0.0020376, 0.0037609, -0.0016803, 0.0016250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062367
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062803, upper bound: 0.0062372
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9815614, 0.9895260, 0.9806578, 0.9893626, -0.0078012, 0.0088682
1: -0.0045204, -0.0038738, -0.0045336, -0.0039145, -0.0006059, 0.0006598
2: 0.0104751, 0.0139020, 0.0106908, 0.0139716, -0.0034966, 0.0032112
3: -0.0076897, -0.0060409, -0.0077355, -0.0061391, -0.0015506, 0.0016946
4: 0.0025553, 0.0036035, 0.0025971, 0.0036780, -0.0011227, 0.0010065
5: 0.0121343, 0.0204330, 0.0124057, 0.0211525, -0.0090182, 0.0080273
6: -0.0026330, -0.0015390, -0.0026552, -0.0016079, -0.0010251, 0.0011162
7: -0.0099499, -0.0071195, -0.0100074, -0.0072977, -0.0026522, 0.0028879
8: -0.0047967, -0.0027694, -0.0048269, -0.0024784, -0.0023183, 0.0020575
9: 0.0019722, 0.0036981, 0.0020808, 0.0037332, -0.0017610, 0.0016173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.05 + 597.48 = 600.53 seconds
