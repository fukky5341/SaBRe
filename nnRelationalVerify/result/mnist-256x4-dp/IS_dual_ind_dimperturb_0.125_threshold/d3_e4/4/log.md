## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085992


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0067873, 0.0083648, 0.0067873, 0.0083648, -0.0011005, 0.0011005)
1: (0.0023029, 0.0025308, 0.0023029, 0.0025308, -0.0001590, 0.0001590)
2: (0.0097352, 0.0106074, 0.0097352, 0.0106074, -0.0006085, 0.0006085)
3: (-0.0046119, -0.0037098, -0.0046119, -0.0037098, -0.0006293, 0.0006293)
4: (-0.0000208, 0.0009557, -0.0000208, 0.0009557, -0.0006812, 0.0006812)
5: (0.0032090, 0.0041331, 0.0032090, 0.0041331, -0.0006447, 0.0006447)
6: (-0.0095681, -0.0059015, -0.0095681, -0.0059015, -0.0025579, 0.0025579)
7: (0.0054806, 0.0104743, 0.0054806, 0.0104743, -0.0034837, 0.0034837)
8: (0.9930745, 0.9965922, 0.9930745, 0.9965922, -0.0024540, 0.0024540)
9: (-0.0127939, -0.0096008, -0.0127939, -0.0096008, -0.0022276, 0.0022276)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.03 + 1.38 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0010810, upper bound: 0.0010810

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009613, upper bound: 0.0009831
time: 0.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009872, upper bound: 0.0009872
time: 0.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.21 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 8, lower bound: -0.0009613, upper bound: 0.0009831
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 8, lower bound: -0.0009872, upper bound: 0.0009872

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0069711, 0.0083691, 0.0068721, 0.0083647, -0.0008544, 0.0008610
1: 0.0023294, 0.0025314, 0.0023151, 0.0025308, -0.0001234, 0.0001244
2: 0.0097328, 0.0105057, 0.0097352, 0.0105604, -0.0004760, 0.0004724
3: -0.0046144, -0.0038149, -0.0046118, -0.0037584, -0.0004923, 0.0004886
4: 0.0000929, 0.0009584, 0.0000317, 0.0009556, -0.0005289, 0.0005330
5: 0.0032064, 0.0040254, 0.0032090, 0.0040834, -0.0005044, 0.0005005
6: -0.0095782, -0.0063287, -0.0095679, -0.0060988, -0.0020012, 0.0019859
7: 0.0060625, 0.0104880, 0.0057493, 0.0104740, -0.0027046, 0.0027255
8: 0.9934844, 0.9966018, 0.9932638, 0.9965920, -0.0019052, 0.0019199
9: -0.0128026, -0.0099729, -0.0127937, -0.0097726, -0.0017428, 0.0017294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009029, upper bound: 0.0008927
time: 0.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009123, upper bound: 0.0009276
time: 0.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0068610, 0.0083647, 0.0068152, 0.0083648, -0.0007546, 0.0010876
1: 0.0023135, 0.0025308, 0.0023069, 0.0025308, -0.0001090, 0.0001571
2: 0.0097352, 0.0105666, 0.0097352, 0.0105919, -0.0006013, 0.0004172
3: -0.0046118, -0.0037520, -0.0046119, -0.0037258, -0.0006219, 0.0004315
4: 0.0000248, 0.0009556, -0.0000036, 0.0009557, -0.0004671, 0.0006732
5: 0.0032090, 0.0040899, 0.0032090, 0.0041167, -0.0006371, 0.0004421
6: -0.0095678, -0.0060728, -0.0095680, -0.0059664, -0.0025278, 0.0017540
7: 0.0057139, 0.0104739, 0.0055690, 0.0104741, -0.0023888, 0.0034426
8: 0.9932389, 0.9965919, 0.9931368, 0.9965921, -0.0016827, 0.0024251
9: -0.0127936, -0.0097500, -0.0127938, -0.0096573, -0.0022013, 0.0015274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009239, upper bound: 0.0008983
time: 0.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009330, upper bound: 0.0009330
time: 0.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 8, lower bound: -0.0009029, upper bound: 0.0008927
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 8, lower bound: -0.0009123, upper bound: 0.0009276
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 8, lower bound: -0.0009239, upper bound: 0.0008983
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 8, lower bound: -0.0009330, upper bound: 0.0009330

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0070190, 0.0083689, 0.0069807, 0.0083643, -0.0007984, 0.0007456
1: 0.0023363, 0.0025314, 0.0023308, 0.0025307, -0.0001153, 0.0001077
2: 0.0097329, 0.0104793, 0.0097355, 0.0105004, -0.0004122, 0.0004414
3: -0.0046142, -0.0038423, -0.0046116, -0.0038204, -0.0004263, 0.0004565
4: 0.0001226, 0.0009582, 0.0000989, 0.0009554, -0.0004942, 0.0004615
5: 0.0032066, 0.0039974, 0.0032093, 0.0040198, -0.0004368, 0.0004677
6: -0.0095777, -0.0064400, -0.0095670, -0.0063511, -0.0017330, 0.0018556
7: 0.0062140, 0.0104872, 0.0060929, 0.0104727, -0.0025272, 0.0023602
8: 0.9935911, 0.9966013, 0.9935058, 0.9965911, -0.0017802, 0.0016625
9: -0.0128022, -0.0100697, -0.0127929, -0.0099923, -0.0015091, 0.0016159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0008927
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0008927
time: 0.50 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0069959, 0.0083690, 0.0069309, 0.0083936, -0.0008840, 0.0007606
1: 0.0023330, 0.0025314, 0.0023236, 0.0025349, -0.0001277, 0.0001099
2: 0.0097329, 0.0104920, 0.0097193, 0.0105279, -0.0004205, 0.0004888
3: -0.0046142, -0.0038291, -0.0046283, -0.0037920, -0.0004349, 0.0005055
4: 0.0001083, 0.0009582, 0.0000681, 0.0009735, -0.0005472, 0.0004708
5: 0.0032065, 0.0040109, 0.0031921, 0.0040489, -0.0004456, 0.0005179
6: -0.0095777, -0.0063863, -0.0096350, -0.0062353, -0.0017678, 0.0020547
7: 0.0061409, 0.0104873, 0.0059353, 0.0105653, -0.0027984, 0.0024077
8: 0.9935397, 0.9966014, 0.9933949, 0.9966562, -0.0019712, 0.0016960
9: -0.0128022, -0.0100230, -0.0128521, -0.0098915, -0.0015395, 0.0017894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009177
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009276
time: 0.53 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069056, 0.0083645, 0.0069187, 0.0083644, -0.0006988, 0.0009712
1: 0.0023200, 0.0025307, 0.0023218, 0.0025307, -0.0001010, 0.0001403
2: 0.0097354, 0.0105420, 0.0097354, 0.0105347, -0.0005369, 0.0003864
3: -0.0046117, -0.0037775, -0.0046116, -0.0037850, -0.0005553, 0.0003996
4: 0.0000524, 0.0009555, 0.0000605, 0.0009554, -0.0004326, 0.0006012
5: 0.0032091, 0.0040638, 0.0032092, 0.0040561, -0.0005689, 0.0004094
6: -0.0095674, -0.0061764, -0.0095671, -0.0062069, -0.0022572, 0.0016243
7: 0.0058550, 0.0104732, 0.0058966, 0.0104728, -0.0022122, 0.0030742
8: 0.9933383, 0.9965914, 0.9933676, 0.9965912, -0.0015583, 0.0021655
9: -0.0127932, -0.0098402, -0.0127929, -0.0098668, -0.0019657, 0.0014145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0008983
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0008983
time: 0.51 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068901, 0.0083645, 0.0068764, 0.0083936, -0.0008178, 0.0009865
1: 0.0023177, 0.0025307, 0.0023157, 0.0025349, -0.0001181, 0.0001425
2: 0.0097353, 0.0105505, 0.0097193, 0.0105581, -0.0005454, 0.0004521
3: -0.0046117, -0.0037687, -0.0046284, -0.0037608, -0.0005641, 0.0004676
4: 0.0000428, 0.0009555, 0.0000343, 0.0009735, -0.0005062, 0.0006107
5: 0.0032091, 0.0040728, 0.0031921, 0.0040809, -0.0005779, 0.0004791
6: -0.0095675, -0.0061406, -0.0096351, -0.0061086, -0.0022929, 0.0019008
7: 0.0058062, 0.0104734, 0.0057627, 0.0105654, -0.0025887, 0.0031228
8: 0.9933039, 0.9965914, 0.9932733, 0.9966564, -0.0018235, 0.0021997
9: -0.0127933, -0.0098090, -0.0128522, -0.0097812, -0.0019968, 0.0016553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0009239
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0009330
time: 0.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.99 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0008927
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0008927
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009177
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009276
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0008983
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0008983
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0009239
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 8, lower bound: -0.0008983, upper bound: 0.0009330

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0070801, 0.0083686, 0.0069807, 0.0083643, -0.0007335, 0.0007454
1: 0.0023452, 0.0025313, 0.0023308, 0.0025307, -0.0001060, 0.0001077
2: 0.0097331, 0.0104454, 0.0097355, 0.0105004, -0.0004121, 0.0004055
3: -0.0046141, -0.0038773, -0.0046116, -0.0038204, -0.0004262, 0.0004194
4: 0.0001604, 0.0009580, 0.0000989, 0.0009554, -0.0004541, 0.0004614
5: 0.0032067, 0.0039615, 0.0032093, 0.0040198, -0.0004367, 0.0004297
6: -0.0095770, -0.0065822, -0.0095670, -0.0063511, -0.0017326, 0.0017049
7: 0.0064077, 0.0104863, 0.0060929, 0.0104727, -0.0023219, 0.0023597
8: 0.9937276, 0.9966006, 0.9935058, 0.9965911, -0.0016356, 0.0016622
9: -0.0128016, -0.0101936, -0.0127929, -0.0099923, -0.0015088, 0.0014847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008644
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008927
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070263, 0.0084089, 0.0069807, 0.0083643, -0.0008292, 0.0008214
1: 0.0023374, 0.0025371, 0.0023308, 0.0025307, -0.0001198, 0.0001187
2: 0.0097108, 0.0104752, 0.0097355, 0.0105004, -0.0004542, 0.0004584
3: -0.0046371, -0.0038465, -0.0046116, -0.0038204, -0.0004697, 0.0004741
4: 0.0001271, 0.0009830, 0.0000989, 0.0009554, -0.0005133, 0.0005085
5: 0.0031831, 0.0039930, 0.0032093, 0.0040198, -0.0004812, 0.0004857
6: -0.0096707, -0.0064571, -0.0095670, -0.0063511, -0.0019093, 0.0019273
7: 0.0062373, 0.0106139, 0.0060929, 0.0104727, -0.0026248, 0.0026003
8: 0.9936075, 0.9966905, 0.9935058, 0.9965911, -0.0018489, 0.0018317
9: -0.0128832, -0.0100846, -0.0127929, -0.0099923, -0.0016627, 0.0016784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008644
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008927
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0070801, 0.0083686, 0.0069309, 0.0083936, -0.0007761, 0.0008280
1: 0.0023452, 0.0025313, 0.0023236, 0.0025349, -0.0001121, 0.0001196
2: 0.0097331, 0.0104454, 0.0097193, 0.0105279, -0.0004578, 0.0004291
3: -0.0046141, -0.0038773, -0.0046283, -0.0037920, -0.0004735, 0.0004438
4: 0.0001604, 0.0009580, 0.0000681, 0.0009735, -0.0004804, 0.0005126
5: 0.0032067, 0.0039615, 0.0031921, 0.0040489, -0.0004851, 0.0004546
6: -0.0095770, -0.0065822, -0.0096350, -0.0062353, -0.0019245, 0.0018038
7: 0.0064077, 0.0104863, 0.0059353, 0.0105653, -0.0024566, 0.0026211
8: 0.9937276, 0.9966006, 0.9933949, 0.9966562, -0.0017305, 0.0018463
9: -0.0128016, -0.0101936, -0.0128521, -0.0098915, -0.0016760, 0.0015708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009028
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009177
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070263, 0.0084089, 0.0069309, 0.0083936, -0.0007489, 0.0007615
1: 0.0023374, 0.0025371, 0.0023236, 0.0025349, -0.0001082, 0.0001100
2: 0.0097108, 0.0104752, 0.0097193, 0.0105279, -0.0004210, 0.0004141
3: -0.0046371, -0.0038465, -0.0046283, -0.0037920, -0.0004355, 0.0004283
4: 0.0001271, 0.0009830, 0.0000681, 0.0009735, -0.0004636, 0.0004714
5: 0.0031831, 0.0039930, 0.0031921, 0.0040489, -0.0004461, 0.0004387
6: -0.0096707, -0.0064571, -0.0096350, -0.0062353, -0.0017700, 0.0017407
7: 0.0062373, 0.0106139, 0.0059353, 0.0105653, -0.0023707, 0.0024106
8: 0.9936075, 0.9966905, 0.9933949, 0.9966562, -0.0016700, 0.0016981
9: -0.0128832, -0.0100846, -0.0128521, -0.0098915, -0.0015414, 0.0015159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0008745
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009042
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0069641, 0.0083643, 0.0069187, 0.0083644, -0.0006341, 0.0009709
1: 0.0023284, 0.0025307, 0.0023218, 0.0025307, -0.0000916, 0.0001403
2: 0.0097355, 0.0105096, 0.0097354, 0.0105347, -0.0005368, 0.0003506
3: -0.0046116, -0.0038110, -0.0046116, -0.0037850, -0.0005551, 0.0003626
4: 0.0000886, 0.0009553, 0.0000605, 0.0009554, -0.0003925, 0.0006010
5: 0.0032093, 0.0040295, 0.0032092, 0.0040561, -0.0005687, 0.0003714
6: -0.0095669, -0.0063125, -0.0095671, -0.0062069, -0.0022565, 0.0014737
7: 0.0060404, 0.0104725, 0.0058966, 0.0104728, -0.0020071, 0.0030732
8: 0.9934688, 0.9965909, 0.9933676, 0.9965912, -0.0014138, 0.0021648
9: -0.0127928, -0.0099587, -0.0127929, -0.0098668, -0.0019651, 0.0012834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008644
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008716
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069256, 0.0083935, 0.0069187, 0.0083644, -0.0007288, 0.0010135
1: 0.0023229, 0.0025349, 0.0023218, 0.0025307, -0.0001053, 0.0001464
2: 0.0097193, 0.0105309, 0.0097354, 0.0105347, -0.0005603, 0.0004029
3: -0.0046283, -0.0037890, -0.0046116, -0.0037850, -0.0005795, 0.0004167
4: 0.0000648, 0.0009735, 0.0000605, 0.0009554, -0.0004512, 0.0006274
5: 0.0031921, 0.0040520, 0.0032092, 0.0040561, -0.0005937, 0.0004269
6: -0.0096349, -0.0062231, -0.0095671, -0.0062069, -0.0023556, 0.0016940
7: 0.0059186, 0.0105652, 0.0058966, 0.0104728, -0.0023071, 0.0032081
8: 0.9933831, 0.9966562, 0.9933676, 0.9965912, -0.0016251, 0.0022599
9: -0.0128520, -0.0098809, -0.0127929, -0.0098668, -0.0020514, 0.0014752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008644
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008716
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0069641, 0.0083643, 0.0068764, 0.0083936, -0.0007100, 0.0010342
1: 0.0023284, 0.0025307, 0.0023157, 0.0025349, -0.0001026, 0.0001494
2: 0.0097355, 0.0105096, 0.0097193, 0.0105581, -0.0005718, 0.0003925
3: -0.0046116, -0.0038110, -0.0046284, -0.0037608, -0.0005914, 0.0004060
4: 0.0000886, 0.0009553, 0.0000343, 0.0009735, -0.0004395, 0.0006402
5: 0.0032093, 0.0040295, 0.0031921, 0.0040809, -0.0006059, 0.0004159
6: -0.0095669, -0.0063125, -0.0096351, -0.0061086, -0.0024039, 0.0016502
7: 0.0060404, 0.0104725, 0.0057627, 0.0105654, -0.0022474, 0.0032738
8: 0.9934688, 0.9965909, 0.9932733, 0.9966564, -0.0015831, 0.0023062
9: -0.0127928, -0.0099587, -0.0128522, -0.0097812, -0.0020934, 0.0014370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0009028
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0009096
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069256, 0.0083935, 0.0068764, 0.0083936, -0.0006477, 0.0009878
1: 0.0023229, 0.0025349, 0.0023157, 0.0025349, -0.0000936, 0.0001427
2: 0.0097193, 0.0105309, 0.0097193, 0.0105581, -0.0005461, 0.0003581
3: -0.0046283, -0.0037890, -0.0046284, -0.0037608, -0.0005648, 0.0003704
4: 0.0000648, 0.0009735, 0.0000343, 0.0009735, -0.0004009, 0.0006114
5: 0.0031921, 0.0040520, 0.0031921, 0.0040809, -0.0005786, 0.0003794
6: -0.0096349, -0.0062231, -0.0096351, -0.0061086, -0.0022959, 0.0015054
7: 0.0059186, 0.0105652, 0.0057627, 0.0105654, -0.0020502, 0.0031267
8: 0.9933831, 0.9966562, 0.9932733, 0.9966564, -0.0014442, 0.0022025
9: -0.0128520, -0.0098809, -0.0128522, -0.0097812, -0.0019993, 0.0013110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008745
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008828
time: 0.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.08 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008644
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008927
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008644
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008927
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009028
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009177
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0008745
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008655, upper bound: 0.0009042
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008644
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008716
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008644
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008716
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0009028
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0009096
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008745
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0008927, upper bound: 0.0008828

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0070801, 0.0083686, 0.0070801, 0.0083686, -0.0006236, 0.0006236
1: 0.0023452, 0.0025313, 0.0023452, 0.0025313, -0.0000901, 0.0000901
2: 0.0097331, 0.0104454, 0.0097331, 0.0104454, -0.0003448, 0.0003448
3: -0.0046141, -0.0038773, -0.0046141, -0.0038773, -0.0003566, 0.0003566
4: 0.0001604, 0.0009580, 0.0001604, 0.0009580, -0.0003860, 0.0003860
5: 0.0032067, 0.0039615, 0.0032067, 0.0039615, -0.0003653, 0.0003653
6: -0.0095770, -0.0065822, -0.0095770, -0.0065822, -0.0014494, 0.0014494
7: 0.0064077, 0.0104863, 0.0064077, 0.0104863, -0.0019740, 0.0019740
8: 0.9937276, 0.9966006, 0.9937276, 0.9966006, -0.0013905, 0.0013905
9: -0.0128016, -0.0101936, -0.0128016, -0.0101936, -0.0012622, 0.0012622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007966, upper bound: 0.0008006
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008245
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0070801, 0.0083686, 0.0069641, 0.0083643, -0.0007317, 0.0008532
1: 0.0023452, 0.0025313, 0.0023284, 0.0025307, -0.0001057, 0.0001233
2: 0.0097331, 0.0104454, 0.0097355, 0.0105096, -0.0004717, 0.0004045
3: -0.0046141, -0.0038773, -0.0046116, -0.0038110, -0.0004878, 0.0004184
4: 0.0001604, 0.0009580, 0.0000886, 0.0009553, -0.0004529, 0.0005281
5: 0.0032067, 0.0039615, 0.0032093, 0.0040295, -0.0004998, 0.0004286
6: -0.0095770, -0.0065822, -0.0095669, -0.0063125, -0.0019830, 0.0017007
7: 0.0064077, 0.0104863, 0.0060404, 0.0104725, -0.0023162, 0.0027007
8: 0.9937276, 0.9966006, 0.9934688, 0.9965909, -0.0016316, 0.0019024
9: -0.0128016, -0.0101936, -0.0127928, -0.0099587, -0.0017269, 0.0014810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007966, upper bound: 0.0008374
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008591
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070263, 0.0084089, 0.0070801, 0.0083686, -0.0007193, 0.0006996
1: 0.0023374, 0.0025371, 0.0023452, 0.0025313, -0.0001039, 0.0001011
2: 0.0097108, 0.0104752, 0.0097331, 0.0104454, -0.0003868, 0.0003977
3: -0.0046371, -0.0038465, -0.0046141, -0.0038773, -0.0004000, 0.0004113
4: 0.0001271, 0.0009830, 0.0001604, 0.0009580, -0.0004453, 0.0004331
5: 0.0031831, 0.0039930, 0.0032067, 0.0039615, -0.0004098, 0.0004214
6: -0.0096707, -0.0064571, -0.0095770, -0.0065822, -0.0016261, 0.0016718
7: 0.0062373, 0.0106139, 0.0064077, 0.0104863, -0.0022769, 0.0022146
8: 0.9936075, 0.9966905, 0.9937276, 0.9966006, -0.0016039, 0.0015600
9: -0.0128832, -0.0100846, -0.0128016, -0.0101936, -0.0014161, 0.0014559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008140, upper bound: 0.0007988
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008754, upper bound: 0.0008243
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070263, 0.0084089, 0.0069641, 0.0083643, -0.0008274, 0.0009292
1: 0.0023374, 0.0025371, 0.0023284, 0.0025307, -0.0001195, 0.0001342
2: 0.0097108, 0.0104752, 0.0097355, 0.0105096, -0.0005137, 0.0004574
3: -0.0046371, -0.0038465, -0.0046116, -0.0038110, -0.0005313, 0.0004731
4: 0.0001271, 0.0009830, 0.0000886, 0.0009553, -0.0005122, 0.0005752
5: 0.0031831, 0.0039930, 0.0032093, 0.0040295, -0.0005443, 0.0004847
6: -0.0096707, -0.0064571, -0.0095669, -0.0063125, -0.0021597, 0.0019230
7: 0.0062373, 0.0106139, 0.0060404, 0.0104725, -0.0026190, 0.0029413
8: 0.9936075, 0.9966905, 0.9934688, 0.9965909, -0.0018449, 0.0020719
9: -0.0128832, -0.0100846, -0.0127928, -0.0099587, -0.0018807, 0.0016747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008140, upper bound: 0.0008356
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008754, upper bound: 0.0008584
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0070801, 0.0083686, 0.0070263, 0.0084089, -0.0006996, 0.0007193
1: 0.0023452, 0.0025313, 0.0023374, 0.0025371, -0.0001011, 0.0001039
2: 0.0097331, 0.0104454, 0.0097108, 0.0104752, -0.0003977, 0.0003868
3: -0.0046141, -0.0038773, -0.0046371, -0.0038465, -0.0004113, 0.0004000
4: 0.0001604, 0.0009580, 0.0001271, 0.0009830, -0.0004331, 0.0004453
5: 0.0032067, 0.0039615, 0.0031831, 0.0039930, -0.0004214, 0.0004098
6: -0.0095770, -0.0065822, -0.0096707, -0.0064571, -0.0016718, 0.0016261
7: 0.0064077, 0.0104863, 0.0062373, 0.0106139, -0.0022146, 0.0022769
8: 0.9937276, 0.9966006, 0.9936075, 0.9966905, -0.0015600, 0.0016039
9: -0.0128016, -0.0101936, -0.0128832, -0.0100846, -0.0014559, 0.0014161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007965, upper bound: 0.0008480
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008252, upper bound: 0.0008748
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0070801, 0.0083686, 0.0069256, 0.0083935, -0.0007743, 0.0009097
1: 0.0023452, 0.0025313, 0.0023229, 0.0025349, -0.0001119, 0.0001314
2: 0.0097331, 0.0104454, 0.0097193, 0.0105309, -0.0005030, 0.0004281
3: -0.0046141, -0.0038773, -0.0046283, -0.0037890, -0.0005202, 0.0004428
4: 0.0001604, 0.0009580, 0.0000648, 0.0009735, -0.0004793, 0.0005631
5: 0.0032067, 0.0039615, 0.0031921, 0.0040520, -0.0005329, 0.0004536
6: -0.0095770, -0.0065822, -0.0096349, -0.0062231, -0.0021144, 0.0017997
7: 0.0064077, 0.0104863, 0.0059186, 0.0105652, -0.0024511, 0.0028797
8: 0.9937276, 0.9966006, 0.9933831, 0.9966562, -0.0017266, 0.0020285
9: -0.0128016, -0.0101936, -0.0128520, -0.0098809, -0.0018413, 0.0015673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007965, upper bound: 0.0008682
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008252, upper bound: 0.0008923
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070263, 0.0084089, 0.0070263, 0.0084089, -0.0006386, 0.0006386
1: 0.0023374, 0.0025371, 0.0023374, 0.0025371, -0.0000923, 0.0000923
2: 0.0097108, 0.0104752, 0.0097108, 0.0104752, -0.0003530, 0.0003530
3: -0.0046371, -0.0038465, -0.0046371, -0.0038465, -0.0003651, 0.0003651
4: 0.0001271, 0.0009830, 0.0001271, 0.0009830, -0.0003953, 0.0003953
5: 0.0031831, 0.0039930, 0.0031831, 0.0039930, -0.0003741, 0.0003741
6: -0.0096707, -0.0064571, -0.0096707, -0.0064571, -0.0014842, 0.0014842
7: 0.0062373, 0.0106139, 0.0062373, 0.0106139, -0.0020213, 0.0020213
8: 0.9936075, 0.9966905, 0.9936075, 0.9966905, -0.0014239, 0.0014239
9: -0.0128832, -0.0100846, -0.0128832, -0.0100846, -0.0012925, 0.0012925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008142, upper bound: 0.0008001
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008856, upper bound: 0.0008389
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070263, 0.0084089, 0.0069256, 0.0083935, -0.0007471, 0.0008691
1: 0.0023374, 0.0025371, 0.0023229, 0.0025349, -0.0001079, 0.0001256
2: 0.0097108, 0.0104752, 0.0097193, 0.0105309, -0.0004805, 0.0004131
3: -0.0046371, -0.0038465, -0.0046283, -0.0037890, -0.0004970, 0.0004272
4: 0.0001271, 0.0009830, 0.0000648, 0.0009735, -0.0004625, 0.0005380
5: 0.0031831, 0.0039930, 0.0031921, 0.0040520, -0.0005091, 0.0004377
6: -0.0096707, -0.0064571, -0.0096349, -0.0062231, -0.0020201, 0.0017366
7: 0.0062373, 0.0106139, 0.0059186, 0.0105652, -0.0023651, 0.0027512
8: 0.9936075, 0.9966905, 0.9933831, 0.9966562, -0.0016660, 0.0019380
9: -0.0128832, -0.0100846, -0.0128520, -0.0098809, -0.0017592, 0.0015123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008142, upper bound: 0.0008374
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008856, upper bound: 0.0008716
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0069641, 0.0083643, 0.0070801, 0.0083686, -0.0008532, 0.0007317
1: 0.0023284, 0.0025307, 0.0023452, 0.0025313, -0.0001233, 0.0001057
2: 0.0097355, 0.0105096, 0.0097331, 0.0104454, -0.0004045, 0.0004717
3: -0.0046116, -0.0038110, -0.0046141, -0.0038773, -0.0004184, 0.0004878
4: 0.0000886, 0.0009553, 0.0001604, 0.0009580, -0.0005281, 0.0004529
5: 0.0032093, 0.0040295, 0.0032067, 0.0039615, -0.0004286, 0.0004998
6: -0.0095669, -0.0063125, -0.0095770, -0.0065822, -0.0017007, 0.0019830
7: 0.0060404, 0.0104725, 0.0064077, 0.0104863, -0.0027007, 0.0023162
8: 0.9934688, 0.9965909, 0.9937276, 0.9966006, -0.0019024, 0.0016316
9: -0.0127928, -0.0099587, -0.0128016, -0.0101936, -0.0014810, 0.0017269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008175, upper bound: 0.0008006
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008591, upper bound: 0.0008245
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0069641, 0.0083643, 0.0069641, 0.0083643, -0.0006334, 0.0006334
1: 0.0023284, 0.0025307, 0.0023284, 0.0025307, -0.0000915, 0.0000915
2: 0.0097355, 0.0105096, 0.0097355, 0.0105096, -0.0003502, 0.0003502
3: -0.0046116, -0.0038110, -0.0046116, -0.0038110, -0.0003622, 0.0003622
4: 0.0000886, 0.0009553, 0.0000886, 0.0009553, -0.0003921, 0.0003921
5: 0.0032093, 0.0040295, 0.0032093, 0.0040295, -0.0003710, 0.0003710
6: -0.0095669, -0.0063125, -0.0095669, -0.0063125, -0.0014721, 0.0014721
7: 0.0060404, 0.0104725, 0.0060404, 0.0104725, -0.0020049, 0.0020049
8: 0.9934688, 0.9965909, 0.9934688, 0.9965909, -0.0014123, 0.0014123
9: -0.0127928, -0.0099587, -0.0127928, -0.0099587, -0.0012820, 0.0012820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008175, upper bound: 0.0008050
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008591, upper bound: 0.0008307
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069256, 0.0083935, 0.0070801, 0.0083686, -0.0009097, 0.0007743
1: 0.0023229, 0.0025349, 0.0023452, 0.0025313, -0.0001314, 0.0001119
2: 0.0097193, 0.0105309, 0.0097331, 0.0104454, -0.0004281, 0.0005030
3: -0.0046283, -0.0037890, -0.0046141, -0.0038773, -0.0004428, 0.0005202
4: 0.0000648, 0.0009735, 0.0001604, 0.0009580, -0.0005631, 0.0004793
5: 0.0031921, 0.0040520, 0.0032067, 0.0039615, -0.0004536, 0.0005329
6: -0.0096349, -0.0062231, -0.0095770, -0.0065822, -0.0017997, 0.0021144
7: 0.0059186, 0.0105652, 0.0064077, 0.0104863, -0.0028797, 0.0024511
8: 0.9933831, 0.9966562, 0.9937276, 0.9966006, -0.0020285, 0.0017266
9: -0.0128520, -0.0098809, -0.0128016, -0.0101936, -0.0015673, 0.0018413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008201, upper bound: 0.0007988
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008923, upper bound: 0.0008243
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069256, 0.0083935, 0.0069641, 0.0083643, -0.0007281, 0.0007092
1: 0.0023229, 0.0025349, 0.0023284, 0.0025307, -0.0001052, 0.0001025
2: 0.0097193, 0.0105309, 0.0097355, 0.0105096, -0.0003921, 0.0004026
3: -0.0046283, -0.0037890, -0.0046116, -0.0038110, -0.0004055, 0.0004164
4: 0.0000648, 0.0009735, 0.0000886, 0.0009553, -0.0004507, 0.0004390
5: 0.0031921, 0.0040520, 0.0032093, 0.0040295, -0.0004154, 0.0004265
6: -0.0096349, -0.0062231, -0.0095669, -0.0063125, -0.0016483, 0.0016924
7: 0.0059186, 0.0105652, 0.0060404, 0.0104725, -0.0023049, 0.0022449
8: 0.9933831, 0.9966562, 0.9934688, 0.9965909, -0.0016236, 0.0015813
9: -0.0128520, -0.0098809, -0.0127928, -0.0099587, -0.0014354, 0.0014738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008201, upper bound: 0.0008033
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008923, upper bound: 0.0008303
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0069641, 0.0083643, 0.0070263, 0.0084089, -0.0009292, 0.0008274
1: 0.0023284, 0.0025307, 0.0023374, 0.0025371, -0.0001342, 0.0001195
2: 0.0097355, 0.0105096, 0.0097108, 0.0104752, -0.0004574, 0.0005137
3: -0.0046116, -0.0038110, -0.0046371, -0.0038465, -0.0004731, 0.0005313
4: 0.0000886, 0.0009553, 0.0001271, 0.0009830, -0.0005752, 0.0005122
5: 0.0032093, 0.0040295, 0.0031831, 0.0039930, -0.0004847, 0.0005443
6: -0.0095669, -0.0063125, -0.0096707, -0.0064571, -0.0019230, 0.0021597
7: 0.0060404, 0.0104725, 0.0062373, 0.0106139, -0.0029413, 0.0026190
8: 0.9934688, 0.9965909, 0.9936075, 0.9966905, -0.0020719, 0.0018449
9: -0.0127928, -0.0099587, -0.0128832, -0.0100846, -0.0016747, 0.0018807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008171, upper bound: 0.0008480
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008584, upper bound: 0.0008748
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0069641, 0.0083643, 0.0069256, 0.0083935, -0.0007092, 0.0007281
1: 0.0023284, 0.0025307, 0.0023229, 0.0025349, -0.0001025, 0.0001052
2: 0.0097355, 0.0105096, 0.0097193, 0.0105309, -0.0004026, 0.0003921
3: -0.0046116, -0.0038110, -0.0046283, -0.0037890, -0.0004164, 0.0004055
4: 0.0000886, 0.0009553, 0.0000648, 0.0009735, -0.0004390, 0.0004507
5: 0.0032093, 0.0040295, 0.0031921, 0.0040520, -0.0004265, 0.0004154
6: -0.0095669, -0.0063125, -0.0096349, -0.0062231, -0.0016924, 0.0016483
7: 0.0060404, 0.0104725, 0.0059186, 0.0105652, -0.0022449, 0.0023049
8: 0.9934688, 0.9965909, 0.9933831, 0.9966562, -0.0015813, 0.0016236
9: -0.0127928, -0.0099587, -0.0128520, -0.0098809, -0.0014738, 0.0014354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008171, upper bound: 0.0008542
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008584, upper bound: 0.0008817
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069256, 0.0083935, 0.0070263, 0.0084089, -0.0008691, 0.0007471
1: 0.0023229, 0.0025349, 0.0023374, 0.0025371, -0.0001256, 0.0001079
2: 0.0097193, 0.0105309, 0.0097108, 0.0104752, -0.0004131, 0.0004805
3: -0.0046283, -0.0037890, -0.0046371, -0.0038465, -0.0004272, 0.0004970
4: 0.0000648, 0.0009735, 0.0001271, 0.0009830, -0.0005380, 0.0004625
5: 0.0031921, 0.0040520, 0.0031831, 0.0039930, -0.0004377, 0.0005091
6: -0.0096349, -0.0062231, -0.0096707, -0.0064571, -0.0017366, 0.0020201
7: 0.0059186, 0.0105652, 0.0062373, 0.0106139, -0.0027512, 0.0023651
8: 0.9933831, 0.9966562, 0.9936075, 0.9966905, -0.0019380, 0.0016660
9: -0.0128520, -0.0098809, -0.0128832, -0.0100846, -0.0015123, 0.0017592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008207, upper bound: 0.0008001
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009028, upper bound: 0.0008388
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069256, 0.0083935, 0.0069256, 0.0083935, -0.0006470, 0.0006470
1: 0.0023229, 0.0025349, 0.0023229, 0.0025349, -0.0000935, 0.0000935
2: 0.0097193, 0.0105309, 0.0097193, 0.0105309, -0.0003577, 0.0003577
3: -0.0046283, -0.0037890, -0.0046283, -0.0037890, -0.0003700, 0.0003700
4: 0.0000648, 0.0009735, 0.0000648, 0.0009735, -0.0004005, 0.0004005
5: 0.0031921, 0.0040520, 0.0031921, 0.0040520, -0.0003790, 0.0003790
6: -0.0096349, -0.0062231, -0.0096349, -0.0062231, -0.0015039, 0.0015039
7: 0.0059186, 0.0105652, 0.0059186, 0.0105652, -0.0020482, 0.0020482
8: 0.9933831, 0.9966562, 0.9933831, 0.9966562, -0.0014428, 0.0014428
9: -0.0128520, -0.0098809, -0.0128520, -0.0098809, -0.0013097, 0.0013097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008207, upper bound: 0.0008086
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009028, upper bound: 0.0008438
time: 0.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.02 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0007966, upper bound: 0.0008006
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008245
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0007966, upper bound: 0.0008374
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008591
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008140, upper bound: 0.0007988
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008754, upper bound: 0.0008243
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008140, upper bound: 0.0008356
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008754, upper bound: 0.0008584
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0007965, upper bound: 0.0008480
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008252, upper bound: 0.0008748
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0007965, upper bound: 0.0008682
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008252, upper bound: 0.0008923
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008142, upper bound: 0.0008001
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008856, upper bound: 0.0008389
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008142, upper bound: 0.0008374
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008856, upper bound: 0.0008716
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008175, upper bound: 0.0008006
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008591, upper bound: 0.0008245
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008175, upper bound: 0.0008050
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008591, upper bound: 0.0008307
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008201, upper bound: 0.0007988
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008923, upper bound: 0.0008243
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008201, upper bound: 0.0008033
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008923, upper bound: 0.0008303
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008171, upper bound: 0.0008480
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008584, upper bound: 0.0008748
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008171, upper bound: 0.0008542
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008584, upper bound: 0.0008817
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008207, upper bound: 0.0008001
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0009028, upper bound: 0.0008388
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0008207, upper bound: 0.0008086
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 8, lower bound: -0.0009028, upper bound: 0.0008438

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070459, 0.0084088, 0.0070836, 0.0083686, -0.0006058, 0.0006975
1: 0.0023402, 0.0025371, 0.0023457, 0.0025313, -0.0000875, 0.0001008
2: 0.0097109, 0.0104643, 0.0097331, 0.0104435, -0.0003856, 0.0003349
3: -0.0046370, -0.0038577, -0.0046141, -0.0038793, -0.0003989, 0.0003464
4: 0.0001393, 0.0009829, 0.0001626, 0.0009580, -0.0003750, 0.0004318
5: 0.0031832, 0.0039816, 0.0032067, 0.0039595, -0.0004086, 0.0003549
6: -0.0096703, -0.0065027, -0.0095769, -0.0065902, -0.0016213, 0.0014081
7: 0.0062994, 0.0106134, 0.0064185, 0.0104862, -0.0019176, 0.0022080
8: 0.9936513, 0.9966902, 0.9937353, 0.9966006, -0.0013508, 0.0015554
9: -0.0128829, -0.0101244, -0.0128015, -0.0102005, -0.0014119, 0.0012262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0007939
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0008252
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070459, 0.0084088, 0.0069676, 0.0083642, -0.0007264, 0.0009266
1: 0.0023402, 0.0025371, 0.0023289, 0.0025307, -0.0001049, 0.0001339
2: 0.0097109, 0.0104643, 0.0097355, 0.0105077, -0.0005123, 0.0004016
3: -0.0046370, -0.0038577, -0.0046116, -0.0038129, -0.0005299, 0.0004154
4: 0.0001393, 0.0009829, 0.0000908, 0.0009553, -0.0004497, 0.0005736
5: 0.0031832, 0.0039816, 0.0032093, 0.0040275, -0.0005428, 0.0004255
6: -0.0096703, -0.0065027, -0.0095668, -0.0063206, -0.0021538, 0.0016884
7: 0.0062994, 0.0106134, 0.0060513, 0.0104724, -0.0022994, 0.0029332
8: 0.9936513, 0.9966902, 0.9934765, 0.9965909, -0.0016198, 0.0020662
9: -0.0128829, -0.0101244, -0.0127927, -0.0099657, -0.0018756, 0.0014703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0008172
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0008584
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0071039, 0.0083685, 0.0070292, 0.0084089, -0.0005918, 0.0007168
1: 0.0023486, 0.0025313, 0.0023378, 0.0025371, -0.0000855, 0.0001036
2: 0.0097332, 0.0104323, 0.0097108, 0.0104736, -0.0003963, 0.0003272
3: -0.0046140, -0.0038909, -0.0046371, -0.0038481, -0.0004099, 0.0003384
4: 0.0001751, 0.0009579, 0.0001289, 0.0009830, -0.0003663, 0.0004437
5: 0.0032068, 0.0039476, 0.0031831, 0.0039914, -0.0004199, 0.0003467
6: -0.0095766, -0.0066373, -0.0096706, -0.0064637, -0.0016660, 0.0013755
7: 0.0064828, 0.0104858, 0.0062463, 0.0106139, -0.0018734, 0.0022690
8: 0.9937805, 0.9966003, 0.9936138, 0.9966905, -0.0013196, 0.0015983
9: -0.0128013, -0.0102416, -0.0128831, -0.0100904, -0.0014509, 0.0011979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008140
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008754
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0071706, 0.0084066, 0.0069844, 0.0083934, -0.0006521, 0.0008410
1: 0.0023583, 0.0025368, 0.0023313, 0.0025349, -0.0000942, 0.0001215
2: 0.0097121, 0.0103954, 0.0097194, 0.0104984, -0.0004650, 0.0003605
3: -0.0046358, -0.0039290, -0.0046282, -0.0038225, -0.0004809, 0.0003729
4: 0.0002165, 0.0009815, 0.0001012, 0.0009734, -0.0004037, 0.0005206
5: 0.0031845, 0.0039085, 0.0031922, 0.0040176, -0.0004927, 0.0003820
6: -0.0096652, -0.0067925, -0.0096346, -0.0063596, -0.0019547, 0.0015157
7: 0.0066941, 0.0106064, 0.0061045, 0.0105647, -0.0020643, 0.0026621
8: 0.9939294, 0.9966853, 0.9935139, 0.9966559, -0.0014541, 0.0018753
9: -0.0128784, -0.0103768, -0.0128517, -0.0099997, -0.0017022, 0.0013200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007612, upper bound: 0.0007866
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007812, upper bound: 0.0008502
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0071039, 0.0083685, 0.0069285, 0.0083935, -0.0006741, 0.0009071
1: 0.0023486, 0.0025313, 0.0023233, 0.0025349, -0.0000974, 0.0001310
2: 0.0097332, 0.0104323, 0.0097193, 0.0105293, -0.0005015, 0.0003727
3: -0.0046140, -0.0038909, -0.0046283, -0.0037906, -0.0005187, 0.0003855
4: 0.0001751, 0.0009579, 0.0000666, 0.0009734, -0.0004173, 0.0005615
5: 0.0032068, 0.0039476, 0.0031921, 0.0040503, -0.0005314, 0.0003949
6: -0.0095766, -0.0066373, -0.0096348, -0.0062298, -0.0021083, 0.0015668
7: 0.0064828, 0.0104858, 0.0059278, 0.0105651, -0.0021338, 0.0028713
8: 0.9937805, 0.9966003, 0.9933895, 0.9966562, -0.0015031, 0.0020226
9: -0.0128013, -0.0102416, -0.0128519, -0.0098867, -0.0018360, 0.0013644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008201
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008923
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070459, 0.0084088, 0.0070292, 0.0084089, -0.0005369, 0.0006365
1: 0.0023402, 0.0025371, 0.0023378, 0.0025371, -0.0000776, 0.0000919
2: 0.0097109, 0.0104643, 0.0097108, 0.0104736, -0.0003519, 0.0002969
3: -0.0046370, -0.0038577, -0.0046371, -0.0038481, -0.0003639, 0.0003070
4: 0.0001393, 0.0009829, 0.0001289, 0.0009830, -0.0003324, 0.0003940
5: 0.0031832, 0.0039816, 0.0031831, 0.0039914, -0.0003728, 0.0003145
6: -0.0096703, -0.0065027, -0.0096706, -0.0064637, -0.0014793, 0.0012480
7: 0.0062994, 0.0106134, 0.0062463, 0.0106139, -0.0016996, 0.0020147
8: 0.9936513, 0.9966902, 0.9936138, 0.9966905, -0.0011972, 0.0014192
9: -0.0128829, -0.0101244, -0.0128831, -0.0100904, -0.0012882, 0.0010868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0007948
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0008403
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070459, 0.0084088, 0.0069285, 0.0083935, -0.0006576, 0.0008666
1: 0.0023402, 0.0025371, 0.0023233, 0.0025349, -0.0000950, 0.0001252
2: 0.0097109, 0.0104643, 0.0097193, 0.0105293, -0.0004791, 0.0003636
3: -0.0046370, -0.0038577, -0.0046283, -0.0037906, -0.0004955, 0.0003760
4: 0.0001393, 0.0009829, 0.0000666, 0.0009734, -0.0004071, 0.0005365
5: 0.0031832, 0.0039816, 0.0031921, 0.0040503, -0.0005077, 0.0003852
6: -0.0096703, -0.0065027, -0.0096348, -0.0062298, -0.0020143, 0.0015285
7: 0.0062994, 0.0106134, 0.0059278, 0.0105651, -0.0020816, 0.0027433
8: 0.9936513, 0.9966902, 0.9933895, 0.9966562, -0.0014664, 0.0019324
9: -0.0128829, -0.0101244, -0.0128519, -0.0098867, -0.0017541, 0.0013311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0008141
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0008716
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0070836, 0.0083686, -0.0008003, 0.0007722
1: 0.0023257, 0.0025349, 0.0023457, 0.0025313, -0.0001156, 0.0001116
2: 0.0097194, 0.0105200, 0.0097331, 0.0104435, -0.0004270, 0.0004424
3: -0.0046282, -0.0038002, -0.0046141, -0.0038793, -0.0004416, 0.0004576
4: 0.0000770, 0.0009733, 0.0001626, 0.0009580, -0.0004954, 0.0004780
5: 0.0031922, 0.0040405, 0.0032067, 0.0039595, -0.0004524, 0.0004688
6: -0.0096345, -0.0062688, -0.0095769, -0.0065902, -0.0017949, 0.0018600
7: 0.0059808, 0.0105646, 0.0064185, 0.0104862, -0.0025332, 0.0024445
8: 0.9934269, 0.9966558, 0.9937353, 0.9966006, -0.0017844, 0.0017220
9: -0.0128516, -0.0099207, -0.0128015, -0.0102005, -0.0015631, 0.0016198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008682, upper bound: 0.0007939
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008682, upper bound: 0.0008252
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0069676, 0.0083642, -0.0006219, 0.0007071
1: 0.0023257, 0.0025349, 0.0023289, 0.0025307, -0.0000898, 0.0001022
2: 0.0097194, 0.0105200, 0.0097355, 0.0105077, -0.0003909, 0.0003438
3: -0.0046282, -0.0038002, -0.0046116, -0.0038129, -0.0004043, 0.0003556
4: 0.0000770, 0.0009733, 0.0000908, 0.0009553, -0.0003850, 0.0004377
5: 0.0031922, 0.0040405, 0.0032093, 0.0040275, -0.0004142, 0.0003643
6: -0.0096345, -0.0062688, -0.0095668, -0.0063206, -0.0016435, 0.0014454
7: 0.0059808, 0.0105646, 0.0060513, 0.0104724, -0.0019685, 0.0022382
8: 0.9934269, 0.9966558, 0.9934765, 0.9965909, -0.0013867, 0.0015767
9: -0.0128516, -0.0099207, -0.0127927, -0.0099657, -0.0014312, 0.0012587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008727, upper bound: 0.0007974
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008727, upper bound: 0.0008303
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069866, 0.0083641, 0.0070292, 0.0084089, -0.0008247, 0.0008249
1: 0.0023317, 0.0025307, 0.0023378, 0.0025371, -0.0001191, 0.0001192
2: 0.0097356, 0.0104972, 0.0097108, 0.0104736, -0.0004561, 0.0004559
3: -0.0046115, -0.0038238, -0.0046371, -0.0038481, -0.0004717, 0.0004715
4: 0.0001025, 0.0009552, 0.0001289, 0.0009830, -0.0005105, 0.0005106
5: 0.0032094, 0.0040163, 0.0031831, 0.0039914, -0.0004832, 0.0004831
6: -0.0095665, -0.0063647, -0.0096706, -0.0064637, -0.0019173, 0.0019167
7: 0.0061114, 0.0104720, 0.0062463, 0.0106139, -0.0026104, 0.0026112
8: 0.9935189, 0.9965905, 0.9936138, 0.9966905, -0.0018388, 0.0018394
9: -0.0127924, -0.0100042, -0.0128831, -0.0100904, -0.0016697, 0.0016692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008356, upper bound: 0.0008140
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008356, upper bound: 0.0008754
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069866, 0.0083641, 0.0069285, 0.0083935, -0.0006089, 0.0007257
1: 0.0023317, 0.0025307, 0.0023233, 0.0025349, -0.0000880, 0.0001048
2: 0.0097356, 0.0104972, 0.0097193, 0.0105293, -0.0004012, 0.0003366
3: -0.0046115, -0.0038238, -0.0046283, -0.0037906, -0.0004150, 0.0003482
4: 0.0001025, 0.0009552, 0.0000666, 0.0009734, -0.0003769, 0.0004492
5: 0.0032094, 0.0040163, 0.0031921, 0.0040503, -0.0004251, 0.0003567
6: -0.0095665, -0.0063647, -0.0096348, -0.0062298, -0.0016867, 0.0014152
7: 0.0061114, 0.0104720, 0.0059278, 0.0105651, -0.0019274, 0.0022972
8: 0.9935189, 0.9965905, 0.9933895, 0.9966562, -0.0013577, 0.0016182
9: -0.0127924, -0.0100042, -0.0128519, -0.0098867, -0.0014689, 0.0012324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008389, upper bound: 0.0008182
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008389, upper bound: 0.0008817
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0070292, 0.0084089, -0.0007671, 0.0007451
1: 0.0023257, 0.0025349, 0.0023378, 0.0025371, -0.0001108, 0.0001076
2: 0.0097194, 0.0105200, 0.0097108, 0.0104736, -0.0004119, 0.0004241
3: -0.0046282, -0.0038002, -0.0046371, -0.0038481, -0.0004260, 0.0004386
4: 0.0000770, 0.0009733, 0.0001289, 0.0009830, -0.0004749, 0.0004612
5: 0.0031922, 0.0040405, 0.0031831, 0.0039914, -0.0004365, 0.0004494
6: -0.0096345, -0.0062688, -0.0096706, -0.0064637, -0.0017317, 0.0017830
7: 0.0059808, 0.0105646, 0.0062463, 0.0106139, -0.0024283, 0.0023585
8: 0.9934269, 0.9966558, 0.9936138, 0.9966905, -0.0017105, 0.0016613
9: -0.0128516, -0.0099207, -0.0128831, -0.0100904, -0.0015081, 0.0015527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008687, upper bound: 0.0007948
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008687, upper bound: 0.0008403
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0069285, 0.0083935, -0.0005480, 0.0006450
1: 0.0023257, 0.0025349, 0.0023233, 0.0025349, -0.0000792, 0.0000932
2: 0.0097194, 0.0105200, 0.0097193, 0.0105293, -0.0003566, 0.0003030
3: -0.0046282, -0.0038002, -0.0046283, -0.0037906, -0.0003688, 0.0003133
4: 0.0000770, 0.0009733, 0.0000666, 0.0009734, -0.0003392, 0.0003993
5: 0.0031922, 0.0040405, 0.0031921, 0.0040503, -0.0003778, 0.0003210
6: -0.0096345, -0.0062688, -0.0096348, -0.0062298, -0.0014991, 0.0012737
7: 0.0059808, 0.0105646, 0.0059278, 0.0105651, -0.0017347, 0.0020417
8: 0.9934269, 0.9966558, 0.9933895, 0.9966562, -0.0012219, 0.0014382
9: -0.0128516, -0.0099207, -0.0128519, -0.0098867, -0.0013055, 0.0011092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008759, upper bound: 0.0008019
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008759, upper bound: 0.0008438
time: 0.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.13 seconds
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0007939
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0008252
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0008172
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008484, upper bound: 0.0008584
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008140
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008754
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0007612, upper bound: 0.0007866
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0007812, upper bound: 0.0008502
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008201
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0007996, upper bound: 0.0008923
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0007948
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0008403
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0008141
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008485, upper bound: 0.0008716
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008682, upper bound: 0.0007939
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008682, upper bound: 0.0008252
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008727, upper bound: 0.0007974
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008727, upper bound: 0.0008303
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008356, upper bound: 0.0008140
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008356, upper bound: 0.0008754
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008389, upper bound: 0.0008182
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008389, upper bound: 0.0008817
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008687, upper bound: 0.0007948
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008687, upper bound: 0.0008403
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008759, upper bound: 0.0008019
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 8, lower bound: -0.0008759, upper bound: 0.0008438

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071039, 0.0083685, 0.0070459, 0.0084088, -0.0005917, 0.0006056
1: 0.0023486, 0.0025313, 0.0023402, 0.0025371, -0.0000855, 0.0000875
2: 0.0097332, 0.0104323, 0.0097109, 0.0104643, -0.0003348, 0.0003271
3: -0.0046140, -0.0038909, -0.0046370, -0.0038577, -0.0003463, 0.0003383
4: 0.0001751, 0.0009579, 0.0001393, 0.0009829, -0.0003663, 0.0003749
5: 0.0032068, 0.0039476, 0.0031832, 0.0039816, -0.0003548, 0.0003466
6: -0.0095766, -0.0066373, -0.0096703, -0.0065027, -0.0014077, 0.0013752
7: 0.0064828, 0.0104858, 0.0062994, 0.0106134, -0.0018729, 0.0019171
8: 0.9937805, 0.9966003, 0.9936513, 0.9966902, -0.0013193, 0.0013505
9: -0.0128013, -0.0102416, -0.0128829, -0.0101244, -0.0012259, 0.0011976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007290, upper bound: 0.0008096
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007851, upper bound: 0.0008276
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071039, 0.0083685, 0.0069453, 0.0083934, -0.0006740, 0.0008001
1: 0.0023486, 0.0025313, 0.0023257, 0.0025349, -0.0000974, 0.0001156
2: 0.0097332, 0.0104323, 0.0097194, 0.0105200, -0.0004424, 0.0003726
3: -0.0046140, -0.0038909, -0.0046282, -0.0038002, -0.0004575, 0.0003854
4: 0.0001751, 0.0009579, 0.0000770, 0.0009733, -0.0004172, 0.0004953
5: 0.0032068, 0.0039476, 0.0031922, 0.0040405, -0.0004687, 0.0003948
6: -0.0095766, -0.0066373, -0.0096345, -0.0062688, -0.0018597, 0.0015665
7: 0.0064828, 0.0104858, 0.0059808, 0.0105646, -0.0021334, 0.0025327
8: 0.9937805, 0.9966003, 0.9934269, 0.9966558, -0.0015028, 0.0017841
9: -0.0128013, -0.0102416, -0.0128516, -0.0099207, -0.0016195, 0.0013641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007290, upper bound: 0.0008138
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007851, upper bound: 0.0008317
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070459, 0.0084088, 0.0069453, 0.0083934, -0.0006575, 0.0007669
1: 0.0023402, 0.0025371, 0.0023257, 0.0025349, -0.0000950, 0.0001108
2: 0.0097109, 0.0104643, 0.0097194, 0.0105200, -0.0004240, 0.0003635
3: -0.0046370, -0.0038577, -0.0046282, -0.0038002, -0.0004385, 0.0003759
4: 0.0001393, 0.0009829, 0.0000770, 0.0009733, -0.0004070, 0.0004747
5: 0.0031832, 0.0039816, 0.0031922, 0.0040405, -0.0004493, 0.0003851
6: -0.0096703, -0.0065027, -0.0096345, -0.0062688, -0.0017826, 0.0015281
7: 0.0062994, 0.0106134, 0.0059808, 0.0105646, -0.0020812, 0.0024277
8: 0.9936513, 0.9966902, 0.9934269, 0.9966558, -0.0014660, 0.0017101
9: -0.0128829, -0.0101244, -0.0128516, -0.0099207, -0.0015523, 0.0013308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007662, upper bound: 0.0008177
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008328, upper bound: 0.0008386
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0071706, 0.0084066, -0.0008845, 0.0006521
1: 0.0023257, 0.0025349, 0.0023583, 0.0025368, -0.0001278, 0.0000942
2: 0.0097194, 0.0105200, 0.0097121, 0.0103954, -0.0003605, 0.0004890
3: -0.0046282, -0.0038002, -0.0046358, -0.0039290, -0.0003729, 0.0005058
4: 0.0000770, 0.0009733, 0.0002165, 0.0009815, -0.0005475, 0.0004037
5: 0.0031922, 0.0040405, 0.0031845, 0.0039085, -0.0003820, 0.0005181
6: -0.0096345, -0.0062688, -0.0096652, -0.0067925, -0.0015156, 0.0020558
7: 0.0059808, 0.0105646, 0.0066941, 0.0106064, -0.0027999, 0.0020642
8: 0.9934269, 0.9966558, 0.9939294, 0.9966853, -0.0019723, 0.0014540
9: -0.0128516, -0.0099207, -0.0128784, -0.0103768, -0.0013199, 0.0017903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007866, upper bound: 0.0007579
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008502, upper bound: 0.0007789
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0071039, 0.0083685, -0.0008001, 0.0006740
1: 0.0023257, 0.0025349, 0.0023486, 0.0025313, -0.0001156, 0.0000974
2: 0.0097194, 0.0105200, 0.0097332, 0.0104323, -0.0003726, 0.0004424
3: -0.0046282, -0.0038002, -0.0046140, -0.0038909, -0.0003854, 0.0004575
4: 0.0000770, 0.0009733, 0.0001751, 0.0009579, -0.0004953, 0.0004172
5: 0.0031922, 0.0040405, 0.0032068, 0.0039476, -0.0003948, 0.0004687
6: -0.0096345, -0.0062688, -0.0095766, -0.0066373, -0.0015665, 0.0018597
7: 0.0059808, 0.0105646, 0.0064828, 0.0104858, -0.0025327, 0.0021334
8: 0.9934269, 0.9966558, 0.9937805, 0.9966003, -0.0017841, 0.0015028
9: -0.0128516, -0.0099207, -0.0128013, -0.0102416, -0.0013641, 0.0016195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007866, upper bound: 0.0007890
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008502, upper bound: 0.0008073
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0070678, 0.0083946, -0.0007042, 0.0005820
1: 0.0023257, 0.0025349, 0.0023434, 0.0025351, -0.0001017, 0.0000841
2: 0.0097194, 0.0105200, 0.0097187, 0.0104523, -0.0003218, 0.0003893
3: -0.0046282, -0.0038002, -0.0046289, -0.0038702, -0.0003328, 0.0004027
4: 0.0000770, 0.0009733, 0.0001528, 0.0009741, -0.0004359, 0.0003603
5: 0.0031922, 0.0040405, 0.0031915, 0.0039688, -0.0003409, 0.0004125
6: -0.0096345, -0.0062688, -0.0096373, -0.0065534, -0.0013527, 0.0016367
7: 0.0059808, 0.0105646, 0.0063685, 0.0105684, -0.0022291, 0.0018423
8: 0.9934269, 0.9966558, 0.9937000, 0.9966585, -0.0015702, 0.0012978
9: -0.0128516, -0.0099207, -0.0128541, -0.0101685, -0.0011780, 0.0014253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007908, upper bound: 0.0007609
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008556, upper bound: 0.0007837
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0069866, 0.0083641, -0.0006217, 0.0006087
1: 0.0023257, 0.0025349, 0.0023317, 0.0025307, -0.0000898, 0.0000879
2: 0.0097194, 0.0105200, 0.0097356, 0.0104972, -0.0003365, 0.0003437
3: -0.0046282, -0.0038002, -0.0046115, -0.0038238, -0.0003481, 0.0003555
4: 0.0000770, 0.0009733, 0.0001025, 0.0009552, -0.0003848, 0.0003768
5: 0.0031922, 0.0040405, 0.0032094, 0.0040163, -0.0003566, 0.0003642
6: -0.0096345, -0.0062688, -0.0095665, -0.0063647, -0.0014148, 0.0014450
7: 0.0059808, 0.0105646, 0.0061114, 0.0104720, -0.0019680, 0.0019269
8: 0.9934269, 0.9966558, 0.9935189, 0.9965905, -0.0013863, 0.0013573
9: -0.0128516, -0.0099207, -0.0127924, -0.0100042, -0.0012321, 0.0012584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007908, upper bound: 0.0007898
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008556, upper bound: 0.0008130
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069866, 0.0083641, 0.0070459, 0.0084088, -0.0008245, 0.0007263
1: 0.0023317, 0.0025307, 0.0023402, 0.0025371, -0.0001191, 0.0001049
2: 0.0097356, 0.0104972, 0.0097109, 0.0104643, -0.0004015, 0.0004558
3: -0.0046115, -0.0038238, -0.0046370, -0.0038577, -0.0004153, 0.0004715
4: 0.0001025, 0.0009552, 0.0001393, 0.0009829, -0.0005104, 0.0004496
5: 0.0032094, 0.0040163, 0.0031832, 0.0039816, -0.0004255, 0.0004830
6: -0.0095665, -0.0063647, -0.0096703, -0.0065027, -0.0016881, 0.0019164
7: 0.0061114, 0.0104720, 0.0062994, 0.0106134, -0.0026099, 0.0022990
8: 0.9935189, 0.9965905, 0.9936513, 0.9966902, -0.0018385, 0.0016195
9: -0.0127924, -0.0100042, -0.0128829, -0.0101244, -0.0014701, 0.0016689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007597, upper bound: 0.0008096
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008185, upper bound: 0.0008276
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069866, 0.0083641, 0.0069453, 0.0083934, -0.0006087, 0.0006217
1: 0.0023317, 0.0025307, 0.0023257, 0.0025349, -0.0000879, 0.0000898
2: 0.0097356, 0.0104972, 0.0097194, 0.0105200, -0.0003437, 0.0003365
3: -0.0046115, -0.0038238, -0.0046282, -0.0038002, -0.0003555, 0.0003481
4: 0.0001025, 0.0009552, 0.0000770, 0.0009733, -0.0003768, 0.0003848
5: 0.0032094, 0.0040163, 0.0031922, 0.0040405, -0.0003642, 0.0003566
6: -0.0095665, -0.0063647, -0.0096345, -0.0062688, -0.0014450, 0.0014148
7: 0.0061114, 0.0104720, 0.0059808, 0.0105646, -0.0019269, 0.0019680
8: 0.9935189, 0.9965905, 0.9934269, 0.9966558, -0.0013573, 0.0013863
9: -0.0127924, -0.0100042, -0.0128516, -0.0099207, -0.0012584, 0.0012321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007628, upper bound: 0.0008105
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008225, upper bound: 0.0008344
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0071418, 0.0084253, -0.0008444, 0.0006214
1: 0.0023257, 0.0025349, 0.0023541, 0.0025395, -0.0001220, 0.0000898
2: 0.0097194, 0.0105200, 0.0097017, 0.0104113, -0.0003436, 0.0004669
3: -0.0046282, -0.0038002, -0.0046465, -0.0039126, -0.0003553, 0.0004828
4: 0.0000770, 0.0009733, 0.0001986, 0.0009931, -0.0005227, 0.0003847
5: 0.0031922, 0.0040405, 0.0031735, 0.0039254, -0.0003640, 0.0004947
6: -0.0096345, -0.0062688, -0.0097087, -0.0067255, -0.0014443, 0.0019627
7: 0.0059808, 0.0105646, 0.0066028, 0.0106656, -0.0026730, 0.0019670
8: 0.9934269, 0.9966558, 0.9938651, 0.9967269, -0.0018829, 0.0013856
9: -0.0128516, -0.0099207, -0.0129162, -0.0103184, -0.0012578, 0.0017092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007870, upper bound: 0.0007585
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008504, upper bound: 0.0007792
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0070459, 0.0084088, -0.0007669, 0.0006575
1: 0.0023257, 0.0025349, 0.0023402, 0.0025371, -0.0001108, 0.0000950
2: 0.0097194, 0.0105200, 0.0097109, 0.0104643, -0.0003635, 0.0004240
3: -0.0046282, -0.0038002, -0.0046370, -0.0038577, -0.0003759, 0.0004385
4: 0.0000770, 0.0009733, 0.0001393, 0.0009829, -0.0004747, 0.0004070
5: 0.0031922, 0.0040405, 0.0031832, 0.0039816, -0.0003851, 0.0004493
6: -0.0096345, -0.0062688, -0.0096703, -0.0065027, -0.0015281, 0.0017826
7: 0.0059808, 0.0105646, 0.0062994, 0.0106134, -0.0024277, 0.0020812
8: 0.9934269, 0.9966558, 0.9936513, 0.9966902, -0.0017101, 0.0014660
9: -0.0128516, -0.0099207, -0.0128829, -0.0101244, -0.0013308, 0.0015523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007870, upper bound: 0.0008006
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008504, upper bound: 0.0008216
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0070482, 0.0084072, -0.0006259, 0.0005170
1: 0.0023257, 0.0025349, 0.0023406, 0.0025369, -0.0000904, 0.0000747
2: 0.0097194, 0.0105200, 0.0097117, 0.0104631, -0.0002859, 0.0003461
3: -0.0046282, -0.0038002, -0.0046361, -0.0038590, -0.0002956, 0.0003579
4: 0.0000770, 0.0009733, 0.0001406, 0.0009819, -0.0003875, 0.0003201
5: 0.0031922, 0.0040405, 0.0031841, 0.0039802, -0.0003029, 0.0003667
6: -0.0096345, -0.0062688, -0.0096667, -0.0065079, -0.0012017, 0.0014548
7: 0.0059808, 0.0105646, 0.0063064, 0.0106085, -0.0019813, 0.0016367
8: 0.9934269, 0.9966558, 0.9936563, 0.9966867, -0.0013957, 0.0011529
9: -0.0128516, -0.0099207, -0.0128797, -0.0101289, -0.0010465, 0.0012669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007950, upper bound: 0.0007662
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008577, upper bound: 0.0007866
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069453, 0.0083934, 0.0069453, 0.0083934, -0.0005478, 0.0005478
1: 0.0023257, 0.0025349, 0.0023257, 0.0025349, -0.0000791, 0.0000791
2: 0.0097194, 0.0105200, 0.0097194, 0.0105200, -0.0003029, 0.0003029
3: -0.0046282, -0.0038002, -0.0046282, -0.0038002, -0.0003132, 0.0003132
4: 0.0000770, 0.0009733, 0.0000770, 0.0009733, -0.0003391, 0.0003391
5: 0.0031922, 0.0040405, 0.0031922, 0.0040405, -0.0003209, 0.0003209
6: -0.0096345, -0.0062688, -0.0096345, -0.0062688, -0.0012733, 0.0012733
7: 0.0059808, 0.0105646, 0.0059808, 0.0105646, -0.0017341, 0.0017341
8: 0.9934269, 0.9966558, 0.9934269, 0.9966558, -0.0012215, 0.0012215
9: -0.0128516, -0.0099207, -0.0128516, -0.0099207, -0.0011088, 0.0011088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007950, upper bound: 0.0008040
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008577, upper bound: 0.0008261
time: 0.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.44 seconds
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007290, upper bound: 0.0008096
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007851, upper bound: 0.0008276
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007290, upper bound: 0.0008138
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007851, upper bound: 0.0008317
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007662, upper bound: 0.0008177
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008328, upper bound: 0.0008386
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007866, upper bound: 0.0007579
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008502, upper bound: 0.0007789
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007866, upper bound: 0.0007890
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008502, upper bound: 0.0008073
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007908, upper bound: 0.0007609
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008556, upper bound: 0.0007837
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007908, upper bound: 0.0007898
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008556, upper bound: 0.0008130
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007597, upper bound: 0.0008096
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008185, upper bound: 0.0008276
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007628, upper bound: 0.0008105
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008225, upper bound: 0.0008344
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007870, upper bound: 0.0007585
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008504, upper bound: 0.0007792
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007870, upper bound: 0.0008006
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008504, upper bound: 0.0008216
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007950, upper bound: 0.0007662
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008577, upper bound: 0.0007866
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0007950, upper bound: 0.0008040
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0008577, upper bound: 0.0008261

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.40 + 174.70 = 178.10 seconds
