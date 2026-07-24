## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00710256


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0037822, 0.0086200, 0.0037822, 0.0086200, -0.0048377, 0.0048377)
1: (-0.0002319, 0.0047782, -0.0002319, 0.0047782, -0.0050101, 0.0050101)
2: (-0.0258005, -0.0055088, -0.0258005, -0.0055088, -0.0202917, 0.0202917)
3: (-0.0020206, 0.0080132, -0.0020206, 0.0080132, -0.0100337, 0.0100337)
4: (0.0115328, 0.0185113, 0.0115328, 0.0185113, -0.0069784, 0.0069784)
5: (-0.0032831, 0.0100350, -0.0032831, 0.0100350, -0.0133181, 0.0133181)
6: (0.9943639, 1.0042140, 0.9943639, 1.0042140, -0.0098501, 0.0098501)
7: (0.0074935, 0.0201257, 0.0074935, 0.0201257, -0.0093597, 0.0093597)
8: (0.0018148, 0.0072936, 0.0018148, 0.0072936, -0.0054788, 0.0054788)
9: (-0.0276323, -0.0139874, -0.0276323, -0.0139874, -0.0136449, 0.0136449)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.51 + 2.65 = 4.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0086858, upper bound: 0.0086858

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0085672, upper bound: 0.0084670
time: 1.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0085943, upper bound: 0.0085943
time: 1.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.79
Output dim: 6, lower bound: -0.0085672, upper bound: 0.0084670
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.79
Output dim: 6, lower bound: -0.0085943, upper bound: 0.0085943

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0040361, 0.0085966, 0.0038729, 0.0086116, -0.0045756, 0.0047237
1: 0.0000099, 0.0047329, -0.0001455, 0.0047620, -0.0047521, 0.0048785
2: -0.0251953, -0.0058024, -0.0255844, -0.0055544, -0.0196409, 0.0197819
3: -0.0019943, 0.0074880, -0.0020165, 0.0078256, -0.0098200, 0.0095045
4: 0.0116601, 0.0183530, 0.0115526, 0.0184547, -0.0067947, 0.0068004
5: -0.0032595, 0.0092954, -0.0032746, 0.0097708, -0.0130303, 0.0125701
6: 0.9943987, 1.0037158, 0.9943693, 1.0040359, -0.0096372, 0.0093465
7: 0.0077238, 0.0198392, 0.0075293, 0.0200234, -0.0090596, 0.0090499
8: 0.0020390, 0.0072038, 0.0018949, 0.0072615, -0.0052225, 0.0053089
9: -0.0271239, -0.0141314, -0.0274507, -0.0140098, -0.0131141, 0.0133193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0085056, upper bound: 0.0084193
time: 1.48 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0085001, upper bound: 0.0083993
time: 1.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0039239, 0.0086069, 0.0038370, 0.0086149, -0.0046910, 0.0047699
1: -0.0000970, 0.0047529, -0.0001797, 0.0047684, -0.0048654, 0.0049327
2: -0.0254628, -0.0055780, -0.0256699, -0.0055358, -0.0199270, 0.0200919
3: -0.0020144, 0.0077202, -0.0020181, 0.0078999, -0.0099143, 0.0097383
4: 0.0115628, 0.0184229, 0.0115445, 0.0184771, -0.0069143, 0.0068784
5: -0.0032699, 0.0096223, -0.0032780, 0.0098755, -0.0131454, 0.0129003
6: 0.9943721, 1.0039359, 0.9943671, 1.0041065, -0.0097345, 0.0095689
7: 0.0075478, 0.0199659, 0.0075147, 0.0200639, -0.0092505, 0.0091040
8: 0.0019399, 0.0072435, 0.0018632, 0.0072742, -0.0053343, 0.0053803
9: -0.0273486, -0.0140213, -0.0275226, -0.0140006, -0.0133480, 0.0135013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0085330, upper bound: 0.0085402
time: 2.36 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0085269, upper bound: 0.0085269
time: 1.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.55 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 6, lower bound: -0.0085056, upper bound: 0.0084193
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 6, lower bound: -0.0085001, upper bound: 0.0083993
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 6, lower bound: -0.0085330, upper bound: 0.0085402
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 6, lower bound: -0.0085269, upper bound: 0.0085269

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0040493, 0.0085954, 0.0038774, 0.0086112, -0.0045619, 0.0047180
1: 0.0000225, 0.0047306, -0.0001412, 0.0047612, -0.0047387, 0.0048718
2: -0.0251638, -0.0060934, -0.0255737, -0.0056425, -0.0195213, 0.0194803
3: -0.0019683, 0.0074607, -0.0020086, 0.0078163, -0.0097847, 0.0094693
4: 0.0117861, 0.0183447, 0.0115908, 0.0184519, -0.0066658, 0.0067540
5: -0.0032582, 0.0092569, -0.0032742, 0.0097577, -0.0130160, 0.0125311
6: 0.9944332, 1.0036898, 0.9943797, 1.0040272, -0.0095940, 0.0093101
7: 0.0079521, 0.0198243, 0.0075984, 0.0200183, -0.0088262, 0.0088836
8: 0.0020507, 0.0071992, 0.0018988, 0.0072599, -0.0052093, 0.0053003
9: -0.0270974, -0.0142741, -0.0274417, -0.0140530, -0.0130444, 0.0131676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077895, upper bound: 0.0078036
time: 1.40 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077297, upper bound: 0.0076606
time: 1.40 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0039160, 0.0086077, 0.0038824, 0.0086108, -0.0046948, 0.0047252
1: -0.0001045, 0.0047544, -0.0001365, 0.0047603, -0.0048649, 0.0048908
2: -0.0254817, -0.0062070, -0.0255617, -0.0057540, -0.0197277, 0.0193547
3: -0.0019582, 0.0077365, -0.0019986, 0.0078060, -0.0097641, 0.0097352
4: 0.0118354, 0.0184279, 0.0116391, 0.0184488, -0.0066134, 0.0067888
5: -0.0032706, 0.0096454, -0.0032738, 0.0097432, -0.0130138, 0.0129192
6: 0.9944467, 1.0039515, 0.9943930, 1.0040174, -0.0095707, 0.0095586
7: 0.0080412, 0.0199748, 0.0076859, 0.0200127, -0.0086919, 0.0089997
8: 0.0019329, 0.0072463, 0.0019033, 0.0072582, -0.0053253, 0.0053430
9: -0.0273645, -0.0143298, -0.0274317, -0.0141077, -0.0132568, 0.0131018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077832, upper bound: 0.0077779
time: 1.37 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077165, upper bound: 0.0075698
time: 1.25 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0039389, 0.0086056, 0.0038415, 0.0086145, -0.0046756, 0.0047641
1: -0.0000827, 0.0047503, -0.0001755, 0.0047677, -0.0048503, 0.0049258
2: -0.0254271, -0.0058675, -0.0256593, -0.0056239, -0.0198032, 0.0197918
3: -0.0019885, 0.0076892, -0.0020103, 0.0078907, -0.0098792, 0.0096994
4: 0.0116883, 0.0184136, 0.0115827, 0.0184743, -0.0067861, 0.0068309
5: -0.0032685, 0.0095787, -0.0032776, 0.0098624, -0.0131309, 0.0128562
6: 0.9944064, 1.0039067, 0.9943776, 1.0040977, -0.0096913, 0.0095292
7: 0.0077749, 0.0199490, 0.0075838, 0.0200589, -0.0090239, 0.0090163
8: 0.0019532, 0.0072382, 0.0018671, 0.0072727, -0.0053195, 0.0053711
9: -0.0273186, -0.0141633, -0.0275137, -0.0140438, -0.0132748, 0.0133503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0080193, upper bound: 0.0079720
time: 1.74 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077808, upper bound: 0.0078993
time: 1.19 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0038105, 0.0086174, 0.0038465, 0.0086141, -0.0048036, 0.0047709
1: -0.0002050, 0.0047732, -0.0001707, 0.0047668, -0.0049717, 0.0049439
2: -0.0257331, -0.0059766, -0.0256474, -0.0057355, -0.0199977, 0.0196708
3: -0.0019788, 0.0079547, -0.0020003, 0.0078803, -0.0098591, 0.0099550
4: 0.0117355, 0.0184936, 0.0116310, 0.0184712, -0.0067357, 0.0068626
5: -0.0032805, 0.0099527, -0.0032771, 0.0098479, -0.0131283, 0.0132298
6: 0.9944193, 1.0041585, 0.9943908, 1.0040878, -0.0096685, 0.0097677
7: 0.0078605, 0.0200938, 0.0076713, 0.0200532, -0.0088901, 0.0091203
8: 0.0018398, 0.0072836, 0.0018715, 0.0072709, -0.0054311, 0.0054121
9: -0.0275757, -0.0142168, -0.0275036, -0.0140986, -0.0134771, 0.0132868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0078159, upper bound: 0.0079713
time: 1.39 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077694, upper bound: 0.0077694
time: 1.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.99 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0077895, upper bound: 0.0078036
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0077297, upper bound: 0.0076606
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0077832, upper bound: 0.0077779
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0077165, upper bound: 0.0075698
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0080193, upper bound: 0.0079720
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0077808, upper bound: 0.0078993
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0078159, upper bound: 0.0079713
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 6, lower bound: -0.0077694, upper bound: 0.0077694

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0042070, 0.0085809, 0.0038774, 0.0086112, -0.0044042, 0.0047035
1: 0.0001727, 0.0047025, -0.0001412, 0.0047612, -0.0045885, 0.0048437
2: -0.0247879, -0.0061313, -0.0255737, -0.0056425, -0.0191453, 0.0194423
3: -0.0019650, 0.0071345, -0.0020086, 0.0078163, -0.0097813, 0.0091431
4: 0.0118026, 0.0182464, 0.0115908, 0.0184519, -0.0066494, 0.0066557
5: -0.0032435, 0.0087975, -0.0032742, 0.0097577, -0.0130013, 0.0120717
6: 0.9944377, 1.0033802, 0.9943797, 1.0040272, -0.0095896, 0.0090005
7: 0.0079818, 0.0196463, 0.0075984, 0.0200183, -0.0087937, 0.0087291
8: 0.0021900, 0.0071434, 0.0018988, 0.0072599, -0.0050700, 0.0052446
9: -0.0267816, -0.0142927, -0.0274417, -0.0140530, -0.0127286, 0.0131490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076274, upper bound: 0.0076572
time: 1.23 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077018, upper bound: 0.0076976
time: 1.44 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0045630, 0.0085481, 0.0040613, 0.0085943, -0.0040313, 0.0044868
1: 0.0005119, 0.0046390, 0.0000339, 0.0047284, -0.0042166, 0.0046050
2: -0.0239391, -0.0051325, -0.0251353, -0.0056994, -0.0182397, 0.0200028
3: -0.0020542, 0.0063979, -0.0020035, 0.0074360, -0.0094901, 0.0084015
4: 0.0113697, 0.0180245, 0.0116154, 0.0183373, -0.0069676, 0.0064091
5: -0.0032104, 0.0077604, -0.0032571, 0.0092221, -0.0124325, 0.0110175
6: 0.9943192, 1.0026814, 0.9943866, 1.0036664, -0.0093472, 0.0082948
7: 0.0071983, 0.0192446, 0.0076430, 0.0198108, -0.0092562, 0.0085157
8: 0.0025043, 0.0070175, 0.0020612, 0.0071949, -0.0046906, 0.0049563
9: -0.0260686, -0.0138028, -0.0270734, -0.0140809, -0.0119877, 0.0132706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075701, upper bound: 0.0075196
time: 1.36 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076423, upper bound: 0.0075360
time: 1.37 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0040757, 0.0085930, 0.0038824, 0.0086108, -0.0045351, 0.0047105
1: 0.0000476, 0.0047259, -0.0001365, 0.0047603, -0.0047128, 0.0048624
2: -0.0251010, -0.0062445, -0.0255617, -0.0057540, -0.0193470, 0.0193172
3: -0.0019548, 0.0074062, -0.0019986, 0.0078060, -0.0097608, 0.0094049
4: 0.0118516, 0.0183283, 0.0116391, 0.0184488, -0.0065972, 0.0066893
5: -0.0032558, 0.0091802, -0.0032738, 0.0097432, -0.0129989, 0.0124540
6: 0.9944512, 1.0036381, 0.9943930, 1.0040174, -0.0095662, 0.0092452
7: 0.0080706, 0.0197946, 0.0076859, 0.0200127, -0.0086597, 0.0088339
8: 0.0020740, 0.0071899, 0.0019033, 0.0072582, -0.0051842, 0.0052866
9: -0.0270447, -0.0143482, -0.0274317, -0.0141077, -0.0129370, 0.0130834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076842, upper bound: 0.0076142
time: 1.57 seconds

## Relational analysis of IS_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076953, upper bound: 0.0076917
time: 1.42 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0044029, 0.0085628, 0.0040664, 0.0085938, -0.0041909, 0.0044964
1: 0.0003593, 0.0046675, 0.0000388, 0.0047275, -0.0043682, 0.0046287
2: -0.0243209, -0.0052097, -0.0251231, -0.0058107, -0.0185101, 0.0199134
3: -0.0020473, 0.0067292, -0.0019936, 0.0074254, -0.0094727, 0.0087228
4: 0.0114032, 0.0181243, 0.0116636, 0.0183341, -0.0069309, 0.0064607
5: -0.0032253, 0.0082269, -0.0032566, 0.0092072, -0.0124325, 0.0114835
6: 0.9943284, 1.0029956, 0.9943997, 1.0036563, -0.0093279, 0.0085959
7: 0.0072589, 0.0194253, 0.0077303, 0.0198050, -0.0092523, 0.0086514
8: 0.0023630, 0.0070742, 0.0020658, 0.0071931, -0.0048302, 0.0050084
9: -0.0263893, -0.0138406, -0.0270632, -0.0141355, -0.0122538, 0.0132226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075616, upper bound: 0.0074621
time: 1.51 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076290, upper bound: 0.0074740
time: 1.38 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0039389, 0.0086056, 0.0040051, 0.0085995, -0.0046606, 0.0046004
1: -0.0000827, 0.0047503, -0.0000196, 0.0047385, -0.0048211, 0.0047699
2: -0.0254271, -0.0058675, -0.0252692, -0.0056617, -0.0197654, 0.0194017
3: -0.0019885, 0.0076892, -0.0020069, 0.0075521, -0.0095406, 0.0096961
4: 0.0116883, 0.0184136, 0.0115991, 0.0183723, -0.0066840, 0.0068145
5: -0.0032685, 0.0095787, -0.0032623, 0.0093857, -0.0126542, 0.0128410
6: 0.9944064, 1.0039067, 0.9943820, 1.0037766, -0.0093701, 0.0095248
7: 0.0077749, 0.0199490, 0.0076135, 0.0198742, -0.0088615, 0.0089838
8: 0.0019532, 0.0072382, 0.0020117, 0.0072148, -0.0052616, 0.0052265
9: -0.0273186, -0.0141633, -0.0271859, -0.0140624, -0.0132562, 0.0130226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079134, upper bound: 0.0078157
time: 1.27 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079359, upper bound: 0.0078634
time: 1.94 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041227, 0.0085886, 0.0043567, 0.0085671, -0.0044444, 0.0042319
1: 0.0000924, 0.0047175, 0.0003153, 0.0046758, -0.0045834, 0.0044022
2: -0.0249889, -0.0059216, -0.0244310, -0.0046284, -0.0203605, 0.0185094
3: -0.0019837, 0.0073089, -0.0020992, 0.0068248, -0.0088084, 0.0094081
4: 0.0117117, 0.0182990, 0.0111513, 0.0181531, -0.0064414, 0.0071477
5: -0.0032514, 0.0090432, -0.0032296, 0.0083614, -0.0116128, 0.0122728
6: 0.9944128, 1.0035458, 0.9942594, 1.0030863, -0.0086735, 0.0092863
7: 0.0078173, 0.0197415, 0.0068029, 0.0194774, -0.0084562, 0.0094909
8: 0.0021155, 0.0071732, 0.0023221, 0.0070905, -0.0049750, 0.0048511
9: -0.0269505, -0.0141899, -0.0264818, -0.0135555, -0.0133949, 0.0122919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076877, upper bound: 0.0077440
time: 1.40 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076922, upper bound: 0.0077915
time: 1.29 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0039777, 0.0086020, 0.0038465, 0.0086141, -0.0046364, 0.0047555
1: -0.0000457, 0.0047434, -0.0001707, 0.0047668, -0.0048125, 0.0049141
2: -0.0253346, -0.0060131, -0.0256474, -0.0057355, -0.0195991, 0.0196342
3: -0.0019755, 0.0076089, -0.0020003, 0.0078803, -0.0098558, 0.0096092
4: 0.0117514, 0.0183894, 0.0116310, 0.0184712, -0.0067198, 0.0067584
5: -0.0032649, 0.0094656, -0.0032771, 0.0098479, -0.0131128, 0.0127427
6: 0.9944238, 1.0038304, 0.9943908, 1.0040878, -0.0096640, 0.0094396
7: 0.0078891, 0.0199051, 0.0076713, 0.0200532, -0.0088584, 0.0088753
8: 0.0019874, 0.0072245, 0.0018715, 0.0072709, -0.0052834, 0.0053530
9: -0.0272408, -0.0142347, -0.0275036, -0.0140986, -0.0131423, 0.0132689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077208, upper bound: 0.0078084
time: 1.35 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077275, upper bound: 0.0078868
time: 1.43 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0042766, 0.0085745, 0.0040312, 0.0085971, -0.0043204, 0.0045433
1: 0.0002391, 0.0046900, 0.0000052, 0.0047338, -0.0044948, 0.0046848
2: -0.0246219, -0.0049263, -0.0252071, -0.0057916, -0.0188302, 0.0202808
3: -0.0020726, 0.0069904, -0.0019953, 0.0074982, -0.0095708, 0.0089857
4: 0.0112804, 0.0182030, 0.0116554, 0.0183561, -0.0070757, 0.0065476
5: -0.0032371, 0.0085947, -0.0032599, 0.0093098, -0.0125469, 0.0118546
6: 0.9942947, 1.0032434, 0.9943974, 1.0037255, -0.0094308, 0.0088460
7: 0.0070366, 0.0195678, 0.0077154, 0.0198448, -0.0094950, 0.0087030
8: 0.0022514, 0.0071188, 0.0020347, 0.0072056, -0.0049541, 0.0050841
9: -0.0266421, -0.0137017, -0.0271337, -0.0141261, -0.0125160, 0.0134321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076069, upper bound: 0.0076690
time: 1.42 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076803, upper bound: 0.0076803
time: 1.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.44 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076274, upper bound: 0.0076572
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0077018, upper bound: 0.0076976
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0075701, upper bound: 0.0075196
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076423, upper bound: 0.0075360
IS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076842, upper bound: 0.0076142
IS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076953, upper bound: 0.0076917
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0075616, upper bound: 0.0074621
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076290, upper bound: 0.0074740
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0079134, upper bound: 0.0078157
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0079359, upper bound: 0.0078634
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076877, upper bound: 0.0077440
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076922, upper bound: 0.0077915
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0077208, upper bound: 0.0078084
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0077275, upper bound: 0.0078868
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076069, upper bound: 0.0076690
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 6, lower bound: -0.0076803, upper bound: 0.0076803

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042634, 0.0085757, 0.0040255, 0.0085976, -0.0043342, 0.0045501
1: 0.0002265, 0.0046924, -0.0000001, 0.0047348, -0.0045083, 0.0046925
2: -0.0246534, -0.0061823, -0.0252205, -0.0057727, -0.0188807, 0.0190382
3: -0.0019604, 0.0070178, -0.0019970, 0.0075099, -0.0094703, 0.0090147
4: 0.0118247, 0.0182113, 0.0116471, 0.0183596, -0.0065349, 0.0065641
5: -0.0032383, 0.0086332, -0.0032604, 0.0093262, -0.0125645, 0.0118936
6: 0.9944438, 1.0032694, 0.9943951, 1.0037365, -0.0092927, 0.0088743
7: 0.0080219, 0.0195827, 0.0077005, 0.0198512, -0.0086022, 0.0085586
8: 0.0022398, 0.0071235, 0.0020296, 0.0072076, -0.0049678, 0.0050938
9: -0.0266686, -0.0143177, -0.0271450, -0.0141168, -0.0125518, 0.0128273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076274, upper bound: 0.0076572
time: 1.57 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076274, upper bound: 0.0076572
time: 1.35 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0042675, 0.0085753, 0.0040134, 0.0085987, -0.0043312, 0.0045619
1: 0.0002304, 0.0046917, -0.0000117, 0.0047370, -0.0045066, 0.0047034
2: -0.0246435, -0.0061910, -0.0252495, -0.0055935, -0.0190500, 0.0190585
3: -0.0019596, 0.0070092, -0.0020130, 0.0075351, -0.0094947, 0.0090222
4: 0.0118284, 0.0182087, 0.0115695, 0.0183672, -0.0065387, 0.0066392
5: -0.0032379, 0.0086212, -0.0032616, 0.0093617, -0.0125996, 0.0118828
6: 0.9944447, 1.0032614, 0.9943739, 1.0037603, -0.0093156, 0.0088875
7: 0.0080287, 0.0195780, 0.0075600, 0.0198649, -0.0086140, 0.0086978
8: 0.0022434, 0.0071220, 0.0020189, 0.0072119, -0.0049685, 0.0051031
9: -0.0266604, -0.0143220, -0.0271694, -0.0140289, -0.0126314, 0.0128474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077018, upper bound: 0.0076976
time: 1.40 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077018, upper bound: 0.0076976
time: 1.46 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0046110, 0.0085437, 0.0042038, 0.0085812, -0.0039702, 0.0043398
1: 0.0005576, 0.0046304, 0.0001697, 0.0047030, -0.0041454, 0.0044607
2: -0.0238247, -0.0051817, -0.0247954, -0.0058300, -0.0179946, 0.0196137
3: -0.0020498, 0.0062987, -0.0019919, 0.0071410, -0.0091907, 0.0082905
4: 0.0113911, 0.0179946, 0.0116720, 0.0182484, -0.0068573, 0.0063225
5: -0.0032059, 0.0076205, -0.0032438, 0.0088067, -0.0120126, 0.0108644
6: 0.9943250, 1.0025873, 0.9944020, 1.0033863, -0.0090612, 0.0081853
7: 0.0072369, 0.0191904, 0.0077455, 0.0196499, -0.0090728, 0.0083521
8: 0.0025468, 0.0070006, 0.0021872, 0.0071445, -0.0045978, 0.0048134
9: -0.0259724, -0.0138269, -0.0267878, -0.0141449, -0.0118275, 0.0129609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071407, upper bound: 0.0070849
time: 1.23 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071192, upper bound: 0.0070370
time: 1.38 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0046162, 0.0085432, 0.0041912, 0.0085823, -0.0039662, 0.0043520
1: 0.0005625, 0.0046295, 0.0001577, 0.0047053, -0.0041428, 0.0044718
2: -0.0238124, -0.0051909, -0.0248256, -0.0056502, -0.0181622, 0.0196346
3: -0.0020489, 0.0062880, -0.0020079, 0.0071672, -0.0092161, 0.0082959
4: 0.0113951, 0.0179913, 0.0115941, 0.0182563, -0.0068612, 0.0063972
5: -0.0032055, 0.0076055, -0.0032450, 0.0088436, -0.0120490, 0.0108505
6: 0.9943261, 1.0025772, 0.9943807, 1.0034113, -0.0090852, 0.0081965
7: 0.0072442, 0.0191846, 0.0076045, 0.0196642, -0.0090821, 0.0084870
8: 0.0025513, 0.0069987, 0.0021760, 0.0071490, -0.0045977, 0.0048227
9: -0.0259621, -0.0138315, -0.0268132, -0.0140568, -0.0119053, 0.0129818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070860
time: 1.41 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071681, upper bound: 0.0070391
time: 1.15 seconds

## BFS IS instance: IS_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0042032, 0.0085812, 0.0039396, 0.0086055, -0.0044022, 0.0046416
1: 0.0001692, 0.0047031, -0.0000820, 0.0047501, -0.0045810, 0.0047851
2: -0.0247968, -0.0063760, -0.0254253, -0.0058060, -0.0189909, 0.0190493
3: -0.0019431, 0.0071422, -0.0019940, 0.0076876, -0.0096307, 0.0091362
4: 0.0119086, 0.0182488, 0.0116616, 0.0184131, -0.0065045, 0.0065872
5: -0.0032439, 0.0088084, -0.0032684, 0.0095765, -0.0128204, 0.0120769
6: 0.9944667, 1.0033875, 0.9943991, 1.0039051, -0.0094383, 0.0089883
7: 0.0081738, 0.0196506, 0.0077266, 0.0199481, -0.0084963, 0.0086457
8: 0.0021867, 0.0071447, 0.0019538, 0.0072379, -0.0050513, 0.0051909
9: -0.0267891, -0.0144127, -0.0273171, -0.0141331, -0.0126560, 0.0129043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A1_A1_B1

### Relational analysis result of IS_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076842, upper bound: 0.0076142
time: 1.52 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2

### Relational analysis result of IS_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076842, upper bound: 0.0076142
time: 1.57 seconds

## BFS IS instance: IS_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0042060, 0.0085810, 0.0039442, 0.0086051, -0.0043991, 0.0046367
1: 0.0001717, 0.0047026, -0.0000776, 0.0047493, -0.0045776, 0.0047802
2: -0.0247904, -0.0062304, -0.0254144, -0.0058112, -0.0189791, 0.0191839
3: -0.0019561, 0.0071366, -0.0019935, 0.0076781, -0.0096342, 0.0091302
4: 0.0118455, 0.0182471, 0.0116639, 0.0184103, -0.0065648, 0.0065832
5: -0.0032436, 0.0088006, -0.0032680, 0.0095631, -0.0128068, 0.0120686
6: 0.9944495, 1.0033823, 0.9943997, 1.0038961, -0.0094466, 0.0089826
7: 0.0080596, 0.0196475, 0.0077308, 0.0199429, -0.0086204, 0.0086640
8: 0.0021891, 0.0071438, 0.0019578, 0.0072363, -0.0050473, 0.0051860
9: -0.0267837, -0.0143413, -0.0273079, -0.0141357, -0.0126480, 0.0129666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A1_A2_B1

### Relational analysis result of IS_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076953, upper bound: 0.0076917
time: 1.47 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2

### Relational analysis result of IS_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076953, upper bound: 0.0076917
time: 1.37 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044489, 0.0085586, 0.0042078, 0.0085808, -0.0041319, 0.0043507
1: 0.0004031, 0.0046593, 0.0001736, 0.0047023, -0.0042992, 0.0044858
2: -0.0242112, -0.0052588, -0.0247858, -0.0059409, -0.0182703, 0.0195270
3: -0.0020429, 0.0066341, -0.0019820, 0.0071327, -0.0091756, 0.0086160
4: 0.0114245, 0.0180957, 0.0117200, 0.0182459, -0.0068214, 0.0063756
5: -0.0032210, 0.0080929, -0.0032435, 0.0087950, -0.0120161, 0.0113363
6: 0.9943342, 1.0029055, 0.9944152, 1.0033786, -0.0090444, 0.0084903
7: 0.0072974, 0.0193734, 0.0078324, 0.0196454, -0.0090701, 0.0084923
8: 0.0024036, 0.0070579, 0.0021907, 0.0071431, -0.0047395, 0.0048672
9: -0.0262972, -0.0138648, -0.0267799, -0.0141993, -0.0120979, 0.0129151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
time: 1.31 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071154, upper bound: 0.0069639
time: 1.40 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044593, 0.0085576, 0.0041953, 0.0085819, -0.0041226, 0.0043623
1: 0.0004131, 0.0046574, 0.0001616, 0.0047045, -0.0042914, 0.0044958
2: -0.0241862, -0.0052663, -0.0248157, -0.0057748, -0.0184114, 0.0195494
3: -0.0020422, 0.0066124, -0.0019968, 0.0071586, -0.0092008, 0.0086092
4: 0.0114277, 0.0180891, 0.0116481, 0.0182537, -0.0068260, 0.0064410
5: -0.0032201, 0.0080624, -0.0032446, 0.0088315, -0.0120516, 0.0113070
6: 0.9943351, 1.0028849, 0.9943954, 1.0034032, -0.0090681, 0.0084894
7: 0.0073033, 0.0193616, 0.0077022, 0.0196595, -0.0090762, 0.0086018
8: 0.0024128, 0.0070542, 0.0021796, 0.0071475, -0.0047347, 0.0048746
9: -0.0262762, -0.0138684, -0.0268049, -0.0141178, -0.0121583, 0.0129365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071855, upper bound: 0.0070025
time: 1.37 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071634, upper bound: 0.0069657
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0040817, 0.0085924, 0.0040614, 0.0085943, -0.0045126, 0.0045310
1: 0.0000533, 0.0047248, 0.0000340, 0.0047284, -0.0046751, 0.0046908
2: -0.0250867, -0.0059961, -0.0251351, -0.0057140, -0.0193727, 0.0191390
3: -0.0019770, 0.0073938, -0.0020022, 0.0074358, -0.0094128, 0.0093960
4: 0.0117440, 0.0183246, 0.0116217, 0.0183372, -0.0065933, 0.0067029
5: -0.0032552, 0.0091627, -0.0032571, 0.0092218, -0.0124771, 0.0124198
6: 0.9944217, 1.0036262, 0.9943882, 1.0036662, -0.0092444, 0.0092381
7: 0.0078758, 0.0197878, 0.0076545, 0.0198107, -0.0087016, 0.0087935
8: 0.0020793, 0.0071877, 0.0020613, 0.0071949, -0.0051156, 0.0051264
9: -0.0270326, -0.0142264, -0.0270733, -0.0140880, -0.0129446, 0.0128469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B1_A1_A1

### Relational analysis result of IS_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079134, upper bound: 0.0078157
time: 1.37 seconds

## Relational analysis of IS_A2_A1_B1_A1_A2

### Relational analysis result of IS_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079134, upper bound: 0.0078157
time: 1.46 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0040687, 0.0085936, 0.0040663, 0.0085938, -0.0045251, 0.0045273
1: 0.0000410, 0.0047271, 0.0000387, 0.0047276, -0.0046866, 0.0046885
2: -0.0251176, -0.0058031, -0.0251234, -0.0057196, -0.0193980, 0.0193203
3: -0.0019943, 0.0074206, -0.0020017, 0.0074256, -0.0094199, 0.0094223
4: 0.0116603, 0.0183327, 0.0116242, 0.0183342, -0.0066738, 0.0067085
5: -0.0032564, 0.0092005, -0.0032566, 0.0092076, -0.0124640, 0.0124571
6: 0.9943988, 1.0036517, 0.9943889, 1.0036565, -0.0092577, 0.0092628
7: 0.0077244, 0.0198024, 0.0076589, 0.0198052, -0.0088531, 0.0088111
8: 0.0020678, 0.0071923, 0.0020657, 0.0071932, -0.0051254, 0.0051267
9: -0.0270586, -0.0141317, -0.0270635, -0.0140908, -0.0129678, 0.0129317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079359, upper bound: 0.0078634
time: 1.31 seconds

## Relational analysis of IS_A2_A1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0079359, upper bound: 0.0078634
time: 1.83 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042575, 0.0085762, 0.0044072, 0.0085624, -0.0043049, 0.0041690
1: 0.0002209, 0.0046934, 0.0003635, 0.0046667, -0.0044459, 0.0043300
2: -0.0246674, -0.0060510, -0.0243105, -0.0046788, -0.0199886, 0.0182596
3: -0.0019721, 0.0070300, -0.0020947, 0.0067203, -0.0086924, 0.0091246
4: 0.0117678, 0.0182149, 0.0111731, 0.0181216, -0.0063539, 0.0070418
5: -0.0032388, 0.0086504, -0.0032249, 0.0082142, -0.0114531, 0.0118753
6: 0.9944282, 1.0032811, 0.9942654, 1.0029871, -0.0085589, 0.0090157
7: 0.0079188, 0.0195893, 0.0068425, 0.0194204, -0.0082989, 0.0093133
8: 0.0022346, 0.0071256, 0.0023668, 0.0070726, -0.0048381, 0.0047588
9: -0.0266804, -0.0142533, -0.0263806, -0.0135803, -0.0131001, 0.0121273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.27 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042451, 0.0085774, 0.0044120, 0.0085620, -0.0043169, 0.0041654
1: 0.0002090, 0.0046957, 0.0003680, 0.0046659, -0.0044569, 0.0043277
2: -0.0246970, -0.0058629, -0.0242992, -0.0046859, -0.0200111, 0.0184362
3: -0.0019889, 0.0070556, -0.0020940, 0.0067104, -0.0086993, 0.0091497
4: 0.0116863, 0.0182227, 0.0111762, 0.0181186, -0.0064324, 0.0070465
5: -0.0032400, 0.0086865, -0.0032245, 0.0082004, -0.0114404, 0.0119110
6: 0.9944059, 1.0033053, 0.9942662, 1.0029778, -0.0085719, 0.0090391
7: 0.0077713, 0.0196033, 0.0068480, 0.0194150, -0.0084448, 0.0093254
8: 0.0022236, 0.0071299, 0.0023710, 0.0070709, -0.0048473, 0.0047590
9: -0.0267052, -0.0141611, -0.0263710, -0.0135838, -0.0131215, 0.0122100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_A1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.27 seconds

## Relational analysis of IS_A2_A1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0074154
time: 1.37 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0041040, 0.0085904, 0.0039035, 0.0086088, -0.0045049, 0.0046868
1: 0.0000746, 0.0047208, -0.0001164, 0.0047566, -0.0046820, 0.0048372
2: -0.0250335, -0.0061442, -0.0255114, -0.0057872, -0.0192463, 0.0193672
3: -0.0019638, 0.0073476, -0.0019957, 0.0077623, -0.0097261, 0.0093433
4: 0.0118082, 0.0183107, 0.0116534, 0.0184357, -0.0066275, 0.0066572
5: -0.0032531, 0.0090977, -0.0032718, 0.0096817, -0.0129348, 0.0123695
6: 0.9944393, 1.0035825, 0.9943969, 1.0039760, -0.0095367, 0.0091856
7: 0.0079920, 0.0197626, 0.0077119, 0.0199889, -0.0086977, 0.0086917
8: 0.0020989, 0.0071798, 0.0019219, 0.0072507, -0.0051518, 0.0052579
9: -0.0269879, -0.0142991, -0.0273894, -0.0141239, -0.0128640, 0.0130904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077208, upper bound: 0.0078084
time: 1.69 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077208, upper bound: 0.0078084
time: 1.60 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0041065, 0.0085901, 0.0039082, 0.0086084, -0.0045019, 0.0046819
1: 0.0000769, 0.0047204, -0.0001119, 0.0047557, -0.0046788, 0.0048323
2: -0.0250276, -0.0059850, -0.0255002, -0.0057922, -0.0192354, 0.0195153
3: -0.0019780, 0.0073425, -0.0019952, 0.0077526, -0.0097306, 0.0093377
4: 0.0117391, 0.0183091, 0.0116556, 0.0184327, -0.0066936, 0.0066535
5: -0.0032529, 0.0090905, -0.0032714, 0.0096680, -0.0129209, 0.0123619
6: 0.9944204, 1.0035776, 0.9943975, 1.0039667, -0.0095463, 0.0091801
7: 0.0078670, 0.0197598, 0.0077159, 0.0199836, -0.0088335, 0.0087127
8: 0.0021011, 0.0071790, 0.0019261, 0.0072491, -0.0051479, 0.0052529
9: -0.0269830, -0.0142209, -0.0273800, -0.0141264, -0.0128566, 0.0131591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077275, upper bound: 0.0078867
time: 1.62 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0077275, upper bound: 0.0078867
time: 1.42 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043220, 0.0085703, 0.0041713, 0.0085842, -0.0042622, 0.0043989
1: 0.0002822, 0.0046819, 0.0001388, 0.0047088, -0.0044266, 0.0045432
2: -0.0245138, -0.0049751, -0.0248729, -0.0059213, -0.0185925, 0.0198977
3: -0.0020682, 0.0068966, -0.0019837, 0.0072082, -0.0092764, 0.0088803
4: 0.0113016, 0.0181748, 0.0117116, 0.0182687, -0.0069671, 0.0064632
5: -0.0032328, 0.0084626, -0.0032469, 0.0089014, -0.0121343, 0.0117095
6: 0.9943006, 1.0031545, 0.9944128, 1.0034502, -0.0091496, 0.0087417
7: 0.0070749, 0.0195166, 0.0078171, 0.0196866, -0.0093154, 0.0085437
8: 0.0022915, 0.0071028, 0.0021585, 0.0071560, -0.0048646, 0.0049443
9: -0.0265513, -0.0137256, -0.0268530, -0.0141897, -0.0123616, 0.0131274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073177, upper bound: 0.0073537
time: 1.30 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072652, upper bound: 0.0073103
time: 1.48 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043340, 0.0085692, 0.0041596, 0.0085852, -0.0042513, 0.0044096
1: 0.0002937, 0.0046798, 0.0001276, 0.0047109, -0.0044172, 0.0045523
2: -0.0244852, -0.0049818, -0.0249010, -0.0057554, -0.0187298, 0.0199192
3: -0.0020676, 0.0068718, -0.0019985, 0.0072326, -0.0093002, 0.0088703
4: 0.0113044, 0.0181673, 0.0116397, 0.0182760, -0.0069716, 0.0065276
5: -0.0032317, 0.0084276, -0.0032480, 0.0089357, -0.0121675, 0.0116756
6: 0.9943013, 1.0031309, 0.9943931, 1.0034734, -0.0091721, 0.0087378
7: 0.0070801, 0.0195031, 0.0076870, 0.0196999, -0.0093215, 0.0086511
8: 0.0023021, 0.0070985, 0.0021481, 0.0071602, -0.0048581, 0.0049504
9: -0.0265273, -0.0137289, -0.0268766, -0.0141083, -0.0124189, 0.0131477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073755, upper bound: 0.0073568
time: 1.29 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073154, upper bound: 0.0073154
time: 1.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.48 seconds
IS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0076274, upper bound: 0.0076572
IS_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0076274, upper bound: 0.0076572
IS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0077018, upper bound: 0.0076976
IS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0077018, upper bound: 0.0076976
IS_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071407, upper bound: 0.0070849
IS_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071192, upper bound: 0.0070370
IS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070860
IS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071681, upper bound: 0.0070391
IS_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0076842, upper bound: 0.0076142
IS_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0076842, upper bound: 0.0076142
IS_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0076953, upper bound: 0.0076917
IS_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0076953, upper bound: 0.0076917
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071154, upper bound: 0.0069639
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071855, upper bound: 0.0070025
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0071634, upper bound: 0.0069657
IS_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0079134, upper bound: 0.0078157
IS_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0079134, upper bound: 0.0078157
IS_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0079359, upper bound: 0.0078634
IS_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0079359, upper bound: 0.0078634
IS_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
IS_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
IS_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
IS_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0074154
IS_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0077208, upper bound: 0.0078084
IS_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0077208, upper bound: 0.0078084
IS_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0077275, upper bound: 0.0078867
IS_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0077275, upper bound: 0.0078867
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073177, upper bound: 0.0073537
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0072652, upper bound: 0.0073103
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073755, upper bound: 0.0073568
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 6, lower bound: -0.0073154, upper bound: 0.0073154

## BFS IS instance: IS_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0042634, 0.0085757, 0.0041827, 0.0085831, -0.0043197, 0.0043930
1: 0.0002265, 0.0046924, 0.0001495, 0.0047068, -0.0044803, 0.0045429
2: -0.0246534, -0.0061823, -0.0248459, -0.0058116, -0.0188418, 0.0186636
3: -0.0019604, 0.0070178, -0.0019935, 0.0071848, -0.0091452, 0.0090113
4: 0.0118247, 0.0182113, 0.0116640, 0.0182616, -0.0064370, 0.0065473
5: -0.0032383, 0.0086332, -0.0032458, 0.0088684, -0.0121067, 0.0118790
6: 0.9944438, 1.0032694, 0.9943998, 1.0034280, -0.0089842, 0.0088696
7: 0.0080219, 0.0195827, 0.0077310, 0.0196738, -0.0084456, 0.0085256
8: 0.0022398, 0.0071235, 0.0021685, 0.0071520, -0.0049123, 0.0049550
9: -0.0266686, -0.0143177, -0.0268303, -0.0141359, -0.0125327, 0.0125126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068792, upper bound: 0.0069567
time: 1.48 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063802, upper bound: 0.0065714
time: 1.03 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0042634, 0.0085757, 0.0045264, 0.0085515, -0.0042881, 0.0040493
1: 0.0002265, 0.0046924, 0.0004770, 0.0046455, -0.0044190, 0.0042154
2: -0.0246534, -0.0061823, -0.0240264, -0.0047730, -0.0198804, 0.0178441
3: -0.0019604, 0.0070178, -0.0020863, 0.0064737, -0.0084341, 0.0091040
4: 0.0118247, 0.0182113, 0.0112140, 0.0180473, -0.0062226, 0.0069973
5: -0.0032383, 0.0086332, -0.0032138, 0.0078670, -0.0111053, 0.0118470
6: 0.9944438, 1.0032694, 0.9942765, 1.0027533, -0.0083095, 0.0089929
7: 0.0080219, 0.0195827, 0.0069164, 0.0192859, -0.0080242, 0.0091167
8: 0.0022398, 0.0071235, 0.0024720, 0.0070305, -0.0047907, 0.0046514
9: -0.0266686, -0.0143177, -0.0261419, -0.0136265, -0.0130421, 0.0118242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073065, upper bound: 0.0072952
time: 1.55 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071847, upper bound: 0.0072149
time: 1.49 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0042675, 0.0085753, 0.0041707, 0.0085842, -0.0043167, 0.0044046
1: 0.0002304, 0.0046917, 0.0001382, 0.0047089, -0.0044785, 0.0045535
2: -0.0246435, -0.0061910, -0.0248743, -0.0056320, -0.0190115, 0.0186833
3: -0.0019596, 0.0070092, -0.0020095, 0.0072095, -0.0091691, 0.0090188
4: 0.0118284, 0.0182087, 0.0115862, 0.0182691, -0.0064406, 0.0066225
5: -0.0032379, 0.0086212, -0.0032469, 0.0089032, -0.0121411, 0.0118681
6: 0.9944447, 1.0032614, 0.9943785, 1.0034513, -0.0090066, 0.0088829
7: 0.0080287, 0.0195780, 0.0075902, 0.0196873, -0.0084574, 0.0086651
8: 0.0022434, 0.0071220, 0.0021579, 0.0071562, -0.0049128, 0.0049641
9: -0.0266604, -0.0143220, -0.0268542, -0.0140478, -0.0126126, 0.0125322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069148, upper bound: 0.0069741
time: 1.15 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064127, upper bound: 0.0065735
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0042675, 0.0085753, 0.0045218, 0.0085519, -0.0042844, 0.0040535
1: 0.0002304, 0.0046917, 0.0004726, 0.0046463, -0.0044159, 0.0042190
2: -0.0246435, -0.0061910, -0.0240374, -0.0045973, -0.0200463, 0.0178463
3: -0.0019596, 0.0070092, -0.0021020, 0.0064832, -0.0084428, 0.0091112
4: 0.0118284, 0.0182087, 0.0111378, 0.0180502, -0.0062217, 0.0070709
5: -0.0032379, 0.0086212, -0.0032143, 0.0078804, -0.0111184, 0.0118355
6: 0.9944447, 1.0032614, 0.9942557, 1.0027623, -0.0083176, 0.0090058
7: 0.0080287, 0.0195780, 0.0067785, 0.0192911, -0.0081012, 0.0095069
8: 0.0022434, 0.0071220, 0.0024680, 0.0070321, -0.0047887, 0.0046540
9: -0.0266604, -0.0143220, -0.0261511, -0.0135403, -0.0131201, 0.0118291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073633, upper bound: 0.0073068
time: 1.80 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072362, upper bound: 0.0072298
time: 1.20 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046110, 0.0085437, 0.0042930, 0.0085729, -0.0039620, 0.0042507
1: 0.0005576, 0.0046304, 0.0002547, 0.0046871, -0.0041295, 0.0043757
2: -0.0238247, -0.0051817, -0.0245828, -0.0058871, -0.0179376, 0.0194011
3: -0.0020498, 0.0062987, -0.0019868, 0.0069565, -0.0090063, 0.0082854
4: 0.0113911, 0.0179946, 0.0116967, 0.0181928, -0.0068018, 0.0062978
5: -0.0032059, 0.0076205, -0.0032355, 0.0085469, -0.0117529, 0.0108561
6: 0.9943250, 1.0025873, 0.9944087, 1.0032114, -0.0088863, 0.0081786
7: 0.0072369, 0.0191904, 0.0077902, 0.0195493, -0.0089745, 0.0083064
8: 0.0025468, 0.0070006, 0.0022659, 0.0071130, -0.0045662, 0.0047347
9: -0.0259724, -0.0138269, -0.0266093, -0.0141729, -0.0117995, 0.0127824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071407, upper bound: 0.0070849
time: 1.55 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071407, upper bound: 0.0070849
time: 1.32 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0046826, 0.0085371, 0.0043996, 0.0085631, -0.0038805, 0.0041374
1: 0.0006258, 0.0046176, 0.0003563, 0.0046681, -0.0040423, 0.0042614
2: -0.0236540, -0.0052479, -0.0243286, -0.0052792, -0.0183748, 0.0190808
3: -0.0020439, 0.0061505, -0.0020411, 0.0067359, -0.0087798, 0.0081916
4: 0.0114197, 0.0179499, 0.0114333, 0.0181263, -0.0067066, 0.0065166
5: -0.0031993, 0.0074119, -0.0032256, 0.0082363, -0.0114356, 0.0106375
6: 0.9943329, 1.0024467, 0.9943366, 1.0030019, -0.0086690, 0.0081100
7: 0.0072888, 0.0191096, 0.0073134, 0.0194290, -0.0089942, 0.0086196
8: 0.0026100, 0.0069752, 0.0023601, 0.0070753, -0.0044653, 0.0046151
9: -0.0258290, -0.0138594, -0.0263958, -0.0138748, -0.0119543, 0.0125364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069393, upper bound: 0.0070359
time: 1.15 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069393, upper bound: 0.0070370
time: 1.26 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0046162, 0.0085432, 0.0042817, 0.0085740, -0.0039578, 0.0042615
1: 0.0005625, 0.0046295, 0.0002439, 0.0046891, -0.0041266, 0.0043856
2: -0.0238124, -0.0051909, -0.0246098, -0.0057068, -0.0181056, 0.0194189
3: -0.0020489, 0.0062880, -0.0020029, 0.0069799, -0.0090289, 0.0082908
4: 0.0113951, 0.0179913, 0.0116186, 0.0181999, -0.0068048, 0.0063727
5: -0.0032055, 0.0076055, -0.0032366, 0.0085799, -0.0117854, 0.0108421
6: 0.9943261, 1.0025772, 0.9943874, 1.0032336, -0.0089074, 0.0081898
7: 0.0072442, 0.0191846, 0.0076489, 0.0195621, -0.0089842, 0.0084412
8: 0.0025513, 0.0069987, 0.0022559, 0.0071170, -0.0045657, 0.0047428
9: -0.0259621, -0.0138315, -0.0266320, -0.0140845, -0.0118776, 0.0128005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070860
time: 1.43 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070861
time: 1.38 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0046881, 0.0085366, 0.0043821, 0.0085647, -0.0038767, 0.0041544
1: 0.0006310, 0.0046166, 0.0003396, 0.0046712, -0.0040402, 0.0042771
2: -0.0236410, -0.0052566, -0.0243703, -0.0050957, -0.0185453, 0.0191137
3: -0.0020431, 0.0061392, -0.0020574, 0.0067721, -0.0088152, 0.0081967
4: 0.0114235, 0.0179465, 0.0113538, 0.0181372, -0.0067137, 0.0065927
5: -0.0031988, 0.0073960, -0.0032272, 0.0082872, -0.0114860, 0.0106233
6: 0.9943339, 1.0024358, 0.9943148, 1.0030364, -0.0087025, 0.0081210
7: 0.0072957, 0.0191034, 0.0071695, 0.0194487, -0.0090097, 0.0087379
8: 0.0026148, 0.0069733, 0.0023446, 0.0070815, -0.0044667, 0.0046287
9: -0.0258181, -0.0138637, -0.0264308, -0.0137847, -0.0120333, 0.0125671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069785, upper bound: 0.0070378
time: 1.23 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069785, upper bound: 0.0070391
time: 1.34 seconds

## BFS IS instance: IS_A1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0042032, 0.0085812, 0.0041029, 0.0085905, -0.0043872, 0.0044784
1: 0.0001692, 0.0047031, 0.0000735, 0.0047210, -0.0045519, 0.0046296
2: -0.0247968, -0.0063760, -0.0250362, -0.0058439, -0.0189529, 0.0186601
3: -0.0019431, 0.0071422, -0.0019906, 0.0073499, -0.0092930, 0.0091328
4: 0.0119086, 0.0182488, 0.0116780, 0.0183114, -0.0064028, 0.0065708
5: -0.0032439, 0.0088084, -0.0032532, 0.0091010, -0.0123449, 0.0120617
6: 0.9944667, 1.0033875, 0.9944036, 1.0035847, -0.0091180, 0.0089839
7: 0.0081738, 0.0196506, 0.0077564, 0.0197639, -0.0083340, 0.0086133
8: 0.0021867, 0.0071447, 0.0020980, 0.0071802, -0.0049936, 0.0050467
9: -0.0267891, -0.0144127, -0.0269902, -0.0141517, -0.0126373, 0.0125774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068973, upper bound: 0.0069216
time: 1.23 seconds

## Relational analysis of IS_A1_A2_A1_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064062, upper bound: 0.0065418
time: 1.10 seconds

## BFS IS instance: IS_A1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0042032, 0.0085812, 0.0044524, 0.0085583, -0.0043550, 0.0041288
1: 0.0001692, 0.0047031, 0.0004065, 0.0046587, -0.0044895, 0.0042966
2: -0.0247968, -0.0063760, -0.0242028, -0.0047798, -0.0200170, 0.0178268
3: -0.0019431, 0.0071422, -0.0020857, 0.0066268, -0.0085699, 0.0092279
4: 0.0119086, 0.0182488, 0.0112169, 0.0180935, -0.0061848, 0.0070319
5: -0.0032439, 0.0088084, -0.0032207, 0.0080826, -0.0113265, 0.0120292
6: 0.9944667, 1.0033875, 0.9942774, 1.0028986, -0.0084319, 0.0091100
7: 0.0081738, 0.0196506, 0.0069217, 0.0193694, -0.0079413, 0.0093918
8: 0.0021867, 0.0071447, 0.0024067, 0.0070567, -0.0048700, 0.0047381
9: -0.0267891, -0.0144127, -0.0262901, -0.0136298, -0.0131593, 0.0118774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A1_A1_B2_A1

### Relational analysis result of IS_A1_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073553, upper bound: 0.0072562
time: 1.51 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2_A2

### Relational analysis result of IS_A1_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072222, upper bound: 0.0071760
time: 1.30 seconds

## BFS IS instance: IS_A1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042060, 0.0085810, 0.0041072, 0.0085901, -0.0043841, 0.0044738
1: 0.0001717, 0.0047026, 0.0000777, 0.0047203, -0.0045485, 0.0046250
2: -0.0247904, -0.0062304, -0.0250258, -0.0058497, -0.0189407, 0.0187954
3: -0.0019561, 0.0071366, -0.0019901, 0.0073409, -0.0092970, 0.0091267
4: 0.0118455, 0.0182471, 0.0116805, 0.0183087, -0.0064632, 0.0065666
5: -0.0032436, 0.0088006, -0.0032528, 0.0090883, -0.0123320, 0.0120534
6: 0.9944495, 1.0033823, 0.9944043, 1.0035762, -0.0091267, 0.0089780
7: 0.0080596, 0.0196475, 0.0077609, 0.0197590, -0.0084586, 0.0086312
8: 0.0021891, 0.0071438, 0.0021018, 0.0071787, -0.0049896, 0.0050420
9: -0.0267837, -0.0143413, -0.0269815, -0.0141546, -0.0126291, 0.0126402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069118, upper bound: 0.0069578
time: 1.23 seconds

## Relational analysis of IS_A1_A2_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064065, upper bound: 0.0065673
time: 1.13 seconds

## BFS IS instance: IS_A1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042060, 0.0085810, 0.0044572, 0.0085578, -0.0043519, 0.0041237
1: 0.0001717, 0.0047026, 0.0004111, 0.0046578, -0.0044861, 0.0042915
2: -0.0247904, -0.0062304, -0.0241913, -0.0047862, -0.0200041, 0.0179609
3: -0.0019561, 0.0071366, -0.0020851, 0.0066168, -0.0085729, 0.0092217
4: 0.0118455, 0.0182471, 0.0112197, 0.0180904, -0.0062449, 0.0070274
5: -0.0032436, 0.0088006, -0.0032203, 0.0080685, -0.0113122, 0.0120208
6: 0.9944495, 1.0033823, 0.9942782, 1.0028889, -0.0084394, 0.0091041
7: 0.0080596, 0.0196475, 0.0069267, 0.0193639, -0.0080720, 0.0094021
8: 0.0021891, 0.0071438, 0.0024109, 0.0070549, -0.0048659, 0.0047329
9: -0.0267837, -0.0143413, -0.0262804, -0.0136330, -0.0131507, 0.0119391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073564, upper bound: 0.0073028
time: 1.37 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072299, upper bound: 0.0072239
time: 1.74 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0044489, 0.0085586, 0.0042970, 0.0085726, -0.0041237, 0.0042616
1: 0.0004031, 0.0046593, 0.0002585, 0.0046864, -0.0042833, 0.0044008
2: -0.0242112, -0.0052588, -0.0245733, -0.0059965, -0.0182148, 0.0193145
3: -0.0020429, 0.0066341, -0.0019770, 0.0069483, -0.0089911, 0.0086111
4: 0.0114245, 0.0180957, 0.0117441, 0.0181903, -0.0067658, 0.0063515
5: -0.0032210, 0.0080929, -0.0032352, 0.0085353, -0.0117564, 0.0113280
6: 0.9943342, 1.0029055, 0.9944218, 1.0032035, -0.0088693, 0.0084837
7: 0.0072974, 0.0193734, 0.0078761, 0.0195448, -0.0089717, 0.0083461
8: 0.0024036, 0.0070579, 0.0022694, 0.0071116, -0.0047080, 0.0047884
9: -0.0262972, -0.0138648, -0.0266013, -0.0142266, -0.0120706, 0.0127365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
time: 1.51 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
time: 1.27 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0045195, 0.0085521, 0.0044021, 0.0085629, -0.0040434, 0.0041500
1: 0.0004705, 0.0046467, 0.0003586, 0.0046676, -0.0041972, 0.0042881
2: -0.0240427, -0.0053252, -0.0243226, -0.0053842, -0.0186585, 0.0189975
3: -0.0020370, 0.0064878, -0.0020317, 0.0067308, -0.0087677, 0.0085195
4: 0.0114532, 0.0180516, 0.0114788, 0.0181248, -0.0066716, 0.0065727
5: -0.0032145, 0.0078870, -0.0032254, 0.0082290, -0.0114435, 0.0111124
6: 0.9943420, 1.0027666, 0.9943491, 1.0029972, -0.0086551, 0.0084175
7: 0.0073495, 0.0192936, 0.0073958, 0.0194261, -0.0089476, 0.0087621
8: 0.0024660, 0.0070329, 0.0023623, 0.0070744, -0.0046084, 0.0046706
9: -0.0261556, -0.0138973, -0.0263908, -0.0139263, -0.0122293, 0.0124934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069286, upper bound: 0.0069634
time: 1.42 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069286, upper bound: 0.0069639
time: 1.46 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0044593, 0.0085576, 0.0042858, 0.0085736, -0.0041143, 0.0042718
1: 0.0004131, 0.0046574, 0.0002478, 0.0046884, -0.0042753, 0.0044096
2: -0.0241862, -0.0052663, -0.0245999, -0.0058312, -0.0183551, 0.0193336
3: -0.0020422, 0.0066124, -0.0019918, 0.0069714, -0.0090136, 0.0086042
4: 0.0114277, 0.0180891, 0.0116725, 0.0181973, -0.0067696, 0.0064166
5: -0.0032201, 0.0080624, -0.0032362, 0.0085678, -0.0117879, 0.0112986
6: 0.9943351, 1.0028849, 0.9944022, 1.0032256, -0.0088905, 0.0084827
7: 0.0073033, 0.0193616, 0.0077464, 0.0195574, -0.0089783, 0.0085564
8: 0.0024128, 0.0070542, 0.0022596, 0.0071155, -0.0047027, 0.0047946
9: -0.0262762, -0.0138684, -0.0266237, -0.0141455, -0.0121307, 0.0127552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071634, upper bound: 0.0069657
time: 1.33 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071634, upper bound: 0.0069657
time: 1.26 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0045307, 0.0085511, 0.0043864, 0.0085643, -0.0040336, 0.0041647
1: 0.0004811, 0.0046447, 0.0003436, 0.0046705, -0.0041893, 0.0043011
2: -0.0240160, -0.0053323, -0.0243602, -0.0052113, -0.0188047, 0.0190279
3: -0.0020363, 0.0064647, -0.0020471, 0.0067634, -0.0087997, 0.0085118
4: 0.0114563, 0.0180446, 0.0114039, 0.0181346, -0.0066783, 0.0066407
5: -0.0032134, 0.0078544, -0.0032269, 0.0082750, -0.0114884, 0.0110812
6: 0.9943430, 1.0027448, 0.9943286, 1.0030282, -0.0086852, 0.0084162
7: 0.0073551, 0.0192810, 0.0072602, 0.0194439, -0.0090031, 0.0088571
8: 0.0024758, 0.0070289, 0.0023484, 0.0070800, -0.0046041, 0.0046805
9: -0.0261332, -0.0139008, -0.0264223, -0.0138415, -0.0122917, 0.0125215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069652, upper bound: 0.0069652
time: 1.00 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069652, upper bound: 0.0069657
time: 1.28 seconds

## BFS IS instance: IS_A2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0042384, 0.0085780, 0.0040614, 0.0085943, -0.0043559, 0.0045166
1: 0.0002026, 0.0046969, 0.0000340, 0.0047284, -0.0045258, 0.0046628
2: -0.0247130, -0.0060338, -0.0251351, -0.0057140, -0.0189990, 0.0191013
3: -0.0019737, 0.0070695, -0.0020022, 0.0074358, -0.0094094, 0.0090717
4: 0.0117603, 0.0182269, 0.0116217, 0.0183372, -0.0065769, 0.0066051
5: -0.0032406, 0.0087060, -0.0032571, 0.0092218, -0.0124625, 0.0119631
6: 0.9944262, 1.0033185, 0.9943882, 1.0036662, -0.0092400, 0.0089304
7: 0.0079053, 0.0196109, 0.0076545, 0.0198107, -0.0085478, 0.0086397
8: 0.0022177, 0.0071323, 0.0020613, 0.0071949, -0.0049772, 0.0050710
9: -0.0267187, -0.0142449, -0.0270733, -0.0140880, -0.0126307, 0.0128284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070438, upper bound: 0.0068564
time: 1.33 seconds

## Relational analysis of IS_A2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069538, upper bound: 0.0068172
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0045672, 0.0085477, 0.0040614, 0.0085943, -0.0040271, 0.0044863
1: 0.0005158, 0.0046382, 0.0000340, 0.0047284, -0.0042126, 0.0046042
2: -0.0239292, -0.0049969, -0.0251351, -0.0057140, -0.0182153, 0.0201382
3: -0.0020663, 0.0063894, -0.0020022, 0.0074358, -0.0095020, 0.0083916
4: 0.0113110, 0.0180219, 0.0116217, 0.0183372, -0.0070263, 0.0064002
5: -0.0032100, 0.0077483, -0.0032571, 0.0092218, -0.0124319, 0.0110054
6: 0.9943031, 1.0026733, 0.9943882, 1.0036662, -0.0093631, 0.0082851
7: 0.0070920, 0.0192399, 0.0076545, 0.0198107, -0.0092104, 0.0082129
8: 0.0025080, 0.0070161, 0.0020613, 0.0071949, -0.0046869, 0.0049548
9: -0.0260603, -0.0137363, -0.0270733, -0.0140880, -0.0119722, 0.0133370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076190, upper bound: 0.0075181
time: 1.27 seconds

## Relational analysis of IS_A2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0075992, upper bound: 0.0074655
time: 1.69 seconds

## BFS IS instance: IS_A2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0042253, 0.0085792, 0.0040663, 0.0085938, -0.0043685, 0.0045129
1: 0.0001902, 0.0046992, 0.0000387, 0.0047276, -0.0045374, 0.0046605
2: -0.0247442, -0.0058416, -0.0251234, -0.0057196, -0.0190246, 0.0192818
3: -0.0019908, 0.0070966, -0.0020017, 0.0074256, -0.0094165, 0.0090983
4: 0.0116770, 0.0182350, 0.0116242, 0.0183342, -0.0066572, 0.0066109
5: -0.0032418, 0.0087442, -0.0032566, 0.0092076, -0.0124494, 0.0120008
6: 0.9944034, 1.0033443, 0.9943889, 1.0036565, -0.0092531, 0.0089554
7: 0.0077545, 0.0196257, 0.0076589, 0.0198052, -0.0088209, 0.0086569
8: 0.0022061, 0.0071369, 0.0020657, 0.0071932, -0.0049871, 0.0050713
9: -0.0267449, -0.0141506, -0.0270635, -0.0140908, -0.0126541, 0.0129129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070432, upper bound: 0.0068340
time: 1.09 seconds

## Relational analysis of IS_A2_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069538, upper bound: 0.0067949
time: 0.99 seconds

## BFS IS instance: IS_A2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0045623, 0.0085481, 0.0040663, 0.0085938, -0.0040315, 0.0044819
1: 0.0005112, 0.0046391, 0.0000387, 0.0047276, -0.0042163, 0.0046004
2: -0.0239407, -0.0048192, -0.0251234, -0.0057196, -0.0182211, 0.0203042
3: -0.0020821, 0.0063993, -0.0020017, 0.0074256, -0.0095078, 0.0084010
4: 0.0112340, 0.0180249, 0.0116242, 0.0183342, -0.0071002, 0.0064007
5: -0.0032105, 0.0077623, -0.0032566, 0.0092076, -0.0124180, 0.0110189
6: 0.9942821, 1.0026828, 0.9943889, 1.0036565, -0.0093744, 0.0082939
7: 0.0069526, 0.0192453, 0.0076589, 0.0198052, -0.0093466, 0.0082297
8: 0.0025038, 0.0070178, 0.0020657, 0.0071932, -0.0046894, 0.0049521
9: -0.0260699, -0.0136491, -0.0270635, -0.0140908, -0.0119791, 0.0134143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076400, upper bound: 0.0075429
time: 1.31 seconds

## Relational analysis of IS_A2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0076158, upper bound: 0.0074873
time: 1.16 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0043469, 0.0085680, 0.0044072, 0.0085624, -0.0042156, 0.0041608
1: 0.0003060, 0.0046775, 0.0003635, 0.0046667, -0.0043608, 0.0043140
2: -0.0244544, -0.0061087, -0.0243105, -0.0046788, -0.0197756, 0.0182019
3: -0.0019670, 0.0068451, -0.0020947, 0.0067203, -0.0086872, 0.0089398
4: 0.0117928, 0.0181592, 0.0111731, 0.0181216, -0.0063289, 0.0069861
5: -0.0032305, 0.0083901, -0.0032249, 0.0082142, -0.0114448, 0.0116150
6: 0.9944350, 1.0031056, 0.9942654, 1.0029871, -0.0085521, 0.0088402
7: 0.0079641, 0.0194885, 0.0068425, 0.0194204, -0.0082530, 0.0092145
8: 0.0023135, 0.0070940, 0.0023668, 0.0070726, -0.0047591, 0.0047272
9: -0.0265015, -0.0142816, -0.0263806, -0.0135803, -0.0129212, 0.0120990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B2_A1_A1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.38 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0044409, 0.0085593, 0.0044758, 0.0085561, -0.0041152, 0.0040835
1: 0.0003956, 0.0046607, 0.0004288, 0.0046545, -0.0042589, 0.0042319
2: -0.0242302, -0.0054960, -0.0241469, -0.0047380, -0.0194922, 0.0186510
3: -0.0020217, 0.0066505, -0.0020894, 0.0065783, -0.0086000, 0.0087399
4: 0.0115272, 0.0181006, 0.0111988, 0.0180788, -0.0065516, 0.0069018
5: -0.0032218, 0.0081161, -0.0032185, 0.0080143, -0.0112361, 0.0113346
6: 0.9943622, 1.0029211, 0.9942725, 1.0028524, -0.0084902, 0.0086486
7: 0.0074834, 0.0193824, 0.0068889, 0.0193430, -0.0086803, 0.0091783
8: 0.0023965, 0.0070607, 0.0024274, 0.0070484, -0.0046518, 0.0046333
9: -0.0263131, -0.0139811, -0.0262431, -0.0136093, -0.0127038, 0.0122621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B2_A1_A2_A1

### Relational analysis result of IS_A2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
time: 1.25 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
time: 1.21 seconds

## BFS IS instance: IS_A2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0043357, 0.0085690, 0.0044120, 0.0085620, -0.0042263, 0.0041570
1: 0.0002953, 0.0046795, 0.0003680, 0.0046659, -0.0043706, 0.0043115
2: -0.0244810, -0.0059194, -0.0242992, -0.0046859, -0.0197951, 0.0183798
3: -0.0019839, 0.0068682, -0.0020940, 0.0067104, -0.0086943, 0.0089623
4: 0.0117107, 0.0181662, 0.0111762, 0.0181186, -0.0064079, 0.0069900
5: -0.0032316, 0.0084226, -0.0032245, 0.0082004, -0.0114319, 0.0116471
6: 0.9944126, 1.0031276, 0.9942662, 1.0029778, -0.0085652, 0.0088614
7: 0.0078156, 0.0195011, 0.0068480, 0.0194150, -0.0083995, 0.0092272
8: 0.0023036, 0.0070979, 0.0023710, 0.0070709, -0.0047673, 0.0047269
9: -0.0265238, -0.0141887, -0.0263710, -0.0135838, -0.0129401, 0.0121823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B2_A2_A1_A1

### Relational analysis result of IS_A2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.11 seconds

## Relational analysis of IS_A2_A1_B2_A2_A1_A2

### Relational analysis result of IS_A2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.34 seconds

## BFS IS instance: IS_A2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0044273, 0.0085606, 0.0044811, 0.0085556, -0.0041284, 0.0040794
1: 0.0003826, 0.0046632, 0.0004339, 0.0046535, -0.0042710, 0.0042293
2: -0.0242627, -0.0052969, -0.0241342, -0.0047448, -0.0195179, 0.0188373
3: -0.0020395, 0.0066788, -0.0020888, 0.0065673, -0.0086067, 0.0087675
4: 0.0114410, 0.0181091, 0.0112017, 0.0180755, -0.0066345, 0.0069074
5: -0.0032230, 0.0081558, -0.0032180, 0.0079988, -0.0112219, 0.0113738
6: 0.9943388, 1.0029478, 0.9942732, 1.0028421, -0.0085033, 0.0086746
7: 0.0073273, 0.0193978, 0.0068942, 0.0193369, -0.0088095, 0.0092603
8: 0.0023845, 0.0070655, 0.0024321, 0.0070465, -0.0046620, 0.0046334
9: -0.0263404, -0.0138835, -0.0262325, -0.0136126, -0.0127278, 0.0123490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B2_A2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0074154
time: 1.21 seconds

## Relational analysis of IS_A2_A1_B2_A2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0074154
time: 1.22 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041040, 0.0085904, 0.0040664, 0.0085938, -0.0044898, 0.0045239
1: 0.0000746, 0.0047208, 0.0000388, 0.0047275, -0.0046529, 0.0046820
2: -0.0250335, -0.0061442, -0.0251230, -0.0058248, -0.0192088, 0.0189787
3: -0.0019638, 0.0073476, -0.0019923, 0.0074253, -0.0093891, 0.0093399
4: 0.0118082, 0.0183107, 0.0116697, 0.0183341, -0.0065259, 0.0066409
5: -0.0032531, 0.0090977, -0.0032566, 0.0092070, -0.0124602, 0.0123543
6: 0.9944393, 1.0035825, 0.9944014, 1.0036560, -0.0092167, 0.0091811
7: 0.0079920, 0.0197626, 0.0077414, 0.0198050, -0.0085339, 0.0086590
8: 0.0020989, 0.0071798, 0.0020658, 0.0071931, -0.0050942, 0.0051140
9: -0.0269879, -0.0142991, -0.0270631, -0.0141423, -0.0128456, 0.0127640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069732, upper bound: 0.0070756
time: 1.60 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064219, upper bound: 0.0067213
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041040, 0.0085904, 0.0044094, 0.0085622, -0.0044583, 0.0041810
1: 0.0000746, 0.0047208, 0.0003655, 0.0046664, -0.0045918, 0.0043553
2: -0.0250335, -0.0061442, -0.0243054, -0.0047620, -0.0202715, 0.0181611
3: -0.0019638, 0.0073476, -0.0020872, 0.0067158, -0.0086796, 0.0094349
4: 0.0118082, 0.0183107, 0.0112092, 0.0181203, -0.0063121, 0.0071015
5: -0.0032531, 0.0090977, -0.0032247, 0.0082079, -0.0114611, 0.0123224
6: 0.9944393, 1.0035825, 0.9942753, 1.0029830, -0.0085437, 0.0093071
7: 0.0079920, 0.0197626, 0.0069077, 0.0194180, -0.0081360, 0.0094241
8: 0.0020989, 0.0071798, 0.0023687, 0.0070719, -0.0049729, 0.0048112
9: -0.0269879, -0.0142991, -0.0263762, -0.0136211, -0.0133669, 0.0120772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074024, upper bound: 0.0075157
time: 1.25 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073505, upper bound: 0.0074935
time: 1.39 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041065, 0.0085901, 0.0040711, 0.0085934, -0.0044869, 0.0045190
1: 0.0000769, 0.0047204, 0.0000433, 0.0047267, -0.0046498, 0.0046771
2: -0.0250276, -0.0059850, -0.0251118, -0.0058304, -0.0191972, 0.0191268
3: -0.0019780, 0.0073425, -0.0019918, 0.0074155, -0.0093936, 0.0093343
4: 0.0117391, 0.0183091, 0.0116722, 0.0183311, -0.0065920, 0.0066370
5: -0.0032529, 0.0090905, -0.0032562, 0.0091933, -0.0124463, 0.0123467
6: 0.9944204, 1.0035776, 0.9944021, 1.0036470, -0.0092266, 0.0091755
7: 0.0078670, 0.0197598, 0.0077458, 0.0197997, -0.0086718, 0.0086796
8: 0.0021011, 0.0071790, 0.0020699, 0.0071914, -0.0050903, 0.0051090
9: -0.0269830, -0.0142209, -0.0270537, -0.0141451, -0.0128379, 0.0128327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069880, upper bound: 0.0071135
time: 1.69 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064221, upper bound: 0.0067656
time: 1.11 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041065, 0.0085901, 0.0044140, 0.0085618, -0.0044554, 0.0041762
1: 0.0000769, 0.0047204, 0.0003699, 0.0046655, -0.0045886, 0.0043505
2: -0.0250276, -0.0059850, -0.0242944, -0.0047685, -0.0202591, 0.0183095
3: -0.0019780, 0.0073425, -0.0020867, 0.0067063, -0.0086843, 0.0094292
4: 0.0117391, 0.0183091, 0.0112120, 0.0181174, -0.0063783, 0.0070972
5: -0.0032529, 0.0090905, -0.0032243, 0.0081945, -0.0114474, 0.0123148
6: 0.9944204, 1.0035776, 0.9942760, 1.0029739, -0.0085535, 0.0093015
7: 0.0078670, 0.0197598, 0.0069128, 0.0194128, -0.0082784, 0.0094375
8: 0.0021011, 0.0071790, 0.0023728, 0.0070702, -0.0049691, 0.0048062
9: -0.0269830, -0.0142209, -0.0263671, -0.0136242, -0.0133587, 0.0121461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A2_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0074031, upper bound: 0.0075876
time: 1.29 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073556, upper bound: 0.0075658
time: 1.29 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0043220, 0.0085703, 0.0042598, 0.0085760, -0.0042541, 0.0043105
1: 0.0002822, 0.0046819, 0.0002230, 0.0046930, -0.0044108, 0.0044589
2: -0.0245138, -0.0049751, -0.0246621, -0.0059775, -0.0185363, 0.0196869
3: -0.0020682, 0.0068966, -0.0019787, 0.0070253, -0.0090935, 0.0088753
4: 0.0113016, 0.0181748, 0.0117359, 0.0182135, -0.0069120, 0.0064388
5: -0.0032328, 0.0084626, -0.0032386, 0.0086438, -0.0118767, 0.0117013
6: 0.9943006, 1.0031545, 0.9944195, 1.0032767, -0.0089761, 0.0087350
7: 0.0070749, 0.0195166, 0.0078612, 0.0195868, -0.0092149, 0.0083977
8: 0.0022915, 0.0071028, 0.0022366, 0.0071248, -0.0048333, 0.0048662
9: -0.0265513, -0.0137256, -0.0266759, -0.0142173, -0.0123340, 0.0129503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073177, upper bound: 0.0073537
time: 1.16 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073177, upper bound: 0.0073537
time: 1.43 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0043934, 0.0085637, 0.0043594, 0.0085668, -0.0041734, 0.0042043
1: 0.0003503, 0.0046692, 0.0003179, 0.0046753, -0.0043249, 0.0043513
2: -0.0243434, -0.0050362, -0.0244245, -0.0053634, -0.0189800, 0.0193883
3: -0.0020628, 0.0067488, -0.0020335, 0.0068191, -0.0088819, 0.0087823
4: 0.0113280, 0.0181302, 0.0114698, 0.0181514, -0.0068234, 0.0066604
5: -0.0032262, 0.0082544, -0.0032294, 0.0083535, -0.0115797, 0.0114837
6: 0.9943078, 1.0030143, 0.9943466, 1.0030810, -0.0087732, 0.0086677
7: 0.0071228, 0.0194360, 0.0073795, 0.0194743, -0.0091997, 0.0087711
8: 0.0023546, 0.0070775, 0.0023246, 0.0070895, -0.0047349, 0.0047529
9: -0.0264082, -0.0137556, -0.0264763, -0.0139161, -0.0124921, 0.0127208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069293, upper bound: 0.0071531
time: 1.39 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069293, upper bound: 0.0073103
time: 1.37 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0043340, 0.0085692, 0.0042497, 0.0085769, -0.0042430, 0.0043195
1: 0.0002937, 0.0046798, 0.0002134, 0.0046948, -0.0044012, 0.0044664
2: -0.0244852, -0.0049818, -0.0246862, -0.0058123, -0.0186729, 0.0197044
3: -0.0020676, 0.0068718, -0.0019934, 0.0070462, -0.0091138, 0.0088652
4: 0.0113044, 0.0181673, 0.0116643, 0.0182198, -0.0069154, 0.0065030
5: -0.0032317, 0.0084276, -0.0032396, 0.0086732, -0.0119050, 0.0116672
6: 0.9943013, 1.0031309, 0.9943998, 1.0032965, -0.0089952, 0.0087311
7: 0.0070801, 0.0195031, 0.0077316, 0.0195982, -0.0092218, 0.0086041
8: 0.0023021, 0.0070985, 0.0022276, 0.0071283, -0.0048262, 0.0048709
9: -0.0265273, -0.0137289, -0.0266961, -0.0141362, -0.0123910, 0.0129673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073755, upper bound: 0.0073568
time: 1.58 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073755, upper bound: 0.0073568
time: 1.30 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044057, 0.0085626, 0.0043451, 0.0085681, -0.0041624, 0.0042175
1: 0.0003620, 0.0046670, 0.0003043, 0.0046778, -0.0043158, 0.0043627
2: -0.0243141, -0.0050427, -0.0244586, -0.0051901, -0.0191239, 0.0194159
3: -0.0020622, 0.0067233, -0.0020490, 0.0068487, -0.0089109, 0.0087723
4: 0.0113308, 0.0181225, 0.0113947, 0.0181603, -0.0068295, 0.0067278
5: -0.0032251, 0.0082185, -0.0032307, 0.0083952, -0.0116202, 0.0114492
6: 0.9943085, 1.0029901, 0.9943261, 1.0031091, -0.0088006, 0.0086641
7: 0.0071279, 0.0194221, 0.0072436, 0.0194905, -0.0092510, 0.0088637
8: 0.0023655, 0.0070731, 0.0023119, 0.0070946, -0.0047291, 0.0047612
9: -0.0263835, -0.0137587, -0.0265050, -0.0138311, -0.0125525, 0.0127463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069657, upper bound: 0.0071634
time: 1.24 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069657, upper bound: 0.0073154
time: 1.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.10 seconds
IS_A1_A1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0068792, upper bound: 0.0069567
IS_A1_A1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0063802, upper bound: 0.0065714
IS_A1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073065, upper bound: 0.0072952
IS_A1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071847, upper bound: 0.0072149
IS_A1_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069148, upper bound: 0.0069741
IS_A1_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0064127, upper bound: 0.0065735
IS_A1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073633, upper bound: 0.0073068
IS_A1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0072362, upper bound: 0.0072298
IS_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071407, upper bound: 0.0070849
IS_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071407, upper bound: 0.0070849
IS_A1_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069393, upper bound: 0.0070359
IS_A1_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069393, upper bound: 0.0070370
IS_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070860
IS_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070861
IS_A1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069785, upper bound: 0.0070378
IS_A1_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069785, upper bound: 0.0070391
IS_A1_A2_A1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0068973, upper bound: 0.0069216
IS_A1_A2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0064062, upper bound: 0.0065418
IS_A1_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073553, upper bound: 0.0072562
IS_A1_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0072222, upper bound: 0.0071760
IS_A1_A2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069118, upper bound: 0.0069578
IS_A1_A2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0064065, upper bound: 0.0065673
IS_A1_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073564, upper bound: 0.0073028
IS_A1_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0072299, upper bound: 0.0072239
IS_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
IS_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
IS_A1_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069286, upper bound: 0.0069634
IS_A1_A2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069286, upper bound: 0.0069639
IS_A1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071634, upper bound: 0.0069657
IS_A1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0071634, upper bound: 0.0069657
IS_A1_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069652, upper bound: 0.0069652
IS_A1_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069652, upper bound: 0.0069657
IS_A2_A1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0070438, upper bound: 0.0068564
IS_A2_A1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069538, upper bound: 0.0068172
IS_A2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0076190, upper bound: 0.0075181
IS_A2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0075992, upper bound: 0.0074655
IS_A2_A1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0070432, upper bound: 0.0068340
IS_A2_A1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069538, upper bound: 0.0067949
IS_A2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0076400, upper bound: 0.0075429
IS_A2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0076158, upper bound: 0.0074873
IS_A2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
IS_A2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
IS_A2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
IS_A2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
IS_A2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
IS_A2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
IS_A2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0074154
IS_A2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073255, upper bound: 0.0074154
IS_A2_A2_A1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069732, upper bound: 0.0070756
IS_A2_A2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0064219, upper bound: 0.0067213
IS_A2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0074024, upper bound: 0.0075157
IS_A2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073505, upper bound: 0.0074935
IS_A2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069880, upper bound: 0.0071135
IS_A2_A2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0064221, upper bound: 0.0067656
IS_A2_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0074031, upper bound: 0.0075876
IS_A2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073556, upper bound: 0.0075658
IS_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073177, upper bound: 0.0073537
IS_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073177, upper bound: 0.0073537
IS_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069293, upper bound: 0.0071531
IS_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069293, upper bound: 0.0073103
IS_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073755, upper bound: 0.0073568
IS_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0073755, upper bound: 0.0073568
IS_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069657, upper bound: 0.0071634
IS_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.10
Output dim: 6, lower bound: -0.0069657, upper bound: 0.0073154

## BFS IS instance: IS_A1_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0043487, 0.0085678, 0.0045264, 0.0085515, -0.0042027, 0.0040414
1: 0.0003078, 0.0046772, 0.0004770, 0.0046455, -0.0043377, 0.0042002
2: -0.0244500, -0.0062397, -0.0240264, -0.0047730, -0.0196770, 0.0177867
3: -0.0019553, 0.0068412, -0.0020863, 0.0064737, -0.0084289, 0.0089275
4: 0.0118496, 0.0181581, 0.0112140, 0.0180473, -0.0061977, 0.0069441
5: -0.0032304, 0.0083846, -0.0032138, 0.0078670, -0.0110974, 0.0115984
6: 0.9944506, 1.0031018, 0.9942765, 1.0027533, -0.0083027, 0.0088253
7: 0.0080669, 0.0194864, 0.0069164, 0.0192859, -0.0078820, 0.0090258
8: 0.0023151, 0.0070933, 0.0024720, 0.0070305, -0.0047153, 0.0046213
9: -0.0264977, -0.0143459, -0.0261419, -0.0136265, -0.0128712, 0.0117960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073065, upper bound: 0.0072952
time: 1.56 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073065, upper bound: 0.0072952
time: 1.30 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0044470, 0.0085588, 0.0045958, 0.0085451, -0.0040980, 0.0039629
1: 0.0004014, 0.0046596, 0.0005431, 0.0046331, -0.0042317, 0.0041165
2: -0.0242156, -0.0056782, -0.0238609, -0.0048343, -0.0193813, 0.0181826
3: -0.0020054, 0.0066379, -0.0020808, 0.0063300, -0.0083354, 0.0087187
4: 0.0116062, 0.0180968, 0.0112405, 0.0180040, -0.0063978, 0.0068563
5: -0.0032212, 0.0080983, -0.0032074, 0.0076647, -0.0108859, 0.0113056
6: 0.9943840, 1.0029089, 0.9942839, 1.0026169, -0.0082329, 0.0086251
7: 0.0076264, 0.0193755, 0.0069645, 0.0192075, -0.0081841, 0.0090371
8: 0.0024020, 0.0070585, 0.0025334, 0.0070059, -0.0046040, 0.0045252
9: -0.0263008, -0.0140705, -0.0260028, -0.0136566, -0.0126443, 0.0119323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069765, upper bound: 0.0072042
time: 1.05 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069765, upper bound: 0.0072149
time: 1.43 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0043535, 0.0085674, 0.0045218, 0.0085519, -0.0041984, 0.0040456
1: 0.0003123, 0.0046763, 0.0004726, 0.0046463, -0.0043340, 0.0042037
2: -0.0244386, -0.0062488, -0.0240374, -0.0045973, -0.0198413, 0.0177885
3: -0.0019545, 0.0068314, -0.0021020, 0.0064832, -0.0084377, 0.0089333
4: 0.0118535, 0.0181551, 0.0111378, 0.0180502, -0.0061967, 0.0070173
5: -0.0032299, 0.0083707, -0.0032143, 0.0078804, -0.0111104, 0.0115849
6: 0.9944516, 1.0030925, 0.9942557, 1.0027623, -0.0083107, 0.0088369
7: 0.0080740, 0.0194810, 0.0067785, 0.0192911, -0.0079400, 0.0094206
8: 0.0023194, 0.0070916, 0.0024680, 0.0070321, -0.0047127, 0.0046236
9: -0.0264881, -0.0143503, -0.0261511, -0.0135403, -0.0129479, 0.0118008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073633, upper bound: 0.0073068
time: 1.74 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073633, upper bound: 0.0073068
time: 1.39 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0044588, 0.0085577, 0.0045919, 0.0085454, -0.0040866, 0.0039658
1: 0.0004126, 0.0046575, 0.0005393, 0.0046338, -0.0042212, 0.0041182
2: -0.0241875, -0.0056877, -0.0238703, -0.0046617, -0.0195258, 0.0181826
3: -0.0020046, 0.0066135, -0.0020962, 0.0063383, -0.0083428, 0.0087097
4: 0.0116103, 0.0180894, 0.0111657, 0.0180065, -0.0063962, 0.0069237
5: -0.0032201, 0.0080638, -0.0032077, 0.0076763, -0.0108964, 0.0112716
6: 0.9943851, 1.0028859, 0.9942634, 1.0026248, -0.0082396, 0.0086226
7: 0.0076339, 0.0193621, 0.0068290, 0.0192120, -0.0081910, 0.0093361
8: 0.0024123, 0.0070544, 0.0025299, 0.0070073, -0.0045950, 0.0045245
9: -0.0262772, -0.0140751, -0.0260108, -0.0135719, -0.0127053, 0.0119357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072362, upper bound: 0.0072298
time: 1.22 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072362, upper bound: 0.0072298
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046110, 0.0085437, 0.0042677, 0.0085753, -0.0039643, 0.0042759
1: 0.0005576, 0.0046304, 0.0002306, 0.0046916, -0.0041340, 0.0043998
2: -0.0238247, -0.0051817, -0.0246431, -0.0058685, -0.0179562, 0.0194614
3: -0.0020498, 0.0062987, -0.0019884, 0.0070088, -0.0090586, 0.0082871
4: 0.0113911, 0.0179946, 0.0116887, 0.0182086, -0.0068175, 0.0063059
5: -0.0032059, 0.0076205, -0.0032379, 0.0086206, -0.0118265, 0.0108584
6: 0.9943250, 1.0025873, 0.9944065, 1.0032610, -0.0089359, 0.0081808
7: 0.0072369, 0.0191904, 0.0077757, 0.0195778, -0.0090179, 0.0081405
8: 0.0025468, 0.0070006, 0.0022436, 0.0071219, -0.0045752, 0.0047570
9: -0.0259724, -0.0138269, -0.0266599, -0.0141638, -0.0118086, 0.0128330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070251, upper bound: 0.0070849
time: 1.74 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070251, upper bound: 0.0070849
time: 1.26 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0046110, 0.0085437, 0.0046189, 0.0085429, -0.0039319, 0.0039247
1: 0.0005576, 0.0046304, 0.0005652, 0.0046290, -0.0040714, 0.0040652
2: -0.0238247, -0.0051817, -0.0238057, -0.0048249, -0.0189998, 0.0186240
3: -0.0020498, 0.0062987, -0.0020816, 0.0062822, -0.0083319, 0.0083803
4: 0.0113911, 0.0179946, 0.0112365, 0.0179896, -0.0065985, 0.0067581
5: -0.0032059, 0.0076205, -0.0032052, 0.0075973, -0.0108032, 0.0108257
6: 0.9943250, 1.0025873, 0.9942828, 1.0025716, -0.0082465, 0.0083045
7: 0.0072369, 0.0191904, 0.0069571, 0.0191814, -0.0082746, 0.0083295
8: 0.0025468, 0.0070006, 0.0025538, 0.0069978, -0.0044510, 0.0044468
9: -0.0259724, -0.0138269, -0.0259565, -0.0136519, -0.0123205, 0.0121295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070251, upper bound: 0.0070849
time: 1.35 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070251, upper bound: 0.0070849
time: 1.28 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046162, 0.0085432, 0.0042581, 0.0085762, -0.0039600, 0.0042850
1: 0.0005625, 0.0046295, 0.0002215, 0.0046933, -0.0041308, 0.0044080
2: -0.0238124, -0.0051909, -0.0246659, -0.0056884, -0.0181240, 0.0194750
3: -0.0020489, 0.0062880, -0.0020045, 0.0070286, -0.0090776, 0.0082925
4: 0.0113951, 0.0179913, 0.0116107, 0.0182146, -0.0068195, 0.0063807
5: -0.0032055, 0.0076055, -0.0032388, 0.0086485, -0.0118540, 0.0108443
6: 0.9943261, 1.0025772, 0.9943852, 1.0032798, -0.0089537, 0.0081920
7: 0.0072442, 0.0191846, 0.0076344, 0.0195886, -0.0090282, 0.0082745
8: 0.0025513, 0.0069987, 0.0022351, 0.0071253, -0.0045740, 0.0047636
9: -0.0259621, -0.0138315, -0.0266791, -0.0140755, -0.0118866, 0.0128477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070810, upper bound: 0.0070861
time: 1.68 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070810, upper bound: 0.0070860
time: 1.50 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0046162, 0.0085432, 0.0046162, 0.0085432, -0.0039270, 0.0039270
1: 0.0005625, 0.0046295, 0.0005625, 0.0046295, -0.0040670, 0.0040670
2: -0.0238124, -0.0051909, -0.0238124, -0.0046524, -0.0191600, 0.0186215
3: -0.0020489, 0.0062880, -0.0020970, 0.0062880, -0.0083369, 0.0083850
4: 0.0113951, 0.0179913, 0.0111617, 0.0179913, -0.0065963, 0.0068297
5: -0.0032055, 0.0076055, -0.0032055, 0.0076055, -0.0108110, 0.0108110
6: 0.9943261, 1.0025772, 0.9942623, 1.0025769, -0.0082508, 0.0083149
7: 0.0072442, 0.0191846, 0.0068217, 0.0191846, -0.0082824, 0.0087416
8: 0.0025513, 0.0069987, 0.0025513, 0.0069987, -0.0044474, 0.0044474
9: -0.0259621, -0.0138315, -0.0259621, -0.0135673, -0.0123948, 0.0121307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070861
time: 1.32 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071911, upper bound: 0.0070861
time: 1.44 seconds

## BFS IS instance: IS_A1_A2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042873, 0.0085735, 0.0044524, 0.0085583, -0.0042710, 0.0041211
1: 0.0002493, 0.0046881, 0.0004065, 0.0046587, -0.0044094, 0.0042816
2: -0.0245964, -0.0064329, -0.0242028, -0.0047798, -0.0198166, 0.0177700
3: -0.0019380, 0.0069683, -0.0020857, 0.0066268, -0.0085648, 0.0090539
4: 0.0119333, 0.0181964, 0.0112169, 0.0180935, -0.0061602, 0.0069795
5: -0.0032361, 0.0085635, -0.0032207, 0.0080826, -0.0113187, 0.0117842
6: 0.9944735, 1.0032226, 0.9942774, 1.0028986, -0.0084251, 0.0089452
7: 0.0082184, 0.0195557, 0.0069217, 0.0193694, -0.0078948, 0.0092969
8: 0.0022609, 0.0071150, 0.0024067, 0.0070567, -0.0047958, 0.0047083
9: -0.0266207, -0.0144406, -0.0262901, -0.0136298, -0.0129909, 0.0118495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 165

## Relational analysis of IS_A1_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072222, upper bound: 0.0071760
time: 1.56 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072222, upper bound: 0.0071760
time: 1.35 seconds

## BFS IS instance: IS_A1_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0044022, 0.0085629, 0.0045216, 0.0085519, -0.0041497, 0.0040413
1: 0.0003587, 0.0046676, 0.0004724, 0.0046463, -0.0042877, 0.0041952
2: -0.0243225, -0.0058793, -0.0240378, -0.0048413, -0.0194812, 0.0181585
3: -0.0019875, 0.0067307, -0.0020802, 0.0064836, -0.0084710, 0.0088108
4: 0.0116934, 0.0181247, 0.0112435, 0.0180503, -0.0063569, 0.0068812
5: -0.0032254, 0.0082289, -0.0032143, 0.0078809, -0.0111063, 0.0114432
6: 0.9944078, 1.0029970, 0.9942847, 1.0027627, -0.0083549, 0.0087124
7: 0.0077842, 0.0194261, 0.0069699, 0.0192913, -0.0081911, 0.0093105
8: 0.0023623, 0.0070744, 0.0024678, 0.0070322, -0.0046698, 0.0046066
9: -0.0263907, -0.0141691, -0.0261514, -0.0136600, -0.0127307, 0.0119823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 165

## Relational analysis of IS_A1_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070072, upper bound: 0.0071683
time: 1.57 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070072, upper bound: 0.0071760
time: 1.45 seconds

## BFS IS instance: IS_A1_A2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042910, 0.0085731, 0.0044572, 0.0085578, -0.0042668, 0.0041159
1: 0.0002528, 0.0046875, 0.0004111, 0.0046578, -0.0044051, 0.0042764
2: -0.0245876, -0.0062873, -0.0241913, -0.0047862, -0.0198013, 0.0179039
3: -0.0019510, 0.0069607, -0.0020851, 0.0066168, -0.0085678, 0.0090457
4: 0.0118702, 0.0181941, 0.0112197, 0.0180904, -0.0062203, 0.0069744
5: -0.0032357, 0.0085528, -0.0032203, 0.0080685, -0.0113043, 0.0117730
6: 0.9944563, 1.0032153, 0.9942782, 1.0028889, -0.0084326, 0.0089371
7: 0.0081042, 0.0195515, 0.0069267, 0.0193639, -0.0080256, 0.0093076
8: 0.0022641, 0.0071137, 0.0024109, 0.0070549, -0.0047908, 0.0047028
9: -0.0266133, -0.0143692, -0.0262804, -0.0136330, -0.0129804, 0.0119112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 165

## Relational analysis of IS_A1_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072299, upper bound: 0.0072239
time: 1.46 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072299, upper bound: 0.0072239
time: 1.36 seconds

## BFS IS instance: IS_A1_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0044041, 0.0085627, 0.0045264, 0.0085515, -0.0041473, 0.0040363
1: 0.0003605, 0.0046673, 0.0004770, 0.0046455, -0.0042850, 0.0041903
2: -0.0243179, -0.0057134, -0.0240263, -0.0048473, -0.0194707, 0.0183129
3: -0.0020023, 0.0067267, -0.0020796, 0.0064736, -0.0084759, 0.0088063
4: 0.0116215, 0.0181236, 0.0112461, 0.0180473, -0.0064258, 0.0068774
5: -0.0032252, 0.0082233, -0.0032138, 0.0078669, -0.0110921, 0.0114371
6: 0.9943882, 1.0029933, 0.9942854, 1.0027531, -0.0083650, 0.0087079
7: 0.0076540, 0.0194239, 0.0069746, 0.0192858, -0.0083164, 0.0093251
8: 0.0023640, 0.0070737, 0.0024721, 0.0070305, -0.0046665, 0.0046016
9: -0.0263868, -0.0140877, -0.0261418, -0.0136629, -0.0127239, 0.0120541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070084, upper bound: 0.0072162
time: 1.22 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070084, upper bound: 0.0072239
time: 1.29 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0044489, 0.0085586, 0.0042717, 0.0085749, -0.0041261, 0.0042869
1: 0.0004031, 0.0046593, 0.0002344, 0.0046909, -0.0042878, 0.0044250
2: -0.0242112, -0.0052588, -0.0246337, -0.0059796, -0.0182316, 0.0193748
3: -0.0020429, 0.0066341, -0.0019785, 0.0070006, -0.0090435, 0.0086126
4: 0.0114245, 0.0180957, 0.0117368, 0.0182061, -0.0067816, 0.0063588
5: -0.0032210, 0.0080929, -0.0032375, 0.0086091, -0.0118301, 0.0113304
6: 0.9943342, 1.0029055, 0.9944198, 1.0032532, -0.0089190, 0.0084857
7: 0.0072974, 0.0193734, 0.0078628, 0.0195734, -0.0090152, 0.0082054
8: 0.0024036, 0.0070579, 0.0022470, 0.0071205, -0.0047169, 0.0048109
9: -0.0262972, -0.0138648, -0.0266520, -0.0142183, -0.0120789, 0.0127873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070196, upper bound: 0.0070013
time: 1.41 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070196, upper bound: 0.0070013
time: 1.45 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0044489, 0.0085586, 0.0046207, 0.0085428, -0.0040939, 0.0039379
1: 0.0004031, 0.0046593, 0.0005669, 0.0046287, -0.0042255, 0.0040925
2: -0.0242112, -0.0052588, -0.0238015, -0.0049091, -0.0193022, 0.0185426
3: -0.0020429, 0.0066341, -0.0020741, 0.0062785, -0.0083214, 0.0087082
4: 0.0114245, 0.0180957, 0.0112729, 0.0179885, -0.0065640, 0.0068227
5: -0.0032210, 0.0080929, -0.0032050, 0.0075922, -0.0108132, 0.0112979
6: 0.9943342, 1.0029055, 0.9942927, 1.0025680, -0.0082338, 0.0086128
7: 0.0072974, 0.0193734, 0.0070231, 0.0191794, -0.0082814, 0.0084740
8: 0.0024036, 0.0070579, 0.0025553, 0.0069971, -0.0045935, 0.0045026
9: -0.0262972, -0.0138648, -0.0259529, -0.0136932, -0.0126040, 0.0120882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
time: 1.25 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071359, upper bound: 0.0070013
time: 1.29 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0045521, 0.0085491, 0.0042858, 0.0085736, -0.0040215, 0.0042633
1: 0.0005015, 0.0046409, 0.0002478, 0.0046884, -0.0041869, 0.0043931
2: -0.0239652, -0.0053220, -0.0245999, -0.0058312, -0.0181340, 0.0192779
3: -0.0020372, 0.0064206, -0.0019918, 0.0069714, -0.0090086, 0.0084123
4: 0.0114518, 0.0180313, 0.0116725, 0.0181973, -0.0067455, 0.0063588
5: -0.0032114, 0.0077922, -0.0032362, 0.0085678, -0.0117793, 0.0110284
6: 0.9943417, 1.0027028, 0.9944022, 1.0032256, -0.0088839, 0.0083007
7: 0.0073470, 0.0192569, 0.0077464, 0.0195574, -0.0089327, 0.0084616
8: 0.0024947, 0.0070214, 0.0022596, 0.0071155, -0.0046208, 0.0047618
9: -0.0260904, -0.0138957, -0.0266237, -0.0141455, -0.0119449, 0.0127279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071855, upper bound: 0.0070025
time: 1.43 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071855, upper bound: 0.0070025
time: 1.14 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0046338, 0.0085416, 0.0042858, 0.0085736, -0.0039398, 0.0042557
1: 0.0005793, 0.0046263, 0.0002478, 0.0046884, -0.0041091, 0.0043785
2: -0.0237703, -0.0047292, -0.0245999, -0.0058312, -0.0179391, 0.0198707
3: -0.0020902, 0.0062515, -0.0019918, 0.0069714, -0.0090615, 0.0082432
4: 0.0111950, 0.0179803, 0.0116725, 0.0181973, -0.0070023, 0.0063078
5: -0.0032038, 0.0075541, -0.0032362, 0.0085678, -0.0117717, 0.0107903
6: 0.9942714, 1.0025424, 0.9944022, 1.0032256, -0.0089542, 0.0081402
7: 0.0068820, 0.0191647, 0.0077464, 0.0195574, -0.0093506, 0.0083530
8: 0.0025669, 0.0069925, 0.0022596, 0.0071155, -0.0045486, 0.0047329
9: -0.0259268, -0.0136050, -0.0266237, -0.0141455, -0.0117813, 0.0130187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071855, upper bound: 0.0070025
time: 1.31 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071855, upper bound: 0.0070025
time: 1.53 seconds

## BFS IS instance: IS_A2_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0045672, 0.0085477, 0.0041508, 0.0085860, -0.0040189, 0.0043969
1: 0.0005158, 0.0046382, 0.0001192, 0.0047125, -0.0041967, 0.0045190
2: -0.0239292, -0.0049969, -0.0249219, -0.0057717, -0.0181576, 0.0199251
3: -0.0020663, 0.0063894, -0.0019971, 0.0072508, -0.0093171, 0.0083864
4: 0.0113110, 0.0180219, 0.0116467, 0.0182815, -0.0069705, 0.0063752
5: -0.0032100, 0.0077483, -0.0032488, 0.0089614, -0.0121714, 0.0109971
6: 0.9943031, 1.0026733, 0.9943951, 1.0034906, -0.0091875, 0.0082782
7: 0.0070920, 0.0192399, 0.0076997, 0.0197098, -0.0091150, 0.0081663
8: 0.0025080, 0.0070161, 0.0021403, 0.0071633, -0.0046553, 0.0048758
9: -0.0260603, -0.0137363, -0.0268942, -0.0141163, -0.0119440, 0.0131579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073307, upper bound: 0.0074339
time: 1.25 seconds

## Relational analysis of IS_A2_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073307, upper bound: 0.0075181
time: 1.28 seconds

## BFS IS instance: IS_A2_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0046353, 0.0085414, 0.0042483, 0.0085771, -0.0039418, 0.0042931
1: 0.0005807, 0.0046261, 0.0002121, 0.0046951, -0.0041144, 0.0044140
2: -0.0237668, -0.0050564, -0.0246893, -0.0051625, -0.0186043, 0.0196329
3: -0.0020610, 0.0062484, -0.0020515, 0.0070489, -0.0091099, 0.0082999
4: 0.0113368, 0.0179794, 0.0113828, 0.0182207, -0.0068839, 0.0065967
5: -0.0032037, 0.0075498, -0.0032397, 0.0086771, -0.0118808, 0.0107895
6: 0.9943102, 1.0025395, 0.9943228, 1.0032990, -0.0089888, 0.0082167
7: 0.0071387, 0.0191630, 0.0072219, 0.0195997, -0.0091378, 0.0084443
8: 0.0025682, 0.0069920, 0.0022264, 0.0071288, -0.0045606, 0.0047655
9: -0.0259238, -0.0137655, -0.0266988, -0.0138175, -0.0121063, 0.0129333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072645, upper bound: 0.0072977
time: 1.33 seconds

## Relational analysis of IS_A2_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072645, upper bound: 0.0074655
time: 1.54 seconds

## BFS IS instance: IS_A2_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0045623, 0.0085481, 0.0041557, 0.0085856, -0.0040233, 0.0043925
1: 0.0005112, 0.0046391, 0.0001239, 0.0047116, -0.0042004, 0.0045152
2: -0.0239407, -0.0048192, -0.0249102, -0.0057777, -0.0181630, 0.0200910
3: -0.0020821, 0.0063993, -0.0019965, 0.0072406, -0.0093228, 0.0083959
4: 0.0112340, 0.0180249, 0.0116493, 0.0182784, -0.0070445, 0.0063756
5: -0.0032105, 0.0077623, -0.0032483, 0.0089470, -0.0121575, 0.0110106
6: 0.9942821, 1.0026828, 0.9943958, 1.0034808, -0.0091987, 0.0082870
7: 0.0069526, 0.0192453, 0.0077044, 0.0197043, -0.0092521, 0.0081827
8: 0.0025038, 0.0070178, 0.0021446, 0.0071615, -0.0046578, 0.0048732
9: -0.0260699, -0.0136491, -0.0268843, -0.0141193, -0.0119506, 0.0132352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073371, upper bound: 0.0074540
time: 1.33 seconds

## Relational analysis of IS_A2_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073371, upper bound: 0.0075429
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0046310, 0.0085418, 0.0042514, 0.0085768, -0.0039458, 0.0042905
1: 0.0005766, 0.0046268, 0.0002150, 0.0046945, -0.0041179, 0.0044119
2: -0.0237771, -0.0048796, -0.0246821, -0.0051682, -0.0186089, 0.0198025
3: -0.0020767, 0.0062574, -0.0020510, 0.0070427, -0.0091195, 0.0083083
4: 0.0112601, 0.0179821, 0.0113852, 0.0182188, -0.0069586, 0.0065969
5: -0.0032041, 0.0075624, -0.0032394, 0.0086683, -0.0118724, 0.0108018
6: 0.9942892, 1.0025481, 0.9943234, 1.0032932, -0.0090040, 0.0082247
7: 0.0070000, 0.0191679, 0.0072263, 0.0195963, -0.0092757, 0.0084577
8: 0.0025644, 0.0069935, 0.0022291, 0.0071277, -0.0045633, 0.0047644
9: -0.0259325, -0.0136787, -0.0266927, -0.0138203, -0.0121122, 0.0130140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072679, upper bound: 0.0073125
time: 1.24 seconds

## Relational analysis of IS_A2_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072679, upper bound: 0.0074873
time: 1.39 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0043284, 0.0085697, 0.0044072, 0.0085624, -0.0042340, 0.0041625
1: 0.0002884, 0.0046808, 0.0003635, 0.0046667, -0.0043784, 0.0043173
2: -0.0244984, -0.0060913, -0.0243105, -0.0046788, -0.0198196, 0.0182192
3: -0.0019685, 0.0068833, -0.0020947, 0.0067203, -0.0086888, 0.0089780
4: 0.0117852, 0.0181708, 0.0111731, 0.0181216, -0.0063364, 0.0069976
5: -0.0032322, 0.0084438, -0.0032249, 0.0082142, -0.0114465, 0.0116687
6: 0.9944330, 1.0031419, 0.9942654, 1.0029871, -0.0085542, 0.0088764
7: 0.0079505, 0.0195093, 0.0068425, 0.0194204, -0.0080907, 0.0092537
8: 0.0022972, 0.0071005, 0.0023668, 0.0070726, -0.0047754, 0.0047337
9: -0.0265384, -0.0142731, -0.0263806, -0.0135803, -0.0129582, 0.0121075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_A1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.22 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.28 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0046624, 0.0085389, 0.0044072, 0.0085624, -0.0039000, 0.0041317
1: 0.0006066, 0.0046212, 0.0003635, 0.0046667, -0.0040602, 0.0042577
2: -0.0237021, -0.0050510, -0.0243105, -0.0046788, -0.0190233, 0.0192595
3: -0.0020614, 0.0061923, -0.0020947, 0.0067203, -0.0087817, 0.0082870
4: 0.0113344, 0.0179625, 0.0111731, 0.0181216, -0.0067872, 0.0067894
5: -0.0032012, 0.0074708, -0.0032249, 0.0082142, -0.0114154, 0.0106957
6: 0.9943095, 1.0024862, 0.9942654, 1.0029871, -0.0086777, 0.0082208
7: 0.0071344, 0.0191324, 0.0068425, 0.0194204, -0.0083871, 0.0084767
8: 0.0025922, 0.0069824, 0.0023668, 0.0070726, -0.0044805, 0.0046156
9: -0.0258695, -0.0137628, -0.0263806, -0.0135803, -0.0122892, 0.0126177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.50 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073658, upper bound: 0.0074230
time: 1.27 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0044283, 0.0085605, 0.0044758, 0.0085561, -0.0041278, 0.0040847
1: 0.0003836, 0.0046630, 0.0004288, 0.0046545, -0.0042709, 0.0042341
2: -0.0242602, -0.0054772, -0.0241469, -0.0047380, -0.0195221, 0.0186697
3: -0.0020234, 0.0066766, -0.0020894, 0.0065783, -0.0086016, 0.0087660
4: 0.0115191, 0.0181085, 0.0111988, 0.0180788, -0.0065597, 0.0069096
5: -0.0032229, 0.0081527, -0.0032185, 0.0080143, -0.0112373, 0.0113712
6: 0.9943601, 1.0029458, 0.9942725, 1.0028524, -0.0084923, 0.0086733
7: 0.0074688, 0.0193966, 0.0068889, 0.0193430, -0.0085137, 0.0092064
8: 0.0023855, 0.0070652, 0.0024274, 0.0070484, -0.0046629, 0.0046377
9: -0.0263383, -0.0139719, -0.0262431, -0.0136093, -0.0127290, 0.0122712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_A1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
time: 1.59 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073240, upper bound: 0.0073920
time: 1.46 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0047348, 0.0085323, 0.0044758, 0.0085561, -0.0038213, 0.0040564
1: 0.0006756, 0.0046083, 0.0004288, 0.0046545, -0.0039789, 0.0041795
2: -0.0235295, -0.0044172, -0.0241469, -0.0047380, -0.0187914, 0.0197297
3: -0.0021180, 0.0060425, -0.0020894, 0.0065783, -0.0086963, 0.0081319
4: 0.0110598, 0.0179174, 0.0111988, 0.0180788, -0.0070190, 0.0067186
5: -0.0031944, 0.0072598, -0.0032185, 0.0080143, -0.0112087, 0.0104783
6: 0.9942344, 1.0023441, 0.9942725, 1.0028524, -0.0086181, 0.0080717
7: 0.0066373, 0.0190507, 0.0068889, 0.0193430, -0.0089823, 0.0085104
8: 0.0026561, 0.0069568, 0.0024274, 0.0070484, -0.0043922, 0.0045294
9: -0.0257244, -0.0134520, -0.0262431, -0.0136093, -0.0121151, 0.0127912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069786, upper bound: 0.0072057
time: 1.17 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069786, upper bound: 0.0073920
time: 0.99 seconds

## BFS IS instance: IS_A2_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0043151, 0.0085709, 0.0044120, 0.0085620, -0.0042469, 0.0041589
1: 0.0002757, 0.0046832, 0.0003680, 0.0046659, -0.0043902, 0.0043152
2: -0.0245302, -0.0059025, -0.0242992, -0.0046859, -0.0198443, 0.0183966
3: -0.0019854, 0.0069109, -0.0020940, 0.0067104, -0.0086958, 0.0090049
4: 0.0117034, 0.0181790, 0.0111762, 0.0181186, -0.0064152, 0.0070028
5: -0.0032335, 0.0084826, -0.0032245, 0.0082004, -0.0114339, 0.0117071
6: 0.9944106, 1.0031681, 0.9942662, 1.0029778, -0.0085672, 0.0089019
7: 0.0078024, 0.0195244, 0.0068480, 0.0194150, -0.0082396, 0.0092678
8: 0.0022854, 0.0071052, 0.0023710, 0.0070709, -0.0047855, 0.0047342
9: -0.0265651, -0.0141805, -0.0263710, -0.0135838, -0.0129813, 0.0121905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A2_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.03 seconds

## Relational analysis of IS_A2_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.37 seconds

## BFS IS instance: IS_A2_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0046580, 0.0085393, 0.0044120, 0.0085620, -0.0039039, 0.0041274
1: 0.0006024, 0.0046220, 0.0003680, 0.0046659, -0.0040634, 0.0042540
2: -0.0237125, -0.0048741, -0.0242992, -0.0046859, -0.0190265, 0.0194251
3: -0.0020772, 0.0062013, -0.0020940, 0.0067104, -0.0087876, 0.0082953
4: 0.0112578, 0.0179652, 0.0111762, 0.0181186, -0.0068609, 0.0067890
5: -0.0032016, 0.0074834, -0.0032245, 0.0082004, -0.0114019, 0.0107078
6: 0.9942886, 1.0024948, 0.9942662, 1.0029778, -0.0086893, 0.0082286
7: 0.0069956, 0.0191373, 0.0068480, 0.0194150, -0.0085254, 0.0084870
8: 0.0025884, 0.0069839, 0.0023710, 0.0070709, -0.0044826, 0.0046129
9: -0.0258781, -0.0136760, -0.0263710, -0.0135838, -0.0122944, 0.0126950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.12 seconds

## Relational analysis of IS_A2_A1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073665, upper bound: 0.0074508
time: 1.24 seconds

## BFS IS instance: IS_A2_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0044149, 0.0085617, 0.0044811, 0.0085556, -0.0041407, 0.0040806
1: 0.0003708, 0.0046654, 0.0004339, 0.0046535, -0.0042828, 0.0042315
2: -0.0242921, -0.0052790, -0.0241342, -0.0047448, -0.0195474, 0.0188553
3: -0.0020411, 0.0067043, -0.0020888, 0.0065673, -0.0086083, 0.0087931
4: 0.0114332, 0.0181168, 0.0112017, 0.0180755, -0.0066423, 0.0069151
5: -0.0032242, 0.0081917, -0.0032180, 0.0079988, -0.0112230, 0.0114098
6: 0.9943366, 1.0029720, 0.9942732, 1.0028421, -0.0085055, 0.0086988
7: 0.0073133, 0.0194117, 0.0068942, 0.0193369, -0.0086468, 0.0093006
8: 0.0023736, 0.0070699, 0.0024321, 0.0070465, -0.0046729, 0.0046378
9: -0.0263651, -0.0138747, -0.0262325, -0.0136126, -0.0127525, 0.0123578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069786, upper bound: 0.0072196
time: 1.25 seconds

## Relational analysis of IS_A2_A1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069786, upper bound: 0.0074154
time: 1.29 seconds

## BFS IS instance: IS_A2_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0047312, 0.0085326, 0.0044811, 0.0085556, -0.0038244, 0.0040514
1: 0.0006721, 0.0046089, 0.0004339, 0.0046535, -0.0039814, 0.0041750
2: -0.0235381, -0.0042390, -0.0241342, -0.0047448, -0.0187933, 0.0198953
3: -0.0021340, 0.0060499, -0.0020888, 0.0065673, -0.0087012, 0.0081387
4: 0.0109825, 0.0179196, 0.0112017, 0.0180755, -0.0070930, 0.0067179
5: -0.0031948, 0.0072702, -0.0032180, 0.0079988, -0.0111936, 0.0104883
6: 0.9942132, 1.0023512, 0.9942732, 1.0028421, -0.0086289, 0.0080780
7: 0.0064974, 0.0190547, 0.0068942, 0.0193369, -0.0091106, 0.0085234
8: 0.0026530, 0.0069581, 0.0024321, 0.0070465, -0.0043935, 0.0045260
9: -0.0257316, -0.0133645, -0.0262325, -0.0136126, -0.0121190, 0.0128680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069786, upper bound: 0.0072196
time: 1.10 seconds

## Relational analysis of IS_A2_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069786, upper bound: 0.0074154
time: 1.35 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0041907, 0.0085824, 0.0044094, 0.0085622, -0.0043715, 0.0041730
1: 0.0001573, 0.0047054, 0.0003655, 0.0046664, -0.0045091, 0.0043398
2: -0.0248266, -0.0062010, -0.0243054, -0.0047620, -0.0200646, 0.0181044
3: -0.0019587, 0.0071681, -0.0020872, 0.0067158, -0.0086745, 0.0092553
4: 0.0118328, 0.0182566, 0.0112092, 0.0181203, -0.0062875, 0.0070474
5: -0.0032451, 0.0088449, -0.0032247, 0.0082079, -0.0114530, 0.0120696
6: 0.9944460, 1.0034121, 0.9942753, 1.0029830, -0.0085370, 0.0091368
7: 0.0080365, 0.0196647, 0.0069077, 0.0194180, -0.0080904, 0.0093223
8: 0.0021756, 0.0071492, 0.0023687, 0.0070719, -0.0048962, 0.0047805
9: -0.0268141, -0.0143269, -0.0263762, -0.0136211, -0.0131930, 0.0120494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 165

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073505, upper bound: 0.0074935
time: 1.41 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073505, upper bound: 0.0074935
time: 1.23 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042938, 0.0085729, 0.0044780, 0.0085559, -0.0042621, 0.0040949
1: 0.0002554, 0.0046870, 0.0004309, 0.0046541, -0.0043987, 0.0042561
2: -0.0245809, -0.0055800, -0.0241418, -0.0048221, -0.0197588, 0.0185618
3: -0.0020142, 0.0069549, -0.0020819, 0.0065738, -0.0085880, 0.0090367
4: 0.0115637, 0.0181923, 0.0112353, 0.0180775, -0.0065138, 0.0069571
5: -0.0032355, 0.0085446, -0.0032183, 0.0080079, -0.0112434, 0.0117630
6: 0.9943723, 1.0032097, 0.9942824, 1.0028483, -0.0084760, 0.0089273
7: 0.0075494, 0.0195484, 0.0069549, 0.0193405, -0.0085101, 0.0093401
8: 0.0022666, 0.0071127, 0.0024293, 0.0070476, -0.0047810, 0.0046834
9: -0.0266077, -0.0140223, -0.0262388, -0.0136506, -0.0129571, 0.0122165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070073, upper bound: 0.0072848
time: 1.29 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070073, upper bound: 0.0074935
time: 1.41 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0041065, 0.0085901, 0.0040790, 0.0085927, -0.0044862, 0.0045111
1: 0.0000769, 0.0047204, 0.0000508, 0.0047253, -0.0046483, 0.0046695
2: -0.0250276, -0.0059850, -0.0250930, -0.0059611, -0.0190665, 0.0191080
3: -0.0019780, 0.0073425, -0.0019802, 0.0073992, -0.0093772, 0.0093227
4: 0.0117391, 0.0183091, 0.0117288, 0.0183262, -0.0065871, 0.0065803
5: -0.0032529, 0.0090905, -0.0032555, 0.0091704, -0.0124233, 0.0123460
6: 0.9944204, 1.0035776, 0.9944175, 1.0036314, -0.0092109, 0.0091601
7: 0.0078670, 0.0197598, 0.0078484, 0.0197908, -0.0086583, 0.0084846
8: 0.0021011, 0.0071790, 0.0020769, 0.0071887, -0.0050875, 0.0051021
9: -0.0269830, -0.0142209, -0.0270379, -0.0142092, -0.0127737, 0.0128169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.16 + 596.66 = 600.82 seconds
