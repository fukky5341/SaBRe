## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00035568


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044901, -0.0012731, -0.0044901, -0.0012731, -0.0025021, 0.0025021)
1: (-0.0047059, -0.0041246, -0.0047059, -0.0041246, -0.0004523, 0.0004523)
2: (0.0088592, 0.0129371, 0.0088592, 0.0129371, -0.0031439, 0.0031439)
3: (1.0083171, 1.0092779, 1.0083171, 1.0092779, -0.0008201, 0.0008201)
4: (-0.0037102, -0.0030814, -0.0037102, -0.0030814, -0.0004812, 0.0004812)
5: (0.0005153, 0.0029743, 0.0005153, 0.0029743, -0.0019099, 0.0019099)
6: (-0.0025669, -0.0024359, -0.0025669, -0.0024359, -0.0001225, 0.0001225)
7: (-0.0115891, -0.0058695, -0.0115891, -0.0058695, -0.0046249, 0.0046249)
8: (-0.0076056, -0.0010421, -0.0076056, -0.0010421, -0.0050132, 0.0050132)
9: (-0.0036301, -0.0005268, -0.0036301, -0.0005268, -0.0023728, 0.0023728)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.74 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0004241, upper bound: 0.0004241

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004042, upper bound: 0.0004126
time: 0.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004126, upper bound: 0.0004126
time: 0.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.08
Output dim: 3, lower bound: -0.0004042, upper bound: 0.0004126
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.08
Output dim: 3, lower bound: -0.0004126, upper bound: 0.0004126

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0044861, -0.0013865, -0.0044886, -0.0013161, -0.0024554, 0.0023892
1: -0.0047046, -0.0041488, -0.0047054, -0.0041338, -0.0004419, 0.0004275
2: 0.0088646, 0.0127876, 0.0088612, 0.0128804, -0.0030826, 0.0029959
3: 1.0083489, 1.0092692, 1.0083289, 1.0092747, -0.0007845, 0.0008000
4: -0.0036861, -0.0030823, -0.0037011, -0.0030817, -0.0004573, 0.0004712
5: 0.0005184, 0.0028871, 0.0005165, 0.0029412, -0.0018741, 0.0018232
6: -0.0025668, -0.0024374, -0.0025669, -0.0024365, -0.0001209, 0.0001198
7: -0.0114210, -0.0058750, -0.0115251, -0.0058715, -0.0044481, 0.0045528
8: -0.0073502, -0.0010519, -0.0075091, -0.0010457, -0.0047574, 0.0049067
9: -0.0036252, -0.0006500, -0.0036283, -0.0005734, -0.0023215, 0.0022493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0004025
time: 0.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003944, upper bound: 0.0004024
time: 0.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0045542, -0.0013402, -0.0044875, -0.0013022, -0.0025435, 0.0024133
1: -0.0047221, -0.0041382, -0.0047052, -0.0041304, -0.0004628, 0.0004329
2: 0.0087752, 0.0128501, 0.0088626, 0.0128997, -0.0031996, 0.0030267
3: 1.0083383, 1.0092940, 1.0083259, 1.0092746, -0.0007909, 0.0008357
4: -0.0036964, -0.0030677, -0.0037042, -0.0030819, -0.0004622, 0.0004901
5: 0.0004661, 0.0029228, 0.0005173, 0.0029519, -0.0019419, 0.0018416
6: -0.0025672, -0.0024369, -0.0025669, -0.0024365, -0.0001210, 0.0001212
7: -0.0114838, -0.0057768, -0.0115418, -0.0058736, -0.0044904, 0.0046703
8: -0.0074595, -0.0008955, -0.0075430, -0.0010479, -0.0048099, 0.0051056
9: -0.0037024, -0.0005973, -0.0036272, -0.0005564, -0.0024172, 0.0022751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003996, upper bound: 0.0004025
time: 1.06 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004025, upper bound: 0.0004025
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.38 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -0.0003912, upper bound: 0.0004025
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -0.0003944, upper bound: 0.0004024
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -0.0003996, upper bound: 0.0004025
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -0.0004025, upper bound: 0.0004025

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0044817, -0.0014087, -0.0044760, -0.0013794, -0.0023873, 0.0023535
1: -0.0047037, -0.0041498, -0.0047028, -0.0041365, -0.0004332, 0.0004171
2: 0.0088702, 0.0127618, 0.0088774, 0.0128068, -0.0030042, 0.0029526
3: 1.0083497, 1.0092595, 1.0083309, 1.0092477, -0.0007545, 0.0007873
4: -0.0036827, -0.0030831, -0.0036911, -0.0030843, -0.0004509, 0.0004602
5: 0.0005218, 0.0028703, 0.0005261, 0.0028934, -0.0018226, 0.0017961
6: -0.0025666, -0.0024398, -0.0025662, -0.0024430, -0.0001147, 0.0001170
7: -0.0113645, -0.0058825, -0.0113658, -0.0058930, -0.0043700, 0.0043882
8: -0.0073174, -0.0010612, -0.0074139, -0.0010729, -0.0046883, 0.0047959
9: -0.0036208, -0.0006639, -0.0036153, -0.0006132, -0.0022704, 0.0022122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003811
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003789, upper bound: 0.0003921
time: 1.13 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0044810, -0.0014286, -0.0045148, -0.0013945, -0.0024051, 0.0023881
1: -0.0047035, -0.0041505, -0.0047052, -0.0041366, -0.0004379, 0.0004167
2: 0.0088711, 0.0127387, 0.0088363, 0.0127887, -0.0030268, 0.0029879
3: 1.0083499, 1.0092547, 1.0083127, 1.0092528, -0.0007624, 0.0008141
4: -0.0036794, -0.0030833, -0.0036887, -0.0030790, -0.0004549, 0.0004638
5: 0.0005223, 0.0028552, 0.0004972, 0.0028818, -0.0018362, 0.0018219
6: -0.0025665, -0.0024401, -0.0025731, -0.0024405, -0.0001168, 0.0001234
7: -0.0113289, -0.0058839, -0.0113631, -0.0057768, -0.0044875, 0.0044298
8: -0.0072863, -0.0010627, -0.0073908, -0.0010219, -0.0047176, 0.0048356
9: -0.0036201, -0.0006761, -0.0036371, -0.0006225, -0.0022896, 0.0022206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003811
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003822, upper bound: 0.0003921
time: 0.95 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045497, -0.0013631, -0.0044748, -0.0013663, -0.0024758, 0.0023774
1: -0.0047212, -0.0041391, -0.0047026, -0.0041331, -0.0004541, 0.0004226
2: 0.0087809, 0.0128238, 0.0088789, 0.0128251, -0.0031215, 0.0029832
3: 1.0083390, 1.0092846, 1.0083282, 1.0092477, -0.0007609, 0.0008240
4: -0.0036928, -0.0030686, -0.0036941, -0.0030845, -0.0004558, 0.0004789
5: 0.0004696, 0.0029055, 0.0005271, 0.0029036, -0.0018909, 0.0018144
6: -0.0025670, -0.0024393, -0.0025662, -0.0024430, -0.0001149, 0.0001184
7: -0.0114267, -0.0057845, -0.0113829, -0.0058953, -0.0044120, 0.0045036
8: -0.0074258, -0.0009049, -0.0074467, -0.0010754, -0.0047410, 0.0049935
9: -0.0036979, -0.0006112, -0.0036141, -0.0005963, -0.0023659, 0.0022378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003510, upper bound: 0.0003810
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003891, upper bound: 0.0003921
time: 1.00 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045491, -0.0013820, -0.0045137, -0.0013814, -0.0024939, 0.0024122
1: -0.0047210, -0.0041399, -0.0047050, -0.0041332, -0.0004588, 0.0004221
2: 0.0087816, 0.0128013, 0.0088377, 0.0128072, -0.0031443, 0.0030186
3: 1.0083393, 1.0092794, 1.0083098, 1.0092525, -0.0007687, 0.0008487
4: -0.0036897, -0.0030687, -0.0036918, -0.0030793, -0.0004599, 0.0004825
5: 0.0004701, 0.0028913, 0.0004981, 0.0028919, -0.0019047, 0.0018403
6: -0.0025669, -0.0024396, -0.0025731, -0.0024405, -0.0001169, 0.0001248
7: -0.0113929, -0.0057856, -0.0113792, -0.0057788, -0.0045300, 0.0045459
8: -0.0073959, -0.0009061, -0.0074248, -0.0010243, -0.0047701, 0.0050323
9: -0.0036973, -0.0006240, -0.0036360, -0.0006056, -0.0023853, 0.0022463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003810
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003921, upper bound: 0.0003921
time: 1.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003811
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003789, upper bound: 0.0003921
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003811
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003822, upper bound: 0.0003921
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003510, upper bound: 0.0003810
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003891, upper bound: 0.0003921
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003810
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 3, lower bound: -0.0003921, upper bound: 0.0003921

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045435, -0.0017486, -0.0044747, -0.0015034, -0.0022829, 0.0019936
1: -0.0047171, -0.0042215, -0.0047021, -0.0041607, -0.0003908, 0.0003403
2: 0.0087915, 0.0123194, 0.0088793, 0.0126483, -0.0028614, 0.0024933
3: 1.0084475, 1.0092778, 1.0083638, 1.0092422, -0.0006492, 0.0007322
4: -0.0036126, -0.0030709, -0.0036663, -0.0030846, -0.0003791, 0.0004357
5: 0.0004745, 0.0026096, 0.0005272, 0.0027985, -0.0017420, 0.0015203
6: -0.0025675, -0.0024450, -0.0025662, -0.0024457, -0.0001102, 0.0001118
7: -0.0108364, -0.0057729, -0.0111649, -0.0058940, -0.0038261, 0.0042392
8: -0.0065715, -0.0009324, -0.0071518, -0.0010771, -0.0039223, 0.0045105
9: -0.0036832, -0.0010233, -0.0036131, -0.0007379, -0.0021129, 0.0018386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002200, upper bound: 0.0003334
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0044813, -0.0014365, -0.0044760, -0.0013794, -0.0023784, 0.0020784
1: -0.0047035, -0.0041553, -0.0047028, -0.0041365, -0.0004330, 0.0003573
2: 0.0088708, 0.0127261, 0.0088774, 0.0128068, -0.0029966, 0.0025877
3: 1.0083574, 1.0092574, 1.0083309, 1.0092477, -0.0007127, 0.0007854
4: -0.0036771, -0.0030833, -0.0036911, -0.0030843, -0.0003922, 0.0004597
5: 0.0005221, 0.0028491, 0.0005261, 0.0028934, -0.0018161, 0.0015844
6: -0.0025666, -0.0024408, -0.0025662, -0.0024430, -0.0001132, 0.0001184
7: -0.0113172, -0.0058828, -0.0113658, -0.0058930, -0.0040137, 0.0043515
8: -0.0072583, -0.0010627, -0.0074139, -0.0010729, -0.0040594, 0.0047947
9: -0.0036200, -0.0006926, -0.0036153, -0.0006132, -0.0022697, 0.0019055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003697, upper bound: 0.0003551
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003697, upper bound: 0.0003921
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045431, -0.0017666, -0.0045133, -0.0015201, -0.0022654, 0.0020235
1: -0.0047169, -0.0042223, -0.0047044, -0.0041612, -0.0003869, 0.0003395
2: 0.0087921, 0.0122977, 0.0088384, 0.0126267, -0.0028422, 0.0025186
3: 1.0084478, 1.0092727, 1.0083460, 1.0092475, -0.0006589, 0.0007571
4: -0.0036095, -0.0030710, -0.0036631, -0.0030794, -0.0003812, 0.0004329
5: 0.0004749, 0.0025958, 0.0004984, 0.0027857, -0.0017290, 0.0015423
6: -0.0025674, -0.0024453, -0.0025731, -0.0024432, -0.0001127, 0.0001177
7: -0.0108045, -0.0057737, -0.0111569, -0.0057778, -0.0039451, 0.0042263
8: -0.0065441, -0.0009333, -0.0071231, -0.0010261, -0.0039366, 0.0044834
9: -0.0036828, -0.0010350, -0.0036347, -0.0007496, -0.0021006, 0.0018423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003795
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003811
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0044806, -0.0014565, -0.0045148, -0.0013945, -0.0023962, 0.0021238
1: -0.0047033, -0.0041560, -0.0047052, -0.0041366, -0.0004377, 0.0003569
2: 0.0088718, 0.0127028, 0.0088363, 0.0127887, -0.0030193, 0.0026322
3: 1.0083578, 1.0092523, 1.0083127, 1.0092528, -0.0007214, 0.0008122
4: -0.0036739, -0.0030834, -0.0036887, -0.0030790, -0.0003968, 0.0004633
5: 0.0005227, 0.0028339, 0.0004972, 0.0028818, -0.0018297, 0.0016180
6: -0.0025665, -0.0024411, -0.0025731, -0.0024405, -0.0001153, 0.0001247
7: -0.0112821, -0.0058842, -0.0113631, -0.0057768, -0.0041676, 0.0043932
8: -0.0072276, -0.0010642, -0.0073908, -0.0010219, -0.0040939, 0.0048343
9: -0.0036193, -0.0007047, -0.0036371, -0.0006225, -0.0022889, 0.0019162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003725, upper bound: 0.0003551
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003725, upper bound: 0.0003921
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046127, -0.0016985, -0.0044735, -0.0014906, -0.0023682, 0.0020178
1: -0.0047344, -0.0042109, -0.0047019, -0.0041574, -0.0004142, 0.0003450
2: 0.0087000, 0.0123871, 0.0088808, 0.0126657, -0.0029773, 0.0025241
3: 1.0084382, 1.0093015, 1.0083612, 1.0092422, -0.0006551, 0.0007690
4: -0.0036237, -0.0030560, -0.0036692, -0.0030849, -0.0003839, 0.0004552
5: 0.0004214, 0.0026482, 0.0005281, 0.0028083, -0.0018078, 0.0015388
6: -0.0025679, -0.0024444, -0.0025661, -0.0024456, -0.0001105, 0.0001127
7: -0.0109061, -0.0056815, -0.0111814, -0.0058962, -0.0038689, 0.0043581
8: -0.0066910, -0.0007723, -0.0071852, -0.0010795, -0.0039723, 0.0047235
9: -0.0037617, -0.0009662, -0.0036120, -0.0007215, -0.0022180, 0.0018619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0001892, upper bound: 0.0003014
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003718
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045492, -0.0013920, -0.0044748, -0.0013663, -0.0024670, 0.0021057
1: -0.0047209, -0.0041444, -0.0047026, -0.0041331, -0.0004538, 0.0003633
2: 0.0087816, 0.0127875, 0.0088789, 0.0128251, -0.0031140, 0.0026229
3: 1.0083463, 1.0092821, 1.0083282, 1.0092477, -0.0007201, 0.0008218
4: -0.0036871, -0.0030688, -0.0036941, -0.0030845, -0.0003979, 0.0004784
5: 0.0004700, 0.0028835, 0.0005271, 0.0029036, -0.0018844, 0.0016054
6: -0.0025670, -0.0024403, -0.0025662, -0.0024430, -0.0001133, 0.0001202
7: -0.0113785, -0.0057848, -0.0113829, -0.0058953, -0.0040665, 0.0044662
8: -0.0073661, -0.0009064, -0.0074467, -0.0010754, -0.0041208, 0.0049922
9: -0.0036971, -0.0006389, -0.0036141, -0.0005963, -0.0023652, 0.0019355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003783, upper bound: 0.0003529
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003783, upper bound: 0.0003921
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046122, -0.0017172, -0.0045122, -0.0015072, -0.0023389, 0.0020482
1: -0.0047342, -0.0042117, -0.0047043, -0.0041579, -0.0004090, 0.0003442
2: 0.0087006, 0.0123637, 0.0088399, 0.0126441, -0.0029443, 0.0025500
3: 1.0084386, 1.0092962, 1.0083429, 1.0092473, -0.0006647, 0.0007904
4: -0.0036205, -0.0030561, -0.0036662, -0.0030796, -0.0003860, 0.0004504
5: 0.0004218, 0.0026339, 0.0004993, 0.0027955, -0.0017860, 0.0015612
6: -0.0025678, -0.0024446, -0.0025731, -0.0024432, -0.0001126, 0.0001187
7: -0.0108794, -0.0056823, -0.0111742, -0.0057798, -0.0039884, 0.0043199
8: -0.0066610, -0.0007732, -0.0071558, -0.0010285, -0.0039861, 0.0046774
9: -0.0037612, -0.0009782, -0.0036336, -0.0007335, -0.0021981, 0.0018653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0001816, upper bound: 0.0002942
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003431, upper bound: 0.0003718
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045486, -0.0014108, -0.0045137, -0.0013814, -0.0024850, 0.0021510
1: -0.0047207, -0.0041451, -0.0047050, -0.0041332, -0.0004586, 0.0003629
2: 0.0087823, 0.0127647, 0.0088377, 0.0128072, -0.0031367, 0.0026673
3: 1.0083468, 1.0092769, 1.0083098, 1.0092525, -0.0007288, 0.0008465
4: -0.0036841, -0.0030689, -0.0036918, -0.0030793, -0.0004024, 0.0004820
5: 0.0004704, 0.0028692, 0.0004981, 0.0028919, -0.0018983, 0.0016388
6: -0.0025669, -0.0024406, -0.0025731, -0.0024405, -0.0001154, 0.0001265
7: -0.0113447, -0.0057859, -0.0113792, -0.0057788, -0.0042190, 0.0045085
8: -0.0073368, -0.0009075, -0.0074248, -0.0010243, -0.0041542, 0.0050309
9: -0.0036965, -0.0006516, -0.0036360, -0.0006056, -0.0023845, 0.0019450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003810, upper bound: 0.0003529
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003810, upper bound: 0.0003921
time: 1.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0002200, upper bound: 0.0003334
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003697, upper bound: 0.0003551
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003697, upper bound: 0.0003921
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003795
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003529, upper bound: 0.0003811
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003725, upper bound: 0.0003551
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003725, upper bound: 0.0003921
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0001892, upper bound: 0.0003014
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003718
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003783, upper bound: 0.0003529
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003783, upper bound: 0.0003921
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0001816, upper bound: 0.0002942
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003431, upper bound: 0.0003718
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003810, upper bound: 0.0003529
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -0.0003810, upper bound: 0.0003921

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045435, -0.0017486, -0.0044718, -0.0015093, -0.0022786, 0.0019904
1: -0.0047171, -0.0042215, -0.0046995, -0.0041608, -0.0003908, 0.0003343
2: 0.0087915, 0.0123194, 0.0088830, 0.0126445, -0.0028567, 0.0024890
3: 1.0084475, 1.0092778, 1.0083642, 1.0092345, -0.0006259, 0.0007319
4: -0.0036126, -0.0030709, -0.0036661, -0.0030854, -0.0003782, 0.0004353
5: 0.0004745, 0.0026096, 0.0005294, 0.0027943, -0.0017388, 0.0015179
6: -0.0025675, -0.0024450, -0.0025660, -0.0024468, -0.0001078, 0.0001116
7: -0.0108364, -0.0057729, -0.0111336, -0.0058992, -0.0038208, 0.0042154
8: -0.0065715, -0.0009324, -0.0071500, -0.0010886, -0.0039072, 0.0045070
9: -0.0036832, -0.0010233, -0.0036059, -0.0007386, -0.0021122, 0.0018268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003699
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044813, -0.0014365, -0.0045380, -0.0017213, -0.0020285, 0.0023515
1: -0.0047035, -0.0041553, -0.0047161, -0.0042084, -0.0003564, 0.0003975
2: 0.0088708, 0.0127261, 0.0087985, 0.0123625, -0.0025440, 0.0029477
3: 1.0083574, 1.0092574, 1.0084304, 1.0092649, -0.0007269, 0.0006857
4: -0.0036771, -0.0030833, -0.0036208, -0.0030720, -0.0004486, 0.0003883
5: 0.0005221, 0.0028491, 0.0004788, 0.0026311, -0.0015478, 0.0017946
6: -0.0025666, -0.0024408, -0.0025671, -0.0024484, -0.0001090, 0.0001150
7: -0.0113172, -0.0058828, -0.0108339, -0.0057823, -0.0043800, 0.0038198
8: -0.0072583, -0.0010627, -0.0066672, -0.0009439, -0.0046416, 0.0040302
9: -0.0036200, -0.0006926, -0.0036777, -0.0009725, -0.0018967, 0.0021726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0001838
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044813, -0.0014365, -0.0044756, -0.0014075, -0.0020947, 0.0020717
1: -0.0047035, -0.0041553, -0.0047026, -0.0041419, -0.0003727, 0.0003570
2: 0.0088708, 0.0127261, 0.0088779, 0.0127707, -0.0026221, 0.0025819
3: 1.0083574, 1.0092574, 1.0083388, 1.0092454, -0.0007107, 0.0007437
4: -0.0036771, -0.0030833, -0.0036854, -0.0030844, -0.0003918, 0.0003996
5: 0.0005221, 0.0028491, 0.0005265, 0.0028719, -0.0015980, 0.0015796
6: -0.0025666, -0.0024408, -0.0025662, -0.0024440, -0.0001150, 0.0001175
7: -0.0113172, -0.0058828, -0.0113177, -0.0058933, -0.0039906, 0.0039811
8: -0.0072583, -0.0010627, -0.0073544, -0.0010742, -0.0040581, 0.0041499
9: -0.0036200, -0.0006926, -0.0036146, -0.0006417, -0.0019568, 0.0019048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0003404
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003632
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045431, -0.0017666, -0.0045105, -0.0015903, -0.0022289, 0.0020205
1: -0.0047169, -0.0042223, -0.0047037, -0.0041763, -0.0003760, 0.0003387
2: 0.0087921, 0.0122977, 0.0088421, 0.0125340, -0.0027869, 0.0025146
3: 1.0084478, 1.0092727, 1.0083656, 1.0092418, -0.0006524, 0.0007375
4: -0.0036095, -0.0030710, -0.0036483, -0.0030800, -0.0003806, 0.0004230
5: 0.0004749, 0.0025958, 0.0005005, 0.0027317, -0.0017003, 0.0015400
6: -0.0025674, -0.0024453, -0.0025730, -0.0024442, -0.0001112, 0.0001176
7: -0.0108045, -0.0057737, -0.0110519, -0.0057815, -0.0039413, 0.0041776
8: -0.0065441, -0.0009333, -0.0069640, -0.0010326, -0.0039297, 0.0043701
9: -0.0036828, -0.0010350, -0.0036316, -0.0008270, -0.0020421, 0.0018389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003767
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003795
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045431, -0.0017666, -0.0045812, -0.0015459, -0.0022455, 0.0021004
1: -0.0047169, -0.0042223, -0.0047212, -0.0041660, -0.0003845, 0.0003572
2: 0.0087921, 0.0122977, 0.0087490, 0.0125938, -0.0028184, 0.0026206
3: 1.0084478, 1.0092727, 1.0083553, 1.0092663, -0.0006850, 0.0007504
4: -0.0036095, -0.0030710, -0.0036583, -0.0030651, -0.0003973, 0.0004297
5: 0.0004749, 0.0025958, 0.0004461, 0.0027659, -0.0017139, 0.0016016
6: -0.0025674, -0.0024453, -0.0025734, -0.0024436, -0.0001118, 0.0001179
7: -0.0108045, -0.0057737, -0.0111163, -0.0056873, -0.0040458, 0.0041826
8: -0.0065441, -0.0009333, -0.0070714, -0.0008742, -0.0041075, 0.0044521
9: -0.0036828, -0.0010350, -0.0037094, -0.0007750, -0.0020867, 0.0019251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003783
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003811
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044806, -0.0014565, -0.0045809, -0.0017365, -0.0020474, 0.0024013
1: -0.0047033, -0.0041560, -0.0047181, -0.0042090, -0.0003622, 0.0004170
2: 0.0088718, 0.0127028, 0.0087542, 0.0123408, -0.0025698, 0.0030037
3: 1.0083578, 1.0092523, 1.0084125, 1.0092688, -0.0007671, 0.0007123
4: -0.0036739, -0.0030834, -0.0036178, -0.0030667, -0.0004572, 0.0003925
5: 0.0005227, 0.0028339, 0.0004469, 0.0026192, -0.0015624, 0.0018318
6: -0.0025665, -0.0024411, -0.0025742, -0.0024462, -0.0001095, 0.0001216
7: -0.0112821, -0.0058842, -0.0108319, -0.0056603, -0.0045185, 0.0038451
8: -0.0072276, -0.0010642, -0.0066401, -0.0008957, -0.0047415, 0.0040760
9: -0.0036193, -0.0007047, -0.0036969, -0.0009838, -0.0019191, 0.0022300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002896, upper bound: 0.0001816
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003633, upper bound: 0.0003453
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044806, -0.0014565, -0.0045143, -0.0014226, -0.0021101, 0.0021175
1: -0.0047033, -0.0041560, -0.0047049, -0.0041420, -0.0003780, 0.0003567
2: 0.0088718, 0.0127028, 0.0088369, 0.0127531, -0.0026424, 0.0026265
3: 1.0083578, 1.0092523, 1.0083206, 1.0092505, -0.0007195, 0.0007705
4: -0.0036739, -0.0030834, -0.0036831, -0.0030792, -0.0003964, 0.0004029
5: 0.0005227, 0.0028339, 0.0004976, 0.0028603, -0.0016099, 0.0016133
6: -0.0025665, -0.0024411, -0.0025731, -0.0024414, -0.0001168, 0.0001238
7: -0.0112821, -0.0058842, -0.0113152, -0.0057771, -0.0041442, 0.0040199
8: -0.0072276, -0.0010642, -0.0073321, -0.0010232, -0.0040923, 0.0041862
9: -0.0036193, -0.0007047, -0.0036363, -0.0006511, -0.0019750, 0.0019154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002896, upper bound: 0.0003404
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003633, upper bound: 0.0003631
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046127, -0.0016985, -0.0044706, -0.0014965, -0.0023641, 0.0020146
1: -0.0047344, -0.0042109, -0.0046993, -0.0041575, -0.0004141, 0.0003387
2: 0.0087000, 0.0123871, 0.0088845, 0.0126619, -0.0029727, 0.0025198
3: 1.0084382, 1.0093015, 1.0083615, 1.0092344, -0.0006310, 0.0007687
4: -0.0036237, -0.0030560, -0.0036691, -0.0030857, -0.0003830, 0.0004547
5: 0.0004214, 0.0026482, 0.0005303, 0.0028041, -0.0018047, 0.0015364
6: -0.0025679, -0.0024444, -0.0025660, -0.0024468, -0.0001081, 0.0001126
7: -0.0109061, -0.0056815, -0.0111499, -0.0059014, -0.0038636, 0.0043327
8: -0.0066910, -0.0007723, -0.0071835, -0.0010910, -0.0039571, 0.0047200
9: -0.0037617, -0.0009662, -0.0036047, -0.0007222, -0.0022172, 0.0018500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003699
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003718
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045492, -0.0013920, -0.0045370, -0.0017055, -0.0021243, 0.0023750
1: -0.0047209, -0.0041444, -0.0047159, -0.0042049, -0.0003778, 0.0004024
2: 0.0087816, 0.0127875, 0.0087998, 0.0123842, -0.0026669, 0.0029774
3: 1.0083463, 1.0092821, 1.0084280, 1.0092651, -0.0007337, 0.0007216
4: -0.0036871, -0.0030688, -0.0036243, -0.0030722, -0.0004532, 0.0004075
5: 0.0004700, 0.0028835, 0.0004795, 0.0026434, -0.0016213, 0.0018125
6: -0.0025670, -0.0024403, -0.0025671, -0.0024484, -0.0001094, 0.0001163
7: -0.0113785, -0.0057848, -0.0108506, -0.0057840, -0.0044213, 0.0039491
8: -0.0073661, -0.0009064, -0.0067061, -0.0009460, -0.0046911, 0.0042359
9: -0.0036971, -0.0006389, -0.0036767, -0.0009538, -0.0019970, 0.0021961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002952, upper bound: 0.0001837
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003431
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045492, -0.0013920, -0.0044744, -0.0013942, -0.0021600, 0.0020991
1: -0.0047209, -0.0041444, -0.0047024, -0.0041384, -0.0003926, 0.0003631
2: 0.0087816, 0.0127875, 0.0088795, 0.0127896, -0.0027129, 0.0026170
3: 1.0083463, 1.0092821, 1.0083357, 1.0092453, -0.0007181, 0.0007776
4: -0.0036871, -0.0030688, -0.0036885, -0.0030846, -0.0003975, 0.0004150
5: 0.0004700, 0.0028835, 0.0005274, 0.0028822, -0.0016487, 0.0016006
6: -0.0025670, -0.0024403, -0.0025662, -0.0024440, -0.0001146, 0.0001193
7: -0.0113785, -0.0057848, -0.0113348, -0.0058955, -0.0040435, 0.0040378
8: -0.0073661, -0.0009064, -0.0073869, -0.0010766, -0.0041195, 0.0043204
9: -0.0036971, -0.0006389, -0.0036135, -0.0006252, -0.0020425, 0.0019348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002952, upper bound: 0.0003404
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003632
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046122, -0.0017172, -0.0045092, -0.0015130, -0.0023320, 0.0020451
1: -0.0047342, -0.0042117, -0.0047017, -0.0041580, -0.0004089, 0.0003381
2: 0.0087006, 0.0123637, 0.0088437, 0.0126405, -0.0029368, 0.0025458
3: 1.0084386, 1.0092962, 1.0083432, 1.0092396, -0.0006394, 0.0007901
4: -0.0036205, -0.0030561, -0.0036660, -0.0030804, -0.0003851, 0.0004495
5: 0.0004218, 0.0026339, 0.0005016, 0.0027914, -0.0017808, 0.0015588
6: -0.0025678, -0.0024446, -0.0025729, -0.0024444, -0.0001101, 0.0001186
7: -0.0108794, -0.0056823, -0.0111450, -0.0057849, -0.0039833, 0.0043014
8: -0.0066610, -0.0007732, -0.0071540, -0.0010399, -0.0039712, 0.0046695
9: -0.0037612, -0.0009782, -0.0036264, -0.0007342, -0.0021954, 0.0018539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003691
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003718
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045486, -0.0014108, -0.0045800, -0.0017219, -0.0021433, 0.0024253
1: -0.0047207, -0.0041451, -0.0047180, -0.0042055, -0.0003832, 0.0004225
2: 0.0087823, 0.0127647, 0.0087554, 0.0123613, -0.0026925, 0.0030342
3: 1.0083468, 1.0092769, 1.0084100, 1.0092689, -0.0007739, 0.0007457
4: -0.0036841, -0.0030689, -0.0036212, -0.0030669, -0.0004621, 0.0004117
5: 0.0004704, 0.0028692, 0.0004476, 0.0026306, -0.0016360, 0.0018502
6: -0.0025669, -0.0024406, -0.0025741, -0.0024461, -0.0001099, 0.0001229
7: -0.0113447, -0.0057859, -0.0108526, -0.0056620, -0.0045596, 0.0039758
8: -0.0073368, -0.0009075, -0.0066777, -0.0008975, -0.0047943, 0.0042804
9: -0.0036965, -0.0006516, -0.0036960, -0.0009652, -0.0020189, 0.0022560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002942, upper bound: 0.0001816
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003718, upper bound: 0.0003431
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045486, -0.0014108, -0.0045132, -0.0014102, -0.0021758, 0.0021447
1: -0.0047207, -0.0041451, -0.0047048, -0.0041385, -0.0003975, 0.0003627
2: 0.0087823, 0.0127647, 0.0088384, 0.0127703, -0.0027335, 0.0026616
3: 1.0083468, 1.0092769, 1.0083177, 1.0092503, -0.0007268, 0.0008033
4: -0.0036841, -0.0030689, -0.0036861, -0.0030794, -0.0004020, 0.0004184
5: 0.0004704, 0.0028692, 0.0004985, 0.0028700, -0.0016607, 0.0016341
6: -0.0025669, -0.0024406, -0.0025731, -0.0024414, -0.0001165, 0.0001256
7: -0.0113447, -0.0057859, -0.0113313, -0.0057791, -0.0041956, 0.0040756
8: -0.0073368, -0.0009075, -0.0073648, -0.0010256, -0.0041526, 0.0043581
9: -0.0036965, -0.0006516, -0.0036352, -0.0006344, -0.0020609, 0.0019442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002942, upper bound: 0.0003404
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003718, upper bound: 0.0003631
time: 1.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.49 seconds
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003699
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0001838
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002906, upper bound: 0.0003404
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003632
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003767
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003795
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003783
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003811
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002896, upper bound: 0.0001816
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003633, upper bound: 0.0003453
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002896, upper bound: 0.0003404
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003633, upper bound: 0.0003631
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003699
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003718
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002952, upper bound: 0.0001837
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003431
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002952, upper bound: 0.0003404
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003632
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003691
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003718
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002942, upper bound: 0.0001816
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003718, upper bound: 0.0003431
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0002942, upper bound: 0.0003404
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 3, lower bound: -0.0003718, upper bound: 0.0003631

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0044718, -0.0015093, -0.0022702, 0.0019496
1: -0.0047156, -0.0042235, -0.0046995, -0.0041608, -0.0003881, 0.0003303
2: 0.0088011, 0.0122697, 0.0088830, 0.0126445, -0.0028460, 0.0024404
3: 1.0084491, 1.0092607, 1.0083642, 1.0092345, -0.0006243, 0.0007136
4: -0.0036057, -0.0030725, -0.0036661, -0.0030854, -0.0003713, 0.0004336
5: 0.0004804, 0.0025772, 0.0005294, 0.0027943, -0.0017324, 0.0014871
6: -0.0025671, -0.0024493, -0.0025660, -0.0024468, -0.0001074, 0.0001076
7: -0.0107309, -0.0057858, -0.0111336, -0.0058992, -0.0037129, 0.0042017
8: -0.0065058, -0.0009483, -0.0071500, -0.0010886, -0.0038392, 0.0044893
9: -0.0036756, -0.0010505, -0.0036059, -0.0007386, -0.0021038, 0.0017972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003684
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003699
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0044718, -0.0015093, -0.0023198, 0.0019366
1: -0.0047167, -0.0042241, -0.0046995, -0.0041608, -0.0003888, 0.0003290
2: 0.0087601, 0.0122493, 0.0088830, 0.0126445, -0.0028995, 0.0024204
3: 1.0084313, 1.0092610, 1.0083642, 1.0092345, -0.0006514, 0.0007206
4: -0.0036029, -0.0030677, -0.0036661, -0.0030854, -0.0003677, 0.0004401
5: 0.0004501, 0.0025657, 0.0005294, 0.0027943, -0.0017695, 0.0014768
6: -0.0025741, -0.0024475, -0.0025660, -0.0024468, -0.0001139, 0.0001087
7: -0.0107230, -0.0056637, -0.0111336, -0.0058992, -0.0037201, 0.0043523
8: -0.0064797, -0.0009069, -0.0071500, -0.0010886, -0.0037997, 0.0045441
9: -0.0036914, -0.0010620, -0.0036059, -0.0007386, -0.0021228, 0.0017787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003703
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0044785, -0.0014426, -0.0045380, -0.0017213, -0.0020254, 0.0023443
1: -0.0047008, -0.0041554, -0.0047161, -0.0042084, -0.0003500, 0.0003975
2: 0.0088744, 0.0127221, 0.0087985, 0.0123625, -0.0025397, 0.0029397
3: 1.0083578, 1.0092496, 1.0084304, 1.0092649, -0.0007266, 0.0006610
4: -0.0036770, -0.0030841, -0.0036208, -0.0030720, -0.0004476, 0.0003874
5: 0.0005243, 0.0028447, 0.0004788, 0.0026311, -0.0015454, 0.0017892
6: -0.0025665, -0.0024420, -0.0025671, -0.0024484, -0.0001089, 0.0001125
7: -0.0112862, -0.0058879, -0.0108339, -0.0057823, -0.0043555, 0.0038146
8: -0.0072567, -0.0010742, -0.0066672, -0.0009439, -0.0046330, 0.0040149
9: -0.0036128, -0.0006933, -0.0036777, -0.0009725, -0.0018848, 0.0021697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0044785, -0.0014426, -0.0044756, -0.0014075, -0.0020919, 0.0020668
1: -0.0047008, -0.0041554, -0.0047026, -0.0041419, -0.0003656, 0.0003570
2: 0.0088744, 0.0127221, 0.0088779, 0.0127707, -0.0026177, 0.0025776
3: 1.0083578, 1.0092496, 1.0083388, 1.0092454, -0.0007104, 0.0007195
4: -0.0036770, -0.0030841, -0.0036854, -0.0030844, -0.0003915, 0.0003982
5: 0.0005243, 0.0028447, 0.0005265, 0.0028719, -0.0015959, 0.0015760
6: -0.0025665, -0.0024420, -0.0025662, -0.0024440, -0.0001149, 0.0001143
7: -0.0112862, -0.0058879, -0.0113177, -0.0058933, -0.0039626, 0.0039763
8: -0.0072567, -0.0010742, -0.0073544, -0.0010742, -0.0040564, 0.0041284
9: -0.0036128, -0.0006933, -0.0036146, -0.0006417, -0.0019425, 0.0019041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003564
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003632
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0045105, -0.0015903, -0.0021927, 0.0019972
1: -0.0047156, -0.0042235, -0.0047037, -0.0041763, -0.0003717, 0.0003373
2: 0.0088011, 0.0122697, 0.0088421, 0.0125340, -0.0027419, 0.0024889
3: 1.0084491, 1.0092607, 1.0083656, 1.0092418, -0.0006519, 0.0007209
4: -0.0036057, -0.0030725, -0.0036483, -0.0030800, -0.0003773, 0.0004164
5: 0.0004804, 0.0025772, 0.0005005, 0.0027317, -0.0016726, 0.0015225
6: -0.0025671, -0.0024493, -0.0025730, -0.0024442, -0.0001097, 0.0001142
7: -0.0107309, -0.0057858, -0.0110519, -0.0057815, -0.0038591, 0.0041124
8: -0.0065058, -0.0009483, -0.0069640, -0.0010326, -0.0038993, 0.0043044
9: -0.0036756, -0.0010505, -0.0036316, -0.0008270, -0.0020119, 0.0018265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002477, upper bound: 0.0003435
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003720
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0045105, -0.0015903, -0.0022311, 0.0019780
1: -0.0047167, -0.0042241, -0.0047037, -0.0041763, -0.0003742, 0.0003435
2: 0.0087601, 0.0122493, 0.0088421, 0.0125340, -0.0027869, 0.0024760
3: 1.0084313, 1.0092610, 1.0083656, 1.0092418, -0.0006769, 0.0007065
4: -0.0036029, -0.0030677, -0.0036483, -0.0030800, -0.0003773, 0.0004226
5: 0.0004501, 0.0025657, 0.0005005, 0.0027317, -0.0017017, 0.0015088
6: -0.0025741, -0.0024475, -0.0025730, -0.0024442, -0.0001137, 0.0001118
7: -0.0107230, -0.0056637, -0.0110519, -0.0057815, -0.0037852, 0.0042058
8: -0.0064797, -0.0009069, -0.0069640, -0.0010326, -0.0039093, 0.0043633
9: -0.0036914, -0.0010620, -0.0036316, -0.0008270, -0.0020379, 0.0018363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002477, upper bound: 0.0003446
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003751
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0045812, -0.0015459, -0.0022170, 0.0020770
1: -0.0047156, -0.0042235, -0.0047212, -0.0041660, -0.0003817, 0.0003559
2: 0.0088011, 0.0122697, 0.0087490, 0.0125938, -0.0027822, 0.0025949
3: 1.0084491, 1.0092607, 1.0083553, 1.0092663, -0.0006840, 0.0007339
4: -0.0036057, -0.0030725, -0.0036583, -0.0030651, -0.0003940, 0.0004243
5: 0.0004804, 0.0025772, 0.0004461, 0.0027659, -0.0016922, 0.0015841
6: -0.0025671, -0.0024493, -0.0025734, -0.0024436, -0.0001108, 0.0001145
7: -0.0107309, -0.0057858, -0.0111163, -0.0056873, -0.0039636, 0.0041278
8: -0.0065058, -0.0009483, -0.0070714, -0.0008742, -0.0040772, 0.0043982
9: -0.0036756, -0.0010505, -0.0037094, -0.0007750, -0.0020632, 0.0019126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002182, upper bound: 0.0003306
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003691
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0045812, -0.0015459, -0.0022511, 0.0020591
1: -0.0047167, -0.0042241, -0.0047212, -0.0041660, -0.0003832, 0.0003617
2: 0.0087601, 0.0122493, 0.0087490, 0.0125938, -0.0028226, 0.0025810
3: 1.0084313, 1.0092610, 1.0083553, 1.0092663, -0.0007099, 0.0007207
4: -0.0036029, -0.0030677, -0.0036583, -0.0030651, -0.0003937, 0.0004301
5: 0.0004501, 0.0025657, 0.0004461, 0.0027659, -0.0017179, 0.0015710
6: -0.0025741, -0.0024475, -0.0025734, -0.0024436, -0.0001151, 0.0001121
7: -0.0107230, -0.0056637, -0.0111163, -0.0056873, -0.0038957, 0.0042257
8: -0.0064797, -0.0009069, -0.0070714, -0.0008742, -0.0040835, 0.0044538
9: -0.0036914, -0.0010620, -0.0037094, -0.0007750, -0.0020862, 0.0019208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002182, upper bound: 0.0003314
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0044777, -0.0014627, -0.0045809, -0.0017365, -0.0020443, 0.0023961
1: -0.0047007, -0.0041561, -0.0047181, -0.0042090, -0.0003557, 0.0004169
2: 0.0088753, 0.0126989, 0.0087542, 0.0123408, -0.0025654, 0.0029992
3: 1.0083579, 1.0092447, 1.0084125, 1.0092688, -0.0007668, 0.0006847
4: -0.0036737, -0.0030842, -0.0036178, -0.0030667, -0.0004569, 0.0003916
5: 0.0005249, 0.0028296, 0.0004469, 0.0026192, -0.0015600, 0.0018280
6: -0.0025664, -0.0024423, -0.0025742, -0.0024462, -0.0001093, 0.0001191
7: -0.0112516, -0.0058893, -0.0108319, -0.0056603, -0.0044882, 0.0038398
8: -0.0072260, -0.0010757, -0.0066401, -0.0008957, -0.0047398, 0.0040605
9: -0.0036121, -0.0007054, -0.0036969, -0.0009838, -0.0019070, 0.0022293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003634, upper bound: 0.0003453
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003634, upper bound: 0.0003453
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0044777, -0.0014627, -0.0045143, -0.0014226, -0.0021073, 0.0021120
1: -0.0047007, -0.0041561, -0.0047049, -0.0041420, -0.0003706, 0.0003567
2: 0.0088753, 0.0126989, 0.0088369, 0.0127531, -0.0026380, 0.0026222
3: 1.0083579, 1.0092447, 1.0083206, 1.0092505, -0.0007192, 0.0007427
4: -0.0036737, -0.0030842, -0.0036831, -0.0030792, -0.0003961, 0.0004015
5: 0.0005249, 0.0028296, 0.0004976, 0.0028603, -0.0016077, 0.0016094
6: -0.0025664, -0.0024423, -0.0025731, -0.0024414, -0.0001167, 0.0001208
7: -0.0112516, -0.0058893, -0.0113152, -0.0057771, -0.0041093, 0.0040150
8: -0.0072260, -0.0010757, -0.0073321, -0.0010232, -0.0040906, 0.0041643
9: -0.0036121, -0.0007054, -0.0036363, -0.0006511, -0.0019603, 0.0019146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003728, upper bound: 0.0003603
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003728, upper bound: 0.0003631
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046050, -0.0017413, -0.0044706, -0.0014965, -0.0023557, 0.0019738
1: -0.0047329, -0.0042128, -0.0046993, -0.0041575, -0.0004116, 0.0003346
2: 0.0087097, 0.0123377, 0.0088845, 0.0126619, -0.0029619, 0.0024711
3: 1.0084398, 1.0092846, 1.0083615, 1.0092344, -0.0006294, 0.0007515
4: -0.0036169, -0.0030575, -0.0036691, -0.0030857, -0.0003760, 0.0004530
5: 0.0004273, 0.0026159, 0.0005303, 0.0028041, -0.0017983, 0.0015056
6: -0.0025675, -0.0024486, -0.0025660, -0.0024468, -0.0001077, 0.0001086
7: -0.0107965, -0.0056946, -0.0111499, -0.0059014, -0.0037559, 0.0043186
8: -0.0066264, -0.0007881, -0.0071835, -0.0010910, -0.0038882, 0.0047022
9: -0.0037544, -0.0009931, -0.0036047, -0.0007222, -0.0022088, 0.0018199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003615
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003615
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046468, -0.0017570, -0.0044706, -0.0014965, -0.0024046, 0.0019608
1: -0.0047341, -0.0042134, -0.0046993, -0.0041575, -0.0004124, 0.0003334
2: 0.0086660, 0.0123161, 0.0088845, 0.0126619, -0.0030159, 0.0024519
3: 1.0084223, 1.0092849, 1.0083615, 1.0092344, -0.0006555, 0.0007549
4: -0.0036139, -0.0030523, -0.0036691, -0.0030857, -0.0003726, 0.0004597
5: 0.0003962, 0.0026037, 0.0005303, 0.0028041, -0.0018350, 0.0014954
6: -0.0025745, -0.0024468, -0.0025660, -0.0024468, -0.0001140, 0.0001098
7: -0.0107971, -0.0055671, -0.0111499, -0.0059014, -0.0037644, 0.0044590
8: -0.0065975, -0.0007397, -0.0071835, -0.0010910, -0.0038513, 0.0047599
9: -0.0037735, -0.0010046, -0.0036047, -0.0007222, -0.0022307, 0.0018025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003633
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003634
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045463, -0.0013981, -0.0045370, -0.0017055, -0.0021212, 0.0023678
1: -0.0047183, -0.0041445, -0.0047159, -0.0042049, -0.0003717, 0.0004023
2: 0.0087852, 0.0127835, 0.0087998, 0.0123842, -0.0026626, 0.0029694
3: 1.0083466, 1.0092746, 1.0084280, 1.0092651, -0.0007334, 0.0006971
4: -0.0036869, -0.0030696, -0.0036243, -0.0030722, -0.0004522, 0.0004066
5: 0.0004722, 0.0028791, 0.0004795, 0.0026434, -0.0016190, 0.0018071
6: -0.0025669, -0.0024415, -0.0025671, -0.0024484, -0.0001093, 0.0001138
7: -0.0113471, -0.0057900, -0.0108506, -0.0057840, -0.0043976, 0.0039437
8: -0.0073644, -0.0009180, -0.0067061, -0.0009460, -0.0046827, 0.0042205
9: -0.0036896, -0.0006396, -0.0036767, -0.0009538, -0.0019853, 0.0021932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045463, -0.0013981, -0.0044744, -0.0013942, -0.0021571, 0.0020941
1: -0.0047183, -0.0041445, -0.0047024, -0.0041384, -0.0003853, 0.0003630
2: 0.0087852, 0.0127835, 0.0088795, 0.0127896, -0.0027084, 0.0026127
3: 1.0083466, 1.0092746, 1.0083357, 1.0092453, -0.0007178, 0.0007532
4: -0.0036869, -0.0030696, -0.0036885, -0.0030846, -0.0003972, 0.0004136
5: 0.0004722, 0.0028791, 0.0005274, 0.0028822, -0.0016464, 0.0015969
6: -0.0025669, -0.0024415, -0.0025662, -0.0024440, -0.0001145, 0.0001162
7: -0.0113471, -0.0057900, -0.0113348, -0.0058955, -0.0040141, 0.0040329
8: -0.0073644, -0.0009180, -0.0073869, -0.0010766, -0.0041177, 0.0042985
9: -0.0036896, -0.0006396, -0.0036135, -0.0006252, -0.0020270, 0.0019340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003564
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003631
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046050, -0.0017413, -0.0045092, -0.0015130, -0.0023042, 0.0020212
1: -0.0047329, -0.0042128, -0.0047017, -0.0041580, -0.0004062, 0.0003367
2: 0.0087097, 0.0123377, 0.0088437, 0.0126405, -0.0029019, 0.0025196
3: 1.0084398, 1.0092846, 1.0083432, 1.0092396, -0.0006386, 0.0007767
4: -0.0036169, -0.0030575, -0.0036660, -0.0030804, -0.0003817, 0.0004443
5: 0.0004273, 0.0026159, 0.0005016, 0.0027914, -0.0017595, 0.0015409
6: -0.0025675, -0.0024486, -0.0025729, -0.0024444, -0.0001092, 0.0001151
7: -0.0107965, -0.0056946, -0.0111450, -0.0057849, -0.0039010, 0.0042415
8: -0.0066264, -0.0007881, -0.0071540, -0.0010399, -0.0039403, 0.0046168
9: -0.0037544, -0.0009931, -0.0036264, -0.0007342, -0.0021723, 0.0018412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003602
time: 2.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003602
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046468, -0.0017570, -0.0045092, -0.0015130, -0.0023413, 0.0020015
1: -0.0047341, -0.0042134, -0.0047017, -0.0041580, -0.0004077, 0.0003424
2: 0.0086660, 0.0123161, 0.0088437, 0.0126405, -0.0029441, 0.0025049
3: 1.0084223, 1.0092849, 1.0083432, 1.0092396, -0.0006613, 0.0007628
4: -0.0036139, -0.0030523, -0.0036660, -0.0030804, -0.0003814, 0.0004499
5: 0.0003962, 0.0026037, 0.0005016, 0.0027914, -0.0017875, 0.0015266
6: -0.0025745, -0.0024468, -0.0025729, -0.0024444, -0.0001132, 0.0001127
7: -0.0107971, -0.0055671, -0.0111450, -0.0057849, -0.0038322, 0.0043402
8: -0.0065975, -0.0007397, -0.0071540, -0.0010399, -0.0039450, 0.0046690
9: -0.0037735, -0.0010046, -0.0036264, -0.0007342, -0.0021931, 0.0018483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003634
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003634
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045457, -0.0014169, -0.0045800, -0.0017219, -0.0021402, 0.0024201
1: -0.0047182, -0.0041452, -0.0047180, -0.0042055, -0.0003769, 0.0004224
2: 0.0087860, 0.0127608, 0.0087554, 0.0123613, -0.0026882, 0.0030297
3: 1.0083468, 1.0092695, 1.0084100, 1.0092689, -0.0007737, 0.0007182
4: -0.0036839, -0.0030697, -0.0036212, -0.0030669, -0.0004618, 0.0004108
5: 0.0004727, 0.0028649, 0.0004476, 0.0026306, -0.0016336, 0.0018464
6: -0.0025668, -0.0024418, -0.0025741, -0.0024461, -0.0001098, 0.0001205
7: -0.0113138, -0.0057912, -0.0108526, -0.0056620, -0.0045306, 0.0039703
8: -0.0073352, -0.0009192, -0.0066777, -0.0008975, -0.0047926, 0.0042650
9: -0.0036890, -0.0006523, -0.0036960, -0.0009652, -0.0020069, 0.0022553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003717, upper bound: 0.0003426
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003717, upper bound: 0.0003426
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045457, -0.0014169, -0.0045132, -0.0014102, -0.0021728, 0.0021391
1: -0.0047182, -0.0041452, -0.0047048, -0.0041385, -0.0003899, 0.0003627
2: 0.0087860, 0.0127608, 0.0088384, 0.0127703, -0.0027289, 0.0026573
3: 1.0083468, 1.0092695, 1.0083177, 1.0092503, -0.0007266, 0.0007748
4: -0.0036839, -0.0030697, -0.0036861, -0.0030794, -0.0004017, 0.0004170
5: 0.0004727, 0.0028649, 0.0004985, 0.0028700, -0.0016585, 0.0016302
6: -0.0025668, -0.0024418, -0.0025731, -0.0024414, -0.0001164, 0.0001225
7: -0.0113138, -0.0057912, -0.0113313, -0.0057791, -0.0041595, 0.0040707
8: -0.0073352, -0.0009192, -0.0073648, -0.0010256, -0.0041509, 0.0043361
9: -0.0036890, -0.0006523, -0.0036352, -0.0006344, -0.0020452, 0.0019435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003535, upper bound: 0.0003564
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003535, upper bound: 0.0003631
time: 1.35 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.92 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003684
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003699
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003703
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003564
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003632
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0002477, upper bound: 0.0003435
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003720
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0002477, upper bound: 0.0003446
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003525, upper bound: 0.0003751
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0002182, upper bound: 0.0003306
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003691
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0002182, upper bound: 0.0003314
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003413, upper bound: 0.0003718
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003634, upper bound: 0.0003453
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003634, upper bound: 0.0003453
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003728, upper bound: 0.0003603
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003728, upper bound: 0.0003631
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003615
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003615
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003633
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003634
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003564
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003631
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003602
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003602
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003634
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003411, upper bound: 0.0003634
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003717, upper bound: 0.0003426
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003717, upper bound: 0.0003426
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003535, upper bound: 0.0003564
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -0.0003535, upper bound: 0.0003631

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0044692, -0.0015785, -0.0022029, 0.0019467
1: -0.0047156, -0.0042235, -0.0046987, -0.0041758, -0.0003726, 0.0003295
2: 0.0088011, 0.0122697, 0.0088864, 0.0125536, -0.0027571, 0.0024367
3: 1.0084491, 1.0092607, 1.0083841, 1.0092292, -0.0006196, 0.0006929
4: -0.0036057, -0.0030725, -0.0036515, -0.0030860, -0.0003707, 0.0004191
5: 0.0004804, 0.0025772, 0.0005314, 0.0027411, -0.0016807, 0.0014849
6: -0.0025671, -0.0024493, -0.0025660, -0.0024477, -0.0001059, 0.0001076
7: -0.0107309, -0.0057858, -0.0110294, -0.0059029, -0.0037091, 0.0040967
8: -0.0065058, -0.0009483, -0.0069943, -0.0010948, -0.0038327, 0.0043338
9: -0.0036756, -0.0010505, -0.0036028, -0.0008151, -0.0020273, 0.0017940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002620, upper bound: 0.0003318
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003609
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0045370, -0.0015342, -0.0022532, 0.0020273
1: -0.0047156, -0.0042235, -0.0047163, -0.0041655, -0.0003859, 0.0003478
2: 0.0088011, 0.0122697, 0.0087972, 0.0126125, -0.0028261, 0.0025404
3: 1.0084491, 1.0092607, 1.0083737, 1.0092547, -0.0006542, 0.0007095
4: -0.0036057, -0.0030725, -0.0036612, -0.0030715, -0.0003870, 0.0004308
5: 0.0004804, 0.0025772, 0.0004794, 0.0027752, -0.0017195, 0.0015468
6: -0.0025671, -0.0024493, -0.0025664, -0.0024472, -0.0001065, 0.0001080
7: -0.0107309, -0.0057858, -0.0110922, -0.0058051, -0.0038151, 0.0041674
8: -0.0065058, -0.0009483, -0.0071007, -0.0009382, -0.0040062, 0.0044627
9: -0.0036756, -0.0010505, -0.0036798, -0.0007628, -0.0020914, 0.0018786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002620, upper bound: 0.0003318
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003625
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0044692, -0.0015785, -0.0022525, 0.0019337
1: -0.0047167, -0.0042241, -0.0046987, -0.0041758, -0.0003733, 0.0003282
2: 0.0087601, 0.0122493, 0.0088864, 0.0125536, -0.0028106, 0.0024167
3: 1.0084313, 1.0092610, 1.0083841, 1.0092292, -0.0006467, 0.0006999
4: -0.0036029, -0.0030677, -0.0036515, -0.0030860, -0.0003671, 0.0004256
5: 0.0004501, 0.0025657, 0.0005314, 0.0027411, -0.0017178, 0.0014746
6: -0.0025741, -0.0024475, -0.0025660, -0.0024477, -0.0001124, 0.0001086
7: -0.0107230, -0.0056637, -0.0110294, -0.0059029, -0.0037163, 0.0042473
8: -0.0064797, -0.0009069, -0.0069943, -0.0010948, -0.0037932, 0.0043886
9: -0.0036914, -0.0010620, -0.0036028, -0.0008151, -0.0020464, 0.0017755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002508, upper bound: 0.0003223
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003620
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0045370, -0.0015342, -0.0023028, 0.0020143
1: -0.0047167, -0.0042241, -0.0047163, -0.0041655, -0.0003865, 0.0003465
2: 0.0087601, 0.0122493, 0.0087972, 0.0126125, -0.0028796, 0.0025204
3: 1.0084313, 1.0092610, 1.0083737, 1.0092547, -0.0006813, 0.0007165
4: -0.0036029, -0.0030677, -0.0036612, -0.0030715, -0.0003834, 0.0004373
5: 0.0004501, 0.0025657, 0.0004794, 0.0027752, -0.0017566, 0.0015365
6: -0.0025741, -0.0024475, -0.0025664, -0.0024472, -0.0001129, 0.0001090
7: -0.0107230, -0.0056637, -0.0110922, -0.0058051, -0.0038224, 0.0043180
8: -0.0064797, -0.0009069, -0.0071007, -0.0009382, -0.0039667, 0.0045175
9: -0.0036914, -0.0010620, -0.0036798, -0.0007628, -0.0021104, 0.0018601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002508, upper bound: 0.0003223
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003633
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044785, -0.0014426, -0.0045359, -0.0017915, -0.0019568, 0.0023421
1: -0.0047008, -0.0041554, -0.0047156, -0.0042235, -0.0003360, 0.0003969
2: 0.0088744, 0.0127221, 0.0088011, 0.0122697, -0.0024497, 0.0029369
3: 1.0083578, 1.0092496, 1.0084491, 1.0092607, -0.0007229, 0.0006423
4: -0.0036770, -0.0030841, -0.0036057, -0.0030725, -0.0004472, 0.0003732
5: 0.0005243, 0.0028447, 0.0004804, 0.0025772, -0.0014927, 0.0017875
6: -0.0025665, -0.0024420, -0.0025671, -0.0024493, -0.0001078, 0.0001120
7: -0.0112862, -0.0058879, -0.0107309, -0.0057858, -0.0043517, 0.0037128
8: -0.0072567, -0.0010742, -0.0065058, -0.0009483, -0.0046283, 0.0038645
9: -0.0036128, -0.0006933, -0.0036756, -0.0010505, -0.0018129, 0.0021675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003440
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044785, -0.0014426, -0.0046050, -0.0017413, -0.0020061, 0.0024010
1: -0.0047008, -0.0041554, -0.0047329, -0.0042128, -0.0003465, 0.0004147
2: 0.0088744, 0.0127221, 0.0087097, 0.0123377, -0.0025136, 0.0030186
3: 1.0083578, 1.0092496, 1.0084398, 1.0092846, -0.0007558, 0.0006526
4: -0.0036770, -0.0030841, -0.0036169, -0.0030575, -0.0004612, 0.0003832
5: 0.0005243, 0.0028447, 0.0004273, 0.0026159, -0.0015306, 0.0018330
6: -0.0025665, -0.0024420, -0.0025675, -0.0024486, -0.0001085, 0.0001122
7: -0.0112862, -0.0058879, -0.0107965, -0.0056946, -0.0044312, 0.0037871
8: -0.0072567, -0.0010742, -0.0066264, -0.0007881, -0.0047826, 0.0039718
9: -0.0036128, -0.0006933, -0.0037544, -0.0009931, -0.0018652, 0.0022447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003441
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044785, -0.0014426, -0.0044807, -0.0014006, -0.0020964, 0.0020683
1: -0.0047008, -0.0041554, -0.0046809, -0.0041354, -0.0003767, 0.0003359
2: 0.0088744, 0.0127221, 0.0088888, 0.0127878, -0.0026325, 0.0025656
3: 1.0083578, 1.0092496, 1.0083178, 1.0091815, -0.0006534, 0.0007646
4: -0.0036770, -0.0030841, -0.0036893, -0.0030895, -0.0003866, 0.0004023
5: 0.0005243, 0.0028447, 0.0005241, 0.0028778, -0.0016001, 0.0015760
6: -0.0025665, -0.0024420, -0.0025703, -0.0024504, -0.0001090, 0.0001210
7: -0.0112862, -0.0058879, -0.0112752, -0.0057894, -0.0040483, 0.0039276
8: -0.0072567, -0.0010742, -0.0074003, -0.0011526, -0.0039800, 0.0041822
9: -0.0036128, -0.0006933, -0.0035623, -0.0006163, -0.0019737, 0.0018532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003546
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003564
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044785, -0.0014426, -0.0044727, -0.0014135, -0.0020871, 0.0020639
1: -0.0047008, -0.0041554, -0.0046999, -0.0041420, -0.0003655, 0.0003503
2: 0.0088744, 0.0127221, 0.0088816, 0.0127669, -0.0026136, 0.0025731
3: 1.0083578, 1.0092496, 1.0083390, 1.0092378, -0.0006873, 0.0007193
4: -0.0036770, -0.0030841, -0.0036853, -0.0030852, -0.0003902, 0.0003980
5: 0.0005243, 0.0028447, 0.0005287, 0.0028676, -0.0015924, 0.0015737
6: -0.0025665, -0.0024420, -0.0025661, -0.0024452, -0.0001117, 0.0001142
7: -0.0112862, -0.0058879, -0.0112865, -0.0058985, -0.0039575, 0.0039504
8: -0.0072567, -0.0010742, -0.0073526, -0.0010857, -0.0040360, 0.0041265
9: -0.0036128, -0.0006933, -0.0036074, -0.0006425, -0.0019417, 0.0018909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003610
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003631
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0045075, -0.0015963, -0.0021880, 0.0019939
1: -0.0047156, -0.0042235, -0.0047010, -0.0041764, -0.0003716, 0.0003315
2: 0.0088011, 0.0122697, 0.0088460, 0.0125301, -0.0027372, 0.0024847
3: 1.0084491, 1.0092607, 1.0083661, 1.0092341, -0.0006269, 0.0007206
4: -0.0036057, -0.0030725, -0.0036481, -0.0030808, -0.0003763, 0.0004159
5: 0.0004804, 0.0025772, 0.0005029, 0.0027275, -0.0016692, 0.0015200
6: -0.0025671, -0.0024493, -0.0025729, -0.0024453, -0.0001070, 0.0001141
7: -0.0107309, -0.0057858, -0.0110223, -0.0057867, -0.0038538, 0.0040833
8: -0.0065058, -0.0009483, -0.0069623, -0.0010441, -0.0038842, 0.0043007
9: -0.0036756, -0.0010505, -0.0036243, -0.0008277, -0.0020112, 0.0018150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002665, upper bound: 0.0002569
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002665, upper bound: 0.0003720
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0045075, -0.0015963, -0.0022268, 0.0019747
1: -0.0047167, -0.0042241, -0.0047010, -0.0041764, -0.0003741, 0.0003373
2: 0.0087601, 0.0122493, 0.0088460, 0.0125301, -0.0027821, 0.0024716
3: 1.0084313, 1.0092610, 1.0083661, 1.0092341, -0.0006499, 0.0007062
4: -0.0036029, -0.0030677, -0.0036481, -0.0030808, -0.0003764, 0.0004221
5: 0.0004501, 0.0025657, 0.0005029, 0.0027275, -0.0016984, 0.0015062
6: -0.0025741, -0.0024475, -0.0025729, -0.0024453, -0.0001110, 0.0001117
7: -0.0107230, -0.0056637, -0.0110223, -0.0057867, -0.0037797, 0.0041790
8: -0.0064797, -0.0009069, -0.0069623, -0.0010441, -0.0038938, 0.0043596
9: -0.0036914, -0.0010620, -0.0036243, -0.0008277, -0.0020372, 0.0018244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002989, upper bound: 0.0003466
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003439, upper bound: 0.0003663
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0045359, -0.0017915, -0.0045783, -0.0015519, -0.0022104, 0.0020739
1: -0.0047156, -0.0042235, -0.0047187, -0.0041661, -0.0003816, 0.0003502
2: 0.0088011, 0.0122697, 0.0087527, 0.0125900, -0.0027746, 0.0025909
3: 1.0084491, 1.0092607, 1.0083555, 1.0092587, -0.0006606, 0.0007336
4: -0.0036057, -0.0030725, -0.0036581, -0.0030659, -0.0003932, 0.0004233
5: 0.0004804, 0.0025772, 0.0004484, 0.0027616, -0.0016871, 0.0015817
6: -0.0025671, -0.0024493, -0.0025733, -0.0024448, -0.0001083, 0.0001144
7: -0.0107309, -0.0057858, -0.0110869, -0.0056923, -0.0039583, 0.0040996
8: -0.0065058, -0.0009483, -0.0070696, -0.0008857, -0.0040624, 0.0043900
9: -0.0036756, -0.0010505, -0.0037022, -0.0007757, -0.0020605, 0.0019012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002534, upper bound: 0.0003222
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003346, upper bound: 0.0003608
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045767, -0.0018063, -0.0045783, -0.0015519, -0.0022441, 0.0020558
1: -0.0047167, -0.0042241, -0.0047187, -0.0041661, -0.0003832, 0.0003554
2: 0.0087601, 0.0122493, 0.0087527, 0.0125900, -0.0028149, 0.0025767
3: 1.0084313, 1.0092610, 1.0083555, 1.0092587, -0.0006842, 0.0007205
4: -0.0036029, -0.0030677, -0.0036581, -0.0030659, -0.0003928, 0.0004292
5: 0.0004501, 0.0025657, 0.0004484, 0.0027616, -0.0017126, 0.0015685
6: -0.0025741, -0.0024475, -0.0025733, -0.0024448, -0.0001125, 0.0001120
7: -0.0107230, -0.0056637, -0.0110869, -0.0056923, -0.0038901, 0.0041994
8: -0.0064797, -0.0009069, -0.0070696, -0.0008857, -0.0040682, 0.0044457
9: -0.0036914, -0.0010620, -0.0037022, -0.0007757, -0.0020834, 0.0019090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002498, upper bound: 0.0003209
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003633
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044777, -0.0014627, -0.0045782, -0.0018063, -0.0019755, 0.0023932
1: -0.0047007, -0.0041561, -0.0047173, -0.0042241, -0.0003416, 0.0004161
2: 0.0088753, 0.0126989, 0.0087578, 0.0122493, -0.0024755, 0.0029953
3: 1.0083579, 1.0092447, 1.0084313, 1.0092630, -0.0007622, 0.0006666
4: -0.0036737, -0.0030842, -0.0036029, -0.0030673, -0.0004562, 0.0003775
5: 0.0005249, 0.0028296, 0.0004489, 0.0025657, -0.0015071, 0.0018258
6: -0.0025664, -0.0024423, -0.0025741, -0.0024472, -0.0001078, 0.0001187
7: -0.0112516, -0.0058893, -0.0107248, -0.0056637, -0.0044845, 0.0037363
8: -0.0072260, -0.0010757, -0.0064797, -0.0009020, -0.0047330, 0.0039105
9: -0.0036121, -0.0007054, -0.0036939, -0.0010620, -0.0018355, 0.0022259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003434
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044777, -0.0014627, -0.0046484, -0.0017570, -0.0020254, 0.0024545
1: -0.0047007, -0.0041561, -0.0047348, -0.0042134, -0.0003517, 0.0004326
2: 0.0088753, 0.0126989, 0.0086634, 0.0123161, -0.0025394, 0.0030801
3: 1.0083579, 1.0092447, 1.0084223, 1.0092871, -0.0007910, 0.0006752
4: -0.0036737, -0.0030842, -0.0036139, -0.0030518, -0.0004701, 0.0003874
5: 0.0005249, 0.0028296, 0.0003949, 0.0026037, -0.0015454, 0.0018734
6: -0.0025664, -0.0024423, -0.0025745, -0.0024464, -0.0001086, 0.0001188
7: -0.0112516, -0.0058893, -0.0107993, -0.0055671, -0.0045413, 0.0038123
8: -0.0072260, -0.0010757, -0.0065975, -0.0007342, -0.0048822, 0.0040170
9: -0.0036121, -0.0007054, -0.0037763, -0.0010046, -0.0018867, 0.0022991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003434
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0044777, -0.0014627, -0.0045116, -0.0014930, -0.0020453, 0.0021092
1: -0.0047007, -0.0041561, -0.0047041, -0.0041571, -0.0003563, 0.0003558
2: 0.0088753, 0.0126989, 0.0088406, 0.0126605, -0.0025541, 0.0026184
3: 1.0083579, 1.0092447, 1.0083405, 1.0092448, -0.0007144, 0.0007237
4: -0.0036737, -0.0030842, -0.0036682, -0.0030797, -0.0003955, 0.0003880
5: 0.0005249, 0.0028296, 0.0004997, 0.0028063, -0.0015598, 0.0016072
6: -0.0025664, -0.0024423, -0.0025731, -0.0024424, -0.0001153, 0.0001203
7: -0.0112516, -0.0058893, -0.0112104, -0.0057808, -0.0041057, 0.0039250
8: -0.0072260, -0.0010757, -0.0071732, -0.0010296, -0.0040839, 0.0040199
9: -0.0036121, -0.0007054, -0.0036332, -0.0007276, -0.0018890, 0.0019113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003577
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003603
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0044777, -0.0014627, -0.0045822, -0.0014485, -0.0020788, 0.0021622
1: -0.0047007, -0.0041561, -0.0047217, -0.0041462, -0.0003661, 0.0003727
2: 0.0088753, 0.0126989, 0.0087475, 0.0127197, -0.0026027, 0.0026968
3: 1.0083579, 1.0092447, 1.0083300, 1.0092691, -0.0007440, 0.0007341
4: -0.0036737, -0.0030842, -0.0036781, -0.0030648, -0.0004091, 0.0003963
5: 0.0005249, 0.0028296, 0.0004454, 0.0028405, -0.0015860, 0.0016487
6: -0.0025664, -0.0024423, -0.0025734, -0.0024419, -0.0001158, 0.0001202
7: -0.0112516, -0.0058893, -0.0112729, -0.0056866, -0.0041478, 0.0039449
8: -0.0072260, -0.0010757, -0.0072811, -0.0008712, -0.0042327, 0.0041121
9: -0.0036121, -0.0007054, -0.0037109, -0.0006745, -0.0019367, 0.0019855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003606
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003632
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046050, -0.0017413, -0.0044692, -0.0015785, -0.0022728, 0.0019960
1: -0.0047329, -0.0042128, -0.0046987, -0.0041758, -0.0003917, 0.0003400
2: 0.0087097, 0.0123377, 0.0088864, 0.0125536, -0.0028506, 0.0025006
3: 1.0084398, 1.0092846, 1.0083841, 1.0092292, -0.0006298, 0.0007258
4: -0.0036169, -0.0030575, -0.0036515, -0.0030860, -0.0003807, 0.0004346
5: 0.0004273, 0.0026159, 0.0005314, 0.0027411, -0.0017344, 0.0015228
6: -0.0025675, -0.0024486, -0.0025660, -0.0024477, -0.0001063, 0.0001082
7: -0.0107965, -0.0056946, -0.0110294, -0.0059029, -0.0037834, 0.0041929
8: -0.0066264, -0.0007881, -0.0069943, -0.0010948, -0.0039400, 0.0045028
9: -0.0037544, -0.0009931, -0.0036028, -0.0008151, -0.0021106, 0.0018463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003073, upper bound: 0.0003190
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003586
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046050, -0.0017413, -0.0045370, -0.0015342, -0.0022274, 0.0019761
1: -0.0047329, -0.0042128, -0.0047163, -0.0041655, -0.0003786, 0.0003369
2: 0.0087097, 0.0123377, 0.0087972, 0.0126125, -0.0027882, 0.0024755
3: 1.0084398, 1.0092846, 1.0083737, 1.0092547, -0.0006375, 0.0007069
4: -0.0036169, -0.0030575, -0.0036612, -0.0030715, -0.0003769, 0.0004239
5: 0.0004273, 0.0026159, 0.0004794, 0.0027752, -0.0016994, 0.0015075
6: -0.0025675, -0.0024486, -0.0025664, -0.0024472, -0.0001070, 0.0001086
7: -0.0107965, -0.0056946, -0.0110922, -0.0058051, -0.0037539, 0.0041329
8: -0.0066264, -0.0007881, -0.0071007, -0.0009382, -0.0038992, 0.0043850
9: -0.0037544, -0.0009931, -0.0036798, -0.0007628, -0.0020524, 0.0018257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003073, upper bound: 0.0003190
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003586
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046468, -0.0017570, -0.0044692, -0.0015785, -0.0023217, 0.0019839
1: -0.0047341, -0.0042134, -0.0046987, -0.0041758, -0.0003926, 0.0003391
2: 0.0086660, 0.0123161, 0.0088864, 0.0125536, -0.0029045, 0.0024834
3: 1.0084223, 1.0092849, 1.0083841, 1.0092292, -0.0006551, 0.0007292
4: -0.0036139, -0.0030523, -0.0036515, -0.0030860, -0.0003778, 0.0004413
5: 0.0003962, 0.0026037, 0.0005314, 0.0027411, -0.0017711, 0.0015134
6: -0.0025745, -0.0024468, -0.0025660, -0.0024477, -0.0001126, 0.0001093
7: -0.0107971, -0.0055671, -0.0110294, -0.0059029, -0.0037780, 0.0043333
8: -0.0065975, -0.0007397, -0.0069943, -0.0010948, -0.0039093, 0.0045604
9: -0.0037735, -0.0010046, -0.0036028, -0.0008151, -0.0021325, 0.0018323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003049
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046468, -0.0017570, -0.0045370, -0.0015342, -0.0022763, 0.0019631
1: -0.0047341, -0.0042134, -0.0047163, -0.0041655, -0.0003793, 0.0003356
2: 0.0086660, 0.0123161, 0.0087972, 0.0126125, -0.0028417, 0.0024563
3: 1.0084223, 1.0092849, 1.0083737, 1.0092547, -0.0006637, 0.0007133
4: -0.0036139, -0.0030523, -0.0036612, -0.0030715, -0.0003735, 0.0004305
5: 0.0003962, 0.0026037, 0.0004794, 0.0027752, -0.0017361, 0.0014973
6: -0.0025745, -0.0024468, -0.0025664, -0.0024472, -0.0001136, 0.0001098
7: -0.0107971, -0.0055671, -0.0110922, -0.0058051, -0.0037624, 0.0042830
8: -0.0065975, -0.0007397, -0.0071007, -0.0009382, -0.0038622, 0.0044409
9: -0.0037735, -0.0010046, -0.0036798, -0.0007628, -0.0020719, 0.0018083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003049
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045463, -0.0013981, -0.0045359, -0.0017915, -0.0020371, 0.0023909
1: -0.0047183, -0.0041445, -0.0047156, -0.0042235, -0.0003539, 0.0004094
2: 0.0087852, 0.0127835, 0.0088011, 0.0122697, -0.0025534, 0.0030036
3: 1.0083466, 1.0092746, 1.0084491, 1.0092607, -0.0007406, 0.0006754
4: -0.0036869, -0.0030696, -0.0036057, -0.0030725, -0.0004582, 0.0003894
5: 0.0004722, 0.0028791, 0.0004804, 0.0025772, -0.0015544, 0.0018252
6: -0.0025669, -0.0024415, -0.0025671, -0.0024493, -0.0001082, 0.0001125
7: -0.0113471, -0.0057900, -0.0107309, -0.0057858, -0.0044145, 0.0038189
8: -0.0073644, -0.0009180, -0.0065058, -0.0009483, -0.0047487, 0.0040377
9: -0.0036896, -0.0006396, -0.0036756, -0.0010505, -0.0018972, 0.0022271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003416
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045463, -0.0013981, -0.0046050, -0.0017413, -0.0019860, 0.0023659
1: -0.0047183, -0.0041445, -0.0047329, -0.0042128, -0.0003432, 0.0004027
2: 0.0087852, 0.0127835, 0.0087097, 0.0123377, -0.0024884, 0.0029673
3: 1.0083466, 1.0092746, 1.0084398, 1.0092846, -0.0007371, 0.0006592
4: -0.0036869, -0.0030696, -0.0036169, -0.0030575, -0.0004520, 0.0003793
5: 0.0004722, 0.0028791, 0.0004273, 0.0026159, -0.0015152, 0.0018057
6: -0.0025669, -0.0024415, -0.0025675, -0.0024486, -0.0001088, 0.0001134
7: -0.0113471, -0.0057900, -0.0107965, -0.0056946, -0.0043916, 0.0037575
8: -0.0073644, -0.0009180, -0.0066264, -0.0007881, -0.0046803, 0.0039305
9: -0.0036896, -0.0006396, -0.0037544, -0.0009931, -0.0018443, 0.0021924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003416
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045463, -0.0013981, -0.0044795, -0.0013873, -0.0021608, 0.0020957
1: -0.0047183, -0.0041445, -0.0046807, -0.0041319, -0.0003970, 0.0003419
2: 0.0087852, 0.0127835, 0.0088903, 0.0128067, -0.0027225, 0.0026007
3: 1.0083466, 1.0092746, 1.0083150, 1.0091815, -0.0006608, 0.0007977
4: -0.0036869, -0.0030696, -0.0036924, -0.0030897, -0.0003923, 0.0004178
5: 0.0004722, 0.0028791, 0.0005250, 0.0028882, -0.0016501, 0.0015970
6: -0.0025669, -0.0024415, -0.0025702, -0.0024504, -0.0001086, 0.0001227
7: -0.0113471, -0.0057900, -0.0112935, -0.0057916, -0.0041015, 0.0039822
8: -0.0073644, -0.0009180, -0.0074354, -0.0011550, -0.0040415, 0.0043537
9: -0.0036896, -0.0006396, -0.0035612, -0.0005982, -0.0020602, 0.0018832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003546
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003564
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045463, -0.0013981, -0.0044715, -0.0014002, -0.0021523, 0.0020912
1: -0.0047183, -0.0041445, -0.0046998, -0.0041385, -0.0003852, 0.0003562
2: 0.0087852, 0.0127835, 0.0088832, 0.0127858, -0.0027043, 0.0026082
3: 1.0083466, 1.0092746, 1.0083359, 1.0092376, -0.0006945, 0.0007530
4: -0.0036869, -0.0030696, -0.0036883, -0.0030854, -0.0003959, 0.0004134
5: 0.0004722, 0.0028791, 0.0005296, 0.0028779, -0.0016429, 0.0015947
6: -0.0025669, -0.0024415, -0.0025660, -0.0024452, -0.0001113, 0.0001161
7: -0.0113471, -0.0057900, -0.0113032, -0.0059008, -0.0040090, 0.0040083
8: -0.0073644, -0.0009180, -0.0073851, -0.0010882, -0.0040980, 0.0042967
9: -0.0036896, -0.0006396, -0.0036063, -0.0006259, -0.0020263, 0.0019207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003610
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003632
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046050, -0.0017413, -0.0045075, -0.0015963, -0.0022578, 0.0020433
1: -0.0047329, -0.0042128, -0.0047010, -0.0041764, -0.0003907, 0.0003420
2: 0.0087097, 0.0123377, 0.0088460, 0.0125301, -0.0028307, 0.0025485
3: 1.0084398, 1.0092846, 1.0083661, 1.0092341, -0.0006371, 0.0007535
4: -0.0036169, -0.0030575, -0.0036481, -0.0030808, -0.0003863, 0.0004314
5: 0.0004273, 0.0026159, 0.0005029, 0.0027275, -0.0017229, 0.0015579
6: -0.0025675, -0.0024486, -0.0025729, -0.0024453, -0.0001074, 0.0001147
7: -0.0107965, -0.0056946, -0.0110223, -0.0057867, -0.0039281, 0.0041794
8: -0.0066264, -0.0007881, -0.0069623, -0.0010441, -0.0039914, 0.0044697
9: -0.0037544, -0.0009931, -0.0036243, -0.0008277, -0.0020945, 0.0018673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003057
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003564
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046050, -0.0017413, -0.0045783, -0.0015519, -0.0021854, 0.0020237
1: -0.0047329, -0.0042128, -0.0047187, -0.0041661, -0.0003743, 0.0003389
2: 0.0087097, 0.0123377, 0.0087527, 0.0125900, -0.0027388, 0.0025239
3: 1.0084398, 1.0092846, 1.0083555, 1.0092587, -0.0006454, 0.0007342
4: -0.0036169, -0.0030575, -0.0036581, -0.0030659, -0.0003827, 0.0004171
5: 0.0004273, 0.0026159, 0.0004484, 0.0027616, -0.0016677, 0.0015429
6: -0.0025675, -0.0024486, -0.0025733, -0.0024448, -0.0001094, 0.0001151
7: -0.0107965, -0.0056946, -0.0110869, -0.0056923, -0.0039002, 0.0040815
8: -0.0066264, -0.0007881, -0.0070696, -0.0008857, -0.0039514, 0.0043206
9: -0.0037544, -0.0009931, -0.0037022, -0.0007757, -0.0020243, 0.0018474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003057
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003564
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046468, -0.0017570, -0.0045075, -0.0015963, -0.0022963, 0.0020246
1: -0.0047341, -0.0042134, -0.0047010, -0.0041764, -0.0003931, 0.0003474
2: 0.0086660, 0.0123161, 0.0088460, 0.0125301, -0.0028757, 0.0025355
3: 1.0084223, 1.0092849, 1.0083661, 1.0092341, -0.0006587, 0.0007387
4: -0.0036139, -0.0030523, -0.0036481, -0.0030808, -0.0003864, 0.0004377
5: 0.0003962, 0.0026037, 0.0005029, 0.0027275, -0.0017520, 0.0015445
6: -0.0025745, -0.0024468, -0.0025729, -0.0024453, -0.0001112, 0.0001123
7: -0.0107971, -0.0055671, -0.0110223, -0.0057867, -0.0038547, 0.0042725
8: -0.0065975, -0.0007397, -0.0069623, -0.0010441, -0.0040003, 0.0045302
9: -0.0037735, -0.0010046, -0.0036243, -0.0008277, -0.0021210, 0.0018756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003037
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046468, -0.0017570, -0.0045783, -0.0015519, -0.0022205, 0.0020037
1: -0.0047341, -0.0042134, -0.0047187, -0.0041661, -0.0003759, 0.0003445
2: 0.0086660, 0.0123161, 0.0087527, 0.0125900, -0.0027792, 0.0025093
3: 1.0084223, 1.0092849, 1.0083555, 1.0092587, -0.0006683, 0.0007203
4: -0.0036139, -0.0030523, -0.0036581, -0.0030659, -0.0003822, 0.0004227
5: 0.0003962, 0.0026037, 0.0004484, 0.0027616, -0.0016942, 0.0015283
6: -0.0025745, -0.0024468, -0.0025733, -0.0024448, -0.0001136, 0.0001127
7: -0.0107971, -0.0055671, -0.0110869, -0.0056923, -0.0038306, 0.0041795
8: -0.0065975, -0.0007397, -0.0070696, -0.0008857, -0.0039557, 0.0043744
9: -0.0037735, -0.0010046, -0.0037022, -0.0007757, -0.0020470, 0.0018541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003037
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045457, -0.0014169, -0.0045782, -0.0018063, -0.0020560, 0.0024382
1: -0.0047182, -0.0041452, -0.0047173, -0.0042241, -0.0003595, 0.0004273
2: 0.0087860, 0.0127608, 0.0087578, 0.0122493, -0.0025793, 0.0030571
3: 1.0083468, 1.0092695, 1.0084313, 1.0092630, -0.0007771, 0.0006982
4: -0.0036839, -0.0030697, -0.0036029, -0.0030673, -0.0004666, 0.0003938
5: 0.0004727, 0.0028649, 0.0004489, 0.0025657, -0.0015690, 0.0018605
6: -0.0025668, -0.0024418, -0.0025741, -0.0024472, -0.0001083, 0.0001192
7: -0.0113138, -0.0057912, -0.0107248, -0.0056637, -0.0045491, 0.0038438
8: -0.0073352, -0.0009192, -0.0064797, -0.0009020, -0.0048473, 0.0040840
9: -0.0036890, -0.0006523, -0.0036939, -0.0010620, -0.0019198, 0.0022813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003408
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045457, -0.0014169, -0.0046484, -0.0017570, -0.0020043, 0.0024191
1: -0.0047182, -0.0041452, -0.0047348, -0.0042134, -0.0003487, 0.0004239
2: 0.0087860, 0.0127608, 0.0086634, 0.0123161, -0.0025130, 0.0030296
3: 1.0083468, 1.0092695, 1.0084223, 1.0092871, -0.0007794, 0.0006827
4: -0.0036839, -0.0030697, -0.0036139, -0.0030518, -0.0004619, 0.0003833
5: 0.0004727, 0.0028649, 0.0003949, 0.0026037, -0.0015293, 0.0018458
6: -0.0025668, -0.0024418, -0.0025745, -0.0024464, -0.0001091, 0.0001201
7: -0.0113138, -0.0057912, -0.0107993, -0.0055671, -0.0045247, 0.0037858
8: -0.0073352, -0.0009192, -0.0065975, -0.0007342, -0.0047936, 0.0039719
9: -0.0036890, -0.0006523, -0.0037763, -0.0010046, -0.0018649, 0.0022567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003408
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045457, -0.0014169, -0.0045177, -0.0014028, -0.0021755, 0.0021401
1: -0.0047182, -0.0041452, -0.0046833, -0.0041320, -0.0004012, 0.0003419
2: 0.0087860, 0.0127608, 0.0088493, 0.0127875, -0.0027421, 0.0026450
3: 1.0083468, 1.0092695, 1.0082984, 1.0091864, -0.0006679, 0.0008182
4: -0.0036839, -0.0030697, -0.0036900, -0.0030844, -0.0003969, 0.0004210
5: 0.0004727, 0.0028649, 0.0004964, 0.0028763, -0.0016615, 0.0016299
6: -0.0025668, -0.0024418, -0.0025769, -0.0024484, -0.0001103, 0.0001290
7: -0.0113138, -0.0057912, -0.0112878, -0.0056849, -0.0042414, 0.0040168
8: -0.0073352, -0.0009192, -0.0074114, -0.0011034, -0.0040775, 0.0043899
9: -0.0036890, -0.0006523, -0.0035837, -0.0006085, -0.0020779, 0.0018943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003540
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003564
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045457, -0.0014169, -0.0045102, -0.0014160, -0.0021681, 0.0021362
1: -0.0047182, -0.0041452, -0.0047021, -0.0041386, -0.0003899, 0.0003562
2: 0.0087860, 0.0127608, 0.0088422, 0.0127666, -0.0027249, 0.0026529
3: 1.0083468, 1.0092695, 1.0083181, 1.0092427, -0.0007017, 0.0007745
4: -0.0036839, -0.0030697, -0.0036859, -0.0030802, -0.0004006, 0.0004168
5: 0.0004727, 0.0028649, 0.0005008, 0.0028658, -0.0016550, 0.0016279
6: -0.0025668, -0.0024418, -0.0025730, -0.0024426, -0.0001132, 0.0001224
7: -0.0113138, -0.0057912, -0.0113016, -0.0057842, -0.0041546, 0.0040462
8: -0.0073352, -0.0009192, -0.0073629, -0.0010370, -0.0041326, 0.0043342
9: -0.0036890, -0.0006523, -0.0036280, -0.0006352, -0.0020444, 0.0019309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003606
time: 1.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003631
time: 1.07 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.28 seconds
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002620, upper bound: 0.0003318
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003609
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002620, upper bound: 0.0003318
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003625
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002508, upper bound: 0.0003223
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003620
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002508, upper bound: 0.0003223
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003633
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003440
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003441
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003546
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003564
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003610
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003631
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002665, upper bound: 0.0002569
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002665, upper bound: 0.0003720
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002989, upper bound: 0.0003466
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003439, upper bound: 0.0003663
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002534, upper bound: 0.0003222
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003346, upper bound: 0.0003608
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002498, upper bound: 0.0003209
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003633
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003434
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003434
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003453
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003577
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003603
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003606
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003694, upper bound: 0.0003632
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003073, upper bound: 0.0003190
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003586
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003073, upper bound: 0.0003190
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003586
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003049
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003049
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003416
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003416
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003546
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003564
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003610
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003632
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003057
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003564
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003057
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003564
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003037
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0002976, upper bound: 0.0003037
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003408
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003408
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003691, upper bound: 0.0003426
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003540
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003564
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003606
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 3, lower bound: -0.0003523, upper bound: 0.0003631

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045357, -0.0018062, -0.0044692, -0.0015785, -0.0022027, 0.0018923
1: -0.0047155, -0.0042267, -0.0046987, -0.0041758, -0.0003725, 0.0003130
2: 0.0088014, 0.0122506, 0.0088864, 0.0125536, -0.0027568, 0.0023596
3: 1.0084549, 1.0092586, 1.0083841, 1.0092292, -0.0005960, 0.0006909
4: -0.0036026, -0.0030725, -0.0036515, -0.0030860, -0.0003575, 0.0004190
5: 0.0004805, 0.0025659, 0.0005314, 0.0027411, -0.0016805, 0.0014428
6: -0.0025668, -0.0024497, -0.0025660, -0.0024477, -0.0001056, 0.0001071
7: -0.0107067, -0.0057861, -0.0110294, -0.0059029, -0.0036514, 0.0040964
8: -0.0064734, -0.0009488, -0.0069943, -0.0010948, -0.0036871, 0.0043332
9: -0.0036753, -0.0010664, -0.0036028, -0.0008151, -0.0020270, 0.0017190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003457, upper bound: 0.0003476
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003457, upper bound: 0.0003653
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045357, -0.0018062, -0.0045370, -0.0015342, -0.0022530, 0.0019722
1: -0.0047155, -0.0042267, -0.0047163, -0.0041655, -0.0003857, 0.0003322
2: 0.0088014, 0.0122506, 0.0087972, 0.0126125, -0.0028258, 0.0024669
3: 1.0084549, 1.0092586, 1.0083737, 1.0092547, -0.0006292, 0.0007075
4: -0.0036026, -0.0030725, -0.0036612, -0.0030715, -0.0003748, 0.0004308
5: 0.0004805, 0.0025659, 0.0004794, 0.0027752, -0.0017193, 0.0015044
6: -0.0025668, -0.0024497, -0.0025664, -0.0024472, -0.0001061, 0.0001075
7: -0.0107067, -0.0057861, -0.0110922, -0.0058051, -0.0037564, 0.0041671
8: -0.0064734, -0.0009488, -0.0071007, -0.0009382, -0.0038714, 0.0044621
9: -0.0036753, -0.0010664, -0.0036798, -0.0007628, -0.0020911, 0.0018087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003373
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003348, upper bound: 0.0003625
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045765, -0.0018216, -0.0044692, -0.0015785, -0.0022523, 0.0018768
1: -0.0047165, -0.0042273, -0.0046987, -0.0041758, -0.0003732, 0.0003118
2: 0.0087604, 0.0122297, 0.0088864, 0.0125536, -0.0028103, 0.0023375
3: 1.0084374, 1.0092591, 1.0083841, 1.0092292, -0.0006224, 0.0006979
4: -0.0035998, -0.0030678, -0.0036515, -0.0030860, -0.0003540, 0.0004255
5: 0.0004503, 0.0025540, 0.0005314, 0.0027411, -0.0017176, 0.0014307
6: -0.0025738, -0.0024479, -0.0025660, -0.0024477, -0.0001121, 0.0001082
7: -0.0106987, -0.0056640, -0.0110294, -0.0059029, -0.0036538, 0.0042470
8: -0.0064466, -0.0009075, -0.0069943, -0.0010948, -0.0036501, 0.0043880
9: -0.0036911, -0.0010778, -0.0036028, -0.0008151, -0.0020460, 0.0017030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003439, upper bound: 0.0003477
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003439, upper bound: 0.0003663
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045765, -0.0018216, -0.0045370, -0.0015342, -0.0023026, 0.0019566
1: -0.0047165, -0.0042273, -0.0047163, -0.0041655, -0.0003864, 0.0003310
2: 0.0087604, 0.0122297, 0.0087972, 0.0126125, -0.0028793, 0.0024448
3: 1.0084374, 1.0092591, 1.0083737, 1.0092547, -0.0006556, 0.0007145
4: -0.0035998, -0.0030678, -0.0036612, -0.0030715, -0.0003713, 0.0004373
5: 0.0004503, 0.0025540, 0.0004794, 0.0027752, -0.0017565, 0.0014923
6: -0.0025738, -0.0024479, -0.0025664, -0.0024472, -0.0001127, 0.0001086
7: -0.0106987, -0.0056640, -0.0110922, -0.0058051, -0.0037588, 0.0043177
8: -0.0064466, -0.0009075, -0.0071007, -0.0009382, -0.0038343, 0.0045169
9: -0.0036911, -0.0010778, -0.0036798, -0.0007628, -0.0021101, 0.0017926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003368
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003633
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0045359, -0.0017915, -0.0019475, 0.0023024
1: -0.0046992, -0.0041572, -0.0047156, -0.0042235, -0.0003299, 0.0003956
2: 0.0088851, 0.0126744, 0.0088011, 0.0122697, -0.0024380, 0.0028910
3: 1.0083590, 1.0092323, 1.0084491, 1.0092607, -0.0007216, 0.0006240
4: -0.0036705, -0.0030858, -0.0036057, -0.0030725, -0.0004409, 0.0003709
5: 0.0005307, 0.0028134, 0.0004804, 0.0025772, -0.0014856, 0.0017575
6: -0.0025660, -0.0024461, -0.0025671, -0.0024493, -0.0001074, 0.0001082
7: -0.0111823, -0.0059023, -0.0107309, -0.0057858, -0.0042511, 0.0036980
8: -0.0071951, -0.0010918, -0.0065058, -0.0009483, -0.0045690, 0.0038355
9: -0.0036044, -0.0007190, -0.0036756, -0.0010505, -0.0017955, 0.0021429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002569, upper bound: 0.0002664
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002569, upper bound: 0.0003533
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0045359, -0.0017915, -0.0019947, 0.0022898
1: -0.0047015, -0.0041572, -0.0047156, -0.0042235, -0.0003319, 0.0003951
2: 0.0088445, 0.0126567, 0.0088011, 0.0122697, -0.0024861, 0.0028739
3: 1.0083407, 1.0092371, 1.0084491, 1.0092607, -0.0007472, 0.0006312
4: -0.0036681, -0.0030806, -0.0036057, -0.0030725, -0.0004383, 0.0003766
5: 0.0005021, 0.0028020, 0.0004804, 0.0025772, -0.0015208, 0.0017478
6: -0.0025729, -0.0024436, -0.0025671, -0.0024493, -0.0001141, 0.0001096
7: -0.0111804, -0.0057861, -0.0107309, -0.0057858, -0.0042427, 0.0038432
8: -0.0071714, -0.0010412, -0.0065058, -0.0009483, -0.0045426, 0.0038871
9: -0.0036259, -0.0007283, -0.0036756, -0.0010505, -0.0018166, 0.0021314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002569, upper bound: 0.0002664
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002569, upper bound: 0.0002665
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0046050, -0.0017413, -0.0019969, 0.0023613
1: -0.0046992, -0.0041572, -0.0047329, -0.0042128, -0.0003405, 0.0004133
2: 0.0088851, 0.0126744, 0.0087097, 0.0123377, -0.0025018, 0.0029726
3: 1.0083590, 1.0092323, 1.0084398, 1.0092846, -0.0007545, 0.0006343
4: -0.0036705, -0.0030858, -0.0036169, -0.0030575, -0.0004549, 0.0003809
5: 0.0005307, 0.0028134, 0.0004273, 0.0026159, -0.0015235, 0.0018030
6: -0.0025660, -0.0024461, -0.0025675, -0.0024486, -0.0001080, 0.0001084
7: -0.0111823, -0.0059023, -0.0107965, -0.0056946, -0.0043305, 0.0037723
8: -0.0071951, -0.0010918, -0.0066264, -0.0007881, -0.0047233, 0.0039427
9: -0.0036044, -0.0007190, -0.0037544, -0.0009931, -0.0018478, 0.0022201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003057, upper bound: 0.0003056
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003564, upper bound: 0.0003403
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0046050, -0.0017413, -0.0020441, 0.0023487
1: -0.0047015, -0.0041572, -0.0047329, -0.0042128, -0.0003425, 0.0004129
2: 0.0088445, 0.0126567, 0.0087097, 0.0123377, -0.0025500, 0.0029556
3: 1.0083407, 1.0092371, 1.0084398, 1.0092846, -0.0007801, 0.0006415
4: -0.0036681, -0.0030806, -0.0036169, -0.0030575, -0.0004523, 0.0003866
5: 0.0005021, 0.0028020, 0.0004273, 0.0026159, -0.0015587, 0.0017934
6: -0.0025729, -0.0024436, -0.0025675, -0.0024486, -0.0001148, 0.0001098
7: -0.0111804, -0.0057861, -0.0107965, -0.0056946, -0.0043221, 0.0039175
8: -0.0071714, -0.0010412, -0.0066264, -0.0007881, -0.0046969, 0.0039944
9: -0.0036259, -0.0007283, -0.0037544, -0.0009931, -0.0018689, 0.0022087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003057, upper bound: 0.0003056
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003564, upper bound: 0.0003416
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0044807, -0.0014006, -0.0021481, 0.0020214
1: -0.0047015, -0.0041572, -0.0046809, -0.0041354, -0.0003733, 0.0003314
2: 0.0088445, 0.0126567, 0.0088888, 0.0127878, -0.0026840, 0.0025083
3: 1.0083407, 1.0092371, 1.0083178, 1.0091815, -0.0006818, 0.0007558
4: -0.0036681, -0.0030806, -0.0036893, -0.0030895, -0.0003781, 0.0004079
5: 0.0005021, 0.0028020, 0.0005241, 0.0028778, -0.0016384, 0.0015403
6: -0.0025729, -0.0024436, -0.0025703, -0.0024504, -0.0001157, 0.0001185
7: -0.0111804, -0.0057861, -0.0112752, -0.0057894, -0.0039616, 0.0040920
8: -0.0071714, -0.0010412, -0.0074003, -0.0011526, -0.0038902, 0.0042267
9: -0.0036259, -0.0007283, -0.0035623, -0.0006163, -0.0019845, 0.0018130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003534
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003564
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0044727, -0.0014135, -0.0020787, 0.0020196
1: -0.0046992, -0.0041572, -0.0046999, -0.0041420, -0.0003601, 0.0003464
2: 0.0088851, 0.0126744, 0.0088816, 0.0127669, -0.0026029, 0.0025228
3: 1.0083590, 1.0092323, 1.0083390, 1.0092378, -0.0006860, 0.0007011
4: -0.0036705, -0.0030858, -0.0036853, -0.0030852, -0.0003833, 0.0003963
5: 0.0005307, 0.0028134, 0.0005287, 0.0028676, -0.0015860, 0.0015404
6: -0.0025660, -0.0024461, -0.0025661, -0.0024452, -0.0001113, 0.0001104
7: -0.0111823, -0.0059023, -0.0112865, -0.0058985, -0.0038491, 0.0039363
8: -0.0071951, -0.0010918, -0.0073526, -0.0010857, -0.0039686, 0.0041063
9: -0.0036044, -0.0007190, -0.0036074, -0.0006425, -0.0019283, 0.0018613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003551, upper bound: 0.0003584
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003551, upper bound: 0.0003610
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0044727, -0.0014135, -0.0021383, 0.0020166
1: -0.0047015, -0.0041572, -0.0046999, -0.0041420, -0.0003623, 0.0003459
2: 0.0088445, 0.0126567, 0.0088816, 0.0127669, -0.0026652, 0.0025159
3: 1.0083407, 1.0092371, 1.0083390, 1.0092378, -0.0007127, 0.0007090
4: -0.0036681, -0.0030806, -0.0036853, -0.0030852, -0.0003818, 0.0004037
5: 0.0005021, 0.0028020, 0.0005287, 0.0028676, -0.0016304, 0.0015378
6: -0.0025729, -0.0024436, -0.0025661, -0.0024452, -0.0001186, 0.0001118
7: -0.0111804, -0.0057861, -0.0112865, -0.0058985, -0.0038661, 0.0041090
8: -0.0071714, -0.0010412, -0.0073526, -0.0010857, -0.0039470, 0.0041719
9: -0.0036259, -0.0007283, -0.0036074, -0.0006425, -0.0019528, 0.0018508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003551, upper bound: 0.0003603
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003551, upper bound: 0.0003631
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045331, -0.0017970, -0.0045075, -0.0015963, -0.0021850, 0.0019919
1: -0.0047129, -0.0042236, -0.0047010, -0.0041764, -0.0003625, 0.0003314
2: 0.0088046, 0.0122663, 0.0088460, 0.0125301, -0.0027318, 0.0024845
3: 1.0084493, 1.0092521, 1.0083661, 1.0092341, -0.0006266, 0.0006881
4: -0.0036055, -0.0030733, -0.0036481, -0.0030808, -0.0003762, 0.0004141
5: 0.0004825, 0.0025732, 0.0005029, 0.0027275, -0.0016668, 0.0015187
6: -0.0025669, -0.0024504, -0.0025729, -0.0024453, -0.0001069, 0.0001117
7: -0.0107009, -0.0057907, -0.0110223, -0.0057867, -0.0038318, 0.0040781
8: -0.0065038, -0.0009598, -0.0069623, -0.0010441, -0.0038822, 0.0042742
9: -0.0036683, -0.0010514, -0.0036243, -0.0008277, -0.0019932, 0.0018141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 193
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 134
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 234

Time for candidate selection: 22.84 seconds

### Candidate
type: B, layer: 3, pos: 193

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0001703, upper bound: 0.0003290
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0001667, upper bound: 0.0003289
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045765, -0.0018216, -0.0045075, -0.0015963, -0.0022266, 0.0019207
1: -0.0047165, -0.0042273, -0.0047010, -0.0041764, -0.0003740, 0.0003207
2: 0.0087604, 0.0122297, 0.0088460, 0.0125301, -0.0027819, 0.0023956
3: 1.0084374, 1.0092591, 1.0083661, 1.0092341, -0.0006259, 0.0007042
4: -0.0035998, -0.0030678, -0.0036481, -0.0030808, -0.0003630, 0.0004220
5: 0.0004503, 0.0025540, 0.0005029, 0.0027275, -0.0016983, 0.0014643
6: -0.0025738, -0.0024479, -0.0025729, -0.0024453, -0.0001108, 0.0001112
7: -0.0106987, -0.0056640, -0.0110223, -0.0057867, -0.0037212, 0.0041787
8: -0.0064466, -0.0009075, -0.0069623, -0.0010441, -0.0037461, 0.0043590
9: -0.0036911, -0.0010778, -0.0036243, -0.0008277, -0.0020368, 0.0017481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003439, upper bound: 0.0003477
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003439, upper bound: 0.0003663
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045357, -0.0018062, -0.0045783, -0.0015519, -0.0022101, 0.0020207
1: -0.0047155, -0.0042267, -0.0047187, -0.0041661, -0.0003815, 0.0003345
2: 0.0088014, 0.0122506, 0.0087527, 0.0125900, -0.0027743, 0.0025180
3: 1.0084549, 1.0092586, 1.0083555, 1.0092587, -0.0006366, 0.0007316
4: -0.0036026, -0.0030725, -0.0036581, -0.0030659, -0.0003810, 0.0004233
5: 0.0004805, 0.0025659, 0.0004484, 0.0027616, -0.0016869, 0.0015405
6: -0.0025668, -0.0024497, -0.0025733, -0.0024448, -0.0001081, 0.0001139
7: -0.0107067, -0.0057861, -0.0110869, -0.0056923, -0.0038998, 0.0040992
8: -0.0064734, -0.0009488, -0.0070696, -0.0008857, -0.0039276, 0.0043894
9: -0.0036753, -0.0010664, -0.0037022, -0.0007757, -0.0020602, 0.0018319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003346, upper bound: 0.0003352
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003346, upper bound: 0.0003608
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045765, -0.0018216, -0.0045783, -0.0015519, -0.0022439, 0.0020010
1: -0.0047165, -0.0042273, -0.0047187, -0.0041661, -0.0003830, 0.0003396
2: 0.0087604, 0.0122297, 0.0087527, 0.0125900, -0.0028146, 0.0025035
3: 1.0084374, 1.0092591, 1.0083555, 1.0092587, -0.0006593, 0.0007185
4: -0.0035998, -0.0030678, -0.0036581, -0.0030659, -0.0003804, 0.0004291
5: 0.0004503, 0.0025540, 0.0004484, 0.0027616, -0.0017125, 0.0015263
6: -0.0025738, -0.0024479, -0.0025733, -0.0024448, -0.0001122, 0.0001116
7: -0.0106987, -0.0056640, -0.0110869, -0.0056923, -0.0038295, 0.0041990
8: -0.0064466, -0.0009075, -0.0070696, -0.0008857, -0.0039305, 0.0044451
9: -0.0036911, -0.0010778, -0.0037022, -0.0007757, -0.0020831, 0.0018378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003368
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003329, upper bound: 0.0003633
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0045782, -0.0018063, -0.0019345, 0.0023710
1: -0.0046992, -0.0041572, -0.0047173, -0.0042241, -0.0003287, 0.0004151
2: 0.0088851, 0.0126744, 0.0087578, 0.0122493, -0.0024180, 0.0029707
3: 1.0083590, 1.0092323, 1.0084313, 1.0092630, -0.0007620, 0.0006513
4: -0.0036705, -0.0030858, -0.0036029, -0.0030673, -0.0004531, 0.0003673
5: 0.0005307, 0.0028134, 0.0004489, 0.0025657, -0.0014753, 0.0018090
6: -0.0025660, -0.0024461, -0.0025741, -0.0024472, -0.0001065, 0.0001157
7: -0.0111823, -0.0059023, -0.0107248, -0.0056637, -0.0044223, 0.0036937
8: -0.0071951, -0.0010918, -0.0064797, -0.0009020, -0.0047039, 0.0037960
9: -0.0036044, -0.0007190, -0.0036939, -0.0010620, -0.0017770, 0.0022143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003459, upper bound: 0.0002989
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003632, upper bound: 0.0003438
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0045782, -0.0018063, -0.0019753, 0.0023466
1: -0.0047015, -0.0041572, -0.0047173, -0.0042241, -0.0003377, 0.0004202
2: 0.0088445, 0.0126567, 0.0087578, 0.0122493, -0.0024728, 0.0029483
3: 1.0083407, 1.0092371, 1.0084313, 1.0092630, -0.0007869, 0.0006547
4: -0.0036681, -0.0030806, -0.0036029, -0.0030673, -0.0004513, 0.0003766
5: 0.0005021, 0.0028020, 0.0004489, 0.0025657, -0.0015069, 0.0017911
6: -0.0025729, -0.0024436, -0.0025741, -0.0024472, -0.0001102, 0.0001140
7: -0.0111804, -0.0057861, -0.0107248, -0.0056637, -0.0043509, 0.0037560
8: -0.0071714, -0.0010412, -0.0064797, -0.0009020, -0.0046973, 0.0038964
9: -0.0036259, -0.0007283, -0.0036939, -0.0010620, -0.0018258, 0.0022181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003459, upper bound: 0.0002995
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003632, upper bound: 0.0003461
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0046484, -0.0017570, -0.0019848, 0.0024323
1: -0.0046992, -0.0041572, -0.0047348, -0.0042134, -0.0003396, 0.0004316
2: 0.0088851, 0.0126744, 0.0086634, 0.0123161, -0.0024846, 0.0030555
3: 1.0083590, 1.0092323, 1.0084223, 1.0092871, -0.0007906, 0.0006599
4: -0.0036705, -0.0030858, -0.0036139, -0.0030518, -0.0004670, 0.0003781
5: 0.0005307, 0.0028134, 0.0003949, 0.0026037, -0.0015141, 0.0018567
6: -0.0025660, -0.0024461, -0.0025745, -0.0024464, -0.0001072, 0.0001158
7: -0.0111823, -0.0059023, -0.0107993, -0.0055671, -0.0044791, 0.0037561
8: -0.0071951, -0.0010918, -0.0065975, -0.0007342, -0.0048531, 0.0039121
9: -0.0036044, -0.0007190, -0.0037763, -0.0010046, -0.0018338, 0.0022875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003037, upper bound: 0.0003029
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003564, upper bound: 0.0003397
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0046484, -0.0017570, -0.0020253, 0.0024071
1: -0.0047015, -0.0041572, -0.0047348, -0.0042134, -0.0003478, 0.0004365
2: 0.0088445, 0.0126567, 0.0086634, 0.0123161, -0.0025367, 0.0030301
3: 1.0083407, 1.0092371, 1.0084223, 1.0092871, -0.0008157, 0.0006635
4: -0.0036681, -0.0030806, -0.0036139, -0.0030518, -0.0004646, 0.0003866
5: 0.0005021, 0.0028020, 0.0003949, 0.0026037, -0.0015452, 0.0018379
6: -0.0025729, -0.0024436, -0.0025745, -0.0024464, -0.0001109, 0.0001141
7: -0.0111804, -0.0057861, -0.0107993, -0.0055671, -0.0044284, 0.0038321
8: -0.0071714, -0.0010412, -0.0065975, -0.0007342, -0.0048412, 0.0040029
9: -0.0036259, -0.0007283, -0.0037763, -0.0010046, -0.0018770, 0.0022889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003037, upper bound: 0.0003038
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003564, upper bound: 0.0003415
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0045116, -0.0014930, -0.0020188, 0.0020795
1: -0.0046992, -0.0041572, -0.0047041, -0.0041571, -0.0003452, 0.0003545
2: 0.0088851, 0.0126744, 0.0088406, 0.0126605, -0.0025164, 0.0025862
3: 1.0083590, 1.0092323, 1.0083405, 1.0092448, -0.0007141, 0.0007083
4: -0.0036705, -0.0030858, -0.0036682, -0.0030797, -0.0003915, 0.0003815
5: 0.0005307, 0.0028134, 0.0004997, 0.0028063, -0.0015393, 0.0015850
6: -0.0025660, -0.0024461, -0.0025731, -0.0024424, -0.0001145, 0.0001172
7: -0.0111823, -0.0059023, -0.0112104, -0.0057808, -0.0040231, 0.0038931
8: -0.0071951, -0.0010918, -0.0071732, -0.0010296, -0.0040481, 0.0039425
9: -0.0036044, -0.0007190, -0.0036332, -0.0007276, -0.0018484, 0.0018962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003470, upper bound: 0.0003511
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003470, upper bound: 0.0003591
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0045116, -0.0014930, -0.0020513, 0.0020493
1: -0.0047015, -0.0041572, -0.0047041, -0.0041571, -0.0003532, 0.0003603
2: 0.0088445, 0.0126567, 0.0088406, 0.0126605, -0.0025568, 0.0025573
3: 1.0083407, 1.0092371, 1.0083405, 1.0092448, -0.0007397, 0.0007122
4: -0.0036681, -0.0030806, -0.0036682, -0.0030797, -0.0003891, 0.0003880
5: 0.0005021, 0.0028020, 0.0004997, 0.0028063, -0.0015639, 0.0015626
6: -0.0025729, -0.0024436, -0.0025731, -0.0024424, -0.0001184, 0.0001155
7: -0.0111804, -0.0057861, -0.0112104, -0.0057808, -0.0039469, 0.0039731
8: -0.0071714, -0.0010412, -0.0071732, -0.0010296, -0.0040360, 0.0040167
9: -0.0036259, -0.0007283, -0.0036332, -0.0007276, -0.0018836, 0.0018974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003470, upper bound: 0.0003536
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003470, upper bound: 0.0003617
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044701, -0.0014841, -0.0045822, -0.0014485, -0.0020482, 0.0021325
1: -0.0046992, -0.0041572, -0.0047217, -0.0041462, -0.0003560, 0.0003714
2: 0.0088851, 0.0126744, 0.0087475, 0.0127197, -0.0025630, 0.0026646
3: 1.0083590, 1.0092323, 1.0083300, 1.0092691, -0.0007433, 0.0007187
4: -0.0036705, -0.0030858, -0.0036781, -0.0030648, -0.0004051, 0.0003900
5: 0.0005307, 0.0028134, 0.0004454, 0.0028405, -0.0015624, 0.0016265
6: -0.0025660, -0.0024461, -0.0025734, -0.0024419, -0.0001150, 0.0001172
7: -0.0111823, -0.0059023, -0.0112729, -0.0056866, -0.0040652, 0.0039045
8: -0.0071951, -0.0010918, -0.0072811, -0.0008712, -0.0041969, 0.0040403
9: -0.0036044, -0.0007190, -0.0037109, -0.0006745, -0.0018976, 0.0019704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003540
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003606
time: 1.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045085, -0.0014990, -0.0045822, -0.0014485, -0.0020848, 0.0021051
1: -0.0047015, -0.0041572, -0.0047217, -0.0041462, -0.0003630, 0.0003770
2: 0.0088445, 0.0126567, 0.0087475, 0.0127197, -0.0026054, 0.0026357
3: 1.0083407, 1.0092371, 1.0083300, 1.0092691, -0.0007693, 0.0007229
4: -0.0036681, -0.0030806, -0.0036781, -0.0030648, -0.0004021, 0.0003964
5: 0.0005021, 0.0028020, 0.0004454, 0.0028405, -0.0015900, 0.0016060
6: -0.0025729, -0.0024436, -0.0025734, -0.0024419, -0.0001189, 0.0001154
7: -0.0111804, -0.0057861, -0.0112729, -0.0056866, -0.0040003, 0.0039931
8: -0.0071714, -0.0010412, -0.0072811, -0.0008712, -0.0041789, 0.0041089
9: -0.0036259, -0.0007283, -0.0037109, -0.0006745, -0.0019314, 0.0019696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003564
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003443, upper bound: 0.0003631
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045934, -0.0017456, -0.0044678, -0.0015790, -0.0022391, 0.0020032
1: -0.0047321, -0.0042137, -0.0046986, -0.0041759, -0.0003872, 0.0003412
2: 0.0087233, 0.0123326, 0.0088880, 0.0125530, -0.0028095, 0.0025088
3: 1.0084459, 1.0092813, 1.0083849, 1.0092287, -0.0006219, 0.0007222
4: -0.0036162, -0.0030594, -0.0036514, -0.0030862, -0.0003816, 0.0004286
5: 0.0004361, 0.0026127, 0.0005324, 0.0027408, -0.0017088, 0.0015284
6: -0.0025662, -0.0024490, -0.0025658, -0.0024478, -0.0001048, 0.0001082
7: -0.0107880, -0.0057198, -0.0110285, -0.0059060, -0.0037748, 0.0041267
8: -0.0066191, -0.0008066, -0.0069935, -0.0010971, -0.0039510, 0.0044420
9: -0.0037462, -0.0009963, -0.0036018, -0.0008154, -0.0020831, 0.0018519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003414, upper bound: 0.0003394
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003414, upper bound: 0.0003586
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045934, -0.0017456, -0.0045356, -0.0015347, -0.0021971, 0.0019862
1: -0.0047321, -0.0042137, -0.0047162, -0.0041656, -0.0003743, 0.0003393
2: 0.0087233, 0.0123326, 0.0087989, 0.0126119, -0.0027512, 0.0024882
3: 1.0084459, 1.0092813, 1.0083746, 1.0092542, -0.0006304, 0.0007034
4: -0.0036162, -0.0030594, -0.0036612, -0.0030717, -0.0003788, 0.0004185
5: 0.0004361, 0.0026127, 0.0004804, 0.0027748, -0.0016763, 0.0015154
6: -0.0025662, -0.0024490, -0.0025662, -0.0024473, -0.0001055, 0.0001086
7: -0.0107880, -0.0057198, -0.0110913, -0.0058082, -0.0037545, 0.0040730
8: -0.0066191, -0.0008066, -0.0070998, -0.0009405, -0.0039209, 0.0043289
9: -0.0037462, -0.0009963, -0.0036788, -0.0007631, -0.0020267, 0.0018369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003390
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003586
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046349, -0.0017616, -0.0044678, -0.0015790, -0.0022873, 0.0019862
1: -0.0047333, -0.0042143, -0.0046986, -0.0041759, -0.0003881, 0.0003401
2: 0.0086800, 0.0123104, 0.0088880, 0.0125530, -0.0028621, 0.0024873
3: 1.0084285, 1.0092814, 1.0083849, 1.0092287, -0.0006489, 0.0007257
4: -0.0036131, -0.0030543, -0.0036514, -0.0030862, -0.0003783, 0.0004349
5: 0.0004051, 0.0026002, 0.0005324, 0.0027408, -0.0017449, 0.0015153
6: -0.0025732, -0.0024472, -0.0025658, -0.0024478, -0.0001110, 0.0001092
7: -0.0107876, -0.0055944, -0.0110285, -0.0059060, -0.0037641, 0.0042643
8: -0.0065895, -0.0007587, -0.0069935, -0.0010971, -0.0039154, 0.0044955
9: -0.0037655, -0.0010080, -0.0036018, -0.0008154, -0.0021027, 0.0018355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003397, upper bound: 0.0003393
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003397, upper bound: 0.0003595
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046349, -0.0017616, -0.0045356, -0.0015347, -0.0022437, 0.0019693
1: -0.0047333, -0.0042143, -0.0047162, -0.0041656, -0.0003750, 0.0003380
2: 0.0086800, 0.0123104, 0.0087989, 0.0126119, -0.0028015, 0.0024657
3: 1.0084285, 1.0092814, 1.0083746, 1.0092542, -0.0006588, 0.0007099
4: -0.0036131, -0.0030543, -0.0036612, -0.0030717, -0.0003751, 0.0004245
5: 0.0004051, 0.0026002, 0.0004804, 0.0027748, -0.0017113, 0.0015024
6: -0.0025732, -0.0024472, -0.0025662, -0.0024473, -0.0001121, 0.0001097
7: -0.0107876, -0.0055944, -0.0110913, -0.0058082, -0.0037533, 0.0042169
8: -0.0065895, -0.0007587, -0.0070998, -0.0009405, -0.0038811, 0.0043808
9: -0.0037655, -0.0010080, -0.0036788, -0.0007631, -0.0020450, 0.0018176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003389
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045378, -0.0014395, -0.0045359, -0.0017915, -0.0020280, 0.0023512
1: -0.0047167, -0.0041462, -0.0047156, -0.0042235, -0.0003482, 0.0004079
2: 0.0087960, 0.0127350, 0.0088011, 0.0122697, -0.0025417, 0.0029580
3: 1.0083480, 1.0092573, 1.0084491, 1.0092607, -0.0007392, 0.0006585
4: -0.0036803, -0.0030713, -0.0036057, -0.0030725, -0.0004520, 0.0003872
5: 0.0004787, 0.0028478, 0.0004804, 0.0025772, -0.0015475, 0.0017953
6: -0.0025664, -0.0024456, -0.0025671, -0.0024493, -0.0001078, 0.0001087
7: -0.0112431, -0.0058045, -0.0107309, -0.0057858, -0.0043137, 0.0038039
8: -0.0073019, -0.0009356, -0.0065058, -0.0009483, -0.0046892, 0.0040089
9: -0.0036812, -0.0006655, -0.0036756, -0.0010505, -0.0018800, 0.0022027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003222, upper bound: 0.0002534
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003608, upper bound: 0.0003336
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045792, -0.0014544, -0.0045359, -0.0017915, -0.0020747, 0.0023372
1: -0.0047191, -0.0041463, -0.0047156, -0.0042235, -0.0003506, 0.0004076
2: 0.0087513, 0.0127159, 0.0088011, 0.0122697, -0.0025924, 0.0029397
3: 1.0083303, 1.0092615, 1.0084491, 1.0092607, -0.0007617, 0.0006630
4: -0.0036779, -0.0030656, -0.0036057, -0.0030725, -0.0004493, 0.0003934
5: 0.0004476, 0.0028363, 0.0004804, 0.0025772, -0.0015824, 0.0017845
6: -0.0025733, -0.0024431, -0.0025671, -0.0024493, -0.0001143, 0.0001101
7: -0.0112424, -0.0056917, -0.0107309, -0.0057858, -0.0043052, 0.0039480
8: -0.0072793, -0.0008827, -0.0065058, -0.0009483, -0.0046636, 0.0040653
9: -0.0037037, -0.0006753, -0.0036756, -0.0010505, -0.0019028, 0.0021918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003222, upper bound: 0.0002534
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003608, upper bound: 0.0003346
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045378, -0.0014395, -0.0046050, -0.0017413, -0.0019768, 0.0023262
1: -0.0047167, -0.0041462, -0.0047329, -0.0042128, -0.0003373, 0.0004013
2: 0.0087960, 0.0127350, 0.0087097, 0.0123377, -0.0024767, 0.0029213
3: 1.0083480, 1.0092573, 1.0084398, 1.0092846, -0.0007357, 0.0006417
4: -0.0036803, -0.0030713, -0.0036169, -0.0030575, -0.0004457, 0.0003771
5: 0.0004787, 0.0028478, 0.0004273, 0.0026159, -0.0015081, 0.0017757
6: -0.0025664, -0.0024456, -0.0025675, -0.0024486, -0.0001084, 0.0001095
7: -0.0112431, -0.0058045, -0.0107965, -0.0056946, -0.0042908, 0.0037428
8: -0.0073019, -0.0009356, -0.0066264, -0.0007881, -0.0046208, 0.0039017
9: -0.0036812, -0.0006655, -0.0037544, -0.0009931, -0.0018271, 0.0021681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003111, upper bound: 0.0002999
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003378
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045792, -0.0014544, -0.0046050, -0.0017413, -0.0020244, 0.0023137
1: -0.0047191, -0.0041463, -0.0047329, -0.0042128, -0.0003393, 0.0004009
2: 0.0087513, 0.0127159, 0.0087097, 0.0123377, -0.0025254, 0.0029042
3: 1.0083303, 1.0092615, 1.0084398, 1.0092846, -0.0007610, 0.0006478
4: -0.0036779, -0.0030656, -0.0036169, -0.0030575, -0.0004431, 0.0003829
5: 0.0004476, 0.0028363, 0.0004273, 0.0026159, -0.0015436, 0.0017660
6: -0.0025733, -0.0024431, -0.0025675, -0.0024486, -0.0001151, 0.0001110
7: -0.0112424, -0.0056917, -0.0107965, -0.0056946, -0.0042835, 0.0038896
8: -0.0072793, -0.0008827, -0.0066264, -0.0007881, -0.0045942, 0.0039543
9: -0.0037037, -0.0006753, -0.0037544, -0.0009931, -0.0018489, 0.0021563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003111, upper bound: 0.0002999
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003389
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045792, -0.0014544, -0.0044795, -0.0013873, -0.0022093, 0.0020487
1: -0.0047191, -0.0041463, -0.0046807, -0.0041319, -0.0003937, 0.0003375
2: 0.0087513, 0.0127159, 0.0088903, 0.0128067, -0.0027754, 0.0025436
3: 1.0083303, 1.0092615, 1.0083150, 1.0091815, -0.0006882, 0.0007879
4: -0.0036779, -0.0030656, -0.0036924, -0.0030897, -0.0003837, 0.0004240
5: 0.0004476, 0.0028363, 0.0005250, 0.0028882, -0.0016864, 0.0015612
6: -0.0025733, -0.0024431, -0.0025702, -0.0024504, -0.0001154, 0.0001202
7: -0.0112424, -0.0056917, -0.0112935, -0.0057916, -0.0040130, 0.0041318
8: -0.0072793, -0.0008827, -0.0074354, -0.0011550, -0.0039505, 0.0044032
9: -0.0037037, -0.0006753, -0.0035612, -0.0005982, -0.0020733, 0.0018420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003527
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003522, upper bound: 0.0003527
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045378, -0.0014395, -0.0044715, -0.0014002, -0.0021437, 0.0020470
1: -0.0047167, -0.0041462, -0.0046998, -0.0041385, -0.0003799, 0.0003523
2: 0.0087960, 0.0127350, 0.0088832, 0.0127858, -0.0026932, 0.0025587
3: 1.0083480, 1.0092573, 1.0083359, 1.0092376, -0.0006931, 0.0007357
4: -0.0036803, -0.0030713, -0.0036883, -0.0030854, -0.0003892, 0.0004116
5: 0.0004787, 0.0028478, 0.0005296, 0.0028779, -0.0016364, 0.0015614
6: -0.0025664, -0.0024456, -0.0025660, -0.0024452, -0.0001109, 0.0001123
7: -0.0112431, -0.0058045, -0.0113032, -0.0059008, -0.0039007, 0.0039946
8: -0.0073019, -0.0009356, -0.0073851, -0.0010882, -0.0040315, 0.0042761
9: -0.0036812, -0.0006655, -0.0036063, -0.0006259, -0.0020128, 0.0018914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003636, upper bound: 0.0003578
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003636, upper bound: 0.0003578
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045792, -0.0014544, -0.0044715, -0.0014002, -0.0022007, 0.0020437
1: -0.0047191, -0.0041463, -0.0046998, -0.0041385, -0.0003823, 0.0003517
2: 0.0087513, 0.0127159, 0.0088832, 0.0127858, -0.0027572, 0.0025512
3: 1.0083303, 1.0092615, 1.0083359, 1.0092376, -0.0007190, 0.0007421
4: -0.0036779, -0.0030656, -0.0036883, -0.0030854, -0.0003875, 0.0004197
5: 0.0004476, 0.0028363, 0.0005296, 0.0028779, -0.0016792, 0.0015586
6: -0.0025733, -0.0024431, -0.0025660, -0.0024452, -0.0001182, 0.0001135
7: -0.0112424, -0.0056917, -0.0113032, -0.0059008, -0.0039159, 0.0041524
8: -0.0072793, -0.0008827, -0.0073851, -0.0010882, -0.0040079, 0.0043475
9: -0.0037037, -0.0006753, -0.0036063, -0.0006259, -0.0020403, 0.0018799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003636, upper bound: 0.0003597
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003636, upper bound: 0.0003596
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045934, -0.0017456, -0.0045061, -0.0015968, -0.0022243, 0.0020437
1: -0.0047321, -0.0042137, -0.0047009, -0.0041766, -0.0003862, 0.0003430
2: 0.0087233, 0.0123326, 0.0088476, 0.0125295, -0.0027899, 0.0025515
3: 1.0084459, 1.0092813, 1.0083666, 1.0092336, -0.0006302, 0.0007499
4: -0.0036162, -0.0030594, -0.0036480, -0.0030811, -0.0003868, 0.0004255
5: 0.0004361, 0.0026127, 0.0005040, 0.0027271, -0.0016974, 0.0015586
6: -0.0025662, -0.0024490, -0.0025728, -0.0024454, -0.0001059, 0.0001146
7: -0.0107880, -0.0057198, -0.0110213, -0.0057900, -0.0039130, 0.0041134
8: -0.0066191, -0.0008066, -0.0069614, -0.0010464, -0.0039977, 0.0044092
9: -0.0037462, -0.0009963, -0.0036233, -0.0008281, -0.0020670, 0.0018709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003416, upper bound: 0.0003375
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003416, upper bound: 0.0003564
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045934, -0.0017456, -0.0045769, -0.0015524, -0.0021595, 0.0020272
1: -0.0047321, -0.0042137, -0.0047186, -0.0041662, -0.0003708, 0.0003413
2: 0.0087233, 0.0123326, 0.0087543, 0.0125894, -0.0027069, 0.0025312
3: 1.0084459, 1.0092813, 1.0083563, 1.0092582, -0.0006392, 0.0007306
4: -0.0036162, -0.0030594, -0.0036580, -0.0030661, -0.0003840, 0.0004124
5: 0.0004361, 0.0026127, 0.0004494, 0.0027612, -0.0016480, 0.0015460
6: -0.0025662, -0.0024490, -0.0025731, -0.0024449, -0.0001082, 0.0001150
7: -0.0107880, -0.0057198, -0.0110858, -0.0056956, -0.0038935, 0.0040327
8: -0.0066191, -0.0008066, -0.0070687, -0.0008880, -0.0039673, 0.0042706
9: -0.0037462, -0.0009963, -0.0037012, -0.0007761, -0.0020018, 0.0018554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003370
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003394, upper bound: 0.0003564
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046349, -0.0017616, -0.0045061, -0.0015968, -0.0022626, 0.0020308
1: -0.0047333, -0.0042143, -0.0047009, -0.0041766, -0.0003886, 0.0003493
2: 0.0086800, 0.0123104, 0.0088476, 0.0125295, -0.0028343, 0.0025435
3: 1.0084285, 1.0092814, 1.0083666, 1.0092336, -0.0006532, 0.0007351
4: -0.0036131, -0.0030543, -0.0036480, -0.0030811, -0.0003872, 0.0004316
5: 0.0004051, 0.0026002, 0.0005040, 0.0027271, -0.0017263, 0.0015493
6: -0.0025732, -0.0024472, -0.0025728, -0.0024454, -0.0001097, 0.0001124
7: -0.0107876, -0.0055944, -0.0110213, -0.0057900, -0.0038456, 0.0042065
8: -0.0065895, -0.0007587, -0.0069614, -0.0010464, -0.0040109, 0.0044688
9: -0.0037655, -0.0010080, -0.0036233, -0.0008281, -0.0020935, 0.0018819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003397, upper bound: 0.0003392
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003397, upper bound: 0.0003595
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046349, -0.0017616, -0.0045769, -0.0015524, -0.0021949, 0.0020139
1: -0.0047333, -0.0042143, -0.0047186, -0.0041662, -0.0003724, 0.0003474
2: 0.0086800, 0.0123104, 0.0087543, 0.0125894, -0.0027477, 0.0025221
3: 1.0084285, 1.0092814, 1.0083563, 1.0092582, -0.0006639, 0.0007166
4: -0.0036131, -0.0030543, -0.0036580, -0.0030661, -0.0003842, 0.0004180
5: 0.0004051, 0.0026002, 0.0004494, 0.0027612, -0.0016747, 0.0015364
6: -0.0025732, -0.0024472, -0.0025731, -0.0024449, -0.0001124, 0.0001129
7: -0.0107876, -0.0055944, -0.0110858, -0.0056956, -0.0038277, 0.0041283
8: -0.0065895, -0.0007587, -0.0070687, -0.0008880, -0.0039775, 0.0043251
9: -0.0037655, -0.0010080, -0.0037012, -0.0007761, -0.0020243, 0.0018650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003389
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003374, upper bound: 0.0003595
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045378, -0.0014395, -0.0045782, -0.0018063, -0.0020150, 0.0024177
1: -0.0047167, -0.0041462, -0.0047173, -0.0042241, -0.0003469, 0.0004258
2: 0.0087960, 0.0127350, 0.0087578, 0.0122493, -0.0025217, 0.0030335
3: 1.0083480, 1.0092573, 1.0084313, 1.0092630, -0.0007769, 0.0006858
4: -0.0036803, -0.0030713, -0.0036029, -0.0030673, -0.0004633, 0.0003836
5: 0.0004787, 0.0028478, 0.0004489, 0.0025657, -0.0015372, 0.0018451
6: -0.0025664, -0.0024456, -0.0025741, -0.0024472, -0.0001070, 0.0001162
7: -0.0112431, -0.0058045, -0.0107248, -0.0056637, -0.0044849, 0.0038008
8: -0.0073019, -0.0009356, -0.0064797, -0.0009020, -0.0048139, 0.0039694
9: -0.0036812, -0.0006655, -0.0036939, -0.0010620, -0.0018615, 0.0022671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003209, upper bound: 0.0002498
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003608, upper bound: 0.0003329
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045792, -0.0014544, -0.0045782, -0.0018063, -0.0020564, 0.0023934
1: -0.0047191, -0.0041463, -0.0047173, -0.0042241, -0.0003558, 0.0004309
2: 0.0087513, 0.0127159, 0.0087578, 0.0122493, -0.0025780, 0.0030115
3: 1.0083303, 1.0092615, 1.0084313, 1.0092630, -0.0008000, 0.0006866
4: -0.0036779, -0.0030656, -0.0036029, -0.0030673, -0.0004616, 0.0003930
5: 0.0004476, 0.0028363, 0.0004489, 0.0025657, -0.0015692, 0.0018273
6: -0.0025733, -0.0024431, -0.0025741, -0.0024472, -0.0001105, 0.0001145
7: -0.0112424, -0.0056917, -0.0107248, -0.0056637, -0.0044133, 0.0038675
8: -0.0072793, -0.0008827, -0.0064797, -0.0009020, -0.0048075, 0.0040709
9: -0.0037037, -0.0006753, -0.0036939, -0.0010620, -0.0019105, 0.0022709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003209, upper bound: 0.0002498
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003608, upper bound: 0.0003346
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045378, -0.0014395, -0.0046484, -0.0017570, -0.0019639, 0.0023969
1: -0.0047167, -0.0041462, -0.0047348, -0.0042134, -0.0003360, 0.0004229
2: 0.0087960, 0.0127350, 0.0086634, 0.0123161, -0.0024575, 0.0030050
3: 1.0083480, 1.0092573, 1.0084223, 1.0092871, -0.0007794, 0.0006683
4: -0.0036803, -0.0030713, -0.0036139, -0.0030518, -0.0004587, 0.0003737
5: 0.0004787, 0.0028478, 0.0003949, 0.0026037, -0.0014980, 0.0018291
6: -0.0025664, -0.0024456, -0.0025745, -0.0024464, -0.0001077, 0.0001170
7: -0.0112431, -0.0058045, -0.0107993, -0.0055671, -0.0044620, 0.0037393
8: -0.0073019, -0.0009356, -0.0065975, -0.0007342, -0.0047652, 0.0038648
9: -0.0036812, -0.0006655, -0.0037763, -0.0010046, -0.0018097, 0.0022449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0003088, upper bound: 0.0002975
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003370
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045792, -0.0014544, -0.0046484, -0.0017570, -0.0020042, 0.0023726
1: -0.0047191, -0.0041463, -0.0047348, -0.0042134, -0.0003449, 0.0004277
2: 0.0087513, 0.0127159, 0.0086634, 0.0123161, -0.0025104, 0.0029823
3: 1.0083303, 1.0092615, 1.0084223, 1.0092871, -0.0008034, 0.0006706
4: -0.0036779, -0.0030656, -0.0036139, -0.0030518, -0.0004568, 0.0003824
5: 0.0004476, 0.0028363, 0.0003949, 0.0026037, -0.0015290, 0.0018110
6: -0.0025733, -0.0024431, -0.0025745, -0.0024464, -0.0001115, 0.0001153
7: -0.0112424, -0.0056917, -0.0107993, -0.0055671, -0.0043884, 0.0038064
8: -0.0072793, -0.0008827, -0.0065975, -0.0007342, -0.0047569, 0.0039582
9: -0.0037037, -0.0006753, -0.0037763, -0.0010046, -0.0018554, 0.0022465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.05 + 597.07 = 600.12 seconds
