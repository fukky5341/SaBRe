## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0016185600000000002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013854)
1: (-0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035157)
2: (0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021812)
3: (0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040728, 0.0040728)
4: (-0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035761)
5: (0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013545)
6: (0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051689, 0.0051689)
7: (0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036170, 0.0036170)
8: (-0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038780, 0.0038780)
9: (-0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025616)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 2.83 = 4.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0020232, upper bound: 0.0020232

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019475, upper bound: 0.0018876
time: 1.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019475, upper bound: 0.0019475
time: 1.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.18
Output dim: 7, lower bound: -0.0019475, upper bound: 0.0018876
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.18
Output dim: 7, lower bound: -0.0019475, upper bound: 0.0019475

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0020806, -0.0003829, -0.0021188, -0.0003786, -0.0013485, 0.0013815
1: -0.0095904, -0.0052823, -0.0096874, -0.0052713, -0.0034220, 0.0035058
2: 0.0290801, 0.0317528, 0.0290199, 0.0317597, -0.0021230, 0.0021750
3: 0.0002719, 0.0052626, 0.0002591, 0.0053750, -0.0040613, 0.0039642
4: -0.0086481, -0.0042660, -0.0087468, -0.0042548, -0.0034807, 0.0035660
5: 0.0104625, 0.0121223, 0.0104251, 0.0121266, -0.0013184, 0.0013507
6: 0.0007332, 0.0070671, 0.0007170, 0.0072097, -0.0051543, 0.0050311
7: 0.9785724, 0.9830045, 0.9785610, 0.9831042, -0.0036068, 0.0035205
8: -0.0095381, -0.0047861, -0.0095502, -0.0046791, -0.0038670, 0.0037745
9: -0.0018381, 0.0013008, -0.0019088, 0.0013089, -0.0024933, 0.0025544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018876, upper bound: 0.0018876
time: 1.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018876, upper bound: 0.0018876
time: 2.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0020580, -0.0002336, -0.0020928, -0.0003835, -0.0013663, 0.0015110
1: -0.0095330, -0.0049033, -0.0096214, -0.0052839, -0.0034671, 0.0038344
2: 0.0291157, 0.0319880, 0.0290609, 0.0317519, -0.0021510, 0.0023789
3: -0.0001672, 0.0051961, 0.0002737, 0.0052985, -0.0044419, 0.0040165
4: -0.0085897, -0.0038805, -0.0086796, -0.0042676, -0.0035267, 0.0039002
5: 0.0104846, 0.0122683, 0.0104506, 0.0121217, -0.0013358, 0.0014773
6: 0.0001760, 0.0069826, 0.0007355, 0.0071126, -0.0056374, 0.0050975
7: 0.9781824, 0.9829454, 0.9785739, 0.9830363, -0.0039448, 0.0035670
8: -0.0099561, -0.0048495, -0.0095364, -0.0047519, -0.0042294, 0.0038243
9: -0.0017962, 0.0015770, -0.0018607, 0.0012997, -0.0025262, 0.0027938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018698, upper bound: 0.0019003
time: 1.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0019003
time: 1.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.81 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.81
Output dim: 7, lower bound: -0.0018876, upper bound: 0.0018876
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.81
Output dim: 7, lower bound: -0.0018876, upper bound: 0.0018876
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.81
Output dim: 7, lower bound: -0.0018698, upper bound: 0.0019003
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.81
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0019003

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020806, -0.0003829, -0.0020806, -0.0003829, -0.0013446, 0.0013446
1: -0.0095904, -0.0052823, -0.0095904, -0.0052823, -0.0034120, 0.0034120
2: 0.0290801, 0.0317528, 0.0290801, 0.0317528, -0.0021168, 0.0021168
3: 0.0002719, 0.0052626, 0.0002719, 0.0052626, -0.0039527, 0.0039527
4: -0.0086481, -0.0042660, -0.0086481, -0.0042660, -0.0034706, 0.0034706
5: 0.0104625, 0.0121223, 0.0104625, 0.0121223, -0.0013146, 0.0013146
6: 0.0007332, 0.0070671, 0.0007332, 0.0070671, -0.0050165, 0.0050165
7: 0.9785724, 0.9830045, 0.9785724, 0.9830045, -0.0035103, 0.0035103
8: -0.0095381, -0.0047861, -0.0095381, -0.0047861, -0.0037636, 0.0037636
9: -0.0018381, 0.0013008, -0.0018381, 0.0013008, -0.0024861, 0.0024861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018558, upper bound: 0.0018055
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018559, upper bound: 0.0018427
time: 1.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020806, -0.0003829, -0.0020580, -0.0002336, -0.0014984, 0.0013298
1: -0.0095904, -0.0052823, -0.0095330, -0.0049033, -0.0038025, 0.0033745
2: 0.0290801, 0.0317528, 0.0291157, 0.0319880, -0.0023591, 0.0020936
3: 0.0002719, 0.0052626, -0.0001672, 0.0051961, -0.0039092, 0.0044050
4: -0.0086481, -0.0042660, -0.0085897, -0.0038805, -0.0038678, 0.0034325
5: 0.0104625, 0.0121223, 0.0104846, 0.0122683, -0.0014650, 0.0013001
6: 0.0007332, 0.0070671, 0.0001760, 0.0069826, -0.0049613, 0.0055906
7: 0.9785724, 0.9830045, 0.9781824, 0.9829454, -0.0034717, 0.0039120
8: -0.0095381, -0.0047861, -0.0099561, -0.0048495, -0.0037222, 0.0041943
9: -0.0018381, 0.0013008, -0.0017962, 0.0015770, -0.0027706, 0.0024587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018558, upper bound: 0.0018055
time: 1.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018559, upper bound: 0.0018427
time: 2.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020283, -0.0002391, -0.0020191, -0.0004169, -0.0013065, 0.0014378
1: -0.0094577, -0.0049173, -0.0094344, -0.0053685, -0.0033153, 0.0036487
2: 0.0291624, 0.0319793, 0.0291769, 0.0316994, -0.0020568, 0.0022637
3: -0.0001510, 0.0051089, 0.0003717, 0.0050819, -0.0042269, 0.0038406
4: -0.0085131, -0.0038948, -0.0084894, -0.0043537, -0.0033722, 0.0037114
5: 0.0105136, 0.0122630, 0.0105226, 0.0120891, -0.0012773, 0.0014058
6: 0.0001965, 0.0068720, 0.0008599, 0.0068377, -0.0053645, 0.0048742
7: 0.9781967, 0.9828679, 0.9786610, 0.9828440, -0.0037538, 0.0034108
8: -0.0099407, -0.0049325, -0.0094430, -0.0049582, -0.0040247, 0.0036569
9: -0.0017414, 0.0015668, -0.0017244, 0.0012380, -0.0024156, 0.0026585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018080, upper bound: 0.0017525
time: 1.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018282, upper bound: 0.0018587
time: 1.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020428, -0.0002369, -0.0020546, -0.0003923, -0.0013416, 0.0014352
1: -0.0094945, -0.0049118, -0.0095244, -0.0053061, -0.0034044, 0.0036420
2: 0.0291396, 0.0319827, 0.0291210, 0.0317381, -0.0021121, 0.0022595
3: -0.0001573, 0.0051515, 0.0002994, 0.0051861, -0.0042190, 0.0039438
4: -0.0085505, -0.0038892, -0.0085809, -0.0042902, -0.0034628, 0.0037045
5: 0.0104995, 0.0122651, 0.0104879, 0.0121132, -0.0013116, 0.0014032
6: 0.0001885, 0.0069260, 0.0007681, 0.0069700, -0.0053545, 0.0050052
7: 0.9781911, 0.9829058, 0.9785968, 0.9829364, -0.0037468, 0.0035024
8: -0.0099468, -0.0048919, -0.0095119, -0.0048589, -0.0040172, 0.0037551
9: -0.0017682, 0.0015708, -0.0017900, 0.0012835, -0.0024805, 0.0026536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018352, upper bound: 0.0017525
time: 1.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0018587
time: 1.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.62 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018558, upper bound: 0.0018055
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018559, upper bound: 0.0018427
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018558, upper bound: 0.0018055
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018559, upper bound: 0.0018427
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018080, upper bound: 0.0017525
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018282, upper bound: 0.0018587
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018352, upper bound: 0.0017525
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0018587

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020057, -0.0004163, -0.0020505, -0.0003884, -0.0012708, 0.0012845
1: -0.0094002, -0.0053670, -0.0095141, -0.0052963, -0.0032249, 0.0032597
2: 0.0291981, 0.0317003, 0.0291274, 0.0317442, -0.0020007, 0.0020223
3: 0.0003700, 0.0050423, 0.0002881, 0.0051742, -0.0037762, 0.0037359
4: -0.0084546, -0.0043522, -0.0085705, -0.0042802, -0.0032802, 0.0033157
5: 0.0105358, 0.0120897, 0.0104919, 0.0121169, -0.0012425, 0.0012559
6: 0.0008577, 0.0067874, 0.0007537, 0.0069549, -0.0047925, 0.0047413
7: 0.9786595, 0.9828088, 0.9785867, 0.9829260, -0.0033536, 0.0033177
8: -0.0094447, -0.0049959, -0.0095227, -0.0048703, -0.0035956, 0.0035571
9: -0.0016995, 0.0012391, -0.0017825, 0.0012907, -0.0023497, 0.0023751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017686, upper bound: 0.0017970
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018301, upper bound: 0.0017983
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020436, -0.0003917, -0.0020663, -0.0003863, -0.0012707, 0.0013199
1: -0.0094966, -0.0053046, -0.0095541, -0.0052908, -0.0032245, 0.0033493
2: 0.0291383, 0.0317390, 0.0291026, 0.0317476, -0.0020005, 0.0020779
3: 0.0002977, 0.0051539, 0.0002817, 0.0052206, -0.0038800, 0.0037354
4: -0.0085527, -0.0042887, -0.0086112, -0.0042747, -0.0032798, 0.0034068
5: 0.0104987, 0.0121137, 0.0104765, 0.0121190, -0.0012423, 0.0012904
6: 0.0007660, 0.0069291, 0.0007457, 0.0070137, -0.0049243, 0.0047407
7: 0.9785953, 0.9829079, 0.9785811, 0.9829671, -0.0034458, 0.0033173
8: -0.0095135, -0.0048896, -0.0095287, -0.0048262, -0.0036944, 0.0035567
9: -0.0017697, 0.0012846, -0.0018116, 0.0012946, -0.0023494, 0.0024404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017686, upper bound: 0.0018281
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018301, upper bound: 0.0018301
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020057, -0.0004163, -0.0020283, -0.0002391, -0.0014249, 0.0012708
1: -0.0094002, -0.0053670, -0.0094577, -0.0049173, -0.0036159, 0.0032248
2: 0.0291981, 0.0317003, 0.0291624, 0.0319793, -0.0022433, 0.0020007
3: 0.0003700, 0.0050423, -0.0001510, 0.0051089, -0.0037358, 0.0041888
4: -0.0084546, -0.0043522, -0.0085131, -0.0038948, -0.0036780, 0.0032802
5: 0.0105358, 0.0120897, 0.0105136, 0.0122630, -0.0013931, 0.0012424
6: 0.0008577, 0.0067874, 0.0001965, 0.0068720, -0.0047412, 0.0053162
7: 0.9786595, 0.9828088, 0.9781967, 0.9828679, -0.0033177, 0.0037200
8: -0.0094447, -0.0049959, -0.0099407, -0.0049325, -0.0035571, 0.0039884
9: -0.0016995, 0.0012391, -0.0017414, 0.0015668, -0.0026346, 0.0023497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017550
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017656
time: 1.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020436, -0.0003917, -0.0020428, -0.0002369, -0.0014223, 0.0013050
1: -0.0094966, -0.0053046, -0.0094945, -0.0049118, -0.0036094, 0.0033115
2: 0.0291383, 0.0317390, 0.0291396, 0.0319827, -0.0022393, 0.0020545
3: 0.0002977, 0.0051539, -0.0001573, 0.0051515, -0.0038362, 0.0041813
4: -0.0085527, -0.0042887, -0.0085505, -0.0038892, -0.0036714, 0.0033684
5: 0.0104987, 0.0121137, 0.0104995, 0.0122651, -0.0013906, 0.0012758
6: 0.0007660, 0.0069291, 0.0001885, 0.0069260, -0.0048687, 0.0053066
7: 0.9785953, 0.9829079, 0.9781911, 0.9829058, -0.0034069, 0.0037133
8: -0.0095135, -0.0048896, -0.0099468, -0.0048919, -0.0036527, 0.0039813
9: -0.0017697, 0.0012846, -0.0017682, 0.0015708, -0.0026299, 0.0024128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017846
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017981
time: 1.91 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019280, -0.0002481, -0.0019781, -0.0004241, -0.0012164, 0.0013974
1: -0.0092032, -0.0049401, -0.0093304, -0.0053869, -0.0030867, 0.0035461
2: 0.0293203, 0.0319652, 0.0292414, 0.0316880, -0.0019150, 0.0022000
3: -0.0001246, 0.0048141, 0.0003931, 0.0049614, -0.0041080, 0.0035758
4: -0.0082542, -0.0039179, -0.0083836, -0.0043724, -0.0031397, 0.0036070
5: 0.0106117, 0.0122542, 0.0105627, 0.0120820, -0.0011892, 0.0013662
6: 0.0002300, 0.0064978, 0.0008870, 0.0066847, -0.0052135, 0.0045382
7: 0.9782202, 0.9826061, 0.9786800, 0.9827370, -0.0036482, 0.0031756
8: -0.0099156, -0.0052132, -0.0094227, -0.0050730, -0.0039114, 0.0034048
9: -0.0015560, 0.0015502, -0.0016486, 0.0012246, -0.0022490, 0.0025837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017673, upper bound: 0.0017085
time: 1.80 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017673, upper bound: 0.0017111
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020063, -0.0002447, -0.0020147, -0.0004180, -0.0012522, 0.0014273
1: -0.0094019, -0.0049315, -0.0094231, -0.0053714, -0.0031777, 0.0036220
2: 0.0291971, 0.0319705, 0.0291839, 0.0316976, -0.0019715, 0.0022471
3: -0.0001345, 0.0050442, 0.0003751, 0.0050688, -0.0041960, 0.0036812
4: -0.0084563, -0.0039092, -0.0084779, -0.0043566, -0.0032323, 0.0036842
5: 0.0105352, 0.0122575, 0.0105270, 0.0120880, -0.0012243, 0.0013955
6: 0.0002174, 0.0067899, 0.0008641, 0.0068211, -0.0053252, 0.0046719
7: 0.9782115, 0.9828105, 0.9786639, 0.9828324, -0.0037263, 0.0032692
8: -0.0099250, -0.0049941, -0.0094398, -0.0049707, -0.0039952, 0.0035051
9: -0.0017007, 0.0015564, -0.0017162, 0.0012360, -0.0023153, 0.0026391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017839, upper bound: 0.0018118
time: 1.88 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017839, upper bound: 0.0018120
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019436, -0.0002461, -0.0020149, -0.0003997, -0.0012521, 0.0013979
1: -0.0092428, -0.0049352, -0.0094237, -0.0053248, -0.0031773, 0.0035473
2: 0.0292958, 0.0319682, 0.0291835, 0.0317265, -0.0019712, 0.0022007
3: -0.0001302, 0.0048599, 0.0003211, 0.0050694, -0.0041093, 0.0036808
4: -0.0082945, -0.0039129, -0.0084785, -0.0043092, -0.0032319, 0.0036082
5: 0.0105964, 0.0122561, 0.0105268, 0.0121060, -0.0012241, 0.0013667
6: 0.0002228, 0.0065560, 0.0007956, 0.0068219, -0.0052153, 0.0046714
7: 0.9782152, 0.9826468, 0.9786160, 0.9828328, -0.0036494, 0.0032688
8: -0.0099210, -0.0051696, -0.0094912, -0.0049701, -0.0039127, 0.0035047
9: -0.0015848, 0.0015538, -0.0017166, 0.0012699, -0.0023150, 0.0025846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017924, upper bound: 0.0017085
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017924, upper bound: 0.0017111
time: 1.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020203, -0.0002425, -0.0020498, -0.0003934, -0.0012889, 0.0014246
1: -0.0094373, -0.0049259, -0.0095122, -0.0053089, -0.0032709, 0.0036152
2: 0.0291751, 0.0319740, 0.0291286, 0.0317364, -0.0020293, 0.0022429
3: -0.0001410, 0.0050853, 0.0003027, 0.0051720, -0.0041880, 0.0037892
4: -0.0084924, -0.0039035, -0.0085685, -0.0042931, -0.0033270, 0.0036772
5: 0.0105215, 0.0122596, 0.0104927, 0.0121121, -0.0012602, 0.0013928
6: 0.0002092, 0.0068420, 0.0007723, 0.0069520, -0.0053151, 0.0048089
7: 0.9782056, 0.9828470, 0.9785997, 0.9829239, -0.0037192, 0.0033651
8: -0.0099312, -0.0049550, -0.0095088, -0.0048724, -0.0039876, 0.0036079
9: -0.0017266, 0.0015605, -0.0017811, 0.0012815, -0.0023832, 0.0026340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018118
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018120
time: 1.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.36 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017686, upper bound: 0.0017970
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0018301, upper bound: 0.0017983
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017686, upper bound: 0.0018281
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0018301, upper bound: 0.0018301
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017550
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017656
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017846
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017981
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017673, upper bound: 0.0017085
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017673, upper bound: 0.0017111
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017839, upper bound: 0.0018118
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017839, upper bound: 0.0018120
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017924, upper bound: 0.0017085
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0017924, upper bound: 0.0017111
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018118
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018120

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019662, -0.0004236, -0.0019548, -0.0003976, -0.0012311, 0.0011943
1: -0.0093001, -0.0053855, -0.0092711, -0.0053196, -0.0031241, 0.0030307
2: 0.0292602, 0.0316889, 0.0292782, 0.0317298, -0.0019382, 0.0018803
3: 0.0003914, 0.0049263, 0.0003150, 0.0048927, -0.0035110, 0.0036191
4: -0.0083528, -0.0043709, -0.0083233, -0.0043039, -0.0031777, 0.0030828
5: 0.0105744, 0.0120826, 0.0105856, 0.0121080, -0.0012036, 0.0011677
6: 0.0008848, 0.0066402, 0.0007879, 0.0065975, -0.0044559, 0.0045931
7: 0.9786785, 0.9827058, 0.9786106, 0.9826760, -0.0031180, 0.0032141
8: -0.0094243, -0.0051064, -0.0094970, -0.0051384, -0.0033430, 0.0034460
9: -0.0016265, 0.0012257, -0.0016054, 0.0012737, -0.0022763, 0.0022082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017243, upper bound: 0.0017573
time: 1.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017295, upper bound: 0.0017573
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020010, -0.0004174, -0.0020281, -0.0003940, -0.0012606, 0.0012311
1: -0.0093884, -0.0053699, -0.0094573, -0.0053104, -0.0031989, 0.0031240
2: 0.0292054, 0.0316985, 0.0291627, 0.0317355, -0.0019846, 0.0019382
3: 0.0003733, 0.0050286, 0.0003044, 0.0051084, -0.0036190, 0.0037058
4: -0.0084426, -0.0043551, -0.0085126, -0.0042946, -0.0032538, 0.0031777
5: 0.0105403, 0.0120886, 0.0105138, 0.0121115, -0.0012325, 0.0012036
6: 0.0008619, 0.0067701, 0.0007744, 0.0068713, -0.0045930, 0.0047031
7: 0.9786624, 0.9827967, 0.9786012, 0.9828675, -0.0032140, 0.0032910
8: -0.0094415, -0.0050089, -0.0095071, -0.0049330, -0.0034459, 0.0035285
9: -0.0016909, 0.0012370, -0.0017411, 0.0012804, -0.0023307, 0.0022762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017857, upper bound: 0.0017591
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017888, upper bound: 0.0017591
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020045, -0.0003991, -0.0019706, -0.0003958, -0.0012288, 0.0012302
1: -0.0093972, -0.0053234, -0.0093111, -0.0053151, -0.0031182, 0.0031219
2: 0.0292000, 0.0317274, 0.0292533, 0.0317325, -0.0019346, 0.0019368
3: 0.0003195, 0.0050388, 0.0003098, 0.0049391, -0.0036165, 0.0036123
4: -0.0084515, -0.0043078, -0.0083640, -0.0042993, -0.0031718, 0.0031755
5: 0.0105370, 0.0121065, 0.0105701, 0.0121097, -0.0012014, 0.0012028
6: 0.0007936, 0.0067830, 0.0007813, 0.0066565, -0.0045898, 0.0045845
7: 0.9786146, 0.9828056, 0.9786060, 0.9827172, -0.0032117, 0.0032080
8: -0.0094928, -0.0049993, -0.0095020, -0.0050942, -0.0034435, 0.0034395
9: -0.0016973, 0.0012709, -0.0016346, 0.0012770, -0.0022720, 0.0022746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017244, upper bound: 0.0017865
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017295, upper bound: 0.0017865
time: 1.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020388, -0.0003928, -0.0020434, -0.0003918, -0.0012603, 0.0012679
1: -0.0094842, -0.0053075, -0.0094960, -0.0053048, -0.0031983, 0.0032176
2: 0.0291460, 0.0317373, 0.0291387, 0.0317389, -0.0019842, 0.0019962
3: 0.0003010, 0.0051396, 0.0002979, 0.0051533, -0.0037274, 0.0037051
4: -0.0085401, -0.0042916, -0.0085521, -0.0042889, -0.0032532, 0.0032728
5: 0.0105034, 0.0121126, 0.0104989, 0.0121137, -0.0012322, 0.0012396
6: 0.0007701, 0.0069109, 0.0007663, 0.0069283, -0.0047305, 0.0047022
7: 0.9785982, 0.9828951, 0.9785954, 0.9829073, -0.0033102, 0.0032904
8: -0.0095104, -0.0049033, -0.0095133, -0.0048903, -0.0035491, 0.0035278
9: -0.0017607, 0.0012825, -0.0017693, 0.0012845, -0.0023303, 0.0023444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017857, upper bound: 0.0017888
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017888, upper bound: 0.0017888
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019662, -0.0004236, -0.0019280, -0.0002481, -0.0013844, 0.0011761
1: -0.0093001, -0.0053855, -0.0092032, -0.0049401, -0.0035132, 0.0029845
2: 0.0292602, 0.0316889, 0.0293203, 0.0319652, -0.0021796, 0.0018516
3: 0.0003914, 0.0049263, -0.0001246, 0.0048141, -0.0034575, 0.0040699
4: -0.0083528, -0.0043709, -0.0082542, -0.0039179, -0.0035735, 0.0030358
5: 0.0105744, 0.0120826, 0.0106117, 0.0122542, -0.0013535, 0.0011499
6: 0.0008848, 0.0066402, 0.0002300, 0.0064978, -0.0043880, 0.0051652
7: 0.9786785, 0.9827058, 0.9782202, 0.9826061, -0.0030705, 0.0036143
8: -0.0094243, -0.0051064, -0.0099156, -0.0052132, -0.0032920, 0.0038751
9: -0.0016265, 0.0012257, -0.0015560, 0.0015502, -0.0025597, 0.0021746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017117
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017117
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020010, -0.0004174, -0.0020063, -0.0002447, -0.0014146, 0.0012139
1: -0.0093884, -0.0053699, -0.0094019, -0.0049315, -0.0035897, 0.0030804
2: 0.0292054, 0.0316985, 0.0291971, 0.0319705, -0.0022270, 0.0019111
3: 0.0003733, 0.0050286, -0.0001345, 0.0050442, -0.0035685, 0.0041584
4: -0.0084426, -0.0043551, -0.0084563, -0.0039092, -0.0036513, 0.0031333
5: 0.0105403, 0.0120886, 0.0105352, 0.0122575, -0.0013830, 0.0011868
6: 0.0008619, 0.0067701, 0.0002174, 0.0067899, -0.0045289, 0.0052776
7: 0.9786624, 0.9827967, 0.9782115, 0.9828105, -0.0031691, 0.0036930
8: -0.0094415, -0.0050089, -0.0099250, -0.0049941, -0.0033978, 0.0039595
9: -0.0016909, 0.0012370, -0.0017007, 0.0015564, -0.0026155, 0.0022444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017217
time: 1.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017217
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020045, -0.0003991, -0.0019436, -0.0002461, -0.0013851, 0.0012087
1: -0.0093972, -0.0053234, -0.0092428, -0.0049352, -0.0035149, 0.0030672
2: 0.0292000, 0.0317274, 0.0292958, 0.0319682, -0.0021806, 0.0019029
3: 0.0003195, 0.0050388, -0.0001302, 0.0048599, -0.0035532, 0.0040718
4: -0.0084515, -0.0043078, -0.0082945, -0.0039129, -0.0035752, 0.0031199
5: 0.0105370, 0.0121065, 0.0105964, 0.0122561, -0.0013542, 0.0011817
6: 0.0007936, 0.0067830, 0.0002228, 0.0065560, -0.0045095, 0.0051676
7: 0.9786146, 0.9828056, 0.9782152, 0.9826468, -0.0031555, 0.0036161
8: -0.0094928, -0.0049993, -0.0099210, -0.0051696, -0.0033832, 0.0038770
9: -0.0016973, 0.0012709, -0.0015848, 0.0015538, -0.0025610, 0.0022348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017420
time: 1.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017420
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020388, -0.0003928, -0.0020203, -0.0002425, -0.0014120, 0.0012483
1: -0.0094842, -0.0053075, -0.0094373, -0.0049259, -0.0035830, 0.0031677
2: 0.0291460, 0.0317373, 0.0291751, 0.0319740, -0.0022229, 0.0019653
3: 0.0003010, 0.0051396, -0.0001410, 0.0050853, -0.0036696, 0.0041508
4: -0.0085401, -0.0042916, -0.0084924, -0.0039035, -0.0036445, 0.0032221
5: 0.0105034, 0.0121126, 0.0105215, 0.0122596, -0.0013805, 0.0012204
6: 0.0007701, 0.0069109, 0.0002092, 0.0068420, -0.0046572, 0.0052679
7: 0.9785982, 0.9828951, 0.9782056, 0.9828470, -0.0032589, 0.0036862
8: -0.0095104, -0.0049033, -0.0099312, -0.0049550, -0.0034941, 0.0039522
9: -0.0017607, 0.0012825, -0.0017266, 0.0015605, -0.0026106, 0.0023080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017553
time: 1.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017553
time: 1.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019267, -0.0002625, -0.0019747, -0.0004682, -0.0011481, 0.0013727
1: -0.0091999, -0.0049768, -0.0093218, -0.0054988, -0.0029134, 0.0034834
2: 0.0293224, 0.0319424, 0.0292468, 0.0316186, -0.0018075, 0.0021611
3: -0.0000821, 0.0048102, 0.0005226, 0.0049514, -0.0040353, 0.0033750
4: -0.0082509, -0.0039552, -0.0083748, -0.0044862, -0.0029634, 0.0035432
5: 0.0106130, 0.0122400, 0.0105660, 0.0120389, -0.0011225, 0.0013421
6: 0.0002840, 0.0064929, 0.0010514, 0.0066721, -0.0051213, 0.0042833
7: 0.9782580, 0.9826027, 0.9787951, 0.9827281, -0.0035837, 0.0029973
8: -0.0098751, -0.0052169, -0.0092993, -0.0050825, -0.0038422, 0.0032135
9: -0.0015536, 0.0015235, -0.0016423, 0.0011431, -0.0021227, 0.0025380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017153, upper bound: 0.0016522
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0016522
time: 1.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019263, -0.0002713, -0.0020259, -0.0004484, -0.0011805, 0.0014533
1: -0.0091987, -0.0049992, -0.0094517, -0.0054485, -0.0029957, 0.0036878
2: 0.0293231, 0.0319285, 0.0291662, 0.0316497, -0.0018586, 0.0022879
3: -0.0000561, 0.0048089, 0.0004644, 0.0051019, -0.0042722, 0.0034704
4: -0.0082497, -0.0039780, -0.0085070, -0.0044351, -0.0030472, 0.0037511
5: 0.0106134, 0.0122314, 0.0105160, 0.0120583, -0.0011542, 0.0014208
6: 0.0003169, 0.0064912, 0.0009775, 0.0068631, -0.0054219, 0.0044044
7: 0.9782810, 0.9826015, 0.9787433, 0.9828618, -0.0037940, 0.0030820
8: -0.0098504, -0.0052182, -0.0093548, -0.0049391, -0.0040678, 0.0033044
9: -0.0015527, 0.0015072, -0.0017370, 0.0011798, -0.0021827, 0.0026870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017153, upper bound: 0.0016547
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0016547
time: 1.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020050, -0.0002593, -0.0020113, -0.0004621, -0.0011834, 0.0014016
1: -0.0093985, -0.0049686, -0.0094145, -0.0054831, -0.0030030, 0.0035569
2: 0.0291991, 0.0319475, 0.0291892, 0.0316282, -0.0018631, 0.0022067
3: -0.0000916, 0.0050403, 0.0005045, 0.0050588, -0.0041205, 0.0034788
4: -0.0084529, -0.0039469, -0.0084691, -0.0044703, -0.0030545, 0.0036179
5: 0.0105364, 0.0122432, 0.0105303, 0.0120449, -0.0011570, 0.0013704
6: 0.0002719, 0.0067850, 0.0010285, 0.0068084, -0.0052294, 0.0044150
7: 0.9782495, 0.9828070, 0.9787790, 0.9828236, -0.0036593, 0.0030894
8: -0.0098842, -0.0049978, -0.0093166, -0.0049802, -0.0039233, 0.0033124
9: -0.0016983, 0.0015295, -0.0017099, 0.0011545, -0.0021880, 0.0025916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017319, upper bound: 0.0017581
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017314, upper bound: 0.0017581
time: 2.11 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020045, -0.0002678, -0.0020626, -0.0004422, -0.0012184, 0.0014844
1: -0.0093974, -0.0049902, -0.0095448, -0.0054327, -0.0030919, 0.0037669
2: 0.0291999, 0.0319341, 0.0291084, 0.0316596, -0.0019183, 0.0023370
3: -0.0000665, 0.0050390, 0.0004461, 0.0052098, -0.0043638, 0.0035819
4: -0.0084517, -0.0039689, -0.0086017, -0.0044190, -0.0031450, 0.0038316
5: 0.0105369, 0.0122349, 0.0104801, 0.0120644, -0.0011913, 0.0014513
6: 0.0003037, 0.0067832, 0.0009543, 0.0070000, -0.0055382, 0.0045459
7: 0.9782719, 0.9828058, 0.9787270, 0.9829575, -0.0038754, 0.0031810
8: -0.0098603, -0.0049991, -0.0093722, -0.0048364, -0.0041550, 0.0034105
9: -0.0016974, 0.0015137, -0.0018049, 0.0011913, -0.0022528, 0.0027446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017319, upper bound: 0.0017579
time: 1.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017314, upper bound: 0.0017579
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019423, -0.0002606, -0.0020115, -0.0004423, -0.0011836, 0.0013724
1: -0.0092395, -0.0049719, -0.0094149, -0.0054329, -0.0030035, 0.0034828
2: 0.0292978, 0.0319455, 0.0291890, 0.0316594, -0.0018634, 0.0021607
3: -0.0000877, 0.0048561, 0.0004464, 0.0050593, -0.0040346, 0.0034794
4: -0.0082912, -0.0039503, -0.0084696, -0.0044192, -0.0030550, 0.0035426
5: 0.0105977, 0.0122419, 0.0105301, 0.0120643, -0.0011572, 0.0013418
6: 0.0002768, 0.0065511, 0.0009546, 0.0068091, -0.0051205, 0.0044158
7: 0.9782529, 0.9826434, 0.9787273, 0.9828240, -0.0035831, 0.0030899
8: -0.0098805, -0.0051732, -0.0093719, -0.0049797, -0.0038416, 0.0033129
9: -0.0015824, 0.0015270, -0.0017102, 0.0011911, -0.0021884, 0.0025376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017431, upper bound: 0.0016522
time: 1.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017390, upper bound: 0.0016522
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019418, -0.0002694, -0.0020571, -0.0004269, -0.0012143, 0.0014548
1: -0.0092382, -0.0049942, -0.0095306, -0.0053940, -0.0030814, 0.0036917
2: 0.0292986, 0.0319316, 0.0291172, 0.0316836, -0.0019117, 0.0022903
3: -0.0000619, 0.0048546, 0.0004013, 0.0051934, -0.0042766, 0.0035697
4: -0.0082899, -0.0039729, -0.0085873, -0.0043796, -0.0031343, 0.0037550
5: 0.0105982, 0.0122333, 0.0104855, 0.0120793, -0.0011872, 0.0014223
6: 0.0003096, 0.0065493, 0.0008974, 0.0069792, -0.0054276, 0.0045304
7: 0.9782759, 0.9826422, 0.9786872, 0.9829430, -0.0037979, 0.0031701
8: -0.0098559, -0.0051746, -0.0094149, -0.0048521, -0.0040720, 0.0033989
9: -0.0015815, 0.0015108, -0.0017945, 0.0012195, -0.0022452, 0.0026898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017431, upper bound: 0.0016546
time: 1.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017390, upper bound: 0.0016546
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020190, -0.0002570, -0.0020463, -0.0004360, -0.0012220, 0.0013987
1: -0.0094340, -0.0049629, -0.0095035, -0.0054169, -0.0031010, 0.0035494
2: 0.0291771, 0.0319510, 0.0291340, 0.0316693, -0.0019239, 0.0022021
3: -0.0000982, 0.0050814, 0.0004278, 0.0051619, -0.0041118, 0.0035924
4: -0.0084890, -0.0039411, -0.0085596, -0.0044029, -0.0031543, 0.0036104
5: 0.0105228, 0.0122454, 0.0104960, 0.0120705, -0.0011947, 0.0013675
6: 0.0002635, 0.0068371, 0.0009311, 0.0069392, -0.0052184, 0.0045592
7: 0.9782437, 0.9828436, 0.9787108, 0.9829149, -0.0036516, 0.0031903
8: -0.0098904, -0.0049586, -0.0093896, -0.0048820, -0.0039151, 0.0034205
9: -0.0017241, 0.0015336, -0.0017747, 0.0012028, -0.0022594, 0.0025862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017619, upper bound: 0.0017581
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017581
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020185, -0.0002656, -0.0020935, -0.0004206, -0.0012569, 0.0014819
1: -0.0094328, -0.0049845, -0.0096232, -0.0053778, -0.0031896, 0.0037605
2: 0.0291779, 0.0319376, 0.0290597, 0.0316936, -0.0019789, 0.0023330
3: -0.0000731, 0.0050800, 0.0003825, 0.0053006, -0.0043563, 0.0036950
4: -0.0084878, -0.0039631, -0.0086815, -0.0043632, -0.0032444, 0.0038250
5: 0.0105232, 0.0122371, 0.0104499, 0.0120855, -0.0012289, 0.0014488
6: 0.0002953, 0.0068353, 0.0008736, 0.0071153, -0.0055287, 0.0046895
7: 0.9782659, 0.9828423, 0.9786706, 0.9830382, -0.0038687, 0.0032815
8: -0.0098666, -0.0049600, -0.0094327, -0.0047499, -0.0041479, 0.0035182
9: -0.0017232, 0.0015178, -0.0018620, 0.0012313, -0.0023240, 0.0027399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017619, upper bound: 0.0017579
time: 1.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017579
time: 2.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.20 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017243, upper bound: 0.0017573
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017295, upper bound: 0.0017573
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017857, upper bound: 0.0017591
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017888, upper bound: 0.0017591
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017244, upper bound: 0.0017865
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017295, upper bound: 0.0017865
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017857, upper bound: 0.0017888
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017888, upper bound: 0.0017888
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017117
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017117
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017217
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017217
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017420
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017420
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017553
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017553
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017153, upper bound: 0.0016522
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0016522
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017153, upper bound: 0.0016547
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0016547
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017319, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017314, upper bound: 0.0017581
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017319, upper bound: 0.0017579
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017314, upper bound: 0.0017579
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017431, upper bound: 0.0016522
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017390, upper bound: 0.0016522
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017431, upper bound: 0.0016546
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017390, upper bound: 0.0016546
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017619, upper bound: 0.0017581
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017581
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017619, upper bound: 0.0017579
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.20
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017579

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019628, -0.0004677, -0.0019536, -0.0004131, -0.0012043, 0.0011261
1: -0.0092916, -0.0054975, -0.0092681, -0.0053589, -0.0030562, 0.0028577
2: 0.0292655, 0.0316193, 0.0292801, 0.0317053, -0.0018961, 0.0017729
3: 0.0005212, 0.0049164, 0.0003607, 0.0048892, -0.0033105, 0.0035404
4: -0.0083441, -0.0044849, -0.0083202, -0.0043440, -0.0031086, 0.0029068
5: 0.0105777, 0.0120394, 0.0105867, 0.0120928, -0.0011775, 0.0011010
6: 0.0010496, 0.0066277, 0.0008458, 0.0065932, -0.0042015, 0.0044933
7: 0.9787937, 0.9826970, 0.9786512, 0.9826729, -0.0029400, 0.0031442
8: -0.0093007, -0.0051158, -0.0094536, -0.0051417, -0.0031521, 0.0033710
9: -0.0016203, 0.0011440, -0.0016032, 0.0012450, -0.0022268, 0.0020822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017121
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017138
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020137, -0.0004478, -0.0019531, -0.0004207, -0.0012979, 0.0011581
1: -0.0094207, -0.0054470, -0.0092669, -0.0053781, -0.0032935, 0.0029388
2: 0.0291854, 0.0316507, 0.0292808, 0.0316934, -0.0020433, 0.0018233
3: 0.0004626, 0.0050660, 0.0003829, 0.0048879, -0.0034045, 0.0038154
4: -0.0084755, -0.0044335, -0.0083190, -0.0043635, -0.0033501, 0.0029893
5: 0.0105279, 0.0120589, 0.0105872, 0.0120854, -0.0012689, 0.0011323
6: 0.0009752, 0.0068176, 0.0008741, 0.0065915, -0.0043207, 0.0048423
7: 0.9787417, 0.9828299, 0.9786709, 0.9826716, -0.0030234, 0.0033884
8: -0.0093565, -0.0049733, -0.0094324, -0.0051430, -0.0032416, 0.0036329
9: -0.0017145, 0.0011809, -0.0016024, 0.0012310, -0.0023997, 0.0021413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017121
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017138
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019976, -0.0004615, -0.0020269, -0.0004091, -0.0012335, 0.0011623
1: -0.0093799, -0.0054818, -0.0094542, -0.0053487, -0.0031303, 0.0029496
2: 0.0292107, 0.0316291, 0.0291646, 0.0317116, -0.0019420, 0.0018299
3: 0.0005029, 0.0050187, 0.0003488, 0.0051049, -0.0034170, 0.0036263
4: -0.0084339, -0.0044689, -0.0085096, -0.0043336, -0.0031840, 0.0030002
5: 0.0105436, 0.0120455, 0.0105150, 0.0120967, -0.0012060, 0.0011364
6: 0.0010264, 0.0067575, 0.0008309, 0.0068669, -0.0043366, 0.0046022
7: 0.9787775, 0.9827878, 0.9786407, 0.9828644, -0.0030345, 0.0032204
8: -0.0093181, -0.0050183, -0.0094648, -0.0049363, -0.0032535, 0.0034528
9: -0.0016847, 0.0011555, -0.0017389, 0.0012524, -0.0022808, 0.0021491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0017133
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0017152
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020507, -0.0004416, -0.0020265, -0.0004171, -0.0013272, 0.0011967
1: -0.0095146, -0.0054311, -0.0094531, -0.0053691, -0.0033679, 0.0030369
2: 0.0291271, 0.0316605, 0.0291653, 0.0316990, -0.0020895, 0.0018841
3: 0.0004443, 0.0051748, 0.0003724, 0.0051036, -0.0035181, 0.0039015
4: -0.0085710, -0.0044174, -0.0085084, -0.0043543, -0.0034257, 0.0030890
5: 0.0104917, 0.0120650, 0.0105154, 0.0120889, -0.0012976, 0.0011700
6: 0.0009519, 0.0069556, 0.0008607, 0.0068652, -0.0044649, 0.0049516
7: 0.9787254, 0.9829265, 0.9786615, 0.9828632, -0.0031243, 0.0034649
8: -0.0093740, -0.0048698, -0.0094424, -0.0049376, -0.0033498, 0.0037149
9: -0.0017828, 0.0011924, -0.0017381, 0.0012376, -0.0024539, 0.0022127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017428, upper bound: 0.0017133
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017428, upper bound: 0.0017152
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020011, -0.0004418, -0.0019694, -0.0004113, -0.0012012, 0.0011618
1: -0.0093885, -0.0054317, -0.0093082, -0.0053544, -0.0030481, 0.0029483
2: 0.0292053, 0.0316602, 0.0292552, 0.0317081, -0.0018911, 0.0018291
3: 0.0004450, 0.0050288, 0.0003554, 0.0049356, -0.0034155, 0.0035311
4: -0.0084428, -0.0044180, -0.0083610, -0.0043393, -0.0031004, 0.0029989
5: 0.0105403, 0.0120648, 0.0105713, 0.0120946, -0.0011744, 0.0011359
6: 0.0009528, 0.0067703, 0.0008391, 0.0066521, -0.0043347, 0.0044814
7: 0.9787260, 0.9827968, 0.9786465, 0.9827141, -0.0030332, 0.0031359
8: -0.0093733, -0.0050088, -0.0094586, -0.0050975, -0.0032521, 0.0033621
9: -0.0016910, 0.0011920, -0.0016324, 0.0012483, -0.0022209, 0.0021482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017444
time: 2.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017410
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020466, -0.0004263, -0.0019689, -0.0004189, -0.0012965, 0.0011921
1: -0.0095040, -0.0053924, -0.0093069, -0.0053736, -0.0032901, 0.0030252
2: 0.0291337, 0.0316846, 0.0292560, 0.0316962, -0.0020412, 0.0018769
3: 0.0003994, 0.0051625, 0.0003776, 0.0049342, -0.0035046, 0.0038114
4: -0.0085602, -0.0043780, -0.0083597, -0.0043589, -0.0033465, 0.0030772
5: 0.0104958, 0.0120799, 0.0105717, 0.0120872, -0.0012676, 0.0011655
6: 0.0008950, 0.0069400, 0.0008674, 0.0066503, -0.0044478, 0.0048371
7: 0.9786856, 0.9829155, 0.9786662, 0.9827127, -0.0031123, 0.0033848
8: -0.0094167, -0.0048814, -0.0094374, -0.0050988, -0.0033369, 0.0036290
9: -0.0017751, 0.0012207, -0.0016315, 0.0012344, -0.0023972, 0.0022042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017444
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017410
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020354, -0.0004354, -0.0020422, -0.0004069, -0.0012331, 0.0012011
1: -0.0094756, -0.0054156, -0.0094930, -0.0053431, -0.0031293, 0.0030480
2: 0.0291513, 0.0316702, 0.0291405, 0.0317152, -0.0019414, 0.0018910
3: 0.0004263, 0.0051296, 0.0003423, 0.0051498, -0.0035310, 0.0036251
4: -0.0085313, -0.0044016, -0.0085490, -0.0043278, -0.0031830, 0.0031004
5: 0.0105068, 0.0120710, 0.0105000, 0.0120989, -0.0012056, 0.0011743
6: 0.0009291, 0.0068983, 0.0008225, 0.0069239, -0.0044813, 0.0046007
7: 0.9787094, 0.9828863, 0.9786348, 0.9829043, -0.0031358, 0.0032194
8: -0.0093911, -0.0049128, -0.0094711, -0.0048936, -0.0033621, 0.0034517
9: -0.0017544, 0.0012037, -0.0017671, 0.0012566, -0.0022800, 0.0022208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017406, upper bound: 0.0017463
time: 1.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017406, upper bound: 0.0017429
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020832, -0.0004199, -0.0020418, -0.0004149, -0.0013282, 0.0012354
1: -0.0095969, -0.0053762, -0.0094918, -0.0053634, -0.0033705, 0.0031351
2: 0.0290761, 0.0316946, 0.0291413, 0.0317025, -0.0020911, 0.0019450
3: 0.0003806, 0.0052701, 0.0003658, 0.0051484, -0.0036319, 0.0039046
4: -0.0086547, -0.0043615, -0.0085478, -0.0043485, -0.0034284, 0.0031889
5: 0.0104600, 0.0120862, 0.0105005, 0.0120911, -0.0012986, 0.0012079
6: 0.0008712, 0.0070766, 0.0008524, 0.0069221, -0.0046093, 0.0049554
7: 0.9786689, 0.9830111, 0.9786557, 0.9829031, -0.0032254, 0.0034675
8: -0.0094346, -0.0047790, -0.0094486, -0.0048949, -0.0034581, 0.0037178
9: -0.0018428, 0.0012325, -0.0017663, 0.0012418, -0.0024558, 0.0022843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017429, upper bound: 0.0017463
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017429, upper bound: 0.0017429
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019628, -0.0004677, -0.0019267, -0.0002625, -0.0013598, 0.0011078
1: -0.0092916, -0.0054975, -0.0091999, -0.0049768, -0.0034506, 0.0028111
2: 0.0292655, 0.0316193, 0.0293224, 0.0319424, -0.0021407, 0.0017440
3: 0.0005212, 0.0049164, -0.0000821, 0.0048102, -0.0032566, 0.0039973
4: -0.0083441, -0.0044849, -0.0082509, -0.0039552, -0.0035098, 0.0028594
5: 0.0105777, 0.0120394, 0.0106130, 0.0122400, -0.0013294, 0.0010831
6: 0.0010496, 0.0066277, 0.0002840, 0.0064929, -0.0041330, 0.0050731
7: 0.9787937, 0.9826970, 0.9782580, 0.9826027, -0.0028921, 0.0035499
8: -0.0093007, -0.0051158, -0.0098751, -0.0052169, -0.0031008, 0.0038061
9: -0.0016203, 0.0011440, -0.0015536, 0.0015235, -0.0025141, 0.0020482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016637
time: 1.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016631
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020137, -0.0004478, -0.0019263, -0.0002713, -0.0014474, 0.0011397
1: -0.0094207, -0.0054470, -0.0091987, -0.0049992, -0.0036730, 0.0028922
2: 0.0291854, 0.0316507, 0.0293231, 0.0319285, -0.0022787, 0.0017943
3: 0.0004626, 0.0050660, -0.0000561, 0.0048089, -0.0033505, 0.0042550
4: -0.0084755, -0.0044335, -0.0082497, -0.0039780, -0.0037360, 0.0029419
5: 0.0105279, 0.0120589, 0.0106134, 0.0122314, -0.0014151, 0.0011143
6: 0.0009752, 0.0068176, 0.0003169, 0.0064912, -0.0042522, 0.0054001
7: 0.9787417, 0.9828299, 0.9782810, 0.9826015, -0.0029755, 0.0037787
8: -0.0093565, -0.0049733, -0.0098504, -0.0052182, -0.0031902, 0.0040514
9: -0.0017145, 0.0011809, -0.0015527, 0.0015072, -0.0026762, 0.0021073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016637
time: 1.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016631
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019976, -0.0004615, -0.0020050, -0.0002593, -0.0013890, 0.0011450
1: -0.0093799, -0.0054818, -0.0093985, -0.0049686, -0.0035247, 0.0029056
2: 0.0292107, 0.0316291, 0.0291991, 0.0319475, -0.0021867, 0.0018027
3: 0.0005029, 0.0050187, -0.0000916, 0.0050403, -0.0033661, 0.0040832
4: -0.0084339, -0.0044689, -0.0084529, -0.0039469, -0.0035852, 0.0029555
5: 0.0105436, 0.0120455, 0.0105364, 0.0122432, -0.0013580, 0.0011195
6: 0.0010264, 0.0067575, 0.0002719, 0.0067850, -0.0042720, 0.0051821
7: 0.9787775, 0.9827878, 0.9782495, 0.9828070, -0.0029893, 0.0036262
8: -0.0093181, -0.0050183, -0.0098842, -0.0049978, -0.0032050, 0.0038878
9: -0.0016847, 0.0011555, -0.0016983, 0.0015295, -0.0025681, 0.0021171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0016729
time: 2.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0016732
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020507, -0.0004416, -0.0020045, -0.0002678, -0.0014783, 0.0011794
1: -0.0095146, -0.0054311, -0.0093974, -0.0049902, -0.0037515, 0.0029929
2: 0.0291271, 0.0316605, 0.0291999, 0.0319341, -0.0023274, 0.0018568
3: 0.0004443, 0.0051748, -0.0000665, 0.0050390, -0.0034671, 0.0043459
4: -0.0085710, -0.0044174, -0.0084517, -0.0039689, -0.0038159, 0.0030443
5: 0.0104917, 0.0120650, 0.0105369, 0.0122349, -0.0014454, 0.0011531
6: 0.0009519, 0.0069556, 0.0003037, 0.0067832, -0.0044002, 0.0055155
7: 0.9787254, 0.9829265, 0.9782719, 0.9828058, -0.0030790, 0.0038595
8: -0.0093740, -0.0048698, -0.0098603, -0.0049991, -0.0033012, 0.0041380
9: -0.0017828, 0.0011924, -0.0016974, 0.0015137, -0.0027334, 0.0021806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0016729
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0016732
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020011, -0.0004418, -0.0019423, -0.0002606, -0.0013597, 0.0011402
1: -0.0093885, -0.0054317, -0.0092395, -0.0049719, -0.0034504, 0.0028933
2: 0.0292053, 0.0316602, 0.0292978, 0.0319455, -0.0021406, 0.0017950
3: 0.0004450, 0.0050288, -0.0000877, 0.0048561, -0.0033518, 0.0039971
4: -0.0084428, -0.0044180, -0.0082912, -0.0039503, -0.0035096, 0.0029430
5: 0.0105403, 0.0120648, 0.0105977, 0.0122419, -0.0013294, 0.0011147
6: 0.0009528, 0.0067703, 0.0002768, 0.0065511, -0.0042538, 0.0050729
7: 0.9787260, 0.9827968, 0.9782529, 0.9826434, -0.0029766, 0.0035498
8: -0.0093733, -0.0050088, -0.0098805, -0.0051732, -0.0031914, 0.0038059
9: -0.0016910, 0.0011920, -0.0015824, 0.0015270, -0.0025140, 0.0021081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016965
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016927
time: 1.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020466, -0.0004263, -0.0019418, -0.0002694, -0.0014491, 0.0011704
1: -0.0095040, -0.0053924, -0.0092382, -0.0049942, -0.0036772, 0.0029702
2: 0.0291337, 0.0316846, 0.0292986, 0.0319316, -0.0022814, 0.0018427
3: 0.0003994, 0.0051625, -0.0000619, 0.0048546, -0.0034408, 0.0042599
4: -0.0085602, -0.0043780, -0.0082899, -0.0039729, -0.0037403, 0.0030212
5: 0.0104958, 0.0120799, 0.0105982, 0.0122333, -0.0014167, 0.0011443
6: 0.0008950, 0.0069400, 0.0003096, 0.0065493, -0.0043668, 0.0054063
7: 0.9786856, 0.9829155, 0.9782759, 0.9826422, -0.0030557, 0.0037831
8: -0.0094167, -0.0048814, -0.0098559, -0.0051746, -0.0032762, 0.0040561
9: -0.0017751, 0.0012207, -0.0015815, 0.0015108, -0.0026793, 0.0021641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016965
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016927
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020354, -0.0004354, -0.0020190, -0.0002570, -0.0013861, 0.0011813
1: -0.0094756, -0.0054156, -0.0094340, -0.0049629, -0.0035175, 0.0029978
2: 0.0291513, 0.0316702, 0.0291771, 0.0319510, -0.0021823, 0.0018599
3: 0.0004263, 0.0051296, -0.0000982, 0.0050814, -0.0034728, 0.0040748
4: -0.0085313, -0.0044016, -0.0084890, -0.0039411, -0.0035779, 0.0030493
5: 0.0105068, 0.0120710, 0.0105228, 0.0122454, -0.0013552, 0.0011550
6: 0.0009291, 0.0068983, 0.0002635, 0.0068371, -0.0044075, 0.0051715
7: 0.9787094, 0.9828863, 0.9782437, 0.9828436, -0.0030841, 0.0036188
8: -0.0093911, -0.0049128, -0.0098904, -0.0049586, -0.0033067, 0.0038799
9: -0.0017544, 0.0012037, -0.0017241, 0.0015336, -0.0025629, 0.0021843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017091
time: 1.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017049
time: 1.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020832, -0.0004199, -0.0020185, -0.0002656, -0.0014759, 0.0012156
1: -0.0095969, -0.0053762, -0.0094328, -0.0049845, -0.0037453, 0.0030849
2: 0.0290761, 0.0316946, 0.0291779, 0.0319376, -0.0023236, 0.0019139
3: 0.0003806, 0.0052701, -0.0000731, 0.0050800, -0.0035737, 0.0043388
4: -0.0086547, -0.0043615, -0.0084878, -0.0039631, -0.0038096, 0.0031378
5: 0.0104600, 0.0120862, 0.0105232, 0.0122371, -0.0014430, 0.0011885
6: 0.0008712, 0.0070766, 0.0002953, 0.0068353, -0.0045355, 0.0055065
7: 0.9786689, 0.9830111, 0.9782659, 0.9828423, -0.0031737, 0.0038532
8: -0.0094346, -0.0047790, -0.0098666, -0.0049600, -0.0034027, 0.0041312
9: -0.0018428, 0.0012325, -0.0017232, 0.0015178, -0.0027289, 0.0022477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017090
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017049
time: 2.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019240, -0.0003026, -0.0019738, -0.0004846, -0.0011249, 0.0013290
1: -0.0091930, -0.0050785, -0.0093193, -0.0055402, -0.0028546, 0.0033726
2: 0.0293266, 0.0318793, 0.0292483, 0.0315929, -0.0017710, 0.0020924
3: 0.0000358, 0.0048023, 0.0005706, 0.0049486, -0.0039070, 0.0033069
4: -0.0082439, -0.0040588, -0.0083723, -0.0045283, -0.0029036, 0.0034305
5: 0.0106156, 0.0122008, 0.0105670, 0.0120230, -0.0010998, 0.0012994
6: 0.0004336, 0.0064828, 0.0011123, 0.0066685, -0.0049584, 0.0041969
7: 0.9783627, 0.9825957, 0.9788377, 0.9827256, -0.0034697, 0.0029368
8: -0.0097628, -0.0052244, -0.0092536, -0.0050851, -0.0037200, 0.0031487
9: -0.0015486, 0.0014493, -0.0016406, 0.0011130, -0.0020799, 0.0024573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
time: 2.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
time: 2.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0019350, -0.0003188, -0.0019733, -0.0004956, -0.0011484, 0.0013316
1: -0.0092209, -0.0051196, -0.0093180, -0.0055683, -0.0029143, 0.0033792
2: 0.0293093, 0.0318538, 0.0292491, 0.0315754, -0.0018080, 0.0020965
3: 0.0000834, 0.0048346, 0.0006032, 0.0049471, -0.0039147, 0.0033761
4: -0.0082722, -0.0041006, -0.0083710, -0.0045569, -0.0029643, 0.0034372
5: 0.0106049, 0.0121850, 0.0105675, 0.0120121, -0.0011228, 0.0013019
6: 0.0004940, 0.0065238, 0.0011537, 0.0066666, -0.0049682, 0.0042847
7: 0.9784049, 0.9826243, 0.9788665, 0.9827242, -0.0034765, 0.0029982
8: -0.0097175, -0.0051937, -0.0092226, -0.0050866, -0.0037274, 0.0032145
9: -0.0015689, 0.0014194, -0.0016396, 0.0010925, -0.0021234, 0.0024621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
time: 1.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
time: 2.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019236, -0.0003116, -0.0020250, -0.0004646, -0.0011570, 0.0014094
1: -0.0091919, -0.0051014, -0.0094492, -0.0054896, -0.0029360, 0.0035765
2: 0.0293274, 0.0318651, 0.0291677, 0.0316243, -0.0018215, 0.0022189
3: 0.0000623, 0.0048009, 0.0005120, 0.0050991, -0.0041433, 0.0034013
4: -0.0082427, -0.0040820, -0.0085045, -0.0044768, -0.0029864, 0.0036379
5: 0.0106161, 0.0121920, 0.0105169, 0.0120425, -0.0011312, 0.0013780
6: 0.0004672, 0.0064811, 0.0010379, 0.0068595, -0.0052583, 0.0043166
7: 0.9783862, 0.9825944, 0.9787855, 0.9828592, -0.0036795, 0.0030206
8: -0.0097376, -0.0052257, -0.0093095, -0.0049419, -0.0039450, 0.0032385
9: -0.0015477, 0.0014327, -0.0017352, 0.0011498, -0.0021392, 0.0026059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
time: 2.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019345, -0.0003277, -0.0020245, -0.0004757, -0.0011809, 0.0014121
1: -0.0092197, -0.0051420, -0.0094479, -0.0055178, -0.0029966, 0.0035835
2: 0.0293101, 0.0318399, 0.0291685, 0.0316067, -0.0018591, 0.0022232
3: 0.0001094, 0.0048332, 0.0005447, 0.0050976, -0.0041513, 0.0034715
4: -0.0082710, -0.0041233, -0.0085032, -0.0045056, -0.0030481, 0.0036450
5: 0.0106053, 0.0121764, 0.0105174, 0.0120316, -0.0011545, 0.0013806
6: 0.0005270, 0.0065220, 0.0010795, 0.0068576, -0.0052685, 0.0044057
7: 0.9784281, 0.9826230, 0.9788147, 0.9828579, -0.0036866, 0.0030829
8: -0.0096928, -0.0051950, -0.0092783, -0.0049433, -0.0039527, 0.0033054
9: -0.0015680, 0.0014030, -0.0017343, 0.0011292, -0.0021834, 0.0026110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016546
time: 2.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
time: 2.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020024, -0.0003017, -0.0020103, -0.0004784, -0.0011614, 0.0013559
1: -0.0093920, -0.0050762, -0.0094121, -0.0055246, -0.0029471, 0.0034408
2: 0.0292032, 0.0318807, 0.0291907, 0.0316025, -0.0018284, 0.0021347
3: 0.0000331, 0.0050328, 0.0005526, 0.0050560, -0.0039860, 0.0034141
4: -0.0084463, -0.0040564, -0.0084667, -0.0045125, -0.0029977, 0.0034998
5: 0.0105390, 0.0122017, 0.0105312, 0.0120290, -0.0011354, 0.0013256
6: 0.0004301, 0.0067754, 0.0010894, 0.0068049, -0.0050587, 0.0043329
7: 0.9783603, 0.9828004, 0.9788216, 0.9828209, -0.0035398, 0.0030320
8: -0.0097654, -0.0050050, -0.0092708, -0.0049828, -0.0037953, 0.0032507
9: -0.0016935, 0.0014510, -0.0017082, 0.0011243, -0.0021473, 0.0025070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017389
time: 1.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017581
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020143, -0.0003166, -0.0020098, -0.0004895, -0.0011848, 0.0013580
1: -0.0094222, -0.0051141, -0.0094108, -0.0055528, -0.0030066, 0.0034462
2: 0.0291845, 0.0318572, 0.0291915, 0.0315851, -0.0018653, 0.0021380
3: 0.0000770, 0.0050677, 0.0005852, 0.0050545, -0.0039922, 0.0034830
4: -0.0084769, -0.0040949, -0.0084654, -0.0045411, -0.0030582, 0.0035053
5: 0.0105273, 0.0121871, 0.0105317, 0.0120181, -0.0011584, 0.0013277
6: 0.0004859, 0.0068197, 0.0011308, 0.0068030, -0.0050666, 0.0044204
7: 0.9783993, 0.9828314, 0.9788506, 0.9828197, -0.0035454, 0.0030932
8: -0.0097236, -0.0049717, -0.0092397, -0.0049843, -0.0038012, 0.0033164
9: -0.0017155, 0.0014234, -0.0017072, 0.0011038, -0.0021906, 0.0025109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017389
time: 1.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017581
time: 1.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020020, -0.0003109, -0.0020617, -0.0004584, -0.0011954, 0.0014384
1: -0.0093908, -0.0050996, -0.0095424, -0.0054739, -0.0030336, 0.0036503
2: 0.0292039, 0.0318662, 0.0291099, 0.0316340, -0.0018821, 0.0022646
3: 0.0000603, 0.0050314, 0.0004938, 0.0052070, -0.0042286, 0.0035143
4: -0.0084451, -0.0040802, -0.0085992, -0.0044609, -0.0030857, 0.0037129
5: 0.0105394, 0.0121927, 0.0104810, 0.0120485, -0.0011688, 0.0014064
6: 0.0004646, 0.0067736, 0.0010148, 0.0069964, -0.0053667, 0.0044601
7: 0.9783844, 0.9827991, 0.9787694, 0.9829551, -0.0037554, 0.0031209
8: -0.0097396, -0.0050063, -0.0093268, -0.0048391, -0.0040263, 0.0033461
9: -0.0016927, 0.0014340, -0.0018031, 0.0011613, -0.0022103, 0.0026596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017390
time: 1.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017579
time: 2.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020138, -0.0003255, -0.0020612, -0.0004696, -0.0012202, 0.0014406
1: -0.0094210, -0.0051365, -0.0095411, -0.0055021, -0.0030965, 0.0036558
2: 0.0291852, 0.0318433, 0.0291107, 0.0316165, -0.0019211, 0.0022681
3: 0.0001029, 0.0050663, 0.0005266, 0.0052055, -0.0042350, 0.0035872
4: -0.0084758, -0.0041177, -0.0085979, -0.0044896, -0.0031497, 0.0037185
5: 0.0105278, 0.0121785, 0.0104815, 0.0120376, -0.0011930, 0.0014085
6: 0.0005188, 0.0068180, 0.0010564, 0.0069946, -0.0053748, 0.0045526
7: 0.9784222, 0.9828302, 0.9787985, 0.9829537, -0.0037610, 0.0031857
8: -0.0096990, -0.0049730, -0.0092956, -0.0048405, -0.0040324, 0.0034156
9: -0.0017146, 0.0014071, -0.0018022, 0.0011407, -0.0022562, 0.0026636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017390
time: 1.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017579
time: 2.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019396, -0.0003007, -0.0020105, -0.0004581, -0.0011616, 0.0013257
1: -0.0092327, -0.0050736, -0.0094125, -0.0054730, -0.0029479, 0.0033642
2: 0.0293020, 0.0318823, 0.0291905, 0.0316345, -0.0018289, 0.0020872
3: 0.0000301, 0.0048482, 0.0004928, 0.0050566, -0.0038973, 0.0034149
4: -0.0082842, -0.0040537, -0.0084672, -0.0044600, -0.0029985, 0.0034220
5: 0.0106003, 0.0122027, 0.0105311, 0.0120488, -0.0011357, 0.0012961
6: 0.0004264, 0.0065411, 0.0010136, 0.0068055, -0.0049461, 0.0043340
7: 0.9783576, 0.9826364, 0.9787685, 0.9828215, -0.0034611, 0.0030327
8: -0.0097683, -0.0051807, -0.0093277, -0.0049823, -0.0037108, 0.0032516
9: -0.0015774, 0.0014529, -0.0017085, 0.0011619, -0.0021478, 0.0024512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016522
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016522
time: 2.17 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0019492, -0.0003170, -0.0020100, -0.0004698, -0.0011793, 0.0013277
1: -0.0092569, -0.0051149, -0.0094112, -0.0055028, -0.0029927, 0.0033692
2: 0.0292870, 0.0318567, 0.0291913, 0.0316161, -0.0018567, 0.0020903
3: 0.0000779, 0.0048763, 0.0005273, 0.0050550, -0.0039030, 0.0034669
4: -0.0083089, -0.0040957, -0.0084658, -0.0044903, -0.0030441, 0.0034270
5: 0.0105910, 0.0121868, 0.0105316, 0.0120374, -0.0011530, 0.0012981
6: 0.0004870, 0.0065768, 0.0010573, 0.0068036, -0.0049534, 0.0043999
7: 0.9784002, 0.9826613, 0.9787991, 0.9828201, -0.0034662, 0.0030789
8: -0.0097227, -0.0051540, -0.0092949, -0.0049838, -0.0037163, 0.0033010
9: -0.0015951, 0.0014228, -0.0017075, 0.0011402, -0.0021805, 0.0024548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016522
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016522
time: 2.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019392, -0.0003097, -0.0020561, -0.0004427, -0.0011919, 0.0014080
1: -0.0092314, -0.0050964, -0.0095282, -0.0054339, -0.0030245, 0.0035729
2: 0.0293028, 0.0318682, 0.0291187, 0.0316588, -0.0018764, 0.0022166
3: 0.0000566, 0.0048468, 0.0004475, 0.0051905, -0.0041390, 0.0035037
4: -0.0082830, -0.0040770, -0.0085848, -0.0044202, -0.0030764, 0.0036342
5: 0.0106008, 0.0121939, 0.0104865, 0.0120639, -0.0011653, 0.0013765
6: 0.0004599, 0.0065393, 0.0009561, 0.0069756, -0.0052529, 0.0044467
7: 0.9783811, 0.9826351, 0.9787283, 0.9829404, -0.0036757, 0.0031116
8: -0.0097431, -0.0051821, -0.0093709, -0.0048548, -0.0039410, 0.0033361
9: -0.0015765, 0.0014363, -0.0017928, 0.0011904, -0.0022037, 0.0026032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016547
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016547
time: 1.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019487, -0.0003258, -0.0020556, -0.0004548, -0.0012105, 0.0014099
1: -0.0092556, -0.0051372, -0.0095269, -0.0054647, -0.0030719, 0.0035778
2: 0.0292878, 0.0318429, 0.0291195, 0.0316397, -0.0019058, 0.0022197
3: 0.0001038, 0.0048748, 0.0004831, 0.0051890, -0.0041447, 0.0035587
4: -0.0083076, -0.0041184, -0.0085835, -0.0044515, -0.0031246, 0.0036392
5: 0.0105915, 0.0121782, 0.0104870, 0.0120521, -0.0011835, 0.0013784
6: 0.0005199, 0.0065749, 0.0010013, 0.0069737, -0.0052602, 0.0045164
7: 0.9784231, 0.9826601, 0.9787599, 0.9829391, -0.0036808, 0.0031604
8: -0.0096981, -0.0051554, -0.0093369, -0.0048562, -0.0039464, 0.0033884
9: -0.0015942, 0.0014066, -0.0017918, 0.0011680, -0.0022382, 0.0026068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016547
time: 2.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016547
time: 2.20 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020164, -0.0002995, -0.0020454, -0.0004518, -0.0012013, 0.0013503
1: -0.0094275, -0.0050707, -0.0095011, -0.0054570, -0.0030484, 0.0034267
2: 0.0291812, 0.0318841, 0.0291355, 0.0316445, -0.0018913, 0.0021259
3: 0.0000268, 0.0050739, 0.0004743, 0.0051592, -0.0039696, 0.0035315
4: -0.0084823, -0.0040508, -0.0085572, -0.0044437, -0.0031008, 0.0034855
5: 0.0105253, 0.0122038, 0.0104969, 0.0120550, -0.0011745, 0.0013202
6: 0.0004221, 0.0068275, 0.0009901, 0.0069358, -0.0050380, 0.0044819
7: 0.9783546, 0.9828368, 0.9787520, 0.9829126, -0.0035253, 0.0031362
8: -0.0097715, -0.0049659, -0.0093454, -0.0048846, -0.0037797, 0.0033625
9: -0.0017194, 0.0014550, -0.0017730, 0.0011735, -0.0022211, 0.0024967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017389
time: 1.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017581
time: 2.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020282, -0.0003145, -0.0020449, -0.0004636, -0.0012177, 0.0013516
1: -0.0094575, -0.0051087, -0.0094998, -0.0054869, -0.0030901, 0.0034300
2: 0.0291626, 0.0318606, 0.0291363, 0.0316259, -0.0019171, 0.0021280
3: 0.0000708, 0.0051086, 0.0005089, 0.0051577, -0.0039735, 0.0035797
4: -0.0085128, -0.0040894, -0.0085559, -0.0044741, -0.0031432, 0.0034889
5: 0.0105137, 0.0121892, 0.0104974, 0.0120435, -0.0011905, 0.0013215
6: 0.0004779, 0.0068716, 0.0010340, 0.0069339, -0.0050429, 0.0045432
7: 0.9783937, 0.9828677, 0.9787828, 0.9829112, -0.0035288, 0.0031791
8: -0.0097296, -0.0049328, -0.0093124, -0.0048861, -0.0037834, 0.0034085
9: -0.0017412, 0.0014273, -0.0017721, 0.0011518, -0.0022515, 0.0024991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017389
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017581
time: 2.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020159, -0.0003088, -0.0020926, -0.0004364, -0.0012353, 0.0014338
1: -0.0094263, -0.0050941, -0.0096208, -0.0054179, -0.0031349, 0.0036384
2: 0.0291819, 0.0318696, 0.0290612, 0.0316687, -0.0019449, 0.0022573
3: 0.0000539, 0.0050725, 0.0004290, 0.0052978, -0.0042149, 0.0036316
4: -0.0084811, -0.0040746, -0.0086790, -0.0044040, -0.0031887, 0.0037008
5: 0.0105258, 0.0121948, 0.0104508, 0.0120701, -0.0012078, 0.0014018
6: 0.0004565, 0.0068257, 0.0009326, 0.0071117, -0.0053492, 0.0046089
7: 0.9783787, 0.9828357, 0.9787118, 0.9830357, -0.0037431, 0.0032251
8: -0.0097457, -0.0049672, -0.0093885, -0.0047526, -0.0040132, 0.0034578
9: -0.0017185, 0.0014380, -0.0018602, 0.0012020, -0.0022841, 0.0026510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017390
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017579
time: 2.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020277, -0.0003233, -0.0020921, -0.0004485, -0.0012533, 0.0014351
1: -0.0094562, -0.0051310, -0.0096195, -0.0054486, -0.0031805, 0.0036418
2: 0.0291633, 0.0318467, 0.0290621, 0.0316497, -0.0019732, 0.0022594
3: 0.0000966, 0.0051072, 0.0004645, 0.0052963, -0.0042188, 0.0036845
4: -0.0085116, -0.0041121, -0.0086777, -0.0044352, -0.0032351, 0.0037043
5: 0.0105142, 0.0121806, 0.0104513, 0.0120583, -0.0012254, 0.0014031
6: 0.0005107, 0.0068698, 0.0009777, 0.0071098, -0.0053542, 0.0046761
7: 0.9784166, 0.9828663, 0.9787434, 0.9830344, -0.0037466, 0.0032721
8: -0.0097050, -0.0049341, -0.0093546, -0.0047541, -0.0040170, 0.0035082
9: -0.0017403, 0.0014111, -0.0018593, 0.0011797, -0.0023174, 0.0026534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017390
time: 1.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017579
time: 1.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.01 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017121
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017138
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017121
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017138
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0017133
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0017152
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017428, upper bound: 0.0017133
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017428, upper bound: 0.0017152
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017444
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017410
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017444
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017410
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017406, upper bound: 0.0017463
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017406, upper bound: 0.0017429
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017429, upper bound: 0.0017463
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017429, upper bound: 0.0017429
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016637
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016631
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016637
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016631
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0016729
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0016732
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0016729
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0016732
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016965
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016927
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016965
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016927
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017091
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017049
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017090
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017049
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016546
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017389
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017389
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017581
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017390
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017579
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017390
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017579
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016522
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016522
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016522
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016522
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016547
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016965, upper bound: 0.0016547
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016547
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016547
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017389
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017581
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017389
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017581
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017390
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017579
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017390
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017579

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019619, -0.0004841, -0.0019507, -0.0004533, -0.0011591, 0.0011031
1: -0.0092891, -0.0055390, -0.0092609, -0.0054608, -0.0029413, 0.0027994
2: 0.0292670, 0.0315936, 0.0292845, 0.0316421, -0.0018248, 0.0017367
3: 0.0005692, 0.0049136, 0.0004786, 0.0048808, -0.0032429, 0.0034073
4: -0.0083416, -0.0045271, -0.0083129, -0.0044476, -0.0029918, 0.0028474
5: 0.0105786, 0.0120234, 0.0105895, 0.0120536, -0.0011332, 0.0010785
6: 0.0011106, 0.0066241, 0.0009956, 0.0065825, -0.0041157, 0.0043243
7: 0.9788364, 0.9826945, 0.9787560, 0.9826654, -0.0028800, 0.0030260
8: -0.0092550, -0.0051185, -0.0093412, -0.0051496, -0.0030878, 0.0032443
9: -0.0016186, 0.0011138, -0.0015980, 0.0011708, -0.0021431, 0.0020397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016602
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017121
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019614, -0.0004951, -0.0019608, -0.0004679, -0.0011620, 0.0011270
1: -0.0092878, -0.0055670, -0.0092863, -0.0054979, -0.0029488, 0.0028599
2: 0.0292678, 0.0315763, 0.0292688, 0.0316191, -0.0018294, 0.0017743
3: 0.0006016, 0.0049121, 0.0005216, 0.0049103, -0.0033131, 0.0034160
4: -0.0083403, -0.0045556, -0.0083387, -0.0044853, -0.0029994, 0.0029090
5: 0.0105791, 0.0120127, 0.0105797, 0.0120393, -0.0011361, 0.0011019
6: 0.0011517, 0.0066222, 0.0010501, 0.0066199, -0.0042047, 0.0043354
7: 0.9788651, 0.9826931, 0.9787942, 0.9826916, -0.0029423, 0.0030337
8: -0.0092241, -0.0051199, -0.0093003, -0.0051216, -0.0031546, 0.0032526
9: -0.0016176, 0.0010934, -0.0016165, 0.0011438, -0.0021485, 0.0020838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016608
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017138
time: 1.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020127, -0.0004640, -0.0019503, -0.0004608, -0.0012523, 0.0011348
1: -0.0094182, -0.0054880, -0.0092597, -0.0054799, -0.0031779, 0.0028796
2: 0.0291869, 0.0316253, 0.0292853, 0.0316303, -0.0019716, 0.0017865
3: 0.0005101, 0.0050631, 0.0005008, 0.0048795, -0.0033359, 0.0036815
4: -0.0084729, -0.0044752, -0.0083117, -0.0044670, -0.0032325, 0.0029291
5: 0.0105289, 0.0120431, 0.0105899, 0.0120462, -0.0012244, 0.0011094
6: 0.0010356, 0.0068139, 0.0010237, 0.0065809, -0.0042337, 0.0046723
7: 0.9787839, 0.9828273, 0.9787757, 0.9826642, -0.0029625, 0.0032694
8: -0.0093112, -0.0049761, -0.0093201, -0.0051509, -0.0031763, 0.0035053
9: -0.0017126, 0.0011510, -0.0015971, 0.0011569, -0.0023155, 0.0020981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016602
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017121
time: 1.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020122, -0.0004751, -0.0019603, -0.0004758, -0.0012553, 0.0011589
1: -0.0094169, -0.0055163, -0.0092851, -0.0055180, -0.0031855, 0.0029410
2: 0.0291877, 0.0316077, 0.0292695, 0.0316067, -0.0019763, 0.0018246
3: 0.0005430, 0.0050616, 0.0005449, 0.0049090, -0.0034070, 0.0036902
4: -0.0084716, -0.0045040, -0.0083376, -0.0045057, -0.0032402, 0.0029915
5: 0.0105294, 0.0120322, 0.0105801, 0.0120315, -0.0012273, 0.0011331
6: 0.0010772, 0.0068119, 0.0010796, 0.0066182, -0.0043239, 0.0046834
7: 0.9788131, 0.9828259, 0.9788148, 0.9826903, -0.0030256, 0.0032772
8: -0.0092800, -0.0049775, -0.0092782, -0.0051229, -0.0032440, 0.0035137
9: -0.0017117, 0.0011304, -0.0016157, 0.0011292, -0.0023210, 0.0021428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016608
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017138
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019967, -0.0004779, -0.0020242, -0.0004524, -0.0011855, 0.0011405
1: -0.0093775, -0.0055233, -0.0094473, -0.0054587, -0.0030083, 0.0028942
2: 0.0292122, 0.0316034, 0.0291689, 0.0316434, -0.0018664, 0.0017956
3: 0.0005510, 0.0050159, 0.0004762, 0.0050968, -0.0033528, 0.0034850
4: -0.0084315, -0.0045111, -0.0085025, -0.0044454, -0.0030599, 0.0029439
5: 0.0105446, 0.0120295, 0.0105177, 0.0120544, -0.0011590, 0.0011151
6: 0.0010875, 0.0067540, 0.0009925, 0.0068566, -0.0042551, 0.0044229
7: 0.9788203, 0.9827853, 0.9787539, 0.9828572, -0.0029775, 0.0030949
8: -0.0092723, -0.0050210, -0.0093435, -0.0049440, -0.0031924, 0.0033182
9: -0.0016829, 0.0011253, -0.0017338, 0.0011723, -0.0021919, 0.0021088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016481
time: 1.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017132
time: 1.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019962, -0.0004889, -0.0020374, -0.0004655, -0.0011878, 0.0011643
1: -0.0093762, -0.0055513, -0.0094807, -0.0054919, -0.0030142, 0.0029546
2: 0.0292130, 0.0315860, 0.0291482, 0.0316228, -0.0018700, 0.0018331
3: 0.0005835, 0.0050144, 0.0005147, 0.0051355, -0.0034228, 0.0034918
4: -0.0084302, -0.0045396, -0.0085365, -0.0044792, -0.0030659, 0.0030054
5: 0.0105451, 0.0120187, 0.0105048, 0.0120416, -0.0011613, 0.0011384
6: 0.0011287, 0.0067521, 0.0010413, 0.0069057, -0.0043440, 0.0044315
7: 0.9788491, 0.9827841, 0.9787880, 0.9828916, -0.0030397, 0.0031009
8: -0.0092414, -0.0050224, -0.0093069, -0.0049072, -0.0032591, 0.0033247
9: -0.0016820, 0.0011048, -0.0017581, 0.0011482, -0.0021962, 0.0021528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016453
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017152
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020497, -0.0004578, -0.0020238, -0.0004604, -0.0012791, 0.0011739
1: -0.0095121, -0.0054723, -0.0094462, -0.0054790, -0.0032460, 0.0029790
2: 0.0291287, 0.0316350, 0.0291696, 0.0316308, -0.0020138, 0.0018482
3: 0.0004919, 0.0051719, 0.0004997, 0.0050955, -0.0034511, 0.0037603
4: -0.0085684, -0.0044592, -0.0085014, -0.0044661, -0.0033017, 0.0030302
5: 0.0104927, 0.0120491, 0.0105181, 0.0120465, -0.0012506, 0.0011478
6: 0.0010124, 0.0069519, 0.0010224, 0.0068550, -0.0043799, 0.0047723
7: 0.9787678, 0.9829238, 0.9787747, 0.9828561, -0.0030648, 0.0033394
8: -0.0093286, -0.0048725, -0.0093211, -0.0049452, -0.0032860, 0.0035804
9: -0.0017810, 0.0011624, -0.0017330, 0.0011575, -0.0023651, 0.0021706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016803, upper bound: 0.0016481
time: 2.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016803, upper bound: 0.0017133
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020492, -0.0004689, -0.0020369, -0.0004742, -0.0012813, 0.0011991
1: -0.0095107, -0.0055006, -0.0094796, -0.0055139, -0.0032516, 0.0030428
2: 0.0291295, 0.0316174, 0.0291489, 0.0316091, -0.0020173, 0.0018878
3: 0.0005247, 0.0051703, 0.0005402, 0.0051342, -0.0035250, 0.0037668
4: -0.0085670, -0.0044880, -0.0085353, -0.0045016, -0.0033074, 0.0030951
5: 0.0104932, 0.0120382, 0.0105052, 0.0120331, -0.0012528, 0.0011723
6: 0.0010541, 0.0069499, 0.0010737, 0.0069041, -0.0044736, 0.0047806
7: 0.9787968, 0.9829225, 0.9788106, 0.9828905, -0.0031304, 0.0033452
8: -0.0092973, -0.0048740, -0.0092826, -0.0049084, -0.0033563, 0.0035866
9: -0.0017800, 0.0011418, -0.0017573, 0.0011321, -0.0023691, 0.0022170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016453
time: 2.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017152
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020001, -0.0004576, -0.0019666, -0.0004515, -0.0011523, 0.0011401
1: -0.0093861, -0.0054718, -0.0093010, -0.0054564, -0.0029240, 0.0028933
2: 0.0292068, 0.0316353, 0.0292596, 0.0316449, -0.0018141, 0.0017950
3: 0.0004914, 0.0050259, 0.0004735, 0.0049274, -0.0033517, 0.0033874
4: -0.0084403, -0.0044588, -0.0083537, -0.0044431, -0.0029742, 0.0029429
5: 0.0105412, 0.0120493, 0.0105740, 0.0120553, -0.0011266, 0.0011147
6: 0.0010118, 0.0067667, 0.0009891, 0.0066416, -0.0042537, 0.0042990
7: 0.9787673, 0.9827942, 0.9787514, 0.9827068, -0.0029766, 0.0030082
8: -0.0093291, -0.0050115, -0.0093461, -0.0051053, -0.0031914, 0.0032253
9: -0.0016892, 0.0011628, -0.0016272, 0.0011740, -0.0021305, 0.0021081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016991
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017444
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019996, -0.0004693, -0.0019758, -0.0004661, -0.0011546, 0.0011578
1: -0.0093848, -0.0055015, -0.0093245, -0.0054934, -0.0029300, 0.0029381
2: 0.0292077, 0.0316168, 0.0292450, 0.0316219, -0.0018178, 0.0018228
3: 0.0005259, 0.0050244, 0.0005164, 0.0049546, -0.0034037, 0.0033943
4: -0.0084389, -0.0044890, -0.0083776, -0.0044807, -0.0029803, 0.0029886
5: 0.0105417, 0.0120379, 0.0105650, 0.0120410, -0.0011289, 0.0011320
6: 0.0010555, 0.0067647, 0.0010435, 0.0066762, -0.0043197, 0.0043078
7: 0.9787979, 0.9827928, 0.9787894, 0.9827309, -0.0030227, 0.0030144
8: -0.0092963, -0.0050130, -0.0093053, -0.0050794, -0.0032409, 0.0032319
9: -0.0016883, 0.0011411, -0.0016444, 0.0011470, -0.0021348, 0.0021408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016924
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017410
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020456, -0.0004421, -0.0019661, -0.0004590, -0.0012473, 0.0011699
1: -0.0095015, -0.0054323, -0.0092998, -0.0054755, -0.0031652, 0.0029688
2: 0.0291352, 0.0316598, 0.0292604, 0.0316330, -0.0019637, 0.0018419
3: 0.0004457, 0.0051597, 0.0004957, 0.0049260, -0.0034392, 0.0036668
4: -0.0085577, -0.0044186, -0.0083525, -0.0044625, -0.0032196, 0.0030198
5: 0.0104968, 0.0120645, 0.0105745, 0.0120479, -0.0012195, 0.0011438
6: 0.0009538, 0.0069364, 0.0010172, 0.0066399, -0.0043648, 0.0046536
7: 0.9787267, 0.9829130, 0.9787710, 0.9827055, -0.0030543, 0.0032564
8: -0.0093726, -0.0048842, -0.0093250, -0.0051066, -0.0032747, 0.0034913
9: -0.0017733, 0.0011915, -0.0016264, 0.0011601, -0.0023062, 0.0021631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016991
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017444
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020451, -0.0004541, -0.0019754, -0.0004740, -0.0012496, 0.0011887
1: -0.0095002, -0.0054629, -0.0093233, -0.0055133, -0.0031710, 0.0030164
2: 0.0291361, 0.0316408, 0.0292458, 0.0316095, -0.0019673, 0.0018714
3: 0.0004811, 0.0051581, 0.0005395, 0.0049532, -0.0034944, 0.0036735
4: -0.0085563, -0.0044497, -0.0083764, -0.0045010, -0.0032255, 0.0030682
5: 0.0104973, 0.0120527, 0.0105654, 0.0120333, -0.0012217, 0.0011622
6: 0.0009987, 0.0069344, 0.0010729, 0.0066744, -0.0044349, 0.0046621
7: 0.9787582, 0.9829117, 0.9788100, 0.9827297, -0.0031033, 0.0032623
8: -0.0093389, -0.0048856, -0.0092832, -0.0050807, -0.0033272, 0.0034977
9: -0.0017724, 0.0011692, -0.0016435, 0.0011325, -0.0023105, 0.0021978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016924
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017411
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020344, -0.0004513, -0.0020394, -0.0004503, -0.0011824, 0.0011806
1: -0.0094732, -0.0054557, -0.0094859, -0.0054532, -0.0030005, 0.0029959
2: 0.0291528, 0.0316453, 0.0291449, 0.0316468, -0.0018615, 0.0018587
3: 0.0004727, 0.0051268, 0.0004699, 0.0051416, -0.0034706, 0.0034759
4: -0.0085288, -0.0044424, -0.0085418, -0.0044399, -0.0030520, 0.0030473
5: 0.0105077, 0.0120555, 0.0105028, 0.0120565, -0.0011560, 0.0011542
6: 0.0009881, 0.0068947, 0.0009845, 0.0069135, -0.0044047, 0.0044114
7: 0.9787507, 0.9828838, 0.9787482, 0.9828970, -0.0030822, 0.0030869
8: -0.0093468, -0.0049154, -0.0093495, -0.0049014, -0.0033046, 0.0033096
9: -0.0017527, 0.0011745, -0.0017620, 0.0011763, -0.0021862, 0.0021829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016901
time: 2.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017463
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020339, -0.0004630, -0.0020522, -0.0004633, -0.0011838, 0.0011971
1: -0.0094718, -0.0054856, -0.0095183, -0.0054863, -0.0030040, 0.0030379
2: 0.0291537, 0.0316268, 0.0291248, 0.0316263, -0.0018637, 0.0018847
3: 0.0005073, 0.0051253, 0.0005082, 0.0051791, -0.0035192, 0.0034800
4: -0.0085275, -0.0044728, -0.0085747, -0.0044735, -0.0030556, 0.0030900
5: 0.0105082, 0.0120440, 0.0104903, 0.0120437, -0.0011574, 0.0011704
6: 0.0010320, 0.0068927, 0.0010331, 0.0069610, -0.0044664, 0.0044166
7: 0.9787814, 0.9828824, 0.9787821, 0.9829302, -0.0031253, 0.0030905
8: -0.0093139, -0.0049169, -0.0093131, -0.0048657, -0.0033509, 0.0033135
9: -0.0017517, 0.0011528, -0.0017855, 0.0011522, -0.0021888, 0.0022134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016804
time: 2.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017429
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020822, -0.0004357, -0.0020390, -0.0004583, -0.0012782, 0.0012140
1: -0.0095944, -0.0054163, -0.0094848, -0.0054735, -0.0032436, 0.0030807
2: 0.0290776, 0.0316697, 0.0291456, 0.0316342, -0.0020123, 0.0019113
3: 0.0004271, 0.0052673, 0.0004934, 0.0051402, -0.0035689, 0.0037576
4: -0.0086522, -0.0044023, -0.0085406, -0.0044605, -0.0032993, 0.0031336
5: 0.0104610, 0.0120707, 0.0105032, 0.0120487, -0.0012497, 0.0011869
6: 0.0009302, 0.0070730, 0.0010143, 0.0069117, -0.0045294, 0.0047688
7: 0.9787101, 0.9830086, 0.9787690, 0.9828957, -0.0031694, 0.0033370
8: -0.0093903, -0.0047817, -0.0093272, -0.0049027, -0.0033982, 0.0035778
9: -0.0018410, 0.0012032, -0.0017611, 0.0011616, -0.0023633, 0.0022447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016901
time: 1.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017464
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020817, -0.0004478, -0.0020517, -0.0004720, -0.0012795, 0.0012322
1: -0.0095931, -0.0054468, -0.0095171, -0.0055082, -0.0032468, 0.0031269
2: 0.0290784, 0.0316508, 0.0291256, 0.0316127, -0.0020143, 0.0019400
3: 0.0004625, 0.0052657, 0.0005336, 0.0051777, -0.0036224, 0.0037613
4: -0.0086508, -0.0044334, -0.0085736, -0.0044958, -0.0033025, 0.0031806
5: 0.0104615, 0.0120589, 0.0104907, 0.0120353, -0.0012509, 0.0012047
6: 0.0009751, 0.0070710, 0.0010653, 0.0069593, -0.0045973, 0.0047735
7: 0.9787415, 0.9830073, 0.9788048, 0.9829291, -0.0032170, 0.0033403
8: -0.0093566, -0.0047832, -0.0092889, -0.0048670, -0.0034491, 0.0035813
9: -0.0018400, 0.0011810, -0.0017847, 0.0011362, -0.0023657, 0.0022783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016803
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017429
time: 2.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019619, -0.0004841, -0.0019240, -0.0003026, -0.0013161, 0.0010848
1: -0.0092891, -0.0055390, -0.0091930, -0.0050785, -0.0033397, 0.0027527
2: 0.0292670, 0.0315936, 0.0293266, 0.0318793, -0.0020720, 0.0017078
3: 0.0005692, 0.0049136, 0.0000358, 0.0048023, -0.0031889, 0.0038689
4: -0.0083416, -0.0045271, -0.0082439, -0.0040588, -0.0033970, 0.0028000
5: 0.0105786, 0.0120234, 0.0106156, 0.0122008, -0.0012867, 0.0010606
6: 0.0011106, 0.0066241, 0.0004336, 0.0064828, -0.0040471, 0.0049101
7: 0.9788364, 0.9826945, 0.9783627, 0.9825957, -0.0028320, 0.0034359
8: -0.0092550, -0.0051185, -0.0097628, -0.0052244, -0.0030363, 0.0036838
9: -0.0016186, 0.0011138, -0.0015486, 0.0014493, -0.0024334, 0.0020057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016001
time: 1.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016637
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019614, -0.0004951, -0.0019350, -0.0003188, -0.0013187, 0.0011040
1: -0.0092878, -0.0055670, -0.0092209, -0.0051196, -0.0033464, 0.0028015
2: 0.0292678, 0.0315763, 0.0293093, 0.0318538, -0.0020761, 0.0017381
3: 0.0006016, 0.0049121, 0.0000834, 0.0048346, -0.0032454, 0.0038766
4: -0.0083403, -0.0045556, -0.0082722, -0.0041006, -0.0034038, 0.0028496
5: 0.0105791, 0.0120127, 0.0106049, 0.0121850, -0.0012893, 0.0010794
6: 0.0011517, 0.0066222, 0.0004940, 0.0065238, -0.0041189, 0.0049199
7: 0.9788651, 0.9826931, 0.9784049, 0.9826243, -0.0028822, 0.0034427
8: -0.0092241, -0.0051199, -0.0097175, -0.0051937, -0.0030902, 0.0036911
9: -0.0016176, 0.0010934, -0.0015689, 0.0014194, -0.0024382, 0.0020412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015976
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016631
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020127, -0.0004640, -0.0019236, -0.0003116, -0.0014035, 0.0011164
1: -0.0094182, -0.0054880, -0.0091919, -0.0051014, -0.0035616, 0.0028329
2: 0.0291869, 0.0316253, 0.0293274, 0.0318651, -0.0022096, 0.0017576
3: 0.0005101, 0.0050631, 0.0000623, 0.0048009, -0.0032818, 0.0041259
4: -0.0084729, -0.0044752, -0.0082427, -0.0040820, -0.0036228, 0.0028816
5: 0.0105289, 0.0120431, 0.0106161, 0.0121920, -0.0013722, 0.0010915
6: 0.0010356, 0.0068139, 0.0004672, 0.0064811, -0.0041650, 0.0052364
7: 0.9787839, 0.9828273, 0.9783862, 0.9825944, -0.0029145, 0.0036642
8: -0.0093112, -0.0049761, -0.0097376, -0.0052257, -0.0031248, 0.0039286
9: -0.0017126, 0.0011510, -0.0015477, 0.0014327, -0.0025950, 0.0020641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016001
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016637
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020122, -0.0004751, -0.0019345, -0.0003277, -0.0014062, 0.0011359
1: -0.0094169, -0.0055163, -0.0092197, -0.0051420, -0.0035684, 0.0028825
2: 0.0291877, 0.0316077, 0.0293101, 0.0318399, -0.0022139, 0.0017883
3: 0.0005430, 0.0050616, 0.0001094, 0.0048332, -0.0033393, 0.0041339
4: -0.0084716, -0.0045040, -0.0082710, -0.0041233, -0.0036297, 0.0029320
5: 0.0105294, 0.0120322, 0.0106053, 0.0121764, -0.0013748, 0.0011106
6: 0.0010772, 0.0068119, 0.0005270, 0.0065220, -0.0042379, 0.0052464
7: 0.9788131, 0.9828259, 0.9784281, 0.9826230, -0.0029655, 0.0036712
8: -0.0092800, -0.0049775, -0.0096928, -0.0051950, -0.0031795, 0.0039361
9: -0.0017117, 0.0011304, -0.0015680, 0.0014030, -0.0026000, 0.0021002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015976
time: 1.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016631
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019967, -0.0004779, -0.0020024, -0.0003017, -0.0013432, 0.0011232
1: -0.0093775, -0.0055233, -0.0093920, -0.0050762, -0.0034085, 0.0028502
2: 0.0292122, 0.0316034, 0.0292032, 0.0318807, -0.0021147, 0.0017683
3: 0.0005510, 0.0050159, 0.0000331, 0.0050328, -0.0033018, 0.0039486
4: -0.0084315, -0.0045111, -0.0084463, -0.0040564, -0.0034670, 0.0028991
5: 0.0105446, 0.0120295, 0.0105390, 0.0122017, -0.0013132, 0.0010981
6: 0.0010875, 0.0067540, 0.0004301, 0.0067754, -0.0041904, 0.0050113
7: 0.9788203, 0.9827853, 0.9783603, 0.9828004, -0.0029322, 0.0035067
8: -0.0092723, -0.0050210, -0.0097654, -0.0050050, -0.0031438, 0.0037597
9: -0.0016829, 0.0011253, -0.0016935, 0.0014510, -0.0024835, 0.0020767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015938
time: 1.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016729
time: 1.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019962, -0.0004889, -0.0020143, -0.0003166, -0.0013453, 0.0011419
1: -0.0093762, -0.0055513, -0.0094222, -0.0051141, -0.0034139, 0.0028978
2: 0.0292130, 0.0315860, 0.0291845, 0.0318572, -0.0021180, 0.0017978
3: 0.0005835, 0.0050144, 0.0000770, 0.0050677, -0.0033570, 0.0039548
4: -0.0084302, -0.0045396, -0.0084769, -0.0040949, -0.0034725, 0.0029476
5: 0.0105451, 0.0120187, 0.0105273, 0.0121871, -0.0013153, 0.0011165
6: 0.0011287, 0.0067521, 0.0004859, 0.0068197, -0.0042605, 0.0050192
7: 0.9788491, 0.9827841, 0.9783993, 0.9828314, -0.0029813, 0.0035122
8: -0.0092414, -0.0050224, -0.0097236, -0.0049717, -0.0031964, 0.0037656
9: -0.0016820, 0.0011048, -0.0017155, 0.0014234, -0.0024874, 0.0021114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015903
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016732
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020497, -0.0004578, -0.0020020, -0.0003109, -0.0014324, 0.0011566
1: -0.0095121, -0.0054723, -0.0093908, -0.0050996, -0.0036349, 0.0029350
2: 0.0291287, 0.0316350, 0.0292039, 0.0318662, -0.0022551, 0.0018209
3: 0.0004919, 0.0051719, 0.0000603, 0.0050314, -0.0034000, 0.0042108
4: -0.0085684, -0.0044592, -0.0084451, -0.0040802, -0.0036973, 0.0029854
5: 0.0104927, 0.0120491, 0.0105394, 0.0121927, -0.0014004, 0.0011308
6: 0.0010124, 0.0069519, 0.0004646, 0.0067736, -0.0043151, 0.0053441
7: 0.9787678, 0.9829238, 0.9783844, 0.9827991, -0.0030195, 0.0037395
8: -0.0093286, -0.0048725, -0.0097396, -0.0050063, -0.0032373, 0.0040094
9: -0.0017810, 0.0011624, -0.0016927, 0.0014340, -0.0026484, 0.0021385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015938
time: 2.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016729
time: 2.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020492, -0.0004689, -0.0020138, -0.0003255, -0.0014344, 0.0011767
1: -0.0095107, -0.0055006, -0.0094210, -0.0051365, -0.0036401, 0.0029859
2: 0.0291295, 0.0316174, 0.0291852, 0.0318433, -0.0022583, 0.0018525
3: 0.0005247, 0.0051703, 0.0001029, 0.0050663, -0.0034591, 0.0042168
4: -0.0085670, -0.0044880, -0.0084758, -0.0041177, -0.0037026, 0.0030372
5: 0.0104932, 0.0120382, 0.0105278, 0.0121785, -0.0014024, 0.0011504
6: 0.0010541, 0.0069499, 0.0005188, 0.0068180, -0.0043900, 0.0053517
7: 0.9787968, 0.9829225, 0.9784222, 0.9828302, -0.0030719, 0.0037449
8: -0.0092973, -0.0048740, -0.0096990, -0.0049730, -0.0032936, 0.0040151
9: -0.0017800, 0.0011418, -0.0017146, 0.0014071, -0.0026522, 0.0021756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015903
time: 2.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016732
time: 2.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020001, -0.0004576, -0.0019396, -0.0003007, -0.0013129, 0.0011185
1: -0.0093861, -0.0054718, -0.0092327, -0.0050736, -0.0033318, 0.0028384
2: 0.0292068, 0.0316353, 0.0293020, 0.0318823, -0.0020671, 0.0017610
3: 0.0004914, 0.0050259, 0.0000301, 0.0048482, -0.0032882, 0.0038597
4: -0.0084403, -0.0044588, -0.0082842, -0.0040537, -0.0033890, 0.0028872
5: 0.0105412, 0.0120493, 0.0106003, 0.0122027, -0.0012837, 0.0010936
6: 0.0010118, 0.0067667, 0.0004264, 0.0065411, -0.0041731, 0.0048985
7: 0.9787673, 0.9827942, 0.9783576, 0.9826364, -0.0029202, 0.0034277
8: -0.0093291, -0.0050115, -0.0097683, -0.0051807, -0.0031309, 0.0036751
9: -0.0016892, 0.0011628, -0.0015774, 0.0014529, -0.0024276, 0.0020681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016420
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016965
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019996, -0.0004693, -0.0019492, -0.0003170, -0.0013149, 0.0011334
1: -0.0093848, -0.0055015, -0.0092569, -0.0051149, -0.0033367, 0.0028761
2: 0.0292077, 0.0316168, 0.0292870, 0.0318567, -0.0020701, 0.0017844
3: 0.0005259, 0.0050244, 0.0000779, 0.0048763, -0.0033319, 0.0038655
4: -0.0084389, -0.0044890, -0.0083089, -0.0040957, -0.0033940, 0.0029255
5: 0.0105417, 0.0120379, 0.0105910, 0.0121868, -0.0012856, 0.0011081
6: 0.0010555, 0.0067647, 0.0004870, 0.0065768, -0.0042286, 0.0049058
7: 0.9787979, 0.9827928, 0.9784002, 0.9826613, -0.0029589, 0.0034328
8: -0.0092963, -0.0050130, -0.0097227, -0.0051540, -0.0031725, 0.0036805
9: -0.0016883, 0.0011411, -0.0015951, 0.0014228, -0.0024312, 0.0020956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016325
time: 1.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016927
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020456, -0.0004421, -0.0019392, -0.0003097, -0.0014022, 0.0011483
1: -0.0095015, -0.0054323, -0.0092314, -0.0050964, -0.0035583, 0.0029139
2: 0.0291352, 0.0316598, 0.0293028, 0.0318682, -0.0022076, 0.0018078
3: 0.0004457, 0.0051597, 0.0000566, 0.0048468, -0.0033756, 0.0041222
4: -0.0085577, -0.0044186, -0.0082830, -0.0040770, -0.0036194, 0.0029639
5: 0.0104968, 0.0120645, 0.0106008, 0.0121939, -0.0013709, 0.0011227
6: 0.0009538, 0.0069364, 0.0004599, 0.0065393, -0.0042841, 0.0052315
7: 0.9787267, 0.9829130, 0.9783811, 0.9826351, -0.0029978, 0.0036608
8: -0.0093726, -0.0048842, -0.0097431, -0.0051821, -0.0032141, 0.0039249
9: -0.0017733, 0.0011915, -0.0015765, 0.0014363, -0.0025926, 0.0021231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016420
time: 1.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016965
time: 2.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020451, -0.0004541, -0.0019487, -0.0003258, -0.0014041, 0.0011642
1: -0.0095002, -0.0054629, -0.0092556, -0.0051372, -0.0035631, 0.0029543
2: 0.0291361, 0.0316408, 0.0292878, 0.0318429, -0.0022106, 0.0018329
3: 0.0004811, 0.0051581, 0.0001038, 0.0048748, -0.0034225, 0.0041277
4: -0.0085563, -0.0044497, -0.0083076, -0.0041184, -0.0036243, 0.0030051
5: 0.0104973, 0.0120527, 0.0105915, 0.0121782, -0.0013728, 0.0011382
6: 0.0009987, 0.0069344, 0.0005199, 0.0065749, -0.0043435, 0.0052386
7: 0.9787582, 0.9829117, 0.9784231, 0.9826601, -0.0030394, 0.0036657
8: -0.0093389, -0.0048856, -0.0096981, -0.0051554, -0.0032587, 0.0039302
9: -0.0017724, 0.0011692, -0.0015942, 0.0014066, -0.0025962, 0.0021526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016325
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016927
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020344, -0.0004513, -0.0020164, -0.0002995, -0.0013377, 0.0011607
1: -0.0094732, -0.0054557, -0.0094275, -0.0050707, -0.0033946, 0.0029455
2: 0.0291528, 0.0316453, 0.0291812, 0.0318841, -0.0021060, 0.0018274
3: 0.0004727, 0.0051268, 0.0000268, 0.0050739, -0.0034123, 0.0039325
4: -0.0085288, -0.0044424, -0.0084823, -0.0040508, -0.0034529, 0.0029961
5: 0.0105077, 0.0120555, 0.0105253, 0.0122038, -0.0013079, 0.0011348
6: 0.0009881, 0.0068947, 0.0004221, 0.0068275, -0.0043306, 0.0049909
7: 0.9787507, 0.9828838, 0.9783546, 0.9828368, -0.0030303, 0.0034924
8: -0.0093468, -0.0049154, -0.0097715, -0.0049659, -0.0032490, 0.0037444
9: -0.0017527, 0.0011745, -0.0017194, 0.0014550, -0.0024734, 0.0021462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016356
time: 2.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0017091
time: 2.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020339, -0.0004630, -0.0020282, -0.0003145, -0.0013390, 0.0011737
1: -0.0094718, -0.0054856, -0.0094575, -0.0051087, -0.0033980, 0.0029784
2: 0.0291537, 0.0316268, 0.0291626, 0.0318606, -0.0021081, 0.0018478
3: 0.0005073, 0.0051253, 0.0000708, 0.0051086, -0.0034503, 0.0039364
4: -0.0085275, -0.0044728, -0.0085128, -0.0040894, -0.0034563, 0.0030295
5: 0.0105082, 0.0120440, 0.0105137, 0.0121892, -0.0013092, 0.0011475
6: 0.0010320, 0.0068927, 0.0004779, 0.0068716, -0.0043789, 0.0049958
7: 0.9787814, 0.9828824, 0.9783937, 0.9828677, -0.0030642, 0.0034958
8: -0.0093139, -0.0049169, -0.0097296, -0.0049328, -0.0032853, 0.0037481
9: -0.0017517, 0.0011528, -0.0017412, 0.0014273, -0.0024758, 0.0021701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016247
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0017049
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020822, -0.0004357, -0.0020159, -0.0003088, -0.0014278, 0.0011942
1: -0.0095944, -0.0054163, -0.0094263, -0.0050941, -0.0036233, 0.0030303
2: 0.0290776, 0.0316697, 0.0291819, 0.0318696, -0.0022479, 0.0018800
3: 0.0004271, 0.0052673, 0.0000539, 0.0050725, -0.0035105, 0.0041974
4: -0.0086522, -0.0044023, -0.0084811, -0.0040746, -0.0036855, 0.0030824
5: 0.0104610, 0.0120707, 0.0105258, 0.0121948, -0.0013960, 0.0011675
6: 0.0009302, 0.0070730, 0.0004565, 0.0068257, -0.0044553, 0.0053271
7: 0.9787101, 0.9830086, 0.9783787, 0.9828357, -0.0031176, 0.0037276
8: -0.0093903, -0.0047817, -0.0097457, -0.0049672, -0.0033425, 0.0039966
9: -0.0018410, 0.0012032, -0.0017185, 0.0014380, -0.0026400, 0.0022079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016356
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0017091
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020817, -0.0004478, -0.0020277, -0.0003233, -0.0014290, 0.0012087
1: -0.0095931, -0.0054468, -0.0094562, -0.0051310, -0.0036263, 0.0030674
2: 0.0290784, 0.0316508, 0.0291633, 0.0318467, -0.0022498, 0.0019030
3: 0.0004625, 0.0052657, 0.0000966, 0.0051072, -0.0035534, 0.0042009
4: -0.0086508, -0.0044334, -0.0085116, -0.0041121, -0.0036886, 0.0031200
5: 0.0104615, 0.0120589, 0.0105142, 0.0121806, -0.0013971, 0.0011818
6: 0.0009751, 0.0070710, 0.0005107, 0.0068698, -0.0045097, 0.0053315
7: 0.9787415, 0.9830073, 0.9784166, 0.9828663, -0.0031557, 0.0037308
8: -0.0093566, -0.0047832, -0.0097050, -0.0049341, -0.0033834, 0.0040000
9: -0.0018400, 0.0011810, -0.0017403, 0.0014111, -0.0026422, 0.0022349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016247
time: 1.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0017049
time: 1.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019240, -0.0003026, -0.0019619, -0.0004841, -0.0010848, 0.0013161
1: -0.0091930, -0.0050785, -0.0092891, -0.0055390, -0.0027527, 0.0033397
2: 0.0293266, 0.0318793, 0.0292670, 0.0315936, -0.0017078, 0.0020720
3: 0.0000358, 0.0048023, 0.0005692, 0.0049136, -0.0038689, 0.0031889
4: -0.0082439, -0.0040588, -0.0083416, -0.0045271, -0.0028000, 0.0033970
5: 0.0106156, 0.0122008, 0.0105786, 0.0120234, -0.0010606, 0.0012867
6: 0.0004336, 0.0064828, 0.0011106, 0.0066241, -0.0049101, 0.0040471
7: 0.9783627, 0.9825957, 0.9788364, 0.9826945, -0.0034359, 0.0028320
8: -0.0097628, -0.0052244, -0.0092550, -0.0051185, -0.0036838, 0.0030363
9: -0.0015486, 0.0014493, -0.0016186, 0.0011138, -0.0020057, 0.0024334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016221
time: 1.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
time: 1.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019240, -0.0003026, -0.0019386, -0.0003316, -0.0011521, 0.0012098
1: -0.0091930, -0.0050785, -0.0092300, -0.0051520, -0.0029236, 0.0030700
2: 0.0293266, 0.0318793, 0.0293037, 0.0318337, -0.0018138, 0.0019046
3: 0.0000358, 0.0048023, 0.0001209, 0.0048450, -0.0035564, 0.0033869
4: -0.0082439, -0.0040588, -0.0082814, -0.0041335, -0.0029738, 0.0031227
5: 0.0106156, 0.0122008, 0.0106014, 0.0121725, -0.0011264, 0.0011828
6: 0.0004336, 0.0064828, 0.0005416, 0.0065371, -0.0045135, 0.0042984
7: 0.9783627, 0.9825957, 0.9784383, 0.9826336, -0.0031584, 0.0030078
8: -0.0097628, -0.0052244, -0.0096818, -0.0051837, -0.0033863, 0.0032248
9: -0.0015486, 0.0014493, -0.0015755, 0.0013958, -0.0021302, 0.0022368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016221
time: 2.21 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
time: 2.22 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0019350, -0.0003188, -0.0019614, -0.0004951, -0.0011040, 0.0013187
1: -0.0092209, -0.0051196, -0.0092878, -0.0055670, -0.0028015, 0.0033464
2: 0.0293093, 0.0318538, 0.0292678, 0.0315763, -0.0017381, 0.0020761
3: 0.0000834, 0.0048346, 0.0006016, 0.0049121, -0.0038766, 0.0032454
4: -0.0082722, -0.0041006, -0.0083403, -0.0045556, -0.0028496, 0.0034038
5: 0.0106049, 0.0121850, 0.0105791, 0.0120127, -0.0010794, 0.0012893
6: 0.0004940, 0.0065238, 0.0011517, 0.0066222, -0.0049199, 0.0041189
7: 0.9784049, 0.9826243, 0.9788651, 0.9826931, -0.0034427, 0.0028822
8: -0.0097175, -0.0051937, -0.0092241, -0.0051199, -0.0036911, 0.0030902
9: -0.0015689, 0.0014194, -0.0016176, 0.0010934, -0.0020412, 0.0024382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016221
time: 1.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019350, -0.0003188, -0.0019381, -0.0003434, -0.0011752, 0.0012126
1: -0.0092209, -0.0051196, -0.0092288, -0.0051821, -0.0029822, 0.0030771
2: 0.0293093, 0.0318538, 0.0293044, 0.0318150, -0.0018502, 0.0019091
3: 0.0000834, 0.0048346, 0.0001558, 0.0048437, -0.0035647, 0.0034547
4: -0.0082722, -0.0041006, -0.0082802, -0.0041641, -0.0030334, 0.0031300
5: 0.0106049, 0.0121850, 0.0106018, 0.0121609, -0.0011490, 0.0011855
6: 0.0004940, 0.0065238, 0.0005859, 0.0065354, -0.0045241, 0.0043845
7: 0.9784049, 0.9826243, 0.9784693, 0.9826324, -0.0031657, 0.0030681
8: -0.0097175, -0.0051937, -0.0096486, -0.0051850, -0.0033942, 0.0032895
9: -0.0015689, 0.0014194, -0.0015746, 0.0013739, -0.0021729, 0.0022420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016221
time: 2.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
time: 2.14 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019236, -0.0003116, -0.0020127, -0.0004640, -0.0011164, 0.0014035
1: -0.0091919, -0.0051014, -0.0094182, -0.0054880, -0.0028329, 0.0035616
2: 0.0293274, 0.0318651, 0.0291869, 0.0316253, -0.0017576, 0.0022096
3: 0.0000623, 0.0048009, 0.0005101, 0.0050631, -0.0041260, 0.0032818
4: -0.0082427, -0.0040820, -0.0084729, -0.0044752, -0.0028816, 0.0036228
5: 0.0106161, 0.0121920, 0.0105289, 0.0120431, -0.0010915, 0.0013722
6: 0.0004672, 0.0064811, 0.0010356, 0.0068139, -0.0052364, 0.0041650
7: 0.9783862, 0.9825944, 0.9787839, 0.9828273, -0.0036642, 0.0029145
8: -0.0097376, -0.0052257, -0.0093112, -0.0049761, -0.0039286, 0.0031248
9: -0.0015477, 0.0014327, -0.0017126, 0.0011510, -0.0020641, 0.0025950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016275
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
time: 2.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019236, -0.0003116, -0.0019896, -0.0003142, -0.0011848, 0.0013032
1: -0.0091919, -0.0051014, -0.0093594, -0.0051079, -0.0030067, 0.0033071
2: 0.0293274, 0.0318651, 0.0292234, 0.0318611, -0.0018653, 0.0020517
3: 0.0000623, 0.0048009, 0.0000698, 0.0049950, -0.0038311, 0.0034831
4: -0.0082427, -0.0040820, -0.0084131, -0.0040886, -0.0030583, 0.0033639
5: 0.0106161, 0.0121920, 0.0105515, 0.0121895, -0.0011584, 0.0012741
6: 0.0004672, 0.0064811, 0.0004767, 0.0067274, -0.0048621, 0.0044205
7: 0.9783862, 0.9825944, 0.9783930, 0.9827667, -0.0034023, 0.0030932
8: -0.0097376, -0.0052257, -0.0097305, -0.0050410, -0.0036478, 0.0033164
9: -0.0015477, 0.0014327, -0.0016698, 0.0014279, -0.0021907, 0.0024096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016275
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016546
time: 2.17 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0019345, -0.0003277, -0.0020122, -0.0004751, -0.0011359, 0.0014062
1: -0.0092197, -0.0051420, -0.0094169, -0.0055163, -0.0028825, 0.0035684
2: 0.0293101, 0.0318399, 0.0291877, 0.0316077, -0.0017883, 0.0022139
3: 0.0001094, 0.0048332, 0.0005430, 0.0050616, -0.0041339, 0.0033393
4: -0.0082710, -0.0041233, -0.0084716, -0.0045040, -0.0029320, 0.0036297
5: 0.0106053, 0.0121764, 0.0105294, 0.0120322, -0.0011106, 0.0013748
6: 0.0005270, 0.0065220, 0.0010772, 0.0068119, -0.0052464, 0.0042379
7: 0.9784281, 0.9826230, 0.9788131, 0.9828259, -0.0036712, 0.0029655
8: -0.0096928, -0.0051950, -0.0092800, -0.0049775, -0.0039361, 0.0031795
9: -0.0015680, 0.0014030, -0.0017117, 0.0011304, -0.0021002, 0.0026000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016275
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
time: 1.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019345, -0.0003277, -0.0019891, -0.0003260, -0.0012082, 0.0013061
1: -0.0092197, -0.0051420, -0.0093582, -0.0051379, -0.0030660, 0.0033143
2: 0.0293101, 0.0318399, 0.0292242, 0.0318424, -0.0019022, 0.0020562
3: 0.0001094, 0.0048332, 0.0001046, 0.0049936, -0.0038395, 0.0035518
4: -0.0082710, -0.0041233, -0.0084119, -0.0041191, -0.0031186, 0.0033712
5: 0.0106053, 0.0121764, 0.0105520, 0.0121780, -0.0011813, 0.0012769
6: 0.0005270, 0.0065220, 0.0005209, 0.0067257, -0.0048728, 0.0045077
7: 0.9784281, 0.9826230, 0.9784238, 0.9827656, -0.0034097, 0.0031543
8: -0.0096928, -0.0051950, -0.0096974, -0.0050423, -0.0036558, 0.0033819
9: -0.0015680, 0.0014030, -0.0016689, 0.0014061, -0.0022339, 0.0024148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016275
time: 2.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020024, -0.0003017, -0.0019161, -0.0004877, -0.0011901, 0.0012747
1: -0.0093920, -0.0050762, -0.0091730, -0.0055482, -0.0030201, 0.0032348
2: 0.0292032, 0.0318807, 0.0293390, 0.0315879, -0.0018737, 0.0020069
3: 0.0000331, 0.0050328, 0.0005800, 0.0047791, -0.0037474, 0.0034986
4: -0.0084463, -0.0040564, -0.0082236, -0.0045365, -0.0030719, 0.0032904
5: 0.0105390, 0.0122017, 0.0106233, 0.0120199, -0.0011636, 0.0012463
6: 0.0004301, 0.0067754, 0.0011242, 0.0064534, -0.0047559, 0.0044402
7: 0.9783603, 0.9828004, 0.9788460, 0.9825751, -0.0033280, 0.0031070
8: -0.0097654, -0.0050050, -0.0092448, -0.0052465, -0.0035681, 0.0033312
9: -0.0016935, 0.0014510, -0.0015340, 0.0011071, -0.0022004, 0.0023569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017128
time: 1.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017389
time: 1.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020024, -0.0003017, -0.0019929, -0.0004829, -0.0011572, 0.0013096
1: -0.0093920, -0.0050762, -0.0093678, -0.0055361, -0.0029366, 0.0033233
2: 0.0292032, 0.0318807, 0.0292182, 0.0315954, -0.0018219, 0.0020618
3: 0.0000331, 0.0050328, 0.0005658, 0.0050047, -0.0038499, 0.0034020
4: -0.0084463, -0.0040564, -0.0084216, -0.0045241, -0.0029871, 0.0033803
5: 0.0105390, 0.0122017, 0.0105483, 0.0120246, -0.0011314, 0.0012804
6: 0.0004301, 0.0067754, 0.0011062, 0.0067397, -0.0048860, 0.0043175
7: 0.9783603, 0.9828004, 0.9788334, 0.9827754, -0.0034190, 0.0030212
8: -0.0097654, -0.0050050, -0.0092582, -0.0050317, -0.0036657, 0.0032392
9: -0.0016935, 0.0014510, -0.0016759, 0.0011160, -0.0021397, 0.0024214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017306
time: 2.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017581
time: 2.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020143, -0.0003166, -0.0019156, -0.0004993, -0.0012141, 0.0012770
1: -0.0094222, -0.0051141, -0.0091718, -0.0055776, -0.0030811, 0.0032405
2: 0.0291845, 0.0318572, 0.0293398, 0.0315696, -0.0019115, 0.0020104
3: 0.0000770, 0.0050677, 0.0006140, 0.0047776, -0.0037540, 0.0035693
4: -0.0084769, -0.0040949, -0.0082222, -0.0045664, -0.0031339, 0.0032962
5: 0.0105273, 0.0121871, 0.0106238, 0.0120085, -0.0011871, 0.0012485
6: 0.0004859, 0.0068197, 0.0011674, 0.0064516, -0.0047643, 0.0045298
7: 0.9783993, 0.9828314, 0.9788762, 0.9825737, -0.0033338, 0.0031698
8: -0.0097236, -0.0049717, -0.0092123, -0.0052479, -0.0035744, 0.0033985
9: -0.0017155, 0.0014234, -0.0015331, 0.0010857, -0.0022449, 0.0023611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017128
time: 1.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017389
time: 2.11 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020143, -0.0003166, -0.0019924, -0.0004940, -0.0011806, 0.0013124
1: -0.0094222, -0.0051141, -0.0093665, -0.0055642, -0.0029960, 0.0033303
2: 0.0291845, 0.0318572, 0.0292190, 0.0315780, -0.0018587, 0.0020662
3: 0.0000770, 0.0050677, 0.0005984, 0.0050032, -0.0038580, 0.0034707
4: -0.0084769, -0.0040949, -0.0084203, -0.0045527, -0.0030474, 0.0033875
5: 0.0105273, 0.0121871, 0.0105488, 0.0120137, -0.0011543, 0.0012831
6: 0.0004859, 0.0068197, 0.0011476, 0.0067378, -0.0048963, 0.0044048
7: 0.9783993, 0.9828314, 0.9788623, 0.9827740, -0.0034262, 0.0030823
8: -0.0097236, -0.0049717, -0.0092272, -0.0050331, -0.0036735, 0.0033047
9: -0.0017155, 0.0014234, -0.0016749, 0.0010955, -0.0021829, 0.0024265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017306
time: 1.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017581
time: 1.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020020, -0.0003109, -0.0019656, -0.0004685, -0.0012260, 0.0013560
1: -0.0093908, -0.0050996, -0.0092986, -0.0054993, -0.0031111, 0.0034410
2: 0.0292039, 0.0318662, 0.0292611, 0.0316182, -0.0019301, 0.0021348
3: 0.0000603, 0.0050314, 0.0005233, 0.0049246, -0.0039862, 0.0036041
4: -0.0084451, -0.0040802, -0.0083513, -0.0044868, -0.0031645, 0.0035001
5: 0.0105394, 0.0121927, 0.0105749, 0.0120387, -0.0011986, 0.0013257
6: 0.0004646, 0.0067736, 0.0010523, 0.0066381, -0.0050590, 0.0045740
7: 0.9783844, 0.9827991, 0.9787956, 0.9827043, -0.0035401, 0.0032007
8: -0.0097396, -0.0050063, -0.0092987, -0.0051080, -0.0037955, 0.0034316
9: -0.0016927, 0.0014340, -0.0016255, 0.0011427, -0.0022668, 0.0025071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017171
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017390
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020020, -0.0003109, -0.0020449, -0.0004628, -0.0011910, 0.0013905
1: -0.0093908, -0.0050996, -0.0094999, -0.0054849, -0.0030224, 0.0035285
2: 0.0292039, 0.0318662, 0.0291362, 0.0316272, -0.0018751, 0.0021891
3: 0.0000603, 0.0050314, 0.0005066, 0.0051578, -0.0040876, 0.0035013
4: -0.0084451, -0.0040802, -0.0085560, -0.0044721, -0.0030743, 0.0035891
5: 0.0105394, 0.0121927, 0.0104974, 0.0120443, -0.0011644, 0.0013594
6: 0.0004646, 0.0067736, 0.0010310, 0.0069340, -0.0051877, 0.0044436
7: 0.9783844, 0.9827991, 0.9787807, 0.9829113, -0.0036301, 0.0031094
8: -0.0097396, -0.0050063, -0.0093146, -0.0048860, -0.0038920, 0.0033338
9: -0.0016927, 0.0014340, -0.0017721, 0.0011532, -0.0022021, 0.0025709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017352
time: 2.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017579
time: 2.17 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020138, -0.0003255, -0.0019651, -0.0004796, -0.0012508, 0.0013581
1: -0.0094210, -0.0051365, -0.0092973, -0.0055277, -0.0031740, 0.0034464
2: 0.0291852, 0.0318433, 0.0292619, 0.0316006, -0.0019692, 0.0021382
3: 0.0001029, 0.0050663, 0.0005561, 0.0049231, -0.0039925, 0.0036770
4: -0.0084758, -0.0041177, -0.0083500, -0.0045156, -0.0032285, 0.0035056
5: 0.0105278, 0.0121785, 0.0105754, 0.0120278, -0.0012229, 0.0013278
6: 0.0005188, 0.0068180, 0.0010939, 0.0066362, -0.0050670, 0.0046665
7: 0.9784222, 0.9828302, 0.9788247, 0.9827030, -0.0035457, 0.0032654
8: -0.0096990, -0.0049730, -0.0092675, -0.0051094, -0.0038015, 0.0035010
9: -0.0017146, 0.0014071, -0.0016246, 0.0011221, -0.0023126, 0.0025111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017172
time: 1.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017390
time: 2.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020138, -0.0003255, -0.0020444, -0.0004739, -0.0012159, 0.0013932
1: -0.0094210, -0.0051365, -0.0094986, -0.0055131, -0.0030855, 0.0035355
2: 0.0291852, 0.0318433, 0.0291371, 0.0316097, -0.0019142, 0.0021934
3: 0.0001029, 0.0050663, 0.0005392, 0.0051562, -0.0040957, 0.0035744
4: -0.0084758, -0.0041177, -0.0085547, -0.0045007, -0.0031384, 0.0035962
5: 0.0105278, 0.0121785, 0.0104979, 0.0120334, -0.0011887, 0.0013621
6: 0.0005188, 0.0068180, 0.0010724, 0.0069321, -0.0051979, 0.0045363
7: 0.9784222, 0.9828302, 0.9788097, 0.9829100, -0.0036373, 0.0031743
8: -0.0096990, -0.0049730, -0.0092836, -0.0048874, -0.0038997, 0.0034033
9: -0.0017146, 0.0014071, -0.0017712, 0.0011327, -0.0022481, 0.0025760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017352
time: 2.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017579
time: 2.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019396, -0.0003007, -0.0020001, -0.0004576, -0.0011185, 0.0013129
1: -0.0092327, -0.0050736, -0.0093861, -0.0054718, -0.0028384, 0.0033318
2: 0.0293020, 0.0318823, 0.0292068, 0.0316353, -0.0017610, 0.0020671
3: 0.0000301, 0.0048482, 0.0004914, 0.0050259, -0.0038597, 0.0032882
4: -0.0082842, -0.0040537, -0.0084403, -0.0044588, -0.0028872, 0.0033890
5: 0.0106003, 0.0122027, 0.0105412, 0.0120493, -0.0010936, 0.0012837
6: 0.0004264, 0.0065411, 0.0010118, 0.0067667, -0.0048985, 0.0041731
7: 0.9783576, 0.9826364, 0.9787673, 0.9827942, -0.0034277, 0.0029202
8: -0.0097683, -0.0051807, -0.0093291, -0.0050115, -0.0036751, 0.0031309
9: -0.0015774, 0.0014529, -0.0016892, 0.0011628, -0.0020681, 0.0024276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016167
time: 2.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
time: 2.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019396, -0.0003007, -0.0019738, -0.0003060, -0.0011899, 0.0012019
1: -0.0092327, -0.0050736, -0.0093193, -0.0050870, -0.0030195, 0.0030500
2: 0.0293020, 0.0318823, 0.0292483, 0.0318740, -0.0018733, 0.0018922
3: 0.0000301, 0.0048482, 0.0000456, 0.0049485, -0.0035333, 0.0034979
4: -0.0082842, -0.0040537, -0.0083723, -0.0040674, -0.0030713, 0.0031024
5: 0.0106003, 0.0122027, 0.0105670, 0.0121976, -0.0011633, 0.0011751
6: 0.0004264, 0.0065411, 0.0004460, 0.0066685, -0.0044842, 0.0044393
7: 0.9783576, 0.9826364, 0.9783714, 0.9827255, -0.0031378, 0.0031064
8: -0.0097683, -0.0051807, -0.0097535, -0.0050852, -0.0033642, 0.0033305
9: -0.0015774, 0.0014529, -0.0016405, 0.0014432, -0.0022000, 0.0022223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016167
time: 2.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
time: 2.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0019492, -0.0003170, -0.0019996, -0.0004693, -0.0011334, 0.0013149
1: -0.0092569, -0.0051149, -0.0093848, -0.0055015, -0.0028761, 0.0033367
2: 0.0292870, 0.0318567, 0.0292077, 0.0316168, -0.0017844, 0.0020701
3: 0.0000779, 0.0048763, 0.0005259, 0.0050244, -0.0038655, 0.0033319
4: -0.0083089, -0.0040957, -0.0084389, -0.0044890, -0.0029255, 0.0033940
5: 0.0105910, 0.0121868, 0.0105417, 0.0120379, -0.0011081, 0.0012856
6: 0.0004870, 0.0065768, 0.0010555, 0.0067647, -0.0049058, 0.0042286
7: 0.9784002, 0.9826613, 0.9787979, 0.9827928, -0.0034328, 0.0029589
8: -0.0097227, -0.0051540, -0.0092963, -0.0050130, -0.0036805, 0.0031725
9: -0.0015951, 0.0014228, -0.0016883, 0.0011411, -0.0020956, 0.0024312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016167
time: 2.20 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016618, upper bound: 0.0016522
time: 1.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019492, -0.0003170, -0.0019733, -0.0003188, -0.0012069, 0.0012041
1: -0.0092569, -0.0051149, -0.0093181, -0.0051197, -0.0030626, 0.0030555
2: 0.0292870, 0.0318567, 0.0292490, 0.0318538, -0.0019001, 0.0018957
3: 0.0000779, 0.0048763, 0.0000835, 0.0049472, -0.0035397, 0.0035479
4: -0.0083089, -0.0040957, -0.0083711, -0.0041006, -0.0031152, 0.0031080
5: 0.0105910, 0.0121868, 0.0105674, 0.0121850, -0.0011800, 0.0011772
6: 0.0004870, 0.0065768, 0.0004940, 0.0066667, -0.0044923, 0.0045027
7: 0.9784002, 0.9826613, 0.9784050, 0.9827243, -0.0031435, 0.0031508
8: -0.0097227, -0.0051540, -0.0097175, -0.0050865, -0.0033703, 0.0033782
9: -0.0015951, 0.0014228, -0.0016397, 0.0014194, -0.0022315, 0.0022263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016167
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
time: 2.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019392, -0.0003097, -0.0020456, -0.0004421, -0.0011483, 0.0014022
1: -0.0092314, -0.0050964, -0.0095015, -0.0054323, -0.0029139, 0.0035583
2: 0.0293028, 0.0318682, 0.0291352, 0.0316598, -0.0018078, 0.0022076
3: 0.0000566, 0.0048468, 0.0004457, 0.0051597, -0.0041222, 0.0033756
4: -0.0082830, -0.0040770, -0.0085577, -0.0044186, -0.0029639, 0.0036194
5: 0.0106008, 0.0121939, 0.0104968, 0.0120645, -0.0011227, 0.0013709
6: 0.0004599, 0.0065393, 0.0009538, 0.0069364, -0.0052315, 0.0042841
7: 0.9783811, 0.9826351, 0.9787267, 0.9829130, -0.0036608, 0.0029978
8: -0.0097431, -0.0051821, -0.0093726, -0.0048842, -0.0039249, 0.0032141
9: -0.0015765, 0.0014363, -0.0017733, 0.0011915, -0.0021231, 0.0025926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016226
time: 2.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016546
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019392, -0.0003097, -0.0020186, -0.0002931, -0.0012203, 0.0012969
1: -0.0092314, -0.0050964, -0.0094330, -0.0050543, -0.0030966, 0.0032910
2: 0.0293028, 0.0318682, 0.0291777, 0.0318943, -0.0019212, 0.0020418
3: 0.0000566, 0.0048468, 0.0000077, 0.0050803, -0.0038125, 0.0035873
4: -0.0082830, -0.0040770, -0.0084880, -0.0040341, -0.0031498, 0.0033475
5: 0.0106008, 0.0121939, 0.0105232, 0.0122102, -0.0011931, 0.0012680
6: 0.0004599, 0.0065393, 0.0003979, 0.0068357, -0.0048386, 0.0045527
7: 0.9783811, 0.9826351, 0.9783377, 0.9828426, -0.0033858, 0.0031858
8: -0.0097431, -0.0051821, -0.0097896, -0.0049597, -0.0036301, 0.0034157
9: -0.0015765, 0.0014363, -0.0017234, 0.0014670, -0.0022562, 0.0023979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016226
time: 2.17 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
time: 2.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0019487, -0.0003258, -0.0020451, -0.0004541, -0.0011642, 0.0014041
1: -0.0092556, -0.0051372, -0.0095002, -0.0054629, -0.0029543, 0.0035631
2: 0.0292878, 0.0318429, 0.0291361, 0.0316408, -0.0018329, 0.0022106
3: 0.0001038, 0.0048748, 0.0004811, 0.0051581, -0.0041277, 0.0034225
4: -0.0083076, -0.0041184, -0.0085563, -0.0044497, -0.0030051, 0.0036243
5: 0.0105915, 0.0121782, 0.0104973, 0.0120527, -0.0011382, 0.0013728
6: 0.0005199, 0.0065749, 0.0009987, 0.0069344, -0.0052386, 0.0043435
7: 0.9784231, 0.9826601, 0.9787582, 0.9829117, -0.0036657, 0.0030394
8: -0.0096981, -0.0051554, -0.0093389, -0.0048856, -0.0039302, 0.0032587
9: -0.0015942, 0.0014066, -0.0017724, 0.0011692, -0.0021526, 0.0025962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016226
time: 2.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016546
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019487, -0.0003258, -0.0020181, -0.0003056, -0.0012385, 0.0012990
1: -0.0092556, -0.0051372, -0.0094318, -0.0050860, -0.0031429, 0.0032963
2: 0.0292878, 0.0318429, 0.0291785, 0.0318747, -0.0019499, 0.0020451
3: 0.0001038, 0.0048748, 0.0000444, 0.0050789, -0.0038186, 0.0036409
4: -0.0083076, -0.0041184, -0.0084868, -0.0040663, -0.0031968, 0.0033529
5: 0.0105915, 0.0121782, 0.0105236, 0.0121980, -0.0012109, 0.0012700
6: 0.0005199, 0.0065749, 0.0004445, 0.0068339, -0.0048463, 0.0046208
7: 0.9784231, 0.9826601, 0.9783704, 0.9828414, -0.0033912, 0.0032334
8: -0.0096981, -0.0051554, -0.0097547, -0.0049610, -0.0036359, 0.0034667
9: -0.0015942, 0.0014066, -0.0017226, 0.0014439, -0.0022899, 0.0024017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016226
time: 2.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
time: 1.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020164, -0.0002995, -0.0019539, -0.0004586, -0.0012281, 0.0012693
1: -0.0094275, -0.0050707, -0.0092689, -0.0054744, -0.0031166, 0.0032209
2: 0.0291812, 0.0318841, 0.0292796, 0.0316337, -0.0019335, 0.0019983
3: 0.0000268, 0.0050739, 0.0004944, 0.0048901, -0.0037313, 0.0036104
4: -0.0084823, -0.0040508, -0.0083210, -0.0044614, -0.0031701, 0.0032762
5: 0.0105253, 0.0122038, 0.0105864, 0.0120483, -0.0012007, 0.0012409
6: 0.0004221, 0.0068275, 0.0010156, 0.0065943, -0.0047355, 0.0045820
7: 0.9783546, 0.9828368, 0.9787700, 0.9826736, -0.0033137, 0.0032063
8: -0.0097715, -0.0049659, -0.0093262, -0.0051408, -0.0035528, 0.0034376
9: -0.0017194, 0.0014550, -0.0016038, 0.0011609, -0.0022708, 0.0023468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017070
time: 1.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017389
time: 1.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020164, -0.0002995, -0.0020269, -0.0004562, -0.0011972, 0.0013049
1: -0.0094275, -0.0050707, -0.0094542, -0.0054681, -0.0030381, 0.0033113
2: 0.0291812, 0.0318841, 0.0291646, 0.0316376, -0.0018849, 0.0020544
3: 0.0000268, 0.0050739, 0.0004871, 0.0051048, -0.0038360, 0.0035196
4: -0.0084823, -0.0040508, -0.0085095, -0.0044550, -0.0030903, 0.0033682
5: 0.0105253, 0.0122038, 0.0105150, 0.0120507, -0.0011705, 0.0012758
6: 0.0004221, 0.0068275, 0.0010064, 0.0068668, -0.0048684, 0.0044668
7: 0.9783546, 0.9828368, 0.9787635, 0.9828643, -0.0034067, 0.0031256
8: -0.0097715, -0.0049659, -0.0093331, -0.0049364, -0.0036525, 0.0033512
9: -0.0017194, 0.0014550, -0.0017388, 0.0011655, -0.0022136, 0.0024127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017242
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017581
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020282, -0.0003145, -0.0019534, -0.0004715, -0.0012461, 0.0012707
1: -0.0094575, -0.0051087, -0.0092676, -0.0055070, -0.0031623, 0.0032245
2: 0.0291626, 0.0318606, 0.0292804, 0.0316135, -0.0019619, 0.0020005
3: 0.0000708, 0.0051086, 0.0005322, 0.0048886, -0.0037355, 0.0036633
4: -0.0085128, -0.0040894, -0.0083197, -0.0044946, -0.0032166, 0.0032799
5: 0.0105137, 0.0121892, 0.0105869, 0.0120358, -0.0012183, 0.0012423
6: 0.0004779, 0.0068716, 0.0010635, 0.0065924, -0.0047408, 0.0046493
7: 0.9783937, 0.9828677, 0.9788034, 0.9826723, -0.0033174, 0.0032533
8: -0.0097296, -0.0049328, -0.0092903, -0.0051422, -0.0035568, 0.0034881
9: -0.0017412, 0.0014273, -0.0016029, 0.0011371, -0.0023041, 0.0023494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017070
time: 2.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017389
time: 1.60 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.18 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016602
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017121
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016608
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017138
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016602
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017121
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016608
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017138
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016481
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017132
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016453
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017152
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016803, upper bound: 0.0016481
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016803, upper bound: 0.0017133
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016453
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017152
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016991
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017444
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016924
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017410
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016991
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017444
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016924
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017411
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016901
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017463
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016804
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017429
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016901
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017464
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016803
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017429
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016001
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016637
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015976
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016631
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016001
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016637
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015976
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016631
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015938
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016729
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015903
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016732
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015938
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016729
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015903
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016732
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016420
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016965
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016325
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016927
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016420
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016965
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016325
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016927
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016356
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0017091
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016247
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0017049
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016356
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0017091
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016247
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0017049
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016221
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016221
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016221
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016221
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016275
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016275
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016546
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016275
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016275
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017128
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017389
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017306
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017128
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017389
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017306
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017581
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017171
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017390
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017352
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017579
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017172
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017390
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017352
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017579
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016167
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016167
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016522
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016167
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016618, upper bound: 0.0016522
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016167
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016522
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016226
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016546
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016226
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016637, upper bound: 0.0016547
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016226
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016546
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016226
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016547
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017070
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017389
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017242
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016281, upper bound: 0.0017581
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017070
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 7, lower bound: -0.0016226, upper bound: 0.0017389
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017581
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017390
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 7, lower bound: -0.0016662, upper bound: 0.0017579
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017390
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017579

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.11 + 598.02 = 602.13 seconds
