## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0071008


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0072958, 0.0058223, -0.0072958, 0.0058223, -0.0131181, 0.0131181)
1: (0.9968639, 1.0115018, 0.9968639, 1.0115018, -0.0134915, 0.0134915)
2: (-0.0067976, 0.0064179, -0.0067976, 0.0064179, -0.0132154, 0.0132154)
3: (-0.0003281, 0.0026445, -0.0003281, 0.0026445, -0.0027107, 0.0027107)
4: (-0.0077486, 0.0017800, -0.0077486, 0.0017800, -0.0095286, 0.0095286)
5: (-0.0025431, 0.0092563, -0.0025431, 0.0092563, -0.0117994, 0.0117994)
6: (-0.0103348, 0.0020602, -0.0103348, 0.0020602, -0.0123950, 0.0123950)
7: (-0.0059883, 0.0005387, -0.0059883, 0.0005387, -0.0065270, 0.0065270)
8: (-0.0141420, -0.0011354, -0.0141420, -0.0011354, -0.0130066, 0.0130066)
9: (-0.0057393, 0.0079594, -0.0057393, 0.0079594, -0.0136987, 0.0136987)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.82 = 3.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0113717, upper bound: 0.0113717

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0112087, upper bound: 0.0112243
time: 0.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0112095, upper bound: 0.0112095
time: 1.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 1, lower bound: -0.0112087, upper bound: 0.0112243
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 1, lower bound: -0.0112095, upper bound: 0.0112095

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0072548, 0.0057687, -0.0072833, 0.0058059, -0.0130608, 0.0130520
1: 0.9970167, 1.0114206, 0.9969126, 1.0114770, -0.0133061, 0.0133570
2: -0.0067443, 0.0063755, -0.0067813, 0.0064049, -0.0131493, 0.0131568
3: -0.0003166, 0.0025811, -0.0003246, 0.0026242, -0.0026780, 0.0026442
4: -0.0076947, 0.0017661, -0.0077321, 0.0017757, -0.0094705, 0.0094982
5: -0.0025245, 0.0091889, -0.0025374, 0.0092357, -0.0117602, 0.0117263
6: -0.0102516, 0.0020536, -0.0103094, 0.0020582, -0.0123098, 0.0123630
7: -0.0058814, 0.0005090, -0.0059541, 0.0005296, -0.0064110, 0.0064631
8: -0.0141005, -0.0014661, -0.0141294, -0.0012410, -0.0128595, 0.0126633
9: -0.0056650, 0.0079351, -0.0057166, 0.0079520, -0.0136170, 0.0136517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107858, upper bound: 0.0107664
time: 1.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110166, upper bound: 0.0110303
time: 0.96 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0075742, 0.0061868, -0.0072727, 0.0057922, -0.0133664, 0.0134596
1: 0.9970174, 1.0120546, 0.9969327, 1.0114561, -0.0132985, 0.0140429
2: -0.0071596, 0.0067062, -0.0067676, 0.0063940, -0.0135536, 0.0134739
3: -0.0004060, 0.0025808, -0.0003216, 0.0026160, -0.0027940, 0.0026482
4: -0.0081151, 0.0018744, -0.0077183, 0.0017722, -0.0098872, 0.0095927
5: -0.0026699, 0.0097149, -0.0025326, 0.0092184, -0.0118883, 0.0122475
6: -0.0109009, 0.0021050, -0.0102880, 0.0020565, -0.0129574, 0.0123930
7: -0.0058809, 0.0007407, -0.0059402, 0.0005220, -0.0064029, 0.0066809
8: -0.0144244, -0.0014676, -0.0141187, -0.0012840, -0.0131403, 0.0126511
9: -0.0062442, 0.0081250, -0.0056975, 0.0079457, -0.0141899, 0.0138225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107507
time: 0.99 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0110165, upper bound: 0.0110165
time: 1.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.63 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 1, lower bound: -0.0107858, upper bound: 0.0107664
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 1, lower bound: -0.0110166, upper bound: 0.0110303
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 1, lower bound: -0.0107890, upper bound: 0.0107507
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.63
Output dim: 1, lower bound: -0.0110165, upper bound: 0.0110165

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0070219, 0.0054637, -0.0067500, 0.0051078, -0.0121297, 0.0122137
1: 0.9970215, 1.0109580, 0.9966976, 1.0104182, -0.0121949, 0.0129332
2: -0.0064414, 0.0061342, -0.0060879, 0.0058526, -0.0122940, 0.0122221
3: -0.0002514, 0.0025790, -0.0001753, 0.0027134, -0.0026082, 0.0024429
4: -0.0073881, 0.0016871, -0.0070302, 0.0015948, -0.0089829, 0.0087173
5: -0.0024184, 0.0088053, -0.0022945, 0.0083574, -0.0107758, 0.0110998
6: -0.0097780, 0.0020161, -0.0092251, 0.0019724, -0.0117504, 0.0112413
7: -0.0058779, 0.0003400, -0.0061046, 0.0001428, -0.0060207, 0.0064447
8: -0.0138644, -0.0014767, -0.0135887, -0.0007758, -0.0130886, 0.0121120
9: -0.0052426, 0.0077965, -0.0047495, 0.0076348, -0.0128774, 0.0125460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104346, upper bound: 0.0102349
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106393, upper bound: 0.0106232
time: 0.93 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0072133, 0.0057143, -0.0070328, 0.0054780, -0.0126913, 0.0127471
1: 0.9970174, 1.0113379, 0.9969170, 1.0109794, -0.0127391, 0.0132756
2: -0.0066903, 0.0063324, -0.0064556, 0.0061455, -0.0128358, 0.0127880
3: -0.0003050, 0.0025808, -0.0002545, 0.0026225, -0.0026655, 0.0025281
4: -0.0076400, 0.0017520, -0.0074024, 0.0016908, -0.0093308, 0.0091544
5: -0.0025056, 0.0091205, -0.0024233, 0.0088232, -0.0113287, 0.0115438
6: -0.0101671, 0.0020469, -0.0098001, 0.0020179, -0.0121850, 0.0118470
7: -0.0058809, 0.0004789, -0.0059512, 0.0003479, -0.0062288, 0.0064300
8: -0.0140584, -0.0014676, -0.0138754, -0.0012503, -0.0128082, 0.0124079
9: -0.0055897, 0.0079104, -0.0052623, 0.0078030, -0.0133927, 0.0131727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106285, upper bound: 0.0104663
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108732, upper bound: 0.0108903
time: 1.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0073461, 0.0058882, -0.0067385, 0.0050928, -0.0124389, 0.0126267
1: 0.9970225, 1.0116017, 0.9967169, 1.0103952, -0.0121869, 0.0136199
2: -0.0068630, 0.0064700, -0.0060730, 0.0058408, -0.0127038, 0.0125430
3: -0.0003422, 0.0025787, -0.0001721, 0.0027055, -0.0027284, 0.0024470
4: -0.0078148, 0.0017971, -0.0070152, 0.0015910, -0.0094058, 0.0088122
5: -0.0025661, 0.0093392, -0.0022893, 0.0083386, -0.0109047, 0.0116285
6: -0.0104372, 0.0020683, -0.0092019, 0.0019705, -0.0124077, 0.0112702
7: -0.0058774, 0.0005752, -0.0060912, 0.0001345, -0.0060119, 0.0066664
8: -0.0141931, -0.0014783, -0.0135771, -0.0008173, -0.0133758, 0.0120988
9: -0.0058306, 0.0079894, -0.0047287, 0.0076280, -0.0134585, 0.0127181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104409, upper bound: 0.0102286
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106440, upper bound: 0.0106081
time: 1.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0075311, 0.0061304, -0.0070225, 0.0054645, -0.0129956, 0.0131528
1: 0.9970182, 1.0119690, 0.9969370, 1.0109591, -0.0127286, 0.0139585
2: -0.0071035, 0.0066615, -0.0064422, 0.0061348, -0.0132384, 0.0131038
3: -0.0003939, 0.0025805, -0.0002516, 0.0026142, -0.0027814, 0.0025316
4: -0.0080583, 0.0018598, -0.0073889, 0.0016873, -0.0097456, 0.0092487
5: -0.0026503, 0.0096438, -0.0024187, 0.0088062, -0.0114565, 0.0120625
6: -0.0108132, 0.0020981, -0.0097792, 0.0020162, -0.0128294, 0.0118773
7: -0.0058804, 0.0007094, -0.0059372, 0.0003405, -0.0062208, 0.0066466
8: -0.0143806, -0.0014690, -0.0138650, -0.0012933, -0.0130873, 0.0123960
9: -0.0061660, 0.0080994, -0.0052437, 0.0077969, -0.0139629, 0.0133430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106297, upper bound: 0.0104499
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108740, upper bound: 0.0108740
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0104346, upper bound: 0.0102349
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0106393, upper bound: 0.0106232
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0106285, upper bound: 0.0104663
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0108732, upper bound: 0.0108903
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0104409, upper bound: 0.0102286
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0106440, upper bound: 0.0106081
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0106297, upper bound: 0.0104499
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 1, lower bound: -0.0108740, upper bound: 0.0108740

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0065708, 0.0048732, -0.0065634, 0.0048635, -0.0114342, 0.0114365
1: 0.9970204, 1.0100621, 0.9967004, 1.0100473, -0.0117680, 0.0119972
2: -0.0058549, 0.0056670, -0.0058452, 0.0056593, -0.0115142, 0.0115122
3: -0.0001251, 0.0025795, -0.0001230, 0.0027123, -0.0024562, 0.0023612
4: -0.0067943, 0.0015340, -0.0067846, 0.0015315, -0.0083259, 0.0083186
5: -0.0022129, 0.0080623, -0.0022095, 0.0080501, -0.0102630, 0.0102719
6: -0.0088608, 0.0019436, -0.0088457, 0.0019424, -0.0108032, 0.0107893
7: -0.0058787, 0.0000128, -0.0061027, 0.0000074, -0.0058861, 0.0061155
8: -0.0134070, -0.0014743, -0.0133995, -0.0007818, -0.0126252, 0.0119252
9: -0.0044244, 0.0075282, -0.0044110, 0.0075238, -0.0119483, 0.0119392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104346, upper bound: 0.0102349
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104346, upper bound: 0.0102349
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0068557, 0.0052461, -0.0067053, 0.0050493, -0.0119049, 0.0119514
1: 0.9970248, 1.0106277, 0.9966985, 1.0103292, -0.0121037, 0.0124410
2: -0.0062253, 0.0059621, -0.0060298, 0.0058063, -0.0120316, 0.0119918
3: -0.0002049, 0.0025777, -0.0001628, 0.0027131, -0.0024831, 0.0024264
4: -0.0071693, 0.0016307, -0.0069714, 0.0015797, -0.0087490, 0.0086021
5: -0.0023427, 0.0085315, -0.0022742, 0.0082839, -0.0106265, 0.0108057
6: -0.0094400, 0.0019894, -0.0091343, 0.0019652, -0.0114052, 0.0111237
7: -0.0058756, 0.0002195, -0.0061040, 0.0001104, -0.0059860, 0.0063235
8: -0.0136959, -0.0014839, -0.0135434, -0.0007776, -0.0129182, 0.0120596
9: -0.0049411, 0.0076977, -0.0046684, 0.0076082, -0.0125494, 0.0123661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106393, upper bound: 0.0106232
time: 2.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106393, upper bound: 0.0106232
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0067621, 0.0051237, -0.0068449, 0.0052321, -0.0119942, 0.0119686
1: 0.9970161, 1.0104420, 0.9969198, 1.0106065, -0.0122944, 0.0123254
2: -0.0061037, 0.0058652, -0.0062114, 0.0059509, -0.0120547, 0.0120766
3: -0.0001787, 0.0025814, -0.0002019, 0.0026213, -0.0025079, 0.0024413
4: -0.0070462, 0.0015990, -0.0071552, 0.0016271, -0.0086733, 0.0087542
5: -0.0023001, 0.0083775, -0.0023378, 0.0085138, -0.0108139, 0.0107153
6: -0.0092499, 0.0019743, -0.0094182, 0.0019877, -0.0112375, 0.0113925
7: -0.0058819, 0.0001516, -0.0059492, 0.0002117, -0.0060935, 0.0061008
8: -0.0136010, -0.0014645, -0.0136850, -0.0012563, -0.0123447, 0.0122205
9: -0.0047715, 0.0076420, -0.0049217, 0.0076913, -0.0124628, 0.0125637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106285, upper bound: 0.0104663
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0106285, upper bound: 0.0104663
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0070434, 0.0054919, -0.0069857, 0.0054164, -0.0124598, 0.0124776
1: 0.9970207, 1.0110006, 0.9969180, 1.0108860, -0.0126498, 0.0127714
2: -0.0064694, 0.0061565, -0.0063944, 0.0060968, -0.0125662, 0.0125509
3: -0.0002574, 0.0025794, -0.0002413, 0.0026221, -0.0025367, 0.0025112
4: -0.0074164, 0.0016944, -0.0073405, 0.0016748, -0.0090912, 0.0090349
5: -0.0024282, 0.0088407, -0.0024019, 0.0087457, -0.0111739, 0.0112426
6: -0.0098217, 0.0020196, -0.0097044, 0.0020103, -0.0118320, 0.0117240
7: -0.0058786, 0.0003556, -0.0059505, 0.0003138, -0.0061923, 0.0063062
8: -0.0138862, -0.0014747, -0.0138277, -0.0012522, -0.0126340, 0.0123530
9: -0.0052816, 0.0078093, -0.0051770, 0.0077750, -0.0130566, 0.0129863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108732, upper bound: 0.0108903
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108732, upper bound: 0.0108903
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0068761, 0.0052729, -0.0065522, 0.0048489, -0.0117250, 0.0118251
1: 0.9970284, 1.0106685, 0.9967198, 1.0100253, -0.0117591, 0.0126593
2: -0.0062519, 0.0059833, -0.0058307, 0.0056478, -0.0118997, 0.0118140
3: -0.0002106, 0.0025763, -0.0001199, 0.0027043, -0.0025768, 0.0023650
4: -0.0071962, 0.0016376, -0.0067699, 0.0015278, -0.0087240, 0.0084075
5: -0.0023520, 0.0085652, -0.0022045, 0.0080318, -0.0103838, 0.0107697
6: -0.0094816, 0.0019927, -0.0088231, 0.0019406, -0.0114222, 0.0108157
7: -0.0058733, 0.0002343, -0.0060892, -0.0000007, -0.0058726, 0.0063235
8: -0.0137166, -0.0014911, -0.0133882, -0.0008234, -0.0128932, 0.0118971
9: -0.0049782, 0.0077098, -0.0043908, 0.0075172, -0.0124954, 0.0121006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099728, upper bound: 0.0099386
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103303, upper bound: 0.0101102
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0071826, 0.0056742, -0.0066938, 0.0050342, -0.0122168, 0.0123680
1: 0.9970257, 1.0112770, 0.9967179, 1.0103064, -0.0120953, 0.0131306
2: -0.0066504, 0.0063007, -0.0060148, 0.0057944, -0.0124448, 0.0123155
3: -0.0002964, 0.0025774, -0.0001595, 0.0027051, -0.0026043, 0.0024303
4: -0.0075996, 0.0017416, -0.0069562, 0.0015758, -0.0091754, 0.0086978
5: -0.0024916, 0.0090699, -0.0022689, 0.0082649, -0.0107565, 0.0113389
6: -0.0101047, 0.0020420, -0.0091109, 0.0019633, -0.0120681, 0.0111529
7: -0.0058751, 0.0004566, -0.0060906, 0.0001020, -0.0059771, 0.0065472
8: -0.0140273, -0.0014854, -0.0135317, -0.0008192, -0.0132082, 0.0120464
9: -0.0055341, 0.0078921, -0.0046475, 0.0076014, -0.0131354, 0.0125397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100692, upper bound: 0.0102522
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105341, upper bound: 0.0104983
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0070583, 0.0055115, -0.0068349, 0.0052190, -0.0122773, 0.0123464
1: 0.9970237, 1.0110303, 0.9969397, 1.0105867, -0.0122822, 0.0129805
2: -0.0064888, 0.0061720, -0.0061983, 0.0059406, -0.0124294, 0.0123703
3: -0.0002616, 0.0025782, -0.0001991, 0.0026131, -0.0026281, 0.0024449
4: -0.0074361, 0.0016994, -0.0071420, 0.0016237, -0.0090597, 0.0088414
5: -0.0024350, 0.0088653, -0.0023332, 0.0084974, -0.0109323, 0.0111985
6: -0.0098521, 0.0020220, -0.0093979, 0.0019861, -0.0118381, 0.0114199
7: -0.0058764, 0.0003665, -0.0059353, 0.0002044, -0.0060808, 0.0063018
8: -0.0139013, -0.0014814, -0.0136748, -0.0012993, -0.0126020, 0.0121935
9: -0.0053087, 0.0078182, -0.0049035, 0.0076853, -0.0129940, 0.0127217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101036, upper bound: 0.0101375
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105226, upper bound: 0.0103378
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0073630, 0.0059103, -0.0069755, 0.0054030, -0.0127660, 0.0128858
1: 0.9970213, 1.0116352, 0.9969378, 1.0108658, -0.0126394, 0.0134556
2: -0.0068850, 0.0064875, -0.0063811, 0.0060861, -0.0129711, 0.0128686
3: -0.0003469, 0.0025792, -0.0002384, 0.0026138, -0.0026539, 0.0025146
4: -0.0078371, 0.0018028, -0.0073270, 0.0016713, -0.0095084, 0.0091298
5: -0.0025737, 0.0093670, -0.0023973, 0.0087288, -0.0113026, 0.0117642
6: -0.0104715, 0.0020710, -0.0096836, 0.0020087, -0.0124802, 0.0117547
7: -0.0058781, 0.0005875, -0.0059366, 0.0003064, -0.0061845, 0.0065241
8: -0.0142102, -0.0014761, -0.0138173, -0.0012952, -0.0129150, 0.0123412
9: -0.0058612, 0.0079994, -0.0051584, 0.0077689, -0.0136301, 0.0131578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108740, upper bound: 0.0108704
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0108740, upper bound: 0.0108704
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0104346, upper bound: 0.0102349
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0104346, upper bound: 0.0102349
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0106393, upper bound: 0.0106232
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0106393, upper bound: 0.0106232
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0106285, upper bound: 0.0104663
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0106285, upper bound: 0.0104663
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0108732, upper bound: 0.0108903
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0108732, upper bound: 0.0108903
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0099728, upper bound: 0.0099386
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0103303, upper bound: 0.0101102
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0100692, upper bound: 0.0102522
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0105341, upper bound: 0.0104983
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0101036, upper bound: 0.0101375
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0105226, upper bound: 0.0103378
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0108740, upper bound: 0.0108704
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 1, lower bound: -0.0108740, upper bound: 0.0108704

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0065708, 0.0048732, -0.0065344, 0.0048256, -0.0113963, 0.0114076
1: 0.9970204, 1.0100621, 0.9968004, 1.0099899, -0.0117120, 0.0118873
2: -0.0058549, 0.0056670, -0.0058076, 0.0056293, -0.0114842, 0.0114746
3: -0.0001251, 0.0025795, -0.0001149, 0.0026708, -0.0024134, 0.0023528
4: -0.0067943, 0.0015340, -0.0067465, 0.0015217, -0.0083160, 0.0082805
5: -0.0022129, 0.0080623, -0.0021964, 0.0080024, -0.0102153, 0.0102587
6: -0.0088608, 0.0019436, -0.0087869, 0.0019377, -0.0107985, 0.0107304
7: -0.0058787, 0.0000128, -0.0060327, -0.0000136, -0.0058651, 0.0060455
8: -0.0134070, -0.0014743, -0.0133701, -0.0009983, -0.0124087, 0.0118959
9: -0.0044244, 0.0075282, -0.0043585, 0.0075066, -0.0119310, 0.0118867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101304, upper bound: 0.0098304
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103258, upper bound: 0.0101147
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0065708, 0.0048732, -0.0068568, 0.0052477, -0.0118184, 0.0117300
1: 0.9970204, 1.0100621, 0.9968078, 1.0106301, -0.0123891, 0.0119257
2: -0.0058549, 0.0056670, -0.0062268, 0.0059633, -0.0118181, 0.0118938
3: -0.0001251, 0.0025795, -0.0002052, 0.0026677, -0.0024275, 0.0024764
4: -0.0067943, 0.0015340, -0.0071708, 0.0016311, -0.0084254, 0.0087049
5: -0.0022129, 0.0080623, -0.0023432, 0.0085334, -0.0107463, 0.0104055
6: -0.0088608, 0.0019436, -0.0094424, 0.0019896, -0.0108504, 0.0113859
7: -0.0058787, 0.0000128, -0.0060276, 0.0002203, -0.0060990, 0.0060403
8: -0.0134070, -0.0014743, -0.0136970, -0.0010141, -0.0123929, 0.0122228
9: -0.0044244, 0.0075282, -0.0049432, 0.0076983, -0.0121228, 0.0124714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101304, upper bound: 0.0098304
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103258, upper bound: 0.0101147
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0068557, 0.0052461, -0.0066764, 0.0050114, -0.0118671, 0.0119225
1: 0.9970248, 1.0106277, 0.9967985, 1.0102719, -0.0120469, 0.0123301
2: -0.0062253, 0.0059621, -0.0059922, 0.0057764, -0.0120017, 0.0119542
3: -0.0002049, 0.0025777, -0.0001547, 0.0026716, -0.0024390, 0.0024180
4: -0.0071693, 0.0016307, -0.0069333, 0.0015699, -0.0087392, 0.0085640
5: -0.0023427, 0.0085315, -0.0022610, 0.0082363, -0.0105789, 0.0107925
6: -0.0094400, 0.0019894, -0.0090755, 0.0019605, -0.0114006, 0.0110649
7: -0.0058756, 0.0002195, -0.0060341, 0.0000894, -0.0059650, 0.0062535
8: -0.0136959, -0.0014839, -0.0135141, -0.0009939, -0.0127019, 0.0120302
9: -0.0049411, 0.0076977, -0.0046160, 0.0075910, -0.0125322, 0.0123136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103061, upper bound: 0.0100923
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105330, upper bound: 0.0105112
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0068557, 0.0052461, -0.0070014, 0.0054370, -0.0122926, 0.0122476
1: 0.9970248, 1.0106277, 0.9968058, 1.0109173, -0.0127304, 0.0123779
2: -0.0062253, 0.0059621, -0.0064148, 0.0061130, -0.0123383, 0.0123769
3: -0.0002049, 0.0025777, -0.0002457, 0.0026686, -0.0024544, 0.0025433
4: -0.0071693, 0.0016307, -0.0073611, 0.0016801, -0.0088494, 0.0089918
5: -0.0023427, 0.0085315, -0.0024091, 0.0087715, -0.0111142, 0.0109406
6: -0.0094400, 0.0019894, -0.0097364, 0.0020128, -0.0114529, 0.0117257
7: -0.0058756, 0.0002195, -0.0060290, 0.0003252, -0.0062008, 0.0062484
8: -0.0136959, -0.0014839, -0.0138436, -0.0010097, -0.0126861, 0.0123598
9: -0.0049411, 0.0076977, -0.0052054, 0.0077843, -0.0127255, 0.0129031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103061, upper bound: 0.0100923
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105330, upper bound: 0.0105112
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0067621, 0.0051237, -0.0068164, 0.0051947, -0.0119568, 0.0119401
1: 0.9970161, 1.0104420, 0.9970237, 1.0105498, -0.0122367, 0.0122163
2: -0.0061037, 0.0058652, -0.0061742, 0.0059213, -0.0120250, 0.0120394
3: -0.0001787, 0.0025814, -0.0001939, 0.0025782, -0.0024655, 0.0024329
4: -0.0070462, 0.0015990, -0.0071176, 0.0016174, -0.0086636, 0.0087165
5: -0.0023001, 0.0083775, -0.0023248, 0.0084668, -0.0107668, 0.0107023
6: -0.0092499, 0.0019743, -0.0093601, 0.0019831, -0.0112329, 0.0113344
7: -0.0058819, 0.0001516, -0.0058765, 0.0001909, -0.0060728, 0.0060281
8: -0.0136010, -0.0014645, -0.0136560, -0.0014810, -0.0121200, 0.0121915
9: -0.0047715, 0.0076420, -0.0048698, 0.0076743, -0.0124458, 0.0125119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0102921
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0104470
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0067621, 0.0051237, -0.0071214, 0.0055940, -0.0123561, 0.0122451
1: 0.9970161, 1.0104420, 0.9970244, 1.0111555, -0.0129184, 0.0122464
2: -0.0061037, 0.0058652, -0.0065708, 0.0062372, -0.0123409, 0.0124360
3: -0.0001787, 0.0025814, -0.0002792, 0.0025779, -0.0024776, 0.0025563
4: -0.0070462, 0.0015990, -0.0075190, 0.0017208, -0.0087670, 0.0091180
5: -0.0023001, 0.0083775, -0.0024637, 0.0089691, -0.0112691, 0.0108412
6: -0.0092499, 0.0019743, -0.0099802, 0.0020321, -0.0112820, 0.0119545
7: -0.0058819, 0.0001516, -0.0058760, 0.0004122, -0.0062940, 0.0060276
8: -0.0136010, -0.0014645, -0.0139652, -0.0014825, -0.0121185, 0.0125007
9: -0.0047715, 0.0076420, -0.0054230, 0.0078557, -0.0126272, 0.0130650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0102921
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0104470
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0070434, 0.0054919, -0.0069572, 0.0053791, -0.0124225, 0.0124491
1: 0.9970207, 1.0110006, 0.9970217, 1.0108296, -0.0125916, 0.0126641
2: -0.0064694, 0.0061565, -0.0063574, 0.0060672, -0.0125366, 0.0125138
3: -0.0002574, 0.0025794, -0.0002333, 0.0025790, -0.0024952, 0.0025028
4: -0.0074164, 0.0016944, -0.0073030, 0.0016651, -0.0090815, 0.0089973
5: -0.0024282, 0.0088407, -0.0023889, 0.0086987, -0.0111269, 0.0112296
6: -0.0098217, 0.0020196, -0.0096465, 0.0020057, -0.0118274, 0.0116661
7: -0.0058786, 0.0003556, -0.0058778, 0.0002931, -0.0061717, 0.0062335
8: -0.0138862, -0.0014747, -0.0137988, -0.0014769, -0.0124093, 0.0123241
9: -0.0052816, 0.0078093, -0.0051253, 0.0077581, -0.0130396, 0.0129346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0106652
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0108713
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0070434, 0.0054919, -0.0072725, 0.0057918, -0.0128352, 0.0127644
1: 0.9970207, 1.0110006, 0.9970223, 1.0114555, -0.0132813, 0.0126936
2: -0.0064694, 0.0061565, -0.0067673, 0.0063937, -0.0128631, 0.0129238
3: -0.0002574, 0.0025794, -0.0003215, 0.0025787, -0.0025030, 0.0026256
4: -0.0074164, 0.0016944, -0.0077179, 0.0017721, -0.0091885, 0.0094123
5: -0.0024282, 0.0088407, -0.0025325, 0.0092179, -0.0116461, 0.0113732
6: -0.0098217, 0.0020196, -0.0102874, 0.0020565, -0.0118782, 0.0123070
7: -0.0058786, 0.0003556, -0.0058773, 0.0005218, -0.0064004, 0.0062330
8: -0.0138862, -0.0014747, -0.0141184, -0.0014784, -0.0124077, 0.0126437
9: -0.0052816, 0.0078093, -0.0056970, 0.0079456, -0.0132271, 0.0135063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0106652
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0108713
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0067275, 0.0050784, -0.0062031, 0.0043918, -0.0111193, 0.0112815
1: 0.9970307, 1.0103734, 0.9967059, 1.0093323, -0.0110474, 0.0123597
2: -0.0060587, 0.0058293, -0.0053768, 0.0052862, -0.0113449, 0.0112061
3: -0.0001690, 0.0025753, -0.0000222, 0.0027100, -0.0025348, 0.0022629
4: -0.0070006, 0.0015872, -0.0063104, 0.0015154, -0.0085160, 0.0078976
5: -0.0022843, 0.0083205, -0.0020455, 0.0074568, -0.0097411, 0.0103659
6: -0.0091795, 0.0019688, -0.0081132, 0.0018844, -0.0110639, 0.0100820
7: -0.0058715, 0.0001265, -0.0060989, -0.0002539, -0.0056176, 0.0062254
8: -0.0135659, -0.0014964, -0.0130342, -0.0007935, -0.0127724, 0.0115378
9: -0.0047087, 0.0076214, -0.0037577, 0.0073095, -0.0120183, 0.0113791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094234, upper bound: 0.0094471
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093560, upper bound: 0.0092442
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0068281, 0.0052101, -0.0063964, 0.0046448, -0.0114730, 0.0116064
1: 0.9970292, 1.0105731, 0.9967223, 1.0097158, -0.0114048, 0.0125383
2: -0.0061895, 0.0059335, -0.0056281, 0.0054864, -0.0116759, 0.0115616
3: -0.0001972, 0.0025760, -0.0000763, 0.0027032, -0.0025484, 0.0023046
4: -0.0071331, 0.0016214, -0.0065648, 0.0015078, -0.0086409, 0.0081861
5: -0.0023301, 0.0084862, -0.0021335, 0.0077751, -0.0101052, 0.0106196
6: -0.0093841, 0.0019850, -0.0085062, 0.0019155, -0.0112996, 0.0104912
7: -0.0058727, 0.0001995, -0.0060874, -0.0001137, -0.0057590, 0.0062869
8: -0.0136680, -0.0014929, -0.0132302, -0.0008290, -0.0128389, 0.0117373
9: -0.0048912, 0.0076813, -0.0041082, 0.0074245, -0.0123157, 0.0117895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098032, upper bound: 0.0096325
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097520, upper bound: 0.0094301
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0070381, 0.0054850, -0.0063474, 0.0045808, -0.0116189, 0.0118324
1: 0.9970281, 1.0109901, 0.9967040, 1.0096186, -0.0113902, 0.0128389
2: -0.0064625, 0.0061510, -0.0055645, 0.0054357, -0.0118982, 0.0117154
3: -0.0002559, 0.0025764, -0.0000626, 0.0027108, -0.0025623, 0.0023293
4: -0.0074094, 0.0016926, -0.0065003, 0.0015163, -0.0089257, 0.0081929
5: -0.0024258, 0.0088319, -0.0021112, 0.0076945, -0.0101203, 0.0109431
6: -0.0098109, 0.0020187, -0.0084067, 0.0019076, -0.0117185, 0.0104255
7: -0.0058734, 0.0003518, -0.0061002, -0.0001492, -0.0057242, 0.0064520
8: -0.0138808, -0.0014906, -0.0131806, -0.0007894, -0.0130914, 0.0116900
9: -0.0052720, 0.0078062, -0.0040194, 0.0073954, -0.0126673, 0.0118256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095964, upper bound: 0.0098744
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095721, upper bound: 0.0097886
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0071344, 0.0056110, -0.0065409, 0.0048341, -0.0119685, 0.0121519
1: 0.9970264, 1.0111812, 0.9967203, 1.0100030, -0.0117383, 0.0130045
2: -0.0065877, 0.0062507, -0.0058161, 0.0056361, -0.0122238, 0.0120668
3: -0.0002829, 0.0025771, -0.0001167, 0.0027040, -0.0025752, 0.0023660
4: -0.0075362, 0.0017252, -0.0067550, 0.0015239, -0.0090601, 0.0084803
5: -0.0024696, 0.0089905, -0.0021993, 0.0080132, -0.0104828, 0.0111898
6: -0.0100067, 0.0020342, -0.0088001, 0.0019388, -0.0119454, 0.0108344
7: -0.0058745, 0.0004216, -0.0060888, -0.0000089, -0.0058657, 0.0065105
8: -0.0139784, -0.0014871, -0.0133768, -0.0008247, -0.0131537, 0.0118896
9: -0.0054466, 0.0078634, -0.0043703, 0.0075105, -0.0129570, 0.0122338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100823, upper bound: 0.0101361
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100618, upper bound: 0.0100309
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0069109, 0.0053184, -0.0064817, 0.0047566, -0.0116675, 0.0118001
1: 0.9970263, 1.0107375, 0.9969186, 1.0098853, -0.0115786, 0.0126803
2: -0.0062971, 0.0060192, -0.0057391, 0.0055748, -0.0118719, 0.0117583
3: -0.0002203, 0.0025771, -0.0001002, 0.0026218, -0.0025817, 0.0023453
4: -0.0072420, 0.0016494, -0.0066772, 0.0015039, -0.0087458, 0.0083266
5: -0.0023678, 0.0086224, -0.0021724, 0.0079157, -0.0102835, 0.0107948
6: -0.0095522, 0.0019983, -0.0086798, 0.0019292, -0.0114815, 0.0106781
7: -0.0058747, 0.0002595, -0.0059500, -0.0000518, -0.0058229, 0.0062095
8: -0.0137518, -0.0014867, -0.0133168, -0.0012539, -0.0124979, 0.0118301
9: -0.0050412, 0.0077305, -0.0042630, 0.0074753, -0.0125165, 0.0119935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099843
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099326
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0070112, 0.0054498, -0.0066799, 0.0050160, -0.0120273, 0.0121297
1: 0.9970247, 1.0109369, 0.9969423, 1.0102788, -0.0119341, 0.0128583
2: -0.0064276, 0.0061232, -0.0059968, 0.0057800, -0.0122076, 0.0121199
3: -0.0002484, 0.0025778, -0.0001557, 0.0026120, -0.0025992, 0.0023884
4: -0.0073740, 0.0016835, -0.0069380, 0.0015711, -0.0089451, 0.0086214
5: -0.0024135, 0.0087877, -0.0022626, 0.0082420, -0.0106556, 0.0110503
6: -0.0097563, 0.0020144, -0.0090827, 0.0019611, -0.0117174, 0.0110971
7: -0.0058758, 0.0003323, -0.0059335, 0.0000920, -0.0059678, 0.0062658
8: -0.0138536, -0.0014832, -0.0135177, -0.0013049, -0.0125487, 0.0120345
9: -0.0052232, 0.0077902, -0.0046224, 0.0075931, -0.0128163, 0.0124125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0101572
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0103181
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0073630, 0.0059103, -0.0069572, 0.0053791, -0.0127421, 0.0128675
1: 0.9970213, 1.0116352, 0.9970217, 1.0108296, -0.0126283, 0.0133593
2: -0.0068850, 0.0064875, -0.0063574, 0.0060672, -0.0129522, 0.0128448
3: -0.0003469, 0.0025792, -0.0002333, 0.0025790, -0.0026204, 0.0025185
4: -0.0078371, 0.0018028, -0.0073030, 0.0016651, -0.0095022, 0.0091058
5: -0.0025737, 0.0093670, -0.0023889, 0.0086987, -0.0112725, 0.0117559
6: -0.0104715, 0.0020710, -0.0096465, 0.0020057, -0.0124772, 0.0117175
7: -0.0058781, 0.0005875, -0.0058778, 0.0002931, -0.0061712, 0.0064653
8: -0.0142102, -0.0014761, -0.0137988, -0.0014769, -0.0127333, 0.0123227
9: -0.0058612, 0.0079994, -0.0051253, 0.0077581, -0.0136192, 0.0131247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0106384
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0108551
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0073630, 0.0059103, -0.0072725, 0.0057918, -0.0131548, 0.0131828
1: 0.9970213, 1.0116352, 0.9970223, 1.0114555, -0.0131628, 0.0132468
2: -0.0068850, 0.0064875, -0.0067673, 0.0063937, -0.0132787, 0.0132547
3: -0.0003469, 0.0025792, -0.0003215, 0.0025787, -0.0025598, 0.0025647
4: -0.0078371, 0.0018028, -0.0077179, 0.0017721, -0.0096091, 0.0095207
5: -0.0025737, 0.0093670, -0.0025325, 0.0092179, -0.0117917, 0.0118995
6: -0.0104715, 0.0020710, -0.0102874, 0.0020565, -0.0125279, 0.0123585
7: -0.0058781, 0.0005875, -0.0058773, 0.0005218, -0.0063999, 0.0064648
8: -0.0142102, -0.0014761, -0.0141184, -0.0014784, -0.0127318, 0.0126423
9: -0.0058612, 0.0079994, -0.0056970, 0.0079456, -0.0138067, 0.0136964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0106384
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0108551
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.45 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0101304, upper bound: 0.0098304
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103258, upper bound: 0.0101147
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0101304, upper bound: 0.0098304
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103258, upper bound: 0.0101147
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103061, upper bound: 0.0100923
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105330, upper bound: 0.0105112
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103061, upper bound: 0.0100923
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105330, upper bound: 0.0105112
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0102921
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0104470
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0102921
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0103894, upper bound: 0.0104470
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0106652
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0108713
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0106652
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105997, upper bound: 0.0108713
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0094234, upper bound: 0.0094471
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0093560, upper bound: 0.0092442
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0098032, upper bound: 0.0096325
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0097520, upper bound: 0.0094301
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0095964, upper bound: 0.0098744
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0095721, upper bound: 0.0097886
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0100823, upper bound: 0.0101361
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0100618, upper bound: 0.0100309
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099843
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099326
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0101572
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0103181
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0106384
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0108551
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0106384
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 1, lower bound: -0.0105930, upper bound: 0.0108551

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0062271, 0.0044232, -0.0063840, 0.0046286, -0.0108557, 0.0108072
1: 0.9970112, 1.0093795, 0.9968031, 1.0096912, -0.0114003, 0.0111727
2: -0.0054080, 0.0053110, -0.0056120, 0.0054735, -0.0108815, 0.0109230
3: -0.0000289, 0.0025834, -0.0000728, 0.0026698, -0.0023137, 0.0023018
4: -0.0063419, 0.0014175, -0.0065485, 0.0014707, -0.0078127, 0.0079659
5: -0.0020564, 0.0074963, -0.0021278, 0.0077547, -0.0098111, 0.0096242
6: -0.0081620, 0.0018883, -0.0084810, 0.0019135, -0.0100755, 0.0103693
7: -0.0058852, -0.0002365, -0.0060310, -0.0001227, -0.0057625, 0.0057944
8: -0.0130586, -0.0014541, -0.0132176, -0.0010035, -0.0120551, 0.0117635
9: -0.0038012, 0.0073238, -0.0040857, 0.0074171, -0.0112183, 0.0114095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097501, upper bound: 0.0092258
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096884, upper bound: 0.0092096
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0064135, 0.0046673, -0.0064863, 0.0047626, -0.0111762, 0.0111537
1: 0.9970232, 1.0097501, 0.9968011, 1.0098943, -0.0115849, 0.0115203
2: -0.0056504, 0.0055042, -0.0057451, 0.0055796, -0.0112300, 0.0112492
3: -0.0000811, 0.0025784, -0.0001015, 0.0026705, -0.0023500, 0.0023226
4: -0.0065874, 0.0014807, -0.0066832, 0.0015054, -0.0080928, 0.0081639
5: -0.0021413, 0.0078034, -0.0021745, 0.0079233, -0.0100646, 0.0099779
6: -0.0085411, 0.0019183, -0.0086891, 0.0019300, -0.0104711, 0.0106074
7: -0.0058769, -0.0001013, -0.0060321, -0.0000484, -0.0058284, 0.0059309
8: -0.0132476, -0.0014799, -0.0133214, -0.0009999, -0.0122477, 0.0118415
9: -0.0041393, 0.0074347, -0.0042713, 0.0074780, -0.0116173, 0.0117060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099034, upper bound: 0.0095561
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098442, upper bound: 0.0095484
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0062271, 0.0044232, -0.0067097, 0.0050551, -0.0112822, 0.0111330
1: 0.9970112, 1.0093795, 0.9968103, 1.0103381, -0.0120867, 0.0112070
2: -0.0054080, 0.0053110, -0.0060356, 0.0058109, -0.0112189, 0.0113466
3: -0.0000289, 0.0025834, -0.0001640, 0.0026667, -0.0023252, 0.0024283
4: -0.0063419, 0.0014175, -0.0069772, 0.0015812, -0.0079231, 0.0083947
5: -0.0020564, 0.0074963, -0.0022762, 0.0082912, -0.0103476, 0.0097725
6: -0.0081620, 0.0018883, -0.0091433, 0.0019659, -0.0101279, 0.0110316
7: -0.0058852, -0.0002365, -0.0060258, 0.0001136, -0.0059988, 0.0057893
8: -0.0130586, -0.0014541, -0.0135479, -0.0010195, -0.0120391, 0.0120938
9: -0.0038012, 0.0073238, -0.0046765, 0.0076109, -0.0114120, 0.0120003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096322, upper bound: 0.0091494
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095598, upper bound: 0.0091279
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0064135, 0.0046673, -0.0068062, 0.0051814, -0.0115949, 0.0114736
1: 0.9970232, 1.0097501, 0.9968088, 1.0105298, -0.0122592, 0.0115782
2: -0.0056504, 0.0055042, -0.0061610, 0.0059109, -0.0115613, 0.0116652
3: -0.0000811, 0.0025784, -0.0001910, 0.0026674, -0.0023755, 0.0024448
4: -0.0065874, 0.0014807, -0.0071042, 0.0016139, -0.0082013, 0.0085850
5: -0.0021413, 0.0078034, -0.0023202, 0.0084501, -0.0105914, 0.0101236
6: -0.0085411, 0.0019183, -0.0093395, 0.0019814, -0.0105226, 0.0112578
7: -0.0058769, -0.0001013, -0.0060270, 0.0001836, -0.0060605, 0.0059257
8: -0.0132476, -0.0014799, -0.0136457, -0.0010159, -0.0122317, 0.0121658
9: -0.0041393, 0.0074347, -0.0048515, 0.0076682, -0.0118076, 0.0122862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098181, upper bound: 0.0094853
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097516, upper bound: 0.0094660
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0065131, 0.0047977, -0.0065257, 0.0048142, -0.0113273, 0.0113235
1: 0.9970096, 1.0099478, 0.9968010, 1.0099727, -0.0117348, 0.0116347
2: -0.0057799, 0.0056073, -0.0057963, 0.0056203, -0.0114003, 0.0114036
3: -0.0001090, 0.0025841, -0.0001125, 0.0026706, -0.0023396, 0.0023649
4: -0.0067185, 0.0015145, -0.0067350, 0.0015188, -0.0082373, 0.0082495
5: -0.0021867, 0.0079674, -0.0021924, 0.0079881, -0.0101748, 0.0101598
6: -0.0087436, 0.0019343, -0.0087692, 0.0019363, -0.0106799, 0.0107035
7: -0.0058864, -0.0000290, -0.0060324, -0.0000199, -0.0058665, 0.0060034
8: -0.0133486, -0.0014505, -0.0133613, -0.0009992, -0.0123494, 0.0119109
9: -0.0043200, 0.0074939, -0.0043427, 0.0075014, -0.0118214, 0.0118367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100712, upper bound: 0.0098056
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100052, upper bound: 0.0097951
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0067062, 0.0050504, -0.0066300, 0.0049506, -0.0116568, 0.0116804
1: 0.9970274, 1.0103310, 0.9967993, 1.0101796, -0.0119214, 0.0119882
2: -0.0060309, 0.0058072, -0.0059318, 0.0057283, -0.0117592, 0.0117390
3: -0.0001630, 0.0025766, -0.0001417, 0.0026713, -0.0023841, 0.0023884
4: -0.0069725, 0.0015800, -0.0068722, 0.0015541, -0.0085266, 0.0084522
5: -0.0022746, 0.0082853, -0.0022399, 0.0081598, -0.0104344, 0.0105251
6: -0.0091360, 0.0019653, -0.0089811, 0.0019531, -0.0110891, 0.0109465
7: -0.0058738, 0.0001110, -0.0060335, 0.0000557, -0.0059296, 0.0061446
8: -0.0135443, -0.0014892, -0.0134670, -0.0009956, -0.0125487, 0.0119778
9: -0.0046700, 0.0076087, -0.0045318, 0.0075634, -0.0122334, 0.0121405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102559, upper bound: 0.0101547
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101814, upper bound: 0.0101453
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0065131, 0.0047977, -0.0068550, 0.0052453, -0.0117585, 0.0116528
1: 0.9970096, 1.0099478, 0.9968081, 1.0106268, -0.0124273, 0.0116812
2: -0.0057799, 0.0056073, -0.0062245, 0.0059614, -0.0117414, 0.0118318
3: -0.0001090, 0.0025841, -0.0002047, 0.0026676, -0.0023539, 0.0024924
4: -0.0067185, 0.0015145, -0.0071685, 0.0016305, -0.0083490, 0.0086830
5: -0.0021867, 0.0079674, -0.0023424, 0.0085305, -0.0107171, 0.0103098
6: -0.0087436, 0.0019343, -0.0094387, 0.0019893, -0.0107329, 0.0113730
7: -0.0058864, -0.0000290, -0.0060273, 0.0002190, -0.0061054, 0.0059983
8: -0.0133486, -0.0014505, -0.0136952, -0.0010150, -0.0123336, 0.0122448
9: -0.0043200, 0.0074939, -0.0049400, 0.0076973, -0.0120173, 0.0124339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099903, upper bound: 0.0099852
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099903, upper bound: 0.0100923
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0067062, 0.0050504, -0.0069514, 0.0053715, -0.0120777, 0.0120018
1: 0.9970274, 1.0103310, 0.9968067, 1.0108180, -0.0125989, 0.0120477
2: -0.0060309, 0.0058072, -0.0063498, 0.0060612, -0.0120921, 0.0121570
3: -0.0001630, 0.0025766, -0.0002317, 0.0026682, -0.0024079, 0.0025127
4: -0.0069725, 0.0015800, -0.0072954, 0.0016632, -0.0086357, 0.0088753
5: -0.0022746, 0.0082853, -0.0023863, 0.0086892, -0.0109638, 0.0106716
6: -0.0091360, 0.0019653, -0.0096347, 0.0020048, -0.0111409, 0.0116001
7: -0.0058738, 0.0001110, -0.0060284, 0.0002889, -0.0061628, 0.0061394
8: -0.0135443, -0.0014892, -0.0137929, -0.0010115, -0.0125328, 0.0123037
9: -0.0046700, 0.0076087, -0.0051148, 0.0077546, -0.0124246, 0.0127235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101534, upper bound: 0.0102974
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101534, upper bound: 0.0105112
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0062771, 0.0044887, -0.0068164, 0.0051947, -0.0114718, 0.0113051
1: 0.9967967, 1.0094790, 0.9970237, 1.0105498, -0.0124051, 0.0112145
2: -0.0054731, 0.0053629, -0.0061742, 0.0059213, -0.0113944, 0.0115371
3: -0.0000429, 0.0026723, -0.0001939, 0.0025782, -0.0022877, 0.0025123
4: -0.0064078, 0.0014736, -0.0071176, 0.0016174, -0.0080252, 0.0085911
5: -0.0020792, 0.0075787, -0.0023248, 0.0084668, -0.0105459, 0.0099035
6: -0.0082638, 0.0018963, -0.0093601, 0.0019831, -0.0102468, 0.0112564
7: -0.0060353, -0.0002002, -0.0058765, 0.0001909, -0.0062263, 0.0056763
8: -0.0131093, -0.0009900, -0.0136560, -0.0014810, -0.0116283, 0.0126660
9: -0.0038919, 0.0073536, -0.0048698, 0.0076743, -0.0115662, 0.0122234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100674, upper bound: 0.0101166
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103405, upper bound: 0.0102191
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0065564, 0.0048543, -0.0068164, 0.0051947, -0.0117510, 0.0116707
1: 0.9970198, 1.0100336, 0.9970237, 1.0105498, -0.0122340, 0.0117327
2: -0.0058362, 0.0056521, -0.0061742, 0.0059213, -0.0117575, 0.0118263
3: -0.0001211, 0.0025799, -0.0001939, 0.0025782, -0.0023628, 0.0024299
4: -0.0067754, 0.0015292, -0.0071176, 0.0016174, -0.0083927, 0.0086467
5: -0.0022064, 0.0080386, -0.0023248, 0.0084668, -0.0106731, 0.0103634
6: -0.0088315, 0.0019412, -0.0093601, 0.0019831, -0.0108146, 0.0113013
7: -0.0058793, 0.0000024, -0.0058765, 0.0001909, -0.0060702, 0.0058789
8: -0.0133924, -0.0014725, -0.0136560, -0.0014810, -0.0119114, 0.0121835
9: -0.0043984, 0.0075196, -0.0048698, 0.0076743, -0.0120726, 0.0123895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100674, upper bound: 0.0102413
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103405, upper bound: 0.0104039
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0062771, 0.0044887, -0.0071214, 0.0055940, -0.0118711, 0.0116101
1: 0.9967967, 1.0094790, 0.9970244, 1.0111555, -0.0130775, 0.0112446
2: -0.0054731, 0.0053629, -0.0065708, 0.0062372, -0.0117103, 0.0119337
3: -0.0000429, 0.0026723, -0.0002792, 0.0025779, -0.0022998, 0.0026347
4: -0.0064078, 0.0014736, -0.0075190, 0.0017208, -0.0081287, 0.0089926
5: -0.0020792, 0.0075787, -0.0024637, 0.0089691, -0.0110482, 0.0100424
6: -0.0082638, 0.0018963, -0.0099802, 0.0020321, -0.0102959, 0.0118765
7: -0.0060353, -0.0002002, -0.0058760, 0.0004122, -0.0064475, 0.0056758
8: -0.0131093, -0.0009900, -0.0139652, -0.0014825, -0.0116268, 0.0129752
9: -0.0038919, 0.0073536, -0.0054230, 0.0078557, -0.0117476, 0.0127765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099245, upper bound: 0.0100008
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102801, upper bound: 0.0101655
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0065564, 0.0048543, -0.0071214, 0.0055940, -0.0121504, 0.0119757
1: 0.9970198, 1.0100336, 0.9970244, 1.0111555, -0.0129157, 0.0117690
2: -0.0058362, 0.0056521, -0.0065708, 0.0062372, -0.0120734, 0.0122229
3: -0.0001211, 0.0025799, -0.0002792, 0.0025779, -0.0023786, 0.0025533
4: -0.0067754, 0.0015292, -0.0075190, 0.0017208, -0.0084962, 0.0090482
5: -0.0022064, 0.0080386, -0.0024637, 0.0089691, -0.0111754, 0.0105023
6: -0.0088315, 0.0019412, -0.0099802, 0.0020321, -0.0108637, 0.0119214
7: -0.0058793, 0.0000024, -0.0058760, 0.0004122, -0.0062915, 0.0058784
8: -0.0133924, -0.0014725, -0.0139652, -0.0014825, -0.0119099, 0.0124928
9: -0.0043984, 0.0075196, -0.0054230, 0.0078557, -0.0122540, 0.0129426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099245, upper bound: 0.0099554
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102801, upper bound: 0.0103329
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0065581, 0.0048566, -0.0069572, 0.0053791, -0.0119372, 0.0118139
1: 0.9968008, 1.0100369, 0.9970217, 1.0108296, -0.0127574, 0.0116484
2: -0.0058384, 0.0056539, -0.0063574, 0.0060672, -0.0119057, 0.0120113
3: -0.0001216, 0.0026706, -0.0002333, 0.0025790, -0.0023124, 0.0025817
4: -0.0067777, 0.0015298, -0.0073030, 0.0016651, -0.0084428, 0.0088327
5: -0.0022072, 0.0080415, -0.0023889, 0.0086987, -0.0109059, 0.0104304
6: -0.0088351, 0.0019415, -0.0096465, 0.0020057, -0.0108409, 0.0115880
7: -0.0060324, 0.0000036, -0.0058778, 0.0002931, -0.0063256, 0.0058815
8: -0.0133942, -0.0009990, -0.0137988, -0.0014769, -0.0119173, 0.0127998
9: -0.0044016, 0.0075207, -0.0051253, 0.0077581, -0.0121596, 0.0126460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102320, upper bound: 0.0104190
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105787, upper bound: 0.0106259
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0068341, 0.0052179, -0.0069572, 0.0053791, -0.0122132, 0.0121752
1: 0.9970241, 1.0105852, 0.9970217, 1.0108296, -0.0125890, 0.0121734
2: -0.0061973, 0.0059397, -0.0063574, 0.0060672, -0.0122645, 0.0122971
3: -0.0001988, 0.0025780, -0.0002333, 0.0025790, -0.0023881, 0.0025000
4: -0.0071409, 0.0016234, -0.0073030, 0.0016651, -0.0088061, 0.0089264
5: -0.0023329, 0.0084960, -0.0023889, 0.0086987, -0.0110316, 0.0108849
6: -0.0093962, 0.0019859, -0.0096465, 0.0020057, -0.0114019, 0.0116324
7: -0.0058762, 0.0002038, -0.0058778, 0.0002931, -0.0061693, 0.0060817
8: -0.0136740, -0.0014820, -0.0137988, -0.0014769, -0.0121971, 0.0123168
9: -0.0049020, 0.0076848, -0.0051253, 0.0077581, -0.0126601, 0.0128101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102320, upper bound: 0.0105689
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105787, upper bound: 0.0108291
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0065581, 0.0048566, -0.0072725, 0.0057918, -0.0123499, 0.0121291
1: 0.9968008, 1.0100369, 0.9970223, 1.0114555, -0.0134390, 0.0116779
2: -0.0058384, 0.0056539, -0.0067673, 0.0063937, -0.0122322, 0.0124212
3: -0.0001216, 0.0026706, -0.0003215, 0.0025787, -0.0023202, 0.0027017
4: -0.0067777, 0.0015298, -0.0077179, 0.0017721, -0.0085498, 0.0092477
5: -0.0022072, 0.0080415, -0.0025325, 0.0092179, -0.0114251, 0.0105740
6: -0.0088351, 0.0019415, -0.0102874, 0.0020565, -0.0108916, 0.0122289
7: -0.0060324, 0.0000036, -0.0058773, 0.0005218, -0.0065542, 0.0058810
8: -0.0133942, -0.0009990, -0.0141184, -0.0014784, -0.0119158, 0.0131194
9: -0.0044016, 0.0075207, -0.0056970, 0.0079456, -0.0123471, 0.0132177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100250, upper bound: 0.0103230
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104908, upper bound: 0.0105545
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0068341, 0.0052179, -0.0072725, 0.0057918, -0.0126259, 0.0124904
1: 0.9970241, 1.0105852, 0.9970223, 1.0114555, -0.0132787, 0.0122061
2: -0.0061973, 0.0059397, -0.0067673, 0.0063937, -0.0125910, 0.0127070
3: -0.0001988, 0.0025780, -0.0003215, 0.0025787, -0.0024008, 0.0026228
4: -0.0071409, 0.0016234, -0.0077179, 0.0017721, -0.0089130, 0.0093413
5: -0.0023329, 0.0084960, -0.0025325, 0.0092179, -0.0115508, 0.0110285
6: -0.0093962, 0.0019859, -0.0102874, 0.0020565, -0.0114527, 0.0122734
7: -0.0058762, 0.0002038, -0.0058773, 0.0005218, -0.0063980, 0.0060812
8: -0.0136740, -0.0014820, -0.0141184, -0.0014784, -0.0121956, 0.0126364
9: -0.0049020, 0.0076848, -0.0056970, 0.0079456, -0.0128476, 0.0133819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100250, upper bound: 0.0104619
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104908, upper bound: 0.0107607
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0065776, 0.0048821, -0.0061830, 0.0043655, -0.0109431, 0.0110651
1: 0.9970360, 1.0100757, 0.9967067, 1.0092920, -0.0109997, 0.0120632
2: -0.0058638, 0.0056741, -0.0053506, 0.0052654, -0.0111291, 0.0110247
3: -0.0001270, 0.0025731, -0.0000165, 0.0027097, -0.0024945, 0.0022519
4: -0.0068033, 0.0015364, -0.0062839, 0.0015150, -0.0083184, 0.0078203
5: -0.0022160, 0.0080736, -0.0020363, 0.0074237, -0.0096397, 0.0101099
6: -0.0088747, 0.0019447, -0.0080724, 0.0018812, -0.0107559, 0.0100170
7: -0.0058679, 0.0000178, -0.0060983, -0.0002685, -0.0055993, 0.0061161
8: -0.0134140, -0.0015078, -0.0130139, -0.0007953, -0.0126187, 0.0115061
9: -0.0044369, 0.0075323, -0.0037212, 0.0072976, -0.0117345, 0.0112535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094234, upper bound: 0.0094471
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094234, upper bound: 0.0094471
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063851, 0.0046301, -0.0060432, 0.0041826, -0.0105677, 0.0106734
1: 0.9969429, 1.0096934, 0.9967113, 1.0090148, -0.0108355, 0.0117450
2: -0.0056135, 0.0054747, -0.0051689, 0.0051207, -0.0107341, 0.0106437
3: -0.0000731, 0.0026117, 0.0000226, 0.0027078, -0.0024686, 0.0022512
4: -0.0065500, 0.0014711, -0.0061000, 0.0015129, -0.0080628, 0.0075711
5: -0.0021284, 0.0077566, -0.0019727, 0.0071936, -0.0093219, 0.0097292
6: -0.0084833, 0.0019137, -0.0077883, 0.0018587, -0.0103420, 0.0097020
7: -0.0059330, -0.0001219, -0.0060951, -0.0003699, -0.0055631, 0.0059732
8: -0.0132188, -0.0013064, -0.0128722, -0.0008053, -0.0124135, 0.0115658
9: -0.0040878, 0.0074178, -0.0034678, 0.0072145, -0.0113022, 0.0108856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093560, upper bound: 0.0092442
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093560, upper bound: 0.0092442
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0066730, 0.0050071, -0.0063743, 0.0046160, -0.0112890, 0.0113814
1: 0.9970345, 1.0102651, 0.9967231, 1.0096722, -0.0113558, 0.0122408
2: -0.0059879, 0.0057729, -0.0055994, 0.0054635, -0.0114514, 0.0113723
3: -0.0001537, 0.0025737, -0.0000701, 0.0027029, -0.0025086, 0.0022935
4: -0.0069290, 0.0015688, -0.0065358, 0.0015075, -0.0084364, 0.0081045
5: -0.0022595, 0.0082308, -0.0021234, 0.0077388, -0.0099983, 0.0103542
6: -0.0090688, 0.0019600, -0.0084614, 0.0019119, -0.0109807, 0.0104214
7: -0.0058690, 0.0000870, -0.0060869, -0.0001297, -0.0057393, 0.0061738
8: -0.0135107, -0.0015044, -0.0132078, -0.0008308, -0.0126799, 0.0117035
9: -0.0046100, 0.0075890, -0.0040682, 0.0074114, -0.0120213, 0.0116572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098032, upper bound: 0.0096325
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098032, upper bound: 0.0096325
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0064874, 0.0047640, -0.0062505, 0.0044539, -0.0109413, 0.0110145
1: 0.9969414, 1.0098969, 0.9967279, 1.0094261, -0.0112037, 0.0119213
2: -0.0057465, 0.0055807, -0.0054384, 0.0053353, -0.0110818, 0.0110191
3: -0.0001018, 0.0026123, -0.0000354, 0.0027009, -0.0024823, 0.0022944
4: -0.0066846, 0.0015058, -0.0063728, 0.0015053, -0.0081899, 0.0078785
5: -0.0021750, 0.0079250, -0.0020670, 0.0075349, -0.0097099, 0.0099921
6: -0.0086913, 0.0019301, -0.0082097, 0.0018920, -0.0105833, 0.0101398
7: -0.0059341, -0.0000477, -0.0060835, -0.0002195, -0.0057145, 0.0060359
8: -0.0133225, -0.0013031, -0.0130823, -0.0008410, -0.0124815, 0.0117793
9: -0.0042733, 0.0074786, -0.0038437, 0.0073377, -0.0116110, 0.0113223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097520, upper bound: 0.0094301
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097520, upper bound: 0.0094301
time: 2.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0068833, 0.0052823, -0.0063264, 0.0045533, -0.0114366, 0.0116087
1: 0.9970333, 1.0106827, 0.9967048, 1.0095770, -0.0113424, 0.0125334
2: -0.0062612, 0.0059907, -0.0055372, 0.0054139, -0.0116752, 0.0115278
3: -0.0002126, 0.0025742, -0.0000567, 0.0027105, -0.0025218, 0.0023192
4: -0.0072057, 0.0016401, -0.0064727, 0.0015159, -0.0087216, 0.0081128
5: -0.0023553, 0.0085770, -0.0021016, 0.0076599, -0.0100152, 0.0106786
6: -0.0094962, 0.0019938, -0.0083640, 0.0019042, -0.0114004, 0.0103579
7: -0.0058698, 0.0002395, -0.0060997, -0.0001644, -0.0057053, 0.0063392
8: -0.0137239, -0.0015018, -0.0131593, -0.0007912, -0.0129327, 0.0116574
9: -0.0049912, 0.0077141, -0.0039813, 0.0073829, -0.0123741, 0.0116954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095964, upper bound: 0.0098744
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095964, upper bound: 0.0098744
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0066970, 0.0050384, -0.0061967, 0.0043834, -0.0110804, 0.0112351
1: 0.9969380, 1.0103128, 0.9967094, 1.0093193, -0.0111975, 0.0122333
2: -0.0060190, 0.0057977, -0.0053685, 0.0052796, -0.0112986, 0.0111662
3: -0.0001604, 0.0026138, -0.0000204, 0.0027085, -0.0024964, 0.0023199
4: -0.0069605, 0.0015769, -0.0063019, 0.0015137, -0.0084742, 0.0078788
5: -0.0022704, 0.0082702, -0.0020425, 0.0074462, -0.0097166, 0.0103127
6: -0.0091175, 0.0019639, -0.0081002, 0.0018834, -0.0110008, 0.0100641
7: -0.0059365, 0.0001044, -0.0060964, -0.0002586, -0.0056779, 0.0062008
8: -0.0135350, -0.0012956, -0.0130278, -0.0008013, -0.0127337, 0.0117321
9: -0.0046534, 0.0076033, -0.0037460, 0.0073057, -0.0119591, 0.0113493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095721, upper bound: 0.0097886
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095721, upper bound: 0.0097886
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0069784, 0.0054068, -0.0065186, 0.0048049, -0.0117833, 0.0119254
1: 0.9970319, 1.0108715, 0.9967213, 1.0099585, -0.0116885, 0.0126994
2: -0.0063849, 0.0060892, -0.0057870, 0.0056130, -0.0119979, 0.0118762
3: -0.0002392, 0.0025748, -0.0001105, 0.0027037, -0.0025352, 0.0023551
4: -0.0073309, 0.0016723, -0.0067256, 0.0015163, -0.0088472, 0.0083980
5: -0.0023986, 0.0087336, -0.0021892, 0.0079764, -0.0103750, 0.0109228
6: -0.0096896, 0.0020091, -0.0087547, 0.0019352, -0.0116247, 0.0107639
7: -0.0058708, 0.0003085, -0.0060882, -0.0000251, -0.0058458, 0.0063967
8: -0.0138203, -0.0014986, -0.0133541, -0.0008265, -0.0129938, 0.0118555
9: -0.0051637, 0.0077707, -0.0043298, 0.0074972, -0.0126609, 0.0121005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100823, upper bound: 0.0101361
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100823, upper bound: 0.0101361
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0067947, 0.0051663, -0.0063977, 0.0046466, -0.0114412, 0.0115639
1: 0.9969366, 1.0105067, 0.9967260, 1.0097184, -0.0115445, 0.0124008
2: -0.0061460, 0.0058989, -0.0056298, 0.0054877, -0.0116337, 0.0115287
3: -0.0001878, 0.0026144, -0.0000766, 0.0027017, -0.0025092, 0.0023565
4: -0.0070890, 0.0016100, -0.0065665, 0.0015062, -0.0085952, 0.0081765
5: -0.0023149, 0.0084310, -0.0021341, 0.0077773, -0.0100922, 0.0105651
6: -0.0093160, 0.0019796, -0.0085089, 0.0019157, -0.0112317, 0.0104884
7: -0.0059375, 0.0001752, -0.0060849, -0.0001128, -0.0058248, 0.0062601
8: -0.0136340, -0.0012924, -0.0132315, -0.0008369, -0.0127971, 0.0119392
9: -0.0048305, 0.0076614, -0.0041106, 0.0074253, -0.0122558, 0.0117719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100618, upper bound: 0.0100309
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100618, upper bound: 0.0100309
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0064375, 0.0046987, -0.0064817, 0.0047566, -0.0111941, 0.0111805
1: 0.9968101, 1.0097977, 0.9969186, 1.0098853, -0.0117574, 0.0116695
2: -0.0056816, 0.0055290, -0.0057391, 0.0055748, -0.0112564, 0.0112681
3: -0.0000878, 0.0026668, -0.0001002, 0.0026218, -0.0023996, 0.0024271
4: -0.0066189, 0.0014889, -0.0066772, 0.0015039, -0.0081228, 0.0081660
5: -0.0021522, 0.0078429, -0.0021724, 0.0079157, -0.0100680, 0.0100153
6: -0.0085899, 0.0019221, -0.0086798, 0.0019292, -0.0105191, 0.0106019
7: -0.0060259, -0.0000839, -0.0059500, -0.0000518, -0.0059741, 0.0058661
8: -0.0132719, -0.0010193, -0.0133168, -0.0012539, -0.0120180, 0.0122975
9: -0.0041828, 0.0074490, -0.0042630, 0.0074753, -0.0116581, 0.0117120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099841
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099841
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0067051, 0.0050490, -0.0064817, 0.0047566, -0.0114617, 0.0115308
1: 0.9970300, 1.0103289, 0.9969186, 1.0098853, -0.0115758, 0.0121976
2: -0.0060296, 0.0058061, -0.0057391, 0.0055748, -0.0116044, 0.0115453
3: -0.0001627, 0.0025756, -0.0001002, 0.0026218, -0.0024819, 0.0023424
4: -0.0069712, 0.0015796, -0.0066772, 0.0015039, -0.0084750, 0.0082568
5: -0.0022741, 0.0082836, -0.0021724, 0.0079157, -0.0101898, 0.0104559
6: -0.0091339, 0.0019652, -0.0086798, 0.0019292, -0.0110632, 0.0106450
7: -0.0058721, 0.0001103, -0.0059500, -0.0000518, -0.0058203, 0.0060602
8: -0.0135432, -0.0014946, -0.0133168, -0.0012539, -0.0122893, 0.0118222
9: -0.0046681, 0.0076081, -0.0042630, 0.0074753, -0.0121434, 0.0118711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099326
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099326
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0065371, 0.0048290, -0.0066799, 0.0050160, -0.0115531, 0.0115089
1: 0.9968086, 1.0099952, 0.9969423, 1.0102788, -0.0121091, 0.0118496
2: -0.0058110, 0.0056321, -0.0059968, 0.0057800, -0.0115911, 0.0116289
3: -0.0001157, 0.0026674, -0.0001557, 0.0026120, -0.0024167, 0.0024702
4: -0.0067500, 0.0015226, -0.0069380, 0.0015711, -0.0083210, 0.0084606
5: -0.0021976, 0.0080068, -0.0022626, 0.0082420, -0.0104396, 0.0102694
6: -0.0087922, 0.0019381, -0.0090827, 0.0019611, -0.0107534, 0.0110208
7: -0.0060270, -0.0000117, -0.0059335, 0.0000920, -0.0061190, 0.0059218
8: -0.0133728, -0.0010157, -0.0135177, -0.0013049, -0.0120680, 0.0125019
9: -0.0043633, 0.0075082, -0.0046224, 0.0075931, -0.0119564, 0.0121305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0101522
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0101522
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0068028, 0.0051769, -0.0066799, 0.0050160, -0.0118188, 0.0118568
1: 0.9970282, 1.0105230, 0.9969423, 1.0102788, -0.0119311, 0.0123707
2: -0.0061565, 0.0059073, -0.0059968, 0.0057800, -0.0119366, 0.0119040
3: -0.0001901, 0.0025763, -0.0001557, 0.0026120, -0.0024987, 0.0023855
4: -0.0070997, 0.0016127, -0.0069380, 0.0015711, -0.0086708, 0.0085507
5: -0.0023186, 0.0084444, -0.0022626, 0.0082420, -0.0105606, 0.0107070
6: -0.0093325, 0.0019809, -0.0090827, 0.0019611, -0.0112936, 0.0110635
7: -0.0058733, 0.0001811, -0.0059335, 0.0000920, -0.0059652, 0.0061146
8: -0.0136422, -0.0014911, -0.0135177, -0.0013049, -0.0123373, 0.0120266
9: -0.0048452, 0.0076662, -0.0046224, 0.0075931, -0.0124383, 0.0122886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0103167
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0103167
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0068812, 0.0052795, -0.0069572, 0.0053791, -0.0122602, 0.0122367
1: 0.9968082, 1.0106784, 0.9970217, 1.0108296, -0.0127959, 0.0123288
2: -0.0062585, 0.0059885, -0.0063574, 0.0060672, -0.0123257, 0.0123458
3: -0.0002120, 0.0026676, -0.0002333, 0.0025790, -0.0024353, 0.0025958
4: -0.0072029, 0.0016393, -0.0073030, 0.0016651, -0.0088680, 0.0089423
5: -0.0023543, 0.0085735, -0.0023889, 0.0086987, -0.0110530, 0.0109624
6: -0.0094918, 0.0019935, -0.0096465, 0.0020057, -0.0114976, 0.0116400
7: -0.0060273, 0.0002379, -0.0058778, 0.0002931, -0.0063204, 0.0061158
8: -0.0137217, -0.0010148, -0.0137988, -0.0014769, -0.0122448, 0.0127840
9: -0.0049874, 0.0077128, -0.0051253, 0.0077581, -0.0127454, 0.0128381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100895, upper bound: 0.0103061
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105006, upper bound: 0.0105330
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0071515, 0.0056334, -0.0069572, 0.0053791, -0.0125305, 0.0125906
1: 0.9970248, 1.0112152, 0.9970217, 1.0108296, -0.0126252, 0.0128663
2: -0.0066099, 0.0062684, -0.0063574, 0.0060672, -0.0126772, 0.0126258
3: -0.0002877, 0.0025777, -0.0002333, 0.0025790, -0.0025134, 0.0025157
4: -0.0075586, 0.0017310, -0.0073030, 0.0016651, -0.0092238, 0.0090340
5: -0.0024774, 0.0090187, -0.0023889, 0.0086987, -0.0111761, 0.0114076
6: -0.0100414, 0.0020370, -0.0096465, 0.0020057, -0.0120472, 0.0116835
7: -0.0058757, 0.0004340, -0.0058778, 0.0002931, -0.0061688, 0.0063119
8: -0.0139958, -0.0014835, -0.0137988, -0.0014769, -0.0125188, 0.0123153
9: -0.0054776, 0.0078736, -0.0051253, 0.0077581, -0.0132356, 0.0129989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100895, upper bound: 0.0104561
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0105006, upper bound: 0.0104908
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0068812, 0.0052795, -0.0072725, 0.0057918, -0.0126729, 0.0125520
1: 0.9968082, 1.0106784, 0.9970223, 1.0114555, -0.0133222, 0.0122248
2: -0.0062585, 0.0059885, -0.0067673, 0.0063937, -0.0126522, 0.0127557
3: -0.0002120, 0.0026676, -0.0003215, 0.0025787, -0.0023720, 0.0026465
4: -0.0072029, 0.0016393, -0.0077179, 0.0017721, -0.0089749, 0.0093572
5: -0.0023543, 0.0085735, -0.0025325, 0.0092179, -0.0115722, 0.0111060
6: -0.0094918, 0.0019935, -0.0102874, 0.0020565, -0.0115483, 0.0122809
7: -0.0060273, 0.0002379, -0.0058773, 0.0005218, -0.0065491, 0.0061153
8: -0.0137217, -0.0010148, -0.0141184, -0.0014784, -0.0122433, 0.0131036
9: -0.0049874, 0.0077128, -0.0056970, 0.0079456, -0.0129329, 0.0134098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100206, upper bound: 0.0102939
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104840, upper bound: 0.0105300
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0071515, 0.0056334, -0.0072725, 0.0057918, -0.0129433, 0.0129058
1: 0.9970248, 1.0112152, 0.9970223, 1.0114555, -0.0131601, 0.0127501
2: -0.0066099, 0.0062684, -0.0067673, 0.0063937, -0.0130037, 0.0130357
3: -0.0002877, 0.0025777, -0.0003215, 0.0025787, -0.0024517, 0.0025619
4: -0.0075586, 0.0017310, -0.0077179, 0.0017721, -0.0093307, 0.0094489
5: -0.0024774, 0.0090187, -0.0025325, 0.0092179, -0.0116953, 0.0115512
6: -0.0100414, 0.0020370, -0.0102874, 0.0020565, -0.0120979, 0.0123244
7: -0.0058757, 0.0004340, -0.0058773, 0.0005218, -0.0063975, 0.0063114
8: -0.0139958, -0.0014835, -0.0141184, -0.0014784, -0.0125173, 0.0126349
9: -0.0054776, 0.0078736, -0.0056970, 0.0079456, -0.0134231, 0.0135706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100206, upper bound: 0.0104413
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104840, upper bound: 0.0107468
time: 1.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.80 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0097501, upper bound: 0.0092258
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0096884, upper bound: 0.0092096
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099034, upper bound: 0.0095561
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0098442, upper bound: 0.0095484
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0096322, upper bound: 0.0091494
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0095598, upper bound: 0.0091279
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0098181, upper bound: 0.0094853
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0097516, upper bound: 0.0094660
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100712, upper bound: 0.0098056
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100052, upper bound: 0.0097951
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102559, upper bound: 0.0101547
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0101814, upper bound: 0.0101453
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099903, upper bound: 0.0099852
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099903, upper bound: 0.0100923
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0101534, upper bound: 0.0102974
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0101534, upper bound: 0.0105112
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100674, upper bound: 0.0101166
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0103405, upper bound: 0.0102191
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100674, upper bound: 0.0102413
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0103405, upper bound: 0.0104039
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099245, upper bound: 0.0100008
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102801, upper bound: 0.0101655
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099245, upper bound: 0.0099554
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102801, upper bound: 0.0103329
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102320, upper bound: 0.0104190
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0105787, upper bound: 0.0106259
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102320, upper bound: 0.0105689
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0105787, upper bound: 0.0108291
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100250, upper bound: 0.0103230
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0104908, upper bound: 0.0105545
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100250, upper bound: 0.0104619
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0104908, upper bound: 0.0107607
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0094234, upper bound: 0.0094471
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0094234, upper bound: 0.0094471
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0093560, upper bound: 0.0092442
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0093560, upper bound: 0.0092442
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0098032, upper bound: 0.0096325
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0098032, upper bound: 0.0096325
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0097520, upper bound: 0.0094301
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0097520, upper bound: 0.0094301
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0095964, upper bound: 0.0098744
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0095964, upper bound: 0.0098744
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0095721, upper bound: 0.0097886
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0095721, upper bound: 0.0097886
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100823, upper bound: 0.0101361
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100823, upper bound: 0.0101361
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100618, upper bound: 0.0100309
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100618, upper bound: 0.0100309
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099841
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099841
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099326
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0099285, upper bound: 0.0099326
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0101522
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0101522
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0103167
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0102871, upper bound: 0.0103167
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100895, upper bound: 0.0103061
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0105006, upper bound: 0.0105330
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100895, upper bound: 0.0104561
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0105006, upper bound: 0.0104908
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100206, upper bound: 0.0102939
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0104840, upper bound: 0.0105300
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0100206, upper bound: 0.0104413
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 1, lower bound: -0.0104840, upper bound: 0.0107468

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0062067, 0.0043966, -0.0062438, 0.0044451, -0.0106518, 0.0106404
1: 0.9970120, 1.0093392, 0.9968088, 1.0094130, -0.0111121, 0.0111255
2: -0.0053815, 0.0052900, -0.0054297, 0.0053284, -0.0107099, 0.0107197
3: -0.0000232, 0.0025830, -0.0000336, 0.0026674, -0.0023031, 0.0022629
4: -0.0063152, 0.0014106, -0.0063640, 0.0014680, -0.0077832, 0.0077745
5: -0.0020471, 0.0074628, -0.0020640, 0.0075239, -0.0095710, 0.0095268
6: -0.0081207, 0.0018850, -0.0081960, 0.0018909, -0.0100116, 0.0100810
7: -0.0058846, -0.0002513, -0.0060269, -0.0002244, -0.0056603, 0.0057756
8: -0.0130380, -0.0014559, -0.0130755, -0.0010161, -0.0120218, 0.0116196
9: -0.0037643, 0.0073117, -0.0038315, 0.0073337, -0.0110980, 0.0111432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097089, upper bound: 0.0092208
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097089, upper bound: 0.0092258
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0060665, 0.0042131, -0.0060438, 0.0041833, -0.0102499, 0.0102569
1: 0.9970168, 1.0090610, 0.9967105, 1.0090160, -0.0107990, 0.0109527
2: -0.0051992, 0.0051448, -0.0051697, 0.0051213, -0.0103205, 0.0103145
3: 0.0000161, 0.0025811, 0.0000224, 0.0027081, -0.0023018, 0.0022359
4: -0.0061306, 0.0013723, -0.0061007, 0.0015133, -0.0076440, 0.0074730
5: -0.0019833, 0.0072319, -0.0019729, 0.0071945, -0.0091778, 0.0092048
6: -0.0078356, 0.0018624, -0.0077894, 0.0018588, -0.0096944, 0.0096518
7: -0.0058814, -0.0003530, -0.0060957, -0.0003694, -0.0055120, 0.0057428
8: -0.0128958, -0.0014659, -0.0128728, -0.0008033, -0.0120925, 0.0114069
9: -0.0035100, 0.0072283, -0.0034688, 0.0072148, -0.0107248, 0.0106971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096351, upper bound: 0.0092031
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096351, upper bound: 0.0092096
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0063931, 0.0046406, -0.0063444, 0.0045768, -0.0109699, 0.0109850
1: 0.9970240, 1.0097094, 0.9968072, 1.0096128, -0.0112980, 0.0114759
2: -0.0056239, 0.0054830, -0.0055605, 0.0054326, -0.0110565, 0.0110436
3: -0.0000754, 0.0025781, -0.0000617, 0.0026680, -0.0023404, 0.0022829
4: -0.0065605, 0.0014738, -0.0064964, 0.0014687, -0.0080292, 0.0079702
5: -0.0021320, 0.0077698, -0.0021098, 0.0076895, -0.0098215, 0.0098796
6: -0.0084996, 0.0019150, -0.0084006, 0.0019071, -0.0104068, 0.0103155
7: -0.0058763, -0.0001161, -0.0060280, -0.0001514, -0.0057249, 0.0059119
8: -0.0132269, -0.0014818, -0.0131775, -0.0010128, -0.0122141, 0.0116957
9: -0.0041023, 0.0074226, -0.0040139, 0.0073936, -0.0114959, 0.0114365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098503, upper bound: 0.0095510
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098503, upper bound: 0.0095561
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0062609, 0.0044676, -0.0061551, 0.0043290, -0.0105900, 0.0106227
1: 0.9970286, 1.0094470, 0.9967088, 1.0092369, -0.0109813, 0.0113109
2: -0.0054520, 0.0053461, -0.0053144, 0.0052365, -0.0106886, 0.0106606
3: -0.0000384, 0.0025762, -0.0000087, 0.0027088, -0.0023413, 0.0022571
4: -0.0063866, 0.0014290, -0.0062472, 0.0015140, -0.0079006, 0.0076762
5: -0.0020718, 0.0075521, -0.0020236, 0.0073778, -0.0094496, 0.0095757
6: -0.0082309, 0.0018937, -0.0080157, 0.0018767, -0.0101076, 0.0099094
7: -0.0058730, -0.0002119, -0.0060968, -0.0002887, -0.0055843, 0.0058849
8: -0.0130929, -0.0014918, -0.0129856, -0.0008001, -0.0122929, 0.0114938
9: -0.0038626, 0.0073440, -0.0036707, 0.0072810, -0.0111436, 0.0110147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097905, upper bound: 0.0095398
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097905, upper bound: 0.0095484
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0062067, 0.0043966, -0.0065543, 0.0048516, -0.0110583, 0.0109509
1: 0.9970120, 1.0093392, 0.9968157, 1.0100294, -0.0117799, 0.0111621
2: -0.0053815, 0.0052900, -0.0058334, 0.0056499, -0.0110314, 0.0111234
3: -0.0000232, 0.0025830, -0.0001205, 0.0026645, -0.0023156, 0.0023880
4: -0.0063152, 0.0014106, -0.0067726, 0.0015285, -0.0078436, 0.0081832
5: -0.0020471, 0.0074628, -0.0022054, 0.0080351, -0.0100823, 0.0096682
6: -0.0081207, 0.0018850, -0.0088273, 0.0019409, -0.0100616, 0.0107122
7: -0.0058846, -0.0002513, -0.0060221, 0.0000008, -0.0058855, 0.0057708
8: -0.0130380, -0.0014559, -0.0133903, -0.0010310, -0.0120070, 0.0119344
9: -0.0037643, 0.0073117, -0.0043945, 0.0075184, -0.0112827, 0.0117062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095941, upper bound: 0.0091423
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095941, upper bound: 0.0091494
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0060665, 0.0042131, -0.0063648, 0.0046035, -0.0106701, 0.0105778
1: 0.9970168, 1.0090610, 0.9967173, 1.0096532, -0.0114706, 0.0109889
2: -0.0051992, 0.0051448, -0.0055870, 0.0054537, -0.0106529, 0.0107318
3: 0.0000161, 0.0025811, -0.0000674, 0.0027053, -0.0023158, 0.0023613
4: -0.0061306, 0.0013723, -0.0065232, 0.0015101, -0.0076408, 0.0078955
5: -0.0019833, 0.0072319, -0.0021191, 0.0077231, -0.0097064, 0.0093510
6: -0.0078356, 0.0018624, -0.0084420, 0.0019104, -0.0097460, 0.0103044
7: -0.0058814, -0.0003530, -0.0060909, -0.0001366, -0.0057448, 0.0057379
8: -0.0128958, -0.0014659, -0.0131982, -0.0008183, -0.0120775, 0.0117323
9: -0.0035100, 0.0072283, -0.0040509, 0.0074057, -0.0109157, 0.0112792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094936, upper bound: 0.0091227
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094936, upper bound: 0.0091279
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0063931, 0.0046406, -0.0066518, 0.0049792, -0.0113724, 0.0112924
1: 0.9970240, 1.0097094, 0.9968141, 1.0102230, -0.0119568, 0.0115321
2: -0.0056239, 0.0054830, -0.0059602, 0.0057509, -0.0113748, 0.0114433
3: -0.0000754, 0.0025781, -0.0001478, 0.0026652, -0.0023650, 0.0024047
4: -0.0065605, 0.0014738, -0.0069010, 0.0015615, -0.0081221, 0.0083748
5: -0.0021320, 0.0077698, -0.0022498, 0.0081958, -0.0103278, 0.0100196
6: -0.0084996, 0.0019150, -0.0090255, 0.0019566, -0.0104562, 0.0109405
7: -0.0058763, -0.0001161, -0.0060232, 0.0000716, -0.0059478, 0.0059071
8: -0.0132269, -0.0014818, -0.0134892, -0.0010276, -0.0121993, 0.0120074
9: -0.0041023, 0.0074226, -0.0045714, 0.0075764, -0.0116787, 0.0119940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097595, upper bound: 0.0094669
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097595, upper bound: 0.0094853
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0062609, 0.0044676, -0.0064651, 0.0047348, -0.0109958, 0.0109327
1: 0.9970286, 1.0094470, 0.9967157, 1.0098524, -0.0116439, 0.0113678
2: -0.0054520, 0.0053461, -0.0057175, 0.0055576, -0.0110096, 0.0110636
3: -0.0000384, 0.0025762, -0.0000955, 0.0027059, -0.0023662, 0.0023781
4: -0.0063866, 0.0014290, -0.0066552, 0.0015108, -0.0078974, 0.0080842
5: -0.0020718, 0.0075521, -0.0021648, 0.0078883, -0.0099601, 0.0097169
6: -0.0082309, 0.0018937, -0.0086459, 0.0019266, -0.0101575, 0.0105397
7: -0.0058730, -0.0002119, -0.0060920, -0.0000639, -0.0058091, 0.0058800
8: -0.0130929, -0.0014918, -0.0132999, -0.0008149, -0.0122780, 0.0118081
9: -0.0038626, 0.0073440, -0.0042328, 0.0074654, -0.0113280, 0.0115768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096887, upper bound: 0.0094485
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096887, upper bound: 0.0094660
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0064920, 0.0047700, -0.0063819, 0.0046260, -0.0111180, 0.0111520
1: 0.9970103, 1.0099057, 0.9968068, 1.0096872, -0.0114396, 0.0115858
2: -0.0057524, 0.0055854, -0.0056093, 0.0054715, -0.0112239, 0.0111947
3: -0.0001030, 0.0025837, -0.0000722, 0.0026682, -0.0023289, 0.0023252
4: -0.0066906, 0.0015073, -0.0065458, 0.0014700, -0.0081606, 0.0080531
5: -0.0021770, 0.0079325, -0.0021269, 0.0077514, -0.0099284, 0.0100595
6: -0.0087006, 0.0019309, -0.0084769, 0.0019132, -0.0106138, 0.0104078
7: -0.0058858, -0.0000444, -0.0060283, -0.0001242, -0.0057617, 0.0059839
8: -0.0133271, -0.0014522, -0.0132156, -0.0010119, -0.0123152, 0.0117634
9: -0.0042816, 0.0074813, -0.0040820, 0.0074159, -0.0116975, 0.0115634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100476, upper bound: 0.0098038
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100476, upper bound: 0.0098056
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0063627, 0.0046009, -0.0061933, 0.0043790, -0.0107417, 0.0107941
1: 0.9970149, 1.0096490, 0.9967087, 1.0093126, -0.0111442, 0.0114309
2: -0.0055844, 0.0054516, -0.0053640, 0.0052760, -0.0108604, 0.0108156
3: -0.0000669, 0.0025818, -0.0000194, 0.0027089, -0.0023265, 0.0023001
4: -0.0065205, 0.0014635, -0.0062974, 0.0015141, -0.0080347, 0.0077609
5: -0.0021182, 0.0077198, -0.0020410, 0.0074406, -0.0095588, 0.0097607
6: -0.0084379, 0.0019101, -0.0080933, 0.0018828, -0.0103207, 0.0100034
7: -0.0058826, -0.0001381, -0.0060970, -0.0002610, -0.0056216, 0.0059589
8: -0.0131961, -0.0014622, -0.0130243, -0.0007995, -0.0123966, 0.0115621
9: -0.0040472, 0.0074045, -0.0037399, 0.0073037, -0.0113509, 0.0111443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099766, upper bound: 0.0097928
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099766, upper bound: 0.0097951
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0066838, 0.0050211, -0.0064815, 0.0047563, -0.0114400, 0.0115026
1: 0.9970283, 1.0102866, 0.9968053, 1.0098847, -0.0116234, 0.0119379
2: -0.0060018, 0.0057840, -0.0057388, 0.0055746, -0.0115763, 0.0115228
3: -0.0001567, 0.0025763, -0.0001001, 0.0026688, -0.0023734, 0.0023486
4: -0.0069430, 0.0015724, -0.0066768, 0.0015038, -0.0084468, 0.0082492
5: -0.0022644, 0.0082484, -0.0021723, 0.0079153, -0.0101797, 0.0104206
6: -0.0090905, 0.0019617, -0.0086793, 0.0019292, -0.0110197, 0.0106410
7: -0.0058733, 0.0000947, -0.0060294, -0.0000520, -0.0058213, 0.0061241
8: -0.0135216, -0.0014910, -0.0133165, -0.0010085, -0.0125131, 0.0118255
9: -0.0046293, 0.0075954, -0.0042626, 0.0074751, -0.0121045, 0.0118580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102266, upper bound: 0.0101434
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102266, upper bound: 0.0101547
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0065645, 0.0048649, -0.0063036, 0.0045233, -0.0110878, 0.0111685
1: 0.9970329, 1.0100496, 0.9967071, 1.0095317, -0.0113336, 0.0117867
2: -0.0058467, 0.0056605, -0.0055074, 0.0053903, -0.0112369, 0.0111679
3: -0.0001233, 0.0025744, -0.0000503, 0.0027095, -0.0023722, 0.0023237
4: -0.0067860, 0.0015319, -0.0064426, 0.0015148, -0.0083008, 0.0079745
5: -0.0022100, 0.0080519, -0.0020912, 0.0076223, -0.0098323, 0.0101431
6: -0.0088479, 0.0019425, -0.0083175, 0.0019006, -0.0107485, 0.0102600
7: -0.0058700, 0.0000082, -0.0060980, -0.0001810, -0.0056890, 0.0061063
8: -0.0134006, -0.0015011, -0.0131361, -0.0007962, -0.0126044, 0.0116350
9: -0.0044130, 0.0075245, -0.0039399, 0.0073693, -0.0117823, 0.0114643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101415, upper bound: 0.0101355
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101415, upper bound: 0.0101453
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0065131, 0.0047977, -0.0064375, 0.0046987, -0.0112119, 0.0112353
1: 0.9970096, 1.0099478, 0.9968101, 1.0097977, -0.0115596, 0.0118302
2: -0.0057799, 0.0056073, -0.0056816, 0.0055290, -0.0113089, 0.0112889
3: -0.0001090, 0.0025841, -0.0000878, 0.0026668, -0.0024210, 0.0023546
4: -0.0067185, 0.0015145, -0.0066189, 0.0014889, -0.0082073, 0.0081334
5: -0.0021867, 0.0079674, -0.0021522, 0.0078429, -0.0100296, 0.0101197
6: -0.0087436, 0.0019343, -0.0085899, 0.0019221, -0.0106658, 0.0105242
7: -0.0058864, -0.0000290, -0.0060259, -0.0000839, -0.0058025, 0.0059969
8: -0.0133486, -0.0014505, -0.0132719, -0.0010193, -0.0123293, 0.0118215
9: -0.0043200, 0.0074939, -0.0041828, 0.0074490, -0.0117689, 0.0116767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093474, upper bound: 0.0094397
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093343, upper bound: 0.0093524
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0065131, 0.0047977, -0.0067367, 0.0050904, -0.0116035, 0.0115344
1: 0.9970096, 1.0099478, 0.9968106, 1.0103917, -0.0120261, 0.0116777
2: -0.0057799, 0.0056073, -0.0060706, 0.0058388, -0.0116188, 0.0116779
3: -0.0001090, 0.0025841, -0.0001715, 0.0026666, -0.0023497, 0.0023832
4: -0.0067185, 0.0015145, -0.0070127, 0.0015903, -0.0083088, 0.0085272
5: -0.0021867, 0.0079674, -0.0022885, 0.0083355, -0.0105222, 0.0102559
6: -0.0087436, 0.0019343, -0.0091981, 0.0019702, -0.0107139, 0.0111324
7: -0.0058864, -0.0000290, -0.0060256, 0.0001331, -0.0060195, 0.0059966
8: -0.0133486, -0.0014505, -0.0135752, -0.0010202, -0.0123284, 0.0121247
9: -0.0043200, 0.0074939, -0.0047253, 0.0076269, -0.0119469, 0.0122193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093474, upper bound: 0.0097037
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093343, upper bound: 0.0096281
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0067062, 0.0050504, -0.0065371, 0.0048290, -0.0115352, 0.0115875
1: 0.9970274, 1.0103310, 0.9968086, 1.0099952, -0.0117461, 0.0121803
2: -0.0060309, 0.0058072, -0.0058110, 0.0056321, -0.0116630, 0.0116182
3: -0.0001630, 0.0025766, -0.0001157, 0.0026674, -0.0024713, 0.0023753
4: -0.0069725, 0.0015800, -0.0067500, 0.0015226, -0.0084951, 0.0083299
5: -0.0022746, 0.0082853, -0.0021976, 0.0080068, -0.0102814, 0.0104828
6: -0.0091360, 0.0019653, -0.0087922, 0.0019381, -0.0110742, 0.0107576
7: -0.0058738, 0.0001110, -0.0060270, -0.0000117, -0.0058622, 0.0061380
8: -0.0135443, -0.0014892, -0.0133728, -0.0010157, -0.0125286, 0.0118836
9: -0.0046700, 0.0076087, -0.0043633, 0.0075082, -0.0121781, 0.0119721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095384, upper bound: 0.0097787
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095184, upper bound: 0.0097075
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0067062, 0.0050504, -0.0068313, 0.0052142, -0.0119204, 0.0118817
1: 0.9970274, 1.0103310, 0.9968090, 1.0105793, -0.0121938, 0.0120442
2: -0.0060309, 0.0058072, -0.0061936, 0.0059368, -0.0119677, 0.0120008
3: -0.0001630, 0.0025766, -0.0001980, 0.0026673, -0.0024036, 0.0023997
4: -0.0069725, 0.0015800, -0.0071372, 0.0016224, -0.0085949, 0.0087172
5: -0.0022746, 0.0082853, -0.0023316, 0.0084913, -0.0107659, 0.0106168
6: -0.0091360, 0.0019653, -0.0093904, 0.0019855, -0.0111215, 0.0113557
7: -0.0058738, 0.0001110, -0.0060267, 0.0002018, -0.0060756, 0.0061377
8: -0.0135443, -0.0014892, -0.0136711, -0.0010166, -0.0125277, 0.0121819
9: -0.0046700, 0.0076087, -0.0048969, 0.0076831, -0.0123531, 0.0125056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095384, upper bound: 0.0097787
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095184, upper bound: 0.0100499
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0061276, 0.0042929, -0.0064627, 0.0047316, -0.0108592, 0.0107556
1: 0.9967992, 1.0091821, 0.9970081, 1.0098476, -0.0117033, 0.0108983
2: -0.0052786, 0.0052080, -0.0057143, 0.0055550, -0.0108336, 0.0109223
3: -0.0000010, 0.0026714, -0.0000948, 0.0025847, -0.0022351, 0.0024130
4: -0.0062110, 0.0014725, -0.0066520, 0.0014974, -0.0077083, 0.0081245
5: -0.0020111, 0.0073324, -0.0021637, 0.0078843, -0.0098953, 0.0094961
6: -0.0079597, 0.0018722, -0.0086409, 0.0019262, -0.0098859, 0.0105132
7: -0.0060337, -0.0003087, -0.0058874, -0.0000656, -0.0059680, 0.0055787
8: -0.0129577, -0.0009952, -0.0132974, -0.0014475, -0.0115102, 0.0123022
9: -0.0036207, 0.0072646, -0.0042284, 0.0074639, -0.0110846, 0.0114930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095224, upper bound: 0.0095929
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094714, upper bound: 0.0094367
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0062286, 0.0044252, -0.0066612, 0.0049916, -0.0112202, 0.0110865
1: 0.9967974, 1.0093828, 0.9970261, 1.0102417, -0.0120531, 0.0110880
2: -0.0054100, 0.0053126, -0.0059725, 0.0057607, -0.0111707, 0.0112852
3: -0.0000293, 0.0026720, -0.0001504, 0.0025772, -0.0022585, 0.0024543
4: -0.0063440, 0.0014732, -0.0069134, 0.0015647, -0.0079087, 0.0083866
5: -0.0020571, 0.0074988, -0.0022541, 0.0082113, -0.0102684, 0.0097530
6: -0.0081651, 0.0018885, -0.0090447, 0.0019581, -0.0101232, 0.0109332
7: -0.0060348, -0.0002354, -0.0058747, 0.0000784, -0.0061132, 0.0056393
8: -0.0130601, -0.0009917, -0.0134987, -0.0014865, -0.0115736, 0.0125070
9: -0.0038039, 0.0073247, -0.0045886, 0.0075820, -0.0113860, 0.0119133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098208, upper bound: 0.0097667
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097878, upper bound: 0.0096058
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0064053, 0.0046565, -0.0064627, 0.0047316, -0.0111369, 0.0111191
1: 0.9970219, 1.0097334, 0.9970081, 1.0098476, -0.0115296, 0.0114051
2: -0.0056397, 0.0054956, -0.0057143, 0.0055550, -0.0111947, 0.0112099
3: -0.0000788, 0.0025789, -0.0000948, 0.0025847, -0.0023080, 0.0023304
4: -0.0065765, 0.0014779, -0.0066520, 0.0014974, -0.0080739, 0.0081299
5: -0.0021375, 0.0077898, -0.0021637, 0.0078843, -0.0100218, 0.0099534
6: -0.0085243, 0.0019169, -0.0086409, 0.0019262, -0.0104505, 0.0105579
7: -0.0058776, -0.0001073, -0.0058874, -0.0000656, -0.0058120, 0.0057801
8: -0.0132392, -0.0014776, -0.0132974, -0.0014475, -0.0117918, 0.0118198
9: -0.0041243, 0.0074298, -0.0042284, 0.0074639, -0.0115882, 0.0116581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098060, upper bound: 0.0097916
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097649, upper bound: 0.0096369
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0065080, 0.0047911, -0.0066612, 0.0049916, -0.0114997, 0.0114523
1: 0.9970205, 1.0099376, 0.9970261, 1.0102417, -0.0118776, 0.0116056
2: -0.0057733, 0.0056020, -0.0059725, 0.0057607, -0.0115340, 0.0115746
3: -0.0001075, 0.0025795, -0.0001504, 0.0025772, -0.0023330, 0.0023701
4: -0.0067118, 0.0015128, -0.0069134, 0.0015647, -0.0082765, 0.0084262
5: -0.0021843, 0.0079590, -0.0022541, 0.0082113, -0.0103957, 0.0102132
6: -0.0087333, 0.0019335, -0.0090447, 0.0019581, -0.0106914, 0.0109782
7: -0.0058787, -0.0000327, -0.0058747, 0.0000784, -0.0059572, 0.0058420
8: -0.0133434, -0.0014741, -0.0134987, -0.0014865, -0.0118569, 0.0120246
9: -0.0043107, 0.0074909, -0.0045886, 0.0075820, -0.0118927, 0.0120795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100979, upper bound: 0.0099568
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100617, upper bound: 0.0098093
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0061276, 0.0042929, -0.0067740, 0.0051392, -0.0112667, 0.0110669
1: 0.9967992, 1.0091821, 0.9970053, 1.0104657, -0.0123772, 0.0109465
2: -0.0052786, 0.0052080, -0.0061191, 0.0058775, -0.0111560, 0.0113271
3: -0.0000010, 0.0026714, -0.0001820, 0.0025859, -0.0022534, 0.0025371
4: -0.0062110, 0.0014725, -0.0070618, 0.0016030, -0.0078140, 0.0085343
5: -0.0020111, 0.0073324, -0.0023055, 0.0083970, -0.0104080, 0.0096379
6: -0.0079597, 0.0018722, -0.0092739, 0.0019763, -0.0099360, 0.0111462
7: -0.0060337, -0.0003087, -0.0058895, 0.0001602, -0.0061939, 0.0055807
8: -0.0129577, -0.0009952, -0.0136130, -0.0014410, -0.0115167, 0.0126178
9: -0.0036207, 0.0072646, -0.0047930, 0.0076491, -0.0112698, 0.0120576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093409, upper bound: 0.0094914
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092927, upper bound: 0.0093328
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0062286, 0.0044252, -0.0069669, 0.0053918, -0.0116204, 0.0113922
1: 0.9967974, 1.0093828, 0.9970270, 1.0108488, -0.0127256, 0.0111208
2: -0.0054100, 0.0053126, -0.0063700, 0.0060773, -0.0114873, 0.0116826
3: -0.0000293, 0.0026720, -0.0002360, 0.0025768, -0.0022718, 0.0025786
4: -0.0063440, 0.0014732, -0.0073158, 0.0016684, -0.0080124, 0.0087890
5: -0.0020571, 0.0074988, -0.0023934, 0.0087148, -0.0107718, 0.0098922
6: -0.0081651, 0.0018885, -0.0096662, 0.0020073, -0.0101724, 0.0115547
7: -0.0060348, -0.0002354, -0.0058742, 0.0003002, -0.0063350, 0.0056388
8: -0.0130601, -0.0009917, -0.0138087, -0.0014883, -0.0115718, 0.0128169
9: -0.0038039, 0.0073247, -0.0051429, 0.0077638, -0.0115678, 0.0124676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097317, upper bound: 0.0096980
time: 1.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096984, upper bound: 0.0095327
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0064053, 0.0046565, -0.0067740, 0.0051392, -0.0115444, 0.0114305
1: 0.9970219, 1.0097334, 0.9970053, 1.0104657, -0.0122173, 0.0114696
2: -0.0056397, 0.0054956, -0.0061191, 0.0058775, -0.0115171, 0.0116147
3: -0.0000788, 0.0025789, -0.0001820, 0.0025859, -0.0023323, 0.0024560
4: -0.0065765, 0.0014779, -0.0070618, 0.0016030, -0.0081795, 0.0085397
5: -0.0021375, 0.0077898, -0.0023055, 0.0083970, -0.0105345, 0.0100952
6: -0.0085243, 0.0019169, -0.0092739, 0.0019763, -0.0105006, 0.0111909
7: -0.0058776, -0.0001073, -0.0058895, 0.0001602, -0.0060378, 0.0057822
8: -0.0132392, -0.0014776, -0.0136130, -0.0014410, -0.0117982, 0.0121355
9: -0.0041243, 0.0074298, -0.0047930, 0.0076491, -0.0117734, 0.0122228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095563, upper bound: 0.0096725
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095197, upper bound: 0.0095295
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0065080, 0.0047911, -0.0069669, 0.0053918, -0.0118998, 0.0117580
1: 0.9970205, 1.0099376, 0.9970270, 1.0108488, -0.0125610, 0.0116492
2: -0.0057733, 0.0056020, -0.0063700, 0.0060773, -0.0118506, 0.0119720
3: -0.0001075, 0.0025795, -0.0002360, 0.0025768, -0.0023514, 0.0024959
4: -0.0067118, 0.0015128, -0.0073158, 0.0016684, -0.0083802, 0.0088285
5: -0.0021843, 0.0079590, -0.0023934, 0.0087148, -0.0108991, 0.0103524
6: -0.0087333, 0.0019335, -0.0096662, 0.0020073, -0.0107406, 0.0115997
7: -0.0058787, -0.0000327, -0.0058742, 0.0003002, -0.0061789, 0.0058414
8: -0.0133434, -0.0014741, -0.0138087, -0.0014883, -0.0118551, 0.0123345
9: -0.0043107, 0.0074909, -0.0051429, 0.0077638, -0.0120745, 0.0126338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100042, upper bound: 0.0098811
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099680, upper bound: 0.0097252
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0064078, 0.0046598, -0.0065991, 0.0049103, -0.0113180, 0.0112589
1: 0.9968033, 1.0097386, 0.9970062, 1.0101185, -0.0120506, 0.0113389
2: -0.0056429, 0.0054982, -0.0058917, 0.0056963, -0.0113393, 0.0113899
3: -0.0000795, 0.0026696, -0.0001330, 0.0025854, -0.0022622, 0.0024820
4: -0.0065798, 0.0014788, -0.0068316, 0.0015437, -0.0081235, 0.0083104
5: -0.0021387, 0.0077939, -0.0022258, 0.0081090, -0.0102477, 0.0100197
6: -0.0085294, 0.0019173, -0.0089184, 0.0019481, -0.0104775, 0.0108357
7: -0.0060307, -0.0001055, -0.0058887, 0.0000334, -0.0060641, 0.0057832
8: -0.0132418, -0.0010043, -0.0134358, -0.0014435, -0.0117983, 0.0124315
9: -0.0041289, 0.0074313, -0.0044758, 0.0075451, -0.0116739, 0.0119071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098056, upper bound: 0.0100712
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097951, upper bound: 0.0100052
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0065120, 0.0047962, -0.0068071, 0.0051826, -0.0116946, 0.0116033
1: 0.9968016, 1.0099454, 0.9970243, 1.0105314, -0.0124048, 0.0115226
2: -0.0057784, 0.0056061, -0.0061622, 0.0059118, -0.0116902, 0.0117683
3: -0.0001086, 0.0026703, -0.0001913, 0.0025780, -0.0022825, 0.0025194
4: -0.0067169, 0.0015141, -0.0071054, 0.0016142, -0.0083312, 0.0086195
5: -0.0021861, 0.0079655, -0.0023206, 0.0084516, -0.0106377, 0.0102861
6: -0.0087412, 0.0019341, -0.0093413, 0.0019816, -0.0107228, 0.0112754
7: -0.0060319, -0.0000299, -0.0058761, 0.0001842, -0.0062161, 0.0058462
8: -0.0133474, -0.0010007, -0.0136466, -0.0014824, -0.0118651, 0.0126460
9: -0.0043178, 0.0074932, -0.0048531, 0.0076688, -0.0119866, 0.0123463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101547, upper bound: 0.0102559
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0101453, upper bound: 0.0101814
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0066794, 0.0050153, -0.0065991, 0.0049103, -0.0115896, 0.0116144
1: 0.9970264, 1.0102779, 0.9970062, 1.0101185, -0.0118856, 0.0118566
2: -0.0059961, 0.0057795, -0.0058917, 0.0056963, -0.0116924, 0.0116712
3: -0.0001555, 0.0025771, -0.0001330, 0.0025854, -0.0023376, 0.0024015
4: -0.0069373, 0.0015709, -0.0068316, 0.0015437, -0.0084809, 0.0084025
5: -0.0022624, 0.0082412, -0.0022258, 0.0081090, -0.0103714, 0.0104670
6: -0.0090816, 0.0019610, -0.0089184, 0.0019481, -0.0110297, 0.0108794
7: -0.0058746, 0.0000916, -0.0058887, 0.0000334, -0.0059079, 0.0059802
8: -0.0135171, -0.0014871, -0.0134358, -0.0014435, -0.0120737, 0.0119487
9: -0.0046214, 0.0075928, -0.0044758, 0.0075451, -0.0121665, 0.0120687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100189, upper bound: 0.0102381
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100027, upper bound: 0.0101493
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0067899, 0.0051600, -0.0068071, 0.0051826, -0.0119725, 0.0119671
1: 0.9970248, 1.0104971, 0.9970243, 1.0105314, -0.0122242, 0.0120506
2: -0.0061397, 0.0058939, -0.0061622, 0.0059118, -0.0120515, 0.0120561
3: -0.0001864, 0.0025777, -0.0001913, 0.0025780, -0.0023585, 0.0024334
4: -0.0070827, 0.0016084, -0.0071054, 0.0016142, -0.0086969, 0.0087138
5: -0.0023127, 0.0084231, -0.0023206, 0.0084516, -0.0107643, 0.0107437
6: -0.0093062, 0.0019788, -0.0093413, 0.0019816, -0.0112878, 0.0113201
7: -0.0058756, 0.0001717, -0.0058761, 0.0001842, -0.0060599, 0.0060478
8: -0.0136291, -0.0014837, -0.0136466, -0.0014824, -0.0121468, 0.0121629
9: -0.0048218, 0.0076585, -0.0048531, 0.0076688, -0.0124906, 0.0125116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104203, upper bound: 0.0104742
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0104047, upper bound: 0.0103912
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0064078, 0.0046598, -0.0069208, 0.0053315, -0.0117392, 0.0115806
1: 0.9968033, 1.0097386, 0.9970033, 1.0107573, -0.0127396, 0.0113842
2: -0.0056429, 0.0054982, -0.0063100, 0.0060295, -0.0116725, 0.0118082
3: -0.0000795, 0.0026696, -0.0002231, 0.0025867, -0.0022772, 0.0026043
4: -0.0065798, 0.0014788, -0.0072551, 0.0016528, -0.0082326, 0.0087339
5: -0.0021387, 0.0077939, -0.0023724, 0.0086388, -0.0107775, 0.0101662
6: -0.0085294, 0.0019173, -0.0095725, 0.0019999, -0.0105293, 0.0114898
7: -0.0060307, -0.0001055, -0.0058908, 0.0002667, -0.0062974, 0.0057853
8: -0.0132418, -0.0010043, -0.0137619, -0.0014370, -0.0118048, 0.0127577
9: -0.0041289, 0.0074313, -0.0050593, 0.0077364, -0.0118653, 0.0124906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095542, upper bound: 0.0099342
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095390, upper bound: 0.0098648
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0065120, 0.0047962, -0.0071194, 0.0055914, -0.0121034, 0.0119156
1: 0.9968016, 1.0099454, 0.9970251, 1.0111514, -0.0130843, 0.0115551
2: -0.0057784, 0.0056061, -0.0065683, 0.0062352, -0.0120136, 0.0121744
3: -0.0001086, 0.0026703, -0.0002787, 0.0025776, -0.0022915, 0.0026415
4: -0.0067169, 0.0015141, -0.0075165, 0.0017202, -0.0084371, 0.0090306
5: -0.0021861, 0.0079655, -0.0024628, 0.0089659, -0.0111520, 0.0104283
6: -0.0087412, 0.0019341, -0.0099763, 0.0020318, -0.0107731, 0.0119104
7: -0.0060319, -0.0000299, -0.0058755, 0.0004108, -0.0064427, 0.0058456
8: -0.0133474, -0.0010007, -0.0139633, -0.0014842, -0.0118632, 0.0129626
9: -0.0043178, 0.0074932, -0.0054195, 0.0078545, -0.0121724, 0.0129127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100488, upper bound: 0.0101749
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100401, upper bound: 0.0100897
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0066794, 0.0050153, -0.0069208, 0.0053315, -0.0120108, 0.0119362
1: 0.9970264, 1.0102779, 0.9970033, 1.0107573, -0.0125812, 0.0119112
2: -0.0059961, 0.0057795, -0.0063100, 0.0060295, -0.0120256, 0.0120895
3: -0.0001555, 0.0025771, -0.0002231, 0.0025867, -0.0023593, 0.0025260
4: -0.0069373, 0.0015709, -0.0072551, 0.0016528, -0.0085901, 0.0088260
5: -0.0022624, 0.0082412, -0.0023724, 0.0086388, -0.0109012, 0.0106135
6: -0.0090816, 0.0019610, -0.0095725, 0.0019999, -0.0110815, 0.0115335
7: -0.0058746, 0.0000916, -0.0058908, 0.0002667, -0.0061413, 0.0059823
8: -0.0135171, -0.0014871, -0.0137619, -0.0014370, -0.0120801, 0.0122749
9: -0.0046214, 0.0075928, -0.0050593, 0.0077364, -0.0123578, 0.0126521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097123, upper bound: 0.0100652
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096884, upper bound: 0.0099977
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0067899, 0.0051600, -0.0071194, 0.0055914, -0.0123813, 0.0122794
1: 0.9970248, 1.0104971, 0.9970251, 1.0111514, -0.0129121, 0.0120896
2: -0.0061397, 0.0058939, -0.0065683, 0.0062352, -0.0123750, 0.0124622
3: -0.0001864, 0.0025777, -0.0002787, 0.0025776, -0.0023733, 0.0025580
4: -0.0070827, 0.0016084, -0.0075165, 0.0017202, -0.0088029, 0.0091248
5: -0.0023127, 0.0084231, -0.0024628, 0.0089659, -0.0112786, 0.0108859
6: -0.0093062, 0.0019788, -0.0099763, 0.0020318, -0.0113381, 0.0119551
7: -0.0058756, 0.0001717, -0.0058755, 0.0004108, -0.0062864, 0.0060472
8: -0.0136291, -0.0014837, -0.0139633, -0.0014842, -0.0121449, 0.0124796
9: -0.0048218, 0.0076585, -0.0054195, 0.0078545, -0.0126763, 0.0130780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102974, upper bound: 0.0103878
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102846, upper bound: 0.0102870
time: 2.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0065776, 0.0048821, -0.0061659, 0.0043432, -0.0109208, 0.0110481
1: 0.9970360, 1.0100757, 0.9967875, 1.0092585, -0.0109801, 0.0119433
2: -0.0058638, 0.0056741, -0.0053285, 0.0052477, -0.0111115, 0.0110026
3: -0.0001270, 0.0025731, -0.0000118, 0.0026761, -0.0024483, 0.0022472
4: -0.0068033, 0.0015364, -0.0062615, 0.0014778, -0.0082811, 0.0077979
5: -0.0022160, 0.0080736, -0.0020285, 0.0073956, -0.0096117, 0.0101021
6: -0.0088747, 0.0019447, -0.0080377, 0.0018784, -0.0107531, 0.0099824
7: -0.0058679, 0.0000178, -0.0060417, -0.0002809, -0.0055870, 0.0060595
8: -0.0134140, -0.0015078, -0.0129966, -0.0009703, -0.0124437, 0.0114888
9: -0.0044369, 0.0075323, -0.0036903, 0.0072874, -0.0117243, 0.0112226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094437
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094471
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0065776, 0.0048821, -0.0064891, 0.0047662, -0.0113438, 0.0113712
1: 0.9970360, 1.0100757, 0.9967924, 1.0099000, -0.0115239, 0.0118425
2: -0.0058638, 0.0056741, -0.0057486, 0.0055824, -0.0114462, 0.0114227
3: -0.0001270, 0.0025731, -0.0001022, 0.0026742, -0.0023879, 0.0023002
4: -0.0068033, 0.0015364, -0.0066868, 0.0015063, -0.0083097, 0.0082232
5: -0.0022160, 0.0080736, -0.0021757, 0.0079278, -0.0101438, 0.0102493
6: -0.0088747, 0.0019447, -0.0086947, 0.0019304, -0.0108051, 0.0106393
7: -0.0058679, 0.0000178, -0.0060384, -0.0000465, -0.0058214, 0.0060562
8: -0.0134140, -0.0015078, -0.0133242, -0.0009805, -0.0124335, 0.0118164
9: -0.0044369, 0.0075323, -0.0042763, 0.0074796, -0.0119165, 0.0118086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094437
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094471
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0063851, 0.0046301, -0.0060259, 0.0041599, -0.0105450, 0.0106561
1: 0.9969429, 1.0096934, 0.9967923, 1.0089803, -0.0108141, 0.0116248
2: -0.0056135, 0.0054747, -0.0051464, 0.0051027, -0.0107162, 0.0106212
3: -0.0000731, 0.0026117, 0.0000274, 0.0026742, -0.0024223, 0.0022443
4: -0.0065500, 0.0014711, -0.0060772, 0.0014756, -0.0080256, 0.0075483
5: -0.0021284, 0.0077566, -0.0019648, 0.0071651, -0.0092934, 0.0097213
6: -0.0084833, 0.0019137, -0.0077531, 0.0018559, -0.0103392, 0.0096668
7: -0.0059330, -0.0001219, -0.0060384, -0.0003824, -0.0055506, 0.0059165
8: -0.0132188, -0.0013064, -0.0128547, -0.0009805, -0.0122383, 0.0115482
9: -0.0040878, 0.0074178, -0.0034364, 0.0072042, -0.0112919, 0.0108542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092396
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092442
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0063851, 0.0046301, -0.0063659, 0.0046050, -0.0109901, 0.0109960
1: 0.9969429, 1.0096934, 0.9967969, 1.0096554, -0.0113685, 0.0115280
2: -0.0056135, 0.0054747, -0.0055885, 0.0054548, -0.0110683, 0.0110632
3: -0.0000731, 0.0026117, -0.0000677, 0.0026723, -0.0023621, 0.0023006
4: -0.0065500, 0.0014711, -0.0065247, 0.0014735, -0.0080234, 0.0079958
5: -0.0021284, 0.0077566, -0.0021196, 0.0077249, -0.0098533, 0.0098762
6: -0.0084833, 0.0019137, -0.0084443, 0.0019106, -0.0103939, 0.0103580
7: -0.0059330, -0.0001219, -0.0060352, -0.0001358, -0.0057972, 0.0059133
8: -0.0132188, -0.0013064, -0.0131993, -0.0009905, -0.0122283, 0.0118929
9: -0.0040878, 0.0074178, -0.0040529, 0.0074064, -0.0114941, 0.0114707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092397
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092442
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0066730, 0.0050071, -0.0063563, 0.0045923, -0.0112654, 0.0113633
1: 0.9970345, 1.0102651, 0.9968041, 1.0096362, -0.0113533, 0.0121326
2: -0.0059879, 0.0057729, -0.0055760, 0.0054449, -0.0114327, 0.0113489
3: -0.0001537, 0.0025737, -0.0000650, 0.0026694, -0.0024679, 0.0022980
4: -0.0069290, 0.0015688, -0.0065120, 0.0014702, -0.0083992, 0.0080807
5: -0.0022595, 0.0082308, -0.0021152, 0.0077090, -0.0099685, 0.0103460
6: -0.0090688, 0.0019600, -0.0084247, 0.0019090, -0.0109778, 0.0103847
7: -0.0058690, 0.0000870, -0.0060303, -0.0001428, -0.0057262, 0.0061173
8: -0.0135107, -0.0015044, -0.0131895, -0.0010057, -0.0125050, 0.0116852
9: -0.0046100, 0.0075890, -0.0040355, 0.0074006, -0.0120106, 0.0116245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096177
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096325
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0066730, 0.0050071, -0.0066693, 0.0050022, -0.0116752, 0.0116764
1: 0.9970345, 1.0102651, 0.9968113, 1.0102578, -0.0118737, 0.0120233
2: -0.0059879, 0.0057729, -0.0059830, 0.0057691, -0.0117569, 0.0117559
3: -0.0001537, 0.0025737, -0.0001527, 0.0026663, -0.0024058, 0.0023420
4: -0.0069290, 0.0015688, -0.0069241, 0.0015675, -0.0084964, 0.0084928
5: -0.0022595, 0.0082308, -0.0022578, 0.0082246, -0.0104841, 0.0104886
6: -0.0090688, 0.0019600, -0.0090612, 0.0019594, -0.0110282, 0.0110212
7: -0.0058690, 0.0000870, -0.0060251, 0.0000843, -0.0059533, 0.0061121
8: -0.0135107, -0.0015044, -0.0135069, -0.0010216, -0.0124891, 0.0120026
9: -0.0046100, 0.0075890, -0.0046032, 0.0075868, -0.0121968, 0.0121922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096177
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096325
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0064874, 0.0047640, -0.0062326, 0.0044304, -0.0109178, 0.0109966
1: 0.9969414, 1.0098969, 0.9968088, 1.0093906, -0.0111934, 0.0118129
2: -0.0057465, 0.0055807, -0.0054151, 0.0053167, -0.0110632, 0.0109958
3: -0.0001018, 0.0026123, -0.0000304, 0.0026674, -0.0024415, 0.0022968
4: -0.0066846, 0.0015058, -0.0063492, 0.0014680, -0.0081526, 0.0078549
5: -0.0021750, 0.0079250, -0.0020589, 0.0075053, -0.0096803, 0.0099839
6: -0.0086913, 0.0019301, -0.0081732, 0.0018891, -0.0105805, 0.0101033
7: -0.0059341, -0.0000477, -0.0060269, -0.0002325, -0.0057015, 0.0059792
8: -0.0133225, -0.0013031, -0.0130641, -0.0010161, -0.0123064, 0.0117611
9: -0.0042733, 0.0074786, -0.0038111, 0.0073271, -0.0116003, 0.0112897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094142
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094301
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0064874, 0.0047640, -0.0065465, 0.0048414, -0.0113288, 0.0113106
1: 0.9969414, 1.0098969, 0.9968159, 1.0100141, -0.0117210, 0.0117145
2: -0.0057465, 0.0055807, -0.0058233, 0.0056419, -0.0113884, 0.0114040
3: -0.0001018, 0.0026123, -0.0001183, 0.0026644, -0.0023804, 0.0023432
4: -0.0066846, 0.0015058, -0.0067624, 0.0015258, -0.0082104, 0.0082682
5: -0.0021750, 0.0079250, -0.0022019, 0.0080224, -0.0101973, 0.0101269
6: -0.0086913, 0.0019301, -0.0088115, 0.0019397, -0.0106310, 0.0107416
7: -0.0059341, -0.0000477, -0.0060219, -0.0000048, -0.0059293, 0.0059742
8: -0.0133225, -0.0013031, -0.0133824, -0.0010316, -0.0122909, 0.0120794
9: -0.0042733, 0.0074786, -0.0043805, 0.0075138, -0.0117871, 0.0118591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094142
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094301
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0068833, 0.0052823, -0.0063091, 0.0045307, -0.0114140, 0.0115914
1: 0.9970333, 1.0106827, 0.9967856, 1.0095427, -0.0113225, 0.0124212
2: -0.0062612, 0.0059907, -0.0055147, 0.0053960, -0.0116573, 0.0115054
3: -0.0002126, 0.0025742, -0.0000519, 0.0026769, -0.0024774, 0.0023181
4: -0.0072057, 0.0016401, -0.0064500, 0.0014787, -0.0086843, 0.0080900
5: -0.0023553, 0.0085770, -0.0020938, 0.0076315, -0.0099867, 0.0106708
6: -0.0094962, 0.0019938, -0.0083289, 0.0019015, -0.0113976, 0.0103227
7: -0.0058698, 0.0002395, -0.0060431, -0.0001770, -0.0056928, 0.0062826
8: -0.0137239, -0.0015018, -0.0131418, -0.0009661, -0.0127578, 0.0116399
9: -0.0049912, 0.0077141, -0.0039500, 0.0073726, -0.0123638, 0.0116641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098664
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098664
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0068833, 0.0052823, -0.0066387, 0.0049621, -0.0118454, 0.0119210
1: 0.9970333, 1.0106827, 0.9967903, 1.0101972, -0.0118693, 0.0123127
2: -0.0062612, 0.0059907, -0.0059432, 0.0057374, -0.0119986, 0.0119339
3: -0.0002126, 0.0025742, -0.0001441, 0.0026750, -0.0024171, 0.0023688
4: -0.0072057, 0.0016401, -0.0068838, 0.0015571, -0.0087628, 0.0085238
5: -0.0023553, 0.0085770, -0.0022439, 0.0081742, -0.0105295, 0.0108209
6: -0.0094962, 0.0019938, -0.0089990, 0.0019545, -0.0114507, 0.0109928
7: -0.0058698, 0.0002395, -0.0060398, 0.0000621, -0.0059319, 0.0062793
8: -0.0137239, -0.0015018, -0.0134759, -0.0009761, -0.0127478, 0.0119741
9: -0.0049912, 0.0077141, -0.0045477, 0.0075686, -0.0125599, 0.0122618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098664
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098744
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0066970, 0.0050384, -0.0061791, 0.0043604, -0.0110574, 0.0112175
1: 0.9969380, 1.0103128, 0.9967904, 1.0092844, -0.0111710, 0.0121209
2: -0.0060190, 0.0057977, -0.0053456, 0.0052614, -0.0112804, 0.0111433
3: -0.0001604, 0.0026138, -0.0000154, 0.0026750, -0.0024519, 0.0023155
4: -0.0069605, 0.0015769, -0.0062788, 0.0014765, -0.0084369, 0.0078557
5: -0.0022704, 0.0082702, -0.0020345, 0.0074173, -0.0096877, 0.0103047
6: -0.0091175, 0.0019639, -0.0080645, 0.0018805, -0.0109980, 0.0100283
7: -0.0059365, 0.0001044, -0.0060397, -0.0002713, -0.0056652, 0.0061441
8: -0.0135350, -0.0012956, -0.0130099, -0.0009764, -0.0125586, 0.0117143
9: -0.0046534, 0.0076033, -0.0037142, 0.0072953, -0.0119487, 0.0113174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097790
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097886
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0066970, 0.0050384, -0.0065160, 0.0048014, -0.0114984, 0.0115544
1: 0.9969380, 1.0103128, 0.9967949, 1.0099535, -0.0117236, 0.0120155
2: -0.0060190, 0.0057977, -0.0057836, 0.0056102, -0.0116293, 0.0115813
3: -0.0001604, 0.0026138, -0.0001098, 0.0026731, -0.0023931, 0.0023696
4: -0.0069605, 0.0015769, -0.0067222, 0.0015155, -0.0084760, 0.0082991
5: -0.0022704, 0.0082702, -0.0021880, 0.0079721, -0.0102425, 0.0104582
6: -0.0091175, 0.0019639, -0.0087494, 0.0019347, -0.0110522, 0.0107132
7: -0.0059365, 0.0001044, -0.0060366, -0.0000270, -0.0059095, 0.0061409
8: -0.0135350, -0.0012956, -0.0133515, -0.0009863, -0.0125487, 0.0120558
9: -0.0046534, 0.0076033, -0.0043251, 0.0074956, -0.0121490, 0.0119284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097790
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097886
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0069784, 0.0054068, -0.0065010, 0.0047819, -0.0117603, 0.0119079
1: 0.9970319, 1.0108715, 0.9968019, 1.0099236, -0.0116874, 0.0125912
2: -0.0063849, 0.0060892, -0.0057642, 0.0055948, -0.0119797, 0.0118534
3: -0.0002392, 0.0025748, -0.0001056, 0.0026702, -0.0024949, 0.0023609
4: -0.0073309, 0.0016723, -0.0067025, 0.0015104, -0.0088413, 0.0083749
5: -0.0023986, 0.0087336, -0.0021812, 0.0079475, -0.0103461, 0.0109148
6: -0.0096896, 0.0020091, -0.0087190, 0.0019323, -0.0116219, 0.0107282
7: -0.0058708, 0.0003085, -0.0060317, -0.0000378, -0.0058330, 0.0063402
8: -0.0138203, -0.0014986, -0.0133363, -0.0010014, -0.0128189, 0.0118377
9: -0.0051637, 0.0077707, -0.0042980, 0.0074867, -0.0126504, 0.0120687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101233
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101361
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0069784, 0.0054068, -0.0068152, 0.0051932, -0.0121716, 0.0122221
1: 0.9970319, 1.0108715, 0.9968092, 1.0105476, -0.0122057, 0.0124822
2: -0.0063849, 0.0060892, -0.0061727, 0.0059202, -0.0123051, 0.0122619
3: -0.0002392, 0.0025748, -0.0001935, 0.0026671, -0.0024341, 0.0024048
4: -0.0073309, 0.0016723, -0.0071161, 0.0016170, -0.0089478, 0.0087884
5: -0.0023986, 0.0087336, -0.0023243, 0.0084649, -0.0108635, 0.0110579
6: -0.0096896, 0.0020091, -0.0093578, 0.0019829, -0.0116724, 0.0113670
7: -0.0058708, 0.0003085, -0.0060266, 0.0001901, -0.0060609, 0.0063350
8: -0.0138203, -0.0014986, -0.0136549, -0.0010172, -0.0128031, 0.0121562
9: -0.0051637, 0.0077707, -0.0048678, 0.0076736, -0.0128373, 0.0126384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101233
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101361
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0067947, 0.0051663, -0.0063803, 0.0046239, -0.0114185, 0.0115466
1: 0.9969366, 1.0105067, 0.9968069, 1.0096840, -0.0115402, 0.0122924
2: -0.0061460, 0.0058989, -0.0056072, 0.0054698, -0.0116158, 0.0115061
3: -0.0001878, 0.0026144, -0.0000718, 0.0026681, -0.0024687, 0.0023620
4: -0.0070890, 0.0016100, -0.0065437, 0.0014695, -0.0085585, 0.0081537
5: -0.0023149, 0.0084310, -0.0021262, 0.0077487, -0.0100636, 0.0105572
6: -0.0093160, 0.0019796, -0.0084736, 0.0019129, -0.0112289, 0.0104532
7: -0.0059375, 0.0001752, -0.0060282, -0.0001253, -0.0058122, 0.0062035
8: -0.0136340, -0.0012924, -0.0132140, -0.0010119, -0.0126221, 0.0119216
9: -0.0048305, 0.0076614, -0.0040791, 0.0074149, -0.0122454, 0.0117405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100142
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100309
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0067947, 0.0051663, -0.0066949, 0.0050357, -0.0118303, 0.0118612
1: 0.9969366, 1.0105067, 0.9968139, 1.0103086, -0.0120604, 0.0121869
2: -0.0061460, 0.0058989, -0.0060163, 0.0057956, -0.0119415, 0.0119152
3: -0.0001878, 0.0026144, -0.0001598, 0.0026652, -0.0024102, 0.0024056
4: -0.0070890, 0.0016100, -0.0069577, 0.0015762, -0.0086652, 0.0085677
5: -0.0023149, 0.0084310, -0.0022695, 0.0082668, -0.0105817, 0.0107005
6: -0.0093160, 0.0019796, -0.0091132, 0.0019635, -0.0112795, 0.0110928
7: -0.0059375, 0.0001752, -0.0060233, 0.0001029, -0.0060404, 0.0061985
8: -0.0136340, -0.0012924, -0.0135329, -0.0010274, -0.0126067, 0.0122405
9: -0.0048305, 0.0076614, -0.0046496, 0.0076020, -0.0124325, 0.0123110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100142
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100309
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0064375, 0.0046987, -0.0064627, 0.0047316, -0.0111691, 0.0111614
1: 0.9968101, 1.0097977, 0.9970081, 1.0098476, -0.0117495, 0.0115629
2: -0.0056816, 0.0055290, -0.0057143, 0.0055550, -0.0112366, 0.0112433
3: -0.0000878, 0.0026668, -0.0000948, 0.0025847, -0.0023577, 0.0024276
4: -0.0066189, 0.0014889, -0.0066520, 0.0014974, -0.0081163, 0.0081409
5: -0.0021522, 0.0078429, -0.0021637, 0.0078843, -0.0100365, 0.0100065
6: -0.0085899, 0.0019221, -0.0086409, 0.0019262, -0.0105160, 0.0105631
7: -0.0060259, -0.0000839, -0.0058874, -0.0000656, -0.0059602, 0.0058035
8: -0.0132719, -0.0010193, -0.0132974, -0.0014475, -0.0118245, 0.0122781
9: -0.0041828, 0.0074490, -0.0042284, 0.0074639, -0.0116467, 0.0116773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093462, upper bound: 0.0094808
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092906, upper bound: 0.0093195
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0064375, 0.0046987, -0.0067740, 0.0051392, -0.0115767, 0.0114727
1: 0.9968101, 1.0097977, 0.9970053, 1.0104657, -0.0122689, 0.0114638
2: -0.0056816, 0.0055290, -0.0061191, 0.0058775, -0.0115591, 0.0116481
3: -0.0000878, 0.0026668, -0.0001820, 0.0025859, -0.0022953, 0.0024756
4: -0.0066189, 0.0014889, -0.0070618, 0.0016030, -0.0082219, 0.0085506
5: -0.0021522, 0.0078429, -0.0023055, 0.0083970, -0.0105492, 0.0101483
6: -0.0085899, 0.0019221, -0.0092739, 0.0019763, -0.0105661, 0.0111961
7: -0.0060259, -0.0000839, -0.0058895, 0.0001602, -0.0061861, 0.0058056
8: -0.0132719, -0.0010193, -0.0136130, -0.0014410, -0.0118309, 0.0125938
9: -0.0041828, 0.0074490, -0.0047930, 0.0076491, -0.0118319, 0.0122419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093462, upper bound: 0.0094808
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092906, upper bound: 0.0093195
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0067051, 0.0050490, -0.0064627, 0.0047316, -0.0114367, 0.0115117
1: 0.9970300, 1.0103289, 0.9970081, 1.0098476, -0.0115592, 0.0120758
2: -0.0060296, 0.0058061, -0.0057143, 0.0055550, -0.0115846, 0.0115204
3: -0.0001627, 0.0025756, -0.0000948, 0.0025847, -0.0024347, 0.0023418
4: -0.0069712, 0.0015796, -0.0066520, 0.0014974, -0.0084685, 0.0082316
5: -0.0022741, 0.0082836, -0.0021637, 0.0078843, -0.0101584, 0.0104472
6: -0.0091339, 0.0019652, -0.0086409, 0.0019262, -0.0110601, 0.0106061
7: -0.0058721, 0.0001103, -0.0058874, -0.0000656, -0.0058065, 0.0059976
8: -0.0135432, -0.0014946, -0.0132974, -0.0014475, -0.0120958, 0.0118028
9: -0.0046681, 0.0076081, -0.0042284, 0.0074639, -0.0121320, 0.0118365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095564, upper bound: 0.0096608
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095223, upper bound: 0.0095158
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0067051, 0.0050490, -0.0067740, 0.0051392, -0.0118443, 0.0118230
1: 0.9970300, 1.0103289, 0.9970053, 1.0104657, -0.0120977, 0.0119747
2: -0.0060296, 0.0058061, -0.0061191, 0.0058775, -0.0119070, 0.0119252
3: -0.0001627, 0.0025756, -0.0001820, 0.0025859, -0.0023745, 0.0023914
4: -0.0069712, 0.0015796, -0.0070618, 0.0016030, -0.0085742, 0.0086414
5: -0.0022741, 0.0082836, -0.0023055, 0.0083970, -0.0106711, 0.0105890
6: -0.0091339, 0.0019652, -0.0092739, 0.0019763, -0.0111102, 0.0112391
7: -0.0058721, 0.0001103, -0.0058895, 0.0001602, -0.0060323, 0.0059997
8: -0.0135432, -0.0014946, -0.0136130, -0.0014410, -0.0121022, 0.0121184
9: -0.0046681, 0.0076081, -0.0047930, 0.0076491, -0.0123172, 0.0124011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095564, upper bound: 0.0096608
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095223, upper bound: 0.0095158
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0065371, 0.0048290, -0.0066612, 0.0049916, -0.0115287, 0.0114903
1: 0.9968086, 1.0099952, 0.9970261, 1.0102417, -0.0121126, 0.0117493
2: -0.0058110, 0.0056321, -0.0059725, 0.0057607, -0.0115717, 0.0116046
3: -0.0001157, 0.0026674, -0.0001504, 0.0025772, -0.0023785, 0.0024783
4: -0.0067500, 0.0015226, -0.0069134, 0.0015647, -0.0083147, 0.0084360
5: -0.0021976, 0.0080068, -0.0022541, 0.0082113, -0.0104089, 0.0102609
6: -0.0087922, 0.0019381, -0.0090447, 0.0019581, -0.0107504, 0.0109829
7: -0.0060270, -0.0000117, -0.0058747, 0.0000784, -0.0061054, 0.0058631
8: -0.0133728, -0.0010157, -0.0134987, -0.0014865, -0.0118863, 0.0124830
9: -0.0043633, 0.0075082, -0.0045886, 0.0075820, -0.0119453, 0.0120967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097342, upper bound: 0.0096789
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096895, upper bound: 0.0095106
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0065371, 0.0048290, -0.0069669, 0.0053918, -0.0119289, 0.0117960
1: 0.9968086, 1.0099952, 0.9970270, 1.0108488, -0.0126150, 0.0116477
2: -0.0058110, 0.0056321, -0.0063700, 0.0060773, -0.0118883, 0.0120021
3: -0.0001157, 0.0026674, -0.0002360, 0.0025768, -0.0023162, 0.0025189
4: -0.0067500, 0.0015226, -0.0073158, 0.0016684, -0.0084184, 0.0088384
5: -0.0021976, 0.0080068, -0.0023934, 0.0087148, -0.0109123, 0.0104002
6: -0.0087922, 0.0019381, -0.0096662, 0.0020073, -0.0107995, 0.0116044
7: -0.0060270, -0.0000117, -0.0058742, 0.0003002, -0.0063272, 0.0058625
8: -0.0133728, -0.0010157, -0.0138087, -0.0014883, -0.0118845, 0.0127929
9: -0.0043633, 0.0075082, -0.0051429, 0.0077638, -0.0121271, 0.0126511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097342, upper bound: 0.0096789
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096895, upper bound: 0.0095106
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0068028, 0.0051769, -0.0066612, 0.0049916, -0.0117944, 0.0118381
1: 0.9970282, 1.0105230, 0.9970261, 1.0102417, -0.0119340, 0.0122680
2: -0.0061565, 0.0059073, -0.0059725, 0.0057607, -0.0119172, 0.0118798
3: -0.0001901, 0.0025763, -0.0001504, 0.0025772, -0.0024562, 0.0023935
4: -0.0070997, 0.0016127, -0.0069134, 0.0015647, -0.0086644, 0.0085262
5: -0.0023186, 0.0084444, -0.0022541, 0.0082113, -0.0105299, 0.0106985
6: -0.0093325, 0.0019809, -0.0090447, 0.0019581, -0.0112906, 0.0110256
7: -0.0058733, 0.0001811, -0.0058747, 0.0000784, -0.0059517, 0.0060558
8: -0.0136422, -0.0014911, -0.0134987, -0.0014865, -0.0121557, 0.0120076
9: -0.0048452, 0.0076662, -0.0045886, 0.0075820, -0.0124272, 0.0122548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100051, upper bound: 0.0098704
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099681, upper bound: 0.0097060
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0068028, 0.0051769, -0.0069669, 0.0053918, -0.0121946, 0.0121438
1: 0.9970282, 1.0105230, 0.9970270, 1.0108488, -0.0124492, 0.0121583
2: -0.0061565, 0.0059073, -0.0063700, 0.0060773, -0.0122338, 0.0122773
3: -0.0001901, 0.0025763, -0.0002360, 0.0025768, -0.0023944, 0.0024346
4: -0.0070997, 0.0016127, -0.0073158, 0.0016684, -0.0087681, 0.0089285
5: -0.0023186, 0.0084444, -0.0023934, 0.0087148, -0.0110334, 0.0108377
6: -0.0093325, 0.0019809, -0.0096662, 0.0020073, -0.0113398, 0.0116471
7: -0.0058733, 0.0001811, -0.0058742, 0.0003002, -0.0061734, 0.0060552
8: -0.0136422, -0.0014911, -0.0138087, -0.0014883, -0.0121539, 0.0123176
9: -0.0048452, 0.0076662, -0.0051429, 0.0077638, -0.0126090, 0.0128091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100051, upper bound: 0.0098704
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0099681, upper bound: 0.0097060
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0067367, 0.0050904, -0.0065991, 0.0049103, -0.0116469, 0.0116895
1: 0.9968106, 1.0103917, 0.9970062, 1.0101185, -0.0120870, 0.0120302
2: -0.0060706, 0.0058388, -0.0058917, 0.0056963, -0.0117669, 0.0117305
3: -0.0001715, 0.0026666, -0.0001330, 0.0025854, -0.0023874, 0.0024946
4: -0.0070127, 0.0015903, -0.0068316, 0.0015437, -0.0085563, 0.0084219
5: -0.0022885, 0.0083355, -0.0022258, 0.0081090, -0.0103975, 0.0105614
6: -0.0091981, 0.0019702, -0.0089184, 0.0019481, -0.0111462, 0.0108886
7: -0.0060256, 0.0001331, -0.0058887, 0.0000334, -0.0060589, 0.0060218
8: -0.0135752, -0.0010202, -0.0134358, -0.0014435, -0.0121318, 0.0124156
9: -0.0047253, 0.0076269, -0.0044758, 0.0075451, -0.0122704, 0.0121027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096486, upper bound: 0.0099241
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096281, upper bound: 0.0098573
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0068313, 0.0052142, -0.0068071, 0.0051826, -0.0120139, 0.0120213
1: 0.9968090, 1.0105793, 0.9970243, 1.0105314, -0.0124627, 0.0121982
2: -0.0061936, 0.0059368, -0.0061622, 0.0059118, -0.0121053, 0.0120990
3: -0.0001980, 0.0026673, -0.0001913, 0.0025780, -0.0024042, 0.0025450
4: -0.0071372, 0.0016224, -0.0071054, 0.0016142, -0.0087514, 0.0087278
5: -0.0023316, 0.0084913, -0.0023206, 0.0084516, -0.0107831, 0.0108119
6: -0.0093904, 0.0019855, -0.0093413, 0.0019816, -0.0113720, 0.0113268
7: -0.0060267, 0.0002018, -0.0058761, 0.0001842, -0.0062110, 0.0060778
8: -0.0136711, -0.0010166, -0.0136466, -0.0014824, -0.0121888, 0.0126300
9: -0.0048969, 0.0076831, -0.0048531, 0.0076688, -0.0125657, 0.0125362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100637, upper bound: 0.0101640
time: 1.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100499, upper bound: 0.0100680
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0070002, 0.0054353, -0.0065991, 0.0049103, -0.0119104, 0.0120344
1: 0.9970273, 1.0109147, 0.9970062, 1.0101185, -0.0119157, 0.0125564
2: -0.0064132, 0.0061117, -0.0058917, 0.0056963, -0.0121095, 0.0120034
3: -0.0002453, 0.0025768, -0.0001330, 0.0025854, -0.0024643, 0.0024146
4: -0.0073595, 0.0016797, -0.0068316, 0.0015437, -0.0089032, 0.0085113
5: -0.0024085, 0.0087695, -0.0022258, 0.0081090, -0.0105175, 0.0109953
6: -0.0097338, 0.0020126, -0.0089184, 0.0019481, -0.0116819, 0.0109310
7: -0.0058740, 0.0003243, -0.0058887, 0.0000334, -0.0059074, 0.0062130
8: -0.0138424, -0.0014887, -0.0134358, -0.0014435, -0.0123989, 0.0119471
9: -0.0052032, 0.0077836, -0.0044758, 0.0075451, -0.0127483, 0.0122594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098247, upper bound: 0.0100746
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098023, upper bound: 0.0099918
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0071057, 0.0055735, -0.0068071, 0.0051826, -0.0122883, 0.0123806
1: 0.9970256, 1.0111243, 0.9970243, 1.0105314, -0.0122882, 0.0127390
2: -0.0065505, 0.0062210, -0.0061622, 0.0059118, -0.0124622, 0.0123832
3: -0.0002749, 0.0025774, -0.0001913, 0.0025780, -0.0024828, 0.0024590
4: -0.0074984, 0.0017155, -0.0071054, 0.0016142, -0.0091127, 0.0088209
5: -0.0024566, 0.0089433, -0.0023206, 0.0084516, -0.0109081, 0.0112639
6: -0.0099484, 0.0020296, -0.0093413, 0.0019816, -0.0119300, 0.0113709
7: -0.0058751, 0.0004008, -0.0058761, 0.0001842, -0.0060594, 0.0062769
8: -0.0139494, -0.0014853, -0.0136466, -0.0014824, -0.0124670, 0.0121613
9: -0.0053946, 0.0078464, -0.0048531, 0.0076688, -0.0130634, 0.0126995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103265, upper bound: 0.0103822
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0103040, upper bound: 0.0102707
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0067367, 0.0050904, -0.0069208, 0.0053315, -0.0120681, 0.0120112
1: 0.9968106, 1.0103917, 0.9970033, 1.0107573, -0.0126245, 0.0119237
2: -0.0060706, 0.0058388, -0.0063100, 0.0060295, -0.0121001, 0.0121488
3: -0.0001715, 0.0026666, -0.0002231, 0.0025867, -0.0023231, 0.0025433
4: -0.0070127, 0.0015903, -0.0072551, 0.0016528, -0.0086655, 0.0088454
5: -0.0022885, 0.0083355, -0.0023724, 0.0086388, -0.0109273, 0.0107079
6: -0.0091981, 0.0019702, -0.0095725, 0.0019999, -0.0111980, 0.0115428
7: -0.0060256, 0.0001331, -0.0058908, 0.0002667, -0.0062923, 0.0060239
8: -0.0135752, -0.0010202, -0.0137619, -0.0014370, -0.0121382, 0.0127418
9: -0.0047253, 0.0076269, -0.0050593, 0.0077364, -0.0124617, 0.0126862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095479, upper bound: 0.0099020
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095321, upper bound: 0.0098336
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0068313, 0.0052142, -0.0071194, 0.0055914, -0.0124227, 0.0123336
1: 0.9968090, 1.0105793, 0.9970251, 1.0111514, -0.0129752, 0.0120947
2: -0.0061936, 0.0059368, -0.0065683, 0.0062352, -0.0124288, 0.0125051
3: -0.0001980, 0.0026673, -0.0002787, 0.0025776, -0.0023420, 0.0025845
4: -0.0071372, 0.0016224, -0.0075165, 0.0017202, -0.0088574, 0.0091389
5: -0.0023316, 0.0084913, -0.0024628, 0.0089659, -0.0112974, 0.0109541
6: -0.0093904, 0.0019855, -0.0099763, 0.0020318, -0.0114222, 0.0119617
7: -0.0060267, 0.0002018, -0.0058755, 0.0004108, -0.0064375, 0.0060773
8: -0.0136711, -0.0010166, -0.0139633, -0.0014842, -0.0121869, 0.0129466
9: -0.0048969, 0.0076831, -0.0054195, 0.0078545, -0.0127514, 0.0131026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100430, upper bound: 0.0101584
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0100309, upper bound: 0.0100617
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0070002, 0.0054353, -0.0069208, 0.0053315, -0.0123316, 0.0123562
1: 0.9970273, 1.0109147, 0.9970033, 1.0107573, -0.0124612, 0.0124462
2: -0.0064132, 0.0061117, -0.0063100, 0.0060295, -0.0124428, 0.0124218
3: -0.0002453, 0.0025768, -0.0002231, 0.0025867, -0.0024044, 0.0024612
4: -0.0073595, 0.0016797, -0.0072551, 0.0016528, -0.0090123, 0.0089348
5: -0.0024085, 0.0087695, -0.0023724, 0.0086388, -0.0110473, 0.0111419
6: -0.0097338, 0.0020126, -0.0095725, 0.0019999, -0.0117337, 0.0115852
7: -0.0058740, 0.0003243, -0.0058908, 0.0002667, -0.0061408, 0.0062150
8: -0.0138424, -0.0014887, -0.0137619, -0.0014370, -0.0124054, 0.0122733
9: -0.0052032, 0.0077836, -0.0050593, 0.0077364, -0.0129396, 0.0128429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097075, upper bound: 0.0100484
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096878, upper bound: 0.0099703
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0071057, 0.0055735, -0.0071194, 0.0055914, -0.0126972, 0.0126929
1: 0.9970256, 1.0111243, 0.9970251, 1.0111514, -0.0128056, 0.0126236
2: -0.0065505, 0.0062210, -0.0065683, 0.0062352, -0.0127857, 0.0127893
3: -0.0002749, 0.0025774, -0.0002787, 0.0025776, -0.0024220, 0.0024983
4: -0.0074984, 0.0017155, -0.0075165, 0.0017202, -0.0092186, 0.0092320
5: -0.0024566, 0.0089433, -0.0024628, 0.0089659, -0.0114224, 0.0114061
6: -0.0099484, 0.0020296, -0.0099763, 0.0020318, -0.0119803, 0.0120059
7: -0.0058751, 0.0004008, -0.0058755, 0.0004108, -0.0062859, 0.0062763
8: -0.0139494, -0.0014853, -0.0139633, -0.0014842, -0.0124652, 0.0124779
9: -0.0053946, 0.0078464, -0.0054195, 0.0078545, -0.0132491, 0.0132658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102928, upper bound: 0.0103685
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0102745, upper bound: 0.0102567
time: 1.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.04 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097089, upper bound: 0.0092208
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097089, upper bound: 0.0092258
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096351, upper bound: 0.0092031
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096351, upper bound: 0.0092096
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098503, upper bound: 0.0095510
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098503, upper bound: 0.0095561
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097905, upper bound: 0.0095398
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097905, upper bound: 0.0095484
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095941, upper bound: 0.0091423
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095941, upper bound: 0.0091494
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0094936, upper bound: 0.0091227
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0094936, upper bound: 0.0091279
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097595, upper bound: 0.0094669
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097595, upper bound: 0.0094853
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096887, upper bound: 0.0094485
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096887, upper bound: 0.0094660
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100476, upper bound: 0.0098038
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100476, upper bound: 0.0098056
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0099766, upper bound: 0.0097928
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0099766, upper bound: 0.0097951
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0102266, upper bound: 0.0101434
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0102266, upper bound: 0.0101547
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0101415, upper bound: 0.0101355
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0101415, upper bound: 0.0101453
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093474, upper bound: 0.0094397
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093343, upper bound: 0.0093524
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093474, upper bound: 0.0097037
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093343, upper bound: 0.0096281
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095384, upper bound: 0.0097787
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095184, upper bound: 0.0097075
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095384, upper bound: 0.0097787
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095184, upper bound: 0.0100499
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095224, upper bound: 0.0095929
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0094714, upper bound: 0.0094367
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098208, upper bound: 0.0097667
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097878, upper bound: 0.0096058
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098060, upper bound: 0.0097916
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097649, upper bound: 0.0096369
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100979, upper bound: 0.0099568
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100617, upper bound: 0.0098093
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093409, upper bound: 0.0094914
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092927, upper bound: 0.0093328
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097317, upper bound: 0.0096980
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096984, upper bound: 0.0095327
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095563, upper bound: 0.0096725
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095197, upper bound: 0.0095295
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100042, upper bound: 0.0098811
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0099680, upper bound: 0.0097252
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098056, upper bound: 0.0100712
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097951, upper bound: 0.0100052
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0101547, upper bound: 0.0102559
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0101453, upper bound: 0.0101814
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100189, upper bound: 0.0102381
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100027, upper bound: 0.0101493
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0104203, upper bound: 0.0104742
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0104047, upper bound: 0.0103912
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095542, upper bound: 0.0099342
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095390, upper bound: 0.0098648
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100488, upper bound: 0.0101749
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100401, upper bound: 0.0100897
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097123, upper bound: 0.0100652
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096884, upper bound: 0.0099977
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0102974, upper bound: 0.0103878
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0102846, upper bound: 0.0102870
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094437
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094471
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094437
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093584, upper bound: 0.0094471
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092396
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092442
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092397
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092971, upper bound: 0.0092442
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096177
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096325
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096177
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097270, upper bound: 0.0096325
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094142
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094301
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094142
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096798, upper bound: 0.0094301
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098664
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098664
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098664
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095552, upper bound: 0.0098744
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097790
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097886
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097790
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095406, upper bound: 0.0097886
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101233
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101361
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101233
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100325, upper bound: 0.0101361
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100142
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100309
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100142
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100166, upper bound: 0.0100309
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093462, upper bound: 0.0094808
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092906, upper bound: 0.0093195
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0093462, upper bound: 0.0094808
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0092906, upper bound: 0.0093195
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095564, upper bound: 0.0096608
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095223, upper bound: 0.0095158
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095564, upper bound: 0.0096608
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095223, upper bound: 0.0095158
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097342, upper bound: 0.0096789
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096895, upper bound: 0.0095106
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097342, upper bound: 0.0096789
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096895, upper bound: 0.0095106
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100051, upper bound: 0.0098704
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0099681, upper bound: 0.0097060
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100051, upper bound: 0.0098704
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0099681, upper bound: 0.0097060
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096486, upper bound: 0.0099241
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096281, upper bound: 0.0098573
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100637, upper bound: 0.0101640
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100499, upper bound: 0.0100680
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098247, upper bound: 0.0100746
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0098023, upper bound: 0.0099918
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0103265, upper bound: 0.0103822
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0103040, upper bound: 0.0102707
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095479, upper bound: 0.0099020
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0095321, upper bound: 0.0098336
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100430, upper bound: 0.0101584
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0100309, upper bound: 0.0100617
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0097075, upper bound: 0.0100484
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0096878, upper bound: 0.0099703
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0102928, upper bound: 0.0103685
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 1, lower bound: -0.0102745, upper bound: 0.0102567

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0059103, 0.0040085, -0.0062438, 0.0044451, -0.0103554, 0.0102523
1: 0.9967800, 1.0087507, 0.9968088, 1.0094130, -0.0111811, 0.0105246
2: -0.0049961, 0.0049830, -0.0054297, 0.0053284, -0.0103244, 0.0104127
3: 0.0000598, 0.0026793, -0.0000336, 0.0026674, -0.0021971, 0.0022830
4: -0.0059250, 0.0014813, -0.0063640, 0.0014680, -0.0073930, 0.0078452
5: -0.0019121, 0.0069746, -0.0020640, 0.0075239, -0.0094360, 0.0090386
6: -0.0075180, 0.0018373, -0.0081960, 0.0018909, -0.0094089, 0.0100333
7: -0.0060470, -0.0004663, -0.0060269, -0.0002244, -0.0058226, 0.0055606
8: -0.0127374, -0.0009539, -0.0130755, -0.0010161, -0.0117213, 0.0121217
9: -0.0032267, 0.0071354, -0.0038315, 0.0073337, -0.0105604, 0.0109669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084850, upper bound: 0.0077659
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082380, upper bound: 0.0077474
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0061811, 0.0043631, -0.0062438, 0.0044451, -0.0106262, 0.0106069
1: 0.9970112, 1.0092887, 0.9968088, 1.0094130, -0.0111126, 0.0111448
2: -0.0053482, 0.0052635, -0.0054297, 0.0053284, -0.0106766, 0.0106932
3: -0.0000160, 0.0025834, -0.0000336, 0.0026674, -0.0023321, 0.0022630
4: -0.0062815, 0.0014019, -0.0063640, 0.0014680, -0.0077495, 0.0077658
5: -0.0020355, 0.0074207, -0.0020640, 0.0075239, -0.0095593, 0.0094847
6: -0.0080686, 0.0018809, -0.0081960, 0.0018909, -0.0099596, 0.0100769
7: -0.0058853, -0.0002699, -0.0060269, -0.0002244, -0.0056609, 0.0057571
8: -0.0130120, -0.0014540, -0.0130755, -0.0010161, -0.0119959, 0.0116215
9: -0.0037178, 0.0072965, -0.0038315, 0.0073337, -0.0110516, 0.0111280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084850, upper bound: 0.0077659
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082380, upper bound: 0.0078737
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057684, 0.0038227, -0.0060438, 0.0041833, -0.0099517, 0.0098665
1: 0.9967846, 1.0084689, 0.9967105, 1.0090160, -0.0108679, 0.0103536
2: -0.0048115, 0.0048360, -0.0051697, 0.0051213, -0.0099328, 0.0100057
3: 0.0000995, 0.0026774, 0.0000224, 0.0027081, -0.0021961, 0.0022561
4: -0.0057382, 0.0014791, -0.0061007, 0.0015133, -0.0072515, 0.0075799
5: -0.0018475, 0.0067409, -0.0019729, 0.0071945, -0.0090420, 0.0087138
6: -0.0072294, 0.0018144, -0.0077894, 0.0018588, -0.0090882, 0.0096039
7: -0.0060438, -0.0005693, -0.0060957, -0.0003694, -0.0056744, 0.0055265
8: -0.0125935, -0.0009638, -0.0128728, -0.0008033, -0.0117902, 0.0119089
9: -0.0029693, 0.0070510, -0.0034688, 0.0072148, -0.0101841, 0.0105198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084234, upper bound: 0.0077411
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081482, upper bound: 0.0077191
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0060468, 0.0041872, -0.0060438, 0.0041833, -0.0102301, 0.0102310
1: 0.9970158, 1.0090218, 0.9967105, 1.0090160, -0.0107994, 0.0109716
2: -0.0051735, 0.0051243, -0.0051697, 0.0051213, -0.0102948, 0.0102940
3: 0.0000216, 0.0025815, 0.0000224, 0.0027081, -0.0023309, 0.0022360
4: -0.0061046, 0.0013727, -0.0061007, 0.0015133, -0.0076179, 0.0074734
5: -0.0019743, 0.0071994, -0.0019729, 0.0071945, -0.0091687, 0.0091723
6: -0.0077954, 0.0018592, -0.0077894, 0.0018588, -0.0096542, 0.0096487
7: -0.0058820, -0.0003673, -0.0060957, -0.0003694, -0.0055125, 0.0057284
8: -0.0128758, -0.0014641, -0.0128728, -0.0008033, -0.0120725, 0.0114087
9: -0.0034742, 0.0072166, -0.0034688, 0.0072148, -0.0106890, 0.0106854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084234, upper bound: 0.0078784
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081482, upper bound: 0.0078420
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060964, 0.0042522, -0.0063444, 0.0045768, -0.0106733, 0.0105966
1: 0.9968000, 1.0091205, 0.9968072, 1.0096128, -0.0113686, 0.0108668
2: -0.0052381, 0.0051758, -0.0055605, 0.0054326, -0.0106707, 0.0107363
3: 0.0000077, 0.0026709, -0.0000617, 0.0026680, -0.0022351, 0.0023026
4: -0.0061700, 0.0014720, -0.0064964, 0.0014687, -0.0076388, 0.0079684
5: -0.0019969, 0.0072812, -0.0021098, 0.0076895, -0.0096864, 0.0093910
6: -0.0078965, 0.0018672, -0.0084006, 0.0019071, -0.0098036, 0.0102678
7: -0.0060329, -0.0003313, -0.0060280, -0.0001514, -0.0058815, 0.0056967
8: -0.0129262, -0.0009974, -0.0131775, -0.0010128, -0.0119133, 0.0121801
9: -0.0035643, 0.0072461, -0.0040139, 0.0073936, -0.0109579, 0.0112601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095379, upper bound: 0.0093704
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095379, upper bound: 0.0095510
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063753, 0.0046173, -0.0063444, 0.0045768, -0.0109521, 0.0109617
1: 0.9970232, 1.0096741, 0.9968072, 1.0096128, -0.0112985, 0.0114862
2: -0.0056007, 0.0054646, -0.0055605, 0.0054326, -0.0110333, 0.0110251
3: -0.0000704, 0.0025784, -0.0000617, 0.0026680, -0.0023693, 0.0022830
4: -0.0065371, 0.0014678, -0.0064964, 0.0014687, -0.0080058, 0.0079641
5: -0.0021239, 0.0077404, -0.0021098, 0.0076895, -0.0098134, 0.0098503
6: -0.0084634, 0.0019121, -0.0084006, 0.0019071, -0.0103705, 0.0103127
7: -0.0058769, -0.0001290, -0.0060280, -0.0001514, -0.0057255, 0.0058990
8: -0.0132089, -0.0014799, -0.0131775, -0.0010128, -0.0121961, 0.0116976
9: -0.0040700, 0.0074120, -0.0040139, 0.0073936, -0.0114636, 0.0114259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095379, upper bound: 0.0093721
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095379, upper bound: 0.0095561
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0059709, 0.0040879, -0.0061551, 0.0043290, -0.0103000, 0.0102430
1: 0.9968047, 1.0088711, 0.9967088, 1.0092369, -0.0110520, 0.0107092
2: -0.0050749, 0.0050458, -0.0053144, 0.0052365, -0.0103115, 0.0103602
3: 0.0000428, 0.0026690, -0.0000087, 0.0027088, -0.0022366, 0.0022771
4: -0.0060048, 0.0014699, -0.0062472, 0.0015140, -0.0075188, 0.0077171
5: -0.0019397, 0.0070745, -0.0020236, 0.0073778, -0.0093175, 0.0090981
6: -0.0076412, 0.0018470, -0.0080157, 0.0018767, -0.0095179, 0.0098628
7: -0.0060297, -0.0004223, -0.0060968, -0.0002887, -0.0057410, 0.0056745
8: -0.0127989, -0.0010074, -0.0129856, -0.0008001, -0.0119988, 0.0119783
9: -0.0033366, 0.0071714, -0.0036707, 0.0072810, -0.0106177, 0.0108421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094782, upper bound: 0.0093576
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094782, upper bound: 0.0095398
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0062447, 0.0044463, -0.0061551, 0.0043290, -0.0105737, 0.0106014
1: 0.9970278, 1.0094149, 0.9967088, 1.0092369, -0.0109817, 0.0113192
2: -0.0054309, 0.0053293, -0.0053144, 0.0052365, -0.0106674, 0.0106437
3: -0.0000338, 0.0025765, -0.0000087, 0.0027088, -0.0023697, 0.0022573
4: -0.0063651, 0.0014234, -0.0062472, 0.0015140, -0.0078791, 0.0076707
5: -0.0020644, 0.0075253, -0.0020236, 0.0073778, -0.0094422, 0.0095489
6: -0.0081978, 0.0018911, -0.0080157, 0.0018767, -0.0100745, 0.0099068
7: -0.0058736, -0.0002238, -0.0060968, -0.0002887, -0.0055849, 0.0058730
8: -0.0130764, -0.0014901, -0.0129856, -0.0008001, -0.0122763, 0.0114955
9: -0.0038331, 0.0073343, -0.0036707, 0.0072810, -0.0111141, 0.0110050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094782, upper bound: 0.0093604
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094782, upper bound: 0.0095484
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0059103, 0.0040085, -0.0065543, 0.0048516, -0.0107619, 0.0105628
1: 0.9967800, 1.0087507, 0.9968157, 1.0100294, -0.0118489, 0.0105611
2: -0.0049961, 0.0049830, -0.0058334, 0.0056499, -0.0106460, 0.0108164
3: 0.0000598, 0.0026793, -0.0001205, 0.0026645, -0.0022096, 0.0024081
4: -0.0059250, 0.0014813, -0.0067726, 0.0015285, -0.0074535, 0.0082539
5: -0.0019121, 0.0069746, -0.0022054, 0.0080351, -0.0099473, 0.0091800
6: -0.0075180, 0.0018373, -0.0088273, 0.0019409, -0.0094589, 0.0106645
7: -0.0060470, -0.0004663, -0.0060221, 0.0000008, -0.0060479, 0.0055558
8: -0.0127374, -0.0009539, -0.0133903, -0.0010310, -0.0117064, 0.0124364
9: -0.0032267, 0.0071354, -0.0043945, 0.0075184, -0.0107451, 0.0115299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083727, upper bound: 0.0076624
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081434, upper bound: 0.0076446
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0061811, 0.0043631, -0.0065543, 0.0048516, -0.0110327, 0.0109174
1: 0.9970112, 1.0092887, 0.9968157, 1.0100294, -0.0117804, 0.0111814
2: -0.0053482, 0.0052635, -0.0058334, 0.0056499, -0.0109981, 0.0110969
3: -0.0000160, 0.0025834, -0.0001205, 0.0026645, -0.0023447, 0.0023881
4: -0.0062815, 0.0014019, -0.0067726, 0.0015285, -0.0078099, 0.0081745
5: -0.0020355, 0.0074207, -0.0022054, 0.0080351, -0.0100706, 0.0096261
6: -0.0080686, 0.0018809, -0.0088273, 0.0019409, -0.0100095, 0.0107081
7: -0.0058853, -0.0002699, -0.0060221, 0.0000008, -0.0058861, 0.0057522
8: -0.0130120, -0.0014540, -0.0133903, -0.0010310, -0.0119810, 0.0119363
9: -0.0037178, 0.0072965, -0.0043945, 0.0075184, -0.0112362, 0.0116910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083727, upper bound: 0.0078189
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081434, upper bound: 0.0077963
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057684, 0.0038227, -0.0063648, 0.0046035, -0.0103719, 0.0101875
1: 0.9967846, 1.0084689, 0.9967173, 1.0096532, -0.0115395, 0.0103899
2: -0.0048115, 0.0048360, -0.0055870, 0.0054537, -0.0102652, 0.0104230
3: 0.0000995, 0.0026774, -0.0000674, 0.0027053, -0.0022101, 0.0023816
4: -0.0057382, 0.0014791, -0.0065232, 0.0015101, -0.0072483, 0.0080024
5: -0.0018475, 0.0067409, -0.0021191, 0.0077231, -0.0095706, 0.0088600
6: -0.0072294, 0.0018144, -0.0084420, 0.0019104, -0.0091399, 0.0102565
7: -0.0060438, -0.0005693, -0.0060909, -0.0001366, -0.0059072, 0.0055216
8: -0.0125935, -0.0009638, -0.0131982, -0.0008183, -0.0117752, 0.0122344
9: -0.0029693, 0.0070510, -0.0040509, 0.0074057, -0.0103750, 0.0111019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083124, upper bound: 0.0076294
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0080326, upper bound: 0.0076047
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0060468, 0.0041872, -0.0063648, 0.0046035, -0.0106503, 0.0105520
1: 0.9970158, 1.0090218, 0.9967173, 1.0096532, -0.0114710, 0.0110079
2: -0.0051735, 0.0051243, -0.0055870, 0.0054537, -0.0106272, 0.0107113
3: 0.0000216, 0.0025815, -0.0000674, 0.0027053, -0.0023449, 0.0023615
4: -0.0061046, 0.0013727, -0.0065232, 0.0015101, -0.0076147, 0.0078959
5: -0.0019743, 0.0071994, -0.0021191, 0.0077231, -0.0096974, 0.0093185
6: -0.0077954, 0.0018592, -0.0084420, 0.0019104, -0.0097058, 0.0103013
7: -0.0058820, -0.0003673, -0.0060909, -0.0001366, -0.0057454, 0.0057236
8: -0.0128758, -0.0014641, -0.0131982, -0.0008183, -0.0120575, 0.0117341
9: -0.0034742, 0.0072166, -0.0040509, 0.0074057, -0.0108799, 0.0112675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083124, upper bound: 0.0077819
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0080326, upper bound: 0.0077559
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060964, 0.0042522, -0.0066518, 0.0049792, -0.0110757, 0.0109040
1: 0.9968000, 1.0091205, 0.9968141, 1.0102230, -0.0120274, 0.0109231
2: -0.0052381, 0.0051758, -0.0059602, 0.0057509, -0.0109891, 0.0111360
3: 0.0000077, 0.0026709, -0.0001478, 0.0026652, -0.0022597, 0.0024244
4: -0.0061700, 0.0014720, -0.0069010, 0.0015615, -0.0077316, 0.0083730
5: -0.0019969, 0.0072812, -0.0022498, 0.0081958, -0.0101926, 0.0095310
6: -0.0078965, 0.0018672, -0.0090255, 0.0019566, -0.0098531, 0.0108928
7: -0.0060329, -0.0003313, -0.0060232, 0.0000716, -0.0061045, 0.0056919
8: -0.0129262, -0.0009974, -0.0134892, -0.0010276, -0.0118985, 0.0124918
9: -0.0035643, 0.0072461, -0.0045714, 0.0075764, -0.0111407, 0.0118175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094100, upper bound: 0.0092844
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094100, upper bound: 0.0094669
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063753, 0.0046173, -0.0066518, 0.0049792, -0.0113545, 0.0112691
1: 0.9970232, 1.0096741, 0.9968141, 1.0102230, -0.0119573, 0.0115422
2: -0.0056007, 0.0054646, -0.0059602, 0.0057509, -0.0113516, 0.0114248
3: -0.0000704, 0.0025784, -0.0001478, 0.0026652, -0.0023939, 0.0024049
4: -0.0065371, 0.0014678, -0.0069010, 0.0015615, -0.0080986, 0.0083687
5: -0.0021239, 0.0077404, -0.0022498, 0.0081958, -0.0103196, 0.0099903
6: -0.0084634, 0.0019121, -0.0090255, 0.0019566, -0.0104200, 0.0109376
7: -0.0058769, -0.0001290, -0.0060232, 0.0000716, -0.0059484, 0.0058942
8: -0.0132089, -0.0014799, -0.0134892, -0.0010276, -0.0121813, 0.0120092
9: -0.0040700, 0.0074120, -0.0045714, 0.0075764, -0.0116464, 0.0119834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094100, upper bound: 0.0092859
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094100, upper bound: 0.0094853
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0059709, 0.0040879, -0.0064651, 0.0047348, -0.0107057, 0.0105530
1: 0.9968047, 1.0088711, 0.9967157, 1.0098524, -0.0117147, 0.0107661
2: -0.0050749, 0.0050458, -0.0057175, 0.0055576, -0.0106325, 0.0107632
3: 0.0000428, 0.0026690, -0.0000955, 0.0027059, -0.0022615, 0.0023981
4: -0.0060048, 0.0014699, -0.0066552, 0.0015108, -0.0075156, 0.0081251
5: -0.0019397, 0.0070745, -0.0021648, 0.0078883, -0.0098280, 0.0092393
6: -0.0076412, 0.0018470, -0.0086459, 0.0019266, -0.0095678, 0.0104930
7: -0.0060297, -0.0004223, -0.0060920, -0.0000639, -0.0059659, 0.0056696
8: -0.0127989, -0.0010074, -0.0132999, -0.0008149, -0.0119839, 0.0122925
9: -0.0033366, 0.0071714, -0.0042328, 0.0074654, -0.0108020, 0.0114043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092984, upper bound: 0.0092652
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092984, upper bound: 0.0094485
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0062447, 0.0044463, -0.0064651, 0.0047348, -0.0109795, 0.0109114
1: 0.9970278, 1.0094149, 0.9967157, 1.0098524, -0.0116443, 0.0113761
2: -0.0054309, 0.0053293, -0.0057175, 0.0055576, -0.0109884, 0.0110467
3: -0.0000338, 0.0025765, -0.0000955, 0.0027059, -0.0023946, 0.0023783
4: -0.0063651, 0.0014234, -0.0066552, 0.0015108, -0.0078760, 0.0080787
5: -0.0020644, 0.0075253, -0.0021648, 0.0078883, -0.0099527, 0.0096901
6: -0.0081978, 0.0018911, -0.0086459, 0.0019266, -0.0101243, 0.0105370
7: -0.0058736, -0.0002238, -0.0060920, -0.0000639, -0.0058097, 0.0058682
8: -0.0130764, -0.0014901, -0.0132999, -0.0008149, -0.0122615, 0.0118098
9: -0.0038331, 0.0073343, -0.0042328, 0.0074654, -0.0112984, 0.0115671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092984, upper bound: 0.0092672
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092984, upper bound: 0.0094660
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0061920, 0.0043773, -0.0063819, 0.0046260, -0.0108179, 0.0107592
1: 0.9967880, 1.0093100, 0.9968068, 1.0096872, -0.0115065, 0.0109661
2: -0.0053623, 0.0052747, -0.0056093, 0.0054715, -0.0108338, 0.0108840
3: -0.0000191, 0.0026760, -0.0000722, 0.0026682, -0.0022199, 0.0023474
4: -0.0062958, 0.0014776, -0.0065458, 0.0014700, -0.0077658, 0.0080234
5: -0.0020404, 0.0074385, -0.0021269, 0.0077514, -0.0097918, 0.0095654
6: -0.0080907, 0.0018826, -0.0084769, 0.0019132, -0.0100038, 0.0103595
7: -0.0060414, -0.0002620, -0.0060283, -0.0001242, -0.0059173, 0.0057663
8: -0.0130230, -0.0009712, -0.0132156, -0.0010119, -0.0120111, 0.0122444
9: -0.0037375, 0.0073029, -0.0040820, 0.0074159, -0.0111534, 0.0113849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095447, upper bound: 0.0095224
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095447, upper bound: 0.0098038
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0064551, 0.0047217, -0.0063819, 0.0046260, -0.0110811, 0.0111037
1: 0.9970095, 1.0098321, 0.9968068, 1.0096872, -0.0114401, 0.0115914
2: -0.0057045, 0.0055472, -0.0056093, 0.0054715, -0.0111759, 0.0111566
3: -0.0000927, 0.0025841, -0.0000722, 0.0026682, -0.0023598, 0.0023253
4: -0.0066421, 0.0014948, -0.0065458, 0.0014700, -0.0081121, 0.0080406
5: -0.0021602, 0.0078719, -0.0021269, 0.0077514, -0.0099116, 0.0099988
6: -0.0086256, 0.0019249, -0.0084769, 0.0019132, -0.0105388, 0.0104019
7: -0.0058864, -0.0000711, -0.0060283, -0.0001242, -0.0057623, 0.0059572
8: -0.0132898, -0.0014504, -0.0132156, -0.0010119, -0.0122779, 0.0117652
9: -0.0042147, 0.0074594, -0.0040820, 0.0074159, -0.0116306, 0.0115414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095447, upper bound: 0.0095224
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095447, upper bound: 0.0098056
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0060661, 0.0042125, -0.0061933, 0.0043790, -0.0104451, 0.0104058
1: 0.9967929, 1.0090599, 0.9967087, 1.0093126, -0.0112112, 0.0108157
2: -0.0051987, 0.0051444, -0.0053640, 0.0052760, -0.0104747, 0.0105084
3: 0.0000162, 0.0026740, -0.0000194, 0.0027089, -0.0022183, 0.0023224
4: -0.0061301, 0.0014754, -0.0062974, 0.0015141, -0.0076442, 0.0077728
5: -0.0019831, 0.0072312, -0.0020410, 0.0074406, -0.0094237, 0.0092722
6: -0.0078348, 0.0018624, -0.0080933, 0.0018828, -0.0097176, 0.0099556
7: -0.0060381, -0.0003533, -0.0060970, -0.0002610, -0.0057771, 0.0057437
8: -0.0128954, -0.0009814, -0.0130243, -0.0007995, -0.0120959, 0.0120429
9: -0.0035093, 0.0072281, -0.0037399, 0.0073037, -0.0108130, 0.0109679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093705, upper bound: 0.0094706
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093705, upper bound: 0.0097928
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0063256, 0.0045523, -0.0061933, 0.0043790, -0.0107046, 0.0107455
1: 0.9970142, 1.0095754, 0.9967087, 1.0093126, -0.0111448, 0.0114292
2: -0.0055361, 0.0054131, -0.0053640, 0.0052760, -0.0108122, 0.0107771
3: -0.0000565, 0.0025822, -0.0000194, 0.0027089, -0.0023564, 0.0023001
4: -0.0064717, 0.0014509, -0.0062974, 0.0015141, -0.0079858, 0.0077483
5: -0.0021013, 0.0076586, -0.0020410, 0.0074406, -0.0095419, 0.0096996
6: -0.0083624, 0.0019041, -0.0080933, 0.0018828, -0.0102452, 0.0099974
7: -0.0058832, -0.0001650, -0.0060970, -0.0002610, -0.0056222, 0.0059320
8: -0.0131585, -0.0014603, -0.0130243, -0.0007995, -0.0123590, 0.0115640
9: -0.0039799, 0.0073824, -0.0037399, 0.0073037, -0.0112836, 0.0111223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093705, upper bound: 0.0094714
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093705, upper bound: 0.0097951
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0063836, 0.0046281, -0.0064815, 0.0047563, -0.0111399, 0.0111096
1: 0.9968044, 1.0096904, 0.9968053, 1.0098847, -0.0116914, 0.0113216
2: -0.0056115, 0.0054731, -0.0057388, 0.0055746, -0.0111860, 0.0112119
3: -0.0000727, 0.0026692, -0.0001001, 0.0026688, -0.0022652, 0.0023690
4: -0.0065480, 0.0014706, -0.0066768, 0.0015038, -0.0080517, 0.0081474
5: -0.0021277, 0.0077541, -0.0021723, 0.0079153, -0.0100430, 0.0099263
6: -0.0084802, 0.0019134, -0.0086793, 0.0019292, -0.0104094, 0.0105927
7: -0.0060300, -0.0001230, -0.0060294, -0.0000520, -0.0059780, 0.0059064
8: -0.0132172, -0.0010065, -0.0133165, -0.0010085, -0.0122088, 0.0123100
9: -0.0040850, 0.0074169, -0.0042626, 0.0074751, -0.0115601, 0.0116795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097136, upper bound: 0.0098113
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097136, upper bound: 0.0101434
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0066656, 0.0049974, -0.0064815, 0.0047563, -0.0114219, 0.0114788
1: 0.9970275, 1.0102504, 0.9968053, 1.0098847, -0.0116239, 0.0119617
2: -0.0059782, 0.0057652, -0.0057388, 0.0055746, -0.0115528, 0.0115040
3: -0.0001517, 0.0025766, -0.0001001, 0.0026688, -0.0024049, 0.0023488
4: -0.0069192, 0.0015662, -0.0066768, 0.0015038, -0.0084229, 0.0082431
5: -0.0022561, 0.0082186, -0.0021723, 0.0079153, -0.0101714, 0.0103908
6: -0.0090537, 0.0019588, -0.0086793, 0.0019292, -0.0109829, 0.0106381
7: -0.0058738, 0.0000816, -0.0060294, -0.0000520, -0.0058219, 0.0061110
8: -0.0135032, -0.0014893, -0.0133165, -0.0010085, -0.0124947, 0.0118272
9: -0.0045965, 0.0075846, -0.0042626, 0.0074751, -0.0120716, 0.0118472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097136, upper bound: 0.0098208
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0097136, upper bound: 0.0101547
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0062653, 0.0044733, -0.0063036, 0.0045233, -0.0107887, 0.0107769
1: 0.9968092, 1.0094556, 0.9967071, 1.0095317, -0.0114017, 0.0111664
2: -0.0054577, 0.0053507, -0.0055074, 0.0053903, -0.0108480, 0.0108581
3: -0.0000396, 0.0026672, -0.0000503, 0.0027095, -0.0022648, 0.0023443
4: -0.0063923, 0.0014678, -0.0064426, 0.0015148, -0.0079071, 0.0079105
5: -0.0020738, 0.0075593, -0.0020912, 0.0076223, -0.0096961, 0.0096505
6: -0.0082398, 0.0018944, -0.0083175, 0.0019006, -0.0101403, 0.0102119
7: -0.0060266, -0.0002088, -0.0060980, -0.0001810, -0.0058456, 0.0058893
8: -0.0130973, -0.0010169, -0.0131361, -0.0007962, -0.0123012, 0.0121192
9: -0.0038705, 0.0073465, -0.0039399, 0.0073693, -0.0112398, 0.0112864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095445, upper bound: 0.0097814
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095445, upper bound: 0.0097814
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0065452, 0.0048397, -0.0063036, 0.0045233, -0.0110686, 0.0111433
1: 0.9970321, 1.0100117, 0.9967071, 1.0095317, -0.0113341, 0.0118047
2: -0.0058217, 0.0056406, -0.0055074, 0.0053903, -0.0112119, 0.0111480
3: -0.0001180, 0.0025747, -0.0000503, 0.0027095, -0.0024029, 0.0023239
4: -0.0067607, 0.0015254, -0.0064426, 0.0015148, -0.0082756, 0.0079680
5: -0.0022013, 0.0080203, -0.0020912, 0.0076223, -0.0098236, 0.0101115
6: -0.0088089, 0.0019394, -0.0083175, 0.0019006, -0.0107094, 0.0102570
7: -0.0058706, -0.0000057, -0.0060980, -0.0001810, -0.0056895, 0.0060923
8: -0.0133811, -0.0014994, -0.0131361, -0.0007962, -0.0125849, 0.0116367
9: -0.0043782, 0.0075130, -0.0039399, 0.0073693, -0.0117475, 0.0114529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095445, upper bound: 0.0097878
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095445, upper bound: 0.0101453
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0063666, 0.0046059, -0.0064153, 0.0046696, -0.0110362, 0.0110212
1: 0.9970153, 1.0096569, 0.9968110, 1.0097533, -0.0115091, 0.0115282
2: -0.0055895, 0.0054556, -0.0056527, 0.0055060, -0.0110954, 0.0111083
3: -0.0000680, 0.0025817, -0.0000816, 0.0026664, -0.0023817, 0.0023432
4: -0.0065256, 0.0014648, -0.0065897, 0.0014813, -0.0080069, 0.0080545
5: -0.0021199, 0.0077262, -0.0021421, 0.0078063, -0.0099262, 0.0098683
6: -0.0084458, 0.0019107, -0.0085446, 0.0019185, -0.0103643, 0.0104554
7: -0.0058823, -0.0001353, -0.0060253, -0.0001000, -0.0057823, 0.0058901
8: -0.0132001, -0.0014630, -0.0132494, -0.0010209, -0.0121792, 0.0117864
9: -0.0040543, 0.0074068, -0.0041425, 0.0074357, -0.0114900, 0.0115493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083744, upper bound: 0.0079861
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0079777, upper bound: 0.0079461
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0061689, 0.0043470, -0.0062888, 0.0045040, -0.0106729, 0.0106358
1: 0.9969198, 1.0092641, 0.9968156, 1.0095022, -0.0113558, 0.0112381
2: -0.0053323, 0.0052508, -0.0054882, 0.0053750, -0.0107073, 0.0107390
3: -0.0000126, 0.0026213, -0.0000462, 0.0026646, -0.0023569, 0.0023439
4: -0.0062654, 0.0014168, -0.0064232, 0.0014649, -0.0077303, 0.0078400
5: -0.0020299, 0.0074005, -0.0020845, 0.0075980, -0.0096278, 0.0094850
6: -0.0080437, 0.0018789, -0.0082875, 0.0018982, -0.0099419, 0.0101664
7: -0.0059492, -0.0002787, -0.0060222, -0.0001918, -0.0057574, 0.0057435
8: -0.0129996, -0.0012565, -0.0131211, -0.0010306, -0.0119690, 0.0118646
9: -0.0036956, 0.0072892, -0.0039131, 0.0073605, -0.0110561, 0.0112023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083459, upper bound: 0.0079345
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0079443, upper bound: 0.0078933
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0063666, 0.0046059, -0.0067145, 0.0050614, -0.0114280, 0.0113205
1: 0.9970153, 1.0096569, 0.9968114, 1.0103475, -0.0119759, 0.0113776
2: -0.0055895, 0.0054556, -0.0060418, 0.0058159, -0.0114053, 0.0114974
3: -0.0000680, 0.0025817, -0.0001654, 0.0026663, -0.0023109, 0.0023720
4: -0.0065256, 0.0014648, -0.0069835, 0.0015828, -0.0081085, 0.0084484
5: -0.0021199, 0.0077262, -0.0022784, 0.0082991, -0.0104191, 0.0100046
6: -0.0084458, 0.0019107, -0.0091531, 0.0019667, -0.0104125, 0.0110638
7: -0.0058823, -0.0001353, -0.0060251, 0.0001171, -0.0059994, 0.0058898
8: -0.0132001, -0.0014630, -0.0135528, -0.0010218, -0.0121782, 0.0120898
9: -0.0040543, 0.0074068, -0.0046852, 0.0076137, -0.0116680, 0.0120920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098137, upper bound: 0.0096281
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098137, upper bound: 0.0096281
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0061689, 0.0043470, -0.0065922, 0.0049012, -0.0110701, 0.0109392
1: 0.9969198, 1.0092641, 0.9968161, 1.0101047, -0.0118279, 0.0110791
2: -0.0053323, 0.0052508, -0.0058827, 0.0056892, -0.0110215, 0.0111335
3: -0.0000126, 0.0026213, -0.0001311, 0.0026643, -0.0022856, 0.0023730
4: -0.0062654, 0.0014168, -0.0068225, 0.0015413, -0.0078067, 0.0082393
5: -0.0020299, 0.0074005, -0.0022227, 0.0080976, -0.0101274, 0.0096231
6: -0.0080437, 0.0018789, -0.0089043, 0.0019470, -0.0099907, 0.0107832
7: -0.0059492, -0.0002787, -0.0060218, 0.0000283, -0.0059775, 0.0057431
8: -0.0129996, -0.0012565, -0.0134287, -0.0010319, -0.0119677, 0.0121722
9: -0.0036956, 0.0072892, -0.0044632, 0.0075409, -0.0112366, 0.0117524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087443, upper bound: 0.0081772
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083606, upper bound: 0.0081352
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0065531, 0.0048500, -0.0065144, 0.0047993, -0.0113524, 0.0113644
1: 0.9970332, 1.0100271, 0.9968094, 1.0099500, -0.0116961, 0.0118745
2: -0.0058318, 0.0056487, -0.0057815, 0.0056086, -0.0114404, 0.0114302
3: -0.0001201, 0.0025743, -0.0001093, 0.0026671, -0.0024317, 0.0023642
4: -0.0067710, 0.0015280, -0.0067201, 0.0015149, -0.0082859, 0.0082481
5: -0.0022049, 0.0080332, -0.0021872, 0.0079694, -0.0101743, 0.0102204
6: -0.0088248, 0.0019407, -0.0087461, 0.0019345, -0.0107593, 0.0106868
7: -0.0058698, -0.0000001, -0.0060265, -0.0000281, -0.0058417, 0.0060264
8: -0.0133891, -0.0015017, -0.0133498, -0.0010174, -0.0123717, 0.0118481
9: -0.0043924, 0.0075177, -0.0043222, 0.0074947, -0.0118870, 0.0118399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091825, upper bound: 0.0095907
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091825, upper bound: 0.0097787
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063762, 0.0046184, -0.0063898, 0.0046362, -0.0110124, 0.0110082
1: 0.9969357, 1.0096757, 0.9968140, 1.0097029, -0.0115396, 0.0115980
2: -0.0056018, 0.0054655, -0.0056195, 0.0054796, -0.0110814, 0.0110850
3: -0.0000706, 0.0026147, -0.0000744, 0.0026652, -0.0024075, 0.0023637
4: -0.0065382, 0.0014680, -0.0065561, 0.0014727, -0.0080109, 0.0080242
5: -0.0021243, 0.0077419, -0.0021305, 0.0077643, -0.0098886, 0.0098724
6: -0.0084651, 0.0019122, -0.0084928, 0.0019144, -0.0103796, 0.0104051
7: -0.0059381, -0.0001284, -0.0060233, -0.0001185, -0.0058196, 0.0058950
8: -0.0132097, -0.0012907, -0.0132235, -0.0010272, -0.0121826, 0.0119328
9: -0.0040716, 0.0074125, -0.0040962, 0.0074206, -0.0114921, 0.0115087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091562, upper bound: 0.0095028
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0091562, upper bound: 0.0097075
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0065531, 0.0048500, -0.0068090, 0.0051850, -0.0117381, 0.0116590
1: 0.9970332, 1.0100271, 0.9968098, 1.0105352, -0.0121434, 0.0117381
2: -0.0058318, 0.0056487, -0.0061646, 0.0059137, -0.0117455, 0.0118133
3: -0.0001201, 0.0025743, -0.0001918, 0.0026669, -0.0023643, 0.0023887
4: -0.0067710, 0.0015280, -0.0071078, 0.0016149, -0.0083859, 0.0086359
5: -0.0022049, 0.0080332, -0.0023214, 0.0084546, -0.0106594, 0.0103546
6: -0.0088248, 0.0019407, -0.0093451, 0.0019819, -0.0108067, 0.0112858
7: -0.0058698, -0.0000001, -0.0060262, 0.0001856, -0.0060554, 0.0060261
8: -0.0133891, -0.0015017, -0.0136485, -0.0010183, -0.0123708, 0.0121468
9: -0.0043924, 0.0075177, -0.0048564, 0.0076699, -0.0120622, 0.0123741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095851, upper bound: 0.0098959
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095851, upper bound: 0.0101421
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0063762, 0.0046184, -0.0066885, 0.0050273, -0.0114035, 0.0113069
1: 0.9969357, 1.0096757, 0.9968144, 1.0102960, -0.0119909, 0.0114563
2: -0.0056018, 0.0054655, -0.0060080, 0.0057890, -0.0113908, 0.0114735
3: -0.0000706, 0.0026147, -0.0001581, 0.0026650, -0.0023399, 0.0023898
4: -0.0065382, 0.0014680, -0.0069493, 0.0015740, -0.0081122, 0.0084174
5: -0.0021243, 0.0077419, -0.0022666, 0.0082563, -0.0103806, 0.0100084
6: -0.0084651, 0.0019122, -0.0091002, 0.0019625, -0.0104277, 0.0110125
7: -0.0059381, -0.0001284, -0.0060229, 0.0000982, -0.0060363, 0.0058945
8: -0.0132097, -0.0012907, -0.0135264, -0.0010285, -0.0121812, 0.0122357
9: -0.0040716, 0.0074125, -0.0046380, 0.0075983, -0.0116698, 0.0120505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095624, upper bound: 0.0098104
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095624, upper bound: 0.0098104
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0059887, 0.0041111, -0.0064418, 0.0047043, -0.0106930, 0.0105529
1: 0.9968048, 1.0089062, 0.9970090, 1.0098059, -0.0116559, 0.0106206
2: -0.0050980, 0.0050642, -0.0056872, 0.0055334, -0.0106314, 0.0107513
3: 0.0000378, 0.0026690, -0.0000890, 0.0025843, -0.0021959, 0.0024024
4: -0.0060282, 0.0014698, -0.0066246, 0.0014903, -0.0075185, 0.0080944
5: -0.0019478, 0.0071037, -0.0021542, 0.0078499, -0.0097977, 0.0092579
6: -0.0076773, 0.0018499, -0.0085986, 0.0019228, -0.0096001, 0.0104485
7: -0.0060296, -0.0004095, -0.0058868, -0.0000808, -0.0059489, 0.0054774
8: -0.0128169, -0.0010077, -0.0132763, -0.0014492, -0.0113677, 0.0122685
9: -0.0033688, 0.0071820, -0.0041906, 0.0074515, -0.0108203, 0.0113726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092233, upper bound: 0.0095794
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092233, upper bound: 0.0095929
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0057710, 0.0038262, -0.0063075, 0.0045285, -0.0102995, 0.0101337
1: 0.9967119, 1.0084742, 0.9970136, 1.0095394, -0.0114884, 0.0102748
2: -0.0048150, 0.0048388, -0.0055125, 0.0053943, -0.0102093, 0.0103513
3: 0.0000988, 0.0027076, -0.0000514, 0.0025824, -0.0021682, 0.0023985
4: -0.0057417, 0.0015127, -0.0064478, 0.0014447, -0.0071865, 0.0079605
5: -0.0018487, 0.0067453, -0.0020930, 0.0076287, -0.0094774, 0.0088383
6: -0.0072349, 0.0018149, -0.0083255, 0.0019012, -0.0091361, 0.0101404
7: -0.0060948, -0.0005673, -0.0058835, -0.0001782, -0.0059166, 0.0053162
8: -0.0125962, -0.0008063, -0.0131401, -0.0014593, -0.0111370, 0.0123338
9: -0.0029742, 0.0070526, -0.0039470, 0.0073716, -0.0103458, 0.0109996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.41 + 596.70 = 600.11 seconds
