## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00506088


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037831, -0.0004361, -0.0037831, -0.0004361, -0.0033470, 0.0033470)
1: (-0.0041680, 0.0040053, -0.0041680, 0.0040053, -0.0063485, 0.0063485)
2: (0.0031502, 0.0100214, 0.0031502, 0.0100214, -0.0050988, 0.0050988)
3: (-0.0045141, -0.0034873, -0.0045141, -0.0034873, -0.0010269, 0.0010269)
4: (0.0016619, 0.0078023, 0.0016619, 0.0078023, -0.0055977, 0.0055977)
5: (-0.0029737, 0.0031841, -0.0029737, 0.0031841, -0.0053953, 0.0053953)
6: (-0.0068101, -0.0033009, -0.0068101, -0.0033009, -0.0033923, 0.0033923)
7: (-0.0023583, 0.0037299, -0.0023583, 0.0037299, -0.0057371, 0.0057371)
8: (-0.0008638, 0.0000232, -0.0008638, 0.0000232, -0.0008870, 0.0008870)
9: (0.9972829, 1.0118368, 0.9972829, 1.0118368, -0.0112919, 0.0112919)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.59 = 2.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0063784, upper bound: 0.0063784

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058086, upper bound: 0.0060466
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060786, upper bound: 0.0060786
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 9, lower bound: -0.0058086, upper bound: 0.0060466
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 9, lower bound: -0.0060786, upper bound: 0.0060786

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0033132, -0.0004410, -0.0036254, -0.0004459, -0.0028672, 0.0031844
1: -0.0042638, 0.0028398, -0.0041607, 0.0036135, -0.0056032, 0.0050041
2: 0.0030449, 0.0092882, 0.0031558, 0.0097732, -0.0047002, 0.0042054
3: -0.0044283, -0.0034925, -0.0044857, -0.0034924, -0.0009359, 0.0009932
4: 0.0019845, 0.0077819, 0.0017702, 0.0077627, -0.0052415, 0.0054606
5: -0.0029655, 0.0027473, -0.0029374, 0.0030391, -0.0052303, 0.0049344
6: -0.0067629, -0.0034038, -0.0067793, -0.0033361, -0.0033085, 0.0032590
7: -0.0020410, 0.0037482, -0.0022481, 0.0037023, -0.0053315, 0.0055896
8: -0.0008488, -0.0000382, -0.0008569, 0.0000035, -0.0008523, 0.0008187
9: 0.9986828, 1.0119581, 0.9977472, 1.0117953, -0.0097058, 0.0106937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0054725
time: 0.70 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0058611
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0036903, -0.0004433, -0.0037773, -0.0004366, -0.0032538, 0.0033340
1: -0.0041628, 0.0037654, -0.0041677, 0.0039904, -0.0063317, 0.0050222
2: 0.0031541, 0.0098518, 0.0031504, 0.0100111, -0.0050812, 0.0043770
3: -0.0044941, -0.0034911, -0.0045129, -0.0034875, -0.0010066, 0.0010218
4: 0.0017589, 0.0077763, 0.0016677, 0.0078007, -0.0054727, 0.0055621
5: -0.0029502, 0.0030703, -0.0029723, 0.0031773, -0.0053617, 0.0051926
6: -0.0067904, -0.0033415, -0.0068089, -0.0033033, -0.0033695, 0.0033605
7: -0.0022586, 0.0037117, -0.0023523, 0.0037288, -0.0055846, 0.0057024
8: -0.0008592, 0.0000083, -0.0008635, 0.0000223, -0.0008814, 0.0008696
9: 0.9976149, 1.0118092, 0.9973032, 1.0118351, -0.0102106, 0.0112422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058906, upper bound: 0.0055029
time: 0.72 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058906, upper bound: 0.0058907
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.72 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0054725
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0058611
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 9, lower bound: -0.0058906, upper bound: 0.0055029
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 9, lower bound: -0.0058906, upper bound: 0.0058907

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0036214, -0.0005010, -0.0027676, 0.0029998
1: -0.0037844, 0.0026818, -0.0040104, 0.0036078, -0.0050656, 0.0045141
2: 0.0034838, 0.0091255, 0.0032888, 0.0097660, -0.0042456, 0.0037970
3: -0.0044250, -0.0035443, -0.0044844, -0.0035079, -0.0009170, 0.0009401
4: 0.0021194, 0.0074215, 0.0017779, 0.0076525, -0.0050238, 0.0051094
5: -0.0025576, 0.0025986, -0.0028105, 0.0030327, -0.0048183, 0.0046358
6: -0.0065863, -0.0034534, -0.0067258, -0.0033405, -0.0031366, 0.0031645
7: -0.0019374, 0.0034595, -0.0022376, 0.0036118, -0.0051711, 0.0053155
8: -0.0007838, -0.0000490, -0.0008372, 0.0000027, -0.0007865, 0.0007745
9: 0.9990694, 1.0109725, 0.9977618, 1.0114797, -0.0088898, 0.0097126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0052939
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0054725
time: 0.68 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0036232, -0.0004667, -0.0028390, 0.0030960
1: -0.0040656, 0.0028282, -0.0041113, 0.0036100, -0.0049733, 0.0048775
2: 0.0032125, 0.0092715, 0.0031992, 0.0097691, -0.0041343, 0.0041525
3: -0.0044256, -0.0035238, -0.0044849, -0.0035005, -0.0009251, 0.0009611
4: 0.0020015, 0.0076260, 0.0017745, 0.0077244, -0.0051865, 0.0051690
5: -0.0027927, 0.0027322, -0.0028958, 0.0030354, -0.0048330, 0.0048783
6: -0.0066802, -0.0034138, -0.0067596, -0.0033386, -0.0031914, 0.0032284
7: -0.0020182, 0.0036256, -0.0022425, 0.0036735, -0.0052766, 0.0053817
8: -0.0008139, -0.0000405, -0.0008477, 0.0000030, -0.0007775, 0.0008072
9: 0.9987171, 1.0115892, 0.9977556, 1.0117074, -0.0095872, 0.0095635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0056116
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0058611
time: 0.69 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0037731, -0.0004915, -0.0031465, 0.0031483
1: -0.0036748, 0.0035840, -0.0040176, 0.0039845, -0.0058479, 0.0045070
2: 0.0035906, 0.0096900, 0.0032832, 0.0100039, -0.0046428, 0.0039379
3: -0.0044903, -0.0035427, -0.0045115, -0.0035031, -0.0009873, 0.0009688
4: 0.0019122, 0.0074146, 0.0016754, 0.0076922, -0.0052499, 0.0052083
5: -0.0025355, 0.0028903, -0.0028466, 0.0031709, -0.0049484, 0.0048899
6: -0.0066172, -0.0034031, -0.0067565, -0.0033077, -0.0031990, 0.0032647
7: -0.0021357, 0.0034153, -0.0023418, 0.0036393, -0.0054178, 0.0054282
8: -0.0007947, -0.0000022, -0.0008439, 0.0000215, -0.0008162, 0.0008156
9: 0.9980376, 1.0107963, 0.9973178, 1.0115211, -0.0093628, 0.0102539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0052939
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0053394
time: 0.75 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0037750, -0.0004574, -0.0031774, 0.0032486
1: -0.0039674, 0.0037511, -0.0041183, 0.0039868, -0.0058008, 0.0049273
2: 0.0033183, 0.0098352, 0.0031939, 0.0100070, -0.0045603, 0.0043285
3: -0.0044910, -0.0035230, -0.0045121, -0.0034956, -0.0009955, 0.0009891
4: 0.0017766, 0.0076270, 0.0016720, 0.0077625, -0.0054166, 0.0052741
5: -0.0027863, 0.0030547, -0.0029303, 0.0031735, -0.0049638, 0.0051333
6: -0.0067112, -0.0033517, -0.0067893, -0.0033058, -0.0032563, 0.0033296
7: -0.0022354, 0.0035968, -0.0023468, 0.0036995, -0.0055274, 0.0054971
8: -0.0008242, 0.0000062, -0.0008543, 0.0000217, -0.0008128, 0.0008544
9: 0.9976520, 1.0114602, 0.9973118, 1.0117472, -0.0100926, 0.0101337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0056078
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0056536
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.04 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0052939
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0054725
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0056116
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0056078, upper bound: 0.0058611
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0052939
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0053394
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0056078
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 9, lower bound: -0.0058610, upper bound: 0.0056536

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0033100, -0.0004960, -0.0027726, 0.0026884
1: -0.0037844, 0.0026818, -0.0041166, 0.0028353, -0.0041886, 0.0042054
2: 0.0034838, 0.0091255, 0.0031813, 0.0092810, -0.0036761, 0.0037089
3: -0.0044250, -0.0035443, -0.0044272, -0.0035083, -0.0009166, 0.0008829
4: 0.0021194, 0.0074215, 0.0019920, 0.0076695, -0.0050388, 0.0049070
5: -0.0025576, 0.0025986, -0.0028400, 0.0027409, -0.0045429, 0.0046555
6: -0.0065863, -0.0034534, -0.0067071, -0.0034082, -0.0030705, 0.0031476
7: -0.0019374, 0.0034595, -0.0020305, 0.0036599, -0.0051943, 0.0050826
8: -0.0007838, -0.0000490, -0.0008285, -0.0000390, -0.0007449, 0.0007627
9: 0.9990694, 1.0109725, 0.9986976, 1.0116501, -0.0088730, 0.0087061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0052939
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0052938
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0036864, -0.0004983, -0.0027703, 0.0030648
1: -0.0037844, 0.0026818, -0.0040126, 0.0037597, -0.0053890, 0.0045134
2: 0.0034838, 0.0091255, 0.0032870, 0.0098446, -0.0044474, 0.0038024
3: -0.0044250, -0.0035443, -0.0044928, -0.0035067, -0.0009183, 0.0009485
4: 0.0021194, 0.0074215, 0.0017667, 0.0076670, -0.0050408, 0.0051151
5: -0.0025576, 0.0025986, -0.0028239, 0.0030636, -0.0048509, 0.0046512
6: -0.0065863, -0.0034534, -0.0067374, -0.0033460, -0.0031308, 0.0031764
7: -0.0019374, 0.0034595, -0.0022479, 0.0036217, -0.0051865, 0.0053353
8: -0.0007838, -0.0000490, -0.0008395, 0.0000076, -0.0007914, 0.0007773
9: 0.9990694, 1.0109725, 0.9976307, 1.0114945, -0.0089081, 0.0099905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0054725
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0054725
time: 0.99 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0033113, -0.0004625, -0.0028433, 0.0027841
1: -0.0040656, 0.0028282, -0.0042133, 0.0028370, -0.0040534, 0.0046168
2: 0.0032125, 0.0092715, 0.0030863, 0.0092841, -0.0035628, 0.0040834
3: -0.0044256, -0.0035238, -0.0044276, -0.0035004, -0.0009252, 0.0009038
4: 0.0020015, 0.0076260, 0.0019887, 0.0077431, -0.0052015, 0.0049782
5: -0.0027927, 0.0027322, -0.0029214, 0.0027436, -0.0045661, 0.0048954
6: -0.0066802, -0.0034138, -0.0067424, -0.0034062, -0.0031281, 0.0032099
7: -0.0020182, 0.0036256, -0.0020355, 0.0037170, -0.0052978, 0.0051559
8: -0.0008139, -0.0000405, -0.0008400, -0.0000387, -0.0007337, 0.0007995
9: 0.9987171, 1.0115892, 0.9986913, 1.0118632, -0.0095690, 0.0085986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0056116
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0056116
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0036881, -0.0004641, -0.0028417, 0.0031609
1: -0.0040656, 0.0028282, -0.0041134, 0.0037618, -0.0053800, 0.0048771
2: 0.0032125, 0.0092715, 0.0031976, 0.0098476, -0.0043912, 0.0041577
3: -0.0044256, -0.0035238, -0.0044934, -0.0034992, -0.0009264, 0.0009696
4: 0.0020015, 0.0076260, 0.0017633, 0.0077381, -0.0052029, 0.0051743
5: -0.0027927, 0.0027322, -0.0029084, 0.0030664, -0.0048663, 0.0048932
6: -0.0066802, -0.0034138, -0.0067708, -0.0033440, -0.0031848, 0.0032399
7: -0.0020182, 0.0036256, -0.0022529, 0.0036827, -0.0052913, 0.0053982
8: -0.0008139, -0.0000405, -0.0008499, 0.0000078, -0.0007828, 0.0008095
9: 0.9987171, 1.0115892, 0.9976239, 1.0117215, -0.0096045, 0.0098984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0058611
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0058611
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0033100, -0.0004960, -0.0031420, 0.0026852
1: -0.0036748, 0.0035840, -0.0041166, 0.0028353, -0.0045213, 0.0055385
2: 0.0035906, 0.0096900, 0.0031813, 0.0092810, -0.0037732, 0.0045294
3: -0.0044903, -0.0035427, -0.0044272, -0.0035083, -0.0009820, 0.0008845
4: 0.0019122, 0.0074146, 0.0019920, 0.0076695, -0.0052200, 0.0049045
5: -0.0025355, 0.0028903, -0.0028400, 0.0027409, -0.0045362, 0.0049428
6: -0.0066172, -0.0034031, -0.0067071, -0.0034082, -0.0031002, 0.0031978
7: -0.0021357, 0.0034153, -0.0020305, 0.0036599, -0.0054273, 0.0050733
8: -0.0007947, -0.0000022, -0.0008285, -0.0000390, -0.0007557, 0.0008121
9: 0.9980376, 1.0107963, 0.9986976, 1.0116501, -0.0101438, 0.0087370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0052939
time: 0.71 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0052939
time: 0.72 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0036864, -0.0004983, -0.0031397, 0.0030616
1: -0.0036748, 0.0035840, -0.0040126, 0.0037597, -0.0044767, 0.0045010
2: 0.0035906, 0.0096900, 0.0032870, 0.0098446, -0.0039155, 0.0039251
3: -0.0044903, -0.0035427, -0.0044928, -0.0035067, -0.0009837, 0.0009501
4: 0.0019122, 0.0074146, 0.0017667, 0.0076670, -0.0052198, 0.0050997
5: -0.0025355, 0.0028903, -0.0028239, 0.0030636, -0.0047622, 0.0048613
6: -0.0066172, -0.0034031, -0.0067374, -0.0033460, -0.0031758, 0.0032449
7: -0.0021357, 0.0034153, -0.0022479, 0.0036217, -0.0053892, 0.0052916
8: -0.0007947, -0.0000022, -0.0008395, 0.0000076, -0.0007920, 0.0008080
9: 0.9980376, 1.0107963, 0.9976307, 1.0114945, -0.0093227, 0.0092045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0053394
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0053394
time: 0.71 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0033113, -0.0004625, -0.0032189, 0.0027850
1: -0.0039674, 0.0037511, -0.0042133, 0.0028370, -0.0043886, 0.0058363
2: 0.0033183, 0.0098352, 0.0030863, 0.0092841, -0.0036636, 0.0048530
3: -0.0044910, -0.0035230, -0.0044276, -0.0035004, -0.0009907, 0.0009046
4: 0.0017766, 0.0076270, 0.0019887, 0.0077431, -0.0054092, 0.0049821
5: -0.0027863, 0.0030547, -0.0029214, 0.0027436, -0.0045618, 0.0052027
6: -0.0067112, -0.0033517, -0.0067424, -0.0034062, -0.0031606, 0.0032701
7: -0.0022354, 0.0035968, -0.0020355, 0.0037170, -0.0055493, 0.0051520
8: -0.0008242, 0.0000062, -0.0008400, -0.0000387, -0.0007474, 0.0008461
9: 0.9976520, 1.0114602, 0.9986913, 1.0118632, -0.0108512, 0.0086316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056078
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056078
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0036881, -0.0004641, -0.0031687, 0.0031617
1: -0.0039674, 0.0037511, -0.0041134, 0.0037618, -0.0043598, 0.0049211
2: 0.0033183, 0.0098352, 0.0031976, 0.0098476, -0.0037988, 0.0043164
3: -0.0044910, -0.0035230, -0.0044934, -0.0034992, -0.0009918, 0.0009704
4: 0.0017766, 0.0076270, 0.0017633, 0.0077381, -0.0053872, 0.0051696
5: -0.0027863, 0.0030547, -0.0029084, 0.0030664, -0.0047816, 0.0051055
6: -0.0067112, -0.0033517, -0.0067708, -0.0033440, -0.0032314, 0.0033101
7: -0.0022354, 0.0035968, -0.0022529, 0.0036827, -0.0054990, 0.0053632
8: -0.0008242, 0.0000062, -0.0008499, 0.0000078, -0.0007794, 0.0008469
9: 0.9976520, 1.0114602, 0.9976239, 1.0117215, -0.0100542, 0.0090845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056536
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056536
time: 0.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.74 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0052939
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0052938
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0054725
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0053616, upper bound: 0.0054725
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0056116
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0056116
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0058611
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0052941, upper bound: 0.0058611
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0052939
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0052939
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0053394
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0055806, upper bound: 0.0053394
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056078
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056078
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056536
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 9, lower bound: -0.0054725, upper bound: 0.0056536

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0032685, -0.0006216, -0.0026469, 0.0026469
1: -0.0037844, 0.0026818, -0.0037844, 0.0026818, -0.0038631, 0.0038631
2: 0.0034838, 0.0091255, 0.0034838, 0.0091255, -0.0034106, 0.0034106
3: -0.0044250, -0.0035443, -0.0044250, -0.0035443, -0.0008807, 0.0008807
4: 0.0021194, 0.0074215, 0.0021194, 0.0074215, -0.0048036, 0.0048036
5: -0.0025576, 0.0025986, -0.0025576, 0.0025986, -0.0043787, 0.0043787
6: -0.0065863, -0.0034534, -0.0065863, -0.0034534, -0.0030319, 0.0030319
7: -0.0019374, 0.0034595, -0.0019374, 0.0034595, -0.0050122, 0.0050122
8: -0.0007838, -0.0000490, -0.0007838, -0.0000490, -0.0007156, 0.0007156
9: 0.9990694, 1.0109725, 0.9990694, 1.0109725, -0.0082114, 0.0082114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0050645
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0051936
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0033058, -0.0005272, -0.0027414, 0.0026841
1: -0.0037844, 0.0026818, -0.0040656, 0.0028282, -0.0041212, 0.0042758
2: 0.0034838, 0.0091255, 0.0032125, 0.0092715, -0.0036670, 0.0037469
3: -0.0044250, -0.0035443, -0.0044256, -0.0035238, -0.0009012, 0.0008813
4: 0.0021194, 0.0074215, 0.0020015, 0.0076260, -0.0049843, 0.0048982
5: -0.0025576, 0.0025986, -0.0027927, 0.0027322, -0.0045342, 0.0046109
6: -0.0065863, -0.0034534, -0.0066802, -0.0034138, -0.0030650, 0.0031130
7: -0.0019374, 0.0034595, -0.0020182, 0.0036256, -0.0051387, 0.0050705
8: -0.0007838, -0.0000490, -0.0008139, -0.0000405, -0.0007372, 0.0007618
9: 0.9990694, 1.0109725, 0.9987171, 1.0115892, -0.0088650, 0.0086872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0050645
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0051936
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0036380, -0.0006249, -0.0026437, 0.0030164
1: -0.0037844, 0.0026818, -0.0036748, 0.0035840, -0.0051962, 0.0041958
2: 0.0034838, 0.0091255, 0.0035906, 0.0096900, -0.0042310, 0.0035077
3: -0.0044250, -0.0035443, -0.0044903, -0.0035427, -0.0008822, 0.0009461
4: 0.0021194, 0.0074215, 0.0019122, 0.0074146, -0.0048011, 0.0049848
5: -0.0025576, 0.0025986, -0.0025355, 0.0028903, -0.0046659, 0.0043720
6: -0.0065863, -0.0034534, -0.0066172, -0.0034031, -0.0030821, 0.0030616
7: -0.0019374, 0.0034595, -0.0021357, 0.0034153, -0.0050030, 0.0052452
8: -0.0007838, -0.0000490, -0.0007947, -0.0000022, -0.0007650, 0.0007297
9: 0.9990694, 1.0109725, 0.9980376, 1.0107963, -0.0082424, 0.0094822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0052307
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0053706
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0032685, -0.0006216, -0.0036814, -0.0005264, -0.0027422, 0.0030598
1: -0.0037844, 0.0026818, -0.0039674, 0.0037511, -0.0053406, 0.0044994
2: 0.0034838, 0.0091255, 0.0033183, 0.0098352, -0.0044366, 0.0038028
3: -0.0044250, -0.0035443, -0.0044910, -0.0035230, -0.0009020, 0.0009468
4: 0.0021194, 0.0074215, 0.0017766, 0.0076270, -0.0049918, 0.0051059
5: -0.0025576, 0.0025986, -0.0027863, 0.0030547, -0.0048415, 0.0046204
6: -0.0065863, -0.0034534, -0.0067112, -0.0033517, -0.0031252, 0.0031456
7: -0.0019374, 0.0034595, -0.0022354, 0.0035968, -0.0051357, 0.0053220
8: -0.0007838, -0.0000490, -0.0008242, 0.0000062, -0.0007900, 0.0007751
9: 0.9990694, 1.0109725, 0.9976520, 1.0114602, -0.0089075, 0.0099693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0052307
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0053706
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0032685, -0.0006216, -0.0026841, 0.0027414
1: -0.0040656, 0.0028282, -0.0037844, 0.0026818, -0.0042758, 0.0041212
2: 0.0032125, 0.0092715, 0.0034838, 0.0091255, -0.0037469, 0.0036670
3: -0.0044256, -0.0035238, -0.0044250, -0.0035443, -0.0008813, 0.0009012
4: 0.0020015, 0.0076260, 0.0021194, 0.0074215, -0.0048982, 0.0049843
5: -0.0027927, 0.0027322, -0.0025576, 0.0025986, -0.0046109, 0.0045342
6: -0.0066802, -0.0034138, -0.0065863, -0.0034534, -0.0031130, 0.0030650
7: -0.0020182, 0.0036256, -0.0019374, 0.0034595, -0.0050705, 0.0051387
8: -0.0008139, -0.0000405, -0.0007838, -0.0000490, -0.0007618, 0.0007372
9: 0.9987171, 1.0115892, 0.9990694, 1.0109725, -0.0086872, 0.0088650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0055113
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0055113
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0033058, -0.0005272, -0.0027786, 0.0027786
1: -0.0040656, 0.0028282, -0.0040656, 0.0028282, -0.0040164, 0.0040164
2: 0.0032125, 0.0092715, 0.0032125, 0.0092715, -0.0035514, 0.0035514
3: -0.0044256, -0.0035238, -0.0044256, -0.0035238, -0.0009018, 0.0009018
4: 0.0020015, 0.0076260, 0.0020015, 0.0076260, -0.0049667, 0.0049667
5: -0.0027927, 0.0027322, -0.0027927, 0.0027322, -0.0045552, 0.0045552
6: -0.0066802, -0.0034138, -0.0066802, -0.0034138, -0.0031207, 0.0031207
7: -0.0020182, 0.0036256, -0.0020182, 0.0036256, -0.0051392, 0.0051392
8: -0.0008139, -0.0000405, -0.0008139, -0.0000405, -0.0007280, 0.0007280
9: 0.9987171, 1.0115892, 0.9987171, 1.0115892, -0.0085770, 0.0085770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0054076
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0055113
time: 0.98 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0036380, -0.0006249, -0.0026809, 0.0031108
1: -0.0040656, 0.0028282, -0.0036748, 0.0035840, -0.0056089, 0.0044539
2: 0.0032125, 0.0092715, 0.0035906, 0.0096900, -0.0045673, 0.0037641
3: -0.0044256, -0.0035238, -0.0044903, -0.0035427, -0.0008829, 0.0009666
4: 0.0020015, 0.0076260, 0.0019122, 0.0074146, -0.0048957, 0.0051654
5: -0.0027927, 0.0027322, -0.0025355, 0.0028903, -0.0048982, 0.0045275
6: -0.0066802, -0.0034138, -0.0066172, -0.0034031, -0.0031632, 0.0030947
7: -0.0020182, 0.0036256, -0.0021357, 0.0034153, -0.0050612, 0.0053717
8: -0.0008139, -0.0000405, -0.0007947, -0.0000022, -0.0008113, 0.0007513
9: 0.9987171, 1.0115892, 0.9980376, 1.0107963, -0.0087182, 0.0101358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0056580
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0057598
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0033058, -0.0005272, -0.0036814, -0.0005264, -0.0027794, 0.0031542
1: -0.0040656, 0.0028282, -0.0039674, 0.0037511, -0.0053479, 0.0043516
2: 0.0032125, 0.0092715, 0.0033183, 0.0098352, -0.0043759, 0.0036522
3: -0.0044256, -0.0035238, -0.0044910, -0.0035230, -0.0009026, 0.0009672
4: 0.0020015, 0.0076260, 0.0017766, 0.0076270, -0.0049706, 0.0051620
5: -0.0027927, 0.0027322, -0.0027863, 0.0030547, -0.0048542, 0.0045508
6: -0.0066802, -0.0034138, -0.0067112, -0.0033517, -0.0031771, 0.0031532
7: -0.0020182, 0.0036256, -0.0022354, 0.0035968, -0.0051353, 0.0053797
8: -0.0008139, -0.0000405, -0.0008242, 0.0000062, -0.0007774, 0.0007417
9: 0.9987171, 1.0115892, 0.9976520, 1.0114602, -0.0086100, 0.0098698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0056580
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0057598
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0032685, -0.0006216, -0.0030164, 0.0026437
1: -0.0036748, 0.0035840, -0.0037844, 0.0026818, -0.0041958, 0.0051961
2: 0.0035906, 0.0096900, 0.0034838, 0.0091255, -0.0035077, 0.0042310
3: -0.0044903, -0.0035427, -0.0044250, -0.0035443, -0.0009461, 0.0008822
4: 0.0019122, 0.0074146, 0.0021194, 0.0074215, -0.0049848, 0.0048011
5: -0.0025355, 0.0028903, -0.0025576, 0.0025986, -0.0043720, 0.0046659
6: -0.0066172, -0.0034031, -0.0065863, -0.0034534, -0.0030616, 0.0030821
7: -0.0021357, 0.0034153, -0.0019374, 0.0034595, -0.0052452, 0.0050030
8: -0.0007947, -0.0000022, -0.0007838, -0.0000490, -0.0007297, 0.0007650
9: 0.9980376, 1.0107963, 0.9990694, 1.0109725, -0.0094822, 0.0082424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0051936
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054759, upper bound: 0.0051936
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0033058, -0.0005272, -0.0031108, 0.0026809
1: -0.0036748, 0.0035840, -0.0040656, 0.0028282, -0.0044539, 0.0056089
2: 0.0035906, 0.0096900, 0.0032125, 0.0092715, -0.0037641, 0.0045673
3: -0.0044903, -0.0035427, -0.0044256, -0.0035238, -0.0009666, 0.0008829
4: 0.0019122, 0.0074146, 0.0020015, 0.0076260, -0.0051654, 0.0048957
5: -0.0025355, 0.0028903, -0.0027927, 0.0027322, -0.0045275, 0.0048982
6: -0.0066172, -0.0034031, -0.0066802, -0.0034138, -0.0030947, 0.0031632
7: -0.0021357, 0.0034153, -0.0020182, 0.0036256, -0.0053717, 0.0050612
8: -0.0007947, -0.0000022, -0.0008139, -0.0000405, -0.0007513, 0.0008113
9: 0.9980376, 1.0107963, 0.9987171, 1.0115892, -0.0101358, 0.0087182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0051936
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054759, upper bound: 0.0051936
time: 0.72 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0036380, -0.0006249, -0.0030131, 0.0030131
1: -0.0036748, 0.0035840, -0.0036748, 0.0035840, -0.0041575, 0.0041575
2: 0.0035906, 0.0096900, 0.0035906, 0.0096900, -0.0036288, 0.0036288
3: -0.0044903, -0.0035427, -0.0044903, -0.0035427, -0.0009476, 0.0009476
4: 0.0019122, 0.0074146, 0.0019122, 0.0074146, -0.0049891, 0.0049891
5: -0.0025355, 0.0028903, -0.0025355, 0.0028903, -0.0045897, 0.0045897
6: -0.0066172, -0.0034031, -0.0066172, -0.0034031, -0.0031335, 0.0031335
7: -0.0021357, 0.0034153, -0.0021357, 0.0034153, -0.0052138, 0.0052138
8: -0.0007947, -0.0000022, -0.0007947, -0.0000022, -0.0007620, 0.0007620
9: 0.9980376, 1.0107963, 0.9980376, 1.0107963, -0.0086702, 0.0086702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0051039
time: 0.71 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0052333
time: 0.72 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0036380, -0.0006249, -0.0036814, -0.0005264, -0.0030821, 0.0030305
1: -0.0036748, 0.0035840, -0.0039674, 0.0037511, -0.0044232, 0.0045735
2: 0.0035906, 0.0096900, 0.0033183, 0.0098352, -0.0039055, 0.0039599
3: -0.0044903, -0.0035427, -0.0044910, -0.0035230, -0.0009674, 0.0009483
4: 0.0019122, 0.0074146, 0.0017766, 0.0076270, -0.0051661, 0.0050907
5: -0.0025355, 0.0028903, -0.0027863, 0.0030547, -0.0047534, 0.0048137
6: -0.0066172, -0.0034031, -0.0067112, -0.0033517, -0.0031701, 0.0032123
7: -0.0021357, 0.0034153, -0.0022354, 0.0035968, -0.0053372, 0.0052780
8: -0.0007947, -0.0000022, -0.0008242, 0.0000062, -0.0007855, 0.0008048
9: 0.9980376, 1.0107963, 0.9976520, 1.0114602, -0.0093105, 0.0091850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0051039
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0052332
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0032685, -0.0006216, -0.0030598, 0.0027422
1: -0.0039674, 0.0037511, -0.0037844, 0.0026818, -0.0044994, 0.0053406
2: 0.0033183, 0.0098352, 0.0034838, 0.0091255, -0.0038028, 0.0044366
3: -0.0044910, -0.0035230, -0.0044250, -0.0035443, -0.0009468, 0.0009020
4: 0.0017766, 0.0076270, 0.0021194, 0.0074215, -0.0051059, 0.0049918
5: -0.0027863, 0.0030547, -0.0025576, 0.0025986, -0.0046204, 0.0048415
6: -0.0067112, -0.0033517, -0.0065863, -0.0034534, -0.0031456, 0.0031252
7: -0.0022354, 0.0035968, -0.0019374, 0.0034595, -0.0053220, 0.0051357
8: -0.0008242, 0.0000062, -0.0007838, -0.0000490, -0.0007751, 0.0007900
9: 0.9976520, 1.0114602, 0.9990694, 1.0109725, -0.0099693, 0.0089075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0055074
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053706, upper bound: 0.0055074
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0033058, -0.0005272, -0.0031542, 0.0027794
1: -0.0039674, 0.0037511, -0.0040656, 0.0028282, -0.0043516, 0.0053479
2: 0.0033183, 0.0098352, 0.0032125, 0.0092715, -0.0036522, 0.0043759
3: -0.0044910, -0.0035230, -0.0044256, -0.0035238, -0.0009672, 0.0009026
4: 0.0017766, 0.0076270, 0.0020015, 0.0076260, -0.0051620, 0.0049706
5: -0.0027863, 0.0030547, -0.0027927, 0.0027322, -0.0045508, 0.0048542
6: -0.0067112, -0.0033517, -0.0066802, -0.0034138, -0.0031532, 0.0031771
7: -0.0022354, 0.0035968, -0.0020182, 0.0036256, -0.0053797, 0.0051353
8: -0.0008242, 0.0000062, -0.0008139, -0.0000405, -0.0007417, 0.0007774
9: 0.9976520, 1.0114602, 0.9987171, 1.0115892, -0.0098698, 0.0086100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0055074
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053706, upper bound: 0.0055074
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0036380, -0.0006249, -0.0030305, 0.0030821
1: -0.0039674, 0.0037511, -0.0036748, 0.0035840, -0.0045735, 0.0044232
2: 0.0033183, 0.0098352, 0.0035906, 0.0096900, -0.0039599, 0.0039055
3: -0.0044910, -0.0035230, -0.0044903, -0.0035427, -0.0009483, 0.0009674
4: 0.0017766, 0.0076270, 0.0019122, 0.0074146, -0.0050907, 0.0051661
5: -0.0027863, 0.0030547, -0.0025355, 0.0028903, -0.0048137, 0.0047534
6: -0.0067112, -0.0033517, -0.0066172, -0.0034031, -0.0032123, 0.0031701
7: -0.0022354, 0.0035968, -0.0021357, 0.0034153, -0.0052780, 0.0053372
8: -0.0008242, 0.0000062, -0.0007947, -0.0000022, -0.0008048, 0.0007855
9: 0.9976520, 1.0114602, 0.9980376, 1.0107963, -0.0091850, 0.0093105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052561, upper bound: 0.0055488
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053928, upper bound: 0.0055488
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0036814, -0.0005264, -0.0036814, -0.0005264, -0.0031402, 0.0031402
1: -0.0039674, 0.0037511, -0.0039674, 0.0037511, -0.0043266, 0.0043266
2: 0.0033183, 0.0098352, 0.0033183, 0.0098352, -0.0037861, 0.0037861
3: -0.0044910, -0.0035230, -0.0044910, -0.0035230, -0.0009681, 0.0009681
4: 0.0017766, 0.0076270, 0.0017766, 0.0076270, -0.0051575, 0.0051575
5: -0.0027863, 0.0030547, -0.0027863, 0.0030547, -0.0047704, 0.0047704
6: -0.0067112, -0.0033517, -0.0067112, -0.0033517, -0.0032236, 0.0032236
7: -0.0022354, 0.0035968, -0.0022354, 0.0035968, -0.0053444, 0.0053444
8: -0.0008242, 0.0000062, -0.0008242, 0.0000062, -0.0007741, 0.0007741
9: 0.9976520, 1.0114602, 0.9976520, 1.0114602, -0.0090611, 0.0090611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053928, upper bound: 0.0054449
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053928, upper bound: 0.0055489
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.78 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0050645
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0051936
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0050645
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0051936
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0052307
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0053706
IS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0052307
IS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052630, upper bound: 0.0053706
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0055113
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0055113
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0054076
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0055113
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0056580
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0057598
IS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0056580
IS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0051936, upper bound: 0.0057598
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0051936
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0054759, upper bound: 0.0051936
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0051936
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0054759, upper bound: 0.0051936
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0051039
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0052333
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0051039
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0054996, upper bound: 0.0052332
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0055074
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053706, upper bound: 0.0055074
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0055074
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053706, upper bound: 0.0055074
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0052561, upper bound: 0.0055488
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053928, upper bound: 0.0055488
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053928, upper bound: 0.0054449
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 9, lower bound: -0.0053928, upper bound: 0.0055489

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0032630, -0.0006236, -0.0025943, 0.0026225
1: -0.0037179, 0.0026543, -0.0037775, 0.0026791, -0.0038168, 0.0038109
2: 0.0035981, 0.0091067, 0.0034956, 0.0091236, -0.0032826, 0.0033741
3: -0.0043849, -0.0035512, -0.0044207, -0.0035450, -0.0008399, 0.0008695
4: 0.0021321, 0.0072644, 0.0021207, 0.0074047, -0.0047794, 0.0046559
5: -0.0024348, 0.0025822, -0.0025446, 0.0025969, -0.0042666, 0.0043565
6: -0.0064867, -0.0034582, -0.0065754, -0.0034539, -0.0029321, 0.0030161
7: -0.0019213, 0.0032731, -0.0019357, 0.0034402, -0.0049652, 0.0048086
8: -0.0007760, -0.0000730, -0.0007830, -0.0000515, -0.0007059, 0.0006814
9: 0.9991087, 1.0107142, 0.9990734, 1.0109454, -0.0081513, 0.0079629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0051354
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0051354
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032253, -0.0006028, -0.0032411, -0.0006295, -0.0025958, 0.0026383
1: -0.0037195, 0.0027624, -0.0037565, 0.0026717, -0.0038312, 0.0038615
2: 0.0036027, 0.0092742, 0.0035274, 0.0091180, -0.0033147, 0.0035465
3: -0.0043885, -0.0034970, -0.0044091, -0.0035473, -0.0008411, 0.0009121
4: 0.0019686, 0.0073196, 0.0021249, 0.0073600, -0.0048886, 0.0047059
5: -0.0024869, 0.0026904, -0.0025096, 0.0025919, -0.0043097, 0.0044224
6: -0.0065211, -0.0033544, -0.0065464, -0.0034556, -0.0029675, 0.0030906
7: -0.0021640, 0.0032973, -0.0019303, 0.0033854, -0.0051937, 0.0048692
8: -0.0008142, -0.0000816, -0.0007808, -0.0000604, -0.0007538, 0.0006863
9: 0.9988109, 1.0107478, 0.9990852, 1.0108705, -0.0083746, 0.0080272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0051898
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050931, upper bound: 0.0050931
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0033002, -0.0005293, -0.0026885, 0.0026598
1: -0.0037179, 0.0026543, -0.0040586, 0.0028255, -0.0040749, 0.0042230
2: 0.0035981, 0.0091067, 0.0032243, 0.0092698, -0.0035388, 0.0037114
3: -0.0043849, -0.0035512, -0.0044213, -0.0035246, -0.0008603, 0.0008701
4: 0.0021321, 0.0072644, 0.0020029, 0.0076092, -0.0049601, 0.0047505
5: -0.0024348, 0.0025822, -0.0027796, 0.0027307, -0.0044220, 0.0045889
6: -0.0064867, -0.0034582, -0.0066691, -0.0034144, -0.0029652, 0.0030970
7: -0.0019213, 0.0032731, -0.0020167, 0.0036060, -0.0050919, 0.0048670
8: -0.0007760, -0.0000730, -0.0008130, -0.0000430, -0.0007277, 0.0007275
9: 0.9991087, 1.0107142, 0.9987208, 1.0115627, -0.0088059, 0.0084381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054076, upper bound: 0.0050645
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054076, upper bound: 0.0050645
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032253, -0.0006028, -0.0032787, -0.0005357, -0.0026896, 0.0026759
1: -0.0037195, 0.0027624, -0.0040374, 0.0028185, -0.0040895, 0.0042705
2: 0.0036027, 0.0092742, 0.0032580, 0.0092647, -0.0035702, 0.0038728
3: -0.0043885, -0.0034970, -0.0044097, -0.0035271, -0.0008614, 0.0009127
4: 0.0019686, 0.0073196, 0.0020070, 0.0075645, -0.0050674, 0.0048003
5: -0.0024869, 0.0026904, -0.0027441, 0.0027261, -0.0044647, 0.0046521
6: -0.0065211, -0.0033544, -0.0066400, -0.0034162, -0.0030006, 0.0031707
7: -0.0021640, 0.0032973, -0.0020119, 0.0035519, -0.0053181, 0.0049277
8: -0.0008142, -0.0000816, -0.0008107, -0.0000512, -0.0007630, 0.0007291
9: 0.9988109, 1.0107478, 0.9987314, 1.0114824, -0.0090204, 0.0085008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052613, upper bound: 0.0051271
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053494, upper bound: 0.0050264
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0036330, -0.0006267, -0.0025911, 0.0029925
1: -0.0037179, 0.0026543, -0.0036680, 0.0035809, -0.0051509, 0.0041422
2: 0.0035981, 0.0091067, 0.0036021, 0.0096881, -0.0041022, 0.0034720
3: -0.0043849, -0.0035512, -0.0044861, -0.0035435, -0.0008414, 0.0009348
4: 0.0021321, 0.0072644, 0.0019136, 0.0073981, -0.0047772, 0.0048368
5: -0.0024348, 0.0025822, -0.0025225, 0.0028886, -0.0045535, 0.0043502
6: -0.0064867, -0.0034582, -0.0066069, -0.0034037, -0.0029822, 0.0030462
7: -0.0019213, 0.0032731, -0.0021337, 0.0033958, -0.0049563, 0.0050414
8: -0.0007760, -0.0000730, -0.0007939, -0.0000046, -0.0007554, 0.0006954
9: 0.9991087, 1.0107142, 0.9980419, 1.0107703, -0.0081827, 0.0092323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0053627
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0053627
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032253, -0.0006028, -0.0036099, -0.0006324, -0.0025928, 0.0030072
1: -0.0037195, 0.0027624, -0.0036477, 0.0035723, -0.0051666, 0.0041970
2: 0.0036027, 0.0092742, 0.0036370, 0.0096826, -0.0041316, 0.0036378
3: -0.0043885, -0.0034970, -0.0044755, -0.0035460, -0.0008425, 0.0009785
4: 0.0019686, 0.0073196, 0.0019183, 0.0073566, -0.0048862, 0.0048862
5: -0.0024869, 0.0026904, -0.0024899, 0.0028834, -0.0045954, 0.0044161
6: -0.0065211, -0.0033544, -0.0065782, -0.0034055, -0.0030174, 0.0031200
7: -0.0021640, 0.0032973, -0.0021274, 0.0033448, -0.0051842, 0.0051008
8: -0.0008142, -0.0000816, -0.0007915, -0.0000142, -0.0008001, 0.0007001
9: 0.9988109, 1.0107478, 0.9980550, 1.0106908, -0.0084032, 0.0092923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0053996
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050931, upper bound: 0.0053318
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0036764, -0.0005285, -0.0026894, 0.0030359
1: -0.0037179, 0.0026543, -0.0039607, 0.0037482, -0.0052952, 0.0044458
2: 0.0035981, 0.0091067, 0.0033294, 0.0098335, -0.0043080, 0.0037676
3: -0.0043849, -0.0035512, -0.0044869, -0.0035238, -0.0008611, 0.0009357
4: 0.0021321, 0.0072644, 0.0017780, 0.0076106, -0.0049681, 0.0049579
5: -0.0024348, 0.0025822, -0.0027735, 0.0030531, -0.0047290, 0.0045985
6: -0.0064867, -0.0034582, -0.0067005, -0.0033523, -0.0030253, 0.0031301
7: -0.0019213, 0.0032731, -0.0022337, 0.0035772, -0.0050888, 0.0051182
8: -0.0007760, -0.0000730, -0.0008234, 0.0000037, -0.0007797, 0.0007407
9: 0.9991087, 1.0107142, 0.9976562, 1.0114341, -0.0088492, 0.0097196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054063, upper bound: 0.0052307
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054063, upper bound: 0.0052307
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032253, -0.0006028, -0.0036541, -0.0005348, -0.0026904, 0.0030513
1: -0.0037195, 0.0027624, -0.0039385, 0.0037400, -0.0053108, 0.0044967
2: 0.0036027, 0.0092742, 0.0033670, 0.0098283, -0.0043378, 0.0039285
3: -0.0043885, -0.0034970, -0.0044758, -0.0035265, -0.0008620, 0.0009788
4: 0.0019686, 0.0073196, 0.0017826, 0.0075665, -0.0050754, 0.0050072
5: -0.0024869, 0.0026904, -0.0027381, 0.0030481, -0.0047707, 0.0046619
6: -0.0065211, -0.0033544, -0.0066715, -0.0033543, -0.0030605, 0.0032037
7: -0.0021640, 0.0032973, -0.0022284, 0.0035247, -0.0053165, 0.0051778
8: -0.0008142, -0.0000816, -0.0008209, -0.0000055, -0.0008088, 0.0007393
9: 0.9988109, 1.0107478, 0.9976686, 1.0113558, -0.0090603, 0.0097797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052559, upper bound: 0.0053006
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053467, upper bound: 0.0052217
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0033002, -0.0005293, -0.0032179, -0.0006405, -0.0026598, 0.0026885
1: -0.0040586, 0.0028255, -0.0037179, 0.0026543, -0.0042230, 0.0040749
2: 0.0032243, 0.0092698, 0.0035981, 0.0091067, -0.0037114, 0.0035388
3: -0.0044213, -0.0035246, -0.0043849, -0.0035512, -0.0008701, 0.0008603
4: 0.0020029, 0.0076092, 0.0021321, 0.0072644, -0.0047505, 0.0049601
5: -0.0027796, 0.0027307, -0.0024348, 0.0025822, -0.0045889, 0.0044220
6: -0.0066691, -0.0034144, -0.0064867, -0.0034582, -0.0030970, 0.0029652
7: -0.0020167, 0.0036060, -0.0019213, 0.0032731, -0.0048670, 0.0050919
8: -0.0008130, -0.0000430, -0.0007760, -0.0000730, -0.0007275, 0.0007277
9: 0.9987208, 1.0115627, 0.9991087, 1.0107142, -0.0084381, 0.0088059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0054076
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0055113
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032787, -0.0005357, -0.0032253, -0.0006028, -0.0026759, 0.0026896
1: -0.0040374, 0.0028185, -0.0037195, 0.0027624, -0.0042705, 0.0040895
2: 0.0032580, 0.0092647, 0.0036027, 0.0092742, -0.0038728, 0.0035702
3: -0.0044097, -0.0035271, -0.0043885, -0.0034970, -0.0009127, 0.0008614
4: 0.0020070, 0.0075645, 0.0019686, 0.0073196, -0.0048003, 0.0050674
5: -0.0027441, 0.0027261, -0.0024869, 0.0026904, -0.0046521, 0.0044647
6: -0.0066400, -0.0034162, -0.0065211, -0.0033544, -0.0031707, 0.0030006
7: -0.0020119, 0.0035519, -0.0021640, 0.0032973, -0.0049277, 0.0053181
8: -0.0008107, -0.0000512, -0.0008142, -0.0000816, -0.0007291, 0.0007630
9: 0.9987314, 1.0114824, 0.9988109, 1.0107478, -0.0085008, 0.0090204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051271, upper bound: 0.0052613
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050264, upper bound: 0.0053494
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0033002, -0.0005293, -0.0027260, 0.0027527
1: -0.0039981, 0.0028021, -0.0040586, 0.0028255, -0.0039689, 0.0039640
2: 0.0033250, 0.0092548, 0.0032243, 0.0092698, -0.0034260, 0.0035155
3: -0.0043855, -0.0035314, -0.0044213, -0.0035246, -0.0008609, 0.0008900
4: 0.0020143, 0.0074683, 0.0020029, 0.0076092, -0.0049424, 0.0048176
5: -0.0026697, 0.0027175, -0.0027796, 0.0027307, -0.0044418, 0.0045326
6: -0.0065764, -0.0034190, -0.0066691, -0.0034144, -0.0030197, 0.0031047
7: -0.0020035, 0.0034380, -0.0020167, 0.0036060, -0.0050932, 0.0049370
8: -0.0008058, -0.0000645, -0.0008130, -0.0000430, -0.0007183, 0.0006940
9: 0.9987524, 1.0113355, 0.9987208, 1.0115627, -0.0085186, 0.0083274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0054076
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0054076
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032631, -0.0005095, -0.0032787, -0.0005357, -0.0027274, 0.0027692
1: -0.0039974, 0.0029119, -0.0040374, 0.0028185, -0.0039805, 0.0040094
2: 0.0033405, 0.0094077, 0.0032580, 0.0092647, -0.0034461, 0.0036746
3: -0.0043880, -0.0034787, -0.0044097, -0.0035271, -0.0008610, 0.0009309
4: 0.0018585, 0.0075049, 0.0020070, 0.0075645, -0.0050481, 0.0048528
5: -0.0027046, 0.0028209, -0.0027441, 0.0027261, -0.0044714, 0.0045999
6: -0.0066031, -0.0033140, -0.0066400, -0.0034162, -0.0030452, 0.0031760
7: -0.0022369, 0.0034441, -0.0020119, 0.0035519, -0.0053011, 0.0049807
8: -0.0008450, -0.0000708, -0.0008107, -0.0000512, -0.0007701, 0.0006982
9: 0.9984772, 1.0113324, 0.9987314, 1.0114824, -0.0087370, 0.0083649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0054343
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050316, upper bound: 0.0053494
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0036330, -0.0006267, -0.0026286, 0.0030854
1: -0.0039981, 0.0028021, -0.0036680, 0.0035809, -0.0055577, 0.0044001
2: 0.0033250, 0.0092548, 0.0036021, 0.0096881, -0.0044462, 0.0037261
3: -0.0043855, -0.0035314, -0.0044861, -0.0035435, -0.0008420, 0.0009547
4: 0.0020143, 0.0074683, 0.0019136, 0.0073981, -0.0048713, 0.0050186
5: -0.0026697, 0.0027175, -0.0025225, 0.0028886, -0.0047874, 0.0045042
6: -0.0065764, -0.0034190, -0.0066069, -0.0034037, -0.0030621, 0.0030793
7: -0.0020035, 0.0034380, -0.0021337, 0.0033958, -0.0050155, 0.0051693
8: -0.0008058, -0.0000645, -0.0007939, -0.0000046, -0.0008005, 0.0007182
9: 0.9987524, 1.0113355, 0.9980419, 1.0107703, -0.0086529, 0.0098894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0056580
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0056580
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032631, -0.0005095, -0.0036099, -0.0006324, -0.0026306, 0.0031005
1: -0.0039974, 0.0029119, -0.0036477, 0.0035723, -0.0055719, 0.0044404
2: 0.0033405, 0.0094077, 0.0036370, 0.0096826, -0.0044469, 0.0038602
3: -0.0043880, -0.0034787, -0.0044755, -0.0035460, -0.0008421, 0.0009968
4: 0.0018585, 0.0075049, 0.0019183, 0.0073566, -0.0049716, 0.0050474
5: -0.0027046, 0.0028209, -0.0024899, 0.0028834, -0.0048081, 0.0045596
6: -0.0066031, -0.0033140, -0.0065782, -0.0034055, -0.0030861, 0.0031501
7: -0.0022369, 0.0034441, -0.0021274, 0.0033448, -0.0052218, 0.0052106
8: -0.0008450, -0.0000708, -0.0007915, -0.0000142, -0.0008308, 0.0007207
9: 0.9984772, 1.0113324, 0.9980550, 1.0106908, -0.0088449, 0.0099101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051271, upper bound: 0.0055470
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050264, upper bound: 0.0056222
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0036764, -0.0005285, -0.0027269, 0.0031288
1: -0.0039981, 0.0028021, -0.0039607, 0.0037482, -0.0053015, 0.0042980
2: 0.0033250, 0.0092548, 0.0033294, 0.0098335, -0.0042497, 0.0036168
3: -0.0043855, -0.0035314, -0.0044869, -0.0035238, -0.0008617, 0.0009555
4: 0.0020143, 0.0074683, 0.0017780, 0.0076106, -0.0049470, 0.0050126
5: -0.0026697, 0.0027175, -0.0027735, 0.0030531, -0.0047405, 0.0045288
6: -0.0065764, -0.0034190, -0.0067005, -0.0033523, -0.0030761, 0.0031377
7: -0.0020035, 0.0034380, -0.0022337, 0.0035772, -0.0050896, 0.0051772
8: -0.0008058, -0.0000645, -0.0008234, 0.0000037, -0.0007678, 0.0007076
9: 0.9987524, 1.0113355, 0.9976562, 1.0114341, -0.0085526, 0.0096188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0056580
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0056580
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032631, -0.0005095, -0.0036541, -0.0005348, -0.0027282, 0.0031446
1: -0.0039974, 0.0029119, -0.0039385, 0.0037400, -0.0053143, 0.0043469
2: 0.0033405, 0.0094077, 0.0033670, 0.0098283, -0.0042673, 0.0037721
3: -0.0043880, -0.0034787, -0.0044758, -0.0035265, -0.0008615, 0.0009971
4: 0.0018585, 0.0075049, 0.0017826, 0.0075665, -0.0050518, 0.0050471
5: -0.0027046, 0.0028209, -0.0027381, 0.0030481, -0.0047691, 0.0045966
6: -0.0066031, -0.0033140, -0.0066715, -0.0033543, -0.0031014, 0.0032090
7: -0.0022369, 0.0034441, -0.0022284, 0.0035247, -0.0053000, 0.0052198
8: -0.0008450, -0.0000708, -0.0008209, -0.0000055, -0.0008195, 0.0007116
9: 0.9984772, 1.0113324, 0.9976686, 1.0113558, -0.0087683, 0.0096518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051306, upper bound: 0.0055470
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050316, upper bound: 0.0056222
time: 0.77 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0036330, -0.0006267, -0.0032179, -0.0006405, -0.0029925, 0.0025911
1: -0.0036680, 0.0035809, -0.0037179, 0.0026543, -0.0041422, 0.0051509
2: 0.0036021, 0.0096881, 0.0035981, 0.0091067, -0.0034719, 0.0041022
3: -0.0044861, -0.0035435, -0.0043849, -0.0035512, -0.0009348, 0.0008414
4: 0.0019136, 0.0073981, 0.0021321, 0.0072644, -0.0048368, 0.0047772
5: -0.0025225, 0.0028886, -0.0024348, 0.0025822, -0.0043502, 0.0045535
6: -0.0066069, -0.0034037, -0.0064867, -0.0034582, -0.0030462, 0.0029822
7: -0.0021337, 0.0033958, -0.0019213, 0.0032731, -0.0050414, 0.0049563
8: -0.0007939, -0.0000046, -0.0007760, -0.0000730, -0.0006954, 0.0007554
9: 0.9980419, 1.0107703, 0.9991087, 1.0107142, -0.0092323, 0.0081827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0051354
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0052630
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0036099, -0.0006324, -0.0032253, -0.0006028, -0.0030072, 0.0025928
1: -0.0036477, 0.0035723, -0.0037195, 0.0027624, -0.0041970, 0.0051666
2: 0.0036370, 0.0096826, 0.0036027, 0.0092742, -0.0036378, 0.0041316
3: -0.0044755, -0.0035460, -0.0043885, -0.0034970, -0.0009785, 0.0008425
4: 0.0019183, 0.0073566, 0.0019686, 0.0073196, -0.0048862, 0.0048862
5: -0.0024899, 0.0028834, -0.0024869, 0.0026904, -0.0044161, 0.0045954
6: -0.0065782, -0.0034055, -0.0065211, -0.0033544, -0.0031200, 0.0030174
7: -0.0021274, 0.0033448, -0.0021640, 0.0032973, -0.0051008, 0.0051842
8: -0.0007915, -0.0000142, -0.0008142, -0.0000816, -0.0007001, 0.0008001
9: 0.9980550, 1.0106908, 0.9988109, 1.0107478, -0.0092923, 0.0084032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053996, upper bound: 0.0049788
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053318, upper bound: 0.0050931
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0036330, -0.0006267, -0.0032553, -0.0005476, -0.0030854, 0.0026286
1: -0.0036680, 0.0035809, -0.0039981, 0.0028021, -0.0044001, 0.0055577
2: 0.0036021, 0.0096881, 0.0033250, 0.0092548, -0.0037261, 0.0044462
3: -0.0044861, -0.0035435, -0.0043855, -0.0035314, -0.0009547, 0.0008420
4: 0.0019136, 0.0073981, 0.0020143, 0.0074683, -0.0050186, 0.0048713
5: -0.0025225, 0.0028886, -0.0026697, 0.0027175, -0.0045042, 0.0047874
6: -0.0066069, -0.0034037, -0.0065764, -0.0034190, -0.0030793, 0.0030621
7: -0.0021337, 0.0033958, -0.0020035, 0.0034380, -0.0051693, 0.0050155
8: -0.0007939, -0.0000046, -0.0008058, -0.0000645, -0.0007182, 0.0008005
9: 0.9980419, 1.0107703, 0.9987524, 1.0113355, -0.0098894, 0.0086529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056580, upper bound: 0.0050645
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056580, upper bound: 0.0051936
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0036099, -0.0006324, -0.0032631, -0.0005095, -0.0031005, 0.0026306
1: -0.0036477, 0.0035723, -0.0039974, 0.0029119, -0.0044404, 0.0055719
2: 0.0036370, 0.0096826, 0.0033405, 0.0094077, -0.0038602, 0.0044469
3: -0.0044755, -0.0035460, -0.0043880, -0.0034787, -0.0009968, 0.0008421
4: 0.0019183, 0.0073566, 0.0018585, 0.0075049, -0.0050474, 0.0049716
5: -0.0024899, 0.0028834, -0.0027046, 0.0028209, -0.0045596, 0.0048081
6: -0.0065782, -0.0034055, -0.0066031, -0.0033140, -0.0031501, 0.0030861
7: -0.0021274, 0.0033448, -0.0022369, 0.0034441, -0.0052106, 0.0052218
8: -0.0007915, -0.0000142, -0.0008450, -0.0000708, -0.0007207, 0.0008308
9: 0.9980550, 1.0106908, 0.9984772, 1.0113324, -0.0099101, 0.0088449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055471, upper bound: 0.0051271
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056222, upper bound: 0.0050264
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0036330, -0.0006267, -0.0029513, 0.0029903
1: -0.0036097, 0.0035535, -0.0036680, 0.0035809, -0.0041113, 0.0041090
2: 0.0037012, 0.0096718, 0.0036021, 0.0096881, -0.0035039, 0.0035915
3: -0.0044504, -0.0035500, -0.0044861, -0.0035435, -0.0009069, 0.0009361
4: 0.0019263, 0.0072613, 0.0019136, 0.0073981, -0.0049653, 0.0048482
5: -0.0024145, 0.0028740, -0.0025225, 0.0028886, -0.0044825, 0.0045680
6: -0.0065203, -0.0034083, -0.0066069, -0.0034037, -0.0030383, 0.0031179
7: -0.0021167, 0.0032308, -0.0021337, 0.0033958, -0.0051664, 0.0050175
8: -0.0007868, -0.0000255, -0.0007939, -0.0000046, -0.0007523, 0.0007282
9: 0.9980801, 1.0105479, 0.9980419, 1.0107703, -0.0086098, 0.0084294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053882, upper bound: 0.0051786
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053882, upper bound: 0.0051786
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035935, -0.0006111, -0.0036099, -0.0006324, -0.0029611, 0.0029988
1: -0.0036105, 0.0036734, -0.0036477, 0.0035723, -0.0041274, 0.0041593
2: 0.0037211, 0.0098275, 0.0036370, 0.0096826, -0.0035288, 0.0037649
3: -0.0044549, -0.0034994, -0.0044755, -0.0035460, -0.0009089, 0.0009761
4: 0.0017662, 0.0073031, 0.0019183, 0.0073566, -0.0050745, 0.0048823
5: -0.0024555, 0.0029803, -0.0024899, 0.0028834, -0.0045142, 0.0046337
6: -0.0065462, -0.0033035, -0.0065782, -0.0034055, -0.0030637, 0.0031927
7: -0.0023565, 0.0032465, -0.0021274, 0.0033448, -0.0053944, 0.0050585
8: -0.0008217, -0.0000364, -0.0007915, -0.0000142, -0.0008055, 0.0007321
9: 0.9978067, 1.0105419, 0.9980550, 1.0106908, -0.0088341, 0.0084747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052717, upper bound: 0.0052324
time: 0.78 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053642, upper bound: 0.0051614
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0036764, -0.0005285, -0.0030040, 0.0030110
1: -0.0036097, 0.0035535, -0.0039607, 0.0037482, -0.0043770, 0.0045246
2: 0.0037012, 0.0096718, 0.0033294, 0.0098335, -0.0037805, 0.0039235
3: -0.0044504, -0.0035500, -0.0044869, -0.0035238, -0.0009266, 0.0009369
4: 0.0019263, 0.0072613, 0.0017780, 0.0076106, -0.0051420, 0.0049497
5: -0.0024145, 0.0028740, -0.0027735, 0.0030531, -0.0046461, 0.0047919
6: -0.0065203, -0.0034083, -0.0067005, -0.0033523, -0.0030750, 0.0031965
7: -0.0021167, 0.0032308, -0.0022337, 0.0035772, -0.0052896, 0.0050818
8: -0.0007868, -0.0000255, -0.0008234, 0.0000037, -0.0007759, 0.0007709
9: 0.9980801, 1.0105479, 0.9976562, 1.0114341, -0.0092509, 0.0089436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056836, upper bound: 0.0051039
time: 0.73 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056836, upper bound: 0.0051039
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035935, -0.0006111, -0.0036541, -0.0005348, -0.0030203, 0.0030430
1: -0.0036105, 0.0036734, -0.0039385, 0.0037400, -0.0043934, 0.0045716
2: 0.0037211, 0.0098275, 0.0033670, 0.0098283, -0.0038048, 0.0040860
3: -0.0044549, -0.0034994, -0.0044758, -0.0035265, -0.0009284, 0.0009764
4: 0.0017662, 0.0073031, 0.0017826, 0.0075665, -0.0052501, 0.0049837
5: -0.0024555, 0.0029803, -0.0027381, 0.0030481, -0.0046773, 0.0048555
6: -0.0065462, -0.0033035, -0.0066715, -0.0033543, -0.0031003, 0.0032708
7: -0.0023565, 0.0032465, -0.0022284, 0.0035247, -0.0055165, 0.0051229
8: -0.0008217, -0.0000364, -0.0008209, -0.0000055, -0.0008162, 0.0007745
9: 0.9978067, 1.0105419, 0.9976686, 1.0113558, -0.0094664, 0.0089873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055705, upper bound: 0.0051673
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056442, upper bound: 0.0050849
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0036764, -0.0005285, -0.0032179, -0.0006405, -0.0030359, 0.0026894
1: -0.0039607, 0.0037482, -0.0037179, 0.0026543, -0.0044458, 0.0052952
2: 0.0033294, 0.0098335, 0.0035981, 0.0091067, -0.0037676, 0.0043080
3: -0.0044869, -0.0035238, -0.0043849, -0.0035512, -0.0009357, 0.0008611
4: 0.0017780, 0.0076106, 0.0021321, 0.0072644, -0.0049579, 0.0049681
5: -0.0027735, 0.0030531, -0.0024348, 0.0025822, -0.0045985, 0.0047290
6: -0.0067005, -0.0033523, -0.0064867, -0.0034582, -0.0031301, 0.0030253
7: -0.0022337, 0.0035772, -0.0019213, 0.0032731, -0.0051182, 0.0050888
8: -0.0008234, 0.0000037, -0.0007760, -0.0000730, -0.0007407, 0.0007797
9: 0.9976562, 1.0114341, 0.9991087, 1.0107142, -0.0097196, 0.0088492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0054063
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0055074
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0036541, -0.0005348, -0.0032253, -0.0006028, -0.0030513, 0.0026904
1: -0.0039385, 0.0037400, -0.0037195, 0.0027624, -0.0044967, 0.0053108
2: 0.0033670, 0.0098283, 0.0036027, 0.0092742, -0.0039285, 0.0043378
3: -0.0044758, -0.0035265, -0.0043885, -0.0034970, -0.0009788, 0.0008620
4: 0.0017826, 0.0075665, 0.0019686, 0.0073196, -0.0050072, 0.0050754
5: -0.0027381, 0.0030481, -0.0024869, 0.0026904, -0.0046619, 0.0047707
6: -0.0066715, -0.0033543, -0.0065211, -0.0033544, -0.0032037, 0.0030605
7: -0.0022284, 0.0035247, -0.0021640, 0.0032973, -0.0051778, 0.0053165
8: -0.0008209, -0.0000055, -0.0008142, -0.0000816, -0.0007393, 0.0008088
9: 0.9976686, 1.0113558, 0.9988109, 1.0107478, -0.0097797, 0.0090603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053006, upper bound: 0.0052559
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052217, upper bound: 0.0053467
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0036764, -0.0005285, -0.0032553, -0.0005476, -0.0031288, 0.0027269
1: -0.0039607, 0.0037482, -0.0039981, 0.0028021, -0.0042981, 0.0053015
2: 0.0033294, 0.0098335, 0.0033250, 0.0092548, -0.0036168, 0.0042497
3: -0.0044869, -0.0035238, -0.0043855, -0.0035314, -0.0009555, 0.0008617
4: 0.0017780, 0.0076106, 0.0020143, 0.0074683, -0.0050126, 0.0049470
5: -0.0027735, 0.0030531, -0.0026697, 0.0027175, -0.0045288, 0.0047405
6: -0.0067005, -0.0033523, -0.0065764, -0.0034190, -0.0031377, 0.0030760
7: -0.0022337, 0.0035772, -0.0020035, 0.0034380, -0.0051772, 0.0050896
8: -0.0008234, 0.0000037, -0.0008058, -0.0000645, -0.0007076, 0.0007678
9: 0.9976562, 1.0114341, 0.9987524, 1.0113355, -0.0096188, 0.0085526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052348, upper bound: 0.0054063
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052348, upper bound: 0.0055073
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0036541, -0.0005348, -0.0032631, -0.0005095, -0.0031446, 0.0027282
1: -0.0039385, 0.0037400, -0.0039974, 0.0029119, -0.0043469, 0.0053143
2: 0.0033670, 0.0098283, 0.0033405, 0.0094077, -0.0037721, 0.0042673
3: -0.0044758, -0.0035265, -0.0043880, -0.0034787, -0.0009971, 0.0008615
4: 0.0017826, 0.0075665, 0.0018585, 0.0075049, -0.0050471, 0.0050518
5: -0.0027381, 0.0030481, -0.0027046, 0.0028209, -0.0045966, 0.0047691
6: -0.0066715, -0.0033543, -0.0066031, -0.0033140, -0.0032090, 0.0031014
7: -0.0022284, 0.0035247, -0.0022369, 0.0034441, -0.0052198, 0.0053000
8: -0.0008209, -0.0000055, -0.0008450, -0.0000708, -0.0007116, 0.0008195
9: 0.9976686, 1.0113558, 0.9984772, 1.0113324, -0.0096518, 0.0087683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051300, upper bound: 0.0054319
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052255, upper bound: 0.0053467
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0036764, -0.0005285, -0.0035914, -0.0006426, -0.0030110, 0.0030040
1: -0.0039607, 0.0037482, -0.0036097, 0.0035535, -0.0045246, 0.0043770
2: 0.0033294, 0.0098335, 0.0037012, 0.0096718, -0.0039235, 0.0037805
3: -0.0044869, -0.0035238, -0.0044504, -0.0035500, -0.0009369, 0.0009266
4: 0.0017780, 0.0076106, 0.0019263, 0.0072613, -0.0049497, 0.0051420
5: -0.0027735, 0.0030531, -0.0024145, 0.0028740, -0.0047919, 0.0046461
6: -0.0067005, -0.0033523, -0.0065203, -0.0034083, -0.0031965, 0.0030750
7: -0.0022337, 0.0035772, -0.0021167, 0.0032308, -0.0050818, 0.0052896
8: -0.0008234, 0.0000037, -0.0007868, -0.0000255, -0.0007709, 0.0007759
9: 0.9976562, 1.0114341, 0.9980801, 1.0105479, -0.0089436, 0.0092509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052560, upper bound: 0.0054449
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052560, upper bound: 0.0055488
time: 0.74 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0036541, -0.0005348, -0.0035935, -0.0006111, -0.0030430, 0.0030203
1: -0.0039385, 0.0037400, -0.0036105, 0.0036734, -0.0045716, 0.0043934
2: 0.0033670, 0.0098283, 0.0037211, 0.0098275, -0.0040860, 0.0038048
3: -0.0044758, -0.0035265, -0.0044549, -0.0034994, -0.0009764, 0.0009284
4: 0.0017826, 0.0075665, 0.0017662, 0.0073031, -0.0049837, 0.0052501
5: -0.0027381, 0.0030481, -0.0024555, 0.0029803, -0.0048555, 0.0046773
6: -0.0066715, -0.0033543, -0.0065462, -0.0033035, -0.0032708, 0.0031003
7: -0.0022284, 0.0035247, -0.0023565, 0.0032465, -0.0051229, 0.0055165
8: -0.0008209, -0.0000055, -0.0008217, -0.0000364, -0.0007745, 0.0008162
9: 0.9976686, 1.0113558, 0.9978067, 1.0105419, -0.0089873, 0.0094664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053239, upper bound: 0.0053001
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052484, upper bound: 0.0053974
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036352, -0.0005462, -0.0036764, -0.0005285, -0.0030625, 0.0031209
1: -0.0039025, 0.0037230, -0.0039607, 0.0037482, -0.0042790, 0.0042781
2: 0.0034264, 0.0098184, 0.0033294, 0.0098335, -0.0036619, 0.0037499
3: -0.0044516, -0.0035309, -0.0044869, -0.0035238, -0.0009277, 0.0009560
4: 0.0017904, 0.0074737, 0.0017780, 0.0076106, -0.0051337, 0.0050138
5: -0.0026671, 0.0030393, -0.0027735, 0.0030531, -0.0046617, 0.0047483
6: -0.0066113, -0.0033572, -0.0067005, -0.0033523, -0.0031266, 0.0032080
7: -0.0022195, 0.0034090, -0.0022337, 0.0035772, -0.0052972, 0.0051466
8: -0.0008161, -0.0000171, -0.0008234, 0.0000037, -0.0007645, 0.0007407
9: 0.9976916, 1.0112109, 0.9976562, 1.0114341, -0.0090019, 0.0088182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052607, upper bound: 0.0054449
time: 0.77 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052607, upper bound: 0.0054449
time: 0.93 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0036399, -0.0005157, -0.0036541, -0.0005348, -0.0030745, 0.0031384
1: -0.0038971, 0.0038326, -0.0039385, 0.0037400, -0.0042896, 0.0043245
2: 0.0034569, 0.0099649, 0.0033670, 0.0098283, -0.0036777, 0.0039135
3: -0.0044556, -0.0034804, -0.0044758, -0.0035265, -0.0009291, 0.0009954
4: 0.0016332, 0.0074996, 0.0017826, 0.0075665, -0.0052392, 0.0050413
5: -0.0026923, 0.0031408, -0.0027381, 0.0030481, -0.0046859, 0.0048129
6: -0.0066328, -0.0032506, -0.0066715, -0.0033543, -0.0031471, 0.0032794
7: -0.0024508, 0.0034109, -0.0022284, 0.0035247, -0.0055127, 0.0051831
8: -0.0008521, -0.0000264, -0.0008209, -0.0000055, -0.0008165, 0.0007439
9: 0.9974241, 1.0112014, 0.9976686, 1.0113558, -0.0092181, 0.0088476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051522, upper bound: 0.0054715
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052485, upper bound: 0.0053974
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.98 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0051354
IS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0051354
IS_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0051898
IS_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050931, upper bound: 0.0050931
IS_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0054076, upper bound: 0.0050645
IS_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0054076, upper bound: 0.0050645
IS_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052613, upper bound: 0.0051271
IS_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053494, upper bound: 0.0050264
IS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0053627
IS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051354, upper bound: 0.0053627
IS_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0053996
IS_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050931, upper bound: 0.0053318
IS_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0054063, upper bound: 0.0052307
IS_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0054063, upper bound: 0.0052307
IS_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052559, upper bound: 0.0053006
IS_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053467, upper bound: 0.0052217
IS_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0054076
IS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0055113
IS_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051271, upper bound: 0.0052613
IS_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050264, upper bound: 0.0053494
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0054076
IS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0054076
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0054343
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050316, upper bound: 0.0053494
IS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0056580
IS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050645, upper bound: 0.0056580
IS_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051271, upper bound: 0.0055470
IS_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050264, upper bound: 0.0056222
IS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0056580
IS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050692, upper bound: 0.0056580
IS_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051306, upper bound: 0.0055470
IS_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0050316, upper bound: 0.0056222
IS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0051354
IS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053627, upper bound: 0.0052630
IS_A2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053996, upper bound: 0.0049788
IS_A2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053318, upper bound: 0.0050931
IS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0056580, upper bound: 0.0050645
IS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0056580, upper bound: 0.0051936
IS_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0055471, upper bound: 0.0051271
IS_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0056222, upper bound: 0.0050264
IS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053882, upper bound: 0.0051786
IS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053882, upper bound: 0.0051786
IS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052717, upper bound: 0.0052324
IS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053642, upper bound: 0.0051614
IS_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0056836, upper bound: 0.0051039
IS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0056836, upper bound: 0.0051039
IS_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0055705, upper bound: 0.0051673
IS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0056442, upper bound: 0.0050849
IS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0054063
IS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052307, upper bound: 0.0055074
IS_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053006, upper bound: 0.0052559
IS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052217, upper bound: 0.0053467
IS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052348, upper bound: 0.0054063
IS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052348, upper bound: 0.0055073
IS_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051300, upper bound: 0.0054319
IS_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052255, upper bound: 0.0053467
IS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052560, upper bound: 0.0054449
IS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052560, upper bound: 0.0055488
IS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0053239, upper bound: 0.0053001
IS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052484, upper bound: 0.0053974
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052607, upper bound: 0.0054449
IS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052607, upper bound: 0.0054449
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0051522, upper bound: 0.0054715
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 9, lower bound: -0.0052485, upper bound: 0.0053974

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0032179, -0.0006405, -0.0025774, 0.0025774
1: -0.0037179, 0.0026543, -0.0037179, 0.0026543, -0.0037743, 0.0037743
2: 0.0035981, 0.0091067, 0.0035981, 0.0091067, -0.0032619, 0.0032619
3: -0.0043849, -0.0035512, -0.0043849, -0.0035512, -0.0008337, 0.0008337
4: 0.0021321, 0.0072644, 0.0021321, 0.0072644, -0.0046482, 0.0046482
5: -0.0024348, 0.0025822, -0.0024348, 0.0025822, -0.0042574, 0.0042574
6: -0.0064867, -0.0034582, -0.0064867, -0.0034582, -0.0029275, 0.0029275
7: -0.0019213, 0.0032731, -0.0019213, 0.0032731, -0.0047858, 0.0047858
8: -0.0007760, -0.0000730, -0.0007760, -0.0000730, -0.0006758, 0.0006758
9: 0.9991087, 1.0107142, 0.9991087, 1.0107142, -0.0079329, 0.0079329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0048787
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0049802
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0032253, -0.0006028, -0.0026151, 0.0025848
1: -0.0037179, 0.0026543, -0.0037195, 0.0027624, -0.0038522, 0.0037391
2: 0.0035981, 0.0091067, 0.0036027, 0.0092742, -0.0034645, 0.0032975
3: -0.0043849, -0.0035512, -0.0043885, -0.0034970, -0.0008879, 0.0008372
4: 0.0021321, 0.0072644, 0.0019686, 0.0073196, -0.0046978, 0.0048026
5: -0.0024348, 0.0025822, -0.0024869, 0.0026904, -0.0043596, 0.0043014
6: -0.0064867, -0.0034582, -0.0065211, -0.0033544, -0.0030313, 0.0029633
7: -0.0019213, 0.0032731, -0.0021640, 0.0032973, -0.0048451, 0.0050660
8: -0.0007760, -0.0000730, -0.0008142, -0.0000816, -0.0006868, 0.0007360
9: 0.9991087, 1.0107142, 0.9988109, 1.0107478, -0.0079851, 0.0082306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0048787
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0049802
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0031724, -0.0006436, -0.0025791, 0.0025691
1: -0.0037189, 0.0027595, -0.0037418, 0.0025947, -0.0037009, 0.0038456
2: 0.0036031, 0.0092730, 0.0035372, 0.0090836, -0.0032741, 0.0035176
3: -0.0043870, -0.0034972, -0.0043695, -0.0035521, -0.0008348, 0.0008723
4: 0.0019695, 0.0073184, 0.0021493, 0.0073250, -0.0048231, 0.0046702
5: -0.0024858, 0.0026886, -0.0024781, 0.0025463, -0.0042559, 0.0043645
6: -0.0065199, -0.0033547, -0.0065137, -0.0034642, -0.0029525, 0.0030476
7: -0.0021635, 0.0032949, -0.0019155, 0.0033203, -0.0051176, 0.0048450
8: -0.0008141, -0.0000828, -0.0007756, -0.0000927, -0.0007214, 0.0006784
9: 0.9988145, 1.0107462, 0.9991799, 1.0108290, -0.0082825, 0.0079203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0031182, -0.0005723, -0.0026098, 0.0025057
1: -0.0037081, 0.0026755, -0.0037650, 0.0024167, -0.0036394, 0.0039718
2: 0.0036091, 0.0092352, 0.0035342, 0.0090103, -0.0032543, 0.0034689
3: -0.0043757, -0.0034998, -0.0043734, -0.0034755, -0.0009002, 0.0008736
4: 0.0019918, 0.0072965, 0.0021678, 0.0073134, -0.0047709, 0.0046759
5: -0.0024660, 0.0026437, -0.0024889, 0.0024614, -0.0042515, 0.0043188
6: -0.0065015, -0.0033608, -0.0065013, -0.0034428, -0.0029638, 0.0030209
7: -0.0021552, 0.0032676, -0.0020205, 0.0033038, -0.0050896, 0.0049095
8: -0.0008111, -0.0000937, -0.0008434, -0.0000955, -0.0007156, 0.0007497
9: 0.9989144, 1.0107208, 0.9993728, 1.0108525, -0.0081818, 0.0078657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0050931
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0050931
time: 0.99 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0032553, -0.0005476, -0.0026703, 0.0026148
1: -0.0037179, 0.0026543, -0.0039981, 0.0028021, -0.0040322, 0.0041811
2: 0.0035981, 0.0091067, 0.0033250, 0.0092548, -0.0035161, 0.0036059
3: -0.0043849, -0.0035512, -0.0043855, -0.0035314, -0.0008535, 0.0008343
4: 0.0021321, 0.0072644, 0.0020143, 0.0074683, -0.0048300, 0.0047423
5: -0.0024348, 0.0025822, -0.0026697, 0.0027175, -0.0044114, 0.0044913
6: -0.0064867, -0.0034582, -0.0065764, -0.0034190, -0.0029606, 0.0030073
7: -0.0019213, 0.0032731, -0.0020035, 0.0034380, -0.0049137, 0.0048449
8: -0.0007760, -0.0000730, -0.0008058, -0.0000645, -0.0006986, 0.0007210
9: 0.9991087, 1.0107142, 0.9987524, 1.0113355, -0.0085900, 0.0084031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052288, upper bound: 0.0049966
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053186, upper bound: 0.0049074
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0032631, -0.0005095, -0.0027084, 0.0026226
1: -0.0037179, 0.0026543, -0.0039974, 0.0029119, -0.0040956, 0.0041408
2: 0.0035981, 0.0091067, 0.0033405, 0.0094077, -0.0036869, 0.0036003
3: -0.0043849, -0.0035512, -0.0043880, -0.0034787, -0.0009062, 0.0008368
4: 0.0021321, 0.0072644, 0.0018585, 0.0075049, -0.0048588, 0.0048881
5: -0.0024348, 0.0025822, -0.0027046, 0.0028209, -0.0045031, 0.0045138
6: -0.0064867, -0.0034582, -0.0066031, -0.0033140, -0.0030614, 0.0030320
7: -0.0019213, 0.0032731, -0.0022369, 0.0034441, -0.0049553, 0.0051036
8: -0.0007760, -0.0000730, -0.0008450, -0.0000708, -0.0007052, 0.0007719
9: 0.9991087, 1.0107142, 0.9984772, 1.0113324, -0.0085974, 0.0086723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052288, upper bound: 0.0049966
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053186, upper bound: 0.0049074
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0032098, -0.0005500, -0.0026727, 0.0026065
1: -0.0037189, 0.0027595, -0.0040228, 0.0027342, -0.0039549, 0.0042530
2: 0.0036031, 0.0092730, 0.0032673, 0.0092299, -0.0035151, 0.0038475
3: -0.0043870, -0.0034972, -0.0043700, -0.0035320, -0.0008550, 0.0008728
4: 0.0019695, 0.0073184, 0.0020338, 0.0075327, -0.0050058, 0.0047593
5: -0.0024858, 0.0026886, -0.0027161, 0.0026787, -0.0043980, 0.0045988
6: -0.0065199, -0.0033547, -0.0066087, -0.0034257, -0.0029845, 0.0031296
7: -0.0021635, 0.0032949, -0.0019963, 0.0034911, -0.0052459, 0.0049013
8: -0.0008141, -0.0000828, -0.0008057, -0.0000827, -0.0007313, 0.0007229
9: 0.9988145, 1.0107462, 0.9988286, 1.0114442, -0.0089363, 0.0083608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0031524, -0.0004757, -0.0027064, 0.0025399
1: -0.0037081, 0.0026755, -0.0040652, 0.0025758, -0.0038941, 0.0043852
2: 0.0036091, 0.0092352, 0.0032590, 0.0091630, -0.0034863, 0.0038261
3: -0.0043757, -0.0034998, -0.0043751, -0.0034505, -0.0009252, 0.0008753
4: 0.0019918, 0.0072965, 0.0020498, 0.0075269, -0.0049602, 0.0047594
5: -0.0024660, 0.0026437, -0.0027340, 0.0025987, -0.0043817, 0.0045730
6: -0.0065015, -0.0033608, -0.0065964, -0.0034057, -0.0029954, 0.0031027
7: -0.0021552, 0.0032676, -0.0020954, 0.0034718, -0.0052159, 0.0049599
8: -0.0008111, -0.0000937, -0.0008737, -0.0000866, -0.0007245, 0.0007800
9: 0.9989144, 1.0107208, 0.9990121, 1.0114831, -0.0088886, 0.0082887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052571, upper bound: 0.0050264
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052571, upper bound: 0.0050264
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0035914, -0.0006426, -0.0025752, 0.0029509
1: -0.0037179, 0.0026543, -0.0036097, 0.0035535, -0.0051160, 0.0040935
2: 0.0035981, 0.0091067, 0.0037012, 0.0096718, -0.0040746, 0.0033650
3: -0.0043849, -0.0035512, -0.0044504, -0.0035500, -0.0008349, 0.0008992
4: 0.0021321, 0.0072644, 0.0019263, 0.0072613, -0.0046498, 0.0048275
5: -0.0024348, 0.0025822, -0.0024145, 0.0028740, -0.0045411, 0.0042552
6: -0.0064867, -0.0034582, -0.0065203, -0.0034083, -0.0029772, 0.0029617
7: -0.0019213, 0.0032731, -0.0021167, 0.0032308, -0.0047809, 0.0050162
8: -0.0007760, -0.0000730, -0.0007868, -0.0000255, -0.0007268, 0.0006893
9: 0.9991087, 1.0107142, 0.9980801, 1.0105479, -0.0079655, 0.0091911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0051463
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0052205
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0035935, -0.0006111, -0.0026067, 0.0029530
1: -0.0037179, 0.0026543, -0.0036105, 0.0036734, -0.0052221, 0.0040811
2: 0.0035981, 0.0091067, 0.0037211, 0.0098275, -0.0042699, 0.0033635
3: -0.0043849, -0.0035512, -0.0044549, -0.0034994, -0.0008855, 0.0009037
4: 0.0021321, 0.0072644, 0.0017662, 0.0073031, -0.0046807, 0.0049770
5: -0.0024348, 0.0025822, -0.0024555, 0.0029803, -0.0046345, 0.0042835
6: -0.0064867, -0.0034582, -0.0065462, -0.0033035, -0.0030818, 0.0029846
7: -0.0019213, 0.0032731, -0.0023565, 0.0032465, -0.0048230, 0.0052939
8: -0.0007760, -0.0000730, -0.0008217, -0.0000364, -0.0007357, 0.0007484
9: 0.9991087, 1.0107142, 0.9978067, 1.0105419, -0.0079838, 0.0094876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0051463
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0052205
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0035357, -0.0006462, -0.0025765, 0.0029324
1: -0.0037189, 0.0027595, -0.0036333, 0.0034834, -0.0050371, 0.0041794
2: 0.0036031, 0.0092730, 0.0036461, 0.0096436, -0.0040750, 0.0036116
3: -0.0043870, -0.0034972, -0.0044349, -0.0035508, -0.0008362, 0.0009377
4: 0.0019695, 0.0073184, 0.0019465, 0.0073254, -0.0048192, 0.0048484
5: -0.0024858, 0.0026886, -0.0024622, 0.0028336, -0.0045313, 0.0043577
6: -0.0065199, -0.0033547, -0.0065435, -0.0034141, -0.0030023, 0.0030751
7: -0.0021635, 0.0032949, -0.0021097, 0.0032796, -0.0051070, 0.0050748
8: -0.0008141, -0.0000828, -0.0007868, -0.0000465, -0.0007675, 0.0006926
9: 0.9988145, 1.0107462, 0.9981599, 1.0106546, -0.0083168, 0.0091532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0034840, -0.0005637, -0.0026184, 0.0028715
1: -0.0037081, 0.0026755, -0.0036809, 0.0033280, -0.0049695, 0.0042859
2: 0.0036091, 0.0092352, 0.0036326, 0.0095784, -0.0040426, 0.0035859
3: -0.0043757, -0.0034998, -0.0044399, -0.0034729, -0.0009028, 0.0009400
4: 0.0019918, 0.0072965, 0.0019641, 0.0073213, -0.0047751, 0.0048522
5: -0.0024660, 0.0026437, -0.0024797, 0.0027574, -0.0045183, 0.0043297
6: -0.0065015, -0.0033608, -0.0065319, -0.0033934, -0.0030134, 0.0030513
7: -0.0021552, 0.0032676, -0.0022099, 0.0032643, -0.0050830, 0.0051289
8: -0.0008111, -0.0000937, -0.0008541, -0.0000488, -0.0007623, 0.0007604
9: 0.9989144, 1.0107208, 0.9983432, 1.0106931, -0.0082653, 0.0090803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0053318
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0053318
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0036352, -0.0005462, -0.0026717, 0.0029947
1: -0.0037179, 0.0026543, -0.0039025, 0.0037230, -0.0052591, 0.0043966
2: 0.0035981, 0.0091067, 0.0034264, 0.0098184, -0.0042814, 0.0036647
3: -0.0043849, -0.0035512, -0.0044516, -0.0035309, -0.0008540, 0.0009003
4: 0.0021321, 0.0072644, 0.0017904, 0.0074737, -0.0048419, 0.0049482
5: -0.0024348, 0.0025822, -0.0026671, 0.0030393, -0.0047160, 0.0045020
6: -0.0064867, -0.0034582, -0.0066113, -0.0033572, -0.0030202, 0.0030447
7: -0.0019213, 0.0032731, -0.0022195, 0.0034090, -0.0049121, 0.0050933
8: -0.0007760, -0.0000730, -0.0008161, -0.0000171, -0.0007531, 0.0007337
9: 0.9991087, 1.0107142, 0.9976916, 1.0112109, -0.0086399, 0.0096785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053931, upper bound: 0.0050335
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053165, upper bound: 0.0050927
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032179, -0.0006405, -0.0036399, -0.0005157, -0.0027021, 0.0029994
1: -0.0037179, 0.0026543, -0.0038971, 0.0038326, -0.0053527, 0.0043693
2: 0.0035981, 0.0091067, 0.0034569, 0.0099649, -0.0044467, 0.0036419
3: -0.0043849, -0.0035512, -0.0044556, -0.0034804, -0.0009045, 0.0009043
4: 0.0021321, 0.0072644, 0.0016332, 0.0074996, -0.0048599, 0.0050937
5: -0.0024348, 0.0025822, -0.0026923, 0.0031408, -0.0048103, 0.0045167
6: -0.0064867, -0.0034582, -0.0066328, -0.0032506, -0.0031222, 0.0030626
7: -0.0019213, 0.0032731, -0.0024508, 0.0034109, -0.0049488, 0.0053598
8: -0.0007760, -0.0000730, -0.0008521, -0.0000264, -0.0007496, 0.0007790
9: 0.9991087, 1.0107142, 0.9974241, 1.0112014, -0.0086138, 0.0099502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053931, upper bound: 0.0050335
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053165, upper bound: 0.0050927
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0035783, -0.0005491, -0.0026736, 0.0029750
1: -0.0037189, 0.0027595, -0.0039241, 0.0036458, -0.0051703, 0.0044783
2: 0.0036031, 0.0092730, 0.0033760, 0.0097879, -0.0042758, 0.0039041
3: -0.0043870, -0.0034972, -0.0044351, -0.0035314, -0.0008555, 0.0009379
4: 0.0019695, 0.0073184, 0.0018111, 0.0075371, -0.0050110, 0.0049627
5: -0.0024858, 0.0026886, -0.0027121, 0.0029982, -0.0047007, 0.0046073
6: -0.0065199, -0.0033547, -0.0066377, -0.0033639, -0.0030441, 0.0031596
7: -0.0021635, 0.0032949, -0.0022116, 0.0034612, -0.0052409, 0.0051490
8: -0.0008141, -0.0000828, -0.0008162, -0.0000372, -0.0007768, 0.0007334
9: 0.9988145, 1.0107462, 0.9977753, 1.0113204, -0.0089803, 0.0096313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0035275, -0.0004623, -0.0027197, 0.0029151
1: -0.0037081, 0.0026755, -0.0039799, 0.0035038, -0.0051102, 0.0046084
2: 0.0036091, 0.0092352, 0.0033539, 0.0097286, -0.0042415, 0.0038931
3: -0.0043757, -0.0034998, -0.0044413, -0.0034482, -0.0009276, 0.0009415
4: 0.0019918, 0.0072965, 0.0018290, 0.0075414, -0.0049753, 0.0049584
5: -0.0024660, 0.0026437, -0.0027384, 0.0029209, -0.0046807, 0.0045929
6: -0.0065015, -0.0033608, -0.0066312, -0.0033443, -0.0030539, 0.0031383
7: -0.0021552, 0.0032676, -0.0023068, 0.0034495, -0.0052213, 0.0051930
8: -0.0008111, -0.0000937, -0.0008839, -0.0000401, -0.0007710, 0.0007902
9: 0.9989144, 1.0107208, 0.9979419, 1.0113884, -0.0089602, 0.0095518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052556, upper bound: 0.0052217
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052556, upper bound: 0.0052217
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0032179, -0.0006405, -0.0026148, 0.0026703
1: -0.0039981, 0.0028021, -0.0037179, 0.0026543, -0.0041811, 0.0040322
2: 0.0033250, 0.0092548, 0.0035981, 0.0091067, -0.0036059, 0.0035161
3: -0.0043855, -0.0035314, -0.0043849, -0.0035512, -0.0008343, 0.0008535
4: 0.0020143, 0.0074683, 0.0021321, 0.0072644, -0.0047423, 0.0048300
5: -0.0026697, 0.0027175, -0.0024348, 0.0025822, -0.0044913, 0.0044114
6: -0.0065764, -0.0034190, -0.0064867, -0.0034582, -0.0030073, 0.0029606
7: -0.0020035, 0.0034380, -0.0019213, 0.0032731, -0.0048449, 0.0049137
8: -0.0008058, -0.0000645, -0.0007760, -0.0000730, -0.0007210, 0.0006986
9: 0.9987524, 1.0113355, 0.9991087, 1.0107142, -0.0084031, 0.0085900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052288
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0053186
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032631, -0.0005095, -0.0032179, -0.0006405, -0.0026226, 0.0027084
1: -0.0039974, 0.0029119, -0.0037179, 0.0026543, -0.0041408, 0.0040956
2: 0.0033405, 0.0094077, 0.0035981, 0.0091067, -0.0036003, 0.0036869
3: -0.0043880, -0.0034787, -0.0043849, -0.0035512, -0.0008368, 0.0009062
4: 0.0018585, 0.0075049, 0.0021321, 0.0072644, -0.0048881, 0.0048588
5: -0.0027046, 0.0028209, -0.0024348, 0.0025822, -0.0045138, 0.0045031
6: -0.0066031, -0.0033140, -0.0064867, -0.0034582, -0.0030320, 0.0030614
7: -0.0022369, 0.0034441, -0.0019213, 0.0032731, -0.0051036, 0.0049553
8: -0.0008450, -0.0000708, -0.0007760, -0.0000730, -0.0007719, 0.0007052
9: 0.9984772, 1.0113324, 0.9991087, 1.0107142, -0.0086723, 0.0085974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052613
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0053494
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032098, -0.0005500, -0.0032227, -0.0006033, -0.0026065, 0.0026727
1: -0.0040228, 0.0027342, -0.0037189, 0.0027595, -0.0042530, 0.0039549
2: 0.0032673, 0.0092299, 0.0036031, 0.0092730, -0.0038475, 0.0035151
3: -0.0043700, -0.0035320, -0.0043870, -0.0034972, -0.0008728, 0.0008550
4: 0.0020338, 0.0075327, 0.0019695, 0.0073184, -0.0047593, 0.0050058
5: -0.0027161, 0.0026787, -0.0024858, 0.0026886, -0.0045988, 0.0043980
6: -0.0066087, -0.0034257, -0.0065199, -0.0033547, -0.0031296, 0.0029845
7: -0.0019963, 0.0034911, -0.0021635, 0.0032949, -0.0049013, 0.0052459
8: -0.0008057, -0.0000827, -0.0008141, -0.0000828, -0.0007229, 0.0007313
9: 0.9988286, 1.0114442, 0.9988145, 1.0107462, -0.0083608, 0.0089363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0051853
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052613
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0031524, -0.0004757, -0.0031821, -0.0006124, -0.0025399, 0.0027064
1: -0.0040652, 0.0025758, -0.0037081, 0.0026755, -0.0043852, 0.0038941
2: 0.0032590, 0.0091630, 0.0036091, 0.0092352, -0.0038261, 0.0034863
3: -0.0043751, -0.0034505, -0.0043757, -0.0034998, -0.0008753, 0.0009252
4: 0.0020498, 0.0075269, 0.0019918, 0.0072965, -0.0047594, 0.0049602
5: -0.0027340, 0.0025987, -0.0024660, 0.0026437, -0.0045730, 0.0043817
6: -0.0065964, -0.0034057, -0.0065015, -0.0033608, -0.0031027, 0.0029954
7: -0.0020954, 0.0034718, -0.0021552, 0.0032676, -0.0049599, 0.0052159
8: -0.0008737, -0.0000866, -0.0008111, -0.0000937, -0.0007800, 0.0007245
9: 0.9990121, 1.0114831, 0.9989144, 1.0107208, -0.0082887, 0.0088886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0052571
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0053494
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0032553, -0.0005476, -0.0027077, 0.0027077
1: -0.0039981, 0.0028021, -0.0039981, 0.0028021, -0.0039264, 0.0039264
2: 0.0033250, 0.0092548, 0.0033250, 0.0092548, -0.0034056, 0.0034056
3: -0.0043855, -0.0035314, -0.0043855, -0.0035314, -0.0008541, 0.0008541
4: 0.0020143, 0.0074683, 0.0020143, 0.0074683, -0.0048101, 0.0048101
5: -0.0026697, 0.0027175, -0.0026697, 0.0027175, -0.0044325, 0.0044325
6: -0.0065764, -0.0034190, -0.0065764, -0.0034190, -0.0030151, 0.0030152
7: -0.0020035, 0.0034380, -0.0020035, 0.0034380, -0.0049149, 0.0049149
8: -0.0008058, -0.0000645, -0.0008058, -0.0000645, -0.0006884, 0.0006884
9: 0.9987524, 1.0113355, 0.9987524, 1.0113355, -0.0082985, 0.0082985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0051853
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0052571
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0032631, -0.0005095, -0.0027458, 0.0027155
1: -0.0039981, 0.0028021, -0.0039974, 0.0029119, -0.0039988, 0.0038939
2: 0.0033250, 0.0092548, 0.0033405, 0.0094077, -0.0035975, 0.0034287
3: -0.0043855, -0.0035314, -0.0043880, -0.0034787, -0.0009068, 0.0008566
4: 0.0020143, 0.0074683, 0.0018585, 0.0075049, -0.0048458, 0.0049606
5: -0.0026697, 0.0027175, -0.0027046, 0.0028209, -0.0045367, 0.0044638
6: -0.0065764, -0.0034190, -0.0066031, -0.0033140, -0.0031159, 0.0030413
7: -0.0020035, 0.0034380, -0.0022369, 0.0034441, -0.0049571, 0.0051780
8: -0.0008058, -0.0000645, -0.0008450, -0.0000708, -0.0006987, 0.0007462
9: 0.9987524, 1.0113355, 0.9984772, 1.0113324, -0.0083279, 0.0085962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0051853
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0052571
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032605, -0.0005100, -0.0032098, -0.0005500, -0.0027105, 0.0026998
1: -0.0039968, 0.0029088, -0.0040228, 0.0027342, -0.0038554, 0.0039929
2: 0.0033409, 0.0094064, 0.0032673, 0.0092299, -0.0034032, 0.0036467
3: -0.0043865, -0.0034789, -0.0043700, -0.0035320, -0.0008546, 0.0008911
4: 0.0018595, 0.0075038, 0.0020338, 0.0075327, -0.0049856, 0.0048151
5: -0.0027036, 0.0028191, -0.0027161, 0.0026787, -0.0044130, 0.0045441
6: -0.0066019, -0.0033144, -0.0066087, -0.0034257, -0.0030295, 0.0031348
7: -0.0022364, 0.0034418, -0.0019963, 0.0034911, -0.0052279, 0.0049549
8: -0.0008448, -0.0000720, -0.0008057, -0.0000827, -0.0007318, 0.0006905
9: 0.9984809, 1.0113308, 0.9988286, 1.0114442, -0.0086488, 0.0082487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0032183, -0.0005190, -0.0031524, -0.0004757, -0.0027426, 0.0026334
1: -0.0039864, 0.0028270, -0.0040652, 0.0025758, -0.0038012, 0.0041155
2: 0.0033465, 0.0093697, 0.0032590, 0.0091630, -0.0033869, 0.0036111
3: -0.0043757, -0.0034815, -0.0043751, -0.0034505, -0.0009252, 0.0008936
4: 0.0018815, 0.0074840, 0.0020498, 0.0075269, -0.0049391, 0.0048244
5: -0.0026857, 0.0027745, -0.0027340, 0.0025987, -0.0044121, 0.0045049
6: -0.0065842, -0.0033204, -0.0065964, -0.0034057, -0.0030423, 0.0031079
7: -0.0022280, 0.0034141, -0.0020954, 0.0034718, -0.0051976, 0.0050141
8: -0.0008420, -0.0000834, -0.0008737, -0.0000866, -0.0007337, 0.0007662
9: 0.9985780, 1.0113076, 0.9990121, 1.0114831, -0.0085589, 0.0082073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0053494
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0053494
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0035914, -0.0006426, -0.0026127, 0.0030438
1: -0.0039981, 0.0028021, -0.0036097, 0.0035535, -0.0055227, 0.0043514
2: 0.0033250, 0.0092548, 0.0037012, 0.0096718, -0.0044186, 0.0036191
3: -0.0043855, -0.0035314, -0.0044504, -0.0035500, -0.0008355, 0.0009190
4: 0.0020143, 0.0074683, 0.0019263, 0.0072613, -0.0047438, 0.0050093
5: -0.0026697, 0.0027175, -0.0024145, 0.0028740, -0.0047749, 0.0044092
6: -0.0065764, -0.0034190, -0.0065203, -0.0034083, -0.0030570, 0.0029947
7: -0.0020035, 0.0034380, -0.0021167, 0.0032308, -0.0048400, 0.0051441
8: -0.0008058, -0.0000645, -0.0007868, -0.0000255, -0.0007719, 0.0007121
9: 0.9987524, 1.0113355, 0.9980801, 1.0105479, -0.0084356, 0.0098481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050783, upper bound: 0.0054663
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049899, upper bound: 0.0055205
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0035935, -0.0006111, -0.0026442, 0.0030459
1: -0.0039981, 0.0028021, -0.0036105, 0.0036734, -0.0056288, 0.0043390
2: 0.0033250, 0.0092548, 0.0037211, 0.0098275, -0.0046139, 0.0036177
3: -0.0043855, -0.0035314, -0.0044549, -0.0034994, -0.0008861, 0.0009235
4: 0.0020143, 0.0074683, 0.0017662, 0.0073031, -0.0047747, 0.0051588
5: -0.0026697, 0.0027175, -0.0024555, 0.0029803, -0.0048684, 0.0044375
6: -0.0065764, -0.0034190, -0.0065462, -0.0033035, -0.0031616, 0.0030177
7: -0.0020035, 0.0034380, -0.0023565, 0.0032465, -0.0048822, 0.0054218
8: -0.0008058, -0.0000645, -0.0008217, -0.0000364, -0.0007694, 0.0007571
9: 0.9987524, 1.0113355, 0.9978067, 1.0105419, -0.0084539, 0.0101447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050783, upper bound: 0.0054662
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049899, upper bound: 0.0055205
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0031938, -0.0005238, -0.0036072, -0.0006330, -0.0025608, 0.0030833
1: -0.0039825, 0.0028280, -0.0036472, 0.0035689, -0.0055543, 0.0043113
2: 0.0033503, 0.0093729, 0.0036373, 0.0096811, -0.0044205, 0.0038107
3: -0.0043480, -0.0034835, -0.0044740, -0.0035461, -0.0008018, 0.0009905
4: 0.0018845, 0.0074735, 0.0019193, 0.0073554, -0.0049315, 0.0049854
5: -0.0026764, 0.0027742, -0.0024889, 0.0028815, -0.0047539, 0.0044941
6: -0.0065717, -0.0033234, -0.0065769, -0.0034059, -0.0030445, 0.0031341
7: -0.0022210, 0.0033825, -0.0021268, 0.0033424, -0.0051955, 0.0051379
8: -0.0008401, -0.0001025, -0.0007913, -0.0000154, -0.0008247, 0.0006829
9: 0.9985744, 1.0112939, 0.9980588, 1.0106895, -0.0087151, 0.0098250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055470
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055470
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0031344, -0.0004472, -0.0035663, -0.0006417, -0.0024926, 0.0031191
1: -0.0040244, 0.0026625, -0.0036368, 0.0034888, -0.0056875, 0.0042473
2: 0.0033400, 0.0093055, 0.0036427, 0.0096438, -0.0044062, 0.0037825
3: -0.0043538, -0.0034007, -0.0044627, -0.0035487, -0.0008050, 0.0010620
4: 0.0019006, 0.0074689, 0.0019428, 0.0073365, -0.0049331, 0.0049418
5: -0.0026950, 0.0026955, -0.0024720, 0.0028367, -0.0047325, 0.0044798
6: -0.0065604, -0.0033026, -0.0065583, -0.0034117, -0.0030191, 0.0031454
7: -0.0023209, 0.0033637, -0.0021179, 0.0033149, -0.0052513, 0.0051091
8: -0.0009101, -0.0001059, -0.0007888, -0.0000259, -0.0008842, 0.0006829
9: 0.9987555, 1.0113358, 0.9981597, 1.0106680, -0.0086453, 0.0097916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0056222
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0056222
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0036352, -0.0005462, -0.0027091, 0.0030876
1: -0.0039981, 0.0028021, -0.0039025, 0.0037230, -0.0052671, 0.0042485
2: 0.0033250, 0.0092548, 0.0034264, 0.0098184, -0.0042229, 0.0035080
3: -0.0043855, -0.0035314, -0.0044516, -0.0035309, -0.0008546, 0.0009202
4: 0.0020143, 0.0074683, 0.0017904, 0.0074737, -0.0048197, 0.0050033
5: -0.0026697, 0.0027175, -0.0026671, 0.0030393, -0.0047286, 0.0044326
6: -0.0065764, -0.0034190, -0.0066113, -0.0033572, -0.0030710, 0.0030521
7: -0.0020035, 0.0034380, -0.0022195, 0.0034090, -0.0049121, 0.0051524
8: -0.0008058, -0.0000645, -0.0008161, -0.0000171, -0.0007394, 0.0007015
9: 0.9987524, 1.0113355, 0.9976916, 1.0112109, -0.0083383, 0.0095779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0054663
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0055205
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0032553, -0.0005476, -0.0036399, -0.0005157, -0.0027396, 0.0030923
1: -0.0039981, 0.0028021, -0.0038971, 0.0038326, -0.0053693, 0.0042348
2: 0.0033250, 0.0092548, 0.0034569, 0.0099649, -0.0044111, 0.0035050
3: -0.0043855, -0.0035314, -0.0044556, -0.0034804, -0.0009051, 0.0009242
4: 0.0020143, 0.0074683, 0.0016332, 0.0074996, -0.0048418, 0.0051519
5: -0.0026697, 0.0027175, -0.0026923, 0.0031408, -0.0048245, 0.0044538
6: -0.0065764, -0.0034190, -0.0066328, -0.0032506, -0.0031721, 0.0030714
7: -0.0020035, 0.0034380, -0.0024508, 0.0034109, -0.0049499, 0.0054252
8: -0.0008058, -0.0000645, -0.0008521, -0.0000264, -0.0007480, 0.0007593
9: 0.9987524, 1.0113355, 0.9974241, 1.0112014, -0.0083380, 0.0098671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0054662
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0055205
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0031938, -0.0005238, -0.0036513, -0.0005354, -0.0026584, 0.0031274
1: -0.0039825, 0.0028280, -0.0039379, 0.0037365, -0.0052978, 0.0042234
2: 0.0033503, 0.0093729, 0.0033673, 0.0098268, -0.0042386, 0.0037309
3: -0.0043480, -0.0034835, -0.0044742, -0.0035267, -0.0008213, 0.0009907
4: 0.0018845, 0.0074735, 0.0017837, 0.0075654, -0.0050137, 0.0049839
5: -0.0026764, 0.0027742, -0.0027371, 0.0030463, -0.0047130, 0.0045389
6: -0.0065717, -0.0033234, -0.0066703, -0.0033546, -0.0030595, 0.0031934
7: -0.0022210, 0.0033825, -0.0022278, 0.0035223, -0.0052745, 0.0051461
8: -0.0008401, -0.0001025, -0.0008207, -0.0000067, -0.0008118, 0.0006731
9: 0.9985744, 1.0112939, 0.9976727, 1.0113544, -0.0086539, 0.0095618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050009, upper bound: 0.0055471
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050009, upper bound: 0.0055471
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0031344, -0.0004472, -0.0036104, -0.0005440, -0.0025904, 0.0031632
1: -0.0040244, 0.0026625, -0.0039282, 0.0036596, -0.0054150, 0.0041679
2: 0.0033400, 0.0093055, 0.0033724, 0.0097911, -0.0042081, 0.0037147
3: -0.0043538, -0.0034007, -0.0044637, -0.0035293, -0.0008245, 0.0010630
4: 0.0019006, 0.0074689, 0.0018074, 0.0075488, -0.0050246, 0.0049388
5: -0.0026950, 0.0026955, -0.0027223, 0.0030006, -0.0046791, 0.0045385
6: -0.0065604, -0.0033026, -0.0066531, -0.0033607, -0.0030336, 0.0032085
7: -0.0023209, 0.0033637, -0.0022195, 0.0034961, -0.0053355, 0.0051152
8: -0.0009101, -0.0001059, -0.0008183, -0.0000178, -0.0008869, 0.0006761
9: 0.9987555, 1.0113358, 0.9977679, 1.0113347, -0.0086147, 0.0094874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0056222
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0056222
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0032179, -0.0006405, -0.0029509, 0.0025752
1: -0.0036097, 0.0035535, -0.0037179, 0.0026543, -0.0040935, 0.0051160
2: 0.0037012, 0.0096718, 0.0035981, 0.0091067, -0.0033650, 0.0040746
3: -0.0044504, -0.0035500, -0.0043849, -0.0035512, -0.0008992, 0.0008349
4: 0.0019263, 0.0072613, 0.0021321, 0.0072644, -0.0048275, 0.0046498
5: -0.0024145, 0.0028740, -0.0024348, 0.0025822, -0.0042552, 0.0045411
6: -0.0065203, -0.0034083, -0.0064867, -0.0034582, -0.0029617, 0.0029772
7: -0.0021167, 0.0032308, -0.0019213, 0.0032731, -0.0050162, 0.0047809
8: -0.0007868, -0.0000255, -0.0007760, -0.0000730, -0.0006893, 0.0007268
9: 0.9980801, 1.0105479, 0.9991087, 1.0107142, -0.0091910, 0.0079655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051463, upper bound: 0.0051409
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0050579
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035935, -0.0006111, -0.0032179, -0.0006405, -0.0029530, 0.0026067
1: -0.0036105, 0.0036734, -0.0037179, 0.0026543, -0.0040811, 0.0052221
2: 0.0037211, 0.0098275, 0.0035981, 0.0091067, -0.0033635, 0.0042699
3: -0.0044549, -0.0034994, -0.0043849, -0.0035512, -0.0009037, 0.0008855
4: 0.0017662, 0.0073031, 0.0021321, 0.0072644, -0.0049770, 0.0046807
5: -0.0024555, 0.0029803, -0.0024348, 0.0025822, -0.0042835, 0.0046345
6: -0.0065462, -0.0033035, -0.0064867, -0.0034582, -0.0029846, 0.0030818
7: -0.0023565, 0.0032465, -0.0019213, 0.0032731, -0.0052939, 0.0048230
8: -0.0008217, -0.0000364, -0.0007760, -0.0000730, -0.0007484, 0.0007357
9: 0.9978067, 1.0105419, 0.9991087, 1.0107142, -0.0094876, 0.0079838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051463, upper bound: 0.0051898
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0050931
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035357, -0.0006462, -0.0032227, -0.0006033, -0.0029324, 0.0025765
1: -0.0036333, 0.0034834, -0.0037189, 0.0027595, -0.0041794, 0.0050371
2: 0.0036461, 0.0096436, 0.0036031, 0.0092730, -0.0036116, 0.0040750
3: -0.0044349, -0.0035508, -0.0043870, -0.0034972, -0.0009377, 0.0008362
4: 0.0019465, 0.0073254, 0.0019695, 0.0073184, -0.0048484, 0.0048192
5: -0.0024622, 0.0028336, -0.0024858, 0.0026886, -0.0043577, 0.0045313
6: -0.0065435, -0.0034141, -0.0065199, -0.0033547, -0.0030751, 0.0030023
7: -0.0021097, 0.0032796, -0.0021635, 0.0032949, -0.0050748, 0.0051070
8: -0.0007868, -0.0000465, -0.0008141, -0.0000828, -0.0006926, 0.0007675
9: 0.9981599, 1.0106546, 0.9988145, 1.0107462, -0.0091532, 0.0083168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052868, upper bound: 0.0048757
time: 1.07 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052868, upper bound: 0.0049788
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0034840, -0.0005637, -0.0031821, -0.0006124, -0.0028715, 0.0026184
1: -0.0036809, 0.0033280, -0.0037081, 0.0026755, -0.0042859, 0.0049695
2: 0.0036326, 0.0095784, 0.0036091, 0.0092352, -0.0035859, 0.0040426
3: -0.0044399, -0.0034729, -0.0043757, -0.0034998, -0.0009400, 0.0009028
4: 0.0019641, 0.0073213, 0.0019918, 0.0072965, -0.0048522, 0.0047751
5: -0.0024797, 0.0027574, -0.0024660, 0.0026437, -0.0043297, 0.0045183
6: -0.0065319, -0.0033934, -0.0065015, -0.0033608, -0.0030513, 0.0030134
7: -0.0022099, 0.0032643, -0.0021552, 0.0032676, -0.0051289, 0.0050830
8: -0.0008541, -0.0000488, -0.0008111, -0.0000937, -0.0007604, 0.0007623
9: 0.9983432, 1.0106931, 0.9989144, 1.0107208, -0.0090803, 0.0082653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0049802
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0050931
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0032553, -0.0005476, -0.0030438, 0.0026127
1: -0.0036097, 0.0035535, -0.0039981, 0.0028021, -0.0043514, 0.0055227
2: 0.0037012, 0.0096718, 0.0033250, 0.0092548, -0.0036191, 0.0044186
3: -0.0044504, -0.0035500, -0.0043855, -0.0035314, -0.0009190, 0.0008355
4: 0.0019263, 0.0072613, 0.0020143, 0.0074683, -0.0050093, 0.0047438
5: -0.0024145, 0.0028740, -0.0026697, 0.0027175, -0.0044092, 0.0047749
6: -0.0065203, -0.0034083, -0.0065764, -0.0034190, -0.0029947, 0.0030570
7: -0.0021167, 0.0032308, -0.0020035, 0.0034380, -0.0051441, 0.0048400
8: -0.0007868, -0.0000255, -0.0008058, -0.0000645, -0.0007121, 0.0007719
9: 0.9980801, 1.0105479, 0.9987524, 1.0113355, -0.0098481, 0.0084356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0050783
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049899
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035935, -0.0006111, -0.0032553, -0.0005476, -0.0030459, 0.0026442
1: -0.0036105, 0.0036734, -0.0039981, 0.0028021, -0.0043390, 0.0056288
2: 0.0037211, 0.0098275, 0.0033250, 0.0092548, -0.0036177, 0.0046139
3: -0.0044549, -0.0034994, -0.0043855, -0.0035314, -0.0009235, 0.0008861
4: 0.0017662, 0.0073031, 0.0020143, 0.0074683, -0.0051588, 0.0047747
5: -0.0024555, 0.0029803, -0.0026697, 0.0027175, -0.0044375, 0.0048684
6: -0.0065462, -0.0033035, -0.0065764, -0.0034190, -0.0030177, 0.0031616
7: -0.0023565, 0.0032465, -0.0020035, 0.0034380, -0.0054218, 0.0048822
8: -0.0008217, -0.0000364, -0.0008058, -0.0000645, -0.0007571, 0.0007694
9: 0.9978067, 1.0105419, 0.9987524, 1.0113355, -0.0101447, 0.0084539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0051271
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0050264
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0036072, -0.0006330, -0.0031938, -0.0005238, -0.0030833, 0.0025608
1: -0.0036472, 0.0035689, -0.0039825, 0.0028280, -0.0043113, 0.0055543
2: 0.0036373, 0.0096811, 0.0033503, 0.0093729, -0.0038107, 0.0044205
3: -0.0044740, -0.0035461, -0.0043480, -0.0034835, -0.0009905, 0.0008018
4: 0.0019193, 0.0073554, 0.0018845, 0.0074735, -0.0049854, 0.0049315
5: -0.0024889, 0.0028815, -0.0026764, 0.0027742, -0.0044941, 0.0047539
6: -0.0065769, -0.0034059, -0.0065717, -0.0033234, -0.0031341, 0.0030445
7: -0.0021268, 0.0033424, -0.0022210, 0.0033825, -0.0051379, 0.0051955
8: -0.0007913, -0.0000154, -0.0008401, -0.0001025, -0.0006829, 0.0008247
9: 0.9980588, 1.0106895, 0.9985744, 1.0112939, -0.0098250, 0.0087151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0049966
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0051271
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0035663, -0.0006417, -0.0031344, -0.0004472, -0.0031191, 0.0024926
1: -0.0036368, 0.0034888, -0.0040244, 0.0026625, -0.0042473, 0.0056875
2: 0.0036427, 0.0096438, 0.0033400, 0.0093055, -0.0037825, 0.0044062
3: -0.0044627, -0.0035487, -0.0043538, -0.0034007, -0.0010620, 0.0008050
4: 0.0019428, 0.0073365, 0.0019006, 0.0074689, -0.0049418, 0.0049331
5: -0.0024720, 0.0028367, -0.0026950, 0.0026955, -0.0044798, 0.0047325
6: -0.0065583, -0.0034117, -0.0065604, -0.0033026, -0.0031454, 0.0030191
7: -0.0021179, 0.0033149, -0.0023209, 0.0033637, -0.0051091, 0.0052513
8: -0.0007888, -0.0000259, -0.0009101, -0.0001059, -0.0006829, 0.0008842
9: 0.9981597, 1.0106680, 0.9987555, 1.0113358, -0.0097916, 0.0086453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049074
time: 1.07 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0050264
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0035914, -0.0006426, -0.0029412, 0.0029412
1: -0.0036097, 0.0035535, -0.0036097, 0.0035535, -0.0040722, 0.0040722
2: 0.0037012, 0.0096718, 0.0037012, 0.0096718, -0.0034822, 0.0034822
3: -0.0044504, -0.0035500, -0.0044504, -0.0035500, -0.0009005, 0.0009005
4: 0.0019263, 0.0072613, 0.0019263, 0.0072613, -0.0048403, 0.0048403
5: -0.0024145, 0.0028740, -0.0024145, 0.0028740, -0.0044732, 0.0044732
6: -0.0065203, -0.0034083, -0.0065203, -0.0034083, -0.0030335, 0.0030335
7: -0.0021167, 0.0032308, -0.0021167, 0.0032308, -0.0049936, 0.0049936
8: -0.0007868, -0.0000255, -0.0007868, -0.0000255, -0.0007226, 0.0007226
9: 0.9980801, 1.0105479, 0.9980801, 1.0105479, -0.0083983, 0.0083983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053915, upper bound: 0.0049297
time: 0.73 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053322, upper bound: 0.0050410
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0035935, -0.0006111, -0.0029803, 0.0029509
1: -0.0036097, 0.0035535, -0.0036105, 0.0036734, -0.0041497, 0.0040377
2: 0.0037012, 0.0096718, 0.0037211, 0.0098275, -0.0036859, 0.0035103
3: -0.0044504, -0.0035500, -0.0044549, -0.0034994, -0.0009510, 0.0009049
4: 0.0019263, 0.0072613, 0.0017662, 0.0073031, -0.0048744, 0.0049945
5: -0.0024145, 0.0028740, -0.0024555, 0.0029803, -0.0045748, 0.0045058
6: -0.0065203, -0.0034083, -0.0065462, -0.0033035, -0.0031377, 0.0030593
7: -0.0021167, 0.0032308, -0.0023565, 0.0032465, -0.0050375, 0.0052742
8: -0.0007868, -0.0000255, -0.0008217, -0.0000364, -0.0007325, 0.0007819
9: 0.9980801, 1.0105479, 0.9978067, 1.0105419, -0.0084333, 0.0086972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053915, upper bound: 0.0049297
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053322, upper bound: 0.0050410
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0035907, -0.0006116, -0.0035357, -0.0006462, -0.0029445, 0.0029241
1: -0.0036099, 0.0036701, -0.0036333, 0.0034834, -0.0039928, 0.0041431
2: 0.0037215, 0.0098261, 0.0036461, 0.0096436, -0.0034879, 0.0037346
3: -0.0044534, -0.0034995, -0.0044349, -0.0035508, -0.0009026, 0.0009354
4: 0.0017672, 0.0073020, 0.0019465, 0.0073254, -0.0050061, 0.0048468
5: -0.0024545, 0.0029785, -0.0024622, 0.0028336, -0.0044602, 0.0045718
6: -0.0065449, -0.0033039, -0.0065435, -0.0034141, -0.0030489, 0.0031475
7: -0.0023559, 0.0032441, -0.0021097, 0.0032796, -0.0053140, 0.0050349
8: -0.0008215, -0.0000376, -0.0007868, -0.0000465, -0.0007670, 0.0007240
9: 0.9978105, 1.0105405, 0.9981599, 1.0106546, -0.0087373, 0.0083675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051780, upper bound: 0.0052324
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051780, upper bound: 0.0052324
time: 0.95 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035498, -0.0006206, -0.0034840, -0.0005637, -0.0029861, 0.0028633
1: -0.0035994, 0.0035882, -0.0036809, 0.0033280, -0.0039470, 0.0042700
2: 0.0037270, 0.0097894, 0.0036326, 0.0095784, -0.0034721, 0.0036896
3: -0.0044421, -0.0035020, -0.0044399, -0.0034729, -0.0009691, 0.0009378
4: 0.0017907, 0.0072824, 0.0019641, 0.0073213, -0.0049592, 0.0048589
5: -0.0024374, 0.0029334, -0.0024797, 0.0027574, -0.0044601, 0.0045304
6: -0.0065259, -0.0033097, -0.0065319, -0.0033934, -0.0030614, 0.0031235
7: -0.0023472, 0.0032158, -0.0022099, 0.0032643, -0.0052910, 0.0050963
8: -0.0008189, -0.0000481, -0.0008541, -0.0000488, -0.0007674, 0.0007965
9: 0.9979107, 1.0105183, 0.9983432, 1.0106931, -0.0086467, 0.0083336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052542, upper bound: 0.0051614
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052542, upper bound: 0.0051614
time: 0.77 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0036352, -0.0005462, -0.0029909, 0.0029419
1: -0.0036097, 0.0035535, -0.0039025, 0.0037230, -0.0043379, 0.0044831
2: 0.0037012, 0.0096718, 0.0034264, 0.0098184, -0.0037570, 0.0038194
3: -0.0044504, -0.0035500, -0.0044516, -0.0035309, -0.0009195, 0.0009016
4: 0.0019263, 0.0072613, 0.0017904, 0.0074737, -0.0050161, 0.0049414
5: -0.0024145, 0.0028740, -0.0026671, 0.0030393, -0.0046354, 0.0046986
6: -0.0065203, -0.0034083, -0.0066113, -0.0033572, -0.0030701, 0.0031104
7: -0.0021167, 0.0032308, -0.0022195, 0.0034090, -0.0051170, 0.0050584
8: -0.0007868, -0.0000255, -0.0008161, -0.0000171, -0.0007472, 0.0007643
9: 0.9980801, 1.0105479, 0.9976916, 1.0112109, -0.0090419, 0.0089074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055393, upper bound: 0.0050370
time: 0.88 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056152, upper bound: 0.0049634
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0035914, -0.0006426, -0.0036399, -0.0005157, -0.0030474, 0.0029687
1: -0.0036097, 0.0035535, -0.0038971, 0.0038326, -0.0044001, 0.0044425
2: 0.0037012, 0.0096718, 0.0034569, 0.0099649, -0.0039275, 0.0038089
3: -0.0044504, -0.0035500, -0.0044556, -0.0034804, -0.0009700, 0.0009056
4: 0.0019263, 0.0072613, 0.0016332, 0.0074996, -0.0050385, 0.0050856
5: -0.0024145, 0.0028740, -0.0026923, 0.0031408, -0.0047269, 0.0047172
6: -0.0065203, -0.0034083, -0.0066328, -0.0032506, -0.0031707, 0.0031297
7: -0.0021167, 0.0032308, -0.0024508, 0.0034109, -0.0051492, 0.0053240
8: -0.0007868, -0.0000255, -0.0008521, -0.0000264, -0.0007546, 0.0008203
9: 0.9980801, 1.0105479, 0.9974241, 1.0112014, -0.0090401, 0.0091720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055393, upper bound: 0.0050370
time: 1.02 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056152, upper bound: 0.0049634
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0035907, -0.0006116, -0.0035783, -0.0005491, -0.0030067, 0.0029660
1: -0.0036099, 0.0036701, -0.0039241, 0.0036458, -0.0042521, 0.0045538
2: 0.0037215, 0.0098261, 0.0033760, 0.0097879, -0.0037491, 0.0040590
3: -0.0044534, -0.0034995, -0.0044351, -0.0035314, -0.0009219, 0.0009356
4: 0.0017672, 0.0073020, 0.0018111, 0.0075371, -0.0051832, 0.0049424
5: -0.0024545, 0.0029785, -0.0027121, 0.0029982, -0.0046098, 0.0047971
6: -0.0065449, -0.0033039, -0.0066377, -0.0033639, -0.0030841, 0.0032265
7: -0.0023559, 0.0032441, -0.0022116, 0.0034612, -0.0054388, 0.0050972
8: -0.0008215, -0.0000376, -0.0008162, -0.0000372, -0.0007843, 0.0007670
9: 0.9978105, 1.0105405, 0.9977753, 1.0113204, -0.0093773, 0.0088466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054914, upper bound: 0.0051673
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054914, upper bound: 0.0051673
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035498, -0.0006206, -0.0035275, -0.0004623, -0.0030874, 0.0029069
1: -0.0035994, 0.0035882, -0.0039799, 0.0035038, -0.0042115, 0.0046866
2: 0.0037270, 0.0097894, 0.0033539, 0.0097286, -0.0037270, 0.0040403
3: -0.0044421, -0.0035020, -0.0044413, -0.0034482, -0.0009939, 0.0009393
4: 0.0017907, 0.0072824, 0.0018290, 0.0075414, -0.0051429, 0.0049479
5: -0.0024374, 0.0029334, -0.0027384, 0.0029209, -0.0045983, 0.0047761
6: -0.0065259, -0.0033097, -0.0066312, -0.0033443, -0.0030950, 0.0032047
7: -0.0023472, 0.0032158, -0.0023068, 0.0034495, -0.0054159, 0.0051543
8: -0.0008189, -0.0000481, -0.0008839, -0.0000401, -0.0007787, 0.0008358
9: 0.9979107, 1.0105183, 0.9979419, 1.0113884, -0.0093400, 0.0087967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055469, upper bound: 0.0050849
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055469, upper bound: 0.0050849
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036352, -0.0005462, -0.0032179, -0.0006405, -0.0029947, 0.0026717
1: -0.0039025, 0.0037230, -0.0037179, 0.0026543, -0.0043966, 0.0052591
2: 0.0034264, 0.0098184, 0.0035981, 0.0091067, -0.0036647, 0.0042814
3: -0.0044516, -0.0035309, -0.0043849, -0.0035512, -0.0009003, 0.0008540
4: 0.0017904, 0.0074737, 0.0021321, 0.0072644, -0.0049482, 0.0048419
5: -0.0026671, 0.0030393, -0.0024348, 0.0025822, -0.0045020, 0.0047160
6: -0.0066113, -0.0033572, -0.0064867, -0.0034582, -0.0030447, 0.0030202
7: -0.0022195, 0.0034090, -0.0019213, 0.0032731, -0.0050933, 0.0049121
8: -0.0008161, -0.0000171, -0.0007760, -0.0000730, -0.0007337, 0.0007531
9: 0.9976916, 1.0112109, 0.9991087, 1.0107142, -0.0096785, 0.0086399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050335, upper bound: 0.0053931
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0053165
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036399, -0.0005157, -0.0032179, -0.0006405, -0.0029994, 0.0027021
1: -0.0038971, 0.0038326, -0.0037179, 0.0026543, -0.0043693, 0.0053527
2: 0.0034569, 0.0099649, 0.0035981, 0.0091067, -0.0036419, 0.0044467
3: -0.0044556, -0.0034804, -0.0043849, -0.0035512, -0.0009043, 0.0009045
4: 0.0016332, 0.0074996, 0.0021321, 0.0072644, -0.0050937, 0.0048599
5: -0.0026923, 0.0031408, -0.0024348, 0.0025822, -0.0045167, 0.0048103
6: -0.0066328, -0.0032506, -0.0064867, -0.0034582, -0.0030626, 0.0031222
7: -0.0024508, 0.0034109, -0.0019213, 0.0032731, -0.0053598, 0.0049488
8: -0.0008521, -0.0000264, -0.0007760, -0.0000730, -0.0007790, 0.0007496
9: 0.9974241, 1.0112014, 0.9991087, 1.0107142, -0.0099502, 0.0086138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050335, upper bound: 0.0054319
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0053467
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035783, -0.0005491, -0.0032227, -0.0006033, -0.0029750, 0.0026736
1: -0.0039241, 0.0036458, -0.0037189, 0.0027595, -0.0044783, 0.0051703
2: 0.0033760, 0.0097879, 0.0036031, 0.0092730, -0.0039041, 0.0042758
3: -0.0044351, -0.0035314, -0.0043870, -0.0034972, -0.0009379, 0.0008555
4: 0.0018111, 0.0075371, 0.0019695, 0.0073184, -0.0049627, 0.0050110
5: -0.0027121, 0.0029982, -0.0024858, 0.0026886, -0.0046073, 0.0047007
6: -0.0066377, -0.0033639, -0.0065199, -0.0033547, -0.0031596, 0.0030441
7: -0.0022116, 0.0034612, -0.0021635, 0.0032949, -0.0051490, 0.0052409
8: -0.0008162, -0.0000372, -0.0008141, -0.0000828, -0.0007334, 0.0007768
9: 0.9977753, 1.0113204, 0.9988145, 1.0107462, -0.0096313, 0.0089803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051595, upper bound: 0.0051755
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051595, upper bound: 0.0052559
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035275, -0.0004623, -0.0031821, -0.0006124, -0.0029151, 0.0027197
1: -0.0039799, 0.0035038, -0.0037081, 0.0026755, -0.0046084, 0.0051102
2: 0.0033539, 0.0097286, 0.0036091, 0.0092352, -0.0038931, 0.0042415
3: -0.0044413, -0.0034482, -0.0043757, -0.0034998, -0.0009415, 0.0009276
4: 0.0018290, 0.0075414, 0.0019918, 0.0072965, -0.0049584, 0.0049753
5: -0.0027384, 0.0029209, -0.0024660, 0.0026437, -0.0045929, 0.0046807
6: -0.0066312, -0.0033443, -0.0065015, -0.0033608, -0.0031383, 0.0030539
7: -0.0023068, 0.0034495, -0.0021552, 0.0032676, -0.0051930, 0.0052213
8: -0.0008839, -0.0000401, -0.0008111, -0.0000937, -0.0007902, 0.0007710
9: 0.9979419, 1.0113884, 0.9989144, 1.0107208, -0.0095518, 0.0089602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0052556
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0053467
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036352, -0.0005462, -0.0032553, -0.0005476, -0.0030876, 0.0027091
1: -0.0039025, 0.0037230, -0.0039981, 0.0028021, -0.0042485, 0.0052671
2: 0.0034264, 0.0098184, 0.0033250, 0.0092548, -0.0035080, 0.0042229
3: -0.0044516, -0.0035309, -0.0043855, -0.0035314, -0.0009202, 0.0008546
4: 0.0017904, 0.0074737, 0.0020143, 0.0074683, -0.0050033, 0.0048197
5: -0.0026671, 0.0030393, -0.0026697, 0.0027175, -0.0044326, 0.0047286
6: -0.0066113, -0.0033572, -0.0065764, -0.0034190, -0.0030521, 0.0030710
7: -0.0022195, 0.0034090, -0.0020035, 0.0034380, -0.0051524, 0.0049121
8: -0.0008161, -0.0000171, -0.0008058, -0.0000645, -0.0007015, 0.0007394
9: 0.9976916, 1.0112109, 0.9987524, 1.0113355, -0.0095779, 0.0083383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0053931
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0053165
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036399, -0.0005157, -0.0032553, -0.0005476, -0.0030923, 0.0027396
1: -0.0038971, 0.0038326, -0.0039981, 0.0028021, -0.0042348, 0.0053693
2: 0.0034569, 0.0099649, 0.0033250, 0.0092548, -0.0035050, 0.0044111
3: -0.0044556, -0.0034804, -0.0043855, -0.0035314, -0.0009242, 0.0009051
4: 0.0016332, 0.0074996, 0.0020143, 0.0074683, -0.0051519, 0.0048418
5: -0.0026923, 0.0031408, -0.0026697, 0.0027175, -0.0044538, 0.0048245
6: -0.0066328, -0.0032506, -0.0065764, -0.0034190, -0.0030714, 0.0031721
7: -0.0024508, 0.0034109, -0.0020035, 0.0034380, -0.0054252, 0.0049499
8: -0.0008521, -0.0000264, -0.0008058, -0.0000645, -0.0007593, 0.0007480
9: 0.9974241, 1.0112014, 0.9987524, 1.0113355, -0.0098671, 0.0083380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0054319
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0053467
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0036513, -0.0005354, -0.0031938, -0.0005238, -0.0031274, 0.0026584
1: -0.0039379, 0.0037365, -0.0039825, 0.0028280, -0.0042234, 0.0052978
2: 0.0033673, 0.0098268, 0.0033503, 0.0093729, -0.0037309, 0.0042386
3: -0.0044742, -0.0035267, -0.0043480, -0.0034835, -0.0009907, 0.0008213
4: 0.0017837, 0.0075654, 0.0018845, 0.0074735, -0.0049839, 0.0050137
5: -0.0027371, 0.0030463, -0.0026764, 0.0027742, -0.0045389, 0.0047130
6: -0.0066703, -0.0033546, -0.0065717, -0.0033234, -0.0031934, 0.0030595
7: -0.0022278, 0.0035223, -0.0022210, 0.0033825, -0.0051461, 0.0052745
8: -0.0008207, -0.0000067, -0.0008401, -0.0001025, -0.0006731, 0.0008118
9: 0.9976727, 1.0113544, 0.9985744, 1.0112939, -0.0095618, 0.0086539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0053271
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0054319
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0036104, -0.0005440, -0.0031344, -0.0004472, -0.0031632, 0.0025904
1: -0.0039282, 0.0036596, -0.0040244, 0.0026625, -0.0041679, 0.0054150
2: 0.0033724, 0.0097911, 0.0033400, 0.0093055, -0.0037147, 0.0042081
3: -0.0044637, -0.0035293, -0.0043538, -0.0034007, -0.0010630, 0.0008245
4: 0.0018074, 0.0075488, 0.0019006, 0.0074689, -0.0049388, 0.0050246
5: -0.0027223, 0.0030006, -0.0026950, 0.0026955, -0.0045385, 0.0046791
6: -0.0066531, -0.0033607, -0.0065604, -0.0033026, -0.0032085, 0.0030336
7: -0.0022195, 0.0034961, -0.0023209, 0.0033637, -0.0051152, 0.0053355
8: -0.0008183, -0.0000178, -0.0009101, -0.0001059, -0.0006761, 0.0008869
9: 0.9977679, 1.0113347, 0.9987555, 1.0113358, -0.0094874, 0.0086147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0052556
time: 0.77 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0053467
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036352, -0.0005462, -0.0035914, -0.0006426, -0.0029419, 0.0029909
1: -0.0039025, 0.0037230, -0.0036097, 0.0035535, -0.0044831, 0.0043379
2: 0.0034264, 0.0098184, 0.0037012, 0.0096718, -0.0038194, 0.0037570
3: -0.0044516, -0.0035309, -0.0044504, -0.0035500, -0.0009016, 0.0009195
4: 0.0017904, 0.0074737, 0.0019263, 0.0072613, -0.0049414, 0.0050161
5: -0.0026671, 0.0030393, -0.0024145, 0.0028740, -0.0046986, 0.0046354
6: -0.0066113, -0.0033572, -0.0065203, -0.0034083, -0.0031104, 0.0030701
7: -0.0022195, 0.0034090, -0.0021167, 0.0032308, -0.0050584, 0.0051170
8: -0.0008161, -0.0000171, -0.0007868, -0.0000255, -0.0007643, 0.0007472
9: 0.9976916, 1.0112109, 0.9980801, 1.0105479, -0.0089074, 0.0090419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0052673
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053666
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036399, -0.0005157, -0.0035914, -0.0006426, -0.0029687, 0.0030474
1: -0.0038971, 0.0038326, -0.0036097, 0.0035535, -0.0044425, 0.0044001
2: 0.0034569, 0.0099649, 0.0037012, 0.0096718, -0.0038089, 0.0039275
3: -0.0044556, -0.0034804, -0.0044504, -0.0035500, -0.0009056, 0.0009700
4: 0.0016332, 0.0074996, 0.0019263, 0.0072613, -0.0050856, 0.0050385
5: -0.0026923, 0.0031408, -0.0024145, 0.0028740, -0.0047172, 0.0047269
6: -0.0066328, -0.0032506, -0.0065203, -0.0034083, -0.0031297, 0.0031707
7: -0.0024508, 0.0034109, -0.0021167, 0.0032308, -0.0053240, 0.0051492
8: -0.0008521, -0.0000264, -0.0007868, -0.0000255, -0.0008203, 0.0007546
9: 0.9974241, 1.0112014, 0.9980801, 1.0105479, -0.0091720, 0.0090401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0053001
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053974
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035783, -0.0005491, -0.0035907, -0.0006116, -0.0029660, 0.0030067
1: -0.0039241, 0.0036458, -0.0036099, 0.0036701, -0.0045538, 0.0042521
2: 0.0033760, 0.0097879, 0.0037215, 0.0098261, -0.0040590, 0.0037491
3: -0.0044351, -0.0035314, -0.0044534, -0.0034995, -0.0009356, 0.0009219
4: 0.0018111, 0.0075371, 0.0017672, 0.0073020, -0.0049424, 0.0051831
5: -0.0027121, 0.0029982, -0.0024545, 0.0029785, -0.0047971, 0.0046098
6: -0.0066377, -0.0033639, -0.0065449, -0.0033039, -0.0032265, 0.0030841
7: -0.0022116, 0.0034612, -0.0023559, 0.0032441, -0.0050972, 0.0054388
8: -0.0008162, -0.0000372, -0.0008215, -0.0000376, -0.0007670, 0.0007843
9: 0.9977753, 1.0113204, 0.9978105, 1.0105405, -0.0088466, 0.0093773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0052204
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0053001
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035275, -0.0004623, -0.0035498, -0.0006206, -0.0029069, 0.0030874
1: -0.0039799, 0.0035038, -0.0035994, 0.0035882, -0.0046866, 0.0042115
2: 0.0033539, 0.0097286, 0.0037270, 0.0097894, -0.0040403, 0.0037270
3: -0.0044413, -0.0034482, -0.0044421, -0.0035020, -0.0009393, 0.0009939
4: 0.0018290, 0.0075414, 0.0017907, 0.0072824, -0.0049479, 0.0051429
5: -0.0027384, 0.0029209, -0.0024374, 0.0029334, -0.0047761, 0.0045983
6: -0.0066312, -0.0033443, -0.0065259, -0.0033097, -0.0032047, 0.0030950
7: -0.0023068, 0.0034495, -0.0023472, 0.0032158, -0.0051543, 0.0054159
8: -0.0008839, -0.0000401, -0.0008189, -0.0000481, -0.0008358, 0.0007787
9: 0.9979419, 1.0113884, 0.9979107, 1.0105183, -0.0087967, 0.0093400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053037
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053974
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036352, -0.0005462, -0.0036352, -0.0005462, -0.0030525, 0.0030525
1: -0.0039025, 0.0037230, -0.0039025, 0.0037230, -0.0042400, 0.0042400
2: 0.0034264, 0.0098184, 0.0034264, 0.0098184, -0.0036410, 0.0036410
3: -0.0044516, -0.0035309, -0.0044516, -0.0035309, -0.0009206, 0.0009206
4: 0.0017904, 0.0074737, 0.0017904, 0.0074737, -0.0050061, 0.0050061
5: -0.0026671, 0.0030393, -0.0026671, 0.0030393, -0.0046524, 0.0046524
6: -0.0066113, -0.0033572, -0.0066113, -0.0033572, -0.0031219, 0.0031219
7: -0.0022195, 0.0034090, -0.0022195, 0.0034090, -0.0051233, 0.0051233
8: -0.0008161, -0.0000171, -0.0008161, -0.0000171, -0.0007351, 0.0007351
9: 0.9976916, 1.0112109, 0.9976916, 1.0112109, -0.0087879, 0.0087879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052859, upper bound: 0.0052204
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052158, upper bound: 0.0053037
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036352, -0.0005462, -0.0036399, -0.0005157, -0.0031044, 0.0030727
1: -0.0039025, 0.0037230, -0.0038971, 0.0038326, -0.0043132, 0.0042087
2: 0.0034264, 0.0098184, 0.0034569, 0.0099649, -0.0038367, 0.0036626
3: -0.0044516, -0.0035309, -0.0044556, -0.0034804, -0.0009712, 0.0009246
4: 0.0017904, 0.0074737, 0.0016332, 0.0074996, -0.0050339, 0.0051573
5: -0.0026671, 0.0030393, -0.0026923, 0.0031408, -0.0047534, 0.0046779
6: -0.0066113, -0.0033572, -0.0066328, -0.0032506, -0.0032228, 0.0031432
7: -0.0022195, 0.0034090, -0.0024508, 0.0034109, -0.0051598, 0.0053912
8: -0.0008161, -0.0000171, -0.0008521, -0.0000264, -0.0007446, 0.0007932
9: 0.9976916, 1.0112109, 0.9974241, 1.0112014, -0.0088119, 0.0090836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052859, upper bound: 0.0052204
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052158, upper bound: 0.0053037
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036371, -0.0005163, -0.0035783, -0.0005491, -0.0030618, 0.0030620
1: -0.0038966, 0.0038290, -0.0039241, 0.0036458, -0.0041575, 0.0043078
2: 0.0034572, 0.0099634, 0.0033760, 0.0097879, -0.0036352, 0.0038839
3: -0.0044540, -0.0034806, -0.0044351, -0.0035314, -0.0009226, 0.0009545
4: 0.0016343, 0.0074986, 0.0018111, 0.0075371, -0.0051720, 0.0050038
5: -0.0026914, 0.0031390, -0.0027121, 0.0029982, -0.0046253, 0.0047526
6: -0.0066315, -0.0032509, -0.0066377, -0.0033639, -0.0031317, 0.0032349
7: -0.0024502, 0.0034086, -0.0022116, 0.0034612, -0.0054348, 0.0051583
8: -0.0008519, -0.0000276, -0.0008162, -0.0000372, -0.0007783, 0.0007357
9: 0.9974282, 1.0112001, 0.9977753, 1.0113204, -0.0091235, 0.0087294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050607, upper bound: 0.0054715
time: 0.83 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050607, upper bound: 0.0054715
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035955, -0.0005252, -0.0035275, -0.0004623, -0.0031332, 0.0030024
1: -0.0038867, 0.0037506, -0.0039799, 0.0035038, -0.0041086, 0.0044309
2: 0.0034624, 0.0099274, 0.0033539, 0.0097286, -0.0036193, 0.0038516
3: -0.0044430, -0.0034831, -0.0044413, -0.0034482, -0.0009949, 0.0009582
4: 0.0016575, 0.0074815, 0.0018290, 0.0075414, -0.0051332, 0.0050154
5: -0.0026761, 0.0030931, -0.0027384, 0.0029209, -0.0046244, 0.0047214
6: -0.0066138, -0.0032568, -0.0066312, -0.0033443, -0.0031448, 0.0032137
7: -0.0024420, 0.0033821, -0.0023068, 0.0034495, -0.0054104, 0.0052168
8: -0.0008494, -0.0000386, -0.0008839, -0.0000401, -0.0007794, 0.0008089
9: 0.9975232, 1.0111798, 0.9979419, 1.0113884, -0.0090477, 0.0086906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051216, upper bound: 0.0053974
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051216, upper bound: 0.0053974
time: 0.84 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.10 seconds
IS_A1_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0048787
IS_A1_A1_B1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0049802
IS_A1_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0048787
IS_A1_A1_B1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0049802
IS_A1_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
IS_A1_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
IS_A1_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0050931
IS_A1_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0050931
IS_A1_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052288, upper bound: 0.0049966
IS_A1_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053186, upper bound: 0.0049074
IS_A1_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052288, upper bound: 0.0049966
IS_A1_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053186, upper bound: 0.0049074
IS_A1_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
IS_A1_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
IS_A1_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052571, upper bound: 0.0050264
IS_A1_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052571, upper bound: 0.0050264
IS_A1_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0051463
IS_A1_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0052205
IS_A1_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051409, upper bound: 0.0051463
IS_A1_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050579, upper bound: 0.0052205
IS_A1_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
IS_A1_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
IS_A1_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0053318
IS_A1_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049802, upper bound: 0.0053318
IS_A1_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053931, upper bound: 0.0050335
IS_A1_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053165, upper bound: 0.0050927
IS_A1_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053931, upper bound: 0.0050335
IS_A1_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053165, upper bound: 0.0050927
IS_A1_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
IS_A1_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
IS_A1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052556, upper bound: 0.0052217
IS_A1_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052556, upper bound: 0.0052217
IS_A1_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052288
IS_A1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0053186
IS_A1_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052613
IS_A1_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0053494
IS_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0051853
IS_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052613
IS_A1_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0052571
IS_A1_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0053494
IS_A1_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0051853
IS_A1_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0052571
IS_A1_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0051853
IS_A1_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0052571
IS_A1_A2_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
IS_A1_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
IS_A1_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0053494
IS_A1_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0053494
IS_A1_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050783, upper bound: 0.0054663
IS_A1_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049899, upper bound: 0.0055205
IS_A1_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050783, upper bound: 0.0054662
IS_A1_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049899, upper bound: 0.0055205
IS_A1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055470
IS_A1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055470
IS_A1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0056222
IS_A1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049074, upper bound: 0.0056222
IS_A1_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0054663
IS_A1_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0055205
IS_A1_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050829, upper bound: 0.0054662
IS_A1_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049953, upper bound: 0.0055205
IS_A1_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050009, upper bound: 0.0055471
IS_A1_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050009, upper bound: 0.0055471
IS_A1_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0056222
IS_A1_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0049136, upper bound: 0.0056222
IS_A2_A1_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051463, upper bound: 0.0051409
IS_A2_A1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0050579
IS_A2_A1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051463, upper bound: 0.0051898
IS_A2_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0050931
IS_A2_A1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052868, upper bound: 0.0048757
IS_A2_A1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052868, upper bound: 0.0049788
IS_A2_A1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0049802
IS_A2_A1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052205, upper bound: 0.0050931
IS_A2_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0050783
IS_A2_A1_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049899
IS_A2_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0051271
IS_A2_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0050264
IS_A2_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0049966
IS_A2_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0054663, upper bound: 0.0051271
IS_A2_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049074
IS_A2_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0050264
IS_A2_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053915, upper bound: 0.0049297
IS_A2_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053322, upper bound: 0.0050410
IS_A2_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053915, upper bound: 0.0049297
IS_A2_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0053322, upper bound: 0.0050410
IS_A2_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051780, upper bound: 0.0052324
IS_A2_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051780, upper bound: 0.0052324
IS_A2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052542, upper bound: 0.0051614
IS_A2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052542, upper bound: 0.0051614
IS_A2_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055393, upper bound: 0.0050370
IS_A2_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0056152, upper bound: 0.0049634
IS_A2_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055393, upper bound: 0.0050370
IS_A2_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0056152, upper bound: 0.0049634
IS_A2_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0054914, upper bound: 0.0051673
IS_A2_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0054914, upper bound: 0.0051673
IS_A2_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055469, upper bound: 0.0050849
IS_A2_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0055469, upper bound: 0.0050849
IS_A2_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050335, upper bound: 0.0053931
IS_A2_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0053165
IS_A2_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050335, upper bound: 0.0054319
IS_A2_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0053467
IS_A2_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051595, upper bound: 0.0051755
IS_A2_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051595, upper bound: 0.0052559
IS_A2_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0052556
IS_A2_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050927, upper bound: 0.0053467
IS_A2_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0053931
IS_A2_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0053165
IS_A2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0054319
IS_A2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0053467
IS_A2_A2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0053271
IS_A2_A2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050375, upper bound: 0.0054319
IS_A2_A2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0052556
IS_A2_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050969, upper bound: 0.0053467
IS_A2_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0052673
IS_A2_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053666
IS_A2_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0053001
IS_A2_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053974
IS_A2_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0052204
IS_A2_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051857, upper bound: 0.0053001
IS_A2_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053037
IS_A2_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051211, upper bound: 0.0053974
IS_A2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052859, upper bound: 0.0052204
IS_A2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052158, upper bound: 0.0053037
IS_A2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052859, upper bound: 0.0052204
IS_A2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0052158, upper bound: 0.0053037
IS_A2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050607, upper bound: 0.0054715
IS_A2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0050607, upper bound: 0.0054715
IS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051216, upper bound: 0.0053974
IS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0051216, upper bound: 0.0053974

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0031516, -0.0006549, -0.0032153, -0.0006410, -0.0025106, 0.0025604
1: -0.0037027, 0.0025778, -0.0037173, 0.0026514, -0.0037584, 0.0036465
2: 0.0036079, 0.0090724, 0.0035985, 0.0091054, -0.0032327, 0.0032216
3: -0.0043465, -0.0035561, -0.0043834, -0.0035514, -0.0007950, 0.0008273
4: 0.0021563, 0.0072293, 0.0021330, 0.0072631, -0.0046124, 0.0045821
5: -0.0024031, 0.0025368, -0.0024336, 0.0025805, -0.0041989, 0.0042042
6: -0.0064536, -0.0034668, -0.0064854, -0.0034585, -0.0028839, 0.0029125
7: -0.0019066, 0.0032082, -0.0019207, 0.0032706, -0.0047615, 0.0047097
8: -0.0007708, -0.0001046, -0.0007758, -0.0000742, -0.0006679, 0.0006366
9: 0.9992027, 1.0106726, 0.9991122, 1.0107126, -0.0078259, 0.0078407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049430, upper bound: 0.0049430
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049430, upper bound: 0.0049430
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0031516, -0.0006549, -0.0032227, -0.0006033, -0.0025483, 0.0025678
1: -0.0037027, 0.0025778, -0.0037189, 0.0027595, -0.0038364, 0.0036113
2: 0.0036079, 0.0090724, 0.0036031, 0.0092730, -0.0034353, 0.0032572
3: -0.0043465, -0.0035561, -0.0043870, -0.0034972, -0.0008493, 0.0008308
4: 0.0021563, 0.0072293, 0.0019695, 0.0073184, -0.0046621, 0.0047365
5: -0.0024031, 0.0025368, -0.0024858, 0.0026886, -0.0043011, 0.0042482
6: -0.0064536, -0.0034668, -0.0065199, -0.0033547, -0.0029877, 0.0029484
7: -0.0019066, 0.0032082, -0.0021635, 0.0032949, -0.0048209, 0.0049899
8: -0.0007708, -0.0001046, -0.0008141, -0.0000828, -0.0006789, 0.0006968
9: 0.9992027, 1.0106726, 0.9988145, 1.0107462, -0.0078782, 0.0081383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0048787
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0048787
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0031516, -0.0006549, -0.0025678, 0.0025483
1: -0.0037189, 0.0027595, -0.0037027, 0.0025778, -0.0036113, 0.0038364
2: 0.0036031, 0.0092730, 0.0036079, 0.0090724, -0.0032572, 0.0034353
3: -0.0043870, -0.0034972, -0.0043465, -0.0035561, -0.0008308, 0.0008493
4: 0.0019695, 0.0073184, 0.0021563, 0.0072293, -0.0047365, 0.0046621
5: -0.0024858, 0.0026886, -0.0024031, 0.0025368, -0.0042482, 0.0043011
6: -0.0065199, -0.0033547, -0.0064536, -0.0034668, -0.0029484, 0.0029877
7: -0.0021635, 0.0032949, -0.0019066, 0.0032082, -0.0049899, 0.0048209
8: -0.0008141, -0.0000828, -0.0007708, -0.0001046, -0.0006968, 0.0006789
9: 0.9988145, 1.0107462, 0.9992027, 1.0106726, -0.0081383, 0.0078782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0031574, -0.0006167, -0.0026060, 0.0025541
1: -0.0037189, 0.0027595, -0.0037048, 0.0026876, -0.0037192, 0.0038316
2: 0.0036031, 0.0092730, 0.0036127, 0.0092400, -0.0033859, 0.0033955
3: -0.0043870, -0.0034972, -0.0043484, -0.0035017, -0.0008853, 0.0008512
4: 0.0019695, 0.0073184, 0.0019931, 0.0072862, -0.0047768, 0.0048058
5: -0.0024858, 0.0026886, -0.0024568, 0.0026441, -0.0043469, 0.0043418
6: -0.0065199, -0.0033547, -0.0064888, -0.0033631, -0.0030475, 0.0030199
7: -0.0021635, 0.0032949, -0.0021489, 0.0032328, -0.0049980, 0.0050495
8: -0.0008141, -0.0000828, -0.0008092, -0.0001137, -0.0006880, 0.0007190
9: 0.9988145, 1.0107462, 0.9989046, 1.0107074, -0.0081278, 0.0081105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0051898
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0030926, -0.0005829, -0.0025992, 0.0024801
1: -0.0037081, 0.0026755, -0.0037263, 0.0023993, -0.0035480, 0.0039654
2: 0.0036091, 0.0092352, 0.0036073, 0.0089986, -0.0032377, 0.0033890
3: -0.0043757, -0.0034998, -0.0043458, -0.0034793, -0.0008964, 0.0008460
4: 0.0019918, 0.0072965, 0.0021749, 0.0072176, -0.0046823, 0.0046677
5: -0.0024660, 0.0026437, -0.0024134, 0.0024513, -0.0042437, 0.0042555
6: -0.0065015, -0.0033608, -0.0064386, -0.0034454, -0.0029596, 0.0029597
7: -0.0021552, 0.0032676, -0.0020118, 0.0031911, -0.0049585, 0.0048864
8: -0.0008111, -0.0000937, -0.0008388, -0.0001081, -0.0006974, 0.0007451
9: 0.9989144, 1.0107208, 0.9993969, 1.0106951, -0.0080373, 0.0078235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0049788
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0050931
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0031016, -0.0005414, -0.0026406, 0.0024891
1: -0.0037081, 0.0026755, -0.0037270, 0.0025053, -0.0036556, 0.0039632
2: 0.0036091, 0.0092352, 0.0036104, 0.0091689, -0.0033645, 0.0033470
3: -0.0043757, -0.0034998, -0.0043534, -0.0034222, -0.0009535, 0.0008536
4: 0.0019918, 0.0072965, 0.0020114, 0.0072735, -0.0047243, 0.0048138
5: -0.0024660, 0.0026437, -0.0024662, 0.0025630, -0.0043447, 0.0042964
6: -0.0065015, -0.0033608, -0.0064754, -0.0033417, -0.0030604, 0.0029930
7: -0.0021552, 0.0032676, -0.0022497, 0.0032151, -0.0049703, 0.0051145
8: -0.0008111, -0.0000937, -0.0008789, -0.0001168, -0.0006900, 0.0007852
9: 0.9989144, 1.0107208, 0.9990979, 1.0107336, -0.0080293, 0.0080572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0049788
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048787, upper bound: 0.0050931
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032153, -0.0006410, -0.0031873, -0.0005622, -0.0026531, 0.0025463
1: -0.0037173, 0.0026514, -0.0039831, 0.0027186, -0.0038985, 0.0041634
2: 0.0035985, 0.0091054, 0.0033344, 0.0092203, -0.0034613, 0.0035805
3: -0.0043834, -0.0035514, -0.0043465, -0.0035363, -0.0008471, 0.0007951
4: 0.0021330, 0.0072631, 0.0020407, 0.0074359, -0.0047676, 0.0047013
5: -0.0024336, 0.0025805, -0.0026415, 0.0026708, -0.0043452, 0.0044373
6: -0.0064854, -0.0034585, -0.0065449, -0.0034284, -0.0029445, 0.0029658
7: -0.0019207, 0.0032706, -0.0019880, 0.0033761, -0.0048412, 0.0048186
8: -0.0007758, -0.0000742, -0.0008007, -0.0000957, -0.0006597, 0.0007136
9: 0.9991122, 1.0107126, 0.9988492, 1.0112972, -0.0085074, 0.0082634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052324, upper bound: 0.0048807
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052324, upper bound: 0.0049947
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0031743, -0.0006505, -0.0031283, -0.0004872, -0.0026871, 0.0024778
1: -0.0037060, 0.0025691, -0.0040275, 0.0025597, -0.0038368, 0.0043092
2: 0.0036045, 0.0090665, 0.0033280, 0.0091529, -0.0034320, 0.0035652
3: -0.0043708, -0.0035542, -0.0043477, -0.0034546, -0.0009162, 0.0007935
4: 0.0021552, 0.0072409, 0.0020569, 0.0074289, -0.0047202, 0.0047006
5: -0.0024134, 0.0025345, -0.0026589, 0.0025900, -0.0043284, 0.0044158
6: -0.0064666, -0.0034645, -0.0065312, -0.0034084, -0.0029551, 0.0029379
7: -0.0019125, 0.0032432, -0.0020871, 0.0033584, -0.0048100, 0.0048757
8: -0.0007729, -0.0000850, -0.0008693, -0.0000997, -0.0006599, 0.0007843
9: 0.9992130, 1.0106866, 0.9990337, 1.0113341, -0.0084608, 0.0081903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039293, upper bound: 0.0027214
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052903, upper bound: 0.0049598
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0032153, -0.0006410, -0.0031938, -0.0005238, -0.0026915, 0.0025528
1: -0.0037173, 0.0026514, -0.0039825, 0.0028280, -0.0039665, 0.0041225
2: 0.0035985, 0.0091054, 0.0033503, 0.0093729, -0.0036373, 0.0035745
3: -0.0043834, -0.0035514, -0.0043480, -0.0034835, -0.0008999, 0.0007966
4: 0.0021330, 0.0072631, 0.0018845, 0.0074735, -0.0047968, 0.0048479
5: -0.0024336, 0.0025805, -0.0026764, 0.0027742, -0.0044376, 0.0044600
6: -0.0064854, -0.0034585, -0.0065717, -0.0033234, -0.0030455, 0.0029904
7: -0.0019207, 0.0032706, -0.0022210, 0.0033825, -0.0048829, 0.0050773
8: -0.0007758, -0.0000742, -0.0008401, -0.0001025, -0.0006684, 0.0007658
9: 0.9991122, 1.0107126, 0.9985744, 1.0112939, -0.0085130, 0.0085423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052613, upper bound: 0.0048141
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052613, upper bound: 0.0049074
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0031743, -0.0006505, -0.0031344, -0.0004472, -0.0027271, 0.0024838
1: -0.0037060, 0.0025691, -0.0040244, 0.0026625, -0.0039031, 0.0042486
2: 0.0036045, 0.0090665, 0.0033400, 0.0093055, -0.0036084, 0.0035489
3: -0.0043708, -0.0035542, -0.0043538, -0.0034007, -0.0009702, 0.0007996
4: 0.0021552, 0.0072409, 0.0019006, 0.0074689, -0.0047524, 0.0048492
5: -0.0024134, 0.0025345, -0.0026950, 0.0026955, -0.0044229, 0.0044348
6: -0.0064666, -0.0034645, -0.0065604, -0.0033026, -0.0030564, 0.0029649
7: -0.0019125, 0.0032432, -0.0023209, 0.0033637, -0.0048548, 0.0051329
8: -0.0007729, -0.0000850, -0.0009101, -0.0001059, -0.0006669, 0.0008251
9: 0.9992130, 1.0106866, 0.9987555, 1.0113358, -0.0084604, 0.0084723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0029805, upper bound: 0.0015783
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053155, upper bound: 0.0048723
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0031873, -0.0005622, -0.0026605, 0.0025840
1: -0.0037189, 0.0027595, -0.0039831, 0.0027186, -0.0038633, 0.0042413
2: 0.0036031, 0.0092730, 0.0033344, 0.0092203, -0.0034969, 0.0037832
3: -0.0043870, -0.0034972, -0.0043465, -0.0035363, -0.0008506, 0.0008493
4: 0.0019695, 0.0073184, 0.0020407, 0.0074359, -0.0049220, 0.0047510
5: -0.0024858, 0.0026886, -0.0026415, 0.0026708, -0.0043892, 0.0045395
6: -0.0065199, -0.0033547, -0.0065449, -0.0034284, -0.0029804, 0.0030696
7: -0.0021635, 0.0032949, -0.0019880, 0.0033761, -0.0051214, 0.0048780
8: -0.0008141, -0.0000828, -0.0008007, -0.0000957, -0.0007184, 0.0007180
9: 0.9988145, 1.0107462, 0.9988492, 1.0112972, -0.0088050, 0.0083157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0031938, -0.0005238, -0.0026989, 0.0025905
1: -0.0037189, 0.0027595, -0.0039825, 0.0028280, -0.0039685, 0.0042350
2: 0.0036031, 0.0092730, 0.0033503, 0.0093729, -0.0036095, 0.0037138
3: -0.0043870, -0.0034972, -0.0043480, -0.0034835, -0.0009035, 0.0008508
4: 0.0019695, 0.0073184, 0.0018845, 0.0074735, -0.0049405, 0.0048871
5: -0.0024858, 0.0026886, -0.0026764, 0.0027742, -0.0044778, 0.0045580
6: -0.0065199, -0.0033547, -0.0065717, -0.0033234, -0.0030764, 0.0030895
7: -0.0021635, 0.0032949, -0.0022210, 0.0033825, -0.0051106, 0.0050872
8: -0.0008141, -0.0000828, -0.0008401, -0.0001025, -0.0007096, 0.0007573
9: 0.9988145, 1.0107462, 0.9985744, 1.0112939, -0.0087525, 0.0085360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0051271
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0031283, -0.0004872, -0.0026949, 0.0025159
1: -0.0037081, 0.0026755, -0.0040275, 0.0025597, -0.0038011, 0.0043846
2: 0.0036091, 0.0092352, 0.0033280, 0.0091529, -0.0034679, 0.0037657
3: -0.0043757, -0.0034998, -0.0043477, -0.0034546, -0.0009211, 0.0008479
4: 0.0019918, 0.0072965, 0.0020569, 0.0074289, -0.0048741, 0.0047509
5: -0.0024660, 0.0026437, -0.0026589, 0.0025900, -0.0043728, 0.0045180
6: -0.0065015, -0.0033608, -0.0065312, -0.0034084, -0.0029913, 0.0030415
7: -0.0021552, 0.0032676, -0.0020871, 0.0033584, -0.0050900, 0.0049374
8: -0.0008111, -0.0000937, -0.0008693, -0.0000997, -0.0007115, 0.0007756
9: 0.9989144, 1.0107208, 0.9990337, 1.0113341, -0.0087612, 0.0082430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0049165
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0050264
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0031344, -0.0004472, -0.0027349, 0.0025219
1: -0.0037081, 0.0026755, -0.0040244, 0.0026625, -0.0039064, 0.0043811
2: 0.0036091, 0.0092352, 0.0033400, 0.0093055, -0.0035806, 0.0036927
3: -0.0043757, -0.0034998, -0.0043538, -0.0034007, -0.0009751, 0.0008540
4: 0.0019918, 0.0072965, 0.0019006, 0.0074689, -0.0048956, 0.0048884
5: -0.0024660, 0.0026437, -0.0026950, 0.0026955, -0.0044631, 0.0045326
6: -0.0065015, -0.0033608, -0.0065604, -0.0033026, -0.0030875, 0.0030637
7: -0.0021552, 0.0032676, -0.0023209, 0.0033637, -0.0050818, 0.0051469
8: -0.0008111, -0.0000937, -0.0009101, -0.0001059, -0.0007052, 0.0008164
9: 0.9989144, 1.0107208, 0.9987555, 1.0113358, -0.0087083, 0.0084655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0049165
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051853, upper bound: 0.0050264
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0031516, -0.0006549, -0.0035886, -0.0006431, -0.0025085, 0.0029337
1: -0.0037027, 0.0025778, -0.0036092, 0.0035502, -0.0051002, 0.0039656
2: 0.0036079, 0.0090724, 0.0037016, 0.0096704, -0.0040448, 0.0033247
3: -0.0043465, -0.0035561, -0.0044489, -0.0035501, -0.0007963, 0.0008928
4: 0.0021563, 0.0072293, 0.0019274, 0.0072601, -0.0046139, 0.0047613
5: -0.0024031, 0.0025368, -0.0024135, 0.0028721, -0.0044821, 0.0042020
6: -0.0064536, -0.0034668, -0.0065190, -0.0034086, -0.0029336, 0.0029466
7: -0.0019066, 0.0032082, -0.0021160, 0.0032284, -0.0047566, 0.0049400
8: -0.0007708, -0.0001046, -0.0007866, -0.0000267, -0.0007189, 0.0006501
9: 0.9992027, 1.0106726, 0.9980841, 1.0105466, -0.0078586, 0.0090977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049430, upper bound: 0.0052124
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049430, upper bound: 0.0052124
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0030926, -0.0005829, -0.0035478, -0.0006521, -0.0024404, 0.0029649
1: -0.0037263, 0.0023993, -0.0035987, 0.0034699, -0.0052192, 0.0039023
2: 0.0036073, 0.0089986, 0.0037071, 0.0096329, -0.0040045, 0.0033054
3: -0.0043458, -0.0034793, -0.0044364, -0.0035528, -0.0007930, 0.0009570
4: 0.0021749, 0.0072176, 0.0019508, 0.0072406, -0.0046189, 0.0047079
5: -0.0024134, 0.0024513, -0.0023962, 0.0028273, -0.0044400, 0.0041976
6: -0.0064386, -0.0034454, -0.0064999, -0.0034145, -0.0029058, 0.0029578
7: -0.0020118, 0.0031911, -0.0021069, 0.0032009, -0.0048195, 0.0049086
8: -0.0008388, -0.0001081, -0.0007841, -0.0000376, -0.0007941, 0.0006506
9: 0.9993969, 1.0106951, 0.9981852, 1.0105247, -0.0078032, 0.0090071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0038480, upper bound: 0.0033413
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050285, upper bound: 0.0052704
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0031516, -0.0006549, -0.0035907, -0.0006116, -0.0025400, 0.0029358
1: -0.0037027, 0.0025778, -0.0036099, 0.0036701, -0.0052064, 0.0039532
2: 0.0036079, 0.0090724, 0.0037215, 0.0098261, -0.0042402, 0.0033233
3: -0.0043465, -0.0035561, -0.0044534, -0.0034995, -0.0008469, 0.0008972
4: 0.0021563, 0.0072293, 0.0017672, 0.0073020, -0.0046449, 0.0049108
5: -0.0024031, 0.0025368, -0.0024545, 0.0029785, -0.0045756, 0.0042302
6: -0.0064536, -0.0034668, -0.0065449, -0.0033039, -0.0030381, 0.0029695
7: -0.0019066, 0.0032082, -0.0023559, 0.0032441, -0.0047987, 0.0052177
8: -0.0007708, -0.0001046, -0.0008215, -0.0000376, -0.0007278, 0.0007092
9: 0.9992027, 1.0106726, 0.9978105, 1.0105405, -0.0078770, 0.0093943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0051463
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049788, upper bound: 0.0051463
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0030926, -0.0005829, -0.0035498, -0.0006206, -0.0024719, 0.0029669
1: -0.0037263, 0.0023993, -0.0035994, 0.0035882, -0.0053246, 0.0038897
2: 0.0036073, 0.0089986, 0.0037270, 0.0097894, -0.0041981, 0.0033041
3: -0.0043458, -0.0034793, -0.0044421, -0.0035020, -0.0008438, 0.0009627
4: 0.0021749, 0.0072176, 0.0017907, 0.0072824, -0.0046491, 0.0048581
5: -0.0024134, 0.0024513, -0.0024374, 0.0029334, -0.0045333, 0.0042249
6: -0.0064386, -0.0034454, -0.0065259, -0.0033097, -0.0030105, 0.0029800
7: -0.0020118, 0.0031911, -0.0023472, 0.0032158, -0.0048623, 0.0051859
8: -0.0008388, -0.0001081, -0.0008189, -0.0000481, -0.0007907, 0.0007095
9: 0.9993969, 1.0106951, 0.9979107, 1.0105183, -0.0078212, 0.0093019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0030582, upper bound: 0.0021081
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050574, upper bound: 0.0051864
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0035178, -0.0006568, -0.0025659, 0.0029145
1: -0.0037189, 0.0027595, -0.0035952, 0.0034648, -0.0049516, 0.0041539
2: 0.0036031, 0.0092730, 0.0037103, 0.0096327, -0.0040540, 0.0035412
3: -0.0043870, -0.0034972, -0.0044108, -0.0035548, -0.0008321, 0.0009137
4: 0.0019695, 0.0073184, 0.0019545, 0.0072297, -0.0047367, 0.0048394
5: -0.0024858, 0.0026886, -0.0023867, 0.0028243, -0.0045208, 0.0042989
6: -0.0065199, -0.0033547, -0.0064853, -0.0034169, -0.0029979, 0.0030200
7: -0.0021635, 0.0032949, -0.0020988, 0.0031654, -0.0049847, 0.0050495
8: -0.0008141, -0.0000828, -0.0007821, -0.0000568, -0.0007480, 0.0006928
9: 0.9988145, 1.0107462, 0.9981848, 1.0105116, -0.0081762, 0.0091043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0035189, -0.0006250, -0.0025977, 0.0029156
1: -0.0037189, 0.0027595, -0.0035960, 0.0035833, -0.0050657, 0.0041491
2: 0.0036031, 0.0092730, 0.0037305, 0.0097888, -0.0041855, 0.0034788
3: -0.0043870, -0.0034972, -0.0044141, -0.0035038, -0.0008831, 0.0009170
4: 0.0019695, 0.0073184, 0.0017937, 0.0072721, -0.0047587, 0.0049773
5: -0.0024858, 0.0026886, -0.0024284, 0.0029325, -0.0046124, 0.0043232
6: -0.0065199, -0.0033547, -0.0065116, -0.0033121, -0.0030973, 0.0030395
7: -0.0021635, 0.0032949, -0.0023398, 0.0031814, -0.0049746, 0.0052773
8: -0.0008141, -0.0000828, -0.0008170, -0.0000689, -0.0007371, 0.0007315
9: 0.9988145, 1.0107462, 0.9979103, 1.0105054, -0.0081346, 0.0093431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048757, upper bound: 0.0053996
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0034650, -0.0005734, -0.0026087, 0.0028526
1: -0.0037081, 0.0026755, -0.0036424, 0.0033092, -0.0048833, 0.0042603
2: 0.0036091, 0.0092352, 0.0036973, 0.0095672, -0.0040212, 0.0035170
3: -0.0043757, -0.0034998, -0.0044127, -0.0034767, -0.0008990, 0.0009129
4: 0.0019918, 0.0072965, 0.0019724, 0.0072251, -0.0046913, 0.0048430
5: -0.0024660, 0.0026437, -0.0024047, 0.0027476, -0.0045079, 0.0042712
6: -0.0065015, -0.0033608, -0.0064732, -0.0033962, -0.0030090, 0.0029961
7: -0.0021552, 0.0032676, -0.0021994, 0.0031511, -0.0049578, 0.0051047
8: -0.0008111, -0.0000937, -0.0008497, -0.0000606, -0.0007491, 0.0007560
9: 0.9989144, 1.0107208, 0.9983696, 1.0105525, -0.0081299, 0.0090300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0017109, upper bound: 0.0035429
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049452, upper bound: 0.0052969
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0034671, -0.0005384, -0.0026437, 0.0028546
1: -0.0037081, 0.0026755, -0.0036429, 0.0034229, -0.0049966, 0.0042587
2: 0.0036091, 0.0092352, 0.0037173, 0.0097248, -0.0041520, 0.0034530
3: -0.0043757, -0.0034998, -0.0044191, -0.0034230, -0.0009527, 0.0009192
4: 0.0019918, 0.0072965, 0.0018124, 0.0072674, -0.0047140, 0.0049817
5: -0.0024660, 0.0026437, -0.0024452, 0.0028542, -0.0046016, 0.0042947
6: -0.0065015, -0.0033608, -0.0064997, -0.0032911, -0.0031091, 0.0030142
7: -0.0021552, 0.0032676, -0.0024396, 0.0031644, -0.0049504, 0.0053294
8: -0.0008111, -0.0000937, -0.0008876, -0.0000708, -0.0007397, 0.0007939
9: 0.9989144, 1.0107208, 0.9980934, 1.0105491, -0.0080830, 0.0092695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0017109, upper bound: 0.0035429
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049452, upper bound: 0.0052969
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0031516, -0.0006549, -0.0036324, -0.0005467, -0.0026049, 0.0029775
1: -0.0037027, 0.0025778, -0.0039020, 0.0037194, -0.0052430, 0.0042687
2: 0.0036079, 0.0090724, 0.0034267, 0.0098170, -0.0042514, 0.0036246
3: -0.0043465, -0.0035561, -0.0044501, -0.0035311, -0.0008153, 0.0008939
4: 0.0021563, 0.0072293, 0.0017915, 0.0074726, -0.0048061, 0.0048818
5: -0.0024031, 0.0025368, -0.0026662, 0.0030375, -0.0046569, 0.0044489
6: -0.0064536, -0.0034668, -0.0066100, -0.0033576, -0.0029766, 0.0030297
7: -0.0019066, 0.0032082, -0.0022188, 0.0034067, -0.0048879, 0.0050170
8: -0.0007708, -0.0001046, -0.0008160, -0.0000183, -0.0007452, 0.0006945
9: 0.9992027, 1.0106726, 0.9976956, 1.0112096, -0.0085334, 0.0095848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052263, upper bound: 0.0050971
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052263, upper bound: 0.0050971
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0030926, -0.0005829, -0.0035913, -0.0005555, -0.0025370, 0.0030083
1: -0.0037263, 0.0023993, -0.0038920, 0.0036424, -0.0053712, 0.0042059
2: 0.0036073, 0.0089986, 0.0034319, 0.0097812, -0.0042143, 0.0036060
3: -0.0043458, -0.0034793, -0.0044383, -0.0035338, -0.0008121, 0.0009589
4: 0.0021749, 0.0072176, 0.0018151, 0.0074554, -0.0048126, 0.0048319
5: -0.0024134, 0.0024513, -0.0026510, 0.0029921, -0.0046188, 0.0044456
6: -0.0064386, -0.0034454, -0.0065922, -0.0033636, -0.0029490, 0.0030407
7: -0.0020118, 0.0031911, -0.0022104, 0.0033803, -0.0049510, 0.0049849
8: -0.0008388, -0.0001081, -0.0008135, -0.0000293, -0.0008095, 0.0006954
9: 0.9993969, 1.0106951, 0.9977913, 1.0111896, -0.0084805, 0.0094981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039299, upper bound: 0.0032004
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052880, upper bound: 0.0051547
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0031516, -0.0006549, -0.0036371, -0.0005163, -0.0026353, 0.0029822
1: -0.0037027, 0.0025778, -0.0038966, 0.0038290, -0.0053367, 0.0042415
2: 0.0036079, 0.0090724, 0.0034572, 0.0099634, -0.0044168, 0.0036017
3: -0.0043465, -0.0035561, -0.0044540, -0.0034806, -0.0008659, 0.0008979
4: 0.0021563, 0.0072293, 0.0016343, 0.0074986, -0.0048242, 0.0050274
5: -0.0024031, 0.0025368, -0.0026914, 0.0031390, -0.0047513, 0.0044637
6: -0.0064536, -0.0034668, -0.0066315, -0.0032509, -0.0030785, 0.0030476
7: -0.0019066, 0.0032082, -0.0024502, 0.0034086, -0.0049246, 0.0052836
8: -0.0007708, -0.0001046, -0.0008519, -0.0000276, -0.0007432, 0.0007473
9: 0.9992027, 1.0106726, 0.9974282, 1.0112001, -0.0085072, 0.0098566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052559, upper bound: 0.0050335
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052559, upper bound: 0.0050335
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0030926, -0.0005829, -0.0035955, -0.0005252, -0.0025674, 0.0030126
1: -0.0037263, 0.0023993, -0.0038867, 0.0037506, -0.0054627, 0.0041785
2: 0.0036073, 0.0089986, 0.0034624, 0.0099274, -0.0043775, 0.0035832
3: -0.0043458, -0.0034793, -0.0044430, -0.0034831, -0.0008627, 0.0009637
4: 0.0021749, 0.0072176, 0.0016575, 0.0074815, -0.0048307, 0.0049761
5: -0.0024134, 0.0024513, -0.0026761, 0.0030931, -0.0047125, 0.0044604
6: -0.0064386, -0.0034454, -0.0066138, -0.0032568, -0.0030511, 0.0030591
7: -0.0020118, 0.0031911, -0.0024420, 0.0033821, -0.0049888, 0.0052515
8: -0.0008388, -0.0001081, -0.0008494, -0.0000386, -0.0008002, 0.0007413
9: 0.9993969, 1.0106951, 0.9975232, 1.0111798, -0.0084550, 0.0097676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0030289, upper bound: 0.0018826
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053132, upper bound: 0.0050575
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0035594, -0.0005609, -0.0026618, 0.0029561
1: -0.0037189, 0.0027595, -0.0038879, 0.0036286, -0.0050843, 0.0044561
2: 0.0036031, 0.0092730, 0.0034353, 0.0097780, -0.0042548, 0.0038427
3: -0.0043870, -0.0034972, -0.0044115, -0.0035359, -0.0008510, 0.0009144
4: 0.0019695, 0.0073184, 0.0018189, 0.0074438, -0.0049313, 0.0049536
5: -0.0024858, 0.0026886, -0.0026410, 0.0029895, -0.0046901, 0.0045500
6: -0.0065199, -0.0033547, -0.0065773, -0.0033668, -0.0030397, 0.0031046
7: -0.0021635, 0.0032949, -0.0022024, 0.0033458, -0.0051171, 0.0051239
8: -0.0008141, -0.0000828, -0.0008113, -0.0000485, -0.0007656, 0.0007286
9: 0.9988145, 1.0107462, 0.9977974, 1.0111755, -0.0088571, 0.0095828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032227, -0.0006033, -0.0035646, -0.0005303, -0.0026924, 0.0029613
1: -0.0037189, 0.0027595, -0.0038827, 0.0037379, -0.0051964, 0.0044496
2: 0.0036031, 0.0092730, 0.0034660, 0.0099252, -0.0043689, 0.0037658
3: -0.0043870, -0.0034972, -0.0044147, -0.0034852, -0.0009018, 0.0009175
4: 0.0019695, 0.0073184, 0.0016608, 0.0074709, -0.0049395, 0.0050904
5: -0.0024858, 0.0026886, -0.0026668, 0.0030920, -0.0047820, 0.0045606
6: -0.0065199, -0.0033547, -0.0065992, -0.0032599, -0.0031367, 0.0031180
7: -0.0021635, 0.0032949, -0.0024343, 0.0033478, -0.0051003, 0.0053436
8: -0.0008141, -0.0000828, -0.0008474, -0.0000585, -0.0007556, 0.0007647
9: 0.9988145, 1.0107462, 0.9975315, 1.0111663, -0.0087805, 0.0098076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051755, upper bound: 0.0053006
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0035083, -0.0004734, -0.0027087, 0.0028958
1: -0.0037081, 0.0026755, -0.0039462, 0.0034863, -0.0050231, 0.0045899
2: 0.0036091, 0.0092352, 0.0034152, 0.0097182, -0.0042202, 0.0038327
3: -0.0043757, -0.0034998, -0.0044152, -0.0034525, -0.0009232, 0.0009153
4: 0.0019918, 0.0072965, 0.0018371, 0.0074483, -0.0048951, 0.0049489
5: -0.0024660, 0.0026437, -0.0026690, 0.0029118, -0.0046696, 0.0045367
6: -0.0065015, -0.0033608, -0.0065694, -0.0033472, -0.0030494, 0.0030820
7: -0.0021552, 0.0032676, -0.0022978, 0.0033338, -0.0050946, 0.0051688
8: -0.0008111, -0.0000937, -0.0008793, -0.0000525, -0.0007586, 0.0007856
9: 0.9989144, 1.0107208, 0.9979655, 1.0112453, -0.0088379, 0.0095014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013077, upper bound: 0.0031218
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052212, upper bound: 0.0051867
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0031821, -0.0006124, -0.0035114, -0.0004410, -0.0027411, 0.0028990
1: -0.0037081, 0.0026755, -0.0039384, 0.0035916, -0.0051332, 0.0045862
2: 0.0036091, 0.0092352, 0.0034425, 0.0098655, -0.0043336, 0.0037551
3: -0.0043757, -0.0034998, -0.0044206, -0.0034001, -0.0009757, 0.0009207
4: 0.0019918, 0.0072965, 0.0016779, 0.0074738, -0.0049010, 0.0050897
5: -0.0024660, 0.0026437, -0.0026922, 0.0030125, -0.0047625, 0.0045449
6: -0.0065015, -0.0033608, -0.0065916, -0.0032393, -0.0031468, 0.0030956
7: -0.0021552, 0.0032676, -0.0025299, 0.0033355, -0.0050798, 0.0053926
8: -0.0008111, -0.0000937, -0.0009172, -0.0000603, -0.0007508, 0.0008235
9: 0.9989144, 1.0107208, 0.9976957, 1.0112324, -0.0087608, 0.0097284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013077, upper bound: 0.0031218
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052212, upper bound: 0.0051867
time: 0.83 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0032153, -0.0006410, -0.0025463, 0.0026531
1: -0.0039831, 0.0027186, -0.0037173, 0.0026514, -0.0041634, 0.0038985
2: 0.0033344, 0.0092203, 0.0035985, 0.0091054, -0.0035805, 0.0034613
3: -0.0043465, -0.0035363, -0.0043834, -0.0035514, -0.0007951, 0.0008471
4: 0.0020407, 0.0074359, 0.0021330, 0.0072631, -0.0047013, 0.0047676
5: -0.0026415, 0.0026708, -0.0024336, 0.0025805, -0.0044373, 0.0043452
6: -0.0065449, -0.0034284, -0.0064854, -0.0034585, -0.0029658, 0.0029445
7: -0.0019880, 0.0033761, -0.0019207, 0.0032706, -0.0048186, 0.0048412
8: -0.0008007, -0.0000957, -0.0007758, -0.0000742, -0.0007136, 0.0006597
9: 0.9988492, 1.0112972, 0.9991122, 1.0107126, -0.0082634, 0.0085074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048807, upper bound: 0.0052324
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048807, upper bound: 0.0052324
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0031743, -0.0006505, -0.0024778, 0.0026871
1: -0.0040275, 0.0025597, -0.0037060, 0.0025691, -0.0043092, 0.0038368
2: 0.0033280, 0.0091529, 0.0036045, 0.0090665, -0.0035652, 0.0034320
3: -0.0043477, -0.0034546, -0.0043708, -0.0035542, -0.0007935, 0.0009162
4: 0.0020569, 0.0074289, 0.0021552, 0.0072409, -0.0047006, 0.0047202
5: -0.0026589, 0.0025900, -0.0024134, 0.0025345, -0.0044158, 0.0043284
6: -0.0065312, -0.0034084, -0.0064666, -0.0034645, -0.0029379, 0.0029551
7: -0.0020871, 0.0033584, -0.0019125, 0.0032432, -0.0048757, 0.0048100
8: -0.0008693, -0.0000997, -0.0007729, -0.0000850, -0.0007843, 0.0006599
9: 0.9990337, 1.0113341, 0.9992130, 1.0106866, -0.0081903, 0.0084608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0027214, upper bound: 0.0039293
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049598, upper bound: 0.0052903
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0031938, -0.0005238, -0.0032153, -0.0006410, -0.0025528, 0.0026915
1: -0.0039825, 0.0028280, -0.0037173, 0.0026514, -0.0041225, 0.0039665
2: 0.0033503, 0.0093729, 0.0035985, 0.0091054, -0.0035745, 0.0036373
3: -0.0043480, -0.0034835, -0.0043834, -0.0035514, -0.0007966, 0.0008999
4: 0.0018845, 0.0074735, 0.0021330, 0.0072631, -0.0048479, 0.0047968
5: -0.0026764, 0.0027742, -0.0024336, 0.0025805, -0.0044600, 0.0044376
6: -0.0065717, -0.0033234, -0.0064854, -0.0034585, -0.0029904, 0.0030455
7: -0.0022210, 0.0033825, -0.0019207, 0.0032706, -0.0050773, 0.0048829
8: -0.0008401, -0.0001025, -0.0007758, -0.0000742, -0.0007658, 0.0006684
9: 0.9985744, 1.0112939, 0.9991122, 1.0107126, -0.0085423, 0.0085130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048141, upper bound: 0.0052613
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048141, upper bound: 0.0052613
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0031344, -0.0004472, -0.0031743, -0.0006505, -0.0024838, 0.0027271
1: -0.0040244, 0.0026625, -0.0037060, 0.0025691, -0.0042486, 0.0039031
2: 0.0033400, 0.0093055, 0.0036045, 0.0090665, -0.0035489, 0.0036084
3: -0.0043538, -0.0034007, -0.0043708, -0.0035542, -0.0007996, 0.0009702
4: 0.0019006, 0.0074689, 0.0021552, 0.0072409, -0.0048492, 0.0047524
5: -0.0026950, 0.0026955, -0.0024134, 0.0025345, -0.0044348, 0.0044229
6: -0.0065604, -0.0033026, -0.0064666, -0.0034645, -0.0029649, 0.0030564
7: -0.0023209, 0.0033637, -0.0019125, 0.0032432, -0.0051329, 0.0048548
8: -0.0009101, -0.0001059, -0.0007729, -0.0000850, -0.0008251, 0.0006669
9: 0.9987555, 1.0113358, 0.9992130, 1.0106866, -0.0084723, 0.0084604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015783, upper bound: 0.0029805
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048723, upper bound: 0.0053155
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0032227, -0.0006033, -0.0025840, 0.0026605
1: -0.0039831, 0.0027186, -0.0037189, 0.0027595, -0.0042413, 0.0038633
2: 0.0033344, 0.0092203, 0.0036031, 0.0092730, -0.0037832, 0.0034969
3: -0.0043465, -0.0035363, -0.0043870, -0.0034972, -0.0008493, 0.0008506
4: 0.0020407, 0.0074359, 0.0019695, 0.0073184, -0.0047510, 0.0049220
5: -0.0026415, 0.0026708, -0.0024858, 0.0026886, -0.0045395, 0.0043892
6: -0.0065449, -0.0034284, -0.0065199, -0.0033547, -0.0030696, 0.0029804
7: -0.0019880, 0.0033761, -0.0021635, 0.0032949, -0.0048780, 0.0051214
8: -0.0008007, -0.0000957, -0.0008141, -0.0000828, -0.0007180, 0.0007184
9: 0.9988492, 1.0112972, 0.9988145, 1.0107462, -0.0083157, 0.0088050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0051853
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0051853
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0031938, -0.0005238, -0.0032227, -0.0006033, -0.0025905, 0.0026989
1: -0.0039825, 0.0028280, -0.0037189, 0.0027595, -0.0042350, 0.0039685
2: 0.0033503, 0.0093729, 0.0036031, 0.0092730, -0.0037138, 0.0036095
3: -0.0043480, -0.0034835, -0.0043870, -0.0034972, -0.0008508, 0.0009035
4: 0.0018845, 0.0074735, 0.0019695, 0.0073184, -0.0048871, 0.0049405
5: -0.0026764, 0.0027742, -0.0024858, 0.0026886, -0.0045580, 0.0044778
6: -0.0065717, -0.0033234, -0.0065199, -0.0033547, -0.0030895, 0.0030764
7: -0.0022210, 0.0033825, -0.0021635, 0.0032949, -0.0050872, 0.0051106
8: -0.0008401, -0.0001025, -0.0008141, -0.0000828, -0.0007573, 0.0007096
9: 0.9985744, 1.0112939, 0.9988145, 1.0107462, -0.0085360, 0.0087525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052613
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0052613
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0031821, -0.0006124, -0.0025159, 0.0026949
1: -0.0040275, 0.0025597, -0.0037081, 0.0026755, -0.0043846, 0.0038011
2: 0.0033280, 0.0091529, 0.0036091, 0.0092352, -0.0037657, 0.0034679
3: -0.0043477, -0.0034546, -0.0043757, -0.0034998, -0.0008479, 0.0009211
4: 0.0020569, 0.0074289, 0.0019918, 0.0072965, -0.0047509, 0.0048741
5: -0.0026589, 0.0025900, -0.0024660, 0.0026437, -0.0045180, 0.0043728
6: -0.0065312, -0.0034084, -0.0065015, -0.0033608, -0.0030415, 0.0029913
7: -0.0020871, 0.0033584, -0.0021552, 0.0032676, -0.0049374, 0.0050900
8: -0.0008693, -0.0000997, -0.0008111, -0.0000937, -0.0007756, 0.0007115
9: 0.9990337, 1.0113341, 0.9989144, 1.0107208, -0.0082430, 0.0087612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048141, upper bound: 0.0052571
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048141, upper bound: 0.0052571
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0031344, -0.0004472, -0.0031821, -0.0006124, -0.0025219, 0.0027349
1: -0.0040244, 0.0026625, -0.0037081, 0.0026755, -0.0043811, 0.0039064
2: 0.0033400, 0.0093055, 0.0036091, 0.0092352, -0.0036927, 0.0035806
3: -0.0043538, -0.0034007, -0.0043757, -0.0034998, -0.0008540, 0.0009751
4: 0.0019006, 0.0074689, 0.0019918, 0.0072965, -0.0048884, 0.0048956
5: -0.0026950, 0.0026955, -0.0024660, 0.0026437, -0.0045326, 0.0044631
6: -0.0065604, -0.0033026, -0.0065015, -0.0033608, -0.0030637, 0.0030875
7: -0.0023209, 0.0033637, -0.0021552, 0.0032676, -0.0051469, 0.0050818
8: -0.0009101, -0.0001059, -0.0008111, -0.0000937, -0.0008164, 0.0007052
9: 0.9987555, 1.0113358, 0.9989144, 1.0107208, -0.0084655, 0.0087083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048141, upper bound: 0.0053494
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048141, upper bound: 0.0053494
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0032527, -0.0005481, -0.0026392, 0.0026905
1: -0.0039831, 0.0027186, -0.0039975, 0.0027988, -0.0039101, 0.0038033
2: 0.0033344, 0.0092203, 0.0033253, 0.0092535, -0.0033778, 0.0033634
3: -0.0043465, -0.0035363, -0.0043841, -0.0035316, -0.0008149, 0.0008477
4: 0.0020407, 0.0074359, 0.0020153, 0.0074671, -0.0047725, 0.0047467
5: -0.0026415, 0.0026708, -0.0026686, 0.0027158, -0.0043765, 0.0043747
6: -0.0065449, -0.0034284, -0.0065752, -0.0034193, -0.0029735, 0.0029995
7: -0.0019880, 0.0033761, -0.0020029, 0.0034357, -0.0048891, 0.0048405
8: -0.0008007, -0.0000957, -0.0008056, -0.0000657, -0.0006807, 0.0006494
9: 0.9988492, 1.0112972, 0.9987560, 1.0113342, -0.0081825, 0.0082093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048852, upper bound: 0.0052324
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048852, upper bound: 0.0052324
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0032115, -0.0005573, -0.0025711, 0.0027243
1: -0.0040275, 0.0025597, -0.0039868, 0.0027209, -0.0040378, 0.0037474
2: 0.0033280, 0.0091529, 0.0033310, 0.0092169, -0.0033440, 0.0033474
3: -0.0043477, -0.0034546, -0.0043716, -0.0035343, -0.0008134, 0.0009169
4: 0.0020569, 0.0074289, 0.0020372, 0.0074465, -0.0047812, 0.0046984
5: -0.0026589, 0.0025900, -0.0026506, 0.0026707, -0.0043415, 0.0043734
6: -0.0065312, -0.0034084, -0.0065569, -0.0034255, -0.0029467, 0.0030123
7: -0.0020871, 0.0033584, -0.0019947, 0.0034086, -0.0049494, 0.0048094
8: -0.0008693, -0.0000997, -0.0008028, -0.0000769, -0.0007567, 0.0006510
9: 0.9990337, 1.0113341, 0.9988542, 1.0113102, -0.0081413, 0.0081360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0025856, upper bound: 0.0037799
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049655, upper bound: 0.0052903
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0032605, -0.0005100, -0.0026773, 0.0026983
1: -0.0039831, 0.0027186, -0.0039968, 0.0029088, -0.0039825, 0.0037707
2: 0.0033344, 0.0092203, 0.0033409, 0.0094064, -0.0035697, 0.0033864
3: -0.0043465, -0.0035363, -0.0043865, -0.0034789, -0.0008676, 0.0008502
4: 0.0020407, 0.0074359, 0.0018595, 0.0075038, -0.0048081, 0.0048972
5: -0.0026415, 0.0026708, -0.0027036, 0.0028191, -0.0044809, 0.0044060
6: -0.0065449, -0.0034284, -0.0066019, -0.0033144, -0.0030743, 0.0030256
7: -0.0019880, 0.0033761, -0.0022364, 0.0034418, -0.0049314, 0.0051035
8: -0.0008007, -0.0000957, -0.0008448, -0.0000720, -0.0006910, 0.0007072
9: 0.9988492, 1.0112972, 0.9984809, 1.0113308, -0.0082118, 0.0085073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0051853
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0051853
time: 1.09 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0032183, -0.0005190, -0.0026093, 0.0027311
1: -0.0040275, 0.0025597, -0.0039864, 0.0028270, -0.0041070, 0.0037142
2: 0.0033280, 0.0091529, 0.0033465, 0.0093697, -0.0035352, 0.0033705
3: -0.0043477, -0.0034546, -0.0043757, -0.0034815, -0.0008662, 0.0009211
4: 0.0020569, 0.0074289, 0.0018815, 0.0074840, -0.0048176, 0.0048496
5: -0.0026589, 0.0025900, -0.0026857, 0.0027745, -0.0044436, 0.0044055
6: -0.0065312, -0.0034084, -0.0065842, -0.0033204, -0.0030474, 0.0030385
7: -0.0020871, 0.0033584, -0.0022280, 0.0034141, -0.0049915, 0.0050724
8: -0.0008693, -0.0000997, -0.0008420, -0.0000834, -0.0007663, 0.0007088
9: 0.9990337, 1.0113341, 0.9985780, 1.0113076, -0.0081717, 0.0084273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0052571
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0052571
time: 1.02 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0032605, -0.0005100, -0.0031873, -0.0005622, -0.0026983, 0.0026773
1: -0.0039968, 0.0029088, -0.0039831, 0.0027186, -0.0037707, 0.0039825
2: 0.0033409, 0.0094064, 0.0033344, 0.0092203, -0.0033864, 0.0035697
3: -0.0043865, -0.0034789, -0.0043465, -0.0035363, -0.0008502, 0.0008676
4: 0.0018595, 0.0075038, 0.0020407, 0.0074359, -0.0048972, 0.0048081
5: -0.0027036, 0.0028191, -0.0026415, 0.0026708, -0.0044060, 0.0044809
6: -0.0066019, -0.0033144, -0.0065449, -0.0034284, -0.0030256, 0.0030743
7: -0.0022364, 0.0034418, -0.0019880, 0.0033761, -0.0051035, 0.0049314
8: -0.0008448, -0.0000720, -0.0008007, -0.0000957, -0.0007072, 0.0006910
9: 0.9984809, 1.0113308, 0.9988492, 1.0112972, -0.0085073, 0.0082118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0032605, -0.0005100, -0.0031938, -0.0005238, -0.0027367, 0.0026838
1: -0.0039968, 0.0029088, -0.0039825, 0.0028280, -0.0038661, 0.0039738
2: 0.0033409, 0.0094064, 0.0033503, 0.0093729, -0.0035067, 0.0035195
3: -0.0043865, -0.0034789, -0.0043480, -0.0034835, -0.0009030, 0.0008691
4: 0.0018595, 0.0075038, 0.0018845, 0.0074735, -0.0049213, 0.0049466
5: -0.0027036, 0.0028191, -0.0026764, 0.0027742, -0.0045048, 0.0045065
6: -0.0066019, -0.0033144, -0.0065717, -0.0033234, -0.0031215, 0.0030951
7: -0.0022364, 0.0034418, -0.0022210, 0.0033825, -0.0050959, 0.0051441
8: -0.0008448, -0.0000720, -0.0008401, -0.0001025, -0.0006978, 0.0007285
9: 0.9984809, 1.0113308, 0.9985744, 1.0112939, -0.0084667, 0.0084412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0054343
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0032183, -0.0005190, -0.0031283, -0.0004872, -0.0027311, 0.0026093
1: -0.0039864, 0.0028270, -0.0040275, 0.0025597, -0.0037142, 0.0041070
2: 0.0033465, 0.0093697, 0.0033280, 0.0091529, -0.0033705, 0.0035352
3: -0.0043757, -0.0034815, -0.0043477, -0.0034546, -0.0009211, 0.0008662
4: 0.0018815, 0.0074840, 0.0020569, 0.0074289, -0.0048496, 0.0048176
5: -0.0026857, 0.0027745, -0.0026589, 0.0025900, -0.0044055, 0.0044436
6: -0.0065842, -0.0033204, -0.0065312, -0.0034084, -0.0030385, 0.0030474
7: -0.0022280, 0.0034141, -0.0020871, 0.0033584, -0.0050724, 0.0049915
8: -0.0008420, -0.0000834, -0.0008693, -0.0000997, -0.0007088, 0.0007663
9: 0.9985780, 1.0113076, 0.9990337, 1.0113341, -0.0084273, 0.0081717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0052613
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0053494
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0032183, -0.0005190, -0.0031344, -0.0004472, -0.0027711, 0.0026153
1: -0.0039864, 0.0028270, -0.0040244, 0.0026625, -0.0038098, 0.0041019
2: 0.0033465, 0.0093697, 0.0033400, 0.0093055, -0.0034896, 0.0034843
3: -0.0043757, -0.0034815, -0.0043538, -0.0034007, -0.0009751, 0.0008722
4: 0.0018815, 0.0074840, 0.0019006, 0.0074689, -0.0048753, 0.0049566
5: -0.0026857, 0.0027745, -0.0026950, 0.0026955, -0.0045031, 0.0044686
6: -0.0065842, -0.0033204, -0.0065604, -0.0033026, -0.0031361, 0.0030690
7: -0.0022280, 0.0034141, -0.0023209, 0.0033637, -0.0050653, 0.0052042
8: -0.0008420, -0.0000834, -0.0009101, -0.0001059, -0.0007007, 0.0008038
9: 0.9985780, 1.0113076, 0.9987555, 1.0113358, -0.0083825, 0.0084010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0052613
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048189, upper bound: 0.0053494
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0035886, -0.0006431, -0.0025441, 0.0030264
1: -0.0039831, 0.0027186, -0.0036092, 0.0035502, -0.0055052, 0.0042176
2: 0.0033344, 0.0092203, 0.0037016, 0.0096704, -0.0043926, 0.0035645
3: -0.0043465, -0.0035363, -0.0044489, -0.0035501, -0.0007964, 0.0009126
4: 0.0020407, 0.0074359, 0.0019274, 0.0072601, -0.0047028, 0.0049468
5: -0.0026415, 0.0026708, -0.0024135, 0.0028721, -0.0047206, 0.0043430
6: -0.0065449, -0.0034284, -0.0065190, -0.0034086, -0.0030155, 0.0029786
7: -0.0019880, 0.0033761, -0.0021160, 0.0032284, -0.0048137, 0.0050716
8: -0.0008007, -0.0000957, -0.0007866, -0.0000267, -0.0007646, 0.0006732
9: 0.9988492, 1.0112972, 0.9980841, 1.0105466, -0.0082962, 0.0097644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048807, upper bound: 0.0055186
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048807, upper bound: 0.0055186
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0035478, -0.0006521, -0.0024762, 0.0030606
1: -0.0040275, 0.0025597, -0.0035987, 0.0034699, -0.0056384, 0.0041553
2: 0.0033280, 0.0091529, 0.0037071, 0.0096329, -0.0043812, 0.0035356
3: -0.0043477, -0.0034546, -0.0044364, -0.0035528, -0.0007949, 0.0009817
4: 0.0020569, 0.0074289, 0.0019508, 0.0072406, -0.0047021, 0.0048998
5: -0.0026589, 0.0025900, -0.0023962, 0.0028273, -0.0047026, 0.0043266
6: -0.0065312, -0.0034084, -0.0064999, -0.0034145, -0.0029876, 0.0029894
7: -0.0020871, 0.0033584, -0.0021069, 0.0032009, -0.0048704, 0.0050401
8: -0.0008693, -0.0000997, -0.0007841, -0.0000376, -0.0008317, 0.0006732
9: 0.9990337, 1.0113341, 0.9981852, 1.0105247, -0.0082227, 0.0097309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0028754, upper bound: 0.0046202
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049598, upper bound: 0.0055616
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0035907, -0.0006116, -0.0025757, 0.0030285
1: -0.0039831, 0.0027186, -0.0036099, 0.0036701, -0.0056114, 0.0042052
2: 0.0033344, 0.0092203, 0.0037215, 0.0098261, -0.0045880, 0.0035630
3: -0.0043465, -0.0035363, -0.0044534, -0.0034995, -0.0008470, 0.0009170
4: 0.0020407, 0.0074359, 0.0017672, 0.0073020, -0.0047337, 0.0050963
5: -0.0026415, 0.0026708, -0.0024545, 0.0029785, -0.0048141, 0.0043713
6: -0.0065449, -0.0034284, -0.0065449, -0.0033039, -0.0031201, 0.0030016
7: -0.0019880, 0.0033761, -0.0023559, 0.0032441, -0.0048558, 0.0053492
8: -0.0008007, -0.0000957, -0.0008215, -0.0000376, -0.0007631, 0.0007258
9: 0.9988492, 1.0112972, 0.9978105, 1.0105405, -0.0083145, 0.0100610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049165, upper bound: 0.0054663
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049165, upper bound: 0.0054663
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0035498, -0.0006206, -0.0025077, 0.0030626
1: -0.0040275, 0.0025597, -0.0035994, 0.0035882, -0.0057438, 0.0041427
2: 0.0033280, 0.0091529, 0.0037270, 0.0097894, -0.0045747, 0.0035344
3: -0.0043477, -0.0034546, -0.0044421, -0.0035020, -0.0008457, 0.0009874
4: 0.0020569, 0.0074289, 0.0017907, 0.0072824, -0.0047322, 0.0050499
5: -0.0026589, 0.0025900, -0.0024374, 0.0029334, -0.0047959, 0.0043539
6: -0.0065312, -0.0034084, -0.0065259, -0.0033097, -0.0030923, 0.0030116
7: -0.0020871, 0.0033584, -0.0023472, 0.0032158, -0.0049133, 0.0053174
8: -0.0008693, -0.0000997, -0.0008189, -0.0000481, -0.0008212, 0.0007192
9: 0.9990337, 1.0113341, 0.9979107, 1.0105183, -0.0082406, 0.0100257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0017187, upper bound: 0.0035788
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049899, upper bound: 0.0054870
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0031938, -0.0005238, -0.0035886, -0.0006431, -0.0025506, 0.0030648
1: -0.0039825, 0.0028280, -0.0036092, 0.0035502, -0.0054644, 0.0042856
2: 0.0033503, 0.0093729, 0.0037016, 0.0096704, -0.0043866, 0.0037404
3: -0.0043480, -0.0034835, -0.0044489, -0.0035501, -0.0007978, 0.0009654
4: 0.0018845, 0.0074735, 0.0019274, 0.0072601, -0.0048494, 0.0049760
5: -0.0026764, 0.0027742, -0.0024135, 0.0028721, -0.0047433, 0.0044354
6: -0.0065717, -0.0033234, -0.0065190, -0.0034086, -0.0030401, 0.0030796
7: -0.0022210, 0.0033825, -0.0021160, 0.0032284, -0.0050724, 0.0051133
8: -0.0008401, -0.0001025, -0.0007866, -0.0000267, -0.0008133, 0.0006819
9: 0.9985744, 1.0112939, 0.9980841, 1.0105466, -0.0085750, 0.0097700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055470
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055470
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0031938, -0.0005238, -0.0035907, -0.0006116, -0.0025822, 0.0030669
1: -0.0039825, 0.0028280, -0.0036099, 0.0036701, -0.0055827, 0.0042877
2: 0.0033503, 0.0093729, 0.0037215, 0.0098261, -0.0045291, 0.0036905
3: -0.0043480, -0.0034835, -0.0044534, -0.0034995, -0.0008484, 0.0009698
4: 0.0018845, 0.0074735, 0.0017672, 0.0073020, -0.0048710, 0.0051143
5: -0.0026764, 0.0027742, -0.0024545, 0.0029785, -0.0048332, 0.0044600
6: -0.0065717, -0.0033234, -0.0065449, -0.0033039, -0.0031396, 0.0030977
7: -0.0022210, 0.0033825, -0.0023559, 0.0032441, -0.0050656, 0.0053406
8: -0.0008401, -0.0001025, -0.0008215, -0.0000376, -0.0008024, 0.0007190
9: 0.9985744, 1.0112939, 0.9978105, 1.0105405, -0.0085389, 0.0100144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055471
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049966, upper bound: 0.0055471
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0031344, -0.0004472, -0.0035478, -0.0006521, -0.0024822, 0.0031006
1: -0.0040244, 0.0026625, -0.0035987, 0.0034699, -0.0055777, 0.0042216
2: 0.0033400, 0.0093055, 0.0037071, 0.0096329, -0.0043649, 0.0037120
3: -0.0043538, -0.0034007, -0.0044364, -0.0035528, -0.0008010, 0.0010357
4: 0.0019006, 0.0074689, 0.0019508, 0.0072406, -0.0048506, 0.0049320
5: -0.0026950, 0.0026955, -0.0023962, 0.0028273, -0.0047216, 0.0044211
6: -0.0065604, -0.0033026, -0.0064999, -0.0034145, -0.0030146, 0.0030907
7: -0.0023209, 0.0033637, -0.0021069, 0.0032009, -0.0051276, 0.0050850
8: -0.0009101, -0.0001059, -0.0007841, -0.0000376, -0.0008725, 0.0006781
9: 0.9987555, 1.0113358, 0.9981852, 1.0105247, -0.0085047, 0.0097305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015917, upper bound: 0.0035713
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048723, upper bound: 0.0055887
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031344, -0.0004472, -0.0035498, -0.0006206, -0.0025137, 0.0031026
1: -0.0040244, 0.0026625, -0.0035994, 0.0035882, -0.0057159, 0.0042249
2: 0.0033400, 0.0093055, 0.0037270, 0.0097894, -0.0045127, 0.0036619
3: -0.0043538, -0.0034007, -0.0044421, -0.0035020, -0.0008518, 0.0010414
4: 0.0019006, 0.0074689, 0.0017907, 0.0072824, -0.0048708, 0.0050709
5: -0.0026950, 0.0026955, -0.0024374, 0.0029334, -0.0048113, 0.0044444
6: -0.0065604, -0.0033026, -0.0065259, -0.0033097, -0.0031142, 0.0031081
7: -0.0023209, 0.0033637, -0.0023472, 0.0032158, -0.0051238, 0.0053114
8: -0.0009101, -0.0001059, -0.0008189, -0.0000481, -0.0008620, 0.0007129
9: 0.9987555, 1.0113358, 0.9979107, 1.0105183, -0.0084676, 0.0099777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015917, upper bound: 0.0035713
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048723, upper bound: 0.0055887
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0036324, -0.0005467, -0.0026405, 0.0030702
1: -0.0039831, 0.0027186, -0.0039020, 0.0037194, -0.0052508, 0.0041254
2: 0.0033344, 0.0092203, 0.0034267, 0.0098170, -0.0041945, 0.0034658
3: -0.0043465, -0.0035363, -0.0044501, -0.0035311, -0.0008154, 0.0009137
4: 0.0020407, 0.0074359, 0.0017915, 0.0074726, -0.0047819, 0.0049398
5: -0.0026415, 0.0026708, -0.0026662, 0.0030375, -0.0046724, 0.0043747
6: -0.0065449, -0.0034284, -0.0066100, -0.0033576, -0.0030293, 0.0030364
7: -0.0019880, 0.0033761, -0.0022188, 0.0034067, -0.0048862, 0.0050778
8: -0.0008007, -0.0000957, -0.0008160, -0.0000183, -0.0007316, 0.0006625
9: 0.9988492, 1.0112972, 0.9976956, 1.0112096, -0.0082224, 0.0094879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048852, upper bound: 0.0055186
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048852, upper bound: 0.0055186
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0035913, -0.0005555, -0.0025728, 0.0031041
1: -0.0040275, 0.0025597, -0.0038920, 0.0036424, -0.0053679, 0.0040690
2: 0.0033280, 0.0091529, 0.0034319, 0.0097812, -0.0041649, 0.0034505
3: -0.0043477, -0.0034546, -0.0044383, -0.0035338, -0.0008139, 0.0009836
4: 0.0020569, 0.0074289, 0.0018151, 0.0074554, -0.0047913, 0.0048931
5: -0.0026589, 0.0025900, -0.0026510, 0.0029921, -0.0046387, 0.0043744
6: -0.0065312, -0.0034084, -0.0065922, -0.0033636, -0.0030028, 0.0030490
7: -0.0020871, 0.0033584, -0.0022104, 0.0033803, -0.0049465, 0.0050465
8: -0.0008693, -0.0000997, -0.0008135, -0.0000293, -0.0008076, 0.0006641
9: 0.9990337, 1.0113341, 0.9977913, 1.0111896, -0.0081821, 0.0094178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0035098, upper bound: 0.0030858
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049655, upper bound: 0.0055616
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0031873, -0.0005622, -0.0036371, -0.0005163, -0.0026710, 0.0030749
1: -0.0039831, 0.0027186, -0.0038966, 0.0038290, -0.0053531, 0.0041117
2: 0.0033344, 0.0092203, 0.0034572, 0.0099634, -0.0043828, 0.0034628
3: -0.0043465, -0.0035363, -0.0044540, -0.0034806, -0.0008659, 0.0009177
4: 0.0020407, 0.0074359, 0.0016343, 0.0074986, -0.0048041, 0.0050885
5: -0.0026415, 0.0026708, -0.0026914, 0.0031390, -0.0047683, 0.0043959
6: -0.0065449, -0.0034284, -0.0066315, -0.0032509, -0.0031304, 0.0030557
7: -0.0019880, 0.0033761, -0.0024502, 0.0034086, -0.0049241, 0.0053506
8: -0.0008007, -0.0000957, -0.0008519, -0.0000276, -0.0007403, 0.0007202
9: 0.9988492, 1.0112972, 0.9974282, 1.0112001, -0.0082221, 0.0097772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0054662
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049208, upper bound: 0.0054662
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0031283, -0.0004872, -0.0035955, -0.0005252, -0.0026032, 0.0031083
1: -0.0040275, 0.0025597, -0.0038867, 0.0037506, -0.0054686, 0.0040550
2: 0.0033280, 0.0091529, 0.0034624, 0.0099274, -0.0043518, 0.0034476
3: -0.0043477, -0.0034546, -0.0044430, -0.0034831, -0.0008646, 0.0009884
4: 0.0020569, 0.0074289, 0.0016575, 0.0074815, -0.0048136, 0.0050412
5: -0.0026589, 0.0025900, -0.0026761, 0.0030931, -0.0047345, 0.0043962
6: -0.0065312, -0.0034084, -0.0066138, -0.0032568, -0.0031039, 0.0030685
7: -0.0020871, 0.0033584, -0.0024420, 0.0033821, -0.0049849, 0.0053193
8: -0.0008693, -0.0000997, -0.0008494, -0.0000386, -0.0008156, 0.0007217
9: 0.9990337, 1.0113341, 0.9975232, 1.0111798, -0.0081814, 0.0097055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.92 + 597.08 = 600.00 seconds
