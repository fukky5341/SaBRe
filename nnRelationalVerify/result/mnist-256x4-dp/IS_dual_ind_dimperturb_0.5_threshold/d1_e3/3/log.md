## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01370172


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041204, -0.0011153, -0.0041204, -0.0011153, -0.0030051, 0.0030051)
1: (0.0175636, 0.0324057, 0.0175636, 0.0324057, -0.0101610, 0.0101610)
2: (0.0203356, 0.0304670, 0.0203356, 0.0304670, -0.0072335, 0.0072335)
3: (0.0058991, 0.0172845, 0.0058991, 0.0172845, -0.0093307, 0.0093307)
4: (-0.0181161, -0.0071157, -0.0181161, -0.0071157, -0.0094716, 0.0094716)
5: (0.0123315, 0.0262829, 0.0123315, 0.0262829, -0.0119701, 0.0119701)
6: (0.0046475, 0.0151365, 0.0046475, 0.0151365, -0.0091271, 0.0091271)
7: (-0.0227235, -0.0113790, -0.0227235, -0.0113790, -0.0101701, 0.0101701)
8: (0.0073679, 0.0189193, 0.0073679, 0.0189193, -0.0094506, 0.0094506)
9: (0.8999735, 0.9498254, 0.8999735, 0.9498254, -0.0372796, 0.0372796)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.56 = 2.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0222192, upper bound: 0.0222192

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198972, upper bound: 0.0210258
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0210258, upper bound: 0.0210258
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 9, lower bound: -0.0198972, upper bound: 0.0210258
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 9, lower bound: -0.0210258, upper bound: 0.0210258

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040931, -0.0012180, -0.0041133, -0.0011432, -0.0029500, 0.0028953
1: 0.0176859, 0.0314734, 0.0175961, 0.0321408, -0.0095937, 0.0088038
2: 0.0203545, 0.0298127, 0.0203405, 0.0302828, -0.0068700, 0.0062284
3: 0.0059903, 0.0164468, 0.0059230, 0.0170439, -0.0087578, 0.0080449
4: -0.0172376, -0.0071600, -0.0178661, -0.0071272, -0.0083305, 0.0089890
5: 0.0124468, 0.0252689, 0.0123618, 0.0259967, -0.0113737, 0.0105934
6: 0.0047149, 0.0143320, 0.0046652, 0.0149051, -0.0086396, 0.0080048
7: -0.0219169, -0.0114743, -0.0224982, -0.0114040, -0.0092172, 0.0097492
8: 0.0074899, 0.0181385, 0.0074000, 0.0186947, -0.0088807, 0.0082067
9: 0.9038190, 0.9495879, 0.9010582, 0.9497629, -0.0315534, 0.0348776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198972, upper bound: 0.0198972
time: 0.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198972, upper bound: 0.0210258
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041618, -0.0011113, -0.0041125, -0.0011469, -0.0030149, 0.0030012
1: 0.0165849, 0.0318183, 0.0176032, 0.0321955, -0.0111213, 0.0091989
2: 0.0197064, 0.0300887, 0.0203405, 0.0303119, -0.0078727, 0.0066522
3: 0.0050654, 0.0167537, 0.0059266, 0.0170963, -0.0097588, 0.0083967
4: -0.0175898, -0.0062514, -0.0179275, -0.0071279, -0.0086672, 0.0098821
5: 0.0112660, 0.0256462, 0.0123662, 0.0260534, -0.0124885, 0.0109918
6: 0.0038333, 0.0146490, 0.0046674, 0.0149623, -0.0094769, 0.0083229
7: -0.0222639, -0.0104924, -0.0225544, -0.0114081, -0.0095117, 0.0106067
8: 0.0066058, 0.0183949, 0.0074050, 0.0187354, -0.0098666, 0.0085575
9: 0.9024206, 0.9535418, 0.9008343, 0.9497557, -0.0331374, 0.0398599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0210258, upper bound: 0.0198972
time: 0.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0210258, upper bound: 0.0210258
time: 0.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 9, lower bound: -0.0198972, upper bound: 0.0198972
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 9, lower bound: -0.0198972, upper bound: 0.0210258
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 9, lower bound: -0.0210258, upper bound: 0.0198972
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 9, lower bound: -0.0210258, upper bound: 0.0210258

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040931, -0.0012180, -0.0040931, -0.0012180, -0.0028751, 0.0028751
1: 0.0176859, 0.0314734, 0.0176859, 0.0314734, -0.0086738, 0.0086738
2: 0.0203545, 0.0298127, 0.0203545, 0.0298127, -0.0061980, 0.0061980
3: 0.0059903, 0.0164468, 0.0059903, 0.0164468, -0.0079363, 0.0079363
4: -0.0172376, -0.0071600, -0.0172376, -0.0071600, -0.0082581, 0.0082581
5: 0.0124468, 0.0252689, 0.0124468, 0.0252689, -0.0104672, 0.0104672
6: 0.0047149, 0.0143320, 0.0047149, 0.0143320, -0.0079203, 0.0079203
7: -0.0219169, -0.0114743, -0.0219169, -0.0114743, -0.0091122, 0.0091122
8: 0.0074899, 0.0181385, 0.0074899, 0.0181385, -0.0080841, 0.0080841
9: 0.9038190, 0.9495879, 0.9038190, 0.9495879, -0.0311518, 0.0311518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0195097
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0195740
time: 0.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040931, -0.0012180, -0.0041618, -0.0011113, -0.0029818, 0.0029438
1: 0.0176859, 0.0314734, 0.0165849, 0.0318183, -0.0094748, 0.0099244
2: 0.0203545, 0.0298127, 0.0197064, 0.0300887, -0.0067572, 0.0070460
3: 0.0059903, 0.0164468, 0.0050654, 0.0167537, -0.0083288, 0.0088192
4: -0.0172376, -0.0071600, -0.0175898, -0.0062514, -0.0090866, 0.0085666
5: 0.0124468, 0.0252689, 0.0112660, 0.0256462, -0.0108932, 0.0114835
6: 0.0047149, 0.0143320, 0.0038333, 0.0146490, -0.0082109, 0.0086969
7: -0.0219169, -0.0114743, -0.0222639, -0.0104924, -0.0099010, 0.0094254
8: 0.0074899, 0.0181385, 0.0066058, 0.0183949, -0.0084803, 0.0089324
9: 0.9038190, 0.9495879, 0.9024206, 0.9535418, -0.0355541, 0.0331942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0206127
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0206842
time: 0.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041618, -0.0011113, -0.0040931, -0.0012180, -0.0029438, 0.0029818
1: 0.0165849, 0.0318183, 0.0176859, 0.0314734, -0.0099244, 0.0094748
2: 0.0197064, 0.0300887, 0.0203545, 0.0298127, -0.0070460, 0.0067572
3: 0.0050654, 0.0167537, 0.0059903, 0.0164468, -0.0088192, 0.0083288
4: -0.0175898, -0.0062514, -0.0172376, -0.0071600, -0.0085666, 0.0090866
5: 0.0112660, 0.0256462, 0.0124468, 0.0252689, -0.0114835, 0.0108932
6: 0.0038333, 0.0146490, 0.0047149, 0.0143320, -0.0086969, 0.0082109
7: -0.0222639, -0.0104924, -0.0219169, -0.0114743, -0.0094254, 0.0099010
8: 0.0066058, 0.0183949, 0.0074899, 0.0181385, -0.0089324, 0.0084803
9: 0.9024206, 0.9535418, 0.9038190, 0.9495879, -0.0331942, 0.0355541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206842, upper bound: 0.0194476
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206842, upper bound: 0.0195479
time: 0.69 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041618, -0.0011113, -0.0041618, -0.0011113, -0.0030505, 0.0030505
1: 0.0165849, 0.0318183, 0.0165849, 0.0318183, -0.0093107, 0.0093107
2: 0.0197064, 0.0300887, 0.0197064, 0.0300887, -0.0066636, 0.0066636
3: 0.0050654, 0.0167537, 0.0050654, 0.0167537, -0.0084373, 0.0084373
4: -0.0175898, -0.0062514, -0.0175898, -0.0062514, -0.0086706, 0.0086706
5: 0.0112660, 0.0256462, 0.0112660, 0.0256462, -0.0110566, 0.0110566
6: 0.0038333, 0.0146490, 0.0038333, 0.0146490, -0.0083482, 0.0083482
7: -0.0222639, -0.0104924, -0.0222639, -0.0104924, -0.0095740, 0.0095740
8: 0.0066058, 0.0183949, 0.0066058, 0.0183949, -0.0086327, 0.0086327
9: 0.9024206, 0.9535418, 0.9024206, 0.9535418, -0.0331512, 0.0331512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206843, upper bound: 0.0194476
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206843, upper bound: 0.0195479
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0195097
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0195740
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0206127
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0195479, upper bound: 0.0206842
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0206842, upper bound: 0.0194476
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0206842, upper bound: 0.0195479
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0206843, upper bound: 0.0194476
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 9, lower bound: -0.0206843, upper bound: 0.0195479

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040772, -0.0012470, -0.0040890, -0.0012260, -0.0028512, 0.0028421
1: 0.0179339, 0.0314702, 0.0177495, 0.0314726, -0.0083883, 0.0085925
2: 0.0204639, 0.0297823, 0.0203828, 0.0298049, -0.0060363, 0.0061090
3: 0.0062061, 0.0164440, 0.0060465, 0.0164461, -0.0076772, 0.0078661
4: -0.0172247, -0.0072983, -0.0172343, -0.0071959, -0.0081730, 0.0080264
5: 0.0126484, 0.0252586, 0.0124993, 0.0252663, -0.0101718, 0.0103810
6: 0.0048708, 0.0143232, 0.0047553, 0.0143298, -0.0076887, 0.0078498
7: -0.0219041, -0.0116030, -0.0219137, -0.0115079, -0.0090171, 0.0088732
8: 0.0077234, 0.0181385, 0.0075500, 0.0181385, -0.0078172, 0.0080122
9: 0.9038353, 0.9487275, 0.9038231, 0.9493637, -0.0308606, 0.0301080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195118
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195118
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041110, -0.0011597, -0.0040871, -0.0012297, -0.0028813, 0.0029274
1: 0.0178279, 0.0315554, 0.0177807, 0.0314722, -0.0086464, 0.0088895
2: 0.0204509, 0.0298892, 0.0203969, 0.0298011, -0.0061036, 0.0063300
3: 0.0061027, 0.0165667, 0.0060664, 0.0164458, -0.0078845, 0.0080133
4: -0.0173189, -0.0072566, -0.0172329, -0.0072152, -0.0082340, 0.0081840
5: 0.0125205, 0.0253701, 0.0125251, 0.0252651, -0.0104542, 0.0104801
6: 0.0048006, 0.0144299, 0.0047759, 0.0143287, -0.0078760, 0.0079289
7: -0.0219900, -0.0114976, -0.0219122, -0.0115262, -0.0090558, 0.0091164
8: 0.0075862, 0.0182502, 0.0075739, 0.0181385, -0.0080707, 0.0081932
9: 0.9034339, 0.9489397, 0.9038250, 0.9492687, -0.0314869, 0.0307535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195754
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195754
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040772, -0.0012470, -0.0041575, -0.0011211, -0.0029562, 0.0029105
1: 0.0179339, 0.0314702, 0.0166498, 0.0318175, -0.0091898, 0.0098432
2: 0.0204639, 0.0297823, 0.0197349, 0.0300810, -0.0065964, 0.0069584
3: 0.0062061, 0.0164440, 0.0051213, 0.0167529, -0.0080696, 0.0087481
4: -0.0172247, -0.0072983, -0.0175862, -0.0062885, -0.0090027, 0.0083348
5: 0.0126484, 0.0252586, 0.0113201, 0.0256433, -0.0105976, 0.0113953
6: 0.0048708, 0.0143232, 0.0038752, 0.0146465, -0.0079792, 0.0086277
7: -0.0219041, -0.0116030, -0.0222603, -0.0105270, -0.0098058, 0.0091863
8: 0.0077234, 0.0181385, 0.0066660, 0.0183949, -0.0082134, 0.0088607
9: 0.9038353, 0.9487275, 0.9024253, 0.9533166, -0.0352662, 0.0321501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206127
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206127
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041110, -0.0011597, -0.0041554, -0.0011259, -0.0029851, 0.0029957
1: 0.0178279, 0.0315554, 0.0166815, 0.0318170, -0.0094481, 0.0101236
2: 0.0204509, 0.0298892, 0.0197499, 0.0300771, -0.0066643, 0.0071673
3: 0.0061027, 0.0165667, 0.0051453, 0.0167526, -0.0082769, 0.0088794
4: -0.0173189, -0.0072566, -0.0175847, -0.0063052, -0.0090461, 0.0084928
5: 0.0125205, 0.0253701, 0.0113444, 0.0256421, -0.0108800, 0.0114776
6: 0.0048006, 0.0144299, 0.0038935, 0.0146455, -0.0081667, 0.0086888
7: -0.0219900, -0.0114976, -0.0222588, -0.0105443, -0.0098309, 0.0094300
8: 0.0075862, 0.0182502, 0.0066936, 0.0183949, -0.0084669, 0.0090252
9: 0.9034339, 0.9489397, 0.9024272, 0.9532178, -0.0358081, 0.0327957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206842
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206842
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041448, -0.0011492, -0.0040890, -0.0012260, -0.0029188, 0.0029399
1: 0.0168357, 0.0318153, 0.0177495, 0.0314726, -0.0096387, 0.0093952
2: 0.0198172, 0.0300587, 0.0203828, 0.0298049, -0.0068827, 0.0066717
3: 0.0052827, 0.0167506, 0.0060465, 0.0164461, -0.0085554, 0.0082584
4: -0.0175758, -0.0063936, -0.0172343, -0.0071959, -0.0084809, 0.0088548
5: 0.0114739, 0.0256350, 0.0124993, 0.0252663, -0.0111783, 0.0108063
6: 0.0039953, 0.0146394, 0.0047553, 0.0143298, -0.0084650, 0.0081404
7: -0.0222499, -0.0106246, -0.0219137, -0.0115079, -0.0093298, 0.0096567
8: 0.0068386, 0.0183949, 0.0075500, 0.0181385, -0.0086609, 0.0084086
9: 0.9024385, 0.9526645, 0.9038231, 0.9493637, -0.0329021, 0.0345056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041786, -0.0010474, -0.0040871, -0.0012297, -0.0029489, 0.0030397
1: 0.0167330, 0.0318992, 0.0177807, 0.0314722, -0.0098609, 0.0095052
2: 0.0198077, 0.0301678, 0.0203969, 0.0298011, -0.0069396, 0.0068359
3: 0.0051897, 0.0168701, 0.0060664, 0.0164458, -0.0087442, 0.0083488
4: -0.0176768, -0.0063450, -0.0172329, -0.0072152, -0.0085314, 0.0090048
5: 0.0113469, 0.0257674, 0.0125251, 0.0252651, -0.0114252, 0.0108835
6: 0.0039222, 0.0147461, 0.0047759, 0.0143287, -0.0086357, 0.0081982
7: -0.0223336, -0.0105167, -0.0219122, -0.0115262, -0.0093505, 0.0098715
8: 0.0067165, 0.0185066, 0.0075739, 0.0181385, -0.0088855, 0.0085143
9: 0.9020081, 0.9528793, 0.9038250, 0.9492687, -0.0333313, 0.0350666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041448, -0.0011492, -0.0041575, -0.0011211, -0.0030238, 0.0030083
1: 0.0168357, 0.0318153, 0.0166498, 0.0318175, -0.0090204, 0.0092281
2: 0.0198172, 0.0300587, 0.0197349, 0.0300810, -0.0065010, 0.0065715
3: 0.0052827, 0.0167506, 0.0051213, 0.0167529, -0.0081768, 0.0083667
4: -0.0175758, -0.0063936, -0.0175862, -0.0062885, -0.0085861, 0.0084397
5: 0.0114739, 0.0256350, 0.0113201, 0.0256433, -0.0107606, 0.0109691
6: 0.0039953, 0.0146394, 0.0038752, 0.0146465, -0.0081172, 0.0082780
7: -0.0222499, -0.0106246, -0.0222603, -0.0105270, -0.0094793, 0.0093351
8: 0.0068386, 0.0183949, 0.0066660, 0.0183949, -0.0083655, 0.0085610
9: 0.9024385, 0.9526645, 0.9024253, 0.9533166, -0.0328611, 0.0321165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041786, -0.0010474, -0.0041554, -0.0011259, -0.0030527, 0.0031080
1: 0.0167330, 0.0318992, 0.0166815, 0.0318170, -0.0092776, 0.0095215
2: 0.0198077, 0.0301678, 0.0197499, 0.0300771, -0.0065698, 0.0067928
3: 0.0051897, 0.0168701, 0.0051453, 0.0167526, -0.0083760, 0.0085132
4: -0.0176768, -0.0063450, -0.0175847, -0.0063052, -0.0086512, 0.0085935
5: 0.0113469, 0.0257674, 0.0113444, 0.0256421, -0.0110332, 0.0110698
6: 0.0039222, 0.0147461, 0.0038935, 0.0146455, -0.0082987, 0.0083606
7: -0.0223336, -0.0105167, -0.0222588, -0.0105443, -0.0095198, 0.0095789
8: 0.0067165, 0.0185066, 0.0066936, 0.0183949, -0.0086062, 0.0087364
9: 0.9020081, 0.9528793, 0.9024272, 0.9532178, -0.0334915, 0.0327264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
time: 0.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.87 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195118
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195118
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195754
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0195118, upper bound: 0.0195754
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206127
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206127
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206842
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0194476, upper bound: 0.0206842
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0194476
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 9, lower bound: -0.0206127, upper bound: 0.0195479

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040772, -0.0012470, -0.0040772, -0.0012470, -0.0028302, 0.0028302
1: 0.0179339, 0.0314702, 0.0179339, 0.0314702, -0.0083826, 0.0083826
2: 0.0204639, 0.0297823, 0.0204639, 0.0297823, -0.0059996, 0.0059996
3: 0.0062061, 0.0164440, 0.0062061, 0.0164440, -0.0076750, 0.0076750
4: -0.0172247, -0.0072983, -0.0172247, -0.0072983, -0.0080067, 0.0080067
5: 0.0126484, 0.0252586, 0.0126484, 0.0252586, -0.0101636, 0.0101636
6: 0.0048708, 0.0143232, 0.0048708, 0.0143232, -0.0076806, 0.0076806
7: -0.0219041, -0.0116030, -0.0219041, -0.0116030, -0.0088470, 0.0088470
8: 0.0077234, 0.0181385, 0.0077234, 0.0181385, -0.0078163, 0.0078163
9: 0.9038353, 0.9487275, 0.9038353, 0.9487275, -0.0300948, 0.0300948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191285, upper bound: 0.0189615
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193549, upper bound: 0.0193352
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040772, -0.0012470, -0.0041110, -0.0011597, -0.0029175, 0.0028640
1: 0.0179339, 0.0314702, 0.0178279, 0.0315554, -0.0086963, 0.0086268
2: 0.0204639, 0.0297823, 0.0204509, 0.0298892, -0.0062294, 0.0060341
3: 0.0062061, 0.0164440, 0.0061027, 0.0165667, -0.0078418, 0.0078176
4: -0.0172247, -0.0072983, -0.0173189, -0.0072566, -0.0080823, 0.0080883
5: 0.0126484, 0.0252586, 0.0125205, 0.0253701, -0.0102930, 0.0103495
6: 0.0048708, 0.0143232, 0.0048006, 0.0144299, -0.0077803, 0.0077876
7: -0.0219041, -0.0116030, -0.0219900, -0.0114976, -0.0090060, 0.0089099
8: 0.0077234, 0.0181385, 0.0075862, 0.0182502, -0.0080155, 0.0080134
9: 0.9038353, 0.9487275, 0.9034339, 0.9489397, -0.0305145, 0.0307818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191285, upper bound: 0.0189615
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193549, upper bound: 0.0193352
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041110, -0.0011597, -0.0040772, -0.0012470, -0.0028640, 0.0029175
1: 0.0178279, 0.0315554, 0.0179339, 0.0314702, -0.0086268, 0.0086963
2: 0.0204509, 0.0298892, 0.0204639, 0.0297823, -0.0060341, 0.0062294
3: 0.0061027, 0.0165667, 0.0062061, 0.0164440, -0.0078176, 0.0078418
4: -0.0173189, -0.0072566, -0.0172247, -0.0072983, -0.0080883, 0.0080823
5: 0.0125205, 0.0253701, 0.0126484, 0.0252586, -0.0103495, 0.0102930
6: 0.0048006, 0.0144299, 0.0048708, 0.0143232, -0.0077876, 0.0077803
7: -0.0219900, -0.0114976, -0.0219041, -0.0116030, -0.0089099, 0.0090060
8: 0.0075862, 0.0182502, 0.0077234, 0.0181385, -0.0080134, 0.0080155
9: 0.9034339, 0.9489397, 0.9038353, 0.9487275, -0.0307818, 0.0305145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191074, upper bound: 0.0190427
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193352, upper bound: 0.0193999
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041110, -0.0011597, -0.0041110, -0.0011597, -0.0029513, 0.0029513
1: 0.0178279, 0.0315554, 0.0178279, 0.0315554, -0.0086981, 0.0086981
2: 0.0204509, 0.0298892, 0.0204509, 0.0298892, -0.0062020, 0.0062020
3: 0.0061027, 0.0165667, 0.0061027, 0.0165667, -0.0078916, 0.0078916
4: -0.0173189, -0.0072566, -0.0173189, -0.0072566, -0.0081655, 0.0081655
5: 0.0125205, 0.0253701, 0.0125205, 0.0253701, -0.0104497, 0.0104497
6: 0.0048006, 0.0144299, 0.0048006, 0.0144299, -0.0078696, 0.0078696
7: -0.0219900, -0.0114976, -0.0219900, -0.0114976, -0.0090905, 0.0090905
8: 0.0075862, 0.0182502, 0.0075862, 0.0182502, -0.0080895, 0.0080895
9: 0.9034339, 0.9489397, 0.9034339, 0.9489397, -0.0307773, 0.0307773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191074, upper bound: 0.0190428
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193352, upper bound: 0.0193999
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040772, -0.0012470, -0.0041448, -0.0011492, -0.0029280, 0.0028978
1: 0.0179339, 0.0314702, 0.0168357, 0.0318153, -0.0091852, 0.0096329
2: 0.0204639, 0.0297823, 0.0198172, 0.0300587, -0.0065623, 0.0068460
3: 0.0062061, 0.0164440, 0.0052827, 0.0167506, -0.0080673, 0.0085532
4: -0.0172247, -0.0072983, -0.0175758, -0.0063936, -0.0088351, 0.0083147
5: 0.0126484, 0.0252586, 0.0114739, 0.0256350, -0.0105890, 0.0111701
6: 0.0048708, 0.0143232, 0.0039953, 0.0146394, -0.0079712, 0.0084569
7: -0.0219041, -0.0116030, -0.0222499, -0.0106246, -0.0096305, 0.0091597
8: 0.0077234, 0.0181385, 0.0068386, 0.0183949, -0.0082126, 0.0086600
9: 0.9038353, 0.9487275, 0.9024385, 0.9526645, -0.0344924, 0.0321363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190837, upper bound: 0.0199844
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193072, upper bound: 0.0204331
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040772, -0.0012470, -0.0041786, -0.0010474, -0.0030299, 0.0029316
1: 0.0179339, 0.0314702, 0.0167330, 0.0318992, -0.0093121, 0.0097995
2: 0.0204639, 0.0297823, 0.0198077, 0.0301678, -0.0067353, 0.0068557
3: 0.0062061, 0.0164440, 0.0051897, 0.0168701, -0.0081774, 0.0086428
4: -0.0172247, -0.0072983, -0.0176768, -0.0063450, -0.0088732, 0.0083857
5: 0.0126484, 0.0252586, 0.0113469, 0.0257674, -0.0106964, 0.0113029
6: 0.0048708, 0.0143232, 0.0039222, 0.0147461, -0.0080496, 0.0085204
7: -0.0219041, -0.0116030, -0.0223336, -0.0105167, -0.0097593, 0.0092045
8: 0.0077234, 0.0181385, 0.0067165, 0.0185066, -0.0083365, 0.0087974
9: 0.9038353, 0.9487275, 0.9020081, 0.9528793, -0.0346667, 0.0326262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190837, upper bound: 0.0199844
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193072, upper bound: 0.0204331
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041110, -0.0011597, -0.0041448, -0.0011492, -0.0029618, 0.0029851
1: 0.0178279, 0.0315554, 0.0168357, 0.0318153, -0.0094295, 0.0099466
2: 0.0204509, 0.0298892, 0.0198172, 0.0300587, -0.0065968, 0.0070757
3: 0.0061027, 0.0165667, 0.0052827, 0.0167506, -0.0082099, 0.0087200
4: -0.0173189, -0.0072566, -0.0175758, -0.0063936, -0.0089167, 0.0083903
5: 0.0125205, 0.0253701, 0.0114739, 0.0256350, -0.0107749, 0.0112995
6: 0.0048006, 0.0144299, 0.0039953, 0.0146394, -0.0080782, 0.0085566
7: -0.0219900, -0.0114976, -0.0222499, -0.0106246, -0.0096933, 0.0093187
8: 0.0075862, 0.0182502, 0.0068386, 0.0183949, -0.0084098, 0.0088592
9: 0.9034339, 0.9489397, 0.9024385, 0.9526645, -0.0351794, 0.0325559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190488, upper bound: 0.0200807
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192672, upper bound: 0.0205040
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041110, -0.0011597, -0.0041786, -0.0010474, -0.0030636, 0.0030189
1: 0.0178279, 0.0315554, 0.0167330, 0.0318992, -0.0094736, 0.0099127
2: 0.0204509, 0.0298892, 0.0198077, 0.0301678, -0.0067488, 0.0070380
3: 0.0061027, 0.0165667, 0.0051897, 0.0168701, -0.0082814, 0.0087513
4: -0.0173189, -0.0072566, -0.0176768, -0.0063450, -0.0089864, 0.0084794
5: 0.0125205, 0.0253701, 0.0113469, 0.0257674, -0.0108795, 0.0114208
6: 0.0048006, 0.0144299, 0.0039222, 0.0147461, -0.0081644, 0.0086294
7: -0.0219900, -0.0114976, -0.0223336, -0.0105167, -0.0098457, 0.0094093
8: 0.0075862, 0.0182502, 0.0067165, 0.0185066, -0.0084777, 0.0089043
9: 0.9034339, 0.9489397, 0.9020081, 0.9528793, -0.0350904, 0.0328152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190488, upper bound: 0.0200807
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192672, upper bound: 0.0205040
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041448, -0.0011492, -0.0040772, -0.0012470, -0.0028978, 0.0029280
1: 0.0168357, 0.0318153, 0.0179339, 0.0314702, -0.0096329, 0.0091852
2: 0.0198172, 0.0300587, 0.0204639, 0.0297823, -0.0068460, 0.0065623
3: 0.0052827, 0.0167506, 0.0062061, 0.0164440, -0.0085532, 0.0080673
4: -0.0175758, -0.0063936, -0.0172247, -0.0072983, -0.0083147, 0.0088351
5: 0.0114739, 0.0256350, 0.0126484, 0.0252586, -0.0111701, 0.0105890
6: 0.0039953, 0.0146394, 0.0048708, 0.0143232, -0.0084569, 0.0079712
7: -0.0222499, -0.0106246, -0.0219041, -0.0116030, -0.0091597, 0.0096305
8: 0.0068386, 0.0183949, 0.0077234, 0.0181385, -0.0086600, 0.0082126
9: 0.9024385, 0.9526645, 0.9038353, 0.9487275, -0.0321363, 0.0344924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204636, upper bound: 0.0192672
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041448, -0.0011492, -0.0041110, -0.0011597, -0.0029851, 0.0029618
1: 0.0168357, 0.0318153, 0.0178279, 0.0315554, -0.0099466, 0.0094295
2: 0.0198172, 0.0300587, 0.0204509, 0.0298892, -0.0070757, 0.0065968
3: 0.0052827, 0.0167506, 0.0061027, 0.0165667, -0.0087200, 0.0082099
4: -0.0175758, -0.0063936, -0.0173189, -0.0072566, -0.0083903, 0.0089167
5: 0.0114739, 0.0256350, 0.0125205, 0.0253701, -0.0112995, 0.0107749
6: 0.0039953, 0.0146394, 0.0048006, 0.0144299, -0.0085566, 0.0080782
7: -0.0222499, -0.0106246, -0.0219900, -0.0114976, -0.0093187, 0.0096933
8: 0.0068386, 0.0183949, 0.0075862, 0.0182502, -0.0088592, 0.0084098
9: 0.9024385, 0.9526645, 0.9034339, 0.9489397, -0.0325559, 0.0351794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204636, upper bound: 0.0192672
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041786, -0.0010474, -0.0040772, -0.0012470, -0.0029316, 0.0030299
1: 0.0167330, 0.0318992, 0.0179339, 0.0314702, -0.0097995, 0.0093121
2: 0.0198077, 0.0301678, 0.0204639, 0.0297823, -0.0068557, 0.0067353
3: 0.0051897, 0.0168701, 0.0062061, 0.0164440, -0.0086428, 0.0081774
4: -0.0176768, -0.0063450, -0.0172247, -0.0072983, -0.0083857, 0.0088732
5: 0.0113469, 0.0257674, 0.0126484, 0.0252586, -0.0113029, 0.0106964
6: 0.0039222, 0.0147461, 0.0048708, 0.0143232, -0.0085204, 0.0080496
7: -0.0223336, -0.0105167, -0.0219041, -0.0116030, -0.0092045, 0.0097593
8: 0.0067165, 0.0185066, 0.0077234, 0.0181385, -0.0087974, 0.0083365
9: 0.9020081, 0.9528793, 0.9038353, 0.9487275, -0.0326262, 0.0346667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041786, -0.0010474, -0.0041110, -0.0011597, -0.0030189, 0.0030636
1: 0.0167330, 0.0318992, 0.0178279, 0.0315554, -0.0099127, 0.0094736
2: 0.0198077, 0.0301678, 0.0204509, 0.0298892, -0.0070380, 0.0067488
3: 0.0051897, 0.0168701, 0.0061027, 0.0165667, -0.0087513, 0.0082814
4: -0.0176768, -0.0063450, -0.0173189, -0.0072566, -0.0084794, 0.0089864
5: 0.0113469, 0.0257674, 0.0125205, 0.0253701, -0.0114208, 0.0108794
6: 0.0039222, 0.0147461, 0.0048006, 0.0144299, -0.0086294, 0.0081644
7: -0.0223336, -0.0105167, -0.0219900, -0.0114976, -0.0094093, 0.0098457
8: 0.0067165, 0.0185066, 0.0075862, 0.0182502, -0.0089042, 0.0084777
9: 0.9020081, 0.9528793, 0.9034339, 0.9489397, -0.0328152, 0.0350904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041448, -0.0011492, -0.0041448, -0.0011492, -0.0029956, 0.0029956
1: 0.0168357, 0.0318153, 0.0168357, 0.0318153, -0.0090143, 0.0090143
2: 0.0198172, 0.0300587, 0.0198172, 0.0300587, -0.0064618, 0.0064618
3: 0.0052827, 0.0167506, 0.0052827, 0.0167506, -0.0081744, 0.0081744
4: -0.0175758, -0.0063936, -0.0175758, -0.0063936, -0.0084200, 0.0084200
5: 0.0114739, 0.0256350, 0.0114739, 0.0256350, -0.0107518, 0.0107518
6: 0.0039953, 0.0146394, 0.0039953, 0.0146394, -0.0081091, 0.0081091
7: -0.0222499, -0.0106246, -0.0222499, -0.0106246, -0.0093090, 0.0093090
8: 0.0068386, 0.0183949, 0.0068386, 0.0183949, -0.0083644, 0.0083644
9: 0.9024385, 0.9526645, 0.9024385, 0.9526645, -0.0321025, 0.0321025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204637, upper bound: 0.0192672
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041448, -0.0011492, -0.0041786, -0.0010474, -0.0030975, 0.0030294
1: 0.0168357, 0.0318153, 0.0167330, 0.0318992, -0.0093266, 0.0092510
2: 0.0198172, 0.0300587, 0.0198077, 0.0301678, -0.0066926, 0.0064952
3: 0.0052827, 0.0167506, 0.0051897, 0.0168701, -0.0083418, 0.0083132
4: -0.0175758, -0.0063936, -0.0176768, -0.0063450, -0.0084944, 0.0085063
5: 0.0114739, 0.0256350, 0.0113469, 0.0257674, -0.0108827, 0.0109312
6: 0.0039953, 0.0146394, 0.0039222, 0.0147461, -0.0082133, 0.0082136
7: -0.0222499, -0.0106246, -0.0223336, -0.0105167, -0.0094614, 0.0093760
8: 0.0068386, 0.0183949, 0.0067165, 0.0185066, -0.0085587, 0.0085549
9: 0.9024385, 0.9526645, 0.9020081, 0.9528793, -0.0324987, 0.0327995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204637, upper bound: 0.0192672
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041786, -0.0010474, -0.0041448, -0.0011492, -0.0030294, 0.0030975
1: 0.0167330, 0.0318992, 0.0168357, 0.0318153, -0.0092510, 0.0093266
2: 0.0198077, 0.0301678, 0.0198172, 0.0300587, -0.0064952, 0.0066926
3: 0.0051897, 0.0168701, 0.0052827, 0.0167506, -0.0083132, 0.0083418
4: -0.0176768, -0.0063450, -0.0175758, -0.0063936, -0.0085063, 0.0084944
5: 0.0113469, 0.0257674, 0.0114739, 0.0256350, -0.0109312, 0.0108827
6: 0.0039222, 0.0147461, 0.0039953, 0.0146394, -0.0082136, 0.0082133
7: -0.0223336, -0.0105167, -0.0222499, -0.0106246, -0.0093760, 0.0094614
8: 0.0067165, 0.0185066, 0.0068386, 0.0183949, -0.0085549, 0.0085587
9: 0.9020081, 0.9528793, 0.9024385, 0.9526645, -0.0327995, 0.0324987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041786, -0.0010474, -0.0041786, -0.0010474, -0.0031312, 0.0031312
1: 0.0167330, 0.0318992, 0.0167330, 0.0318992, -0.0093299, 0.0093299
2: 0.0198077, 0.0301678, 0.0198077, 0.0301678, -0.0066652, 0.0066652
3: 0.0051897, 0.0168701, 0.0051897, 0.0168701, -0.0083826, 0.0083826
4: -0.0176768, -0.0063450, -0.0176768, -0.0063450, -0.0085758, 0.0085758
5: 0.0113469, 0.0257674, 0.0113469, 0.0257674, -0.0110295, 0.0110295
6: 0.0039222, 0.0147461, 0.0039222, 0.0147461, -0.0082931, 0.0082931
7: -0.0223336, -0.0105167, -0.0223336, -0.0105167, -0.0095537, 0.0095537
8: 0.0067165, 0.0185066, 0.0067165, 0.0185066, -0.0086237, 0.0086237
9: 0.9020081, 0.9528793, 0.9020081, 0.9528793, -0.0327547, 0.0327547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0191285, upper bound: 0.0189615
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0193549, upper bound: 0.0193352
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0191285, upper bound: 0.0189615
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0193549, upper bound: 0.0193352
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0191074, upper bound: 0.0190427
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0193352, upper bound: 0.0193999
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0191074, upper bound: 0.0190428
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0193352, upper bound: 0.0193999
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0190837, upper bound: 0.0199844
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0193072, upper bound: 0.0204331
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0190837, upper bound: 0.0199844
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0193072, upper bound: 0.0204331
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0190488, upper bound: 0.0200807
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0192672, upper bound: 0.0205040
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0190488, upper bound: 0.0200807
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0192672, upper bound: 0.0205040
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204636, upper bound: 0.0192672
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204636, upper bound: 0.0192672
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204637, upper bound: 0.0192672
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202341, upper bound: 0.0189386
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204637, upper bound: 0.0192672
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0202139, upper bound: 0.0190297
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 9, lower bound: -0.0204331, upper bound: 0.0193659

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0040627, -0.0012707, -0.0028017, 0.0028261
1: 0.0181921, 0.0316567, 0.0180546, 0.0314697, -0.0081037, 0.0083929
2: 0.0206704, 0.0299115, 0.0205295, 0.0297560, -0.0057136, 0.0060155
3: 0.0063097, 0.0165808, 0.0062790, 0.0164428, -0.0074547, 0.0076975
4: -0.0173488, -0.0074310, -0.0172196, -0.0073582, -0.0080490, 0.0077830
5: 0.0127622, 0.0253943, 0.0127335, 0.0252544, -0.0099419, 0.0101921
6: 0.0049685, 0.0144403, 0.0049314, 0.0143197, -0.0074823, 0.0077121
7: -0.0220428, -0.0117536, -0.0218988, -0.0116879, -0.0088631, 0.0086288
8: 0.0077874, 0.0182481, 0.0077971, 0.0181385, -0.0076392, 0.0078053
9: 0.9032502, 0.9480295, 0.9038427, 0.9484143, -0.0302514, 0.0290502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0190031
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0190031
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0040772, -0.0012470, -0.0028227, 0.0028174
1: 0.0180098, 0.0314699, 0.0179339, 0.0314702, -0.0082355, 0.0083809
2: 0.0205092, 0.0297685, 0.0204639, 0.0297823, -0.0058727, 0.0059778
3: 0.0062548, 0.0164433, 0.0062061, 0.0164440, -0.0076117, 0.0076745
4: -0.0172217, -0.0073388, -0.0172247, -0.0072983, -0.0080055, 0.0079514
5: 0.0127029, 0.0252562, 0.0126484, 0.0252586, -0.0100915, 0.0101625
6: 0.0049108, 0.0143212, 0.0048708, 0.0143232, -0.0076318, 0.0076797
7: -0.0219011, -0.0116561, -0.0219041, -0.0116030, -0.0088458, 0.0087693
8: 0.0077700, 0.0181385, 0.0077234, 0.0181385, -0.0077482, 0.0078147
9: 0.9038395, 0.9485210, 0.9038353, 0.9487275, -0.0300928, 0.0297948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0191401
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0193941
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0040967, -0.0011893, -0.0028831, 0.0028600
1: 0.0181921, 0.0316567, 0.0179470, 0.0315548, -0.0084172, 0.0086385
2: 0.0206704, 0.0299115, 0.0205164, 0.0298633, -0.0059434, 0.0060505
3: 0.0063097, 0.0165808, 0.0061756, 0.0165655, -0.0076216, 0.0078410
4: -0.0173488, -0.0074310, -0.0173139, -0.0073166, -0.0081257, 0.0078642
5: 0.0127622, 0.0253943, 0.0126049, 0.0253659, -0.0100711, 0.0103785
6: 0.0049685, 0.0144403, 0.0048607, 0.0144265, -0.0075817, 0.0078206
7: -0.0220428, -0.0117536, -0.0219848, -0.0115826, -0.0090216, 0.0086912
8: 0.0077874, 0.0182481, 0.0076598, 0.0182502, -0.0078380, 0.0080034
9: 0.9032502, 0.9480295, 0.9034411, 0.9486275, -0.0306827, 0.0297366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0189615
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0189615
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041110, -0.0011597, -0.0029100, 0.0028512
1: 0.0180098, 0.0314699, 0.0178279, 0.0315554, -0.0085456, 0.0086251
2: 0.0205092, 0.0297685, 0.0204509, 0.0298892, -0.0061080, 0.0060122
3: 0.0062548, 0.0164433, 0.0061027, 0.0165667, -0.0077785, 0.0078177
4: -0.0172217, -0.0073388, -0.0173189, -0.0072566, -0.0080810, 0.0080346
5: 0.0127029, 0.0252562, 0.0125205, 0.0253701, -0.0102216, 0.0103484
6: 0.0049108, 0.0143212, 0.0048006, 0.0144299, -0.0077316, 0.0077868
7: -0.0219011, -0.0116561, -0.0219900, -0.0114976, -0.0090048, 0.0088284
8: 0.0077700, 0.0181385, 0.0075862, 0.0182502, -0.0079474, 0.0080122
9: 0.9038395, 0.9485210, 0.9034339, 0.9489397, -0.0305124, 0.0304842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0191074
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0193352
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0040627, -0.0012707, -0.0028368, 0.0029135
1: 0.0180860, 0.0317525, 0.0180546, 0.0314697, -0.0083487, 0.0086889
2: 0.0206564, 0.0300202, 0.0205295, 0.0297560, -0.0057533, 0.0062523
3: 0.0062101, 0.0167013, 0.0062790, 0.0164428, -0.0076013, 0.0078536
4: -0.0174401, -0.0073896, -0.0172196, -0.0073582, -0.0081347, 0.0078589
5: 0.0126355, 0.0255008, 0.0127335, 0.0252544, -0.0101242, 0.0103167
6: 0.0048972, 0.0145426, 0.0049314, 0.0143197, -0.0075913, 0.0078093
7: -0.0221280, -0.0116488, -0.0218988, -0.0116879, -0.0089346, 0.0087824
8: 0.0076520, 0.0183602, 0.0077971, 0.0181385, -0.0078327, 0.0080046
9: 0.9028240, 0.9482442, 0.9038427, 0.9484143, -0.0309108, 0.0294868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190427
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190427
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0040772, -0.0012470, -0.0028565, 0.0029014
1: 0.0179049, 0.0315551, 0.0179339, 0.0314702, -0.0084787, 0.0086944
2: 0.0204964, 0.0298754, 0.0204639, 0.0297823, -0.0059084, 0.0062074
3: 0.0061511, 0.0165660, 0.0062061, 0.0164440, -0.0077553, 0.0078422
4: -0.0173159, -0.0072973, -0.0172247, -0.0072983, -0.0080869, 0.0080271
5: 0.0125759, 0.0253676, 0.0126484, 0.0252586, -0.0102771, 0.0102918
6: 0.0048413, 0.0144279, 0.0048708, 0.0143232, -0.0077386, 0.0077793
7: -0.0219870, -0.0115511, -0.0219041, -0.0116030, -0.0089084, 0.0089257
8: 0.0076329, 0.0182502, 0.0077234, 0.0181385, -0.0079459, 0.0080120
9: 0.9034380, 0.9487324, 0.9038353, 0.9487275, -0.0307794, 0.0302160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0191777
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0193999
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0040967, -0.0011893, -0.0029182, 0.0029474
1: 0.0180860, 0.0317525, 0.0179470, 0.0315548, -0.0084165, 0.0087106
2: 0.0206564, 0.0300202, 0.0205164, 0.0298633, -0.0059111, 0.0062149
3: 0.0062101, 0.0167013, 0.0061756, 0.0165655, -0.0076691, 0.0079142
4: -0.0174401, -0.0073896, -0.0173139, -0.0073166, -0.0082077, 0.0079369
5: 0.0126355, 0.0255008, 0.0126049, 0.0253659, -0.0102194, 0.0104772
6: 0.0048972, 0.0145426, 0.0048607, 0.0144265, -0.0076678, 0.0079014
7: -0.0221280, -0.0116488, -0.0219848, -0.0115826, -0.0091066, 0.0088680
8: 0.0076520, 0.0183602, 0.0076598, 0.0182502, -0.0079056, 0.0080784
9: 0.9028240, 0.9482442, 0.9034411, 0.9486275, -0.0309290, 0.0297053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190427
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190428
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041110, -0.0011597, -0.0029437, 0.0029352
1: 0.0179049, 0.0315551, 0.0178279, 0.0315554, -0.0085592, 0.0086963
2: 0.0204964, 0.0298754, 0.0204509, 0.0298892, -0.0060732, 0.0061798
3: 0.0061511, 0.0165660, 0.0061027, 0.0165667, -0.0078287, 0.0078923
4: -0.0173159, -0.0072973, -0.0173189, -0.0072566, -0.0081642, 0.0081094
5: 0.0125759, 0.0253676, 0.0125205, 0.0253701, -0.0103767, 0.0104486
6: 0.0048413, 0.0144279, 0.0048006, 0.0144299, -0.0078204, 0.0078688
7: -0.0219870, -0.0115511, -0.0219900, -0.0114976, -0.0090892, 0.0090126
8: 0.0076329, 0.0182502, 0.0075862, 0.0182502, -0.0080211, 0.0080896
9: 0.9034380, 0.9487324, 0.9034339, 0.9489397, -0.0307752, 0.0304741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0191777
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0193999
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041308, -0.0011795, -0.0028929, 0.0028941
1: 0.0181921, 0.0316567, 0.0169541, 0.0318147, -0.0089062, 0.0096466
2: 0.0206704, 0.0299115, 0.0198807, 0.0300338, -0.0062813, 0.0068633
3: 0.0063097, 0.0165808, 0.0053596, 0.0167495, -0.0078471, 0.0085765
4: -0.0173488, -0.0074310, -0.0175709, -0.0064557, -0.0088782, 0.0080902
5: 0.0127622, 0.0253943, 0.0115628, 0.0256311, -0.0103667, 0.0111994
6: 0.0049685, 0.0144403, 0.0040585, 0.0146361, -0.0077723, 0.0084903
7: -0.0220428, -0.0117536, -0.0222451, -0.0107088, -0.0096463, 0.0089408
8: 0.0077874, 0.0182481, 0.0069159, 0.0183949, -0.0080357, 0.0086519
9: 0.9032502, 0.9480295, 0.9024454, 0.9523534, -0.0346491, 0.0310909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0200259
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0200259
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041448, -0.0011492, -0.0029205, 0.0028850
1: 0.0180098, 0.0314699, 0.0168357, 0.0318153, -0.0090395, 0.0096312
2: 0.0205092, 0.0297685, 0.0198172, 0.0300587, -0.0064520, 0.0068242
3: 0.0062548, 0.0164433, 0.0052827, 0.0167506, -0.0080040, 0.0085560
4: -0.0172217, -0.0073388, -0.0175758, -0.0063936, -0.0088338, 0.0082577
5: 0.0127029, 0.0252562, 0.0114739, 0.0256350, -0.0105161, 0.0111689
6: 0.0049108, 0.0143212, 0.0039953, 0.0146394, -0.0079224, 0.0084565
7: -0.0219011, -0.0116561, -0.0222499, -0.0106246, -0.0096292, 0.0090738
8: 0.0077700, 0.0181385, 0.0068386, 0.0183949, -0.0081446, 0.0086653
9: 0.9038395, 0.9485210, 0.9024385, 0.9526645, -0.0344903, 0.0318399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0202360
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0204833
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041645, -0.0010811, -0.0029913, 0.0029279
1: 0.0181921, 0.0316567, 0.0168491, 0.0318986, -0.0090330, 0.0098146
2: 0.0206704, 0.0299115, 0.0198708, 0.0301433, -0.0064535, 0.0068750
3: 0.0063097, 0.0165808, 0.0052640, 0.0168690, -0.0079573, 0.0086682
4: -0.0173488, -0.0074310, -0.0176721, -0.0064061, -0.0089165, 0.0081609
5: 0.0127622, 0.0253943, 0.0114342, 0.0257634, -0.0104740, 0.0113336
6: 0.0049685, 0.0144403, 0.0039843, 0.0147427, -0.0078507, 0.0085543
7: -0.0220428, -0.0117536, -0.0223288, -0.0105996, -0.0097758, 0.0089854
8: 0.0077874, 0.0182481, 0.0067908, 0.0185066, -0.0081597, 0.0087900
9: 0.9032502, 0.9480295, 0.9020146, 0.9525682, -0.0348276, 0.0315806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0199843
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0199843
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041786, -0.0010474, -0.0030223, 0.0029188
1: 0.0180098, 0.0314699, 0.0167330, 0.0318992, -0.0091748, 0.0097977
2: 0.0205092, 0.0297685, 0.0198077, 0.0301678, -0.0066316, 0.0068339
3: 0.0062548, 0.0164433, 0.0051897, 0.0168701, -0.0081141, 0.0086453
4: -0.0172217, -0.0073388, -0.0176768, -0.0063450, -0.0088719, 0.0083274
5: 0.0127029, 0.0252562, 0.0113469, 0.0257674, -0.0106234, 0.0113018
6: 0.0049108, 0.0143212, 0.0039222, 0.0147461, -0.0080009, 0.0085200
7: -0.0219011, -0.0116561, -0.0223336, -0.0105167, -0.0097581, 0.0091158
8: 0.0077700, 0.0181385, 0.0067165, 0.0185066, -0.0082685, 0.0088033
9: 0.9038395, 0.9485210, 0.9020081, 0.9528793, -0.0346646, 0.0323242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0202139
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0204331
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041308, -0.0011795, -0.0029280, 0.0029815
1: 0.0180860, 0.0317525, 0.0169541, 0.0318147, -0.0091512, 0.0099426
2: 0.0206564, 0.0300202, 0.0198807, 0.0300338, -0.0063211, 0.0071001
3: 0.0062101, 0.0167013, 0.0053596, 0.0167495, -0.0079937, 0.0087326
4: -0.0174401, -0.0073896, -0.0175709, -0.0064557, -0.0089640, 0.0081662
5: 0.0126355, 0.0255008, 0.0115628, 0.0256311, -0.0105489, 0.0113241
6: 0.0048972, 0.0145426, 0.0040585, 0.0146361, -0.0078814, 0.0085876
7: -0.0221280, -0.0116488, -0.0222451, -0.0107088, -0.0097178, 0.0090944
8: 0.0076520, 0.0183602, 0.0069159, 0.0183949, -0.0082292, 0.0088512
9: 0.9028240, 0.9482442, 0.9024454, 0.9523534, -0.0353085, 0.0315274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041448, -0.0011492, -0.0029543, 0.0029690
1: 0.0179049, 0.0315551, 0.0168357, 0.0318153, -0.0092826, 0.0099448
2: 0.0204964, 0.0298754, 0.0198172, 0.0300587, -0.0064877, 0.0070537
3: 0.0061511, 0.0165660, 0.0052827, 0.0167506, -0.0081476, 0.0087237
4: -0.0173159, -0.0072973, -0.0175758, -0.0063936, -0.0089152, 0.0083333
5: 0.0125759, 0.0253676, 0.0114739, 0.0256350, -0.0107018, 0.0112982
6: 0.0048413, 0.0144279, 0.0039953, 0.0146394, -0.0080292, 0.0085561
7: -0.0219870, -0.0115511, -0.0222499, -0.0106246, -0.0096919, 0.0092302
8: 0.0076329, 0.0182502, 0.0068386, 0.0183949, -0.0083423, 0.0088626
9: 0.9034380, 0.9487324, 0.9024385, 0.9526645, -0.0351769, 0.0322611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0202795
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0205040
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041645, -0.0010811, -0.0030264, 0.0030153
1: 0.0180860, 0.0317525, 0.0168491, 0.0318986, -0.0091919, 0.0099307
2: 0.0206564, 0.0300202, 0.0198708, 0.0301433, -0.0064623, 0.0070545
3: 0.0062101, 0.0167013, 0.0052640, 0.0168690, -0.0080590, 0.0087747
4: -0.0174401, -0.0073896, -0.0176721, -0.0064061, -0.0090302, 0.0082501
5: 0.0126355, 0.0255008, 0.0114342, 0.0257634, -0.0106486, 0.0114514
6: 0.0048972, 0.0145426, 0.0039843, 0.0147427, -0.0079621, 0.0086630
7: -0.0221280, -0.0116488, -0.0223288, -0.0105996, -0.0098621, 0.0091862
8: 0.0076520, 0.0183602, 0.0067908, 0.0185066, -0.0082939, 0.0088980
9: 0.9028240, 0.9482442, 0.9020146, 0.9525682, -0.0352467, 0.0317426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041786, -0.0010474, -0.0030561, 0.0030028
1: 0.0179049, 0.0315551, 0.0167330, 0.0318992, -0.0093272, 0.0099109
2: 0.0204964, 0.0298754, 0.0198077, 0.0301678, -0.0066339, 0.0070158
3: 0.0061511, 0.0165660, 0.0051897, 0.0168701, -0.0082185, 0.0087543
4: -0.0173159, -0.0072973, -0.0176768, -0.0063450, -0.0089850, 0.0084214
5: 0.0125759, 0.0253676, 0.0113469, 0.0257674, -0.0108056, 0.0114196
6: 0.0048413, 0.0144279, 0.0039222, 0.0147461, -0.0081152, 0.0086292
7: -0.0219870, -0.0115511, -0.0223336, -0.0105167, -0.0098444, 0.0093236
8: 0.0076329, 0.0182502, 0.0067165, 0.0185066, -0.0084094, 0.0089106
9: 0.9034380, 0.9487324, 0.9020081, 0.9528793, -0.0350882, 0.0325161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0202795
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0205040
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0040627, -0.0012707, -0.0028706, 0.0029302
1: 0.0170758, 0.0320063, 0.0180546, 0.0314697, -0.0093674, 0.0091766
2: 0.0200154, 0.0301909, 0.0205295, 0.0297560, -0.0065718, 0.0065963
3: 0.0053988, 0.0168827, 0.0062790, 0.0164428, -0.0083612, 0.0080632
4: -0.0176915, -0.0065340, -0.0172196, -0.0073582, -0.0083343, 0.0086350
5: 0.0115979, 0.0257699, 0.0127335, 0.0252544, -0.0109767, 0.0105842
6: 0.0041013, 0.0147464, 0.0049314, 0.0143197, -0.0082851, 0.0079737
7: -0.0223849, -0.0107666, -0.0218988, -0.0116879, -0.0091603, 0.0094256
8: 0.0069091, 0.0185007, 0.0077971, 0.0181385, -0.0085168, 0.0081841
9: 0.9018377, 0.9519727, 0.9038427, 0.9484143, -0.0322148, 0.0335120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0040772, -0.0012470, -0.0028905, 0.0029117
1: 0.0169144, 0.0318150, 0.0179339, 0.0314702, -0.0094733, 0.0091833
2: 0.0198633, 0.0300454, 0.0204639, 0.0297823, -0.0067170, 0.0065425
3: 0.0053300, 0.0167500, 0.0062061, 0.0164440, -0.0084981, 0.0080677
4: -0.0175729, -0.0064338, -0.0172247, -0.0072983, -0.0083131, 0.0087814
5: 0.0115280, 0.0256327, 0.0126484, 0.0252586, -0.0111004, 0.0105875
6: 0.0040348, 0.0146375, 0.0048708, 0.0143232, -0.0084136, 0.0079703
7: -0.0222471, -0.0106774, -0.0219041, -0.0116030, -0.0091581, 0.0095444
8: 0.0068849, 0.0183949, 0.0077234, 0.0181385, -0.0085999, 0.0082121
9: 0.9024426, 0.9524551, 0.9038353, 0.9487275, -0.0321337, 0.0341995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0190934
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0193359
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0040967, -0.0011893, -0.0029520, 0.0029641
1: 0.0170758, 0.0320063, 0.0179470, 0.0315548, -0.0096809, 0.0094222
2: 0.0200154, 0.0301909, 0.0205164, 0.0298633, -0.0068017, 0.0066313
3: 0.0053988, 0.0168827, 0.0061756, 0.0165655, -0.0085281, 0.0082067
4: -0.0176915, -0.0065340, -0.0173139, -0.0073166, -0.0084110, 0.0087162
5: 0.0115979, 0.0257699, 0.0126049, 0.0253659, -0.0111059, 0.0107706
6: 0.0041013, 0.0147464, 0.0048607, 0.0144265, -0.0083846, 0.0080823
7: -0.0223849, -0.0107666, -0.0219848, -0.0115826, -0.0093187, 0.0094881
8: 0.0069091, 0.0185007, 0.0076598, 0.0182502, -0.0087156, 0.0083821
9: 0.9018377, 0.9519727, 0.9034411, 0.9486275, -0.0326462, 0.0341984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041110, -0.0011597, -0.0029777, 0.0029455
1: 0.0169144, 0.0318150, 0.0178279, 0.0315554, -0.0097834, 0.0094276
2: 0.0198633, 0.0300454, 0.0204509, 0.0298892, -0.0069523, 0.0065769
3: 0.0053300, 0.0167500, 0.0061027, 0.0165667, -0.0086649, 0.0082108
4: -0.0175729, -0.0064338, -0.0173189, -0.0072566, -0.0083886, 0.0088645
5: 0.0115280, 0.0256327, 0.0125205, 0.0253701, -0.0112305, 0.0107735
6: 0.0040348, 0.0146375, 0.0048006, 0.0144299, -0.0085134, 0.0080773
7: -0.0222471, -0.0106774, -0.0219900, -0.0114976, -0.0093171, 0.0096035
8: 0.0068849, 0.0183949, 0.0075862, 0.0182502, -0.0087991, 0.0084096
9: 0.9024426, 0.9524551, 0.9034339, 0.9489397, -0.0325534, 0.0348890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0190488
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0192672
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0040627, -0.0012707, -0.0029054, 0.0030303
1: 0.0169682, 0.0321007, 0.0180546, 0.0314697, -0.0095337, 0.0093206
2: 0.0200045, 0.0303035, 0.0205295, 0.0297560, -0.0065858, 0.0067760
3: 0.0053004, 0.0169994, 0.0062790, 0.0164428, -0.0084537, 0.0081712
4: -0.0177966, -0.0064833, -0.0172196, -0.0073582, -0.0084002, 0.0086732
5: 0.0114692, 0.0259043, 0.0127335, 0.0252544, -0.0111090, 0.0106937
6: 0.0040265, 0.0148571, 0.0049314, 0.0143197, -0.0083498, 0.0080541
7: -0.0224730, -0.0106558, -0.0218988, -0.0116879, -0.0091989, 0.0095517
8: 0.0067799, 0.0186110, 0.0077971, 0.0181385, -0.0086548, 0.0083061
9: 0.9013865, 0.9521797, 0.9038427, 0.9484143, -0.0326933, 0.0336948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0040772, -0.0012470, -0.0029241, 0.0030117
1: 0.0168123, 0.0318989, 0.0179339, 0.0314702, -0.0096394, 0.0093102
2: 0.0198550, 0.0301546, 0.0204639, 0.0297823, -0.0067290, 0.0067150
3: 0.0052378, 0.0168694, 0.0062061, 0.0164440, -0.0085876, 0.0081782
4: -0.0176739, -0.0063858, -0.0172247, -0.0072983, -0.0083839, 0.0088187
5: 0.0114008, 0.0257650, 0.0126484, 0.0252586, -0.0112329, 0.0106949
6: 0.0039620, 0.0147440, 0.0048708, 0.0143232, -0.0084762, 0.0080487
7: -0.0223307, -0.0105696, -0.0219041, -0.0116030, -0.0092028, 0.0096721
8: 0.0067627, 0.0185066, 0.0077234, 0.0181385, -0.0087377, 0.0083359
9: 0.9020119, 0.9526696, 0.9038353, 0.9487275, -0.0326236, 0.0343722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0040967, -0.0011893, -0.0029868, 0.0030642
1: 0.0169682, 0.0321007, 0.0179470, 0.0315548, -0.0096474, 0.0094641
2: 0.0200045, 0.0303035, 0.0205164, 0.0298633, -0.0067617, 0.0067779
3: 0.0053004, 0.0169994, 0.0061756, 0.0165655, -0.0085563, 0.0082771
4: -0.0177966, -0.0064833, -0.0173139, -0.0073166, -0.0084978, 0.0087836
5: 0.0114692, 0.0259043, 0.0126049, 0.0253659, -0.0112271, 0.0108735
6: 0.0040265, 0.0148571, 0.0048607, 0.0144265, -0.0084549, 0.0081669
7: -0.0224730, -0.0106558, -0.0219848, -0.0115826, -0.0094103, 0.0096400
8: 0.0067799, 0.0186110, 0.0076598, 0.0182502, -0.0087611, 0.0084491
9: 0.9013865, 0.9521797, 0.9034411, 0.9486275, -0.0328953, 0.0341026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041110, -0.0011597, -0.0030113, 0.0030455
1: 0.0168123, 0.0318989, 0.0178279, 0.0315554, -0.0097615, 0.0094717
2: 0.0198550, 0.0301546, 0.0204509, 0.0298892, -0.0069097, 0.0067285
3: 0.0052378, 0.0168694, 0.0061027, 0.0165667, -0.0086950, 0.0082828
4: -0.0176739, -0.0063858, -0.0173189, -0.0072566, -0.0084777, 0.0089318
5: 0.0114008, 0.0257650, 0.0125205, 0.0253701, -0.0113507, 0.0108780
6: 0.0039620, 0.0147440, 0.0048006, 0.0144299, -0.0085854, 0.0081635
7: -0.0223307, -0.0105696, -0.0219900, -0.0114976, -0.0094077, 0.0097587
8: 0.0067627, 0.0185066, 0.0075862, 0.0182502, -0.0088435, 0.0084781
9: 0.9020119, 0.9526696, 0.9034339, 0.9489397, -0.0328127, 0.0347950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041308, -0.0011795, -0.0029618, 0.0029982
1: 0.0170758, 0.0320063, 0.0169541, 0.0318147, -0.0087428, 0.0090354
2: 0.0200154, 0.0301909, 0.0198807, 0.0300338, -0.0061747, 0.0064811
3: 0.0053988, 0.0168827, 0.0053596, 0.0167495, -0.0079562, 0.0082148
4: -0.0176915, -0.0065340, -0.0175709, -0.0064557, -0.0084701, 0.0081959
5: 0.0115979, 0.0257699, 0.0115628, 0.0256311, -0.0105324, 0.0107931
6: 0.0041013, 0.0147464, 0.0040585, 0.0146361, -0.0079109, 0.0081509
7: -0.0223849, -0.0107666, -0.0222451, -0.0107088, -0.0093332, 0.0090912
8: 0.0069091, 0.0185007, 0.0069159, 0.0183949, -0.0081887, 0.0083637
9: 0.9018377, 0.9519727, 0.9024454, 0.9523534, -0.0323391, 0.0310523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041448, -0.0011492, -0.0029883, 0.0029793
1: 0.0169144, 0.0318150, 0.0168357, 0.0318153, -0.0088644, 0.0090126
2: 0.0198633, 0.0300454, 0.0198172, 0.0300587, -0.0063286, 0.0064396
3: 0.0053300, 0.0167500, 0.0052827, 0.0167506, -0.0081114, 0.0081737
4: -0.0175729, -0.0064338, -0.0175758, -0.0063936, -0.0084187, 0.0083651
5: 0.0115280, 0.0256327, 0.0114739, 0.0256350, -0.0106814, 0.0107506
6: 0.0040348, 0.0146375, 0.0039953, 0.0146394, -0.0080608, 0.0081082
7: -0.0222471, -0.0106774, -0.0222499, -0.0106246, -0.0093076, 0.0092290
8: 0.0068849, 0.0183949, 0.0068386, 0.0183949, -0.0082973, 0.0083615
9: 0.9024426, 0.9524551, 0.9024385, 0.9526645, -0.0321003, 0.0318136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0190934
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0193359
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041645, -0.0010811, -0.0030602, 0.0030320
1: 0.0170758, 0.0320063, 0.0168491, 0.0318986, -0.0090550, 0.0092748
2: 0.0200154, 0.0301909, 0.0198708, 0.0301433, -0.0064062, 0.0065159
3: 0.0053988, 0.0168827, 0.0052640, 0.0168690, -0.0081234, 0.0083552
4: -0.0176915, -0.0065340, -0.0176721, -0.0064061, -0.0085450, 0.0082818
5: 0.0115979, 0.0257699, 0.0114342, 0.0257634, -0.0106630, 0.0109743
6: 0.0041013, 0.0147464, 0.0039843, 0.0147427, -0.0080149, 0.0082576
7: -0.0223849, -0.0107666, -0.0223288, -0.0105996, -0.0094862, 0.0091580
8: 0.0069091, 0.0185007, 0.0067908, 0.0185066, -0.0083824, 0.0085564
9: 0.9018377, 0.9519727, 0.9020146, 0.9525682, -0.0327505, 0.0317490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041786, -0.0010474, -0.0030901, 0.0030131
1: 0.0169144, 0.0318150, 0.0167330, 0.0318992, -0.0091701, 0.0092492
2: 0.0198633, 0.0300454, 0.0198077, 0.0301678, -0.0065638, 0.0064731
3: 0.0053300, 0.0167500, 0.0051897, 0.0168701, -0.0082787, 0.0083129
4: -0.0175729, -0.0064338, -0.0176768, -0.0063450, -0.0084930, 0.0084537
5: 0.0115280, 0.0256327, 0.0113469, 0.0257674, -0.0108126, 0.0109300
6: 0.0040348, 0.0146375, 0.0039222, 0.0147461, -0.0081649, 0.0082126
7: -0.0222471, -0.0106774, -0.0223336, -0.0105167, -0.0094601, 0.0092959
8: 0.0068849, 0.0183949, 0.0067165, 0.0185066, -0.0084916, 0.0085515
9: 0.9024426, 0.9524551, 0.9020081, 0.9528793, -0.0324964, 0.0325070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0190488
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0192672
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041308, -0.0011795, -0.0029966, 0.0030983
1: 0.0169682, 0.0321007, 0.0169541, 0.0318147, -0.0089788, 0.0093236
2: 0.0200045, 0.0303035, 0.0198807, 0.0300338, -0.0062138, 0.0067183
3: 0.0053004, 0.0169994, 0.0053596, 0.0167495, -0.0081011, 0.0083720
4: -0.0177966, -0.0064833, -0.0175709, -0.0064557, -0.0085624, 0.0082720
5: 0.0114692, 0.0259043, 0.0115628, 0.0256311, -0.0107115, 0.0109227
6: 0.0040265, 0.0148571, 0.0040585, 0.0146361, -0.0080198, 0.0082528
7: -0.0224730, -0.0106558, -0.0222451, -0.0107088, -0.0094089, 0.0092426
8: 0.0067799, 0.0186110, 0.0069159, 0.0183949, -0.0083803, 0.0085604
9: 0.9013865, 0.9521797, 0.9024454, 0.9523534, -0.0330049, 0.0314852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041448, -0.0011492, -0.0030219, 0.0030793
1: 0.0168123, 0.0318989, 0.0168357, 0.0318153, -0.0090987, 0.0093247
2: 0.0198550, 0.0301546, 0.0198172, 0.0300587, -0.0063635, 0.0066706
3: 0.0052378, 0.0168694, 0.0052827, 0.0167506, -0.0082514, 0.0083410
4: -0.0176739, -0.0063858, -0.0175758, -0.0063936, -0.0085047, 0.0084390
5: 0.0114008, 0.0257650, 0.0114739, 0.0256350, -0.0108609, 0.0108814
6: 0.0039620, 0.0147440, 0.0039953, 0.0146394, -0.0081648, 0.0082122
7: -0.0223307, -0.0105696, -0.0222499, -0.0106246, -0.0093745, 0.0093801
8: 0.0067627, 0.0185066, 0.0068386, 0.0183949, -0.0084885, 0.0085542
9: 0.9020119, 0.9526696, 0.9024385, 0.9526645, -0.0327971, 0.0322134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041645, -0.0010811, -0.0030950, 0.0031321
1: 0.0169682, 0.0321007, 0.0168491, 0.0318986, -0.0090547, 0.0093491
2: 0.0200045, 0.0303035, 0.0198708, 0.0301433, -0.0063750, 0.0066841
3: 0.0053004, 0.0169994, 0.0052640, 0.0168690, -0.0081640, 0.0084236
4: -0.0177966, -0.0064833, -0.0176721, -0.0064061, -0.0086245, 0.0083478
5: 0.0114692, 0.0259043, 0.0114342, 0.0257634, -0.0108050, 0.0110707
6: 0.0040265, 0.0148571, 0.0039843, 0.0147427, -0.0080937, 0.0083360
7: -0.0224730, -0.0106558, -0.0223288, -0.0105996, -0.0095782, 0.0093341
8: 0.0067799, 0.0186110, 0.0067908, 0.0185066, -0.0084438, 0.0086227
9: 0.9013865, 0.9521797, 0.9020146, 0.9525682, -0.0329844, 0.0316914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041786, -0.0010474, -0.0031237, 0.0031130
1: 0.0168123, 0.0318989, 0.0167330, 0.0318992, -0.0091803, 0.0093281
2: 0.0198550, 0.0301546, 0.0198077, 0.0301678, -0.0065282, 0.0066430
3: 0.0052378, 0.0168694, 0.0051897, 0.0168701, -0.0083201, 0.0083822
4: -0.0176739, -0.0063858, -0.0176768, -0.0063450, -0.0085744, 0.0085201
5: 0.0114008, 0.0257650, 0.0113469, 0.0257674, -0.0109585, 0.0110282
6: 0.0039620, 0.0147440, 0.0039222, 0.0147461, -0.0082442, 0.0082920
7: -0.0223307, -0.0105696, -0.0223336, -0.0105167, -0.0095523, 0.0094739
8: 0.0067627, 0.0185066, 0.0067165, 0.0185066, -0.0085566, 0.0086207
9: 0.9020119, 0.9526696, 0.9020081, 0.9528793, -0.0327524, 0.0324659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.85 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0190031
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0190031
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0191401
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190031, upper bound: 0.0193941
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0189615
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0189615
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0191074
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190427, upper bound: 0.0193352
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190427
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190427
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0191777
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0193999
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190427
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0190428
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0191777
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189615, upper bound: 0.0193999
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0200259
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0200259
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0202360
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189868, upper bound: 0.0204833
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0199843
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0199843
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0202139
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0190297, upper bound: 0.0204331
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0202795
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0205040
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0200807
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0202795
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0189386, upper bound: 0.0205040
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0190934
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0193359
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0190488
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0192672
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0189868
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0190934
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200259, upper bound: 0.0193359
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0189386
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0190488
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0200807, upper bound: 0.0192672
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0190297
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0191424
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 9, lower bound: -0.0199843, upper bound: 0.0193659

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0040723, -0.0012367, -0.0028357, 0.0028357
1: 0.0181921, 0.0316567, 0.0181921, 0.0316567, -0.0082602, 0.0082602
2: 0.0206704, 0.0299115, 0.0206704, 0.0299115, -0.0058459, 0.0058459
3: 0.0063097, 0.0165808, 0.0063097, 0.0165808, -0.0075800, 0.0075800
4: -0.0173488, -0.0074310, -0.0173488, -0.0074310, -0.0079004, 0.0079004
5: 0.0127622, 0.0253943, 0.0127622, 0.0253943, -0.0100785, 0.0100785
6: 0.0049685, 0.0144403, 0.0049685, 0.0144403, -0.0075951, 0.0075951
7: -0.0220428, -0.0117536, -0.0220428, -0.0117536, -0.0087451, 0.0087451
8: 0.0077874, 0.0182481, 0.0077874, 0.0182481, -0.0077429, 0.0077429
9: 0.9032502, 0.9480295, 0.9032502, 0.9480295, -0.0295862, 0.0295862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163217, upper bound: 0.0166957
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163172
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0040697, -0.0012598, -0.0028125, 0.0028330
1: 0.0181921, 0.0316567, 0.0180098, 0.0314699, -0.0081052, 0.0084424
2: 0.0206704, 0.0299115, 0.0205092, 0.0297685, -0.0057333, 0.0060520
3: 0.0063097, 0.0165808, 0.0062548, 0.0164433, -0.0074534, 0.0077335
4: -0.0173488, -0.0074310, -0.0172217, -0.0073388, -0.0080768, 0.0077832
5: 0.0127622, 0.0253943, 0.0127029, 0.0252562, -0.0099422, 0.0102315
6: 0.0049685, 0.0144403, 0.0049108, 0.0143212, -0.0074821, 0.0077428
7: -0.0220428, -0.0117536, -0.0219011, -0.0116561, -0.0088980, 0.0086289
8: 0.0077874, 0.0182481, 0.0077700, 0.0181385, -0.0076359, 0.0078445
9: 0.9032502, 0.9480295, 0.9038395, 0.9485210, -0.0303921, 0.0290510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163217, upper bound: 0.0167021
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163417
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0040723, -0.0012367, -0.0028330, 0.0028125
1: 0.0180098, 0.0314699, 0.0181921, 0.0316567, -0.0084424, 0.0081052
2: 0.0205092, 0.0297685, 0.0206704, 0.0299115, -0.0060520, 0.0057333
3: 0.0062548, 0.0164433, 0.0063097, 0.0165808, -0.0077335, 0.0074534
4: -0.0172217, -0.0073388, -0.0173488, -0.0074310, -0.0077832, 0.0080768
5: 0.0127029, 0.0252562, 0.0127622, 0.0253943, -0.0102315, 0.0099422
6: 0.0049108, 0.0143212, 0.0049685, 0.0144403, -0.0077428, 0.0074821
7: -0.0219011, -0.0116561, -0.0220428, -0.0117536, -0.0086289, 0.0088980
8: 0.0077700, 0.0181385, 0.0077874, 0.0182481, -0.0078445, 0.0076359
9: 0.9038395, 0.9485210, 0.9032502, 0.9480295, -0.0290510, 0.0303921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0169104
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0166697
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0040697, -0.0012598, -0.0028099, 0.0028099
1: 0.0180098, 0.0314699, 0.0180098, 0.0314699, -0.0082336, 0.0082336
2: 0.0205092, 0.0297685, 0.0205092, 0.0297685, -0.0058507, 0.0058507
3: 0.0062548, 0.0164433, 0.0062548, 0.0164433, -0.0076114, 0.0076114
4: -0.0172217, -0.0073388, -0.0172217, -0.0073388, -0.0079502, 0.0079502
5: 0.0127029, 0.0252562, 0.0127029, 0.0252562, -0.0100904, 0.0100904
6: 0.0049108, 0.0143212, 0.0049108, 0.0143212, -0.0076310, 0.0076310
7: -0.0219011, -0.0116561, -0.0219011, -0.0116561, -0.0087681, 0.0087681
8: 0.0077700, 0.0181385, 0.0077700, 0.0181385, -0.0077471, 0.0077471
9: 0.9038395, 0.9485210, 0.9038395, 0.9485210, -0.0297927, 0.0297927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0172749
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0171460
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041075, -0.0011493, -0.0029231, 0.0028708
1: 0.0181921, 0.0316567, 0.0180860, 0.0317525, -0.0085563, 0.0085052
2: 0.0206704, 0.0299115, 0.0206564, 0.0300202, -0.0060827, 0.0058857
3: 0.0063097, 0.0165808, 0.0062101, 0.0167013, -0.0077360, 0.0077266
4: -0.0173488, -0.0074310, -0.0174401, -0.0073896, -0.0079764, 0.0079862
5: 0.0127622, 0.0253943, 0.0126355, 0.0255008, -0.0102031, 0.0102607
6: 0.0049685, 0.0144403, 0.0048972, 0.0145426, -0.0076923, 0.0077041
7: -0.0220428, -0.0117536, -0.0221280, -0.0116488, -0.0088988, 0.0088166
8: 0.0077874, 0.0182481, 0.0076520, 0.0183602, -0.0079422, 0.0079364
9: 0.9032502, 0.9480295, 0.9028240, 0.9482442, -0.0300227, 0.0302456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163001, upper bound: 0.0166137
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041035, -0.0011758, -0.0028966, 0.0028668
1: 0.0181921, 0.0316567, 0.0179049, 0.0315551, -0.0084187, 0.0086879
2: 0.0206704, 0.0299115, 0.0204964, 0.0298754, -0.0059628, 0.0060849
3: 0.0063097, 0.0165808, 0.0061511, 0.0165660, -0.0076199, 0.0078771
4: -0.0173488, -0.0074310, -0.0173159, -0.0072973, -0.0081518, 0.0078646
5: 0.0127622, 0.0253943, 0.0125759, 0.0253676, -0.0100715, 0.0104173
6: 0.0049685, 0.0144403, 0.0048413, 0.0144279, -0.0075817, 0.0078497
7: -0.0220428, -0.0117536, -0.0219870, -0.0115511, -0.0090574, 0.0086915
8: 0.0077874, 0.0182481, 0.0076329, 0.0182502, -0.0078354, 0.0080422
9: 0.9032502, 0.9480295, 0.9034380, 0.9487324, -0.0308118, 0.0297377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163001, upper bound: 0.0166208
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162530
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041075, -0.0011493, -0.0029204, 0.0028477
1: 0.0180098, 0.0314699, 0.0180860, 0.0317525, -0.0087384, 0.0083502
2: 0.0205092, 0.0297685, 0.0206564, 0.0300202, -0.0062888, 0.0057730
3: 0.0062548, 0.0164433, 0.0062101, 0.0167013, -0.0078896, 0.0076000
4: -0.0172217, -0.0073388, -0.0174401, -0.0073896, -0.0078591, 0.0081626
5: 0.0127029, 0.0252562, 0.0126355, 0.0255008, -0.0103561, 0.0101244
6: 0.0049108, 0.0143212, 0.0048972, 0.0145426, -0.0078401, 0.0075911
7: -0.0219011, -0.0116561, -0.0221280, -0.0116488, -0.0087825, 0.0089695
8: 0.0077700, 0.0181385, 0.0076520, 0.0183602, -0.0080438, 0.0078294
9: 0.9038395, 0.9485210, 0.9028240, 0.9482442, -0.0294875, 0.0310516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0168685
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0165952
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041035, -0.0011758, -0.0028939, 0.0028437
1: 0.0180098, 0.0314699, 0.0179049, 0.0315551, -0.0085437, 0.0084768
2: 0.0205092, 0.0297685, 0.0204964, 0.0298754, -0.0060860, 0.0058864
3: 0.0062548, 0.0164433, 0.0061511, 0.0165660, -0.0077790, 0.0077553
4: -0.0172217, -0.0073388, -0.0173159, -0.0072973, -0.0080258, 0.0080331
5: 0.0127029, 0.0252562, 0.0125759, 0.0253676, -0.0102203, 0.0102760
6: 0.0049108, 0.0143212, 0.0048413, 0.0144279, -0.0077306, 0.0077378
7: -0.0219011, -0.0116561, -0.0219870, -0.0115511, -0.0089245, 0.0088270
8: 0.0077700, 0.0181385, 0.0076329, 0.0182502, -0.0079444, 0.0079448
9: 0.9038395, 0.9485210, 0.9034380, 0.9487324, -0.0302139, 0.0304818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0172035
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0170494
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0040723, -0.0012367, -0.0028708, 0.0029231
1: 0.0180860, 0.0317525, 0.0181921, 0.0316567, -0.0085052, 0.0085563
2: 0.0206564, 0.0300202, 0.0206704, 0.0299115, -0.0058857, 0.0060827
3: 0.0062101, 0.0167013, 0.0063097, 0.0165808, -0.0077266, 0.0077360
4: -0.0174401, -0.0073896, -0.0173488, -0.0074310, -0.0079862, 0.0079764
5: 0.0126355, 0.0255008, 0.0127622, 0.0253943, -0.0102607, 0.0102031
6: 0.0048972, 0.0145426, 0.0049685, 0.0144403, -0.0077041, 0.0076923
7: -0.0221280, -0.0116488, -0.0220428, -0.0117536, -0.0088166, 0.0088988
8: 0.0076520, 0.0183602, 0.0077874, 0.0182481, -0.0079364, 0.0079422
9: 0.9028240, 0.9482442, 0.9032502, 0.9480295, -0.0302456, 0.0300227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167104
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162937
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0040697, -0.0012598, -0.0028477, 0.0029204
1: 0.0180860, 0.0317525, 0.0180098, 0.0314699, -0.0083502, 0.0087384
2: 0.0206564, 0.0300202, 0.0205092, 0.0297685, -0.0057730, 0.0062888
3: 0.0062101, 0.0167013, 0.0062548, 0.0164433, -0.0076000, 0.0078896
4: -0.0174401, -0.0073896, -0.0172217, -0.0073388, -0.0081626, 0.0078591
5: 0.0126355, 0.0255008, 0.0127029, 0.0252562, -0.0101244, 0.0103561
6: 0.0048972, 0.0145426, 0.0049108, 0.0143212, -0.0075911, 0.0078401
7: -0.0221280, -0.0116488, -0.0219011, -0.0116561, -0.0089695, 0.0087825
8: 0.0076520, 0.0183602, 0.0077700, 0.0181385, -0.0078294, 0.0080438
9: 0.9028240, 0.9482442, 0.9038395, 0.9485210, -0.0310516, 0.0294875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167289
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0163132
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0040723, -0.0012367, -0.0028668, 0.0028966
1: 0.0179049, 0.0315551, 0.0181921, 0.0316567, -0.0086879, 0.0084187
2: 0.0204964, 0.0298754, 0.0206704, 0.0299115, -0.0060849, 0.0059628
3: 0.0061511, 0.0165660, 0.0063097, 0.0165808, -0.0078771, 0.0076199
4: -0.0173159, -0.0072973, -0.0173488, -0.0074310, -0.0078646, 0.0081518
5: 0.0125759, 0.0253676, 0.0127622, 0.0253943, -0.0104173, 0.0100715
6: 0.0048413, 0.0144279, 0.0049685, 0.0144403, -0.0078497, 0.0075817
7: -0.0219870, -0.0115511, -0.0220428, -0.0117536, -0.0086915, 0.0090574
8: 0.0076329, 0.0182502, 0.0077874, 0.0182481, -0.0080422, 0.0078354
9: 0.9034380, 0.9487324, 0.9032502, 0.9480295, -0.0297377, 0.0308118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0169386
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0166364
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0040697, -0.0012598, -0.0028437, 0.0028939
1: 0.0179049, 0.0315551, 0.0180098, 0.0314699, -0.0084768, 0.0085437
2: 0.0204964, 0.0298754, 0.0205092, 0.0297685, -0.0058864, 0.0060860
3: 0.0061511, 0.0165660, 0.0062548, 0.0164433, -0.0077553, 0.0077790
4: -0.0173159, -0.0072973, -0.0172217, -0.0073388, -0.0080331, 0.0080258
5: 0.0125759, 0.0253676, 0.0127029, 0.0252562, -0.0102760, 0.0102204
6: 0.0048413, 0.0144279, 0.0049108, 0.0143212, -0.0077378, 0.0077306
7: -0.0219870, -0.0115511, -0.0219011, -0.0116561, -0.0088270, 0.0089245
8: 0.0076329, 0.0182502, 0.0077700, 0.0181385, -0.0079448, 0.0079444
9: 0.9034380, 0.9487324, 0.9038395, 0.9485210, -0.0304818, 0.0302139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0173066
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0171038
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041075, -0.0011493, -0.0029582, 0.0029582
1: 0.0180860, 0.0317525, 0.0180860, 0.0317525, -0.0085747, 0.0085747
2: 0.0206564, 0.0300202, 0.0206564, 0.0300202, -0.0060431, 0.0060431
3: 0.0062101, 0.0167013, 0.0062101, 0.0167013, -0.0077940, 0.0077940
4: -0.0174401, -0.0073896, -0.0174401, -0.0073896, -0.0080550, 0.0080550
5: 0.0126355, 0.0255008, 0.0126355, 0.0255008, -0.0103562, 0.0103562
6: 0.0048972, 0.0145426, 0.0048972, 0.0145426, -0.0077809, 0.0077809
7: -0.0221280, -0.0116488, -0.0221280, -0.0116488, -0.0089841, 0.0089841
8: 0.0076520, 0.0183602, 0.0076520, 0.0183602, -0.0080096, 0.0080096
9: 0.9028240, 0.9482442, 0.9028240, 0.9482442, -0.0302404, 0.0302404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167104
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162411
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041035, -0.0011758, -0.0029317, 0.0029542
1: 0.0180860, 0.0317525, 0.0179049, 0.0315551, -0.0084180, 0.0087592
2: 0.0206564, 0.0300202, 0.0204964, 0.0298754, -0.0059306, 0.0062532
3: 0.0062101, 0.0167013, 0.0061511, 0.0165660, -0.0076677, 0.0079503
4: -0.0174401, -0.0073896, -0.0173159, -0.0072973, -0.0082361, 0.0079371
5: 0.0126355, 0.0255008, 0.0125759, 0.0253676, -0.0102197, 0.0105174
6: 0.0048972, 0.0145426, 0.0048413, 0.0144279, -0.0076677, 0.0079316
7: -0.0221280, -0.0116488, -0.0219870, -0.0115511, -0.0091405, 0.0088682
8: 0.0076520, 0.0183602, 0.0076329, 0.0182502, -0.0079022, 0.0081178
9: 0.9028240, 0.9482442, 0.9034380, 0.9487324, -0.0310732, 0.0297061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167289
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162741
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041075, -0.0011493, -0.0029542, 0.0029317
1: 0.0179049, 0.0315551, 0.0180860, 0.0317525, -0.0087592, 0.0084180
2: 0.0204964, 0.0298754, 0.0206564, 0.0300202, -0.0062532, 0.0059306
3: 0.0061511, 0.0165660, 0.0062101, 0.0167013, -0.0079503, 0.0076677
4: -0.0173159, -0.0072973, -0.0174401, -0.0073896, -0.0079371, 0.0082361
5: 0.0125759, 0.0253676, 0.0126355, 0.0255008, -0.0105174, 0.0102197
6: 0.0048413, 0.0144279, 0.0048972, 0.0145426, -0.0079316, 0.0076677
7: -0.0219870, -0.0115511, -0.0221280, -0.0116488, -0.0088682, 0.0091405
8: 0.0076329, 0.0182502, 0.0076520, 0.0183602, -0.0081178, 0.0079022
9: 0.9034380, 0.9487324, 0.9028240, 0.9482442, -0.0297061, 0.0310732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0169386
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0166125
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041035, -0.0011758, -0.0029277, 0.0029277
1: 0.0179049, 0.0315551, 0.0179049, 0.0315551, -0.0085573, 0.0085573
2: 0.0204964, 0.0298754, 0.0204964, 0.0298754, -0.0060507, 0.0060507
3: 0.0061511, 0.0165660, 0.0061511, 0.0165660, -0.0078299, 0.0078299
4: -0.0173159, -0.0072973, -0.0173159, -0.0072973, -0.0081080, 0.0081080
5: 0.0125759, 0.0253676, 0.0125759, 0.0253676, -0.0103755, 0.0103755
6: 0.0048413, 0.0144279, 0.0048413, 0.0144279, -0.0078197, 0.0078197
7: -0.0219870, -0.0115511, -0.0219870, -0.0115511, -0.0090114, 0.0090114
8: 0.0076329, 0.0182502, 0.0076329, 0.0182502, -0.0080214, 0.0080214
9: 0.9034380, 0.9487324, 0.9034380, 0.9487324, -0.0304720, 0.0304720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0173066
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0170825
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041413, -0.0011326, -0.0029398, 0.0029046
1: 0.0181921, 0.0316567, 0.0170758, 0.0320063, -0.0090439, 0.0095240
2: 0.0206704, 0.0299115, 0.0200154, 0.0301909, -0.0064267, 0.0067041
3: 0.0063097, 0.0165808, 0.0053988, 0.0168827, -0.0079457, 0.0084864
4: -0.0173488, -0.0074310, -0.0176915, -0.0065340, -0.0087525, 0.0081857
5: 0.0127622, 0.0253943, 0.0115979, 0.0257699, -0.0104706, 0.0111133
6: 0.0049685, 0.0144403, 0.0041013, 0.0147464, -0.0078567, 0.0083979
7: -0.0220428, -0.0117536, -0.0223849, -0.0107666, -0.0095420, 0.0090423
8: 0.0077874, 0.0182481, 0.0069091, 0.0185007, -0.0081217, 0.0086205
9: 0.9032502, 0.9480295, 0.9018377, 0.9519727, -0.0340480, 0.0315497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163214, upper bound: 0.0178225
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172214
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041374, -0.0011655, -0.0029068, 0.0029008
1: 0.0181921, 0.0316567, 0.0169144, 0.0318150, -0.0089076, 0.0097024
2: 0.0206704, 0.0299115, 0.0198633, 0.0300454, -0.0062980, 0.0068962
3: 0.0063097, 0.0165808, 0.0053300, 0.0167500, -0.0078454, 0.0086200
4: -0.0173488, -0.0074310, -0.0175729, -0.0064338, -0.0089101, 0.0080908
5: 0.0127622, 0.0253943, 0.0115280, 0.0256327, -0.0103672, 0.0112455
6: 0.0049685, 0.0144403, 0.0040348, 0.0146375, -0.0077724, 0.0085247
7: -0.0220428, -0.0117536, -0.0222471, -0.0106774, -0.0096867, 0.0089412
8: 0.0077874, 0.0182481, 0.0068849, 0.0183949, -0.0080323, 0.0086962
9: 0.9032502, 0.9480295, 0.9024426, 0.9524551, -0.0348151, 0.0310920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163214, upper bound: 0.0178230
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172366
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041413, -0.0011326, -0.0029371, 0.0028815
1: 0.0180098, 0.0314699, 0.0170758, 0.0320063, -0.0092261, 0.0093689
2: 0.0205092, 0.0297685, 0.0200154, 0.0301909, -0.0066328, 0.0065915
3: 0.0062548, 0.0164433, 0.0053988, 0.0168827, -0.0080992, 0.0083599
4: -0.0172217, -0.0073388, -0.0176915, -0.0065340, -0.0086352, 0.0083621
5: 0.0127029, 0.0252562, 0.0115979, 0.0257699, -0.0106236, 0.0109770
6: 0.0049108, 0.0143212, 0.0041013, 0.0147464, -0.0080045, 0.0082850
7: -0.0219011, -0.0116561, -0.0223849, -0.0107666, -0.0094257, 0.0091952
8: 0.0077700, 0.0181385, 0.0069091, 0.0185007, -0.0082233, 0.0085136
9: 0.9038395, 0.9485210, 0.9018377, 0.9519727, -0.0335128, 0.0323556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0180716
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0176918
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041374, -0.0011655, -0.0029042, 0.0028776
1: 0.0180098, 0.0314699, 0.0169144, 0.0318150, -0.0090375, 0.0094714
2: 0.0205092, 0.0297685, 0.0198633, 0.0300454, -0.0064316, 0.0066950
3: 0.0062548, 0.0164433, 0.0053300, 0.0167500, -0.0080045, 0.0085012
4: -0.0172217, -0.0073388, -0.0175729, -0.0064338, -0.0087801, 0.0082561
5: 0.0127029, 0.0252562, 0.0115280, 0.0256327, -0.0105147, 0.0110992
6: 0.0049108, 0.0143212, 0.0040348, 0.0146375, -0.0079215, 0.0084134
7: -0.0219011, -0.0116561, -0.0222471, -0.0106774, -0.0095432, 0.0090722
8: 0.0077700, 0.0181385, 0.0068849, 0.0183949, -0.0081445, 0.0086052
9: 0.9038395, 0.9485210, 0.9024426, 0.9524551, -0.0341975, 0.0318374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0183963
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0180658
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041761, -0.0010325, -0.0030399, 0.0029394
1: 0.0181921, 0.0316567, 0.0169682, 0.0321007, -0.0091879, 0.0096903
2: 0.0206704, 0.0299115, 0.0200045, 0.0303035, -0.0066064, 0.0067181
3: 0.0063097, 0.0165808, 0.0053004, 0.0169994, -0.0080537, 0.0085790
4: -0.0173488, -0.0074310, -0.0177966, -0.0064833, -0.0087907, 0.0082517
5: 0.0127622, 0.0253943, 0.0114692, 0.0259043, -0.0105801, 0.0112456
6: 0.0049685, 0.0144403, 0.0040265, 0.0148571, -0.0079371, 0.0084626
7: -0.0220428, -0.0117536, -0.0224730, -0.0106558, -0.0096681, 0.0090809
8: 0.0077874, 0.0182481, 0.0067799, 0.0186110, -0.0082437, 0.0087586
9: 0.9032502, 0.9480295, 0.9013865, 0.9521797, -0.0342308, 0.0320281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163000, upper bound: 0.0177716
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040723, -0.0012367, -0.0041711, -0.0010655, -0.0030068, 0.0029344
1: 0.0181921, 0.0316567, 0.0168123, 0.0318989, -0.0090345, 0.0098714
2: 0.0206704, 0.0299115, 0.0198550, 0.0301546, -0.0064705, 0.0069055
3: 0.0063097, 0.0165808, 0.0052378, 0.0168694, -0.0079554, 0.0087095
4: -0.0173488, -0.0074310, -0.0176739, -0.0063858, -0.0089472, 0.0081616
5: 0.0127622, 0.0253943, 0.0114008, 0.0257650, -0.0104746, 0.0113778
6: 0.0049685, 0.0144403, 0.0039620, 0.0147440, -0.0078507, 0.0085873
7: -0.0220428, -0.0117536, -0.0223307, -0.0105696, -0.0098154, 0.0089859
8: 0.0077874, 0.0182481, 0.0067627, 0.0185066, -0.0081559, 0.0088340
9: 0.9032502, 0.9480295, 0.9020119, 0.9526696, -0.0349897, 0.0315818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163000, upper bound: 0.0177716
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171677
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041761, -0.0010325, -0.0030372, 0.0029163
1: 0.0180098, 0.0314699, 0.0169682, 0.0321007, -0.0093700, 0.0095352
2: 0.0205092, 0.0297685, 0.0200045, 0.0303035, -0.0068125, 0.0066055
3: 0.0062548, 0.0164433, 0.0053004, 0.0169994, -0.0082072, 0.0084524
4: -0.0172217, -0.0073388, -0.0177966, -0.0064833, -0.0086734, 0.0084280
5: 0.0127029, 0.0252562, 0.0114692, 0.0259043, -0.0107331, 0.0111093
6: 0.0049108, 0.0143212, 0.0040265, 0.0148571, -0.0080849, 0.0083496
7: -0.0219011, -0.0116561, -0.0224730, -0.0106558, -0.0095519, 0.0092338
8: 0.0077700, 0.0181385, 0.0067799, 0.0186110, -0.0083453, 0.0086516
9: 0.9038395, 0.9485210, 0.9013865, 0.9521797, -0.0336956, 0.0328340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0180423
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0176532
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040697, -0.0012598, -0.0041711, -0.0010655, -0.0030042, 0.0029112
1: 0.0180098, 0.0314699, 0.0168123, 0.0318989, -0.0091728, 0.0096375
2: 0.0205092, 0.0297685, 0.0198550, 0.0301546, -0.0066110, 0.0067070
3: 0.0062548, 0.0164433, 0.0052378, 0.0168694, -0.0081151, 0.0085901
4: -0.0172217, -0.0073388, -0.0176739, -0.0063858, -0.0088174, 0.0083256
5: 0.0127029, 0.0252562, 0.0114008, 0.0257650, -0.0106219, 0.0112318
6: 0.0049108, 0.0143212, 0.0039620, 0.0147440, -0.0080000, 0.0084759
7: -0.0219011, -0.0116561, -0.0223307, -0.0105696, -0.0096709, 0.0091141
8: 0.0077700, 0.0181385, 0.0067627, 0.0185066, -0.0082683, 0.0087439
9: 0.9038395, 0.9485210, 0.9020119, 0.9526696, -0.0343701, 0.0323216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0183386
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0180129
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041413, -0.0011326, -0.0029749, 0.0029920
1: 0.0180860, 0.0317525, 0.0170758, 0.0320063, -0.0092890, 0.0098200
2: 0.0206564, 0.0300202, 0.0200154, 0.0301909, -0.0064665, 0.0069409
3: 0.0062101, 0.0167013, 0.0053988, 0.0168827, -0.0080923, 0.0086425
4: -0.0174401, -0.0073896, -0.0176915, -0.0065340, -0.0088382, 0.0082617
5: 0.0126355, 0.0255008, 0.0115979, 0.0257699, -0.0106528, 0.0112379
6: 0.0048972, 0.0145426, 0.0041013, 0.0147464, -0.0079657, 0.0084952
7: -0.0221280, -0.0116488, -0.0223849, -0.0107666, -0.0096134, 0.0091959
8: 0.0076520, 0.0183602, 0.0069091, 0.0185007, -0.0083151, 0.0088198
9: 0.9028240, 0.9482442, 0.9018377, 0.9519727, -0.0347074, 0.0319862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178780
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171897
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041374, -0.0011655, -0.0029420, 0.0029882
1: 0.0180860, 0.0317525, 0.0169144, 0.0318150, -0.0091526, 0.0099985
2: 0.0206564, 0.0300202, 0.0198633, 0.0300454, -0.0063377, 0.0071330
3: 0.0062101, 0.0167013, 0.0053300, 0.0167500, -0.0079920, 0.0087760
4: -0.0174401, -0.0073896, -0.0175729, -0.0064338, -0.0089959, 0.0081667
5: 0.0126355, 0.0255008, 0.0115280, 0.0256327, -0.0105495, 0.0113701
6: 0.0048972, 0.0145426, 0.0040348, 0.0146375, -0.0078814, 0.0086219
7: -0.0221280, -0.0116488, -0.0222471, -0.0106774, -0.0097582, 0.0090949
8: 0.0076520, 0.0183602, 0.0068849, 0.0183949, -0.0082257, 0.0088955
9: 0.9028240, 0.9482442, 0.9024426, 0.9524551, -0.0354746, 0.0315285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178784
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0172056
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041413, -0.0011326, -0.0029709, 0.0029655
1: 0.0179049, 0.0315551, 0.0170758, 0.0320063, -0.0094716, 0.0096825
2: 0.0204964, 0.0298754, 0.0200154, 0.0301909, -0.0066656, 0.0068211
3: 0.0061511, 0.0165660, 0.0053988, 0.0168827, -0.0082428, 0.0085264
4: -0.0173159, -0.0072973, -0.0176915, -0.0065340, -0.0087166, 0.0084371
5: 0.0125759, 0.0253676, 0.0115979, 0.0257699, -0.0108094, 0.0111063
6: 0.0048413, 0.0144279, 0.0041013, 0.0147464, -0.0081113, 0.0083846
7: -0.0219870, -0.0115511, -0.0223849, -0.0107666, -0.0094884, 0.0093545
8: 0.0076329, 0.0182502, 0.0069091, 0.0185007, -0.0084209, 0.0087131
9: 0.9034380, 0.9487324, 0.9018377, 0.9519727, -0.0341995, 0.0327753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0181070
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0176661
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041374, -0.0011655, -0.0029380, 0.0029616
1: 0.0179049, 0.0315551, 0.0169144, 0.0318150, -0.0092807, 0.0097815
2: 0.0204964, 0.0298754, 0.0198633, 0.0300454, -0.0064674, 0.0069303
3: 0.0061511, 0.0165660, 0.0053300, 0.0167500, -0.0081484, 0.0086689
4: -0.0173159, -0.0072973, -0.0175729, -0.0064338, -0.0088631, 0.0083317
5: 0.0125759, 0.0253676, 0.0115280, 0.0256327, -0.0107004, 0.0112292
6: 0.0048413, 0.0144279, 0.0040348, 0.0146375, -0.0080283, 0.0085130
7: -0.0219870, -0.0115511, -0.0222471, -0.0106774, -0.0096021, 0.0092286
8: 0.0076329, 0.0182502, 0.0068849, 0.0183949, -0.0083422, 0.0088025
9: 0.9034380, 0.9487324, 0.9024426, 0.9524551, -0.0348865, 0.0322586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0184382
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0180352
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041761, -0.0010325, -0.0030750, 0.0030268
1: 0.0180860, 0.0317525, 0.0169682, 0.0321007, -0.0093283, 0.0098056
2: 0.0206564, 0.0300202, 0.0200045, 0.0303035, -0.0066060, 0.0068937
3: 0.0062101, 0.0167013, 0.0053004, 0.0169994, -0.0081569, 0.0086812
4: -0.0174401, -0.0073896, -0.0177966, -0.0064833, -0.0089017, 0.0083451
5: 0.0126355, 0.0255008, 0.0114692, 0.0259043, -0.0107525, 0.0113639
6: 0.0048972, 0.0145426, 0.0040265, 0.0148571, -0.0080464, 0.0085679
7: -0.0221280, -0.0116488, -0.0224730, -0.0106558, -0.0097561, 0.0092879
8: 0.0076520, 0.0183602, 0.0067799, 0.0186110, -0.0083802, 0.0088651
9: 0.9028240, 0.9482442, 0.9013865, 0.9521797, -0.0346377, 0.0322067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178780
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171555
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041075, -0.0011493, -0.0041711, -0.0010655, -0.0030420, 0.0030218
1: 0.0180860, 0.0317525, 0.0168123, 0.0318989, -0.0091934, 0.0099820
2: 0.0206564, 0.0300202, 0.0198550, 0.0301546, -0.0064793, 0.0070871
3: 0.0062101, 0.0167013, 0.0052378, 0.0168694, -0.0080572, 0.0088166
4: -0.0174401, -0.0073896, -0.0176739, -0.0063858, -0.0090605, 0.0082506
5: 0.0126355, 0.0255008, 0.0114008, 0.0257650, -0.0106491, 0.0114953
6: 0.0048972, 0.0145426, 0.0039620, 0.0147440, -0.0079622, 0.0086966
7: -0.0221280, -0.0116488, -0.0223307, -0.0105696, -0.0099015, 0.0091867
8: 0.0076520, 0.0183602, 0.0067627, 0.0185066, -0.0082904, 0.0089401
9: 0.9028240, 0.9482442, 0.9020119, 0.9526696, -0.0354124, 0.0317437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178784
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171680
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041761, -0.0010325, -0.0030710, 0.0030003
1: 0.0179049, 0.0315551, 0.0169682, 0.0321007, -0.0095128, 0.0096489
2: 0.0204964, 0.0298754, 0.0200045, 0.0303035, -0.0068162, 0.0067812
3: 0.0061511, 0.0165660, 0.0053004, 0.0169994, -0.0083131, 0.0085549
4: -0.0173159, -0.0072973, -0.0177966, -0.0064833, -0.0087838, 0.0085262
5: 0.0125759, 0.0253676, 0.0114692, 0.0259043, -0.0109138, 0.0112274
6: 0.0048413, 0.0144279, 0.0040265, 0.0148571, -0.0081971, 0.0084547
7: -0.0219870, -0.0115511, -0.0224730, -0.0106558, -0.0096401, 0.0094443
8: 0.0076329, 0.0182502, 0.0067799, 0.0186110, -0.0084884, 0.0087577
9: 0.9034380, 0.9487324, 0.9013865, 0.9521797, -0.0341034, 0.0330395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0181070
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0176534
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041035, -0.0011758, -0.0041711, -0.0010655, -0.0030379, 0.0029953
1: 0.0179049, 0.0315551, 0.0168123, 0.0318989, -0.0093253, 0.0097596
2: 0.0204964, 0.0298754, 0.0198550, 0.0301546, -0.0066135, 0.0068873
3: 0.0061511, 0.0165660, 0.0052378, 0.0168694, -0.0082203, 0.0086982
4: -0.0173159, -0.0072973, -0.0176739, -0.0063858, -0.0089305, 0.0084197
5: 0.0125759, 0.0253676, 0.0114008, 0.0257650, -0.0108041, 0.0113495
6: 0.0048413, 0.0144279, 0.0039620, 0.0147440, -0.0081144, 0.0085852
7: -0.0219870, -0.0115511, -0.0223307, -0.0105696, -0.0097575, 0.0093220
8: 0.0076329, 0.0182502, 0.0067627, 0.0185066, -0.0084099, 0.0088502
9: 0.9034380, 0.9487324, 0.9020119, 0.9526696, -0.0347928, 0.0325135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0184381
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0180209
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0040723, -0.0012367, -0.0029046, 0.0029398
1: 0.0170758, 0.0320063, 0.0181921, 0.0316567, -0.0095240, 0.0090439
2: 0.0200154, 0.0301909, 0.0206704, 0.0299115, -0.0067041, 0.0064267
3: 0.0053988, 0.0168827, 0.0063097, 0.0165808, -0.0084864, 0.0079457
4: -0.0176915, -0.0065340, -0.0173488, -0.0074310, -0.0081857, 0.0087525
5: 0.0115979, 0.0257699, 0.0127622, 0.0253943, -0.0111133, 0.0104706
6: 0.0041013, 0.0147464, 0.0049685, 0.0144403, -0.0083979, 0.0078567
7: -0.0223849, -0.0107666, -0.0220428, -0.0117536, -0.0090423, 0.0095420
8: 0.0069091, 0.0185007, 0.0077874, 0.0182481, -0.0086205, 0.0081217
9: 0.9018377, 0.9519727, 0.9032502, 0.9480295, -0.0315497, 0.0340480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166786
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163167
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0040697, -0.0012598, -0.0028815, 0.0029371
1: 0.0170758, 0.0320063, 0.0180098, 0.0314699, -0.0093689, 0.0092261
2: 0.0200154, 0.0301909, 0.0205092, 0.0297685, -0.0065915, 0.0066328
3: 0.0053988, 0.0168827, 0.0062548, 0.0164433, -0.0083599, 0.0080992
4: -0.0176915, -0.0065340, -0.0172217, -0.0073388, -0.0083621, 0.0086352
5: 0.0115979, 0.0257699, 0.0127029, 0.0252562, -0.0109770, 0.0106236
6: 0.0041013, 0.0147464, 0.0049108, 0.0143212, -0.0082850, 0.0080045
7: -0.0223849, -0.0107666, -0.0219011, -0.0116561, -0.0091952, 0.0094257
8: 0.0069091, 0.0185007, 0.0077700, 0.0181385, -0.0085136, 0.0082233
9: 0.9018377, 0.9519727, 0.9038395, 0.9485210, -0.0323556, 0.0335128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166838
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163417
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0040723, -0.0012367, -0.0029008, 0.0029068
1: 0.0169144, 0.0318150, 0.0181921, 0.0316567, -0.0097024, 0.0089076
2: 0.0198633, 0.0300454, 0.0206704, 0.0299115, -0.0068962, 0.0062980
3: 0.0053300, 0.0167500, 0.0063097, 0.0165808, -0.0086200, 0.0078454
4: -0.0175729, -0.0064338, -0.0173488, -0.0074310, -0.0080908, 0.0089101
5: 0.0115280, 0.0256327, 0.0127622, 0.0253943, -0.0112455, 0.0103672
6: 0.0040348, 0.0146375, 0.0049685, 0.0144403, -0.0085247, 0.0077724
7: -0.0222471, -0.0106774, -0.0220428, -0.0117536, -0.0089412, 0.0096867
8: 0.0068849, 0.0183949, 0.0077874, 0.0182481, -0.0086962, 0.0080323
9: 0.9024426, 0.9524551, 0.9032502, 0.9480295, -0.0310920, 0.0348151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0168548
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166526
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0040697, -0.0012598, -0.0028776, 0.0029042
1: 0.0169144, 0.0318150, 0.0180098, 0.0314699, -0.0094714, 0.0090375
2: 0.0198633, 0.0300454, 0.0205092, 0.0297685, -0.0066950, 0.0064316
3: 0.0053300, 0.0167500, 0.0062548, 0.0164433, -0.0085012, 0.0080045
4: -0.0175729, -0.0064338, -0.0172217, -0.0073388, -0.0082561, 0.0087801
5: 0.0115280, 0.0256327, 0.0127029, 0.0252562, -0.0110992, 0.0105147
6: 0.0040348, 0.0146375, 0.0049108, 0.0143212, -0.0084134, 0.0079215
7: -0.0222471, -0.0106774, -0.0219011, -0.0116561, -0.0090722, 0.0095432
8: 0.0068849, 0.0183949, 0.0077700, 0.0181385, -0.0086052, 0.0081445
9: 0.9024426, 0.9524551, 0.9038395, 0.9485210, -0.0318374, 0.0341975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0172089
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0171241
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041075, -0.0011493, -0.0029920, 0.0029749
1: 0.0170758, 0.0320063, 0.0180860, 0.0317525, -0.0098200, 0.0092890
2: 0.0200154, 0.0301909, 0.0206564, 0.0300202, -0.0069409, 0.0064665
3: 0.0053988, 0.0168827, 0.0062101, 0.0167013, -0.0086425, 0.0080923
4: -0.0176915, -0.0065340, -0.0174401, -0.0073896, -0.0082617, 0.0088382
5: 0.0115979, 0.0257699, 0.0126355, 0.0255008, -0.0112379, 0.0106528
6: 0.0041013, 0.0147464, 0.0048972, 0.0145426, -0.0084952, 0.0079657
7: -0.0223849, -0.0107666, -0.0221280, -0.0116488, -0.0091959, 0.0096134
8: 0.0069091, 0.0185007, 0.0076520, 0.0183602, -0.0088198, 0.0083151
9: 0.9018377, 0.9519727, 0.9028240, 0.9482442, -0.0319862, 0.0347074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166107
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041035, -0.0011758, -0.0029655, 0.0029709
1: 0.0170758, 0.0320063, 0.0179049, 0.0315551, -0.0096825, 0.0094716
2: 0.0200154, 0.0301909, 0.0204964, 0.0298754, -0.0068211, 0.0066656
3: 0.0053988, 0.0168827, 0.0061511, 0.0165660, -0.0085264, 0.0082428
4: -0.0176915, -0.0065340, -0.0173159, -0.0072973, -0.0084371, 0.0087166
5: 0.0115979, 0.0257699, 0.0125759, 0.0253676, -0.0111063, 0.0108094
6: 0.0041013, 0.0147464, 0.0048413, 0.0144279, -0.0083846, 0.0081113
7: -0.0223849, -0.0107666, -0.0219870, -0.0115511, -0.0093545, 0.0094884
8: 0.0069091, 0.0185007, 0.0076329, 0.0182502, -0.0087131, 0.0084209
9: 0.9018377, 0.9519727, 0.9034380, 0.9487324, -0.0327753, 0.0341995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166147
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162525
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041075, -0.0011493, -0.0029882, 0.0029420
1: 0.0169144, 0.0318150, 0.0180860, 0.0317525, -0.0099985, 0.0091526
2: 0.0198633, 0.0300454, 0.0206564, 0.0300202, -0.0071330, 0.0063377
3: 0.0053300, 0.0167500, 0.0062101, 0.0167013, -0.0087760, 0.0079920
4: -0.0175729, -0.0064338, -0.0174401, -0.0073896, -0.0081667, 0.0089959
5: 0.0115280, 0.0256327, 0.0126355, 0.0255008, -0.0113702, 0.0105495
6: 0.0040348, 0.0146375, 0.0048972, 0.0145426, -0.0086219, 0.0078814
7: -0.0222471, -0.0106774, -0.0221280, -0.0116488, -0.0090949, 0.0097582
8: 0.0068849, 0.0183949, 0.0076520, 0.0183602, -0.0088955, 0.0082257
9: 0.9024426, 0.9524551, 0.9028240, 0.9482442, -0.0315285, 0.0354746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0168226
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0165663
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041035, -0.0011758, -0.0029616, 0.0029380
1: 0.0169144, 0.0318150, 0.0179049, 0.0315551, -0.0097815, 0.0092807
2: 0.0198633, 0.0300454, 0.0204964, 0.0298754, -0.0069303, 0.0064674
3: 0.0053300, 0.0167500, 0.0061511, 0.0165660, -0.0086689, 0.0081484
4: -0.0175729, -0.0064338, -0.0173159, -0.0072973, -0.0083317, 0.0088631
5: 0.0115280, 0.0256327, 0.0125759, 0.0253676, -0.0112292, 0.0107004
6: 0.0040348, 0.0146375, 0.0048413, 0.0144279, -0.0085130, 0.0080283
7: -0.0222471, -0.0106774, -0.0219870, -0.0115511, -0.0092286, 0.0096021
8: 0.0068849, 0.0183949, 0.0076329, 0.0182502, -0.0088025, 0.0083422
9: 0.9024426, 0.9524551, 0.9034380, 0.9487324, -0.0322586, 0.0348865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0171305
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0170168
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0040723, -0.0012367, -0.0029394, 0.0030399
1: 0.0169682, 0.0321007, 0.0181921, 0.0316567, -0.0096903, 0.0091879
2: 0.0200045, 0.0303035, 0.0206704, 0.0299115, -0.0067181, 0.0066064
3: 0.0053004, 0.0169994, 0.0063097, 0.0165808, -0.0085790, 0.0080537
4: -0.0177966, -0.0064833, -0.0173488, -0.0074310, -0.0082517, 0.0087907
5: 0.0114692, 0.0259043, 0.0127622, 0.0253943, -0.0112456, 0.0105801
6: 0.0040265, 0.0148571, 0.0049685, 0.0144403, -0.0084626, 0.0079371
7: -0.0224730, -0.0106558, -0.0220428, -0.0117536, -0.0090809, 0.0096681
8: 0.0067799, 0.0186110, 0.0077874, 0.0182481, -0.0087586, 0.0082437
9: 0.9013865, 0.9521797, 0.9032502, 0.9480295, -0.0320281, 0.0342308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166788
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162937
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0040697, -0.0012598, -0.0029163, 0.0030372
1: 0.0169682, 0.0321007, 0.0180098, 0.0314699, -0.0095352, 0.0093700
2: 0.0200045, 0.0303035, 0.0205092, 0.0297685, -0.0066055, 0.0068125
3: 0.0053004, 0.0169994, 0.0062548, 0.0164433, -0.0084524, 0.0082072
4: -0.0177966, -0.0064833, -0.0172217, -0.0073388, -0.0084280, 0.0086734
5: 0.0114692, 0.0259043, 0.0127029, 0.0252562, -0.0111093, 0.0107331
6: 0.0040265, 0.0148571, 0.0049108, 0.0143212, -0.0083496, 0.0080849
7: -0.0224730, -0.0106558, -0.0219011, -0.0116561, -0.0092338, 0.0095519
8: 0.0067799, 0.0186110, 0.0077700, 0.0181385, -0.0086516, 0.0083453
9: 0.9013865, 0.9521797, 0.9038395, 0.9485210, -0.0328340, 0.0336956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166917
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163132
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0040723, -0.0012367, -0.0029344, 0.0030068
1: 0.0168123, 0.0318989, 0.0181921, 0.0316567, -0.0098714, 0.0090345
2: 0.0198550, 0.0301546, 0.0206704, 0.0299115, -0.0069055, 0.0064705
3: 0.0052378, 0.0168694, 0.0063097, 0.0165808, -0.0087095, 0.0079554
4: -0.0176739, -0.0063858, -0.0173488, -0.0074310, -0.0081616, 0.0089472
5: 0.0114008, 0.0257650, 0.0127622, 0.0253943, -0.0113778, 0.0104746
6: 0.0039620, 0.0147440, 0.0049685, 0.0144403, -0.0085873, 0.0078507
7: -0.0223307, -0.0105696, -0.0220428, -0.0117536, -0.0089859, 0.0098154
8: 0.0067627, 0.0185066, 0.0077874, 0.0182481, -0.0088340, 0.0081559
9: 0.9020119, 0.9526696, 0.9032502, 0.9480295, -0.0315818, 0.0349897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168869
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166241
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0040697, -0.0012598, -0.0029112, 0.0030042
1: 0.0168123, 0.0318989, 0.0180098, 0.0314699, -0.0096375, 0.0091728
2: 0.0198550, 0.0301546, 0.0205092, 0.0297685, -0.0067070, 0.0066110
3: 0.0052378, 0.0168694, 0.0062548, 0.0164433, -0.0085901, 0.0081151
4: -0.0176739, -0.0063858, -0.0172217, -0.0073388, -0.0083256, 0.0088174
5: 0.0114008, 0.0257650, 0.0127029, 0.0252562, -0.0112318, 0.0106219
6: 0.0039620, 0.0147440, 0.0049108, 0.0143212, -0.0084759, 0.0080000
7: -0.0223307, -0.0105696, -0.0219011, -0.0116561, -0.0091141, 0.0096709
8: 0.0067627, 0.0185066, 0.0077700, 0.0181385, -0.0087439, 0.0082683
9: 0.9020119, 0.9526696, 0.9038395, 0.9485210, -0.0323216, 0.0343701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170855
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041075, -0.0011493, -0.0030268, 0.0030750
1: 0.0169682, 0.0321007, 0.0180860, 0.0317525, -0.0098056, 0.0093283
2: 0.0200045, 0.0303035, 0.0206564, 0.0300202, -0.0068937, 0.0066060
3: 0.0053004, 0.0169994, 0.0062101, 0.0167013, -0.0086812, 0.0081569
4: -0.0177966, -0.0064833, -0.0174401, -0.0073896, -0.0083451, 0.0089017
5: 0.0114692, 0.0259043, 0.0126355, 0.0255008, -0.0113639, 0.0107525
6: 0.0040265, 0.0148571, 0.0048972, 0.0145426, -0.0085679, 0.0080464
7: -0.0224730, -0.0106558, -0.0221280, -0.0116488, -0.0092879, 0.0097561
8: 0.0067799, 0.0186110, 0.0076520, 0.0183602, -0.0088651, 0.0083802
9: 0.9013865, 0.9521797, 0.9028240, 0.9482442, -0.0322067, 0.0346377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166788
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162411
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041035, -0.0011758, -0.0030003, 0.0030710
1: 0.0169682, 0.0321007, 0.0179049, 0.0315551, -0.0096489, 0.0095128
2: 0.0200045, 0.0303035, 0.0204964, 0.0298754, -0.0067812, 0.0068162
3: 0.0053004, 0.0169994, 0.0061511, 0.0165660, -0.0085549, 0.0083131
4: -0.0177966, -0.0064833, -0.0173159, -0.0072973, -0.0085262, 0.0087838
5: 0.0114692, 0.0259043, 0.0125759, 0.0253676, -0.0112274, 0.0109138
6: 0.0040265, 0.0148571, 0.0048413, 0.0144279, -0.0084547, 0.0081971
7: -0.0224730, -0.0106558, -0.0219870, -0.0115511, -0.0094443, 0.0096401
8: 0.0067799, 0.0186110, 0.0076329, 0.0182502, -0.0087577, 0.0084884
9: 0.9013865, 0.9521797, 0.9034380, 0.9487324, -0.0330395, 0.0341034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166917
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162741
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041075, -0.0011493, -0.0030218, 0.0030420
1: 0.0168123, 0.0318989, 0.0180860, 0.0317525, -0.0099820, 0.0091934
2: 0.0198550, 0.0301546, 0.0206564, 0.0300202, -0.0070871, 0.0064793
3: 0.0052378, 0.0168694, 0.0062101, 0.0167013, -0.0088166, 0.0080572
4: -0.0176739, -0.0063858, -0.0174401, -0.0073896, -0.0082506, 0.0090605
5: 0.0114008, 0.0257650, 0.0126355, 0.0255008, -0.0114952, 0.0106491
6: 0.0039620, 0.0147440, 0.0048972, 0.0145426, -0.0086966, 0.0079622
7: -0.0223307, -0.0105696, -0.0221280, -0.0116488, -0.0091867, 0.0099015
8: 0.0067627, 0.0185066, 0.0076520, 0.0183602, -0.0089401, 0.0082904
9: 0.9020119, 0.9526696, 0.9028240, 0.9482442, -0.0317437, 0.0354125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168869
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0165961
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041035, -0.0011758, -0.0029953, 0.0030379
1: 0.0168123, 0.0318989, 0.0179049, 0.0315551, -0.0097596, 0.0093253
2: 0.0198550, 0.0301546, 0.0204964, 0.0298754, -0.0068873, 0.0066135
3: 0.0052378, 0.0168694, 0.0061511, 0.0165660, -0.0086982, 0.0082203
4: -0.0176739, -0.0063858, -0.0173159, -0.0072973, -0.0084197, 0.0089305
5: 0.0114008, 0.0257650, 0.0125759, 0.0253676, -0.0113495, 0.0108041
6: 0.0039620, 0.0147440, 0.0048413, 0.0144279, -0.0085852, 0.0081144
7: -0.0223307, -0.0105696, -0.0219870, -0.0115511, -0.0093219, 0.0097575
8: 0.0067627, 0.0185066, 0.0076329, 0.0182502, -0.0088502, 0.0084099
9: 0.9020119, 0.9526696, 0.9034380, 0.9487324, -0.0325135, 0.0347928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170607
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041413, -0.0011326, -0.0030087, 0.0030087
1: 0.0170758, 0.0320063, 0.0170758, 0.0320063, -0.0089097, 0.0089097
2: 0.0200154, 0.0301909, 0.0200154, 0.0301909, -0.0063112, 0.0063112
3: 0.0053988, 0.0168827, 0.0053988, 0.0168827, -0.0080983, 0.0080983
4: -0.0176915, -0.0065340, -0.0176915, -0.0065340, -0.0083201, 0.0083201
5: 0.0115979, 0.0257699, 0.0115979, 0.0257699, -0.0106811, 0.0106811
6: 0.0041013, 0.0147464, 0.0041013, 0.0147464, -0.0080336, 0.0080336
7: -0.0223849, -0.0107666, -0.0223849, -0.0107666, -0.0092154, 0.0092154
8: 0.0069091, 0.0185007, 0.0069091, 0.0185007, -0.0083021, 0.0083021
9: 0.9018377, 0.9519727, 0.9018377, 0.9519727, -0.0316663, 0.0316663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166862
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163500
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041374, -0.0011655, -0.0029758, 0.0030049
1: 0.0170758, 0.0320063, 0.0169144, 0.0318150, -0.0087443, 0.0090866
2: 0.0200154, 0.0301909, 0.0198633, 0.0300454, -0.0061940, 0.0065195
3: 0.0053988, 0.0168827, 0.0053300, 0.0167500, -0.0079550, 0.0082499
4: -0.0176915, -0.0065340, -0.0175729, -0.0064338, -0.0084965, 0.0081961
5: 0.0115979, 0.0257699, 0.0115280, 0.0256327, -0.0105327, 0.0108323
6: 0.0041013, 0.0147464, 0.0040348, 0.0146375, -0.0079109, 0.0081816
7: -0.0223849, -0.0107666, -0.0222471, -0.0106774, -0.0093692, 0.0090914
8: 0.0069091, 0.0185007, 0.0068849, 0.0183949, -0.0081862, 0.0084028
9: 0.9018377, 0.9519727, 0.9024426, 0.9524551, -0.0324762, 0.0310532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166898
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163654
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041413, -0.0011326, -0.0030049, 0.0029758
1: 0.0169144, 0.0318150, 0.0170758, 0.0320063, -0.0090866, 0.0087443
2: 0.0198633, 0.0300454, 0.0200154, 0.0301909, -0.0065195, 0.0061940
3: 0.0053300, 0.0167500, 0.0053988, 0.0168827, -0.0082499, 0.0079550
4: -0.0175729, -0.0064338, -0.0176915, -0.0065340, -0.0081961, 0.0084965
5: 0.0115280, 0.0256327, 0.0115979, 0.0257699, -0.0108323, 0.0105327
6: 0.0040348, 0.0146375, 0.0041013, 0.0147464, -0.0081816, 0.0079109
7: -0.0222471, -0.0106774, -0.0223849, -0.0107666, -0.0090914, 0.0093692
8: 0.0068849, 0.0183949, 0.0069091, 0.0185007, -0.0084028, 0.0081862
9: 0.9024426, 0.9524551, 0.9018377, 0.9519727, -0.0310532, 0.0324762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0168559
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166703
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041374, -0.0011655, -0.0029719, 0.0029719
1: 0.0169144, 0.0318150, 0.0169144, 0.0318150, -0.0088625, 0.0088625
2: 0.0198633, 0.0300454, 0.0198633, 0.0300454, -0.0063062, 0.0063062
3: 0.0053300, 0.0167500, 0.0053300, 0.0167500, -0.0081107, 0.0081107
4: -0.0175729, -0.0064338, -0.0175729, -0.0064338, -0.0083637, 0.0083637
5: 0.0115280, 0.0256327, 0.0115280, 0.0256327, -0.0106802, 0.0106802
6: 0.0040348, 0.0146375, 0.0040348, 0.0146375, -0.0080599, 0.0080599
7: -0.0222471, -0.0106774, -0.0222471, -0.0106774, -0.0092277, 0.0092277
8: 0.0068849, 0.0183949, 0.0068849, 0.0183949, -0.0082946, 0.0082946
9: 0.9024426, 0.9524551, 0.9024426, 0.9524551, -0.0318113, 0.0318113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0172089
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0171241
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041761, -0.0010325, -0.0031088, 0.0030435
1: 0.0170758, 0.0320063, 0.0169682, 0.0321007, -0.0091978, 0.0091457
2: 0.0200154, 0.0301909, 0.0200045, 0.0303035, -0.0065484, 0.0063503
3: 0.0053988, 0.0168827, 0.0053004, 0.0169994, -0.0082556, 0.0082432
4: -0.0176915, -0.0065340, -0.0177966, -0.0064833, -0.0083962, 0.0084124
5: 0.0115979, 0.0257699, 0.0114692, 0.0259043, -0.0108107, 0.0108602
6: 0.0041013, 0.0147464, 0.0040265, 0.0148571, -0.0081356, 0.0081425
7: -0.0223849, -0.0107666, -0.0224730, -0.0106558, -0.0093668, 0.0092911
8: 0.0069091, 0.0185007, 0.0067799, 0.0186110, -0.0084987, 0.0084938
9: 0.9018377, 0.9519727, 0.9013865, 0.9521797, -0.0320992, 0.0323321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166238
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162738
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041413, -0.0011326, -0.0041711, -0.0010655, -0.0030757, 0.0030385
1: 0.0170758, 0.0320063, 0.0168123, 0.0318989, -0.0090565, 0.0093243
2: 0.0200154, 0.0301909, 0.0198550, 0.0301546, -0.0064250, 0.0065509
3: 0.0053988, 0.0168827, 0.0052378, 0.0168694, -0.0081224, 0.0083900
4: -0.0176915, -0.0065340, -0.0176739, -0.0063858, -0.0085709, 0.0082822
5: 0.0115979, 0.0257699, 0.0114008, 0.0257650, -0.0106634, 0.0110123
6: 0.0041013, 0.0147464, 0.0039620, 0.0147440, -0.0080149, 0.0082857
7: -0.0223849, -0.0107666, -0.0223307, -0.0105696, -0.0095218, 0.0091583
8: 0.0069091, 0.0185007, 0.0067627, 0.0185066, -0.0083810, 0.0085941
9: 0.9018377, 0.9519727, 0.9020119, 0.9526696, -0.0328769, 0.0317500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166279
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162887
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041761, -0.0010325, -0.0031049, 0.0030106
1: 0.0169144, 0.0318150, 0.0169682, 0.0321007, -0.0093747, 0.0089803
2: 0.0198633, 0.0300454, 0.0200045, 0.0303035, -0.0067566, 0.0062330
3: 0.0053300, 0.0167500, 0.0053004, 0.0169994, -0.0084072, 0.0080999
4: -0.0175729, -0.0064338, -0.0177966, -0.0064833, -0.0082722, 0.0085888
5: 0.0115280, 0.0256327, 0.0114692, 0.0259043, -0.0109618, 0.0107118
6: 0.0040348, 0.0146375, 0.0040265, 0.0148571, -0.0082835, 0.0080197
7: -0.0222471, -0.0106774, -0.0224730, -0.0106558, -0.0092427, 0.0094448
8: 0.0068849, 0.0183949, 0.0067799, 0.0186110, -0.0085995, 0.0083779
9: 0.9024426, 0.9524551, 0.9013865, 0.9521797, -0.0314860, 0.0331420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0168227
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0165824
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041374, -0.0011655, -0.0041711, -0.0010655, -0.0030719, 0.0030055
1: 0.0169144, 0.0318150, 0.0168123, 0.0318989, -0.0091682, 0.0090968
2: 0.0198633, 0.0300454, 0.0198550, 0.0301546, -0.0065410, 0.0063412
3: 0.0053300, 0.0167500, 0.0052378, 0.0168694, -0.0082779, 0.0082512
4: -0.0175729, -0.0064338, -0.0176739, -0.0063858, -0.0084376, 0.0084521
5: 0.0115280, 0.0256327, 0.0114008, 0.0257650, -0.0108113, 0.0108597
6: 0.0040348, 0.0146375, 0.0039620, 0.0147440, -0.0081640, 0.0081639
7: -0.0222471, -0.0106774, -0.0223307, -0.0105696, -0.0093788, 0.0092944
8: 0.0068849, 0.0183949, 0.0067627, 0.0185066, -0.0084873, 0.0084856
9: 0.9024426, 0.9524551, 0.9020119, 0.9526696, -0.0322112, 0.0325045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0171305
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0170168
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041413, -0.0011326, -0.0030435, 0.0031088
1: 0.0169682, 0.0321007, 0.0170758, 0.0320063, -0.0091457, 0.0091978
2: 0.0200045, 0.0303035, 0.0200154, 0.0301909, -0.0063503, 0.0065484
3: 0.0053004, 0.0169994, 0.0053988, 0.0168827, -0.0082432, 0.0082556
4: -0.0177966, -0.0064833, -0.0176915, -0.0065340, -0.0084124, 0.0083962
5: 0.0114692, 0.0259043, 0.0115979, 0.0257699, -0.0108602, 0.0108107
6: 0.0040265, 0.0148571, 0.0041013, 0.0147464, -0.0081425, 0.0081356
7: -0.0224730, -0.0106558, -0.0223849, -0.0107666, -0.0092911, 0.0093668
8: 0.0067799, 0.0186110, 0.0069091, 0.0185007, -0.0084938, 0.0084987
9: 0.9013865, 0.9521797, 0.9018377, 0.9519727, -0.0323321, 0.0320992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166984
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163312
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041374, -0.0011655, -0.0030106, 0.0031049
1: 0.0169682, 0.0321007, 0.0169144, 0.0318150, -0.0089803, 0.0093747
2: 0.0200045, 0.0303035, 0.0198633, 0.0300454, -0.0062330, 0.0067566
3: 0.0053004, 0.0169994, 0.0053300, 0.0167500, -0.0080999, 0.0084072
4: -0.0177966, -0.0064833, -0.0175729, -0.0064338, -0.0085888, 0.0082722
5: 0.0114692, 0.0259043, 0.0115280, 0.0256327, -0.0107118, 0.0109618
6: 0.0040265, 0.0148571, 0.0040348, 0.0146375, -0.0080197, 0.0082835
7: -0.0224730, -0.0106558, -0.0222471, -0.0106774, -0.0094448, 0.0092427
8: 0.0067799, 0.0186110, 0.0068849, 0.0183949, -0.0083779, 0.0085995
9: 0.9013865, 0.9521797, 0.9024426, 0.9524551, -0.0331420, 0.0314860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0167077
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163454
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041413, -0.0011326, -0.0030385, 0.0030757
1: 0.0168123, 0.0318989, 0.0170758, 0.0320063, -0.0093243, 0.0090565
2: 0.0198550, 0.0301546, 0.0200154, 0.0301909, -0.0065509, 0.0064250
3: 0.0052378, 0.0168694, 0.0053988, 0.0168827, -0.0083900, 0.0081224
4: -0.0176739, -0.0063858, -0.0176915, -0.0065340, -0.0082822, 0.0085709
5: 0.0114008, 0.0257650, 0.0115979, 0.0257699, -0.0110123, 0.0106634
6: 0.0039620, 0.0147440, 0.0041013, 0.0147464, -0.0082857, 0.0080149
7: -0.0223307, -0.0105696, -0.0223849, -0.0107666, -0.0091583, 0.0095218
8: 0.0067627, 0.0185066, 0.0069091, 0.0185007, -0.0085941, 0.0083810
9: 0.9020119, 0.9526696, 0.9018377, 0.9519727, -0.0317500, 0.0328769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168896
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166456
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041374, -0.0011655, -0.0030055, 0.0030719
1: 0.0168123, 0.0318989, 0.0169144, 0.0318150, -0.0090968, 0.0091682
2: 0.0198550, 0.0301546, 0.0198633, 0.0300454, -0.0063412, 0.0065410
3: 0.0052378, 0.0168694, 0.0053300, 0.0167500, -0.0082512, 0.0082779
4: -0.0176739, -0.0063858, -0.0175729, -0.0064338, -0.0084521, 0.0084376
5: 0.0114008, 0.0257650, 0.0115280, 0.0256327, -0.0108597, 0.0108113
6: 0.0039620, 0.0147440, 0.0040348, 0.0146375, -0.0081639, 0.0081640
7: -0.0223307, -0.0105696, -0.0222471, -0.0106774, -0.0092944, 0.0093788
8: 0.0067627, 0.0185066, 0.0068849, 0.0183949, -0.0084856, 0.0084873
9: 0.9020119, 0.9526696, 0.9024426, 0.9524551, -0.0325045, 0.0322112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170855
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041761, -0.0010325, -0.0031436, 0.0031436
1: 0.0169682, 0.0321007, 0.0169682, 0.0321007, -0.0092185, 0.0092185
2: 0.0200045, 0.0303035, 0.0200045, 0.0303035, -0.0065115, 0.0065115
3: 0.0053004, 0.0169994, 0.0053004, 0.0169994, -0.0083065, 0.0083065
4: -0.0177966, -0.0064833, -0.0177966, -0.0064833, -0.0084723, 0.0084723
5: 0.0114692, 0.0259043, 0.0114692, 0.0259043, -0.0109535, 0.0109535
6: 0.0040265, 0.0148571, 0.0040265, 0.0148571, -0.0082168, 0.0082168
7: -0.0224730, -0.0106558, -0.0224730, -0.0106558, -0.0094578, 0.0094578
8: 0.0067799, 0.0186110, 0.0067799, 0.0186110, -0.0085563, 0.0085563
9: 0.9013865, 0.9521797, 0.9013865, 0.9521797, -0.0322997, 0.0322997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166984
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162955
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041761, -0.0010325, -0.0041711, -0.0010655, -0.0031105, 0.0031386
1: 0.0169682, 0.0321007, 0.0168123, 0.0318989, -0.0090561, 0.0093987
2: 0.0200045, 0.0303035, 0.0198550, 0.0301546, -0.0063941, 0.0067216
3: 0.0053004, 0.0169994, 0.0052378, 0.0168694, -0.0081629, 0.0084591
4: -0.0177966, -0.0064833, -0.0176739, -0.0063858, -0.0086530, 0.0083480
5: 0.0114692, 0.0259043, 0.0114008, 0.0257650, -0.0108052, 0.0111098
6: 0.0040265, 0.0148571, 0.0039620, 0.0147440, -0.0080937, 0.0083653
7: -0.0224730, -0.0106558, -0.0223307, -0.0105696, -0.0096124, 0.0093343
8: 0.0067799, 0.0186110, 0.0067627, 0.0185066, -0.0084418, 0.0086610
9: 0.9013865, 0.9521797, 0.9020119, 0.9526696, -0.0331264, 0.0316923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0167077
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163145
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041761, -0.0010325, -0.0031386, 0.0031105
1: 0.0168123, 0.0318989, 0.0169682, 0.0321007, -0.0093987, 0.0090561
2: 0.0198550, 0.0301546, 0.0200045, 0.0303035, -0.0067216, 0.0063941
3: 0.0052378, 0.0168694, 0.0053004, 0.0169994, -0.0084591, 0.0081629
4: -0.0176739, -0.0063858, -0.0177966, -0.0064833, -0.0083480, 0.0086530
5: 0.0114008, 0.0257650, 0.0114692, 0.0259043, -0.0111098, 0.0108052
6: 0.0039620, 0.0147440, 0.0040265, 0.0148571, -0.0083653, 0.0080937
7: -0.0223307, -0.0105696, -0.0224730, -0.0106558, -0.0093343, 0.0096124
8: 0.0067627, 0.0185066, 0.0067799, 0.0186110, -0.0086610, 0.0084418
9: 0.9020119, 0.9526696, 0.9013865, 0.9521797, -0.0316923, 0.0331264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168896
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166129
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0010655, -0.0041711, -0.0010655, -0.0031055, 0.0031055
1: 0.0168123, 0.0318989, 0.0168123, 0.0318989, -0.0091784, 0.0091784
2: 0.0198550, 0.0301546, 0.0198550, 0.0301546, -0.0065059, 0.0065059
3: 0.0052378, 0.0168694, 0.0052378, 0.0168694, -0.0083208, 0.0083208
4: -0.0176739, -0.0063858, -0.0176739, -0.0063858, -0.0085187, 0.0085187
5: 0.0114008, 0.0257650, 0.0114008, 0.0257650, -0.0109573, 0.0109573
6: 0.0039620, 0.0147440, 0.0039620, 0.0147440, -0.0082432, 0.0082432
7: -0.0223307, -0.0105696, -0.0223307, -0.0105696, -0.0094726, 0.0094725
8: 0.0067627, 0.0185066, 0.0067627, 0.0185066, -0.0085538, 0.0085538
9: 0.9020119, 0.9526696, 0.9020119, 0.9526696, -0.0324636, 0.0324636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170607
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.42 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163217, upper bound: 0.0166957
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163172
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163217, upper bound: 0.0167021
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163417
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0169104
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0166697
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0172749
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0171460
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163001, upper bound: 0.0166137
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163001, upper bound: 0.0166208
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162530
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0168685
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0165952
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0172035
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0170494
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167104
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162937
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167289
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0163132
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0169386
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0166364
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0173066
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0171038
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167104
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162411
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162350, upper bound: 0.0167289
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162741
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0169386
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0166125
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162547, upper bound: 0.0173066
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162530, upper bound: 0.0170825
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163214, upper bound: 0.0178225
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172214
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163214, upper bound: 0.0178230
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172366
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0180716
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0176918
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0183963
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0180658
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163000, upper bound: 0.0177716
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163000, upper bound: 0.0177716
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171677
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0180423
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0176532
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163134, upper bound: 0.0183386
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0163132, upper bound: 0.0180129
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178780
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171897
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178784
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0172056
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0181070
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0176661
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0184382
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0180352
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178780
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171555
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162339, upper bound: 0.0178784
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171680
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0181070
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0176534
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162533, upper bound: 0.0184381
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0162525, upper bound: 0.0180209
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166786
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163167
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166838
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163417
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0168548
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166526
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0172089
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0171241
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166107
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166147
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162525
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0168226
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0165663
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0171305
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0170168
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166788
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162937
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166917
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163132
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168869
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166241
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170855
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166788
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162411
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166917
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162741
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168869
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0165961
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170607
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166862
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163500
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166898
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163654
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0168559
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166703
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0172089
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0171241
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166238
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162738
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166279
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162887
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0168227
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0165824
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0171305
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0170168
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166984
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163312
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0167077
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163454
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168896
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166456
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170855
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166984
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162955
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0167077
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163145
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168896
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166129
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170607

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0040723, -0.0012367, -0.0026405, 0.0026494
1: 0.0188506, 0.0316567, 0.0181921, 0.0316567, -0.0070776, 0.0081976
2: 0.0208018, 0.0295784, 0.0206704, 0.0299115, -0.0056227, 0.0052891
3: 0.0068664, 0.0165629, 0.0063097, 0.0165808, -0.0067365, 0.0075758
4: -0.0172798, -0.0077549, -0.0173488, -0.0074310, -0.0078915, 0.0073972
5: 0.0135111, 0.0253347, 0.0127622, 0.0253943, -0.0089745, 0.0100698
6: 0.0054180, 0.0143917, 0.0049685, 0.0144403, -0.0069274, 0.0075885
7: -0.0219702, -0.0123623, -0.0220428, -0.0117536, -0.0087389, 0.0078120
8: 0.0085528, 0.0182481, 0.0077874, 0.0182481, -0.0066528, 0.0077325
9: 0.9033568, 0.9465656, 0.9032502, 0.9480295, -0.0295633, 0.0272156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163172
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163172
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0039934, -0.0013401, -0.0025184, 0.0031186
1: 0.0192105, 0.0317763, 0.0186414, 0.0316567, -0.0069815, 0.0080869
2: 0.0212826, 0.0298302, 0.0208663, 0.0297881, -0.0052978, 0.0055387
3: 0.0074236, 0.0171280, 0.0067344, 0.0165721, -0.0066872, 0.0083949
4: -0.0181071, -0.0084861, -0.0173153, -0.0078063, -0.0091576, 0.0072369
5: 0.0142199, 0.0261498, 0.0133101, 0.0253652, -0.0088805, 0.0111842
6: 0.0060204, 0.0151158, 0.0053574, 0.0144166, -0.0068309, 0.0086390
7: -0.0225776, -0.0130546, -0.0220074, -0.0122429, -0.0096807, 0.0076977
8: 0.0089573, 0.0187225, 0.0082480, 0.0182481, -0.0066435, 0.0082523
9: 0.9012735, 0.9438298, 0.9033021, 0.9465238, -0.0333281, 0.0267858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145934, upper bound: 0.0134198
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144107, upper bound: 0.0144107
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0040697, -0.0012598, -0.0026174, 0.0026468
1: 0.0188506, 0.0316567, 0.0180098, 0.0314699, -0.0069226, 0.0083797
2: 0.0208018, 0.0295784, 0.0205092, 0.0297685, -0.0055100, 0.0054952
3: 0.0068664, 0.0165629, 0.0062548, 0.0164433, -0.0066100, 0.0077294
4: -0.0172798, -0.0077549, -0.0172217, -0.0073388, -0.0080678, 0.0072800
5: 0.0135111, 0.0253347, 0.0127029, 0.0252562, -0.0088382, 0.0102229
6: 0.0054180, 0.0143917, 0.0049108, 0.0143212, -0.0068144, 0.0077363
7: -0.0219702, -0.0123623, -0.0219011, -0.0116561, -0.0088918, 0.0076958
8: 0.0085528, 0.0182481, 0.0077700, 0.0181385, -0.0065458, 0.0078341
9: 0.9033568, 0.9465656, 0.9038395, 0.9485210, -0.0303693, 0.0266804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166539, upper bound: 0.0163417
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166539, upper bound: 0.0163417
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0039880, -0.0013556, -0.0025029, 0.0031132
1: 0.0192105, 0.0317763, 0.0184532, 0.0314699, -0.0068225, 0.0082683
2: 0.0212826, 0.0298302, 0.0206996, 0.0296439, -0.0051799, 0.0057397
3: 0.0074236, 0.0171280, 0.0066669, 0.0164346, -0.0065572, 0.0085468
4: -0.0181071, -0.0084861, -0.0171870, -0.0077044, -0.0093319, 0.0071171
5: 0.0142199, 0.0261498, 0.0132360, 0.0252261, -0.0087400, 0.0113399
6: 0.0060204, 0.0151158, 0.0052921, 0.0142970, -0.0067150, 0.0087814
7: -0.0225776, -0.0130546, -0.0218644, -0.0121387, -0.0098398, 0.0075792
8: 0.0089573, 0.0187225, 0.0082223, 0.0181385, -0.0065333, 0.0083557
9: 0.9012735, 0.9438298, 0.9038917, 0.9470623, -0.0341397, 0.0262373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149588, upper bound: 0.0134234
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148388, upper bound: 0.0144498
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0040723, -0.0012367, -0.0026366, 0.0026395
1: 0.0186610, 0.0314699, 0.0181921, 0.0316567, -0.0072574, 0.0080475
2: 0.0206414, 0.0294323, 0.0206704, 0.0299115, -0.0058324, 0.0051776
3: 0.0068071, 0.0164252, 0.0063097, 0.0165808, -0.0068840, 0.0074491
4: -0.0171503, -0.0076708, -0.0173488, -0.0074310, -0.0077754, 0.0075701
5: 0.0134471, 0.0251937, 0.0127622, 0.0253943, -0.0091281, 0.0099318
6: 0.0053620, 0.0142708, 0.0049685, 0.0144403, -0.0070725, 0.0074761
7: -0.0218249, -0.0122729, -0.0220428, -0.0117536, -0.0086228, 0.0079690
8: 0.0085303, 0.0181385, 0.0077874, 0.0182481, -0.0067463, 0.0076254
9: 0.9039483, 0.9470383, 0.9032502, 0.9480295, -0.0290237, 0.0280039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0166539
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0166539
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0039954, -0.0013382, -0.0025151, 0.0031103
1: 0.0190119, 0.0315876, 0.0186293, 0.0316567, -0.0071827, 0.0079656
2: 0.0211144, 0.0296830, 0.0208608, 0.0297914, -0.0055071, 0.0054553
3: 0.0073461, 0.0169788, 0.0067231, 0.0165723, -0.0068434, 0.0082775
4: -0.0179715, -0.0083750, -0.0173162, -0.0077967, -0.0090367, 0.0074164
5: 0.0141374, 0.0260014, 0.0132958, 0.0253659, -0.0090352, 0.0110536
6: 0.0059462, 0.0149819, 0.0053471, 0.0144172, -0.0069788, 0.0085284
7: -0.0224307, -0.0129384, -0.0220083, -0.0122305, -0.0095511, 0.0078520
8: 0.0089189, 0.0186002, 0.0082356, 0.0182481, -0.0067463, 0.0081592
9: 0.9018508, 0.9443790, 0.9033009, 0.9465651, -0.0327987, 0.0276234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146453, upper bound: 0.0140472
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144498, upper bound: 0.0148388
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0040697, -0.0012598, -0.0026134, 0.0026368
1: 0.0186610, 0.0314699, 0.0180098, 0.0314699, -0.0070393, 0.0081721
2: 0.0206414, 0.0294323, 0.0205092, 0.0297685, -0.0056242, 0.0052943
3: 0.0068071, 0.0164252, 0.0062548, 0.0164433, -0.0067625, 0.0076070
4: -0.0171503, -0.0076708, -0.0172217, -0.0073388, -0.0079424, 0.0074428
5: 0.0134471, 0.0251937, 0.0127029, 0.0252562, -0.0089850, 0.0100802
6: 0.0053620, 0.0142708, 0.0049108, 0.0143212, -0.0069607, 0.0076249
7: -0.0218249, -0.0122729, -0.0219011, -0.0116561, -0.0087625, 0.0078313
8: 0.0085303, 0.0181385, 0.0077700, 0.0181385, -0.0066489, 0.0077369
9: 0.9039483, 0.9470383, 0.9038395, 0.9485210, -0.0297647, 0.0273914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169951, upper bound: 0.0170749
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169951, upper bound: 0.0170749
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0039874, -0.0013560, -0.0024973, 0.0031023
1: 0.0190119, 0.0315876, 0.0184562, 0.0314699, -0.0069407, 0.0080646
2: 0.0211144, 0.0296830, 0.0207009, 0.0296431, -0.0052953, 0.0055576
3: 0.0073461, 0.0169788, 0.0066697, 0.0164345, -0.0067164, 0.0084258
4: -0.0179715, -0.0083750, -0.0171868, -0.0077067, -0.0091984, 0.0072848
5: 0.0141374, 0.0260014, 0.0132394, 0.0252260, -0.0088855, 0.0111938
6: 0.0059462, 0.0149819, 0.0052946, 0.0142968, -0.0068616, 0.0086641
7: -0.0224307, -0.0129384, -0.0218641, -0.0121418, -0.0096874, 0.0077160
8: 0.0089189, 0.0186002, 0.0082253, 0.0181385, -0.0066460, 0.0082614
9: 0.9018508, 0.9443790, 0.9038921, 0.9470524, -0.0335263, 0.0269860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155492, upper bound: 0.0145125
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154794, upper bound: 0.0155896
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0041075, -0.0011493, -0.0027279, 0.0026846
1: 0.0188506, 0.0316567, 0.0180860, 0.0317525, -0.0073737, 0.0084426
2: 0.0208018, 0.0295784, 0.0206564, 0.0300202, -0.0058594, 0.0053289
3: 0.0068664, 0.0165629, 0.0062101, 0.0167013, -0.0068926, 0.0077224
4: -0.0172798, -0.0077549, -0.0174401, -0.0073896, -0.0079674, 0.0074830
5: 0.0135111, 0.0253347, 0.0126355, 0.0255008, -0.0090991, 0.0102521
6: 0.0054180, 0.0143917, 0.0048972, 0.0145426, -0.0070246, 0.0076975
7: -0.0219702, -0.0123623, -0.0221280, -0.0116488, -0.0088926, 0.0078835
8: 0.0085528, 0.0182481, 0.0076520, 0.0183602, -0.0068521, 0.0079259
9: 0.9033568, 0.9465656, 0.9028240, 0.9482442, -0.0299999, 0.0278751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0040581, -0.0012396, -0.0026189, 0.0031834
1: 0.0192105, 0.0317763, 0.0183835, 0.0317525, -0.0071600, 0.0084482
2: 0.0212826, 0.0298302, 0.0207812, 0.0299441, -0.0055481, 0.0055974
3: 0.0074236, 0.0171280, 0.0064948, 0.0166963, -0.0067231, 0.0086190
4: -0.0181071, -0.0084861, -0.0174194, -0.0076315, -0.0092885, 0.0072513
5: 0.0142199, 0.0261498, 0.0129936, 0.0254831, -0.0089093, 0.0114858
6: 0.0060204, 0.0151158, 0.0051516, 0.0145284, -0.0068472, 0.0088159
7: -0.0225776, -0.0130546, -0.0221060, -0.0119655, -0.0099447, 0.0077121
8: 0.0089573, 0.0187225, 0.0079603, 0.0183602, -0.0066930, 0.0085433
9: 0.9012735, 0.9438298, 0.9028549, 0.9472599, -0.0339549, 0.0269527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0041035, -0.0011758, -0.0027014, 0.0026806
1: 0.0188506, 0.0316567, 0.0179049, 0.0315551, -0.0072362, 0.0086253
2: 0.0208018, 0.0295784, 0.0204964, 0.0298754, -0.0057396, 0.0055280
3: 0.0068664, 0.0165629, 0.0061511, 0.0165660, -0.0067765, 0.0078730
4: -0.0172798, -0.0077549, -0.0173159, -0.0072973, -0.0081429, 0.0073614
5: 0.0135111, 0.0253347, 0.0125759, 0.0253676, -0.0089675, 0.0104087
6: 0.0054180, 0.0143917, 0.0048413, 0.0144279, -0.0069140, 0.0078431
7: -0.0219702, -0.0123623, -0.0219870, -0.0115511, -0.0090512, 0.0077584
8: 0.0085528, 0.0182481, 0.0076329, 0.0182502, -0.0067453, 0.0080317
9: 0.9033568, 0.9465656, 0.9034380, 0.9487324, -0.0307889, 0.0273671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166234, upper bound: 0.0162529
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166234, upper bound: 0.0162529
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0040527, -0.0012643, -0.0025942, 0.0031779
1: 0.0192105, 0.0317763, 0.0182059, 0.0315551, -0.0070084, 0.0086235
2: 0.0212826, 0.0298302, 0.0206203, 0.0297938, -0.0054198, 0.0057963
3: 0.0074236, 0.0171280, 0.0064360, 0.0165608, -0.0065960, 0.0087661
4: -0.0181071, -0.0084861, -0.0172948, -0.0075373, -0.0094633, 0.0071314
5: 0.0142199, 0.0261498, 0.0129322, 0.0253494, -0.0087625, 0.0116355
6: 0.0060204, 0.0151158, 0.0050934, 0.0144131, -0.0067308, 0.0089546
7: -0.0225776, -0.0130546, -0.0219648, -0.0118678, -0.0101011, 0.0075906
8: 0.0089573, 0.0187225, 0.0079419, 0.0182502, -0.0065826, 0.0086430
9: 0.9012735, 0.9438298, 0.9034693, 0.9477551, -0.0347383, 0.0264013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149474, upper bound: 0.0133785
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148009, upper bound: 0.0143674
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0041075, -0.0011493, -0.0027240, 0.0026746
1: 0.0186610, 0.0314699, 0.0180860, 0.0317525, -0.0075534, 0.0082925
2: 0.0206414, 0.0294323, 0.0206564, 0.0300202, -0.0060692, 0.0052173
3: 0.0068071, 0.0164252, 0.0062101, 0.0167013, -0.0070400, 0.0075957
4: -0.0171503, -0.0076708, -0.0174401, -0.0073896, -0.0078514, 0.0076559
5: 0.0134471, 0.0251937, 0.0126355, 0.0255008, -0.0092527, 0.0101140
6: 0.0053620, 0.0142708, 0.0048972, 0.0145426, -0.0071698, 0.0075851
7: -0.0218249, -0.0122729, -0.0221280, -0.0116488, -0.0087764, 0.0080405
8: 0.0085303, 0.0181385, 0.0076520, 0.0183602, -0.0069456, 0.0078189
9: 0.9039483, 0.9470383, 0.9028240, 0.9482442, -0.0294602, 0.0286634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165805
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165805
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0040596, -0.0012374, -0.0026160, 0.0031745
1: 0.0190119, 0.0315876, 0.0183746, 0.0317525, -0.0073609, 0.0083247
2: 0.0211144, 0.0296830, 0.0207774, 0.0299463, -0.0057558, 0.0055137
3: 0.0073461, 0.0169788, 0.0064864, 0.0166964, -0.0068793, 0.0085000
4: -0.0179715, -0.0083750, -0.0174200, -0.0076244, -0.0091667, 0.0074308
5: 0.0141374, 0.0260014, 0.0129832, 0.0254836, -0.0090640, 0.0113528
6: 0.0059462, 0.0149819, 0.0051440, 0.0145289, -0.0069951, 0.0087037
7: -0.0224307, -0.0129384, -0.0221067, -0.0119565, -0.0098132, 0.0078664
8: 0.0089189, 0.0186002, 0.0079512, 0.0183602, -0.0067957, 0.0084486
9: 0.9018508, 0.9443790, 0.9028540, 0.9472894, -0.0334220, 0.0277903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165952
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165952
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0041035, -0.0011758, -0.0026975, 0.0026706
1: 0.0186610, 0.0314699, 0.0179049, 0.0315551, -0.0073494, 0.0084152
2: 0.0206414, 0.0294323, 0.0204964, 0.0298754, -0.0058595, 0.0053301
3: 0.0068071, 0.0164252, 0.0061511, 0.0165660, -0.0069302, 0.0077509
4: -0.0171503, -0.0076708, -0.0173159, -0.0072973, -0.0080181, 0.0075257
5: 0.0134471, 0.0251937, 0.0125759, 0.0253676, -0.0091150, 0.0102658
6: 0.0053620, 0.0142708, 0.0048413, 0.0144279, -0.0070603, 0.0077317
7: -0.0218249, -0.0122729, -0.0219870, -0.0115511, -0.0089189, 0.0078902
8: 0.0085303, 0.0181385, 0.0076329, 0.0182502, -0.0068462, 0.0079346
9: 0.9039483, 0.9470383, 0.9034380, 0.9487324, -0.0301859, 0.0280805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169550, upper bound: 0.0169893
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169550, upper bound: 0.0169893
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0040532, -0.0012635, -0.0025898, 0.0031681
1: 0.0190119, 0.0315876, 0.0182028, 0.0315551, -0.0071274, 0.0084210
2: 0.0211144, 0.0296830, 0.0206190, 0.0297946, -0.0055451, 0.0056163
3: 0.0073461, 0.0169788, 0.0064330, 0.0165609, -0.0067556, 0.0086487
4: -0.0179715, -0.0083750, -0.0172950, -0.0075349, -0.0093316, 0.0072991
5: 0.0141374, 0.0260014, 0.0129286, 0.0253496, -0.0089112, 0.0114950
6: 0.0059462, 0.0149819, 0.0050908, 0.0144133, -0.0068774, 0.0088403
7: -0.0224307, -0.0129384, -0.0219650, -0.0118646, -0.0099528, 0.0077260
8: 0.0089189, 0.0186002, 0.0079387, 0.0182502, -0.0066962, 0.0085536
9: 0.9018508, 0.9443790, 0.9034691, 0.9477653, -0.0341343, 0.0271557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155436, upper bound: 0.0144785
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154468, upper bound: 0.0154999
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0040723, -0.0012367, -0.0026410, 0.0026564
1: 0.0188723, 0.0317525, 0.0181921, 0.0316567, -0.0071663, 0.0084786
2: 0.0208139, 0.0296277, 0.0206704, 0.0299115, -0.0056387, 0.0054208
3: 0.0068686, 0.0166799, 0.0063097, 0.0165808, -0.0067777, 0.0077299
4: -0.0173548, -0.0077806, -0.0173488, -0.0074310, -0.0079713, 0.0073902
5: 0.0135330, 0.0254268, 0.0127622, 0.0253943, -0.0089901, 0.0101872
6: 0.0054314, 0.0144829, 0.0049685, 0.0144403, -0.0069339, 0.0076806
7: -0.0220363, -0.0123763, -0.0220428, -0.0117536, -0.0088055, 0.0078040
8: 0.0085488, 0.0183602, 0.0077874, 0.0182481, -0.0067030, 0.0079282
9: 0.9029542, 0.9465036, 0.9032502, 0.9480295, -0.0302029, 0.0273907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162937
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162937
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040335, -0.0012920, -0.0026140, 0.0031504
1: 0.0191266, 0.0318422, 0.0184124, 0.0316567, -0.0071993, 0.0084897
2: 0.0212754, 0.0299287, 0.0207641, 0.0298513, -0.0054131, 0.0057164
3: 0.0073556, 0.0171634, 0.0065156, 0.0165765, -0.0067786, 0.0085557
4: -0.0181147, -0.0084533, -0.0173324, -0.0076157, -0.0092342, 0.0072816
5: 0.0141124, 0.0261697, 0.0130340, 0.0253800, -0.0090241, 0.0113676
6: 0.0059671, 0.0151281, 0.0051592, 0.0144287, -0.0069024, 0.0087465
7: -0.0225881, -0.0129339, -0.0220254, -0.0120005, -0.0098326, 0.0078410
8: 0.0088531, 0.0187670, 0.0080115, 0.0182481, -0.0067918, 0.0084673
9: 0.9011081, 0.9439582, 0.9032757, 0.9473053, -0.0338253, 0.0269577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0133529
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143201, upper bound: 0.0143757
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0040697, -0.0012598, -0.0026178, 0.0026537
1: 0.0188723, 0.0317525, 0.0180098, 0.0314699, -0.0070113, 0.0086608
2: 0.0208139, 0.0296277, 0.0205092, 0.0297685, -0.0055260, 0.0056269
3: 0.0068686, 0.0166799, 0.0062548, 0.0164433, -0.0066511, 0.0078835
4: -0.0173548, -0.0077806, -0.0172217, -0.0073388, -0.0081476, 0.0072730
5: 0.0135330, 0.0254268, 0.0127029, 0.0252562, -0.0088538, 0.0103403
6: 0.0054314, 0.0144829, 0.0049108, 0.0143212, -0.0068210, 0.0078284
7: -0.0220363, -0.0123763, -0.0219011, -0.0116561, -0.0089584, 0.0076877
8: 0.0085488, 0.0183602, 0.0077700, 0.0181385, -0.0065960, 0.0080298
9: 0.9029542, 0.9465036, 0.9038395, 0.9485210, -0.0310089, 0.0268555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0163131
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0163131
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040296, -0.0013139, -0.0025921, 0.0031465
1: 0.0191266, 0.0318422, 0.0182268, 0.0314699, -0.0070415, 0.0086710
2: 0.0212754, 0.0299287, 0.0205986, 0.0297071, -0.0052963, 0.0059203
3: 0.0073556, 0.0171634, 0.0064553, 0.0164391, -0.0066486, 0.0087059
4: -0.0181147, -0.0084533, -0.0172047, -0.0075214, -0.0094151, 0.0071615
5: 0.0141124, 0.0261697, 0.0129655, 0.0252417, -0.0088840, 0.0115209
6: 0.0059671, 0.0151281, 0.0050981, 0.0143095, -0.0067864, 0.0088909
7: -0.0225881, -0.0129339, -0.0218833, -0.0118984, -0.0099902, 0.0077225
8: 0.0088531, 0.0187670, 0.0079905, 0.0181385, -0.0066816, 0.0085688
9: 0.9011081, 0.9439582, 0.9038648, 0.9478132, -0.0346259, 0.0264102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148925, upper bound: 0.0133622
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147560, upper bound: 0.0144212
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0040723, -0.0012367, -0.0026369, 0.0026465
1: 0.0186864, 0.0315551, 0.0181921, 0.0316567, -0.0073469, 0.0083430
2: 0.0206598, 0.0294776, 0.0206704, 0.0299115, -0.0058389, 0.0053051
3: 0.0068079, 0.0165444, 0.0063097, 0.0165808, -0.0069167, 0.0076127
4: -0.0172286, -0.0076942, -0.0173488, -0.0074310, -0.0078491, 0.0075541
5: 0.0134717, 0.0252920, 0.0127622, 0.0253943, -0.0091287, 0.0100534
6: 0.0053748, 0.0143666, 0.0049685, 0.0144403, -0.0070647, 0.0075682
7: -0.0218950, -0.0122861, -0.0220428, -0.0117536, -0.0086786, 0.0079547
8: 0.0085235, 0.0182502, 0.0077874, 0.0182481, -0.0067928, 0.0078216
9: 0.9035680, 0.9469696, 0.9032502, 0.9480295, -0.0296899, 0.0281377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166234
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166234
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040340, -0.0012913, -0.0026113, 0.0031401
1: 0.0189327, 0.0316482, 0.0184094, 0.0316567, -0.0073945, 0.0083396
2: 0.0211074, 0.0297783, 0.0207627, 0.0298521, -0.0056190, 0.0056131
3: 0.0072798, 0.0170151, 0.0065128, 0.0165765, -0.0069363, 0.0084282
4: -0.0179822, -0.0083385, -0.0173326, -0.0076132, -0.0091124, 0.0074629
5: 0.0140276, 0.0260212, 0.0130304, 0.0253802, -0.0091836, 0.0112299
6: 0.0058922, 0.0149942, 0.0051566, 0.0144288, -0.0070547, 0.0086317
7: -0.0224438, -0.0128171, -0.0220256, -0.0119972, -0.0096997, 0.0080027
8: 0.0088164, 0.0186461, 0.0080085, 0.0182481, -0.0068935, 0.0083607
9: 0.9016901, 0.9445072, 0.9032754, 0.9473151, -0.0332712, 0.0277952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0140092
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0148009
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0040697, -0.0012598, -0.0026138, 0.0026438
1: 0.0186864, 0.0315551, 0.0180098, 0.0314699, -0.0071274, 0.0084673
2: 0.0206598, 0.0294776, 0.0205092, 0.0297685, -0.0056350, 0.0054272
3: 0.0068079, 0.0165444, 0.0062548, 0.0164433, -0.0067947, 0.0077717
4: -0.0172286, -0.0076942, -0.0172217, -0.0073388, -0.0080176, 0.0074277
5: 0.0134717, 0.0252920, 0.0127029, 0.0252562, -0.0089876, 0.0102024
6: 0.0053748, 0.0143666, 0.0049108, 0.0143212, -0.0069528, 0.0077170
7: -0.0218950, -0.0122861, -0.0219011, -0.0116561, -0.0088160, 0.0078185
8: 0.0085235, 0.0182502, 0.0077700, 0.0181385, -0.0066962, 0.0079306
9: 0.9035680, 0.9469696, 0.9038395, 0.9485210, -0.0304343, 0.0275411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040293, -0.0013143, -0.0025882, 0.0031355
1: 0.0189327, 0.0316482, 0.0182286, 0.0314699, -0.0071574, 0.0084622
2: 0.0211074, 0.0297783, 0.0205994, 0.0297066, -0.0054115, 0.0057254
3: 0.0072798, 0.0170151, 0.0064570, 0.0164390, -0.0068093, 0.0085809
4: -0.0179822, -0.0083385, -0.0172046, -0.0075229, -0.0092830, 0.0073320
5: 0.0140276, 0.0260212, 0.0129676, 0.0252415, -0.0090336, 0.0113766
6: 0.0058922, 0.0149942, 0.0050997, 0.0143094, -0.0069375, 0.0087745
7: -0.0224438, -0.0128171, -0.0218832, -0.0119004, -0.0098429, 0.0078607
8: 0.0088164, 0.0186461, 0.0079923, 0.0181385, -0.0067932, 0.0084673
9: 0.9016901, 0.9445072, 0.9038650, 0.9478071, -0.0340038, 0.0271609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154559, upper bound: 0.0144811
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153784, upper bound: 0.0155518
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0041025, -0.0011603, -0.0027174, 0.0026866
1: 0.0188723, 0.0317525, 0.0181025, 0.0317525, -0.0071647, 0.0084670
2: 0.0208139, 0.0296277, 0.0206597, 0.0300119, -0.0057591, 0.0053608
3: 0.0068686, 0.0166799, 0.0062241, 0.0167009, -0.0067905, 0.0077674
4: -0.0173548, -0.0077806, -0.0174383, -0.0073978, -0.0080312, 0.0074517
5: 0.0135330, 0.0254268, 0.0126548, 0.0254993, -0.0090344, 0.0103169
6: 0.0054314, 0.0144829, 0.0049088, 0.0145414, -0.0069808, 0.0077557
7: -0.0220363, -0.0123763, -0.0221261, -0.0116642, -0.0089533, 0.0078592
8: 0.0085488, 0.0183602, 0.0076708, 0.0183602, -0.0067157, 0.0079686
9: 0.9029542, 0.9465036, 0.9028265, 0.9482073, -0.0301493, 0.0274180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162411
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162411
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040298, -0.0012821, -0.0026239, 0.0031467
1: 0.0191266, 0.0318422, 0.0185604, 0.0317525, -0.0072508, 0.0083271
2: 0.0212754, 0.0299287, 0.0208596, 0.0298980, -0.0054520, 0.0056639
3: 0.0073556, 0.0171634, 0.0066626, 0.0166933, -0.0068093, 0.0085697
4: -0.0181147, -0.0084533, -0.0174073, -0.0077719, -0.0092820, 0.0073143
5: 0.0141124, 0.0261697, 0.0132040, 0.0254727, -0.0090584, 0.0114083
6: 0.0059671, 0.0151281, 0.0053010, 0.0145201, -0.0069306, 0.0087861
7: -0.0225881, -0.0129339, -0.0220932, -0.0121450, -0.0098917, 0.0078788
8: 0.0088531, 0.0187670, 0.0081419, 0.0183602, -0.0068229, 0.0084736
9: 0.9011081, 0.9439582, 0.9028730, 0.9466760, -0.0338025, 0.0270808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0133529
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143201, upper bound: 0.0143237
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0041014, -0.0011803, -0.0026974, 0.0026855
1: 0.0188723, 0.0317525, 0.0179118, 0.0315551, -0.0070089, 0.0086689
2: 0.0208139, 0.0296277, 0.0204978, 0.0298719, -0.0056554, 0.0055743
3: 0.0068686, 0.0166799, 0.0061570, 0.0165658, -0.0066643, 0.0079361
4: -0.0173548, -0.0077806, -0.0173152, -0.0073008, -0.0082199, 0.0073339
5: 0.0135330, 0.0254268, 0.0125842, 0.0253670, -0.0088981, 0.0104943
6: 0.0054314, 0.0144829, 0.0048462, 0.0144273, -0.0068677, 0.0079161
7: -0.0220363, -0.0123763, -0.0219862, -0.0115578, -0.0091235, 0.0077433
8: 0.0085488, 0.0183602, 0.0076409, 0.0182502, -0.0066085, 0.0080927
9: 0.9029542, 0.9465036, 0.9034392, 0.9487171, -0.0310182, 0.0268840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0162741
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0162741
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040254, -0.0013041, -0.0026019, 0.0031423
1: 0.0191266, 0.0318422, 0.0183768, 0.0315551, -0.0070898, 0.0084994
2: 0.0212754, 0.0299287, 0.0206944, 0.0297475, -0.0053293, 0.0058690
3: 0.0073556, 0.0171634, 0.0065958, 0.0165579, -0.0066795, 0.0087214
4: -0.0181147, -0.0084533, -0.0172831, -0.0076687, -0.0094640, 0.0071938
5: 0.0141124, 0.0261697, 0.0131293, 0.0253392, -0.0089180, 0.0115689
6: 0.0059671, 0.0151281, 0.0052337, 0.0144049, -0.0068143, 0.0089364
7: -0.0225881, -0.0129339, -0.0219524, -0.0120414, -0.0100503, 0.0077605
8: 0.0088531, 0.0187670, 0.0081173, 0.0182502, -0.0067116, 0.0085725
9: 0.9011081, 0.9439582, 0.9034869, 0.9472061, -0.0346255, 0.0265337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148925, upper bound: 0.0133622
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147560, upper bound: 0.0143735
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0041036, -0.0011578, -0.0027157, 0.0026777
1: 0.0186864, 0.0315551, 0.0180988, 0.0317525, -0.0073465, 0.0083226
2: 0.0206598, 0.0294776, 0.0206590, 0.0300137, -0.0059744, 0.0052516
3: 0.0068079, 0.0165444, 0.0062210, 0.0167010, -0.0069408, 0.0076455
4: -0.0172286, -0.0076942, -0.0174387, -0.0073960, -0.0079177, 0.0076275
5: 0.0134717, 0.0252920, 0.0126506, 0.0254996, -0.0091956, 0.0101853
6: 0.0053748, 0.0143666, 0.0049063, 0.0145417, -0.0071281, 0.0076469
7: -0.0218950, -0.0122861, -0.0221265, -0.0116607, -0.0088426, 0.0080206
8: 0.0085235, 0.0182502, 0.0076666, 0.0183602, -0.0068113, 0.0078680
9: 0.9035680, 0.9469696, 0.9028260, 0.9482155, -0.0296241, 0.0282178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166093
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166093
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040317, -0.0012793, -0.0026233, 0.0031378
1: 0.0189327, 0.0316482, 0.0185479, 0.0317525, -0.0074427, 0.0082089
2: 0.0211074, 0.0297783, 0.0208541, 0.0299013, -0.0056686, 0.0055861
3: 0.0072798, 0.0170151, 0.0066511, 0.0166935, -0.0069698, 0.0084567
4: -0.0179822, -0.0083385, -0.0174081, -0.0077623, -0.0091695, 0.0074967
5: 0.0140276, 0.0260212, 0.0131895, 0.0254735, -0.0092206, 0.0112812
6: 0.0058922, 0.0149942, 0.0052908, 0.0145207, -0.0070825, 0.0086793
7: -0.0224438, -0.0128171, -0.0220941, -0.0121323, -0.0097740, 0.0080411
8: 0.0088164, 0.0186461, 0.0081291, 0.0183602, -0.0069261, 0.0083738
9: 0.9016901, 0.9445072, 0.9028718, 0.9467169, -0.0332766, 0.0279354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0140040
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0147655
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0040970, -0.0011898, -0.0026838, 0.0026711
1: 0.0186864, 0.0315551, 0.0179267, 0.0315551, -0.0071315, 0.0084389
2: 0.0206598, 0.0294776, 0.0205008, 0.0298641, -0.0057596, 0.0053651
3: 0.0068079, 0.0165444, 0.0061697, 0.0165654, -0.0068206, 0.0077956
4: -0.0172286, -0.0076942, -0.0173135, -0.0073082, -0.0080823, 0.0074999
5: 0.0134717, 0.0252920, 0.0126021, 0.0253656, -0.0090524, 0.0103252
6: 0.0053748, 0.0143666, 0.0048569, 0.0144262, -0.0070161, 0.0077888
7: -0.0218950, -0.0122861, -0.0219845, -0.0115721, -0.0089726, 0.0078817
8: 0.0085235, 0.0182502, 0.0076580, 0.0182502, -0.0067151, 0.0079719
9: 0.9035680, 0.9469696, 0.9034415, 0.9486838, -0.0303555, 0.0276076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040249, -0.0013048, -0.0025978, 0.0031310
1: 0.0189327, 0.0316482, 0.0183800, 0.0315551, -0.0072140, 0.0083005
2: 0.0211074, 0.0297783, 0.0206958, 0.0297467, -0.0054482, 0.0056797
3: 0.0072798, 0.0170151, 0.0065987, 0.0165579, -0.0068431, 0.0086056
4: -0.0179822, -0.0083385, -0.0172829, -0.0076712, -0.0093363, 0.0073637
5: 0.0140276, 0.0260212, 0.0131330, 0.0253390, -0.0090700, 0.0114247
6: 0.0058922, 0.0149942, 0.0052363, 0.0144047, -0.0069650, 0.0088234
7: -0.0224438, -0.0128171, -0.0219522, -0.0120447, -0.0099046, 0.0078995
8: 0.0088164, 0.0186461, 0.0081206, 0.0182502, -0.0068240, 0.0084732
9: 0.9016901, 0.9445072, 0.9034872, 0.9471958, -0.0340112, 0.0273030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154559, upper bound: 0.0144782
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153784, upper bound: 0.0155196
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0041413, -0.0011326, -0.0027446, 0.0027184
1: 0.0188506, 0.0316567, 0.0170758, 0.0320063, -0.0078614, 0.0094614
2: 0.0208018, 0.0295784, 0.0200154, 0.0301909, -0.0062035, 0.0061473
3: 0.0068664, 0.0165629, 0.0053988, 0.0168827, -0.0071022, 0.0084823
4: -0.0172798, -0.0077549, -0.0176915, -0.0065340, -0.0087435, 0.0076825
5: 0.0135111, 0.0253347, 0.0115979, 0.0257699, -0.0093666, 0.0111046
6: 0.0054180, 0.0143917, 0.0041013, 0.0147464, -0.0071890, 0.0083914
7: -0.0219702, -0.0123623, -0.0223849, -0.0107666, -0.0095358, 0.0081091
8: 0.0085528, 0.0182481, 0.0069091, 0.0185007, -0.0070316, 0.0086101
9: 0.9033568, 0.9465656, 0.9018377, 0.9519727, -0.0340251, 0.0291791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172214
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172214
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0040612, -0.0012713, -0.0025871, 0.0031864
1: 0.0192105, 0.0317763, 0.0175321, 0.0320063, -0.0077457, 0.0092696
2: 0.0212826, 0.0298302, 0.0202013, 0.0300767, -0.0058897, 0.0063258
3: 0.0074236, 0.0171280, 0.0058148, 0.0168743, -0.0070095, 0.0091897
4: -0.0181071, -0.0084861, -0.0176565, -0.0068882, -0.0098881, 0.0074847
5: 0.0142199, 0.0261498, 0.0121536, 0.0257402, -0.0092257, 0.0120716
6: 0.0060204, 0.0151158, 0.0044848, 0.0147225, -0.0070549, 0.0093193
7: -0.0225776, -0.0130546, -0.0223503, -0.0112503, -0.0103842, 0.0079648
8: 0.0089573, 0.0187225, 0.0073746, 0.0185007, -0.0069840, 0.0090267
9: 0.9012735, 0.9438298, 0.9018893, 0.9505221, -0.0372970, 0.0285722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145934, upper bound: 0.0141791
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144085, upper bound: 0.0152460
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0041374, -0.0011655, -0.0027117, 0.0027145
1: 0.0188506, 0.0316567, 0.0169144, 0.0318150, -0.0077250, 0.0096398
2: 0.0208018, 0.0295784, 0.0198633, 0.0300454, -0.0060747, 0.0063394
3: 0.0068664, 0.0165629, 0.0053300, 0.0167500, -0.0070020, 0.0086159
4: -0.0172798, -0.0077549, -0.0175729, -0.0064338, -0.0089011, 0.0075876
5: 0.0135111, 0.0253347, 0.0115280, 0.0256327, -0.0092632, 0.0112369
6: 0.0054180, 0.0143917, 0.0040348, 0.0146375, -0.0071047, 0.0085181
7: -0.0219702, -0.0123623, -0.0222471, -0.0106774, -0.0096805, 0.0080081
8: 0.0085528, 0.0182481, 0.0068849, 0.0183949, -0.0069421, 0.0086857
9: 0.9033568, 0.9465656, 0.9024426, 0.9524551, -0.0347923, 0.0287214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166275, upper bound: 0.0172366
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166275, upper bound: 0.0172366
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0040551, -0.0012963, -0.0025622, 0.0031803
1: 0.0192105, 0.0317763, 0.0173650, 0.0318150, -0.0076080, 0.0094509
2: 0.0212826, 0.0298302, 0.0200483, 0.0299293, -0.0057604, 0.0065210
3: 0.0074236, 0.0171280, 0.0057474, 0.0167417, -0.0069084, 0.0093275
4: -0.0181071, -0.0084861, -0.0175388, -0.0067892, -0.0100491, 0.0073883
5: 0.0142199, 0.0261498, 0.0120866, 0.0256045, -0.0091195, 0.0122071
6: 0.0060204, 0.0151158, 0.0044197, 0.0146144, -0.0069677, 0.0094473
7: -0.0225776, -0.0130546, -0.0222129, -0.0111640, -0.0105302, 0.0078610
8: 0.0089573, 0.0187225, 0.0073531, 0.0183949, -0.0068917, 0.0091030
9: 0.9012735, 0.9438298, 0.9024935, 0.9510142, -0.0380648, 0.0281040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149556, upper bound: 0.0141791
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148069, upper bound: 0.0152788
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0041413, -0.0011326, -0.0027407, 0.0027084
1: 0.0186610, 0.0314699, 0.0170758, 0.0320063, -0.0080411, 0.0093112
2: 0.0206414, 0.0294323, 0.0200154, 0.0301909, -0.0064132, 0.0060358
3: 0.0068071, 0.0164252, 0.0053988, 0.0168827, -0.0072497, 0.0083555
4: -0.0171503, -0.0076708, -0.0176915, -0.0065340, -0.0086275, 0.0078554
5: 0.0134471, 0.0251937, 0.0115979, 0.0257699, -0.0095202, 0.0109666
6: 0.0053620, 0.0142708, 0.0041013, 0.0147464, -0.0073342, 0.0082790
7: -0.0218249, -0.0122729, -0.0223849, -0.0107666, -0.0094196, 0.0082661
8: 0.0085303, 0.0181385, 0.0069091, 0.0185007, -0.0071250, 0.0085030
9: 0.9039483, 0.9470383, 0.9018377, 0.9519727, -0.0334855, 0.0299674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0176918
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0176918
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0040633, -0.0012686, -0.0025848, 0.0031783
1: 0.0190119, 0.0315876, 0.0175198, 0.0320063, -0.0079466, 0.0091506
2: 0.0211144, 0.0296830, 0.0201960, 0.0300797, -0.0060983, 0.0062446
3: 0.0073461, 0.0169788, 0.0058035, 0.0168745, -0.0071657, 0.0090756
4: -0.0179715, -0.0083750, -0.0176574, -0.0068784, -0.0097705, 0.0076643
5: 0.0141374, 0.0260014, 0.0121382, 0.0257410, -0.0093805, 0.0119443
6: 0.0059462, 0.0149819, 0.0044742, 0.0147231, -0.0072028, 0.0092115
7: -0.0224307, -0.0129384, -0.0223512, -0.0112370, -0.0102572, 0.0081192
8: 0.0089189, 0.0186002, 0.0073619, 0.0185007, -0.0070867, 0.0089365
9: 0.9018508, 0.9443790, 0.9018881, 0.9505620, -0.0367811, 0.0294101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146453, upper bound: 0.0149019
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144498, upper bound: 0.0158381
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0041374, -0.0011655, -0.0027077, 0.0027046
1: 0.0186610, 0.0314699, 0.0169144, 0.0318150, -0.0078432, 0.0094099
2: 0.0206414, 0.0294323, 0.0198633, 0.0300454, -0.0062052, 0.0061386
3: 0.0068071, 0.0164252, 0.0053300, 0.0167500, -0.0071556, 0.0084968
4: -0.0171503, -0.0076708, -0.0175729, -0.0064338, -0.0087724, 0.0077487
5: 0.0134471, 0.0251937, 0.0115280, 0.0256327, -0.0094094, 0.0110890
6: 0.0053620, 0.0142708, 0.0040348, 0.0146375, -0.0072513, 0.0084073
7: -0.0218249, -0.0122729, -0.0222471, -0.0106774, -0.0095376, 0.0081353
8: 0.0085303, 0.0181385, 0.0068849, 0.0183949, -0.0070463, 0.0085950
9: 0.9039483, 0.9470383, 0.9024426, 0.9524551, -0.0341695, 0.0294361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169851, upper bound: 0.0180658
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169851, upper bound: 0.0180658
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0040560, -0.0012951, -0.0025583, 0.0031709
1: 0.0190119, 0.0315876, 0.0173599, 0.0318150, -0.0077251, 0.0092286
2: 0.0211144, 0.0296830, 0.0200461, 0.0299306, -0.0058902, 0.0063343
3: 0.0073461, 0.0169788, 0.0057427, 0.0167418, -0.0070683, 0.0092153
4: -0.0179715, -0.0083750, -0.0175392, -0.0067852, -0.0099157, 0.0075539
5: 0.0141374, 0.0260014, 0.0120802, 0.0256048, -0.0092639, 0.0120632
6: 0.0059462, 0.0149819, 0.0044153, 0.0146147, -0.0071141, 0.0093358
7: -0.0224307, -0.0129384, -0.0222133, -0.0111585, -0.0103698, 0.0079888
8: 0.0089189, 0.0186002, 0.0073478, 0.0183949, -0.0070057, 0.0090219
9: 0.9018508, 0.9443790, 0.9024929, 0.9510307, -0.0374501, 0.0288549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155451, upper bound: 0.0151838
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154750, upper bound: 0.0164999
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0041761, -0.0010325, -0.0028447, 0.0027532
1: 0.0188506, 0.0316567, 0.0169682, 0.0321007, -0.0080053, 0.0096277
2: 0.0208018, 0.0295784, 0.0200045, 0.0303035, -0.0063832, 0.0061613
3: 0.0068664, 0.0165629, 0.0053004, 0.0169994, -0.0072102, 0.0085749
4: -0.0172798, -0.0077549, -0.0177966, -0.0064833, -0.0087817, 0.0077485
5: 0.0135111, 0.0253347, 0.0114692, 0.0259043, -0.0094761, 0.0112370
6: 0.0054180, 0.0143917, 0.0040265, 0.0148571, -0.0072694, 0.0084560
7: -0.0219702, -0.0123623, -0.0224730, -0.0106558, -0.0096619, 0.0081478
8: 0.0085528, 0.0182481, 0.0067799, 0.0186110, -0.0071536, 0.0087481
9: 0.9033568, 0.9465656, 0.9013865, 0.9521797, -0.0342079, 0.0296575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0041192, -0.0011589, -0.0026996, 0.0032444
1: 0.0192105, 0.0317763, 0.0173121, 0.0321007, -0.0078264, 0.0095487
2: 0.0212826, 0.0298302, 0.0201406, 0.0302204, -0.0060714, 0.0063813
3: 0.0074236, 0.0171280, 0.0056140, 0.0169939, -0.0070324, 0.0093795
4: -0.0181071, -0.0084861, -0.0177729, -0.0067453, -0.0100126, 0.0075019
5: 0.0142199, 0.0261498, 0.0118854, 0.0258850, -0.0092561, 0.0123404
6: 0.0060204, 0.0151158, 0.0043135, 0.0148408, -0.0070735, 0.0094773
7: -0.0225776, -0.0130546, -0.0224489, -0.0110122, -0.0106257, 0.0079775
8: 0.0089573, 0.0187225, 0.0071348, 0.0186110, -0.0070170, 0.0092755
9: 0.9012735, 0.9438298, 0.9014195, 0.9511265, -0.0378089, 0.0287098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038772, -0.0014229, -0.0041711, -0.0010655, -0.0028117, 0.0027482
1: 0.0188506, 0.0316567, 0.0168123, 0.0318989, -0.0078519, 0.0098088
2: 0.0208018, 0.0295784, 0.0198550, 0.0301546, -0.0062472, 0.0063487
3: 0.0068664, 0.0165629, 0.0052378, 0.0168694, -0.0071119, 0.0087054
4: -0.0172798, -0.0077549, -0.0176739, -0.0063858, -0.0089382, 0.0076584
5: 0.0135111, 0.0253347, 0.0114008, 0.0257650, -0.0093706, 0.0113692
6: 0.0054180, 0.0143917, 0.0039620, 0.0147440, -0.0071830, 0.0085807
7: -0.0219702, -0.0123623, -0.0223307, -0.0105696, -0.0098092, 0.0080528
8: 0.0085528, 0.0182481, 0.0067627, 0.0185066, -0.0070658, 0.0088235
9: 0.9033568, 0.9465656, 0.9020119, 0.9526696, -0.0349668, 0.0292112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166058, upper bound: 0.0171677
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166058, upper bound: 0.0171677
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0008747, -0.0041134, -0.0011906, -0.0026679, 0.0032386
1: 0.0192105, 0.0317763, 0.0171541, 0.0318989, -0.0076811, 0.0097341
2: 0.0212826, 0.0298302, 0.0199891, 0.0300694, -0.0059374, 0.0065736
3: 0.0074236, 0.0171280, 0.0055508, 0.0168639, -0.0069299, 0.0095156
4: -0.0181071, -0.0084861, -0.0176501, -0.0066473, -0.0101737, 0.0074047
5: 0.0142199, 0.0261498, 0.0118183, 0.0257450, -0.0091475, 0.0124755
6: 0.0060204, 0.0151158, 0.0042488, 0.0147272, -0.0069865, 0.0096044
7: -0.0225776, -0.0130546, -0.0223063, -0.0109261, -0.0107722, 0.0078765
8: 0.0089573, 0.0187225, 0.0071149, 0.0185066, -0.0069260, 0.0093519
9: 0.9012735, 0.9438298, 0.9020451, 0.9516124, -0.0385737, 0.0282308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149444, upper bound: 0.0141426
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147798, upper bound: 0.0151997
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0041761, -0.0010325, -0.0028408, 0.0027432
1: 0.0186610, 0.0314699, 0.0169682, 0.0321007, -0.0081851, 0.0094775
2: 0.0206414, 0.0294323, 0.0200045, 0.0303035, -0.0065929, 0.0060498
3: 0.0068071, 0.0164252, 0.0053004, 0.0169994, -0.0073577, 0.0084481
4: -0.0171503, -0.0076708, -0.0177966, -0.0064833, -0.0086657, 0.0079213
5: 0.0134471, 0.0251937, 0.0114692, 0.0259043, -0.0096297, 0.0110989
6: 0.0053620, 0.0142708, 0.0040265, 0.0148571, -0.0074146, 0.0083436
7: -0.0218249, -0.0122729, -0.0224730, -0.0106558, -0.0095458, 0.0083048
8: 0.0085303, 0.0181385, 0.0067799, 0.0186110, -0.0072471, 0.0086411
9: 0.9039483, 0.9470383, 0.9013865, 0.9521797, -0.0336683, 0.0304458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0041208, -0.0011556, -0.0026977, 0.0032358
1: 0.0190119, 0.0315876, 0.0173020, 0.0321007, -0.0080272, 0.0094273
2: 0.0211144, 0.0296830, 0.0201364, 0.0302228, -0.0062790, 0.0062992
3: 0.0073461, 0.0169788, 0.0056047, 0.0169941, -0.0071886, 0.0092630
4: -0.0179715, -0.0083750, -0.0177736, -0.0067373, -0.0098935, 0.0076814
5: 0.0141374, 0.0260014, 0.0118731, 0.0258856, -0.0094109, 0.0122107
6: 0.0059462, 0.0149819, 0.0043049, 0.0148412, -0.0072214, 0.0093677
7: -0.0224307, -0.0129384, -0.0224496, -0.0110018, -0.0104966, 0.0081319
8: 0.0089189, 0.0186002, 0.0071245, 0.0186110, -0.0071197, 0.0091831
9: 0.9018508, 0.9443790, 0.9014186, 0.9511580, -0.0372867, 0.0295476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038733, -0.0014329, -0.0041711, -0.0010655, -0.0028077, 0.0027382
1: 0.0186610, 0.0314699, 0.0168123, 0.0318989, -0.0079785, 0.0095759
2: 0.0206414, 0.0294323, 0.0198550, 0.0301546, -0.0063845, 0.0061506
3: 0.0068071, 0.0164252, 0.0052378, 0.0168694, -0.0072662, 0.0085857
4: -0.0171503, -0.0076708, -0.0176739, -0.0063858, -0.0088097, 0.0078182
5: 0.0134471, 0.0251937, 0.0114008, 0.0257650, -0.0095166, 0.0112216
6: 0.0053620, 0.0142708, 0.0039620, 0.0147440, -0.0073297, 0.0084699
7: -0.0218249, -0.0122729, -0.0223307, -0.0105696, -0.0096653, 0.0081773
8: 0.0085303, 0.0181385, 0.0067627, 0.0185066, -0.0071701, 0.0087337
9: 0.9039483, 0.9470383, 0.9020119, 0.9526696, -0.0343421, 0.0299203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169505, upper bound: 0.0180077
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169505, upper bound: 0.0180077
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038533, -0.0008851, -0.0041140, -0.0011893, -0.0026640, 0.0032289
1: 0.0190119, 0.0315876, 0.0171504, 0.0318989, -0.0078004, 0.0095068
2: 0.0211144, 0.0296830, 0.0199876, 0.0300703, -0.0060721, 0.0063870
3: 0.0073461, 0.0169788, 0.0055473, 0.0168639, -0.0070898, 0.0094022
4: -0.0179715, -0.0083750, -0.0176504, -0.0066443, -0.0100393, 0.0075701
5: 0.0141374, 0.0260014, 0.0118138, 0.0257452, -0.0092921, 0.0123303
6: 0.0059462, 0.0149819, 0.0042456, 0.0147274, -0.0071329, 0.0094919
7: -0.0224307, -0.0129384, -0.0223066, -0.0109221, -0.0106109, 0.0080029
8: 0.0089189, 0.0186002, 0.0071110, 0.0185066, -0.0070402, 0.0092703
9: 0.9018508, 0.9443790, 0.9020448, 0.9516239, -0.0379545, 0.0289852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155395, upper bound: 0.0151612
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154391, upper bound: 0.0164340
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0041413, -0.0011326, -0.0027451, 0.0027253
1: 0.0188723, 0.0317525, 0.0170758, 0.0320063, -0.0079500, 0.0097424
2: 0.0208139, 0.0296277, 0.0200154, 0.0301909, -0.0062195, 0.0062790
3: 0.0068686, 0.0166799, 0.0053988, 0.0168827, -0.0071434, 0.0086364
4: -0.0173548, -0.0077806, -0.0176915, -0.0065340, -0.0088233, 0.0076755
5: 0.0135330, 0.0254268, 0.0115979, 0.0257699, -0.0093822, 0.0112220
6: 0.0054314, 0.0144829, 0.0041013, 0.0147464, -0.0071956, 0.0084835
7: -0.0220363, -0.0123763, -0.0223849, -0.0107666, -0.0096023, 0.0081011
8: 0.0085488, 0.0183602, 0.0069091, 0.0185007, -0.0070817, 0.0088058
9: 0.9029542, 0.9465036, 0.9018377, 0.9519727, -0.0346648, 0.0293541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171897
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171897
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0041025, -0.0012070, -0.0026990, 0.0032195
1: 0.0191266, 0.0318422, 0.0172945, 0.0320063, -0.0079566, 0.0097179
2: 0.0212754, 0.0299287, 0.0201018, 0.0301358, -0.0059908, 0.0065461
3: 0.0073556, 0.0171634, 0.0055960, 0.0168787, -0.0071017, 0.0094114
4: -0.0181147, -0.0084533, -0.0176749, -0.0067008, -0.0100356, 0.0075323
5: 0.0141124, 0.0261697, 0.0118613, 0.0257558, -0.0093722, 0.0123335
6: 0.0059671, 0.0151281, 0.0042824, 0.0147350, -0.0071286, 0.0094916
7: -0.0225881, -0.0129339, -0.0223684, -0.0109958, -0.0105878, 0.0081106
8: 0.0088531, 0.0187670, 0.0071305, 0.0185007, -0.0071304, 0.0092986
9: 0.9011081, 0.9439582, 0.9018624, 0.9512877, -0.0380543, 0.0287494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0141152
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0152163
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0041374, -0.0011655, -0.0027121, 0.0027215
1: 0.0188723, 0.0317525, 0.0169144, 0.0318150, -0.0078137, 0.0099208
2: 0.0208139, 0.0296277, 0.0198633, 0.0300454, -0.0060907, 0.0064711
3: 0.0068686, 0.0166799, 0.0053300, 0.0167500, -0.0070431, 0.0087699
4: -0.0173548, -0.0077806, -0.0175729, -0.0064338, -0.0089809, 0.0075806
5: 0.0135330, 0.0254268, 0.0115280, 0.0256327, -0.0092789, 0.0113543
6: 0.0054314, 0.0144829, 0.0040348, 0.0146375, -0.0071113, 0.0086102
7: -0.0220363, -0.0123763, -0.0222471, -0.0106774, -0.0097471, 0.0080001
8: 0.0085488, 0.0183602, 0.0068849, 0.0183949, -0.0069923, 0.0088815
9: 0.9029542, 0.9465036, 0.9024426, 0.9524551, -0.0354319, 0.0288965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0172056
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0172056
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040979, -0.0012352, -0.0026708, 0.0032148
1: 0.0191266, 0.0318422, 0.0171274, 0.0318150, -0.0078191, 0.0098985
2: 0.0212754, 0.0299287, 0.0199489, 0.0299913, -0.0058636, 0.0067394
3: 0.0073556, 0.0171634, 0.0055273, 0.0167459, -0.0070006, 0.0095476
4: -0.0181147, -0.0084533, -0.0175561, -0.0066025, -0.0101931, 0.0074355
5: 0.0141124, 0.0261697, 0.0117911, 0.0256188, -0.0092660, 0.0124670
6: 0.0059671, 0.0151281, 0.0042159, 0.0146261, -0.0070415, 0.0096181
7: -0.0225881, -0.0129339, -0.0222302, -0.0109067, -0.0107346, 0.0080072
8: 0.0088531, 0.0187670, 0.0071062, 0.0183949, -0.0070382, 0.0093759
9: 0.9011081, 0.9439582, 0.9024677, 0.9517736, -0.0388329, 0.0282806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148851, upper bound: 0.0141152
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147242, upper bound: 0.0152440
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0041413, -0.0011326, -0.0027410, 0.0027154
1: 0.0186864, 0.0315551, 0.0170758, 0.0320063, -0.0081306, 0.0096068
2: 0.0206598, 0.0294776, 0.0200154, 0.0301909, -0.0064197, 0.0061633
3: 0.0068079, 0.0165444, 0.0053988, 0.0168827, -0.0072824, 0.0085192
4: -0.0172286, -0.0076942, -0.0176915, -0.0065340, -0.0087011, 0.0078394
5: 0.0134717, 0.0252920, 0.0115979, 0.0257699, -0.0095208, 0.0110883
6: 0.0053748, 0.0143666, 0.0041013, 0.0147464, -0.0073264, 0.0083711
7: -0.0218950, -0.0122861, -0.0223849, -0.0107666, -0.0094754, 0.0082518
8: 0.0085235, 0.0182502, 0.0069091, 0.0185007, -0.0071715, 0.0086992
9: 0.9035680, 0.9469696, 0.9018377, 0.9519727, -0.0341517, 0.0301012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176661
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176661
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0041031, -0.0012060, -0.0026966, 0.0032092
1: 0.0189327, 0.0316482, 0.0172913, 0.0320063, -0.0081518, 0.0095685
2: 0.0211074, 0.0297783, 0.0201005, 0.0301366, -0.0061966, 0.0064433
3: 0.0072798, 0.0170151, 0.0055931, 0.0168788, -0.0072594, 0.0092847
4: -0.0179822, -0.0083385, -0.0176751, -0.0066984, -0.0099146, 0.0077136
5: 0.0140276, 0.0260212, 0.0118574, 0.0257560, -0.0095318, 0.0121970
6: 0.0058922, 0.0149942, 0.0042797, 0.0147352, -0.0072809, 0.0093777
7: -0.0224438, -0.0128171, -0.0223687, -0.0109925, -0.0104556, 0.0082723
8: 0.0088164, 0.0186461, 0.0071273, 0.0185007, -0.0072321, 0.0091928
9: 0.9016901, 0.9445072, 0.9018619, 0.9512977, -0.0375038, 0.0295869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0148736
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0158139
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0041374, -0.0011655, -0.0027081, 0.0027115
1: 0.0186864, 0.0315551, 0.0169144, 0.0318150, -0.0079313, 0.0097051
2: 0.0206598, 0.0294776, 0.0198633, 0.0300454, -0.0062159, 0.0062715
3: 0.0068079, 0.0165444, 0.0053300, 0.0167500, -0.0071878, 0.0086616
4: -0.0172286, -0.0076942, -0.0175729, -0.0064338, -0.0088475, 0.0077336
5: 0.0134717, 0.0252920, 0.0115280, 0.0256327, -0.0094120, 0.0112113
6: 0.0053748, 0.0143666, 0.0040348, 0.0146375, -0.0072433, 0.0084994
7: -0.0218950, -0.0122861, -0.0222471, -0.0106774, -0.0095911, 0.0081226
8: 0.0085235, 0.0182502, 0.0068849, 0.0183949, -0.0070935, 0.0087887
9: 0.9035680, 0.9469696, 0.9024426, 0.9524551, -0.0348390, 0.0295857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180351
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180351
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040984, -0.0012342, -0.0026684, 0.0032046
1: 0.0189327, 0.0316482, 0.0171243, 0.0318150, -0.0079355, 0.0096747
2: 0.0211074, 0.0297783, 0.0199477, 0.0299921, -0.0059941, 0.0065416
3: 0.0072798, 0.0170151, 0.0055244, 0.0167459, -0.0071619, 0.0094288
4: -0.0179822, -0.0083385, -0.0175563, -0.0066000, -0.0100611, 0.0076039
5: 0.0140276, 0.0260212, 0.0117872, 0.0256190, -0.0094144, 0.0123226
6: 0.0058922, 0.0149942, 0.0042133, 0.0146263, -0.0071924, 0.0095057
7: -0.0224438, -0.0128171, -0.0222305, -0.0109034, -0.0105807, 0.0081366
8: 0.0088164, 0.0186461, 0.0071030, 0.0183949, -0.0071512, 0.0092858
9: 0.9016901, 0.9445072, 0.9024673, 0.9517835, -0.0381981, 0.0290333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154403, upper bound: 0.0151597
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153655, upper bound: 0.0164598
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0041761, -0.0010325, -0.0028452, 0.0027601
1: 0.0188723, 0.0317525, 0.0169682, 0.0321007, -0.0079198, 0.0097282
2: 0.0208139, 0.0296277, 0.0200045, 0.0303035, -0.0063373, 0.0062172
3: 0.0068686, 0.0166799, 0.0053004, 0.0169994, -0.0071535, 0.0086762
4: -0.0173548, -0.0077806, -0.0177966, -0.0064833, -0.0088908, 0.0077420
5: 0.0135330, 0.0254268, 0.0114692, 0.0259043, -0.0094310, 0.0113528
6: 0.0054314, 0.0144829, 0.0040265, 0.0148571, -0.0072464, 0.0085598
7: -0.0220363, -0.0123763, -0.0224730, -0.0106558, -0.0097491, 0.0081631
8: 0.0085488, 0.0183602, 0.0067799, 0.0186110, -0.0070866, 0.0088518
9: 0.9029542, 0.9465036, 0.9013865, 0.9521797, -0.0346086, 0.0293849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171555
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171555
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040955, -0.0012048, -0.0027012, 0.0032124
1: 0.0191266, 0.0318422, 0.0174585, 0.0321007, -0.0079819, 0.0094710
2: 0.0212754, 0.0299287, 0.0202009, 0.0301851, -0.0060220, 0.0064461
3: 0.0073556, 0.0171634, 0.0057495, 0.0169916, -0.0071294, 0.0093445
4: -0.0181147, -0.0084533, -0.0177631, -0.0068582, -0.0100116, 0.0075674
5: 0.0141124, 0.0261697, 0.0120627, 0.0258769, -0.0094081, 0.0122820
6: 0.0059671, 0.0151281, 0.0044374, 0.0148339, -0.0071582, 0.0094599
7: -0.0225881, -0.0129339, -0.0224387, -0.0111643, -0.0105670, 0.0081528
8: 0.0088531, 0.0187670, 0.0072850, 0.0186110, -0.0071564, 0.0092191
9: 0.9011081, 0.9439582, 0.9014335, 0.9506704, -0.0377283, 0.0288680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0141152
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0151795
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038777, -0.0014160, -0.0041711, -0.0010655, -0.0028121, 0.0027551
1: 0.0188723, 0.0317525, 0.0168123, 0.0318989, -0.0077849, 0.0099046
2: 0.0208139, 0.0296277, 0.0198550, 0.0301546, -0.0062105, 0.0064107
3: 0.0068686, 0.0166799, 0.0052378, 0.0168694, -0.0070538, 0.0088116
4: -0.0173548, -0.0077806, -0.0176739, -0.0063858, -0.0090496, 0.0076475
5: 0.0135330, 0.0254268, 0.0114008, 0.0257650, -0.0093276, 0.0114841
6: 0.0054314, 0.0144829, 0.0039620, 0.0147440, -0.0071622, 0.0086885
7: -0.0220363, -0.0123763, -0.0223307, -0.0105696, -0.0098945, 0.0080619
8: 0.0085488, 0.0183602, 0.0067627, 0.0185066, -0.0069968, 0.0089269
9: 0.9029542, 0.9465036, 0.9020119, 0.9526696, -0.0353833, 0.0289218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0171680
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0171680
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039060, -0.0008831, -0.0040887, -0.0012352, -0.0026708, 0.0032056
1: 0.0191266, 0.0318422, 0.0172994, 0.0318989, -0.0078448, 0.0096529
2: 0.0212754, 0.0299287, 0.0200487, 0.0300328, -0.0058881, 0.0066407
3: 0.0073556, 0.0171634, 0.0056866, 0.0168616, -0.0070285, 0.0094789
4: -0.0181147, -0.0084533, -0.0176403, -0.0067607, -0.0101708, 0.0074709
5: 0.0141124, 0.0261697, 0.0119963, 0.0257368, -0.0093018, 0.0124146
6: 0.0059671, 0.0151281, 0.0043737, 0.0147203, -0.0070711, 0.0095889
7: -0.0225881, -0.0129339, -0.0222964, -0.0110791, -0.0107106, 0.0080490
8: 0.0088531, 0.0187670, 0.0072649, 0.0185066, -0.0070630, 0.0092966
9: 0.9011081, 0.9439582, 0.9020588, 0.9511529, -0.0385058, 0.0283993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148851, upper bound: 0.0141152
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147242, upper bound: 0.0151997
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0041761, -0.0010325, -0.0028411, 0.0027502
1: 0.0186864, 0.0315551, 0.0169682, 0.0321007, -0.0081013, 0.0095771
2: 0.0206598, 0.0294776, 0.0200045, 0.0303035, -0.0065492, 0.0061068
3: 0.0068079, 0.0165444, 0.0053004, 0.0169994, -0.0073038, 0.0085495
4: -0.0172286, -0.0076942, -0.0177966, -0.0064833, -0.0087744, 0.0079177
5: 0.0134717, 0.0252920, 0.0114692, 0.0259043, -0.0095921, 0.0112149
6: 0.0053748, 0.0143666, 0.0040265, 0.0148571, -0.0073936, 0.0084472
7: -0.0218950, -0.0122861, -0.0224730, -0.0106558, -0.0096331, 0.0083244
8: 0.0085235, 0.0182502, 0.0067799, 0.0186110, -0.0071822, 0.0087451
9: 0.9035680, 0.9469696, 0.9013865, 0.9521797, -0.0340696, 0.0301846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176534
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176534
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040975, -0.0012009, -0.0027017, 0.0032037
1: 0.0189327, 0.0316482, 0.0174455, 0.0321007, -0.0081736, 0.0093550
2: 0.0211074, 0.0297783, 0.0201955, 0.0301882, -0.0062380, 0.0063702
3: 0.0072798, 0.0170151, 0.0057374, 0.0169918, -0.0072899, 0.0092345
4: -0.0179822, -0.0083385, -0.0177640, -0.0068481, -0.0099023, 0.0077499
5: 0.0140276, 0.0260212, 0.0120470, 0.0258776, -0.0095705, 0.0121581
6: 0.0058922, 0.0149942, 0.0044265, 0.0148345, -0.0073101, 0.0093558
7: -0.0224438, -0.0128171, -0.0224397, -0.0111507, -0.0104516, 0.0083152
8: 0.0088164, 0.0186461, 0.0072717, 0.0186110, -0.0072595, 0.0091220
9: 0.9016901, 0.9445072, 0.9014323, 0.9507111, -0.0372170, 0.0297229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0148721
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0157998
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0038736, -0.0014259, -0.0041711, -0.0010655, -0.0028081, 0.0027452
1: 0.0186864, 0.0315551, 0.0168123, 0.0318989, -0.0079016, 0.0096817
2: 0.0206598, 0.0294776, 0.0198550, 0.0301546, -0.0063430, 0.0062091
3: 0.0068079, 0.0165444, 0.0052378, 0.0168694, -0.0072112, 0.0086926
4: -0.0172286, -0.0076942, -0.0176739, -0.0063858, -0.0089215, 0.0078119
5: 0.0134717, 0.0252920, 0.0114008, 0.0257650, -0.0094813, 0.0113370
6: 0.0053748, 0.0143666, 0.0039620, 0.0147440, -0.0073110, 0.0085776
7: -0.0218950, -0.0122861, -0.0223307, -0.0105696, -0.0097507, 0.0081925
8: 0.0085235, 0.0182502, 0.0067627, 0.0185066, -0.0071039, 0.0088380
9: 0.9035680, 0.9469696, 0.9020119, 0.9526696, -0.0347585, 0.0296501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180209
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180209
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039026, -0.0008939, -0.0040896, -0.0012338, -0.0026688, 0.0031957
1: 0.0189327, 0.0316482, 0.0172944, 0.0318989, -0.0079593, 0.0094261
2: 0.0211074, 0.0297783, 0.0200466, 0.0300340, -0.0060227, 0.0064511
3: 0.0072798, 0.0170151, 0.0056819, 0.0168617, -0.0071926, 0.0093713
4: -0.0179822, -0.0083385, -0.0176407, -0.0067567, -0.0100454, 0.0076386
5: 0.0140276, 0.0260212, 0.0119901, 0.0257371, -0.0094526, 0.0122742
6: 0.0058922, 0.0149942, 0.0043694, 0.0147205, -0.0072216, 0.0094813
7: -0.0224438, -0.0128171, -0.0222967, -0.0110738, -0.0105598, 0.0081787
8: 0.0088164, 0.0186461, 0.0072598, 0.0185066, -0.0071760, 0.0092121
9: 0.9016901, 0.9445072, 0.9020582, 0.9511690, -0.0378933, 0.0291694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154403, upper bound: 0.0151593
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153655, upper bound: 0.0164390
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039529, -0.0014066, -0.0040723, -0.0012367, -0.0027163, 0.0026658
1: 0.0177399, 0.0320063, 0.0181921, 0.0316567, -0.0085913, 0.0089695
2: 0.0201391, 0.0298727, 0.0206704, 0.0299115, -0.0065640, 0.0059272
3: 0.0059500, 0.0168669, 0.0063097, 0.0165808, -0.0078379, 0.0079390
4: -0.0176267, -0.0068501, -0.0173488, -0.0074310, -0.0081673, 0.0083972
5: 0.0123513, 0.0257143, 0.0127622, 0.0253943, -0.0102394, 0.0104514
6: 0.0045511, 0.0147018, 0.0049685, 0.0144403, -0.0078868, 0.0078423
7: -0.0223201, -0.0113874, -0.0220428, -0.0117536, -0.0090265, 0.0087887
8: 0.0076481, 0.0185007, 0.0077874, 0.0182481, -0.0077421, 0.0081076
9: 0.9019345, 0.9505398, 0.9032502, 0.9480295, -0.0315105, 0.0324034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163167
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163167
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038928, -0.0009043, -0.0040010, -0.0013329, -0.0025599, 0.0030968
1: 0.0182996, 0.0320908, 0.0185964, 0.0316567, -0.0082428, 0.0087271
2: 0.0206883, 0.0300777, 0.0208460, 0.0298004, -0.0060829, 0.0060156
3: 0.0067213, 0.0172944, 0.0066918, 0.0165729, -0.0075224, 0.0084178
4: -0.0182729, -0.0077710, -0.0173186, -0.0077699, -0.0090496, 0.0079079
5: 0.0133699, 0.0263526, 0.0132569, 0.0253681, -0.0096815, 0.0111406
6: 0.0053718, 0.0152634, 0.0053189, 0.0144189, -0.0074767, 0.0085669
7: -0.0227900, -0.0123207, -0.0220109, -0.0121966, -0.0096271, 0.0082535
8: 0.0083166, 0.0188658, 0.0082018, 0.0182481, -0.0074714, 0.0083334
9: 0.9002686, 0.9471854, 0.9032969, 0.9466785, -0.0336027, 0.0310475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155381, upper bound: 0.0134198
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152460, upper bound: 0.0144085
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039529, -0.0014066, -0.0040697, -0.0012598, -0.0026931, 0.0026631
1: 0.0177399, 0.0320063, 0.0180098, 0.0314699, -0.0084363, 0.0091516
2: 0.0201391, 0.0298727, 0.0205092, 0.0297685, -0.0064514, 0.0061333
3: 0.0059500, 0.0168669, 0.0062548, 0.0164433, -0.0077114, 0.0080926
4: -0.0176267, -0.0068501, -0.0172217, -0.0073388, -0.0083437, 0.0082800
5: 0.0123513, 0.0257143, 0.0127029, 0.0252562, -0.0101031, 0.0106044
6: 0.0045511, 0.0147018, 0.0049108, 0.0143212, -0.0077739, 0.0079901
7: -0.0223201, -0.0113874, -0.0219011, -0.0116561, -0.0091794, 0.0086725
8: 0.0076481, 0.0185007, 0.0077700, 0.0181385, -0.0076351, 0.0082092
9: 0.9019345, 0.9505398, 0.9038395, 0.9485210, -0.0323164, 0.0318683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176918, upper bound: 0.0163417
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176918, upper bound: 0.0163417
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038928, -0.0009043, -0.0039961, -0.0013486, -0.0025441, 0.0030918
1: 0.0182996, 0.0320908, 0.0184089, 0.0314699, -0.0080839, 0.0089090
2: 0.0206883, 0.0300777, 0.0206794, 0.0296564, -0.0059653, 0.0062169
3: 0.0067213, 0.0172944, 0.0066253, 0.0164355, -0.0073924, 0.0085696
4: -0.0182729, -0.0077710, -0.0171905, -0.0076699, -0.0092254, 0.0077880
5: 0.0133699, 0.0263526, 0.0131833, 0.0252291, -0.0095411, 0.0112963
6: 0.0053718, 0.0152634, 0.0052547, 0.0142994, -0.0073608, 0.0087094
7: -0.0227900, -0.0123207, -0.0218680, -0.0120920, -0.0097869, 0.0081350
8: 0.0083166, 0.0188658, 0.0081769, 0.0181385, -0.0073612, 0.0084371
9: 0.9002686, 0.9471854, 0.9038866, 0.9472112, -0.0344133, 0.0304992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160462, upper bound: 0.0134234
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158381, upper bound: 0.0144498
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039459, -0.0014172, -0.0040723, -0.0012367, -0.0027093, 0.0026551
1: 0.0175731, 0.0318150, 0.0181921, 0.0316567, -0.0087787, 0.0088334
2: 0.0199905, 0.0297303, 0.0206704, 0.0299115, -0.0067505, 0.0058061
3: 0.0058848, 0.0167340, 0.0063097, 0.0165808, -0.0079695, 0.0078387
4: -0.0175069, -0.0067580, -0.0173488, -0.0074310, -0.0080721, 0.0085462
5: 0.0122828, 0.0255783, 0.0127622, 0.0253943, -0.0103664, 0.0103477
6: 0.0044873, 0.0145929, 0.0049685, 0.0144403, -0.0080070, 0.0077571
7: -0.0221811, -0.0112965, -0.0220428, -0.0117536, -0.0089236, 0.0089285
8: 0.0076235, 0.0183949, 0.0077874, 0.0182481, -0.0078188, 0.0080183
9: 0.9025410, 0.9509988, 0.9032502, 0.9480295, -0.0310519, 0.0331129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166275
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166275
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038862, -0.0009152, -0.0040017, -0.0013322, -0.0025541, 0.0030865
1: 0.0181042, 0.0318965, 0.0185925, 0.0316567, -0.0084734, 0.0086084
2: 0.0205244, 0.0299262, 0.0208443, 0.0298015, -0.0062843, 0.0059165
3: 0.0066312, 0.0171732, 0.0066882, 0.0165730, -0.0076750, 0.0083247
4: -0.0181554, -0.0076655, -0.0173189, -0.0077668, -0.0089444, 0.0080790
5: 0.0132814, 0.0262247, 0.0132523, 0.0253683, -0.0098374, 0.0110343
6: 0.0052929, 0.0151613, 0.0053155, 0.0144192, -0.0076175, 0.0084721
7: -0.0226564, -0.0122147, -0.0220112, -0.0121926, -0.0095176, 0.0084175
8: 0.0082723, 0.0187631, 0.0081979, 0.0182481, -0.0075683, 0.0082641
9: 0.9008329, 0.9477414, 0.9032966, 0.9466919, -0.0331850, 0.0318563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155738, upper bound: 0.0140350
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152788, upper bound: 0.0148069
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039459, -0.0014172, -0.0040697, -0.0012598, -0.0026861, 0.0026525
1: 0.0175731, 0.0318150, 0.0180098, 0.0314699, -0.0085432, 0.0089635
2: 0.0199905, 0.0297303, 0.0205092, 0.0297685, -0.0065520, 0.0059373
3: 0.0058848, 0.0167340, 0.0062548, 0.0164433, -0.0078502, 0.0079978
4: -0.0175069, -0.0067580, -0.0172217, -0.0073388, -0.0082373, 0.0084170
5: 0.0122828, 0.0255783, 0.0127029, 0.0252562, -0.0102205, 0.0104951
6: 0.0044873, 0.0145929, 0.0049108, 0.0143212, -0.0078957, 0.0079062
7: -0.0221811, -0.0112965, -0.0219011, -0.0116561, -0.0090545, 0.0087821
8: 0.0076235, 0.0183949, 0.0077700, 0.0181385, -0.0077293, 0.0081307
9: 0.9025410, 0.9509988, 0.9038395, 0.9485210, -0.0317974, 0.0325002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0178732, upper bound: 0.0170421
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0178732, upper bound: 0.0170421
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038862, -0.0009152, -0.0039956, -0.0013490, -0.0025372, 0.0030804
1: 0.0181042, 0.0318965, 0.0184116, 0.0314699, -0.0082105, 0.0087182
2: 0.0205244, 0.0299262, 0.0206806, 0.0296557, -0.0060711, 0.0060334
3: 0.0066312, 0.0171732, 0.0066278, 0.0164354, -0.0075490, 0.0084789
4: -0.0181554, -0.0076655, -0.0171902, -0.0076720, -0.0091077, 0.0079426
5: 0.0132814, 0.0262247, 0.0131865, 0.0252289, -0.0096816, 0.0111779
6: 0.0052929, 0.0151613, 0.0052570, 0.0142992, -0.0074998, 0.0086126
7: -0.0226564, -0.0122147, -0.0218678, -0.0120948, -0.0096504, 0.0082656
8: 0.0082723, 0.0187631, 0.0081797, 0.0181385, -0.0074735, 0.0083781
9: 0.9008329, 0.9477414, 0.9038868, 0.9472022, -0.0339124, 0.0312102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0163799, upper bound: 0.0145016
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0162859, upper bound: 0.0155564
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039529, -0.0014066, -0.0041075, -0.0011493, -0.0028036, 0.0027009
1: 0.0177399, 0.0320063, 0.0180860, 0.0317525, -0.0088874, 0.0092145
2: 0.0201391, 0.0298727, 0.0206564, 0.0300202, -0.0068008, 0.0059670
3: 0.0059500, 0.0168669, 0.0062101, 0.0167013, -0.0079940, 0.0080856
4: -0.0176267, -0.0068501, -0.0174401, -0.0073896, -0.0082433, 0.0084830
5: 0.0123513, 0.0257143, 0.0126355, 0.0255008, -0.0103640, 0.0106336
6: 0.0045511, 0.0147018, 0.0048972, 0.0145426, -0.0079841, 0.0079514
7: -0.0223201, -0.0113874, -0.0221280, -0.0116488, -0.0091801, 0.0088602
8: 0.0076481, 0.0185007, 0.0076520, 0.0183602, -0.0079414, 0.0083011
9: 0.9019345, 0.9505398, 0.9028240, 0.9482442, -0.0319470, 0.0330629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038928, -0.0009043, -0.0040635, -0.0012311, -0.0026617, 0.0031593
1: 0.0182996, 0.0320908, 0.0183501, 0.0317525, -0.0084199, 0.0090804
2: 0.0206883, 0.0300777, 0.0207668, 0.0299526, -0.0063275, 0.0060734
3: 0.0067213, 0.0172944, 0.0064632, 0.0166968, -0.0075583, 0.0086357
4: -0.0182729, -0.0077710, -0.0174217, -0.0076048, -0.0091775, 0.0079221
5: 0.0133699, 0.0263526, 0.0129540, 0.0254851, -0.0097102, 0.0114338
6: 0.0053718, 0.0152634, 0.0051232, 0.0145300, -0.0074930, 0.0087385
7: -0.0227900, -0.0123207, -0.0221085, -0.0119313, -0.0098842, 0.0082679
8: 0.0083166, 0.0188658, 0.0079262, 0.0183602, -0.0075205, 0.0086185
9: 0.9002686, 0.9471854, 0.9028515, 0.9473714, -0.0342179, 0.0312143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039529, -0.0014066, -0.0041035, -0.0011758, -0.0027771, 0.0026969
1: 0.0177399, 0.0320063, 0.0179049, 0.0315551, -0.0087499, 0.0093971
2: 0.0201391, 0.0298727, 0.0204964, 0.0298754, -0.0066810, 0.0061662
3: 0.0059500, 0.0168669, 0.0061511, 0.0165660, -0.0078779, 0.0082362
4: -0.0176267, -0.0068501, -0.0173159, -0.0072973, -0.0084187, 0.0083614
5: 0.0123513, 0.0257143, 0.0125759, 0.0253676, -0.0102324, 0.0107902
6: 0.0045511, 0.0147018, 0.0048413, 0.0144279, -0.0078735, 0.0080970
7: -0.0223201, -0.0113874, -0.0219870, -0.0115511, -0.0093387, 0.0087352
8: 0.0076481, 0.0185007, 0.0076329, 0.0182502, -0.0078346, 0.0084069
9: 0.9019345, 0.9505398, 0.9034380, 0.9487324, -0.0327361, 0.0325549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162524
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162524
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038928, -0.0009043, -0.0040581, -0.0012557, -0.0026371, 0.0031538
1: 0.0182996, 0.0320908, 0.0181719, 0.0315551, -0.0082686, 0.0092569
2: 0.0206883, 0.0300777, 0.0206060, 0.0298031, -0.0062007, 0.0062721
3: 0.0067213, 0.0172944, 0.0064044, 0.0165614, -0.0074312, 0.0087836
4: -0.0182729, -0.0077710, -0.0172972, -0.0075114, -0.0093530, 0.0078022
5: 0.0133699, 0.0263526, 0.0128925, 0.0253515, -0.0095635, 0.0115844
6: 0.0053718, 0.0152634, 0.0050655, 0.0144148, -0.0073765, 0.0088778
7: -0.0227900, -0.0123207, -0.0219673, -0.0118328, -0.0100408, 0.0081463
8: 0.0083166, 0.0188658, 0.0079073, 0.0182502, -0.0074102, 0.0087186
9: 0.9002686, 0.9471854, 0.9034659, 0.9478643, -0.0349993, 0.0306629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162525
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162525
time: 0.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.08 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163172
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163172, upper bound: 0.0163172
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0145934, upper bound: 0.0134198
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144107, upper bound: 0.0144107
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166539, upper bound: 0.0163417
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166539, upper bound: 0.0163417
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0149588, upper bound: 0.0134234
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148388, upper bound: 0.0144498
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0166539
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0166539
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0146453, upper bound: 0.0140472
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144498, upper bound: 0.0148388
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169951, upper bound: 0.0170749
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169951, upper bound: 0.0170749
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0155492, upper bound: 0.0145125
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154794, upper bound: 0.0155896
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0162321
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166234, upper bound: 0.0162529
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166234, upper bound: 0.0162529
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0149474, upper bound: 0.0133785
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148009, upper bound: 0.0143674
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165805
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165805
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165952
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0165952
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169550, upper bound: 0.0169893
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169550, upper bound: 0.0169893
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0155436, upper bound: 0.0144785
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154468, upper bound: 0.0154999
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162937
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162937
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0133529
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143201, upper bound: 0.0143757
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0163131
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0163131
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148925, upper bound: 0.0133622
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0147560, upper bound: 0.0144212
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166234
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166234
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0140092
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0148009
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154559, upper bound: 0.0144811
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0153784, upper bound: 0.0155518
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162411
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162321, upper bound: 0.0162411
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0133529
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143201, upper bound: 0.0143237
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0162741
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165805, upper bound: 0.0162741
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148925, upper bound: 0.0133622
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0147560, upper bound: 0.0143735
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166093
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162529, upper bound: 0.0166093
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0140040
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0147655
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168940, upper bound: 0.0170360
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154559, upper bound: 0.0144782
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0153784, upper bound: 0.0155196
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172214
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163167, upper bound: 0.0172214
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0145934, upper bound: 0.0141791
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144085, upper bound: 0.0152460
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166275, upper bound: 0.0172366
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166275, upper bound: 0.0172366
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0149556, upper bound: 0.0141791
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148069, upper bound: 0.0152788
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0176918
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163417, upper bound: 0.0176918
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0146453, upper bound: 0.0149019
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144498, upper bound: 0.0158381
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169851, upper bound: 0.0180658
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169851, upper bound: 0.0180658
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0155451, upper bound: 0.0151838
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154750, upper bound: 0.0164999
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162937, upper bound: 0.0171555
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166058, upper bound: 0.0171677
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0166058, upper bound: 0.0171677
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0149444, upper bound: 0.0141426
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0147798, upper bound: 0.0151997
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163131, upper bound: 0.0176532
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169505, upper bound: 0.0180077
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0169505, upper bound: 0.0180077
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0155395, upper bound: 0.0151612
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154391, upper bound: 0.0164340
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171897
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171897
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0141152
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0152163
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0172056
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0172056
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148851, upper bound: 0.0141152
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0147242, upper bound: 0.0152440
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176661
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176661
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0148736
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0158139
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180351
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180351
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154403, upper bound: 0.0151597
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0153655, upper bound: 0.0164598
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171555
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162296, upper bound: 0.0171555
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0144817, upper bound: 0.0141152
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0151795
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0171680
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0165508, upper bound: 0.0171680
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0148851, upper bound: 0.0141152
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0147242, upper bound: 0.0151997
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176534
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162524, upper bound: 0.0176534
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0145558, upper bound: 0.0148721
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0143674, upper bound: 0.0157998
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180209
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0168776, upper bound: 0.0180209
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0154403, upper bound: 0.0151593
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0153655, upper bound: 0.0164390
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163167
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163167
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0155381, upper bound: 0.0134198
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0152460, upper bound: 0.0144085
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0176918, upper bound: 0.0163417
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0176918, upper bound: 0.0163417
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0160462, upper bound: 0.0134234
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0158381, upper bound: 0.0144498
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166275
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166275
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0155738, upper bound: 0.0140350
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0152788, upper bound: 0.0148069
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0178732, upper bound: 0.0170421
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0178732, upper bound: 0.0170421
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0163799, upper bound: 0.0145016
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0162859, upper bound: 0.0155564
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162296
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162524
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162524
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162525
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 9, lower bound: -0.0176661, upper bound: 0.0162525
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0168226
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0165663
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0171305
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0170168
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166788
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162937
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166917
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163132
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168869
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166241
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170855
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166788
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162411
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166917
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162741
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168869
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0165961
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170607
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166862
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163500
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0166898
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172214, upper bound: 0.0163654
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0168559
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0166703
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0172089
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172366, upper bound: 0.0171241
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166238
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162738
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171920, upper bound: 0.0166279
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171897, upper bound: 0.0162887
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0168227
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0165824
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0171305
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0172056, upper bound: 0.0170168
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166984
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163312
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0167077
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163454
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168896
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166456
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170855
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0166984
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0162955
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0167077
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171555, upper bound: 0.0163145
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0168896
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0166129
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0172461
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 9, lower bound: -0.0171677, upper bound: 0.0170607

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.84 + 597.25 = 600.09 seconds
