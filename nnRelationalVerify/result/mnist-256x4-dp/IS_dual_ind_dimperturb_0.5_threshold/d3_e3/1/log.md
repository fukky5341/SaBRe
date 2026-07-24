## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00504846


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0034702, 0.0101244, 0.0034702, 0.0101244, -0.0065322, 0.0065322)
1: (0.0018236, 0.0027850, 0.0018236, 0.0027850, -0.0009437, 0.0009437)
2: (0.0087623, 0.0124413, 0.0087623, 0.0124413, -0.0036115, 0.0036115)
3: (-0.0056180, -0.0018131, -0.0056180, -0.0018131, -0.0037351, 0.0037351)
4: (-0.0020742, 0.0020449, -0.0020742, 0.0020449, -0.0040435, 0.0040435)
5: (0.0021782, 0.0060762, 0.0021782, 0.0060762, -0.0038265, 0.0038265)
6: (-0.0136579, 0.0018083, -0.0136579, 0.0018083, -0.0151825, 0.0151825)
7: (-0.0050194, 0.0160442, -0.0050194, 0.0160442, -0.0206773, 0.0206773)
8: (0.9856781, 1.0005157, 0.9856781, 1.0005157, -0.0145655, 0.0145655)
9: (-0.0163554, -0.0028868, -0.0163554, -0.0028868, -0.0132216, 0.0132216)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 2.32 = 3.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0084066, upper bound: 0.0084066

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078780, upper bound: 0.0080571
time: 0.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080571, upper bound: 0.0080571
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 8, lower bound: -0.0078780, upper bound: 0.0080571
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 8, lower bound: -0.0080571, upper bound: 0.0080571

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0041348, 0.0101222, 0.0037011, 0.0101160, -0.0057008, 0.0060133
1: 0.0019197, 0.0027847, 0.0018570, 0.0027838, -0.0008236, 0.0008688
2: 0.0087636, 0.0120738, 0.0087670, 0.0123136, -0.0033246, 0.0031518
3: -0.0056168, -0.0021931, -0.0056132, -0.0019451, -0.0034385, 0.0032598
4: -0.0016628, 0.0020435, -0.0019312, 0.0020397, -0.0035289, 0.0037224
5: 0.0021795, 0.0056869, 0.0021831, 0.0059410, -0.0035226, 0.0033395
6: -0.0136527, 0.0002636, -0.0136384, 0.0012716, -0.0139767, 0.0132502
7: -0.0029156, 0.0160371, -0.0042886, 0.0160177, -0.0180456, 0.0190350
8: 0.9871600, 1.0005108, 0.9861929, 1.0004970, -0.0127117, 0.0134087
9: -0.0163509, -0.0042320, -0.0163385, -0.0033541, -0.0121715, 0.0115388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078602, upper bound: 0.0078603
time: 0.92 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078602, upper bound: 0.0080571
time: 0.91 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0037404, 0.0101111, 0.0035281, 0.0101217, -0.0056562, 0.0064792
1: 0.0018627, 0.0027831, 0.0018320, 0.0027846, -0.0008172, 0.0009361
2: 0.0087697, 0.0122919, 0.0087639, 0.0124092, -0.0035822, 0.0031272
3: -0.0056104, -0.0019676, -0.0056165, -0.0018462, -0.0037049, 0.0032343
4: -0.0019069, 0.0020367, -0.0020383, 0.0020432, -0.0035013, 0.0040107
5: 0.0021860, 0.0059180, 0.0021798, 0.0060423, -0.0037955, 0.0033134
6: -0.0136270, 0.0011804, -0.0136515, 0.0016736, -0.0150594, 0.0131465
7: -0.0041643, 0.0160021, -0.0048360, 0.0160355, -0.0179044, 0.0205096
8: 0.9862804, 1.0004861, 0.9858073, 1.0005096, -0.0126123, 0.0144474
9: -0.0163285, -0.0034336, -0.0163499, -0.0030041, -0.0131144, 0.0114486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080571, upper bound: 0.0078602
time: 0.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0080571, upper bound: 0.0080571
time: 0.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 8, lower bound: -0.0078602, upper bound: 0.0078603
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 8, lower bound: -0.0078602, upper bound: 0.0080571
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 8, lower bound: -0.0080571, upper bound: 0.0078602
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 8, lower bound: -0.0080571, upper bound: 0.0080571

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041348, 0.0101222, 0.0041348, 0.0101222, -0.0054792, 0.0054792
1: 0.0019197, 0.0027847, 0.0019197, 0.0027847, -0.0007916, 0.0007916
2: 0.0087636, 0.0120738, 0.0087636, 0.0120738, -0.0030293, 0.0030293
3: -0.0056168, -0.0021931, -0.0056168, -0.0021931, -0.0031330, 0.0031330
4: -0.0016628, 0.0020435, -0.0016628, 0.0020435, -0.0033917, 0.0033917
5: 0.0021795, 0.0056869, 0.0021795, 0.0056869, -0.0032097, 0.0032097
6: -0.0136527, 0.0002636, -0.0136527, 0.0002636, -0.0127351, 0.0127351
7: -0.0029156, 0.0160371, -0.0029156, 0.0160371, -0.0173441, 0.0173441
8: 0.9871600, 1.0005108, 0.9871600, 1.0005108, -0.0122175, 0.0122175
9: -0.0163509, -0.0042320, -0.0163509, -0.0042320, -0.0110903, 0.0110903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0074838
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0076091
time: 1.13 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041348, 0.0101222, 0.0037404, 0.0101111, -0.0056961, 0.0060898
1: 0.0019197, 0.0027847, 0.0018627, 0.0027831, -0.0008229, 0.0008798
2: 0.0087636, 0.0120738, 0.0087697, 0.0122919, -0.0033669, 0.0031492
3: -0.0056168, -0.0021931, -0.0056104, -0.0019676, -0.0034822, 0.0032571
4: -0.0016628, 0.0020435, -0.0019069, 0.0020367, -0.0035260, 0.0037697
5: 0.0021795, 0.0056869, 0.0021860, 0.0059180, -0.0035674, 0.0033367
6: -0.0136527, 0.0002636, -0.0136270, 0.0011804, -0.0141543, 0.0132392
7: -0.0029156, 0.0160371, -0.0041643, 0.0160021, -0.0180307, 0.0192770
8: 0.9871600, 1.0005108, 0.9862804, 1.0004861, -0.0127012, 0.0135791
9: -0.0163509, -0.0042320, -0.0163285, -0.0034336, -0.0123262, 0.0115293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0077094
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0078010
time: 1.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037404, 0.0101111, 0.0041348, 0.0101222, -0.0060898, 0.0056961
1: 0.0018627, 0.0027831, 0.0019197, 0.0027847, -0.0008798, 0.0008229
2: 0.0087697, 0.0122919, 0.0087636, 0.0120738, -0.0031492, 0.0033669
3: -0.0056104, -0.0019676, -0.0056168, -0.0021931, -0.0032571, 0.0034822
4: -0.0019069, 0.0020367, -0.0016628, 0.0020435, -0.0037697, 0.0035260
5: 0.0021860, 0.0059180, 0.0021795, 0.0056869, -0.0033367, 0.0035674
6: -0.0136270, 0.0011804, -0.0136527, 0.0002636, -0.0132392, 0.0141543
7: -0.0041643, 0.0160021, -0.0029156, 0.0160371, -0.0192770, 0.0180307
8: 0.9862804, 1.0004861, 0.9871600, 1.0005108, -0.0135791, 0.0127012
9: -0.0163285, -0.0034336, -0.0163509, -0.0042320, -0.0115293, 0.0123262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0074838
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0076091
time: 1.05 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037404, 0.0101111, 0.0037404, 0.0101111, -0.0056461, 0.0056461
1: 0.0018627, 0.0027831, 0.0018627, 0.0027831, -0.0008157, 0.0008157
2: 0.0087697, 0.0122919, 0.0087697, 0.0122919, -0.0031216, 0.0031216
3: -0.0056104, -0.0019676, -0.0056104, -0.0019676, -0.0032285, 0.0032285
4: -0.0019069, 0.0020367, -0.0019069, 0.0020367, -0.0034950, 0.0034950
5: 0.0021860, 0.0059180, 0.0021860, 0.0059180, -0.0033075, 0.0033075
6: -0.0136270, 0.0011804, -0.0136270, 0.0011804, -0.0131230, 0.0131230
7: -0.0041643, 0.0160021, -0.0041643, 0.0160021, -0.0178724, 0.0178724
8: 0.9862804, 1.0004861, 0.9862804, 1.0004861, -0.0125897, 0.0125897
9: -0.0163285, -0.0034336, -0.0163285, -0.0034336, -0.0114281, 0.0114281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0074868
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0076091
time: 1.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0074838
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0076091
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0077094
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0076203, upper bound: 0.0078010
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0074838
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0076091
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0074868
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 8, lower bound: -0.0078009, upper bound: 0.0076091

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0041789, 0.0099141, 0.0041420, 0.0100873, -0.0054049, 0.0052485
1: 0.0019260, 0.0027546, 0.0019207, 0.0027796, -0.0007808, 0.0007582
2: 0.0088786, 0.0120494, 0.0087829, 0.0120699, -0.0029017, 0.0029882
3: -0.0054978, -0.0022184, -0.0055968, -0.0021972, -0.0030011, 0.0030906
4: -0.0016355, 0.0019147, -0.0016583, 0.0020219, -0.0033457, 0.0032489
5: 0.0023014, 0.0056610, 0.0021999, 0.0056827, -0.0030745, 0.0031662
6: -0.0131692, 0.0001610, -0.0135716, 0.0002468, -0.0121989, 0.0125624
7: -0.0027760, 0.0153786, -0.0028929, 0.0159266, -0.0171089, 0.0166138
8: 0.9872583, 1.0000468, 0.9871761, 1.0004330, -0.0120519, 0.0117031
9: -0.0159298, -0.0043213, -0.0162803, -0.0042466, -0.0106233, 0.0109399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0071581
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0072588
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0039328, 0.0099062, 0.0041557, 0.0100368, -0.0057008, 0.0052678
1: 0.0018905, 0.0027535, 0.0019227, 0.0027723, -0.0008236, 0.0007611
2: 0.0088830, 0.0121855, 0.0088108, 0.0120623, -0.0029125, 0.0031518
3: -0.0054932, -0.0020777, -0.0055679, -0.0022051, -0.0030122, 0.0032598
4: -0.0017878, 0.0019098, -0.0016498, 0.0019906, -0.0035289, 0.0032609
5: 0.0023060, 0.0058052, 0.0022295, 0.0056746, -0.0030859, 0.0033395
6: -0.0131506, 0.0007330, -0.0134542, 0.0002150, -0.0122439, 0.0132503
7: -0.0035549, 0.0153533, -0.0028495, 0.0157667, -0.0180458, 0.0166751
8: 0.9867097, 1.0000290, 0.9872066, 1.0003203, -0.0127118, 0.0117463
9: -0.0159137, -0.0038232, -0.0161780, -0.0042743, -0.0106625, 0.0115390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0072746
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0073737
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0041789, 0.0099141, 0.0037474, 0.0100771, -0.0056240, 0.0058587
1: 0.0019260, 0.0027546, 0.0018637, 0.0027782, -0.0008125, 0.0008464
2: 0.0088786, 0.0120494, 0.0087885, 0.0122880, -0.0032391, 0.0031094
3: -0.0054978, -0.0022184, -0.0055910, -0.0019716, -0.0033501, 0.0032159
4: -0.0016355, 0.0019147, -0.0019026, 0.0020156, -0.0034814, 0.0036266
5: 0.0023014, 0.0056610, 0.0022059, 0.0059138, -0.0034320, 0.0032945
6: -0.0131692, 0.0001610, -0.0135480, 0.0011640, -0.0136173, 0.0130717
7: -0.0027760, 0.0153786, -0.0041420, 0.0158945, -0.0178026, 0.0185455
8: 0.9872583, 1.0000468, 0.9862962, 1.0004102, -0.0125405, 0.0130639
9: -0.0159298, -0.0043213, -0.0162597, -0.0034479, -0.0118585, 0.0113835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0073767
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0074215
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039328, 0.0099062, 0.0037604, 0.0100244, -0.0059069, 0.0058778
1: 0.0018905, 0.0027535, 0.0018656, 0.0027705, -0.0008534, 0.0008492
2: 0.0088830, 0.0121855, 0.0088176, 0.0122809, -0.0032497, 0.0032657
3: -0.0054932, -0.0020777, -0.0055609, -0.0019790, -0.0033610, 0.0033776
4: -0.0017878, 0.0019098, -0.0018946, 0.0019830, -0.0036564, 0.0036385
5: 0.0023060, 0.0058052, 0.0022368, 0.0059062, -0.0034432, 0.0034602
6: -0.0131506, 0.0007330, -0.0134255, 0.0011339, -0.0136616, 0.0137291
7: -0.0035549, 0.0153533, -0.0041010, 0.0157276, -0.0186979, 0.0186059
8: 0.9867097, 1.0000290, 0.9863250, 1.0002928, -0.0131712, 0.0131064
9: -0.0159137, -0.0038232, -0.0161530, -0.0034741, -0.0118971, 0.0119559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0074694
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0075202
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037838, 0.0099066, 0.0041420, 0.0100873, -0.0060130, 0.0054727
1: 0.0018689, 0.0027535, 0.0019207, 0.0027796, -0.0008687, 0.0007907
2: 0.0088828, 0.0122679, 0.0087829, 0.0120699, -0.0030257, 0.0033244
3: -0.0054935, -0.0019924, -0.0055968, -0.0021972, -0.0031294, 0.0034383
4: -0.0018801, 0.0019101, -0.0016583, 0.0020219, -0.0037222, 0.0033877
5: 0.0023058, 0.0058925, 0.0021999, 0.0056827, -0.0032059, 0.0035224
6: -0.0131516, 0.0010795, -0.0135716, 0.0002468, -0.0127202, 0.0139759
7: -0.0040269, 0.0153547, -0.0028929, 0.0159266, -0.0190339, 0.0173238
8: 0.9863772, 1.0000300, 0.9871761, 1.0004330, -0.0134079, 0.0122032
9: -0.0159145, -0.0035214, -0.0162803, -0.0042466, -0.0110773, 0.0121708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075123, upper bound: 0.0071581
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0072588
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0035687, 0.0098847, 0.0041557, 0.0100368, -0.0062621, 0.0054819
1: 0.0018379, 0.0027504, 0.0019227, 0.0027723, -0.0009047, 0.0007920
2: 0.0088948, 0.0123868, 0.0088108, 0.0120623, -0.0030308, 0.0034622
3: -0.0054810, -0.0018694, -0.0055679, -0.0022051, -0.0031346, 0.0035808
4: -0.0020132, 0.0018965, -0.0016498, 0.0019906, -0.0038764, 0.0033934
5: 0.0023186, 0.0060185, 0.0022295, 0.0056746, -0.0032113, 0.0036684
6: -0.0131009, 0.0015795, -0.0134542, 0.0002150, -0.0127414, 0.0145550
7: -0.0047078, 0.0152855, -0.0028495, 0.0157667, -0.0198226, 0.0173527
8: 0.9858975, 0.9999813, 0.9872066, 1.0003203, -0.0139634, 0.0122236
9: -0.0158703, -0.0030860, -0.0161780, -0.0042743, -0.0110958, 0.0126751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075123, upper bound: 0.0072746
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0073737
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037838, 0.0099066, 0.0037474, 0.0100771, -0.0055723, 0.0054222
1: 0.0018689, 0.0027535, 0.0018637, 0.0027782, -0.0008050, 0.0007834
2: 0.0088828, 0.0122679, 0.0087885, 0.0122880, -0.0029978, 0.0030808
3: -0.0054935, -0.0019924, -0.0055910, -0.0019716, -0.0031005, 0.0031863
4: -0.0018801, 0.0019101, -0.0019026, 0.0020156, -0.0034493, 0.0033564
5: 0.0023058, 0.0058925, 0.0022059, 0.0059138, -0.0031763, 0.0032642
6: -0.0131516, 0.0010795, -0.0135480, 0.0011640, -0.0126027, 0.0129516
7: -0.0040269, 0.0153547, -0.0041420, 0.0158945, -0.0176389, 0.0171638
8: 0.9863772, 1.0000300, 0.9862962, 1.0004102, -0.0124252, 0.0120905
9: -0.0159145, -0.0035214, -0.0162597, -0.0034479, -0.0109750, 0.0112788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075124, upper bound: 0.0071547
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0072285
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0035687, 0.0098847, 0.0037604, 0.0100244, -0.0058536, 0.0054321
1: 0.0018379, 0.0027504, 0.0018656, 0.0027705, -0.0008457, 0.0007848
2: 0.0088948, 0.0123868, 0.0088176, 0.0122809, -0.0030033, 0.0032363
3: -0.0054810, -0.0018694, -0.0055609, -0.0019790, -0.0031061, 0.0033471
4: -0.0020132, 0.0018965, -0.0018946, 0.0019830, -0.0036235, 0.0033626
5: 0.0023186, 0.0060185, 0.0022368, 0.0059062, -0.0031821, 0.0034290
6: -0.0131009, 0.0015795, -0.0134255, 0.0011339, -0.0126257, 0.0136053
7: -0.0047078, 0.0152855, -0.0041010, 0.0157276, -0.0185293, 0.0171951
8: 0.9858975, 0.9999813, 0.9863250, 1.0002928, -0.0130524, 0.0121126
9: -0.0158703, -0.0030860, -0.0161530, -0.0034741, -0.0109950, 0.0118481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075124, upper bound: 0.0072705
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0073488
time: 1.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.22 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0071581
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0072588
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0072746
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0073737
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0073767
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0074215
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073700, upper bound: 0.0074694
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0073737, upper bound: 0.0075202
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075123, upper bound: 0.0071581
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0072588
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075123, upper bound: 0.0072746
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0073737
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075124, upper bound: 0.0071547
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0072285
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075124, upper bound: 0.0072705
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 8, lower bound: -0.0075202, upper bound: 0.0073488

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0043379, 0.0099029, 0.0046159, 0.0102117, -0.0052145, 0.0046947
1: 0.0019490, 0.0027530, 0.0019892, 0.0027976, -0.0007533, 0.0006783
2: 0.0088848, 0.0119615, 0.0087141, 0.0118078, -0.0025956, 0.0028829
3: -0.0054914, -0.0023093, -0.0056679, -0.0024682, -0.0026845, 0.0029817
4: -0.0015370, 0.0019078, -0.0013649, 0.0020989, -0.0032278, 0.0029061
5: 0.0023080, 0.0055679, 0.0021271, 0.0054050, -0.0027502, 0.0030546
6: -0.0131430, -0.0002086, -0.0138607, -0.0008547, -0.0109118, 0.0121199
7: -0.0022727, 0.0153430, -0.0013927, 0.0163204, -0.0165062, 0.0148610
8: 0.9876130, 1.0000218, 0.9882329, 1.0007104, -0.0116273, 0.0104684
9: -0.0159071, -0.0046431, -0.0165321, -0.0052058, -0.0095025, 0.0105545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070717, upper bound: 0.0068822
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0068901
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041789, 0.0099141, 0.0042603, 0.0100788, -0.0053972, 0.0047949
1: 0.0019260, 0.0027546, 0.0019378, 0.0027784, -0.0007797, 0.0006927
2: 0.0088786, 0.0120494, 0.0087876, 0.0120045, -0.0026510, 0.0029840
3: -0.0054978, -0.0022184, -0.0055919, -0.0022649, -0.0027418, 0.0030862
4: -0.0016355, 0.0019147, -0.0015851, 0.0020167, -0.0033409, 0.0029681
5: 0.0023014, 0.0056610, 0.0022049, 0.0056134, -0.0028088, 0.0031617
6: -0.0131692, 0.0001610, -0.0135519, -0.0000281, -0.0111446, 0.0125445
7: -0.0027760, 0.0153786, -0.0025184, 0.0158998, -0.0170846, 0.0151780
8: 0.9872583, 1.0000468, 0.9874398, 1.0004140, -0.0120347, 0.0106917
9: -0.0159298, -0.0043213, -0.0162631, -0.0044860, -0.0097052, 0.0109243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0072544
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0072588
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040874, 0.0098953, 0.0046300, 0.0101610, -0.0055059, 0.0047147
1: 0.0019128, 0.0027519, 0.0019912, 0.0027903, -0.0007954, 0.0006811
2: 0.0088890, 0.0121000, 0.0087421, 0.0118001, -0.0026066, 0.0030441
3: -0.0054870, -0.0021660, -0.0056390, -0.0024763, -0.0026959, 0.0031483
4: -0.0016921, 0.0019031, -0.0013563, 0.0020676, -0.0034082, 0.0029185
5: 0.0023124, 0.0057146, 0.0021567, 0.0053968, -0.0027618, 0.0032254
6: -0.0131254, 0.0003737, -0.0137430, -0.0008873, -0.0109582, 0.0127972
7: -0.0030657, 0.0153189, -0.0013483, 0.0161601, -0.0174287, 0.0149241
8: 0.9870543, 1.0000049, 0.9882641, 1.0005974, -0.0122772, 0.0105129
9: -0.0158917, -0.0041361, -0.0164296, -0.0052342, -0.0095429, 0.0111444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070718, upper bound: 0.0070059
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0070134
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0039328, 0.0099062, 0.0042731, 0.0100283, -0.0056933, 0.0048049
1: 0.0018905, 0.0027535, 0.0019396, 0.0027711, -0.0008225, 0.0006942
2: 0.0088830, 0.0121855, 0.0088155, 0.0119974, -0.0026565, 0.0031477
3: -0.0054932, -0.0020777, -0.0055631, -0.0022722, -0.0027475, 0.0032555
4: -0.0017878, 0.0019098, -0.0015772, 0.0019854, -0.0035242, 0.0029743
5: 0.0023060, 0.0058052, 0.0022345, 0.0056059, -0.0028147, 0.0033351
6: -0.0131506, 0.0007330, -0.0134346, -0.0000579, -0.0111680, 0.0132328
7: -0.0035549, 0.0153533, -0.0024779, 0.0157400, -0.0180219, 0.0152098
8: 0.9867097, 1.0000290, 0.9874684, 1.0003015, -0.0126950, 0.0107141
9: -0.0159137, -0.0038232, -0.0161609, -0.0045119, -0.0097256, 0.0115237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0073700
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0073737
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0043379, 0.0099029, 0.0042106, 0.0102272, -0.0054915, 0.0053310
1: 0.0019490, 0.0027530, 0.0019306, 0.0027998, -0.0007934, 0.0007702
2: 0.0088848, 0.0119615, 0.0087055, 0.0120319, -0.0029474, 0.0030361
3: -0.0054914, -0.0023093, -0.0056768, -0.0022365, -0.0030483, 0.0031401
4: -0.0015370, 0.0019078, -0.0016158, 0.0021085, -0.0033993, 0.0033000
5: 0.0023080, 0.0055679, 0.0021180, 0.0056425, -0.0031229, 0.0032169
6: -0.0131430, -0.0002086, -0.0138968, 0.0000873, -0.0123907, 0.0127638
7: -0.0022727, 0.0153430, -0.0026756, 0.0163695, -0.0173831, 0.0168751
8: 0.9876130, 1.0000218, 0.9873291, 1.0007448, -0.0122450, 0.0118872
9: -0.0159071, -0.0046431, -0.0165634, -0.0043855, -0.0107904, 0.0111152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070718, upper bound: 0.0071192
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0071246
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041789, 0.0099141, 0.0038844, 0.0100688, -0.0056163, 0.0054398
1: 0.0019260, 0.0027546, 0.0018835, 0.0027769, -0.0008114, 0.0007859
2: 0.0088786, 0.0120494, 0.0087931, 0.0122123, -0.0030075, 0.0031051
3: -0.0054978, -0.0022184, -0.0055862, -0.0020500, -0.0031105, 0.0032114
4: -0.0016355, 0.0019147, -0.0018177, 0.0020105, -0.0034766, 0.0033673
5: 0.0023014, 0.0056610, 0.0022108, 0.0058336, -0.0031866, 0.0032900
6: -0.0131692, 0.0001610, -0.0135286, 0.0008455, -0.0126437, 0.0130538
7: -0.0027760, 0.0153786, -0.0037082, 0.0158680, -0.0177782, 0.0172196
8: 0.9872583, 1.0000468, 0.9866017, 1.0003917, -0.0125233, 0.0121298
9: -0.0159298, -0.0043213, -0.0162428, -0.0037252, -0.0110107, 0.0113678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0074127
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0074215
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040874, 0.0098953, 0.0042243, 0.0101699, -0.0057658, 0.0053496
1: 0.0019128, 0.0027519, 0.0019326, 0.0027915, -0.0008330, 0.0007729
2: 0.0088890, 0.0121000, 0.0087372, 0.0120244, -0.0029577, 0.0031878
3: -0.0054870, -0.0021660, -0.0056440, -0.0022443, -0.0030590, 0.0032969
4: -0.0016921, 0.0019031, -0.0016074, 0.0020730, -0.0035691, 0.0033115
5: 0.0023124, 0.0057146, 0.0021516, 0.0056345, -0.0031338, 0.0033776
6: -0.0131254, 0.0003737, -0.0137635, 0.0000556, -0.0124341, 0.0134013
7: -0.0030657, 0.0153189, -0.0026325, 0.0161880, -0.0182514, 0.0169341
8: 0.9870543, 1.0000049, 0.9873595, 1.0006171, -0.0128567, 0.0119287
9: -0.0158917, -0.0041361, -0.0164474, -0.0044131, -0.0108281, 0.0116704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070718, upper bound: 0.0072101
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0072154
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0039328, 0.0099062, 0.0038975, 0.0100161, -0.0058991, 0.0054482
1: 0.0018905, 0.0027535, 0.0018854, 0.0027693, -0.0008523, 0.0007871
2: 0.0088830, 0.0121855, 0.0088222, 0.0122050, -0.0030122, 0.0032615
3: -0.0054932, -0.0020777, -0.0055561, -0.0020574, -0.0031153, 0.0033732
4: -0.0017878, 0.0019098, -0.0018097, 0.0019778, -0.0036517, 0.0033725
5: 0.0023060, 0.0058052, 0.0022416, 0.0058259, -0.0031915, 0.0034557
6: -0.0131506, 0.0007330, -0.0134062, 0.0008151, -0.0126631, 0.0137112
7: -0.0035549, 0.0153533, -0.0036668, 0.0157013, -0.0186735, 0.0172460
8: 0.9867097, 1.0000290, 0.9866309, 1.0002742, -0.0131540, 0.0121484
9: -0.0159137, -0.0038232, -0.0161362, -0.0037517, -0.0110276, 0.0119404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0075123
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0075202
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0039347, 0.0098961, 0.0046159, 0.0102117, -0.0058333, 0.0049192
1: 0.0018908, 0.0027520, 0.0019892, 0.0027976, -0.0008427, 0.0007107
2: 0.0088886, 0.0121845, 0.0087141, 0.0118078, -0.0027197, 0.0032251
3: -0.0054875, -0.0020787, -0.0056679, -0.0024682, -0.0028128, 0.0033355
4: -0.0017866, 0.0019036, -0.0013649, 0.0020989, -0.0036109, 0.0030451
5: 0.0023119, 0.0058041, 0.0021271, 0.0054050, -0.0028816, 0.0034171
6: -0.0131272, 0.0007286, -0.0138607, -0.0008547, -0.0114335, 0.0135582
7: -0.0035490, 0.0153215, -0.0013927, 0.0163204, -0.0184650, 0.0155715
8: 0.9867139, 1.0000066, 0.9882329, 1.0007104, -0.0130072, 0.0109689
9: -0.0158933, -0.0038270, -0.0165321, -0.0052058, -0.0099568, 0.0118070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072133, upper bound: 0.0068822
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072968, upper bound: 0.0068901
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037838, 0.0099066, 0.0042603, 0.0100788, -0.0060053, 0.0050625
1: 0.0018689, 0.0027535, 0.0019378, 0.0027784, -0.0008676, 0.0007314
2: 0.0088828, 0.0122679, 0.0087876, 0.0120045, -0.0027989, 0.0033202
3: -0.0054935, -0.0019924, -0.0055919, -0.0022649, -0.0028948, 0.0034339
4: -0.0018801, 0.0019101, -0.0015851, 0.0020167, -0.0037174, 0.0031338
5: 0.0023058, 0.0058925, 0.0022049, 0.0056134, -0.0029656, 0.0035179
6: -0.0131516, 0.0010795, -0.0135519, -0.0000281, -0.0117666, 0.0139580
7: -0.0040269, 0.0153547, -0.0025184, 0.0158998, -0.0190096, 0.0160251
8: 0.9863772, 1.0000300, 0.9874398, 1.0004140, -0.0133907, 0.0112884
9: -0.0159145, -0.0035214, -0.0162631, -0.0044860, -0.0102469, 0.0121552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072544
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072588
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037249, 0.0098744, 0.0046300, 0.0101610, -0.0060727, 0.0049289
1: 0.0018604, 0.0027489, 0.0019912, 0.0027903, -0.0008773, 0.0007121
2: 0.0089005, 0.0123004, 0.0087421, 0.0118001, -0.0027251, 0.0033574
3: -0.0054751, -0.0019588, -0.0056390, -0.0024763, -0.0028184, 0.0034724
4: -0.0019165, 0.0018902, -0.0013563, 0.0020676, -0.0037591, 0.0030511
5: 0.0023246, 0.0059270, 0.0021567, 0.0053968, -0.0028874, 0.0035574
6: -0.0130769, 0.0012162, -0.0137430, -0.0008873, -0.0114562, 0.0141146
7: -0.0042131, 0.0152529, -0.0013483, 0.0161601, -0.0192228, 0.0156024
8: 0.9862461, 0.9999584, 0.9882641, 1.0005974, -0.0135410, 0.0109907
9: -0.0158495, -0.0034024, -0.0164296, -0.0052342, -0.0099766, 0.0122916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072133, upper bound: 0.0070059
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072968, upper bound: 0.0070134
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0035687, 0.0098847, 0.0042731, 0.0100283, -0.0062546, 0.0050598
1: 0.0018379, 0.0027504, 0.0019396, 0.0027711, -0.0009036, 0.0007310
2: 0.0088948, 0.0123868, 0.0088155, 0.0119974, -0.0027975, 0.0034580
3: -0.0054810, -0.0018694, -0.0055631, -0.0022722, -0.0028933, 0.0035764
4: -0.0020132, 0.0018965, -0.0015772, 0.0019854, -0.0038717, 0.0031321
5: 0.0023186, 0.0060185, 0.0022345, 0.0056059, -0.0029640, 0.0036639
6: -0.0131009, 0.0015795, -0.0134346, -0.0000579, -0.0117605, 0.0145374
7: -0.0047078, 0.0152855, -0.0024779, 0.0157400, -0.0197987, 0.0160167
8: 0.9858975, 0.9999813, 0.9874684, 1.0003015, -0.0139466, 0.0112825
9: -0.0158703, -0.0030860, -0.0161609, -0.0045119, -0.0102415, 0.0126598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073700
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073737
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0039347, 0.0098961, 0.0042106, 0.0102272, -0.0054043, 0.0048629
1: 0.0018908, 0.0027520, 0.0019306, 0.0027998, -0.0007808, 0.0007025
2: 0.0088886, 0.0121845, 0.0087055, 0.0120319, -0.0026886, 0.0029879
3: -0.0054875, -0.0020787, -0.0056768, -0.0022365, -0.0027807, 0.0030902
4: -0.0017866, 0.0019036, -0.0016158, 0.0021085, -0.0033453, 0.0030102
5: 0.0023119, 0.0058041, 0.0021180, 0.0056425, -0.0028487, 0.0031658
6: -0.0131272, 0.0007286, -0.0138968, 0.0000873, -0.0113027, 0.0125610
7: -0.0035490, 0.0153215, -0.0026756, 0.0163695, -0.0171070, 0.0153934
8: 0.9867139, 1.0000066, 0.9873291, 1.0007448, -0.0120506, 0.0108434
9: -0.0158933, -0.0038270, -0.0165634, -0.0043855, -0.0098429, 0.0109387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072143, upper bound: 0.0068846
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072971, upper bound: 0.0068889
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037838, 0.0099066, 0.0038844, 0.0100688, -0.0055648, 0.0049687
1: 0.0018689, 0.0027535, 0.0018835, 0.0027769, -0.0008040, 0.0007178
2: 0.0088828, 0.0122679, 0.0087931, 0.0122123, -0.0027471, 0.0030766
3: -0.0054935, -0.0019924, -0.0055862, -0.0020500, -0.0028412, 0.0031820
4: -0.0018801, 0.0019101, -0.0018177, 0.0020105, -0.0034447, 0.0030757
5: 0.0023058, 0.0058925, 0.0022108, 0.0058336, -0.0029107, 0.0032599
6: -0.0131516, 0.0010795, -0.0135286, 0.0008455, -0.0115487, 0.0129342
7: -0.0040269, 0.0153547, -0.0037082, 0.0158680, -0.0176152, 0.0157283
8: 0.9863772, 1.0000300, 0.9866017, 1.0003917, -0.0124085, 0.0110793
9: -0.0159145, -0.0035214, -0.0162428, -0.0037252, -0.0100571, 0.0112636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072273
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072285
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037249, 0.0098744, 0.0042243, 0.0101699, -0.0056762, 0.0048721
1: 0.0018604, 0.0027489, 0.0019326, 0.0027915, -0.0008200, 0.0007039
2: 0.0089005, 0.0123004, 0.0087372, 0.0120244, -0.0026937, 0.0031382
3: -0.0054751, -0.0019588, -0.0056440, -0.0022443, -0.0027859, 0.0032457
4: -0.0019165, 0.0018902, -0.0016074, 0.0020730, -0.0035137, 0.0030159
5: 0.0023246, 0.0059270, 0.0021516, 0.0056345, -0.0028541, 0.0033251
6: -0.0130769, 0.0012162, -0.0137635, 0.0000556, -0.0113242, 0.0131930
7: -0.0042131, 0.0152529, -0.0026325, 0.0161880, -0.0179678, 0.0154226
8: 0.9862461, 0.9999584, 0.9873595, 1.0006171, -0.0126569, 0.0108640
9: -0.0158495, -0.0034024, -0.0164474, -0.0044131, -0.0098616, 0.0114891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072143, upper bound: 0.0069971
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072971, upper bound: 0.0070035
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0035687, 0.0098847, 0.0038975, 0.0100161, -0.0058460, 0.0049789
1: 0.0018379, 0.0027504, 0.0018854, 0.0027693, -0.0008446, 0.0007193
2: 0.0088948, 0.0123868, 0.0088222, 0.0122050, -0.0027527, 0.0032321
3: -0.0054810, -0.0018694, -0.0055561, -0.0020574, -0.0028470, 0.0033428
4: -0.0020132, 0.0018965, -0.0018097, 0.0019778, -0.0036188, 0.0030820
5: 0.0023186, 0.0060185, 0.0022416, 0.0058259, -0.0029166, 0.0034246
6: -0.0131009, 0.0015795, -0.0134062, 0.0008151, -0.0115722, 0.0135877
7: -0.0047078, 0.0152855, -0.0036668, 0.0157013, -0.0185052, 0.0157604
8: 0.9858975, 0.9999813, 0.9866309, 1.0002742, -0.0130355, 0.0111019
9: -0.0158703, -0.0030860, -0.0161362, -0.0037517, -0.0100776, 0.0118328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073442
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073488
time: 1.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.16 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0070717, upper bound: 0.0068822
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0068901
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0072544
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0072588
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0070718, upper bound: 0.0070059
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0070134
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0073700
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0073737
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0070718, upper bound: 0.0071192
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0071246
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0074127
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0074215
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0070718, upper bound: 0.0072101
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0071356, upper bound: 0.0072154
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0075123
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072737, upper bound: 0.0075202
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072133, upper bound: 0.0068822
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072968, upper bound: 0.0068901
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072544
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072588
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072133, upper bound: 0.0070059
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072968, upper bound: 0.0070134
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073700
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073737
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072143, upper bound: 0.0068846
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072971, upper bound: 0.0068889
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072273
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0072285
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072143, upper bound: 0.0069971
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0072971, upper bound: 0.0070035
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073442
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 8, lower bound: -0.0074592, upper bound: 0.0073488

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047050, 0.0100519, 0.0047327, 0.0102007, -0.0047788, 0.0045793
1: 0.0020020, 0.0027745, 0.0020060, 0.0027960, -0.0006904, 0.0006616
2: 0.0088024, 0.0117586, 0.0087202, 0.0117433, -0.0025318, 0.0026421
3: -0.0055766, -0.0025192, -0.0056617, -0.0025350, -0.0026185, 0.0027326
4: -0.0013098, 0.0020000, -0.0012927, 0.0020921, -0.0029582, 0.0028347
5: 0.0022207, 0.0053528, 0.0021335, 0.0053367, -0.0026826, 0.0027994
6: -0.0134894, -0.0010618, -0.0138352, -0.0011260, -0.0106437, 0.0111072
7: -0.0011106, 0.0158147, -0.0010232, 0.0162856, -0.0151271, 0.0144958
8: 0.9884315, 1.0003541, 0.9884930, 1.0006858, -0.0106558, 0.0102111
9: -0.0162087, -0.0053862, -0.0165098, -0.0054421, -0.0092690, 0.0096727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0068822
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0068822
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0044666, 0.0098937, 0.0046162, 0.0102117, -0.0048839, 0.0046847
1: 0.0019676, 0.0027517, 0.0019892, 0.0027976, -0.0007056, 0.0006768
2: 0.0088899, 0.0118904, 0.0087141, 0.0118077, -0.0025900, 0.0027002
3: -0.0054861, -0.0023828, -0.0056679, -0.0024684, -0.0026787, 0.0027927
4: -0.0014574, 0.0019021, -0.0013648, 0.0020989, -0.0030232, 0.0028999
5: 0.0023133, 0.0054925, 0.0021271, 0.0054049, -0.0027443, 0.0028610
6: -0.0131217, -0.0005076, -0.0138607, -0.0008553, -0.0108885, 0.0113515
7: -0.0018655, 0.0153139, -0.0013919, 0.0163203, -0.0154598, 0.0148292
8: 0.9878998, 1.0000013, 0.9882334, 1.0007102, -0.0108902, 0.0104460
9: -0.0158885, -0.0049035, -0.0165320, -0.0052063, -0.0094822, 0.0098854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0068901
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0068901
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046533, 0.0100422, 0.0042603, 0.0100788, -0.0048538, 0.0051625
1: 0.0019946, 0.0027731, 0.0019378, 0.0027784, -0.0007012, 0.0007458
2: 0.0088078, 0.0117872, 0.0087876, 0.0120045, -0.0028542, 0.0026835
3: -0.0055710, -0.0024896, -0.0055919, -0.0022649, -0.0029520, 0.0027755
4: -0.0013418, 0.0019940, -0.0015851, 0.0020167, -0.0030046, 0.0031957
5: 0.0022263, 0.0053832, 0.0022049, 0.0056134, -0.0030242, 0.0028434
6: -0.0134668, -0.0009415, -0.0135519, -0.0000281, -0.0119990, 0.0112816
7: -0.0012744, 0.0157840, -0.0025184, 0.0158998, -0.0153645, 0.0163416
8: 0.9883162, 1.0003325, 0.9874398, 1.0004140, -0.0108231, 0.0115114
9: -0.0161890, -0.0052814, -0.0162631, -0.0044860, -0.0104493, 0.0098245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0069653
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0070169
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042965, 0.0099056, 0.0042603, 0.0100788, -0.0049352, 0.0047870
1: 0.0019430, 0.0027534, 0.0019378, 0.0027784, -0.0007130, 0.0006916
2: 0.0088833, 0.0119844, 0.0087876, 0.0120045, -0.0026466, 0.0027285
3: -0.0054929, -0.0022856, -0.0055919, -0.0022649, -0.0027372, 0.0028220
4: -0.0015627, 0.0019094, -0.0015851, 0.0020167, -0.0030550, 0.0029632
5: 0.0023064, 0.0055922, 0.0022049, 0.0056134, -0.0028042, 0.0028910
6: -0.0131493, -0.0001123, -0.0135519, -0.0000281, -0.0111262, 0.0114707
7: -0.0024038, 0.0153515, -0.0025184, 0.0158998, -0.0156221, 0.0151529
8: 0.9875206, 1.0000278, 0.9874398, 1.0004140, -0.0110045, 0.0106741
9: -0.0159125, -0.0045593, -0.0162631, -0.0044860, -0.0096892, 0.0099892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0069033
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0069005
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0044503, 0.0100281, 0.0047467, 0.0101501, -0.0050854, 0.0045942
1: 0.0019652, 0.0027711, 0.0020081, 0.0027887, -0.0007347, 0.0006637
2: 0.0088156, 0.0118994, 0.0087481, 0.0117355, -0.0025400, 0.0028116
3: -0.0055629, -0.0023735, -0.0056327, -0.0025430, -0.0026270, 0.0029079
4: -0.0014675, 0.0019853, -0.0012840, 0.0020608, -0.0031479, 0.0028439
5: 0.0022346, 0.0055021, 0.0021631, 0.0053284, -0.0026913, 0.0029790
6: -0.0134340, -0.0004698, -0.0137176, -0.0011587, -0.0106783, 0.0118198
7: -0.0019169, 0.0157392, -0.0009786, 0.0161255, -0.0160976, 0.0145429
8: 0.9878636, 1.0003009, 0.9885245, 1.0005730, -0.0113395, 0.0102443
9: -0.0161604, -0.0048706, -0.0164074, -0.0054706, -0.0092991, 0.0102932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067461, upper bound: 0.0065914
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067430, upper bound: 0.0066460
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042096, 0.0098866, 0.0046302, 0.0101610, -0.0051761, 0.0047053
1: 0.0019305, 0.0027506, 0.0019912, 0.0027903, -0.0007478, 0.0006798
2: 0.0088938, 0.0120325, 0.0087421, 0.0118000, -0.0026014, 0.0028617
3: -0.0054821, -0.0022359, -0.0056390, -0.0024764, -0.0026905, 0.0029597
4: -0.0016165, 0.0018977, -0.0013561, 0.0020676, -0.0032041, 0.0029126
5: 0.0023175, 0.0056431, 0.0021567, 0.0053967, -0.0027563, 0.0030321
6: -0.0131053, 0.0000898, -0.0137430, -0.0008878, -0.0109364, 0.0120306
7: -0.0026790, 0.0152915, -0.0013475, 0.0161600, -0.0163847, 0.0148944
8: 0.9873267, 0.9999855, 0.9882646, 1.0005974, -0.0115417, 0.0104919
9: -0.0158742, -0.0043833, -0.0164295, -0.0052347, -0.0095239, 0.0104768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0069964
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0070134
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044000, 0.0100261, 0.0042731, 0.0100283, -0.0051432, 0.0051673
1: 0.0019580, 0.0027708, 0.0019396, 0.0027711, -0.0007430, 0.0007465
2: 0.0088167, 0.0119272, 0.0088155, 0.0119974, -0.0028568, 0.0028435
3: -0.0055618, -0.0023448, -0.0055631, -0.0022722, -0.0029547, 0.0029409
4: -0.0014986, 0.0019841, -0.0015772, 0.0019854, -0.0031837, 0.0031986
5: 0.0022358, 0.0055315, 0.0022345, 0.0056059, -0.0030270, 0.0030129
6: -0.0134295, -0.0003528, -0.0134346, -0.0000579, -0.0120101, 0.0119541
7: -0.0020762, 0.0157331, -0.0024779, 0.0157400, -0.0162805, 0.0163567
8: 0.9877514, 1.0002966, 0.9874684, 1.0003015, -0.0114683, 0.0115220
9: -0.0161565, -0.0047688, -0.0161609, -0.0045119, -0.0104589, 0.0104102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0070718
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0071355
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0040493, 0.0098978, 0.0042731, 0.0100283, -0.0052180, 0.0047968
1: 0.0019073, 0.0027522, 0.0019396, 0.0027711, -0.0007539, 0.0006930
2: 0.0088876, 0.0121211, 0.0088155, 0.0119974, -0.0026520, 0.0028849
3: -0.0054884, -0.0021443, -0.0055631, -0.0022722, -0.0027429, 0.0029837
4: -0.0017157, 0.0019046, -0.0015772, 0.0019854, -0.0032300, 0.0029693
5: 0.0023110, 0.0057370, 0.0022345, 0.0056059, -0.0028100, 0.0030567
6: -0.0131311, 0.0004623, -0.0134346, -0.0000579, -0.0111492, 0.0121281
7: -0.0031862, 0.0153267, -0.0024779, 0.0157400, -0.0165174, 0.0151842
8: 0.9869694, 1.0000104, 0.9874684, 1.0003015, -0.0116352, 0.0106961
9: -0.0158967, -0.0040590, -0.0161609, -0.0045119, -0.0097092, 0.0105617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0070171
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0070245
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047050, 0.0100519, 0.0043306, 0.0102171, -0.0050560, 0.0052206
1: 0.0020020, 0.0027745, 0.0019479, 0.0027984, -0.0007304, 0.0007542
2: 0.0088024, 0.0117586, 0.0087111, 0.0119656, -0.0028863, 0.0027953
3: -0.0055766, -0.0025192, -0.0056710, -0.0023051, -0.0029852, 0.0028910
4: -0.0013098, 0.0020000, -0.0015416, 0.0021023, -0.0031297, 0.0032316
5: 0.0022207, 0.0053528, 0.0021239, 0.0055722, -0.0030582, 0.0029618
6: -0.0134894, -0.0010618, -0.0138733, -0.0001914, -0.0121340, 0.0117515
7: -0.0011106, 0.0158147, -0.0022960, 0.0163376, -0.0160045, 0.0165255
8: 0.9884315, 1.0003541, 0.9875965, 1.0007223, -0.0112739, 0.0116409
9: -0.0162087, -0.0053862, -0.0165430, -0.0046282, -0.0105668, 0.0102337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0071192
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0071192
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0044666, 0.0098937, 0.0042109, 0.0102272, -0.0052141, 0.0053209
1: 0.0019676, 0.0027517, 0.0019307, 0.0027998, -0.0007533, 0.0007687
2: 0.0088899, 0.0118904, 0.0087055, 0.0120318, -0.0029418, 0.0028827
3: -0.0054861, -0.0023828, -0.0056768, -0.0022366, -0.0030426, 0.0029815
4: -0.0014574, 0.0019021, -0.0016157, 0.0021085, -0.0032276, 0.0032937
5: 0.0023133, 0.0054925, 0.0021180, 0.0056423, -0.0031170, 0.0030544
6: -0.0131217, -0.0005076, -0.0138967, 0.0000867, -0.0123673, 0.0121190
7: -0.0018655, 0.0153139, -0.0026748, 0.0163694, -0.0165050, 0.0168432
8: 0.9878998, 1.0000013, 0.9873297, 1.0007449, -0.0116265, 0.0118647
9: -0.0158885, -0.0049035, -0.0165634, -0.0043860, -0.0107700, 0.0105538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0071246
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0071246
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046533, 0.0100422, 0.0038844, 0.0100688, -0.0050729, 0.0057482
1: 0.0019946, 0.0027731, 0.0018835, 0.0027769, -0.0007329, 0.0008304
2: 0.0088078, 0.0117872, 0.0087931, 0.0122123, -0.0031780, 0.0028047
3: -0.0055710, -0.0024896, -0.0055862, -0.0020500, -0.0032868, 0.0029007
4: -0.0013418, 0.0019940, -0.0018177, 0.0020105, -0.0031402, 0.0035582
5: 0.0022263, 0.0053832, 0.0022108, 0.0058336, -0.0033673, 0.0029717
6: -0.0134668, -0.0009415, -0.0135286, 0.0008455, -0.0133603, 0.0117909
7: -0.0012744, 0.0157840, -0.0037082, 0.0158680, -0.0160582, 0.0181956
8: 0.9883162, 1.0003325, 0.9866017, 1.0003917, -0.0113117, 0.0128173
9: -0.0161890, -0.0052814, -0.0162428, -0.0037252, -0.0116347, 0.0102680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0071215
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0071956
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042965, 0.0099056, 0.0038844, 0.0100688, -0.0051956, 0.0054319
1: 0.0019430, 0.0027534, 0.0018835, 0.0027769, -0.0007506, 0.0007848
2: 0.0088833, 0.0119844, 0.0087931, 0.0122123, -0.0030032, 0.0028725
3: -0.0054929, -0.0022856, -0.0055862, -0.0020500, -0.0031060, 0.0029709
4: -0.0015627, 0.0019094, -0.0018177, 0.0020105, -0.0032162, 0.0033624
5: 0.0023064, 0.0055922, 0.0022108, 0.0058336, -0.0031820, 0.0030436
6: -0.0131493, -0.0001123, -0.0135286, 0.0008455, -0.0126253, 0.0120760
7: -0.0024038, 0.0153515, -0.0037082, 0.0158680, -0.0164464, 0.0171945
8: 0.9875206, 1.0000278, 0.9866017, 1.0003917, -0.0115852, 0.0121122
9: -0.0159125, -0.0045593, -0.0162428, -0.0037252, -0.0109946, 0.0105163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0071106
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0071343
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0044503, 0.0100281, 0.0043424, 0.0101599, -0.0053455, 0.0052356
1: 0.0019652, 0.0027711, 0.0019497, 0.0027901, -0.0007723, 0.0007564
2: 0.0088156, 0.0118994, 0.0087427, 0.0119590, -0.0028946, 0.0029554
3: -0.0055629, -0.0023735, -0.0056384, -0.0023119, -0.0029938, 0.0030566
4: -0.0014675, 0.0019853, -0.0015342, 0.0020669, -0.0033089, 0.0032409
5: 0.0022346, 0.0055021, 0.0021574, 0.0055653, -0.0030670, 0.0031314
6: -0.0134340, -0.0004698, -0.0137405, -0.0002190, -0.0121690, 0.0124243
7: -0.0019169, 0.0157392, -0.0022584, 0.0161567, -0.0169209, 0.0165731
8: 0.9878636, 1.0003009, 0.9876230, 1.0005950, -0.0119194, 0.0116745
9: -0.0161604, -0.0048706, -0.0164274, -0.0046523, -0.0105973, 0.0108197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067461, upper bound: 0.0067936
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067430, upper bound: 0.0069075
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042096, 0.0098866, 0.0042245, 0.0101698, -0.0054841, 0.0053402
1: 0.0019305, 0.0027506, 0.0019326, 0.0027915, -0.0007923, 0.0007715
2: 0.0088938, 0.0120325, 0.0087372, 0.0120242, -0.0029525, 0.0030320
3: -0.0054821, -0.0022359, -0.0056440, -0.0022444, -0.0030536, 0.0031359
4: -0.0016165, 0.0018977, -0.0016072, 0.0020730, -0.0033948, 0.0033057
5: 0.0023175, 0.0056431, 0.0021516, 0.0056343, -0.0031283, 0.0032126
6: -0.0131053, 0.0000898, -0.0137635, 0.0000550, -0.0124121, 0.0127466
7: -0.0026790, 0.0152915, -0.0026317, 0.0161880, -0.0173597, 0.0169042
8: 0.9873267, 0.9999855, 0.9873601, 1.0006170, -0.0122286, 0.0119077
9: -0.0158742, -0.0043833, -0.0164474, -0.0044136, -0.0108090, 0.0111003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0072005
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0072155
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044000, 0.0100261, 0.0038975, 0.0100161, -0.0053490, 0.0057514
1: 0.0019580, 0.0027708, 0.0018854, 0.0027693, -0.0007728, 0.0008309
2: 0.0088167, 0.0119272, 0.0088222, 0.0122050, -0.0031798, 0.0029573
3: -0.0055618, -0.0023448, -0.0055561, -0.0020574, -0.0032887, 0.0030586
4: -0.0014986, 0.0019841, -0.0018097, 0.0019778, -0.0033111, 0.0035602
5: 0.0022358, 0.0055315, 0.0022416, 0.0058259, -0.0033692, 0.0031334
6: -0.0134295, -0.0003528, -0.0134062, 0.0008151, -0.0133679, 0.0124326
7: -0.0020762, 0.0157331, -0.0036668, 0.0157013, -0.0169321, 0.0182060
8: 0.9877514, 1.0002966, 0.9866309, 1.0002742, -0.0119273, 0.0128247
9: -0.0161565, -0.0047688, -0.0161362, -0.0037517, -0.0116414, 0.0108268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0072133
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0072968
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0040493, 0.0098978, 0.0038975, 0.0100161, -0.0054636, 0.0054401
1: 0.0019073, 0.0027522, 0.0018854, 0.0027693, -0.0007893, 0.0007859
2: 0.0088876, 0.0121211, 0.0088222, 0.0122050, -0.0030077, 0.0030207
3: -0.0054884, -0.0021443, -0.0055561, -0.0020574, -0.0031107, 0.0031241
4: -0.0017157, 0.0019046, -0.0018097, 0.0019778, -0.0033820, 0.0033675
5: 0.0023110, 0.0057370, 0.0022416, 0.0058259, -0.0031868, 0.0032005
6: -0.0131311, 0.0004623, -0.0134062, 0.0008151, -0.0126443, 0.0126988
7: -0.0031862, 0.0153267, -0.0036668, 0.0157013, -0.0172947, 0.0172204
8: 0.9869694, 1.0000104, 0.9866309, 1.0002742, -0.0121828, 0.0121304
9: -0.0158967, -0.0040590, -0.0161362, -0.0037517, -0.0110112, 0.0110587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0072044
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0072251
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043171, 0.0100194, 0.0047327, 0.0102007, -0.0053924, 0.0048388
1: 0.0019460, 0.0027698, 0.0020060, 0.0027960, -0.0007790, 0.0006991
2: 0.0088204, 0.0119730, 0.0087202, 0.0117433, -0.0026753, 0.0029813
3: -0.0055580, -0.0022974, -0.0056617, -0.0025350, -0.0027669, 0.0030834
4: -0.0015499, 0.0019799, -0.0012927, 0.0020921, -0.0033380, 0.0029953
5: 0.0022397, 0.0055801, 0.0021335, 0.0053367, -0.0028346, 0.0031588
6: -0.0134138, -0.0001602, -0.0138352, -0.0011260, -0.0112468, 0.0125334
7: -0.0023386, 0.0157118, -0.0010232, 0.0162856, -0.0170694, 0.0153172
8: 0.9875665, 1.0002816, 0.9884930, 1.0006858, -0.0120240, 0.0107897
9: -0.0161429, -0.0046010, -0.0165098, -0.0054421, -0.0097942, 0.0109146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071213, upper bound: 0.0068822
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071213, upper bound: 0.0068822
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0040684, 0.0098877, 0.0046162, 0.0102117, -0.0055052, 0.0049102
1: 0.0019101, 0.0027508, 0.0019892, 0.0027976, -0.0007953, 0.0007094
2: 0.0088932, 0.0121106, 0.0087141, 0.0118077, -0.0027147, 0.0030437
3: -0.0054827, -0.0021551, -0.0056679, -0.0024684, -0.0028077, 0.0031479
4: -0.0017039, 0.0018984, -0.0013648, 0.0020989, -0.0034078, 0.0030395
5: 0.0023169, 0.0057258, 0.0021271, 0.0054049, -0.0028764, 0.0032249
6: -0.0131077, 0.0004180, -0.0138607, -0.0008553, -0.0114126, 0.0127956
7: -0.0031260, 0.0152948, -0.0013919, 0.0163203, -0.0174266, 0.0155430
8: 0.9870118, 0.9999878, 0.9882334, 1.0007102, -0.0122756, 0.0109488
9: -0.0158763, -0.0040975, -0.0165320, -0.0052063, -0.0099386, 0.0111430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071964, upper bound: 0.0068901
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071964, upper bound: 0.0068901
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042495, 0.0100614, 0.0042603, 0.0100788, -0.0054857, 0.0054477
1: 0.0019362, 0.0027759, 0.0019378, 0.0027784, -0.0007925, 0.0007870
2: 0.0087972, 0.0120104, 0.0087876, 0.0120045, -0.0030119, 0.0030329
3: -0.0055820, -0.0022587, -0.0055919, -0.0022649, -0.0031150, 0.0031368
4: -0.0015918, 0.0020059, -0.0015851, 0.0020167, -0.0033957, 0.0033722
5: 0.0022151, 0.0056197, 0.0022049, 0.0056134, -0.0031912, 0.0032135
6: -0.0135114, -0.0000030, -0.0135519, -0.0000281, -0.0126619, 0.0127503
7: -0.0025526, 0.0158446, -0.0025184, 0.0158998, -0.0173648, 0.0172444
8: 0.9874157, 1.0003752, 0.9874398, 1.0004140, -0.0122321, 0.0121473
9: -0.0162278, -0.0044642, -0.0162631, -0.0044860, -0.0110265, 0.0111035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0069653
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0070169
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039213, 0.0098982, 0.0042603, 0.0100788, -0.0055762, 0.0050540
1: 0.0018888, 0.0027523, 0.0019378, 0.0027784, -0.0008056, 0.0007302
2: 0.0088874, 0.0121919, 0.0087876, 0.0120045, -0.0027942, 0.0030830
3: -0.0054887, -0.0020711, -0.0055919, -0.0022649, -0.0028899, 0.0031885
4: -0.0017949, 0.0019049, -0.0015851, 0.0020167, -0.0034518, 0.0031285
5: 0.0023107, 0.0058119, 0.0022049, 0.0056134, -0.0029606, 0.0032666
6: -0.0131322, 0.0007597, -0.0135519, -0.0000281, -0.0117470, 0.0129607
7: -0.0035914, 0.0153282, -0.0025184, 0.0158998, -0.0176514, 0.0159984
8: 0.9866840, 1.0000114, 0.9874398, 1.0004140, -0.0124340, 0.0112696
9: -0.0158976, -0.0037999, -0.0162631, -0.0044860, -0.0102298, 0.0112868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0069033
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0069006
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0040921, 0.0099941, 0.0047467, 0.0101501, -0.0056398, 0.0048366
1: 0.0019135, 0.0027662, 0.0020081, 0.0027887, -0.0008148, 0.0006987
2: 0.0088344, 0.0120974, 0.0087481, 0.0117355, -0.0026740, 0.0031181
3: -0.0055435, -0.0021687, -0.0056327, -0.0025430, -0.0027656, 0.0032249
4: -0.0016892, 0.0019643, -0.0012840, 0.0020608, -0.0034912, 0.0029939
5: 0.0022545, 0.0057119, 0.0021631, 0.0053284, -0.0028333, 0.0033038
6: -0.0133551, 0.0003628, -0.0137176, -0.0011587, -0.0112415, 0.0131086
7: -0.0030508, 0.0156318, -0.0009786, 0.0161255, -0.0178527, 0.0153100
8: 0.9870648, 1.0002252, 0.9885245, 1.0005730, -0.0125758, 0.0107847
9: -0.0160917, -0.0041456, -0.0164074, -0.0054706, -0.0097896, 0.0114155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069281, upper bound: 0.0065914
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069279, upper bound: 0.0066460
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038515, 0.0098665, 0.0046302, 0.0101610, -0.0057319, 0.0049202
1: 0.0018787, 0.0027477, 0.0019912, 0.0027903, -0.0008281, 0.0007108
2: 0.0089050, 0.0122305, 0.0087421, 0.0118000, -0.0027202, 0.0031690
3: -0.0054705, -0.0020312, -0.0056390, -0.0024764, -0.0028134, 0.0032775
4: -0.0018381, 0.0018852, -0.0013561, 0.0020676, -0.0035481, 0.0030457
5: 0.0023293, 0.0058528, 0.0021567, 0.0053967, -0.0028822, 0.0033577
6: -0.0130584, 0.0009220, -0.0137430, -0.0008878, -0.0114358, 0.0133225
7: -0.0038124, 0.0152276, -0.0013475, 0.0161600, -0.0181441, 0.0155746
8: 0.9865284, 0.9999405, 0.9882646, 1.0005974, -0.0127811, 0.0109710
9: -0.0158333, -0.0036586, -0.0164295, -0.0052347, -0.0099588, 0.0116018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072918, upper bound: 0.0069964
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072918, upper bound: 0.0070134
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0040316, 0.0100245, 0.0042731, 0.0100283, -0.0057274, 0.0054364
1: 0.0019048, 0.0027705, 0.0019396, 0.0027711, -0.0008274, 0.0007854
2: 0.0088176, 0.0121309, 0.0088155, 0.0119974, -0.0030056, 0.0031665
3: -0.0055609, -0.0021341, -0.0055631, -0.0022722, -0.0031086, 0.0032750
4: -0.0017266, 0.0019830, -0.0015772, 0.0019854, -0.0035454, 0.0033652
5: 0.0022367, 0.0057473, 0.0022345, 0.0056059, -0.0031846, 0.0033551
6: -0.0134256, 0.0005034, -0.0134346, -0.0000579, -0.0126357, 0.0133121
7: -0.0032423, 0.0157278, -0.0024779, 0.0157400, -0.0181299, 0.0172087
8: 0.9869300, 1.0002929, 0.9874684, 1.0003015, -0.0127711, 0.0121222
9: -0.0161531, -0.0040231, -0.0161609, -0.0045119, -0.0110037, 0.0115928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0070717
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0071356
time: 1.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036996, 0.0098765, 0.0042731, 0.0100283, -0.0058041, 0.0050517
1: 0.0018568, 0.0027492, 0.0019396, 0.0027711, -0.0008385, 0.0007298
2: 0.0088994, 0.0123145, 0.0088155, 0.0119974, -0.0027929, 0.0032089
3: -0.0054763, -0.0019443, -0.0055631, -0.0022722, -0.0028886, 0.0033188
4: -0.0019322, 0.0018914, -0.0015772, 0.0019854, -0.0035928, 0.0031271
5: 0.0023234, 0.0059418, 0.0022345, 0.0056059, -0.0029592, 0.0034000
6: -0.0130817, 0.0012751, -0.0134346, -0.0000579, -0.0117414, 0.0134904
7: -0.0042933, 0.0152594, -0.0024779, 0.0157400, -0.0183727, 0.0159908
8: 0.9861896, 0.9999630, 0.9874684, 1.0003015, -0.0129421, 0.0112643
9: -0.0158536, -0.0033511, -0.0161609, -0.0045119, -0.0102250, 0.0117480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0070171
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0070245
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043171, 0.0100194, 0.0043306, 0.0102171, -0.0049566, 0.0047524
1: 0.0019460, 0.0027698, 0.0019479, 0.0027984, -0.0007161, 0.0006866
2: 0.0088204, 0.0119730, 0.0087111, 0.0119656, -0.0026275, 0.0027404
3: -0.0055580, -0.0022974, -0.0056710, -0.0023051, -0.0027175, 0.0028343
4: -0.0015499, 0.0019799, -0.0015416, 0.0021023, -0.0030682, 0.0029418
5: 0.0022397, 0.0055801, 0.0021239, 0.0055722, -0.0027839, 0.0029036
6: -0.0134138, -0.0001602, -0.0138733, -0.0001914, -0.0110459, 0.0115206
7: -0.0023386, 0.0157118, -0.0022960, 0.0163376, -0.0156901, 0.0150435
8: 0.9875665, 1.0002816, 0.9875965, 1.0007223, -0.0110524, 0.0105970
9: -0.0161429, -0.0046010, -0.0165430, -0.0046282, -0.0096192, 0.0100326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071228, upper bound: 0.0068846
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071228, upper bound: 0.0068846
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0040684, 0.0098877, 0.0042109, 0.0102272, -0.0050768, 0.0048537
1: 0.0019101, 0.0027508, 0.0019307, 0.0027998, -0.0007335, 0.0007012
2: 0.0088932, 0.0121106, 0.0087055, 0.0120318, -0.0026835, 0.0028068
3: -0.0054827, -0.0021551, -0.0056768, -0.0022366, -0.0027754, 0.0029030
4: -0.0017039, 0.0018984, -0.0016157, 0.0021085, -0.0031426, 0.0030045
5: 0.0023169, 0.0057258, 0.0021180, 0.0056423, -0.0028433, 0.0029740
6: -0.0131077, 0.0004180, -0.0138967, 0.0000867, -0.0112813, 0.0118000
7: -0.0031260, 0.0152948, -0.0026748, 0.0163694, -0.0160705, 0.0153641
8: 0.9870118, 0.9999878, 0.9873297, 1.0007449, -0.0113204, 0.0108228
9: -0.0158763, -0.0040975, -0.0165634, -0.0043860, -0.0098242, 0.0102759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071967, upper bound: 0.0068889
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071967, upper bound: 0.0068889
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042495, 0.0100614, 0.0038844, 0.0100688, -0.0050130, 0.0053430
1: 0.0019362, 0.0027759, 0.0018835, 0.0027769, -0.0007242, 0.0007719
2: 0.0087972, 0.0120104, 0.0087931, 0.0122123, -0.0029540, 0.0027715
3: -0.0055820, -0.0022587, -0.0055862, -0.0020500, -0.0030552, 0.0028665
4: -0.0015918, 0.0020059, -0.0018177, 0.0020105, -0.0031031, 0.0033074
5: 0.0022151, 0.0056197, 0.0022108, 0.0058336, -0.0031299, 0.0029366
6: -0.0135114, -0.0000030, -0.0135286, 0.0008455, -0.0124187, 0.0116515
7: -0.0025526, 0.0158446, -0.0037082, 0.0158680, -0.0158684, 0.0169132
8: 0.9874157, 1.0003752, 0.9866017, 1.0003917, -0.0111780, 0.0119140
9: -0.0162278, -0.0044642, -0.0162428, -0.0037252, -0.0108147, 0.0101467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0069443
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0069982
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039213, 0.0098982, 0.0038844, 0.0100688, -0.0051085, 0.0049608
1: 0.0018888, 0.0027523, 0.0018835, 0.0027769, -0.0007380, 0.0007167
2: 0.0088874, 0.0121919, 0.0087931, 0.0122123, -0.0027427, 0.0028244
3: -0.0054887, -0.0020711, -0.0055862, -0.0020500, -0.0028366, 0.0029211
4: -0.0017949, 0.0019049, -0.0018177, 0.0020105, -0.0031623, 0.0030708
5: 0.0023107, 0.0058119, 0.0022108, 0.0058336, -0.0029060, 0.0029926
6: -0.0131322, 0.0007597, -0.0135286, 0.0008455, -0.0115302, 0.0118736
7: -0.0035914, 0.0153282, -0.0037082, 0.0158680, -0.0161708, 0.0157031
8: 0.9866840, 1.0000114, 0.9866017, 1.0003917, -0.0113910, 0.0110616
9: -0.0158976, -0.0037999, -0.0162428, -0.0037252, -0.0100410, 0.0103400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0068960
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0068907
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0040921, 0.0099941, 0.0043424, 0.0101599, -0.0052451, 0.0047563
1: 0.0019135, 0.0027662, 0.0019497, 0.0027901, -0.0007578, 0.0006871
2: 0.0088344, 0.0120974, 0.0087427, 0.0119590, -0.0026296, 0.0028999
3: -0.0055435, -0.0021687, -0.0056384, -0.0023119, -0.0027197, 0.0029992
4: -0.0016892, 0.0019643, -0.0015342, 0.0020669, -0.0032468, 0.0029442
5: 0.0022545, 0.0057119, 0.0021574, 0.0055653, -0.0027862, 0.0030726
6: -0.0133551, 0.0003628, -0.0137405, -0.0002190, -0.0110549, 0.0121910
7: -0.0030508, 0.0156318, -0.0022584, 0.0161567, -0.0166031, 0.0150559
8: 0.9870648, 1.0002252, 0.9876230, 1.0005950, -0.0116956, 0.0106057
9: -0.0160917, -0.0041456, -0.0164274, -0.0046523, -0.0096271, 0.0106165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069541, upper bound: 0.0066404
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069542, upper bound: 0.0067118
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038515, 0.0098665, 0.0042245, 0.0101698, -0.0053453, 0.0048634
1: 0.0018787, 0.0027477, 0.0019326, 0.0027915, -0.0007722, 0.0007026
2: 0.0089050, 0.0122305, 0.0087372, 0.0120242, -0.0026888, 0.0029553
3: -0.0054705, -0.0020312, -0.0056440, -0.0022444, -0.0027809, 0.0030565
4: -0.0018381, 0.0018852, -0.0016072, 0.0020730, -0.0033088, 0.0030105
5: 0.0023293, 0.0058528, 0.0021516, 0.0056343, -0.0028489, 0.0031312
6: -0.0130584, 0.0009220, -0.0137635, 0.0000550, -0.0113038, 0.0124239
7: -0.0038124, 0.0152276, -0.0026317, 0.0161880, -0.0169202, 0.0153948
8: 0.9865284, 0.9999405, 0.9873601, 1.0006170, -0.0119190, 0.0108444
9: -0.0158333, -0.0036586, -0.0164474, -0.0044136, -0.0098439, 0.0108193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072928, upper bound: 0.0069934
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072928, upper bound: 0.0070035
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0040316, 0.0100245, 0.0038975, 0.0100161, -0.0052963, 0.0053479
1: 0.0019048, 0.0027705, 0.0018854, 0.0027693, -0.0007652, 0.0007726
2: 0.0088176, 0.0121309, 0.0088222, 0.0122050, -0.0029567, 0.0029282
3: -0.0055609, -0.0021341, -0.0055561, -0.0020574, -0.0030580, 0.0030284
4: -0.0017266, 0.0019830, -0.0018097, 0.0019778, -0.0032785, 0.0033104
5: 0.0022367, 0.0057473, 0.0022416, 0.0058259, -0.0031328, 0.0031025
6: -0.0134256, 0.0005034, -0.0134062, 0.0008151, -0.0124300, 0.0123100
7: -0.0032423, 0.0157278, -0.0036668, 0.0157013, -0.0167651, 0.0169285
8: 0.9869300, 1.0002929, 0.9866309, 1.0002742, -0.0118097, 0.0119248
9: -0.0161531, -0.0040231, -0.0161362, -0.0037517, -0.0108246, 0.0107201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0070489
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0071200
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036996, 0.0098765, 0.0038975, 0.0100161, -0.0053746, 0.0049708
1: 0.0018568, 0.0027492, 0.0018854, 0.0027693, -0.0007765, 0.0007181
2: 0.0088994, 0.0123145, 0.0088222, 0.0122050, -0.0027482, 0.0029715
3: -0.0054763, -0.0019443, -0.0055561, -0.0020574, -0.0028424, 0.0030733
4: -0.0019322, 0.0018914, -0.0018097, 0.0019778, -0.0033270, 0.0030770
5: 0.0023234, 0.0059418, 0.0022416, 0.0058259, -0.0029119, 0.0031484
6: -0.0130817, 0.0012751, -0.0134062, 0.0008151, -0.0115535, 0.0124921
7: -0.0042933, 0.0152594, -0.0036668, 0.0157013, -0.0170132, 0.0157349
8: 0.9861896, 0.9999630, 0.9866309, 1.0002742, -0.0119845, 0.0110840
9: -0.0158536, -0.0033511, -0.0161362, -0.0037517, -0.0100613, 0.0108787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0070145
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0070093
time: 1.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0068822
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0068822
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0068901
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0068901
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0069653
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0070169
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0069033
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0069005
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0067461, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0067430, upper bound: 0.0066460
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0069964
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0070134
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0070718
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0071355
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070059, upper bound: 0.0070171
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0070245
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0071192
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069681, upper bound: 0.0071192
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0071246
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070250, upper bound: 0.0071246
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0071215
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0071956
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0071106
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0071343
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0067461, upper bound: 0.0067936
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0067430, upper bound: 0.0069075
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0072005
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071290, upper bound: 0.0072155
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0072133
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0072968
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070058, upper bound: 0.0072044
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0070134, upper bound: 0.0072251
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071213, upper bound: 0.0068822
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071213, upper bound: 0.0068822
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071964, upper bound: 0.0068901
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071964, upper bound: 0.0068901
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0069653
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0070169
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0069033
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0069006
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069281, upper bound: 0.0065914
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069279, upper bound: 0.0066460
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072918, upper bound: 0.0069964
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072918, upper bound: 0.0070134
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0070717
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0071356
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072098, upper bound: 0.0070171
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072145, upper bound: 0.0070245
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071228, upper bound: 0.0068846
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071228, upper bound: 0.0068846
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071967, upper bound: 0.0068889
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0071967, upper bound: 0.0068889
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0069443
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0069982
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0068960
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0068907
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069541, upper bound: 0.0066404
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0069542, upper bound: 0.0067118
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072928, upper bound: 0.0069934
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072928, upper bound: 0.0070035
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0070489
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0071200
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072101, upper bound: 0.0070145
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 8, lower bound: -0.0072146, upper bound: 0.0070093

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0047050, 0.0100519, 0.0047700, 0.0100312, -0.0046069, 0.0045522
1: 0.0020020, 0.0027745, 0.0020114, 0.0027715, -0.0006656, 0.0006577
2: 0.0088024, 0.0117586, 0.0088139, 0.0117226, -0.0025168, 0.0025471
3: -0.0055766, -0.0025192, -0.0055647, -0.0025563, -0.0026030, 0.0026343
4: -0.0013098, 0.0020000, -0.0012696, 0.0019872, -0.0028518, 0.0028179
5: 0.0022207, 0.0053528, 0.0022328, 0.0053148, -0.0026667, 0.0026987
6: -0.0134894, -0.0010618, -0.0134412, -0.0012128, -0.0105807, 0.0107078
7: -0.0011106, 0.0158147, -0.0009050, 0.0157490, -0.0145831, 0.0144100
8: 0.9884315, 1.0003541, 0.9885765, 1.0003078, -0.0102726, 0.0101507
9: -0.0162087, -0.0053862, -0.0161667, -0.0055177, -0.0092141, 0.0093248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0065789
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0065759
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0047050, 0.0100519, 0.0045159, 0.0100161, -0.0046514, 0.0048777
1: 0.0020020, 0.0027745, 0.0019747, 0.0027693, -0.0006720, 0.0007047
2: 0.0088024, 0.0117586, 0.0088222, 0.0118631, -0.0026967, 0.0025716
3: -0.0055766, -0.0025192, -0.0055561, -0.0024111, -0.0027891, 0.0026597
4: -0.0013098, 0.0020000, -0.0014268, 0.0019778, -0.0028793, 0.0030193
5: 0.0022207, 0.0053528, 0.0022416, 0.0054636, -0.0028573, 0.0027248
6: -0.0134894, -0.0010618, -0.0134061, -0.0006223, -0.0113370, 0.0108112
7: -0.0011106, 0.0158147, -0.0017092, 0.0157013, -0.0147239, 0.0154400
8: 0.9884315, 1.0003541, 0.9880099, 1.0002742, -0.0103718, 0.0108763
9: -0.0162087, -0.0053862, -0.0161362, -0.0050034, -0.0098728, 0.0094148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0065789
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0065759
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044666, 0.0098937, 0.0046530, 0.0100422, -0.0047206, 0.0046568
1: 0.0019676, 0.0027517, 0.0019945, 0.0027731, -0.0006820, 0.0006728
2: 0.0088899, 0.0118904, 0.0088078, 0.0117873, -0.0025746, 0.0026099
3: -0.0054861, -0.0023828, -0.0055710, -0.0024894, -0.0026628, 0.0026993
4: -0.0014574, 0.0019021, -0.0013420, 0.0019940, -0.0029221, 0.0028826
5: 0.0023133, 0.0054925, 0.0022264, 0.0053833, -0.0027279, 0.0027653
6: -0.0131217, -0.0005076, -0.0134668, -0.0009408, -0.0108237, 0.0109720
7: -0.0018655, 0.0153139, -0.0012754, 0.0157839, -0.0149429, 0.0147409
8: 0.9878998, 1.0000013, 0.9883154, 1.0003324, -0.0105261, 0.0103838
9: -0.0158885, -0.0049035, -0.0161890, -0.0052808, -0.0094257, 0.0095549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0066081
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0066080
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044666, 0.0098937, 0.0043999, 0.0100261, -0.0047569, 0.0049731
1: 0.0019676, 0.0027517, 0.0019580, 0.0027708, -0.0006872, 0.0007185
2: 0.0088899, 0.0118904, 0.0088167, 0.0119273, -0.0027495, 0.0026300
3: -0.0054861, -0.0023828, -0.0055618, -0.0023447, -0.0028436, 0.0027200
4: -0.0014574, 0.0019021, -0.0014987, 0.0019840, -0.0029446, 0.0030784
5: 0.0023133, 0.0054925, 0.0022358, 0.0055316, -0.0029132, 0.0027866
6: -0.0131217, -0.0005076, -0.0134294, -0.0003526, -0.0115587, 0.0110563
7: -0.0018655, 0.0153139, -0.0020765, 0.0157330, -0.0150577, 0.0157420
8: 0.9878998, 1.0000013, 0.9877512, 1.0002965, -0.0106070, 0.0110890
9: -0.0158885, -0.0049035, -0.0161565, -0.0047686, -0.0100659, 0.0096283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0066081
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0066080
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0047705, 0.0100312, 0.0046450, 0.0102201, -0.0047337, 0.0047069
1: 0.0020115, 0.0027715, 0.0019934, 0.0027988, -0.0006839, 0.0006800
2: 0.0088139, 0.0117224, 0.0087095, 0.0117918, -0.0026023, 0.0026171
3: -0.0055647, -0.0025566, -0.0056727, -0.0024848, -0.0026914, 0.0027068
4: -0.0012693, 0.0019872, -0.0013470, 0.0021041, -0.0029302, 0.0029137
5: 0.0022328, 0.0053145, 0.0021221, 0.0053880, -0.0027573, 0.0027730
6: -0.0134412, -0.0012140, -0.0138803, -0.0009222, -0.0109401, 0.0110024
7: -0.0009034, 0.0157490, -0.0013008, 0.0163470, -0.0149844, 0.0148995
8: 0.9885775, 1.0003078, 0.9882976, 1.0007291, -0.0105553, 0.0104955
9: -0.0161667, -0.0055187, -0.0165491, -0.0052646, -0.0095272, 0.0095814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0066877
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0066875
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0046535, 0.0100422, 0.0043851, 0.0100699, -0.0048443, 0.0048295
1: 0.0019946, 0.0027731, 0.0019558, 0.0027771, -0.0006999, 0.0006977
2: 0.0088078, 0.0117870, 0.0087925, 0.0119354, -0.0026701, 0.0026783
3: -0.0055710, -0.0024897, -0.0055869, -0.0023363, -0.0027616, 0.0027700
4: -0.0013417, 0.0019940, -0.0015078, 0.0020111, -0.0029987, 0.0029896
5: 0.0022264, 0.0053830, 0.0022101, 0.0055403, -0.0028291, 0.0028378
6: -0.0134668, -0.0009421, -0.0135312, -0.0003182, -0.0112252, 0.0112595
7: -0.0012737, 0.0157839, -0.0021233, 0.0158716, -0.0153345, 0.0152878
8: 0.9883167, 1.0003324, 0.9877181, 1.0003941, -0.0108020, 0.0107690
9: -0.0161890, -0.0052819, -0.0162451, -0.0047386, -0.0097754, 0.0098053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0070092
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0070169
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044251, 0.0098953, 0.0046450, 0.0102201, -0.0048422, 0.0043608
1: 0.0019616, 0.0027519, 0.0019934, 0.0027988, -0.0006996, 0.0006300
2: 0.0088890, 0.0119134, 0.0087095, 0.0117918, -0.0024110, 0.0026771
3: -0.0054870, -0.0023591, -0.0056727, -0.0024848, -0.0024936, 0.0027688
4: -0.0014831, 0.0019031, -0.0013470, 0.0021041, -0.0029974, 0.0026994
5: 0.0023124, 0.0055169, 0.0021221, 0.0053880, -0.0025546, 0.0028366
6: -0.0131255, -0.0004111, -0.0138803, -0.0009222, -0.0101357, 0.0112546
7: -0.0019969, 0.0153190, -0.0013008, 0.0163470, -0.0153278, 0.0138040
8: 0.9878073, 1.0000049, 0.9882976, 1.0007291, -0.0107972, 0.0097238
9: -0.0158918, -0.0048195, -0.0165491, -0.0052646, -0.0088266, 0.0098010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0066492
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0066492
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042967, 0.0099056, 0.0043851, 0.0100699, -0.0049259, 0.0044917
1: 0.0019431, 0.0027534, 0.0019558, 0.0027771, -0.0007116, 0.0006489
2: 0.0088833, 0.0119843, 0.0087925, 0.0119354, -0.0024833, 0.0027234
3: -0.0054929, -0.0022857, -0.0055869, -0.0023363, -0.0025684, 0.0028167
4: -0.0015625, 0.0019094, -0.0015078, 0.0020111, -0.0030492, 0.0027804
5: 0.0023064, 0.0055920, 0.0022101, 0.0055403, -0.0026312, 0.0028856
6: -0.0131493, -0.0001128, -0.0135312, -0.0003182, -0.0104399, 0.0114491
7: -0.0024031, 0.0153514, -0.0021233, 0.0158716, -0.0155927, 0.0142183
8: 0.9875211, 1.0000277, 0.9877181, 1.0003941, -0.0109838, 0.0100156
9: -0.0159125, -0.0045597, -0.0162451, -0.0047386, -0.0090915, 0.0099704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0068910
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0069006
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044679, 0.0100269, 0.0049192, 0.0101402, -0.0050560, 0.0044108
1: 0.0019678, 0.0027709, 0.0020330, 0.0027873, -0.0007305, 0.0006372
2: 0.0088163, 0.0118897, 0.0087536, 0.0116402, -0.0024386, 0.0027953
3: -0.0055623, -0.0023836, -0.0056271, -0.0026417, -0.0025221, 0.0028911
4: -0.0014566, 0.0019845, -0.0011772, 0.0020547, -0.0031298, 0.0027304
5: 0.0022353, 0.0054918, 0.0021689, 0.0052274, -0.0025838, 0.0029618
6: -0.0134312, -0.0005106, -0.0136947, -0.0015596, -0.0102519, 0.0117516
7: -0.0018613, 0.0157354, -0.0004327, 0.0160942, -0.0160047, 0.0139622
8: 0.9879028, 1.0002983, 0.9889090, 1.0005510, -0.0112740, 0.0098353
9: -0.0161580, -0.0049062, -0.0163874, -0.0058197, -0.0089278, 0.0102338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0065914
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0065914
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045231, 0.0100207, 0.0049076, 0.0104256, -0.0053302, 0.0044454
1: 0.0019758, 0.0027700, 0.0020313, 0.0028285, -0.0007701, 0.0006422
2: 0.0088197, 0.0118591, 0.0085958, 0.0116466, -0.0024577, 0.0029469
3: -0.0055588, -0.0024152, -0.0057903, -0.0026350, -0.0025419, 0.0030479
4: -0.0014224, 0.0019807, -0.0011844, 0.0022313, -0.0032995, 0.0027518
5: 0.0022389, 0.0054594, 0.0020017, 0.0052342, -0.0026041, 0.0031224
6: -0.0134169, -0.0006390, -0.0143580, -0.0015326, -0.0103323, 0.0123889
7: -0.0016865, 0.0157160, -0.0004694, 0.0169977, -0.0168725, 0.0140717
8: 0.9880259, 1.0002846, 0.9888832, 1.0011873, -0.0118854, 0.0099124
9: -0.0161456, -0.0050180, -0.0169651, -0.0057962, -0.0089978, 0.0107888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0066460
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0066460
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042096, 0.0098866, 0.0049511, 0.0103640, -0.0055003, 0.0043244
1: 0.0019305, 0.0027506, 0.0020376, 0.0028196, -0.0007946, 0.0006247
2: 0.0088938, 0.0120325, 0.0086299, 0.0116225, -0.0023908, 0.0030410
3: -0.0054821, -0.0022359, -0.0057550, -0.0026599, -0.0024727, 0.0031451
4: -0.0016165, 0.0018977, -0.0011575, 0.0021932, -0.0034048, 0.0026768
5: 0.0023175, 0.0056431, 0.0020378, 0.0052087, -0.0025332, 0.0032220
6: -0.0131053, 0.0000898, -0.0142148, -0.0016337, -0.0100510, 0.0127841
7: -0.0026790, 0.0152915, -0.0003318, 0.0168026, -0.0174109, 0.0136886
8: 0.9873267, 0.9999855, 0.9889801, 1.0010500, -0.0122646, 0.0096425
9: -0.0158742, -0.0043833, -0.0168404, -0.0058842, -0.0087528, 0.0111330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066374
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066290
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042096, 0.0098866, 0.0047616, 0.0101512, -0.0051662, 0.0043584
1: 0.0019305, 0.0027506, 0.0020102, 0.0027889, -0.0007464, 0.0006297
2: 0.0088938, 0.0120325, 0.0087475, 0.0117273, -0.0024096, 0.0028563
3: -0.0054821, -0.0022359, -0.0056334, -0.0025516, -0.0024922, 0.0029541
4: -0.0016165, 0.0018977, -0.0012747, 0.0020615, -0.0031980, 0.0026979
5: 0.0023175, 0.0056431, 0.0021625, 0.0053197, -0.0025531, 0.0030264
6: -0.0131053, 0.0000898, -0.0137202, -0.0011934, -0.0101301, 0.0120077
7: -0.0026790, 0.0152915, -0.0009314, 0.0161289, -0.0163535, 0.0137963
8: 0.9873267, 0.9999855, 0.9885578, 1.0005754, -0.0115197, 0.0097184
9: -0.0158742, -0.0043833, -0.0164096, -0.0055008, -0.0088217, 0.0104569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066846
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066809
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045162, 0.0100161, 0.0046582, 0.0101664, -0.0050254, 0.0047131
1: 0.0019748, 0.0027693, 0.0019953, 0.0027911, -0.0007260, 0.0006809
2: 0.0088222, 0.0118630, 0.0087391, 0.0117845, -0.0026058, 0.0027784
3: -0.0055561, -0.0024112, -0.0056420, -0.0024924, -0.0026950, 0.0028736
4: -0.0014267, 0.0019778, -0.0013388, 0.0020709, -0.0031108, 0.0029175
5: 0.0022416, 0.0054635, 0.0021536, 0.0053803, -0.0027609, 0.0029439
6: -0.0134061, -0.0006230, -0.0137555, -0.0009529, -0.0109545, 0.0116805
7: -0.0017083, 0.0157013, -0.0012589, 0.0161771, -0.0159078, 0.0149191
8: 0.9880105, 1.0002742, 0.9883271, 1.0006094, -0.0112058, 0.0105094
9: -0.0161362, -0.0050040, -0.0164404, -0.0052913, -0.0095397, 0.0101719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0067461
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0067430
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044002, 0.0100261, 0.0043979, 0.0100195, -0.0051338, 0.0048280
1: 0.0019580, 0.0027708, 0.0019577, 0.0027698, -0.0007417, 0.0006975
2: 0.0088167, 0.0119271, 0.0088203, 0.0119284, -0.0026693, 0.0028383
3: -0.0055618, -0.0023449, -0.0055581, -0.0023436, -0.0027607, 0.0029356
4: -0.0014985, 0.0019840, -0.0014999, 0.0019800, -0.0031779, 0.0029886
5: 0.0022358, 0.0055314, 0.0022396, 0.0055328, -0.0028282, 0.0030074
6: -0.0134294, -0.0003534, -0.0134141, -0.0003480, -0.0112215, 0.0119324
7: -0.0020755, 0.0157330, -0.0020828, 0.0157121, -0.0162509, 0.0152827
8: 0.9877518, 1.0002965, 0.9877468, 1.0002818, -0.0114474, 0.0107655
9: -0.0161565, -0.0047692, -0.0161431, -0.0047646, -0.0097722, 0.0103912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071290
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071356
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041792, 0.0098884, 0.0046582, 0.0101664, -0.0051231, 0.0043719
1: 0.0019261, 0.0027509, 0.0019953, 0.0027911, -0.0007401, 0.0006316
2: 0.0088928, 0.0120493, 0.0087391, 0.0117845, -0.0024171, 0.0028325
3: -0.0054831, -0.0022185, -0.0056420, -0.0024924, -0.0024999, 0.0029295
4: -0.0016353, 0.0018988, -0.0013388, 0.0020709, -0.0031713, 0.0027063
5: 0.0023164, 0.0056609, 0.0021536, 0.0053803, -0.0025610, 0.0030011
6: -0.0131094, 0.0001604, -0.0137555, -0.0009529, -0.0101614, 0.0119076
7: -0.0027752, 0.0152971, -0.0012589, 0.0161771, -0.0162171, 0.0138390
8: 0.9872590, 0.9999894, 0.9883271, 1.0006094, -0.0114237, 0.0097484
9: -0.0158777, -0.0043218, -0.0164404, -0.0052913, -0.0088490, 0.0103697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0067343
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0067343
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0040495, 0.0098977, 0.0043979, 0.0100195, -0.0052088, 0.0044943
1: 0.0019073, 0.0027522, 0.0019577, 0.0027698, -0.0007525, 0.0006493
2: 0.0088877, 0.0121210, 0.0088203, 0.0119284, -0.0024848, 0.0028798
3: -0.0054884, -0.0021444, -0.0055581, -0.0023436, -0.0025699, 0.0029785
4: -0.0017156, 0.0019046, -0.0014999, 0.0019800, -0.0032244, 0.0027821
5: 0.0023110, 0.0057368, 0.0022396, 0.0055328, -0.0026328, 0.0030513
6: -0.0131311, 0.0004618, -0.0134141, -0.0003480, -0.0104460, 0.0121068
7: -0.0031856, 0.0153267, -0.0020828, 0.0157121, -0.0164884, 0.0142266
8: 0.9869699, 1.0000104, 0.9877468, 1.0002818, -0.0116148, 0.0100215
9: -0.0158967, -0.0040594, -0.0161431, -0.0047646, -0.0090968, 0.0105431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0070144
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0070246
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0047050, 0.0100519, 0.0043681, 0.0100513, -0.0048928, 0.0051897
1: 0.0020020, 0.0027745, 0.0019534, 0.0027744, -0.0007069, 0.0007498
2: 0.0088024, 0.0117586, 0.0088028, 0.0119449, -0.0028693, 0.0027051
3: -0.0055766, -0.0025192, -0.0055762, -0.0023265, -0.0029675, 0.0027977
4: -0.0013098, 0.0020000, -0.0015184, 0.0019996, -0.0030287, 0.0032125
5: 0.0022207, 0.0053528, 0.0022210, 0.0055502, -0.0030401, 0.0028662
6: -0.0134894, -0.0010618, -0.0134879, -0.0002786, -0.0120624, 0.0113721
7: -0.0011106, 0.0158147, -0.0021773, 0.0158126, -0.0154879, 0.0164279
8: 0.9884315, 1.0003541, 0.9876801, 1.0003526, -0.0109100, 0.0115722
9: -0.0162087, -0.0053862, -0.0162074, -0.0047041, -0.0105045, 0.0099034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0068571
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0068562
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0047050, 0.0100519, 0.0041466, 0.0100152, -0.0048919, 0.0054623
1: 0.0020020, 0.0027745, 0.0019214, 0.0027692, -0.0007067, 0.0007891
2: 0.0088024, 0.0117586, 0.0088227, 0.0120673, -0.0030199, 0.0027046
3: -0.0055766, -0.0025192, -0.0055556, -0.0021999, -0.0031234, 0.0027973
4: -0.0013098, 0.0020000, -0.0016555, 0.0019773, -0.0030282, 0.0033812
5: 0.0022207, 0.0053528, 0.0022421, 0.0056800, -0.0031998, 0.0028657
6: -0.0134894, -0.0010618, -0.0134042, 0.0002363, -0.0126958, 0.0113702
7: -0.0011106, 0.0158147, -0.0028784, 0.0156986, -0.0154852, 0.0172906
8: 0.9884315, 1.0003541, 0.9871861, 1.0002724, -0.0109081, 0.0121799
9: -0.0162087, -0.0053862, -0.0161345, -0.0042558, -0.0110561, 0.0099017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0068571
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0068562
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044666, 0.0098937, 0.0042498, 0.0100614, -0.0050543, 0.0052881
1: 0.0019676, 0.0027517, 0.0019363, 0.0027759, -0.0007302, 0.0007640
2: 0.0088899, 0.0118904, 0.0087972, 0.0120103, -0.0029236, 0.0027944
3: -0.0054861, -0.0023828, -0.0055820, -0.0022589, -0.0030238, 0.0028901
4: -0.0014574, 0.0019021, -0.0015916, 0.0020059, -0.0031287, 0.0032734
5: 0.0023133, 0.0054925, 0.0022151, 0.0056195, -0.0030977, 0.0029608
6: -0.0131217, -0.0005076, -0.0135114, -0.0000036, -0.0122909, 0.0117476
7: -0.0018655, 0.0153139, -0.0025518, 0.0158446, -0.0159992, 0.0167391
8: 0.9878998, 1.0000013, 0.9874163, 1.0003752, -0.0112702, 0.0117914
9: -0.0158885, -0.0049035, -0.0162278, -0.0044647, -0.0107035, 0.0102303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0068808
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0068808
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0044666, 0.0098937, 0.0040319, 0.0100245, -0.0050133, 0.0055570
1: 0.0019676, 0.0027517, 0.0019048, 0.0027705, -0.0007243, 0.0008028
2: 0.0088899, 0.0118904, 0.0088176, 0.0121308, -0.0030723, 0.0027717
3: -0.0054861, -0.0023828, -0.0055609, -0.0021343, -0.0031775, 0.0028666
4: -0.0014574, 0.0019021, -0.0017265, 0.0019830, -0.0031033, 0.0034399
5: 0.0023133, 0.0054925, 0.0022367, 0.0057472, -0.0032553, 0.0029368
6: -0.0131217, -0.0005076, -0.0134256, 0.0005028, -0.0129159, 0.0116522
7: -0.0018655, 0.0153139, -0.0032415, 0.0157278, -0.0158693, 0.0175904
8: 0.9878998, 1.0000013, 0.9869305, 1.0002929, -0.0111787, 0.0123910
9: -0.0158885, -0.0049035, -0.0161531, -0.0040236, -0.0112477, 0.0101473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0068808
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0068808
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0047705, 0.0100312, 0.0042915, 0.0101888, -0.0049815, 0.0052928
1: 0.0020115, 0.0027715, 0.0019423, 0.0027943, -0.0007197, 0.0007647
2: 0.0088139, 0.0117224, 0.0087267, 0.0119872, -0.0029262, 0.0027541
3: -0.0055647, -0.0025566, -0.0056549, -0.0022828, -0.0030264, 0.0028484
4: -0.0012693, 0.0019872, -0.0015657, 0.0020848, -0.0030836, 0.0032763
5: 0.0022328, 0.0053145, 0.0021405, 0.0055951, -0.0031005, 0.0029181
6: -0.0134412, -0.0012140, -0.0138076, -0.0001007, -0.0123018, 0.0115783
7: -0.0009034, 0.0157490, -0.0024195, 0.0162481, -0.0157686, 0.0167540
8: 0.9885775, 1.0003078, 0.9875094, 1.0006593, -0.0111078, 0.0118019
9: -0.0161667, -0.0055187, -0.0164858, -0.0045493, -0.0107130, 0.0100829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0068752
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0068752
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0046535, 0.0100422, 0.0040108, 0.0100606, -0.0050641, 0.0054250
1: 0.0019946, 0.0027731, 0.0019017, 0.0027758, -0.0007316, 0.0007838
2: 0.0088078, 0.0117870, 0.0087976, 0.0121424, -0.0029993, 0.0027998
3: -0.0055710, -0.0024897, -0.0055815, -0.0021222, -0.0031021, 0.0028957
4: -0.0013417, 0.0019940, -0.0017395, 0.0020054, -0.0031348, 0.0033582
5: 0.0022264, 0.0053830, 0.0022156, 0.0057595, -0.0031779, 0.0029666
6: -0.0134668, -0.0009421, -0.0135095, 0.0005517, -0.0126092, 0.0117704
7: -0.0012737, 0.0157839, -0.0033081, 0.0158421, -0.0160303, 0.0171726
8: 0.9883167, 1.0003324, 0.9868835, 1.0003735, -0.0112921, 0.0120967
9: -0.0161890, -0.0052819, -0.0162262, -0.0039810, -0.0109806, 0.0102502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071916
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071956
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044251, 0.0098953, 0.0042915, 0.0101888, -0.0051573, 0.0050053
1: 0.0019616, 0.0027519, 0.0019423, 0.0027943, -0.0007451, 0.0007231
2: 0.0088890, 0.0119134, 0.0087267, 0.0119872, -0.0027673, 0.0028513
3: -0.0054870, -0.0023591, -0.0056549, -0.0022828, -0.0028621, 0.0029490
4: -0.0014831, 0.0019031, -0.0015657, 0.0020848, -0.0031925, 0.0030983
5: 0.0023124, 0.0055169, 0.0021405, 0.0055951, -0.0029321, 0.0030211
6: -0.0131255, -0.0004111, -0.0138076, -0.0001007, -0.0116336, 0.0119870
7: -0.0019969, 0.0153190, -0.0024195, 0.0162481, -0.0163253, 0.0158440
8: 0.9878073, 1.0000049, 0.9875094, 1.0006593, -0.0114999, 0.0111608
9: -0.0158918, -0.0048195, -0.0164858, -0.0045493, -0.0101311, 0.0104388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0068886
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0068886
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042967, 0.0099056, 0.0040108, 0.0100606, -0.0051868, 0.0051478
1: 0.0019431, 0.0027534, 0.0019017, 0.0027758, -0.0007493, 0.0007437
2: 0.0088833, 0.0119843, 0.0087976, 0.0121424, -0.0028461, 0.0028677
3: -0.0054929, -0.0022857, -0.0055815, -0.0021222, -0.0029436, 0.0029659
4: -0.0015625, 0.0019094, -0.0017395, 0.0020054, -0.0032107, 0.0031866
5: 0.0023064, 0.0055920, 0.0022156, 0.0057595, -0.0030156, 0.0030384
6: -0.0131493, -0.0001128, -0.0135095, 0.0005517, -0.0119650, 0.0120556
7: -0.0024031, 0.0153514, -0.0033081, 0.0158421, -0.0164187, 0.0162953
8: 0.9875211, 1.0000277, 0.9868835, 1.0003735, -0.0115657, 0.0114787
9: -0.0159125, -0.0045597, -0.0162262, -0.0039810, -0.0104196, 0.0104986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0071290
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0071343
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044679, 0.0100269, 0.0045246, 0.0101502, -0.0053165, 0.0050415
1: 0.0019678, 0.0027709, 0.0019760, 0.0027887, -0.0007681, 0.0007284
2: 0.0088163, 0.0118897, 0.0087481, 0.0118583, -0.0027873, 0.0029394
3: -0.0055623, -0.0023836, -0.0056328, -0.0024160, -0.0028828, 0.0030400
4: -0.0014566, 0.0019845, -0.0014215, 0.0020609, -0.0032910, 0.0031208
5: 0.0022353, 0.0054918, 0.0021631, 0.0054585, -0.0029533, 0.0031144
6: -0.0134312, -0.0005106, -0.0137178, -0.0006424, -0.0117178, 0.0123571
7: -0.0018613, 0.0157354, -0.0016818, 0.0161258, -0.0168293, 0.0159587
8: 0.9879028, 1.0002983, 0.9880292, 1.0005732, -0.0118549, 0.0112416
9: -0.0161580, -0.0049062, -0.0164076, -0.0050210, -0.0102044, 0.0107611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0067936
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0067936
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045231, 0.0100207, 0.0045096, 0.0104413, -0.0055889, 0.0050774
1: 0.0019758, 0.0027700, 0.0019738, 0.0028308, -0.0008074, 0.0007335
2: 0.0088197, 0.0118591, 0.0085871, 0.0118666, -0.0028072, 0.0030900
3: -0.0055588, -0.0024152, -0.0057993, -0.0024075, -0.0029033, 0.0031958
4: -0.0014224, 0.0019807, -0.0014307, 0.0022411, -0.0034596, 0.0031430
5: 0.0022389, 0.0054594, 0.0019925, 0.0054673, -0.0029743, 0.0032740
6: -0.0134169, -0.0006390, -0.0143945, -0.0006076, -0.0118013, 0.0129902
7: -0.0016865, 0.0157160, -0.0017292, 0.0170474, -0.0176916, 0.0160723
8: 0.9880259, 1.0002846, 0.9879958, 1.0012225, -0.0124623, 0.0113217
9: -0.0161456, -0.0050180, -0.0169969, -0.0049907, -0.0102771, 0.0113125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0069075
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0069075
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042096, 0.0098866, 0.0045499, 0.0103846, -0.0058182, 0.0049735
1: 0.0019305, 0.0027506, 0.0019796, 0.0028226, -0.0008406, 0.0007185
2: 0.0088938, 0.0120325, 0.0086185, 0.0118444, -0.0027497, 0.0032167
3: -0.0054821, -0.0022359, -0.0057668, -0.0024305, -0.0028439, 0.0033269
4: -0.0016165, 0.0018977, -0.0014058, 0.0022059, -0.0036015, 0.0030787
5: 0.0023175, 0.0056431, 0.0020258, 0.0054437, -0.0029135, 0.0034083
6: -0.0131053, 0.0000898, -0.0142626, -0.0007012, -0.0115597, 0.0135230
7: -0.0026790, 0.0152915, -0.0016018, 0.0168677, -0.0184171, 0.0157434
8: 0.9873267, 0.9999855, 0.9880856, 1.0010958, -0.0129734, 0.0110900
9: -0.0158742, -0.0043833, -0.0168820, -0.0050721, -0.0100667, 0.0117764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0068792
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0068775
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042096, 0.0098866, 0.0043643, 0.0101609, -0.0054746, 0.0050072
1: 0.0019305, 0.0027506, 0.0019528, 0.0027902, -0.0007909, 0.0007234
2: 0.0088938, 0.0120325, 0.0087422, 0.0119470, -0.0027683, 0.0030268
3: -0.0054821, -0.0022359, -0.0056389, -0.0023244, -0.0028632, 0.0031304
4: -0.0016165, 0.0018977, -0.0015207, 0.0020675, -0.0033889, 0.0030995
5: 0.0023175, 0.0056431, 0.0021568, 0.0055525, -0.0029332, 0.0032070
6: -0.0131053, 0.0000898, -0.0137426, -0.0002698, -0.0116381, 0.0127244
7: -0.0026790, 0.0152915, -0.0021892, 0.0161596, -0.0173296, 0.0158501
8: 0.9873267, 0.9999855, 0.9876717, 1.0005970, -0.0122073, 0.0111651
9: -0.0158742, -0.0043833, -0.0164292, -0.0046965, -0.0101350, 0.0110810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0069255
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0069252
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045162, 0.0100161, 0.0043029, 0.0101358, -0.0052567, 0.0052990
1: 0.0019748, 0.0027693, 0.0019439, 0.0027866, -0.0007594, 0.0007656
2: 0.0088222, 0.0118630, 0.0087561, 0.0119809, -0.0029297, 0.0029063
3: -0.0055561, -0.0024112, -0.0056245, -0.0022892, -0.0030300, 0.0030058
4: -0.0014267, 0.0019778, -0.0015587, 0.0020519, -0.0032540, 0.0032802
5: 0.0022416, 0.0054635, 0.0021715, 0.0055884, -0.0031042, 0.0030794
6: -0.0134061, -0.0006230, -0.0136843, -0.0001270, -0.0123164, 0.0122181
7: -0.0017083, 0.0157013, -0.0023837, 0.0160801, -0.0166400, 0.0167739
8: 0.9880105, 1.0002742, 0.9875348, 1.0005410, -0.0117216, 0.0118159
9: -0.0161362, -0.0050040, -0.0163784, -0.0045721, -0.0107257, 0.0106401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0069281
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0069279
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044002, 0.0100261, 0.0040243, 0.0100080, -0.0053403, 0.0054213
1: 0.0019580, 0.0027708, 0.0019037, 0.0027682, -0.0007715, 0.0007832
2: 0.0088167, 0.0119271, 0.0088267, 0.0121349, -0.0029973, 0.0029525
3: -0.0055618, -0.0023449, -0.0055515, -0.0021299, -0.0031000, 0.0030536
4: -0.0014985, 0.0019840, -0.0017312, 0.0019729, -0.0033057, 0.0033559
5: 0.0022358, 0.0055314, 0.0022464, 0.0057516, -0.0031758, 0.0031283
6: -0.0134294, -0.0003534, -0.0133874, 0.0005204, -0.0126007, 0.0124122
7: -0.0020755, 0.0157330, -0.0032654, 0.0156758, -0.0169044, 0.0171611
8: 0.9877518, 1.0002965, 0.9869136, 1.0002563, -0.0119078, 0.0120886
9: -0.0161565, -0.0047692, -0.0161199, -0.0040083, -0.0109732, 0.0108091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0072918
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0072968
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041792, 0.0098884, 0.0043029, 0.0101358, -0.0054170, 0.0050162
1: 0.0019261, 0.0027509, 0.0019439, 0.0027866, -0.0007826, 0.0007247
2: 0.0088928, 0.0120493, 0.0087561, 0.0119809, -0.0027733, 0.0029949
3: -0.0054831, -0.0022185, -0.0056245, -0.0022892, -0.0028683, 0.0030975
4: -0.0016353, 0.0018988, -0.0015587, 0.0020519, -0.0033532, 0.0031051
5: 0.0023164, 0.0056609, 0.0021715, 0.0055884, -0.0029385, 0.0031733
6: -0.0131094, 0.0001604, -0.0136843, -0.0001270, -0.0116591, 0.0125907
7: -0.0027752, 0.0152971, -0.0023837, 0.0160801, -0.0171475, 0.0158787
8: 0.9872590, 0.9999894, 0.9875348, 1.0005410, -0.0120790, 0.0111853
9: -0.0158777, -0.0043218, -0.0163784, -0.0045721, -0.0101533, 0.0109645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0069511
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0069511
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0040495, 0.0098977, 0.0040243, 0.0100080, -0.0054550, 0.0051483
1: 0.0019073, 0.0027522, 0.0019037, 0.0027682, -0.0007881, 0.0007438
2: 0.0088877, 0.0121210, 0.0088267, 0.0121349, -0.0028464, 0.0030159
3: -0.0054884, -0.0021444, -0.0055515, -0.0021299, -0.0029438, 0.0031192
4: -0.0017156, 0.0019046, -0.0017312, 0.0019729, -0.0033767, 0.0031869
5: 0.0023110, 0.0057368, 0.0022464, 0.0057516, -0.0030159, 0.0031955
6: -0.0131311, 0.0004618, -0.0133874, 0.0005204, -0.0119661, 0.0126789
7: -0.0031856, 0.0153267, -0.0032654, 0.0156758, -0.0172676, 0.0162968
8: 0.9869699, 1.0000104, 0.9869136, 1.0002563, -0.0121637, 0.0114798
9: -0.0158967, -0.0040594, -0.0161199, -0.0040083, -0.0104206, 0.0110414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0072181
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0072251
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0043171, 0.0100194, 0.0047700, 0.0100312, -0.0052205, 0.0048117
1: 0.0019460, 0.0027698, 0.0020114, 0.0027715, -0.0007542, 0.0006952
2: 0.0088204, 0.0119730, 0.0088139, 0.0117226, -0.0026603, 0.0028863
3: -0.0055580, -0.0022974, -0.0055647, -0.0025563, -0.0027514, 0.0029851
4: -0.0015499, 0.0019799, -0.0012696, 0.0019872, -0.0032316, 0.0029785
5: 0.0022397, 0.0055801, 0.0022328, 0.0053148, -0.0028187, 0.0030582
6: -0.0134138, -0.0001602, -0.0134412, -0.0012128, -0.0111838, 0.0121339
7: -0.0023386, 0.0157118, -0.0009050, 0.0157490, -0.0165253, 0.0152314
8: 0.9875665, 1.0002816, 0.9885765, 1.0003078, -0.0116408, 0.0107293
9: -0.0161429, -0.0046010, -0.0161667, -0.0055177, -0.0097394, 0.0105668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066891, upper bound: 0.0065789
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068743, upper bound: 0.0065759
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0043171, 0.0100194, 0.0045159, 0.0100161, -0.0052650, 0.0051371
1: 0.0019460, 0.0027698, 0.0019747, 0.0027693, -0.0007606, 0.0007422
2: 0.0088204, 0.0119730, 0.0088222, 0.0118631, -0.0028402, 0.0029109
3: -0.0055580, -0.0022974, -0.0055561, -0.0024111, -0.0029375, 0.0030106
4: -0.0015499, 0.0019799, -0.0014268, 0.0019778, -0.0032591, 0.0031800
5: 0.0022397, 0.0055801, 0.0022416, 0.0054636, -0.0030093, 0.0030842
6: -0.0134138, -0.0001602, -0.0134061, -0.0006223, -0.0119401, 0.0122373
7: -0.0023386, 0.0157118, -0.0017092, 0.0157013, -0.0166662, 0.0162614
8: 0.9875665, 1.0002816, 0.9880099, 1.0002742, -0.0117400, 0.0114549
9: -0.0161429, -0.0046010, -0.0161362, -0.0050034, -0.0103980, 0.0106568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066891, upper bound: 0.0065789
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068743, upper bound: 0.0065759
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040684, 0.0098877, 0.0046530, 0.0100422, -0.0053419, 0.0048823
1: 0.0019101, 0.0027508, 0.0019945, 0.0027731, -0.0007718, 0.0007053
2: 0.0088932, 0.0121106, 0.0088078, 0.0117873, -0.0026993, 0.0029534
3: -0.0054827, -0.0021551, -0.0055710, -0.0024894, -0.0027917, 0.0030546
4: -0.0017039, 0.0018984, -0.0013420, 0.0019940, -0.0033067, 0.0030222
5: 0.0023169, 0.0057258, 0.0022264, 0.0053833, -0.0028600, 0.0031293
6: -0.0131077, 0.0004180, -0.0134668, -0.0009408, -0.0113478, 0.0124161
7: -0.0031260, 0.0152948, -0.0012754, 0.0157839, -0.0169096, 0.0154547
8: 0.9870118, 0.9999878, 0.9883154, 1.0003324, -0.0119115, 0.0108866
9: -0.0158763, -0.0040975, -0.0161890, -0.0052808, -0.0098822, 0.0108125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067698, upper bound: 0.0066081
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069756, upper bound: 0.0066080
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0040684, 0.0098877, 0.0043999, 0.0100261, -0.0053782, 0.0051986
1: 0.0019101, 0.0027508, 0.0019580, 0.0027708, -0.0007770, 0.0007510
2: 0.0088932, 0.0121106, 0.0088167, 0.0119273, -0.0028741, 0.0029735
3: -0.0054827, -0.0021551, -0.0055618, -0.0023447, -0.0029726, 0.0030753
4: -0.0017039, 0.0018984, -0.0014987, 0.0019840, -0.0033292, 0.0032180
5: 0.0023169, 0.0057258, 0.0022358, 0.0055316, -0.0030453, 0.0031505
6: -0.0131077, 0.0004180, -0.0134294, -0.0003526, -0.0120829, 0.0125004
7: -0.0031260, 0.0152948, -0.0020765, 0.0157330, -0.0170245, 0.0164558
8: 0.9870118, 0.9999878, 0.9877512, 1.0002965, -0.0119924, 0.0115918
9: -0.0158763, -0.0040975, -0.0161565, -0.0047686, -0.0105223, 0.0108859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067698, upper bound: 0.0066081
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069756, upper bound: 0.0066080
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0043681, 0.0100513, 0.0046450, 0.0102201, -0.0053718, 0.0049927
1: 0.0019534, 0.0027744, 0.0019934, 0.0027988, -0.0007761, 0.0007213
2: 0.0088028, 0.0119449, 0.0087095, 0.0117918, -0.0027604, 0.0029699
3: -0.0055762, -0.0023265, -0.0056727, -0.0024848, -0.0028549, 0.0030716
4: -0.0015184, 0.0019996, -0.0013470, 0.0021041, -0.0033252, 0.0030906
5: 0.0022210, 0.0055502, 0.0021221, 0.0053880, -0.0029247, 0.0031468
6: -0.0134879, -0.0002786, -0.0138803, -0.0009222, -0.0116045, 0.0124854
7: -0.0021773, 0.0158126, -0.0013008, 0.0163470, -0.0170041, 0.0158043
8: 0.9876801, 1.0003526, 0.9882976, 1.0007291, -0.0119780, 0.0111329
9: -0.0162074, -0.0047041, -0.0165491, -0.0052646, -0.0101057, 0.0108729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067936, upper bound: 0.0066877
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069075, upper bound: 0.0066875
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0042498, 0.0100614, 0.0043851, 0.0100699, -0.0054762, 0.0051633
1: 0.0019363, 0.0027759, 0.0019558, 0.0027771, -0.0007911, 0.0007459
2: 0.0087972, 0.0120103, 0.0087925, 0.0119354, -0.0028546, 0.0030276
3: -0.0055820, -0.0022589, -0.0055869, -0.0023363, -0.0029524, 0.0031313
4: -0.0015916, 0.0020059, -0.0015078, 0.0020111, -0.0033898, 0.0031961
5: 0.0022151, 0.0056195, 0.0022101, 0.0055403, -0.0030246, 0.0032079
6: -0.0135114, -0.0000036, -0.0135312, -0.0003182, -0.0120008, 0.0127281
7: -0.0025518, 0.0158446, -0.0021233, 0.0158716, -0.0173346, 0.0163441
8: 0.9874163, 1.0003752, 0.9877181, 1.0003941, -0.0122109, 0.0115131
9: -0.0162278, -0.0044647, -0.0162451, -0.0047386, -0.0104509, 0.0110842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0070092
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0070169
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040623, 0.0098884, 0.0046450, 0.0102201, -0.0054795, 0.0046281
1: 0.0019092, 0.0027509, 0.0019934, 0.0027988, -0.0007916, 0.0006686
2: 0.0088928, 0.0121139, 0.0087095, 0.0117918, -0.0025588, 0.0030295
3: -0.0054831, -0.0021516, -0.0056727, -0.0024848, -0.0026464, 0.0031332
4: -0.0017077, 0.0018988, -0.0013470, 0.0021041, -0.0033919, 0.0028649
5: 0.0023164, 0.0057294, 0.0021221, 0.0053880, -0.0027111, 0.0032099
6: -0.0131094, 0.0004322, -0.0138803, -0.0009222, -0.0107570, 0.0127358
7: -0.0031453, 0.0152972, -0.0013008, 0.0163470, -0.0173450, 0.0146501
8: 0.9869982, 0.9999895, 0.9882976, 1.0007291, -0.0122182, 0.0103199
9: -0.0158778, -0.0040852, -0.0165491, -0.0052646, -0.0093677, 0.0110909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068741, upper bound: 0.0066492
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070568, upper bound: 0.0066492
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0039216, 0.0098982, 0.0043851, 0.0100699, -0.0055669, 0.0048043
1: 0.0018889, 0.0027523, 0.0019558, 0.0027771, -0.0008043, 0.0006941
2: 0.0088874, 0.0121917, 0.0087925, 0.0119354, -0.0026562, 0.0030778
3: -0.0054887, -0.0020712, -0.0055869, -0.0023363, -0.0027471, 0.0031832
4: -0.0017948, 0.0019049, -0.0015078, 0.0020111, -0.0034460, 0.0029739
5: 0.0023107, 0.0058118, 0.0022101, 0.0055403, -0.0028143, 0.0032611
6: -0.0131322, 0.0007592, -0.0135312, -0.0003182, -0.0111664, 0.0129391
7: -0.0035907, 0.0153282, -0.0021233, 0.0158716, -0.0176219, 0.0152077
8: 0.9866845, 1.0000113, 0.9877181, 1.0003941, -0.0124132, 0.0107126
9: -0.0158976, -0.0038004, -0.0162451, -0.0047386, -0.0097242, 0.0112679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0068910
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0069005
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041112, 0.0099930, 0.0049192, 0.0101402, -0.0056102, 0.0046532
1: 0.0019162, 0.0027660, 0.0020330, 0.0027873, -0.0008105, 0.0006723
2: 0.0088350, 0.0120869, 0.0087536, 0.0116402, -0.0025726, 0.0031017
3: -0.0055429, -0.0021796, -0.0056271, -0.0026417, -0.0026608, 0.0032080
4: -0.0016774, 0.0019635, -0.0011772, 0.0020547, -0.0034728, 0.0028804
5: 0.0022552, 0.0057007, 0.0021689, 0.0052274, -0.0027259, 0.0032864
6: -0.0133524, 0.0003185, -0.0136947, -0.0015596, -0.0108154, 0.0130396
7: -0.0029905, 0.0156281, -0.0004327, 0.0160942, -0.0177589, 0.0147296
8: 0.9871073, 1.0002227, 0.9889090, 1.0005510, -0.0125097, 0.0103759
9: -0.0160894, -0.0041841, -0.0163874, -0.0058197, -0.0094185, 0.0113555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067683, upper bound: 0.0065914
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067683, upper bound: 0.0065914
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041621, 0.0099871, 0.0049076, 0.0104256, -0.0058817, 0.0046884
1: 0.0019236, 0.0027651, 0.0020313, 0.0028285, -0.0008497, 0.0006773
2: 0.0088383, 0.0120587, 0.0085958, 0.0116466, -0.0025921, 0.0032518
3: -0.0055395, -0.0022087, -0.0057903, -0.0026350, -0.0026809, 0.0033632
4: -0.0016459, 0.0019599, -0.0011844, 0.0022313, -0.0036408, 0.0029022
5: 0.0022586, 0.0056709, 0.0020017, 0.0052342, -0.0027465, 0.0034455
6: -0.0133387, 0.0002001, -0.0143580, -0.0015326, -0.0108971, 0.0136706
7: -0.0028293, 0.0156095, -0.0004694, 0.0169977, -0.0186182, 0.0148409
8: 0.9872209, 1.0002096, 0.9888832, 1.0011873, -0.0131150, 0.0104543
9: -0.0160775, -0.0042872, -0.0169651, -0.0057962, -0.0094897, 0.0119050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068762, upper bound: 0.0066460
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068762, upper bound: 0.0066460
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038515, 0.0098665, 0.0049511, 0.0103640, -0.0060531, 0.0045392
1: 0.0018787, 0.0027477, 0.0020376, 0.0028196, -0.0008745, 0.0006558
2: 0.0089050, 0.0122305, 0.0086299, 0.0116225, -0.0025096, 0.0033466
3: -0.0054705, -0.0020312, -0.0057550, -0.0026599, -0.0025956, 0.0034612
4: -0.0018381, 0.0018852, -0.0011575, 0.0021932, -0.0037470, 0.0028099
5: 0.0023293, 0.0058528, 0.0020378, 0.0052087, -0.0026591, 0.0035459
6: -0.0130584, 0.0009220, -0.0142148, -0.0016337, -0.0105504, 0.0140691
7: -0.0038124, 0.0152276, -0.0003318, 0.0168026, -0.0191608, 0.0143687
8: 0.9865284, 0.9999405, 0.9889801, 1.0010500, -0.0134973, 0.0101216
9: -0.0158333, -0.0036586, -0.0168404, -0.0058842, -0.0091878, 0.0122520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068551, upper bound: 0.0066374
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070260, upper bound: 0.0066290
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0038515, 0.0098665, 0.0047616, 0.0101512, -0.0057220, 0.0046003
1: 0.0018787, 0.0027477, 0.0020102, 0.0027889, -0.0008267, 0.0006646
2: 0.0089050, 0.0122305, 0.0087475, 0.0117273, -0.0025434, 0.0031636
3: -0.0054705, -0.0020312, -0.0056334, -0.0025516, -0.0026305, 0.0032719
4: -0.0018381, 0.0018852, -0.0012747, 0.0020615, -0.0035420, 0.0028477
5: 0.0023293, 0.0058528, 0.0021625, 0.0053197, -0.0026948, 0.0033520
6: -0.0130584, 0.0009220, -0.0137202, -0.0011934, -0.0106924, 0.0132996
7: -0.0038124, 0.0152276, -0.0009314, 0.0161289, -0.0181129, 0.0145621
8: 0.9865284, 0.9999405, 0.9885578, 1.0005754, -0.0127591, 0.0102578
9: -0.0158333, -0.0036586, -0.0164096, -0.0055008, -0.0093114, 0.0115819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068551, upper bound: 0.0066846
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070260, upper bound: 0.0066809
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041466, 0.0100152, 0.0046582, 0.0101664, -0.0056104, 0.0049823
1: 0.0019214, 0.0027692, 0.0019953, 0.0027911, -0.0008105, 0.0007198
2: 0.0088227, 0.0120673, 0.0087391, 0.0117845, -0.0027546, 0.0031018
3: -0.0055556, -0.0021999, -0.0056420, -0.0024924, -0.0028489, 0.0032081
4: -0.0016555, 0.0019773, -0.0013388, 0.0020709, -0.0034729, 0.0030841
5: 0.0022421, 0.0056800, 0.0021536, 0.0053803, -0.0029186, 0.0032865
6: -0.0134042, 0.0002363, -0.0137555, -0.0009529, -0.0115802, 0.0130400
7: -0.0028784, 0.0156986, -0.0012589, 0.0161771, -0.0177594, 0.0157713
8: 0.9871861, 1.0002724, 0.9883271, 1.0006094, -0.0125101, 0.0111096
9: -0.0161345, -0.0042558, -0.0164404, -0.0052913, -0.0100846, 0.0113558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067936, upper bound: 0.0067461
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069075, upper bound: 0.0067430
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0040319, 0.0100245, 0.0043979, 0.0100195, -0.0057180, 0.0051510
1: 0.0019048, 0.0027705, 0.0019577, 0.0027698, -0.0008261, 0.0007442
2: 0.0088176, 0.0121308, 0.0088203, 0.0119284, -0.0028479, 0.0031614
3: -0.0055609, -0.0021343, -0.0055581, -0.0023436, -0.0029454, 0.0032696
4: -0.0017265, 0.0019830, -0.0014999, 0.0019800, -0.0035396, 0.0031886
5: 0.0022367, 0.0057472, 0.0022396, 0.0055328, -0.0030175, 0.0033496
6: -0.0134256, 0.0005028, -0.0134141, -0.0003480, -0.0119724, 0.0132903
7: -0.0032415, 0.0157278, -0.0020828, 0.0157121, -0.0181003, 0.0163054
8: 0.9869305, 1.0002929, 0.9877468, 1.0002818, -0.0127502, 0.0114858
9: -0.0161531, -0.0040236, -0.0161431, -0.0047646, -0.0104261, 0.0115738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0071290
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0071356
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038400, 0.0098675, 0.0046582, 0.0101664, -0.0057031, 0.0046267
1: 0.0018771, 0.0027479, 0.0019953, 0.0027911, -0.0008239, 0.0006684
2: 0.0089044, 0.0122368, 0.0087391, 0.0117845, -0.0025580, 0.0031531
3: -0.0054711, -0.0020246, -0.0056420, -0.0024924, -0.0026456, 0.0032611
4: -0.0018452, 0.0018859, -0.0013388, 0.0020709, -0.0035303, 0.0028640
5: 0.0023287, 0.0058596, 0.0021536, 0.0053803, -0.0027103, 0.0033409
6: -0.0130608, 0.0009488, -0.0137555, -0.0009529, -0.0107536, 0.0132556
7: -0.0038488, 0.0152310, -0.0012589, 0.0161771, -0.0180529, 0.0146455
8: 0.9865027, 0.9999429, 0.9883271, 1.0006094, -0.0127169, 0.0103166
9: -0.0158355, -0.0036353, -0.0164404, -0.0052913, -0.0093647, 0.0115435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068741, upper bound: 0.0067343
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070568, upper bound: 0.0067343
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036998, 0.0098765, 0.0043979, 0.0100195, -0.0057949, 0.0047934
1: 0.0018568, 0.0027492, 0.0019577, 0.0027698, -0.0008372, 0.0006925
2: 0.0088994, 0.0123143, 0.0088203, 0.0119284, -0.0026501, 0.0032039
3: -0.0054763, -0.0019444, -0.0055581, -0.0023436, -0.0027409, 0.0033136
4: -0.0019320, 0.0018914, -0.0014999, 0.0019800, -0.0035872, 0.0029672
5: 0.0023234, 0.0059417, 0.0022396, 0.0055328, -0.0028080, 0.0033947
6: -0.0130817, 0.0012746, -0.0134141, -0.0003480, -0.0111412, 0.0134690
7: -0.0042927, 0.0152594, -0.0020828, 0.0157121, -0.0183436, 0.0151733
8: 0.9861900, 0.9999629, 0.9877468, 1.0002818, -0.0129216, 0.0106884
9: -0.0158536, -0.0033515, -0.0161431, -0.0047646, -0.0097022, 0.0117294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0070144
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0070245
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0043171, 0.0100194, 0.0043681, 0.0100513, -0.0047792, 0.0047234
1: 0.0019460, 0.0027698, 0.0019534, 0.0027744, -0.0006905, 0.0006824
2: 0.0088204, 0.0119730, 0.0088028, 0.0119449, -0.0026115, 0.0026423
3: -0.0055580, -0.0022974, -0.0055762, -0.0023265, -0.0027009, 0.0027328
4: -0.0015499, 0.0019799, -0.0015184, 0.0019996, -0.0029584, 0.0029239
5: 0.0022397, 0.0055801, 0.0022210, 0.0055502, -0.0027670, 0.0027996
6: -0.0134138, -0.0001602, -0.0134879, -0.0002786, -0.0109786, 0.0111082
7: -0.0023386, 0.0157118, -0.0021773, 0.0158126, -0.0151284, 0.0149519
8: 0.9875665, 1.0002816, 0.9876801, 1.0003526, -0.0106567, 0.0105324
9: -0.0161429, -0.0046010, -0.0162074, -0.0047041, -0.0095606, 0.0096735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066970, upper bound: 0.0066311
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068905, upper bound: 0.0066311
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0043171, 0.0100194, 0.0041466, 0.0100152, -0.0048224, 0.0050392
1: 0.0019460, 0.0027698, 0.0019214, 0.0027692, -0.0006967, 0.0007280
2: 0.0088204, 0.0119730, 0.0088227, 0.0120673, -0.0027860, 0.0026662
3: -0.0055580, -0.0022974, -0.0055556, -0.0021999, -0.0028814, 0.0027575
4: -0.0015499, 0.0019799, -0.0016555, 0.0019773, -0.0029852, 0.0031193
5: 0.0022397, 0.0055801, 0.0022421, 0.0056800, -0.0029519, 0.0028250
6: -0.0134138, -0.0001602, -0.0134042, 0.0002363, -0.0117125, 0.0112086
7: -0.0023386, 0.0157118, -0.0028784, 0.0156986, -0.0152652, 0.0159513
8: 0.9875665, 1.0002816, 0.9871861, 1.0002724, -0.0107531, 0.0112365
9: -0.0161429, -0.0046010, -0.0161345, -0.0042558, -0.0101997, 0.0097610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066970, upper bound: 0.0066311
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068905, upper bound: 0.0066311
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040684, 0.0098877, 0.0042498, 0.0100614, -0.0049084, 0.0048223
1: 0.0019101, 0.0027508, 0.0019363, 0.0027759, -0.0007091, 0.0006967
2: 0.0088932, 0.0121106, 0.0087972, 0.0120103, -0.0026661, 0.0027137
3: -0.0054827, -0.0021551, -0.0055820, -0.0022589, -0.0027574, 0.0028067
4: -0.0017039, 0.0018984, -0.0015916, 0.0020059, -0.0030384, 0.0029851
5: 0.0023169, 0.0057258, 0.0022151, 0.0056195, -0.0028249, 0.0028754
6: -0.0131077, 0.0004180, -0.0135114, -0.0000036, -0.0112084, 0.0114086
7: -0.0031260, 0.0152948, -0.0025518, 0.0158446, -0.0155375, 0.0152649
8: 0.9870118, 0.9999878, 0.9874163, 1.0003752, -0.0109449, 0.0107529
9: -0.0158763, -0.0040975, -0.0162278, -0.0044647, -0.0097608, 0.0099351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067772, upper bound: 0.0066441
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069822, upper bound: 0.0066441
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0040684, 0.0098877, 0.0040319, 0.0100245, -0.0049320, 0.0051322
1: 0.0019101, 0.0027508, 0.0019048, 0.0027705, -0.0007125, 0.0007415
2: 0.0088932, 0.0121106, 0.0088176, 0.0121308, -0.0028375, 0.0027268
3: -0.0054827, -0.0021551, -0.0055609, -0.0021343, -0.0029346, 0.0028202
4: -0.0017039, 0.0018984, -0.0017265, 0.0019830, -0.0030530, 0.0031769
5: 0.0023169, 0.0057258, 0.0022367, 0.0057472, -0.0030064, 0.0028892
6: -0.0131077, 0.0004180, -0.0134256, 0.0005028, -0.0119286, 0.0114633
7: -0.0031260, 0.0152948, -0.0032415, 0.0157278, -0.0156121, 0.0162457
8: 0.9870118, 0.9999878, 0.9869305, 1.0002929, -0.0109975, 0.0114438
9: -0.0158763, -0.0040975, -0.0161531, -0.0040236, -0.0103880, 0.0099828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067772, upper bound: 0.0066441
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069822, upper bound: 0.0066441
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0043681, 0.0100513, 0.0042915, 0.0101888, -0.0048993, 0.0048773
1: 0.0019534, 0.0027744, 0.0019423, 0.0027943, -0.0007078, 0.0007046
2: 0.0088028, 0.0119449, 0.0087267, 0.0119872, -0.0026966, 0.0027087
3: -0.0055762, -0.0023265, -0.0056549, -0.0022828, -0.0027889, 0.0028015
4: -0.0015184, 0.0019996, -0.0015657, 0.0020848, -0.0030328, 0.0030192
5: 0.0022210, 0.0055502, 0.0021405, 0.0055951, -0.0028571, 0.0028700
6: -0.0134879, -0.0002786, -0.0138076, -0.0001007, -0.0113363, 0.0113874
7: -0.0021773, 0.0158126, -0.0024195, 0.0162481, -0.0155086, 0.0154391
8: 0.9876801, 1.0003526, 0.9875094, 1.0006593, -0.0109246, 0.0108756
9: -0.0162074, -0.0047041, -0.0164858, -0.0045493, -0.0098722, 0.0099166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068161, upper bound: 0.0067050
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069333, upper bound: 0.0067050
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0042498, 0.0100614, 0.0040108, 0.0100606, -0.0050040, 0.0050194
1: 0.0019363, 0.0027759, 0.0019017, 0.0027758, -0.0007229, 0.0007252
2: 0.0087972, 0.0120103, 0.0087976, 0.0121424, -0.0027751, 0.0027666
3: -0.0055820, -0.0022589, -0.0055815, -0.0021222, -0.0028701, 0.0028613
4: -0.0015916, 0.0020059, -0.0017395, 0.0020054, -0.0030975, 0.0031071
5: 0.0022151, 0.0056195, 0.0022156, 0.0057595, -0.0029404, 0.0029313
6: -0.0135114, -0.0000036, -0.0135095, 0.0005517, -0.0116665, 0.0116306
7: -0.0025518, 0.0158446, -0.0033081, 0.0158421, -0.0158399, 0.0158887
8: 0.9874163, 1.0003752, 0.9868835, 1.0003735, -0.0111580, 0.0111924
9: -0.0162278, -0.0044647, -0.0162262, -0.0039810, -0.0101597, 0.0101285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0069949
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0069982
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040623, 0.0098884, 0.0042915, 0.0101888, -0.0050130, 0.0045167
1: 0.0019092, 0.0027509, 0.0019423, 0.0027943, -0.0007242, 0.0006525
2: 0.0088928, 0.0121139, 0.0087267, 0.0119872, -0.0024972, 0.0027716
3: -0.0054831, -0.0021516, -0.0056549, -0.0022828, -0.0025827, 0.0028665
4: -0.0017077, 0.0018988, -0.0015657, 0.0020848, -0.0031031, 0.0027959
5: 0.0023164, 0.0057294, 0.0021405, 0.0055951, -0.0026459, 0.0029366
6: -0.0131094, 0.0004322, -0.0138076, -0.0001007, -0.0104981, 0.0116517
7: -0.0031453, 0.0152972, -0.0024195, 0.0162481, -0.0158686, 0.0142975
8: 0.9869982, 0.9999895, 0.9875094, 1.0006593, -0.0111781, 0.0100715
9: -0.0158778, -0.0040852, -0.0164858, -0.0045493, -0.0091422, 0.0101468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068768, upper bound: 0.0066628
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070599, upper bound: 0.0066628
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0039216, 0.0098982, 0.0040108, 0.0100606, -0.0050998, 0.0046751
1: 0.0018889, 0.0027523, 0.0019017, 0.0027758, -0.0007368, 0.0006754
2: 0.0088874, 0.0121917, 0.0087976, 0.0121424, -0.0025847, 0.0028195
3: -0.0054887, -0.0020712, -0.0055815, -0.0021222, -0.0026733, 0.0029161
4: -0.0017948, 0.0019049, -0.0017395, 0.0020054, -0.0031568, 0.0028940
5: 0.0023107, 0.0058118, 0.0022156, 0.0057595, -0.0027387, 0.0029874
6: -0.0131322, 0.0007592, -0.0135095, 0.0005517, -0.0108662, 0.0118532
7: -0.0035907, 0.0153282, -0.0033081, 0.0158421, -0.0161431, 0.0147988
8: 0.9866845, 1.0000113, 0.9868835, 1.0003735, -0.0113715, 0.0104246
9: -0.0158976, -0.0038004, -0.0162262, -0.0039810, -0.0094628, 0.0103223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0068861
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0068907
time: 1.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041112, 0.0099930, 0.0045246, 0.0101502, -0.0052154, 0.0045645
1: 0.0019162, 0.0027660, 0.0019760, 0.0027887, -0.0007535, 0.0006594
2: 0.0088350, 0.0120869, 0.0087481, 0.0118583, -0.0025236, 0.0028834
3: -0.0055429, -0.0021796, -0.0056328, -0.0024160, -0.0026100, 0.0029822
4: -0.0016774, 0.0019635, -0.0014215, 0.0020609, -0.0032284, 0.0028255
5: 0.0022552, 0.0057007, 0.0021631, 0.0054585, -0.0026739, 0.0030552
6: -0.0133524, 0.0003185, -0.0137178, -0.0006424, -0.0106091, 0.0121220
7: -0.0029905, 0.0156281, -0.0016818, 0.0161258, -0.0165091, 0.0144487
8: 0.9871073, 1.0002227, 0.9880292, 1.0005732, -0.0116294, 0.0101779
9: -0.0160894, -0.0041841, -0.0164076, -0.0050210, -0.0092389, 0.0105564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067863, upper bound: 0.0066404
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067863, upper bound: 0.0066404
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041621, 0.0099871, 0.0045096, 0.0104413, -0.0054962, 0.0046143
1: 0.0019236, 0.0027651, 0.0019738, 0.0028308, -0.0007940, 0.0006666
2: 0.0088383, 0.0120587, 0.0085871, 0.0118666, -0.0025511, 0.0030387
3: -0.0055395, -0.0022087, -0.0057993, -0.0024075, -0.0026385, 0.0031428
4: -0.0016459, 0.0019599, -0.0014307, 0.0022411, -0.0034022, 0.0028563
5: 0.0022586, 0.0056709, 0.0019925, 0.0054673, -0.0027031, 0.0032197
6: -0.0133387, 0.0002001, -0.0143945, -0.0006076, -0.0107250, 0.0127747
7: -0.0028293, 0.0156095, -0.0017292, 0.0170474, -0.0173980, 0.0146065
8: 0.9872209, 1.0002096, 0.9879958, 1.0012225, -0.0122555, 0.0102891
9: -0.0160775, -0.0042872, -0.0169969, -0.0049907, -0.0093398, 0.0111248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069081, upper bound: 0.0067118
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069081, upper bound: 0.0067118
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038515, 0.0098665, 0.0045499, 0.0103846, -0.0056697, 0.0044749
1: 0.0018787, 0.0027477, 0.0019796, 0.0028226, -0.0008191, 0.0006465
2: 0.0089050, 0.0122305, 0.0086185, 0.0118444, -0.0024741, 0.0031347
3: -0.0054705, -0.0020312, -0.0057668, -0.0024305, -0.0025588, 0.0032420
4: -0.0018381, 0.0018852, -0.0014058, 0.0022059, -0.0035097, 0.0027700
5: 0.0023293, 0.0058528, 0.0020258, 0.0054437, -0.0026214, 0.0033213
6: -0.0130584, 0.0009220, -0.0142626, -0.0007012, -0.0104009, 0.0131780
7: -0.0038124, 0.0152276, -0.0016018, 0.0168677, -0.0179473, 0.0141652
8: 0.9865284, 0.9999405, 0.9880856, 1.0010958, -0.0126425, 0.0099782
9: -0.0158333, -0.0036586, -0.0168820, -0.0050721, -0.0090576, 0.0114760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068659, upper bound: 0.0066988
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070473, upper bound: 0.0066988
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0038515, 0.0098665, 0.0043643, 0.0101609, -0.0053359, 0.0045167
1: 0.0018787, 0.0027477, 0.0019528, 0.0027902, -0.0007709, 0.0006525
2: 0.0089050, 0.0122305, 0.0087422, 0.0119470, -0.0024972, 0.0029501
3: -0.0054705, -0.0020312, -0.0056389, -0.0023244, -0.0025827, 0.0030511
4: -0.0018381, 0.0018852, -0.0015207, 0.0020675, -0.0033030, 0.0027959
5: 0.0023293, 0.0058528, 0.0021568, 0.0055525, -0.0026459, 0.0031258
6: -0.0130584, 0.0009220, -0.0137426, -0.0002698, -0.0104982, 0.0124021
7: -0.0038124, 0.0152276, -0.0021892, 0.0161596, -0.0168906, 0.0142976
8: 0.9865284, 0.9999405, 0.9876717, 1.0005970, -0.0118981, 0.0100715
9: -0.0158333, -0.0036586, -0.0164292, -0.0046965, -0.0091423, 0.0108003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068659, upper bound: 0.0067265
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070473, upper bound: 0.0067265
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041466, 0.0100152, 0.0043029, 0.0101358, -0.0051745, 0.0048850
1: 0.0019214, 0.0027692, 0.0019439, 0.0027866, -0.0007476, 0.0007057
2: 0.0088227, 0.0120673, 0.0087561, 0.0119809, -0.0027008, 0.0028609
3: -0.0055556, -0.0021999, -0.0056245, -0.0022892, -0.0027933, 0.0029588
4: -0.0016555, 0.0019773, -0.0015587, 0.0020519, -0.0032031, 0.0030239
5: 0.0022421, 0.0056800, 0.0021715, 0.0055884, -0.0028616, 0.0030312
6: -0.0134042, 0.0002363, -0.0136843, -0.0001270, -0.0113542, 0.0120270
7: -0.0028784, 0.0156986, -0.0023837, 0.0160801, -0.0163798, 0.0154634
8: 0.9871861, 1.0002724, 0.9875348, 1.0005410, -0.0115383, 0.0108927
9: -0.0161345, -0.0042558, -0.0163784, -0.0045721, -0.0098877, 0.0104737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068161, upper bound: 0.0067835
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069333, upper bound: 0.0067835
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0040319, 0.0100245, 0.0040243, 0.0100080, -0.0052876, 0.0050169
1: 0.0019048, 0.0027705, 0.0019037, 0.0027682, -0.0007639, 0.0007248
2: 0.0088176, 0.0121308, 0.0088267, 0.0121349, -0.0027737, 0.0029234
3: -0.0055609, -0.0021343, -0.0055515, -0.0021299, -0.0028687, 0.0030235
4: -0.0017265, 0.0019830, -0.0017312, 0.0019729, -0.0032731, 0.0031056
5: 0.0022367, 0.0057472, 0.0022464, 0.0057516, -0.0029389, 0.0030975
6: -0.0134256, 0.0005028, -0.0133874, 0.0005204, -0.0116608, 0.0122899
7: -0.0032415, 0.0157278, -0.0032654, 0.0156758, -0.0167378, 0.0158809
8: 0.9869305, 1.0002929, 0.9869136, 1.0002563, -0.0117905, 0.0111869
9: -0.0161531, -0.0040236, -0.0161199, -0.0040083, -0.0101547, 0.0107026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0071153
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0071200
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038400, 0.0098675, 0.0043029, 0.0101358, -0.0052774, 0.0045296
1: 0.0018771, 0.0027479, 0.0019439, 0.0027866, -0.0007624, 0.0006544
2: 0.0089044, 0.0122368, 0.0087561, 0.0119809, -0.0025043, 0.0029177
3: -0.0054711, -0.0020246, -0.0056245, -0.0022892, -0.0025901, 0.0030177
4: -0.0018452, 0.0018859, -0.0015587, 0.0020519, -0.0032668, 0.0028039
5: 0.0023287, 0.0058596, 0.0021715, 0.0055884, -0.0026534, 0.0030915
6: -0.0130608, 0.0009488, -0.0136843, -0.0001270, -0.0105281, 0.0122661
7: -0.0038488, 0.0152310, -0.0023837, 0.0160801, -0.0167054, 0.0143384
8: 0.9865027, 0.9999429, 0.9875348, 1.0005410, -0.0117676, 0.0101002
9: -0.0158355, -0.0036353, -0.0163784, -0.0045721, -0.0091683, 0.0106819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068768, upper bound: 0.0067538
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070599, upper bound: 0.0067538
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036998, 0.0098765, 0.0040243, 0.0100080, -0.0053661, 0.0046739
1: 0.0018568, 0.0027492, 0.0019037, 0.0027682, -0.0007753, 0.0006752
2: 0.0088994, 0.0123143, 0.0088267, 0.0121349, -0.0025841, 0.0029668
3: -0.0054763, -0.0019444, -0.0055515, -0.0021299, -0.0026726, 0.0030684
4: -0.0019320, 0.0018914, -0.0017312, 0.0019729, -0.0033217, 0.0028932
5: 0.0023234, 0.0059417, 0.0022464, 0.0057516, -0.0027380, 0.0031435
6: -0.0130817, 0.0012746, -0.0133874, 0.0005204, -0.0108635, 0.0124724
7: -0.0042927, 0.0152594, -0.0032654, 0.0156758, -0.0169863, 0.0147952
8: 0.9861900, 0.9999629, 0.9869136, 1.0002563, -0.0119655, 0.0104220
9: -0.0158536, -0.0033515, -0.0161199, -0.0040083, -0.0094604, 0.0108615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0070000
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0070093
time: 1.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.25 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0065789
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0065759
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0065789
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0065759
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0066081
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0066080
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0066081
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0066080
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0066877
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0066875
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0070092
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0070169
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0066492
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0066492
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0068910
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0069006
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0066460
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0066460
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066374
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066290
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066846
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066809
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0067461
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0067430
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071290
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071356
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0067343
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0067343
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0070144
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0070246
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0068571
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0068562
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0068571
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0068562
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0068808
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0068808
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0068808
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0068808
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0068752
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0068752
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071916
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071956
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0068886
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0068886
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0071290
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0071343
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0067936
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0067936
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0069075
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0069075
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0068792
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0068775
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0069255
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0069252
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0069281
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0069279
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0072918
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0072968
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0069511
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0069511
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0072181
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0072251
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066891, upper bound: 0.0065789
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068743, upper bound: 0.0065759
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066891, upper bound: 0.0065789
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068743, upper bound: 0.0065759
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067698, upper bound: 0.0066081
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069756, upper bound: 0.0066080
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067698, upper bound: 0.0066081
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069756, upper bound: 0.0066080
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067936, upper bound: 0.0066877
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069075, upper bound: 0.0066875
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0070092
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0070169
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068741, upper bound: 0.0066492
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070568, upper bound: 0.0066492
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0068910
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0069005
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067683, upper bound: 0.0065914
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067683, upper bound: 0.0065914
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068762, upper bound: 0.0066460
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068762, upper bound: 0.0066460
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068551, upper bound: 0.0066374
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070260, upper bound: 0.0066290
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068551, upper bound: 0.0066846
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070260, upper bound: 0.0066809
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067936, upper bound: 0.0067461
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069075, upper bound: 0.0067430
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0071290
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0071356
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068741, upper bound: 0.0067343
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070568, upper bound: 0.0067343
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0070144
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0070245
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066970, upper bound: 0.0066311
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068905, upper bound: 0.0066311
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0066970, upper bound: 0.0066311
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068905, upper bound: 0.0066311
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067772, upper bound: 0.0066441
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069822, upper bound: 0.0066441
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067772, upper bound: 0.0066441
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069822, upper bound: 0.0066441
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068161, upper bound: 0.0067050
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069333, upper bound: 0.0067050
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0069949
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0069982
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068768, upper bound: 0.0066628
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070599, upper bound: 0.0066628
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0068861
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0068907
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067863, upper bound: 0.0066404
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0067863, upper bound: 0.0066404
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069081, upper bound: 0.0067118
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069081, upper bound: 0.0067118
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068659, upper bound: 0.0066988
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070473, upper bound: 0.0066988
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068659, upper bound: 0.0067265
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070473, upper bound: 0.0067265
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068161, upper bound: 0.0067835
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0069333, upper bound: 0.0067835
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0071153
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0071200
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0068768, upper bound: 0.0067538
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0070599, upper bound: 0.0067538
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0070000
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0070093

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0048790, 0.0100412, 0.0047884, 0.0100301, -0.0044152, 0.0045207
1: 0.0020272, 0.0027730, 0.0020141, 0.0027714, -0.0006379, 0.0006531
2: 0.0088084, 0.0116624, 0.0088145, 0.0117125, -0.0024994, 0.0024411
3: -0.0055705, -0.0026187, -0.0055641, -0.0025669, -0.0025850, 0.0025247
4: -0.0012021, 0.0019934, -0.0012582, 0.0019865, -0.0027331, 0.0027984
5: 0.0022269, 0.0052509, 0.0022334, 0.0053040, -0.0026482, 0.0025864
6: -0.0134645, -0.0014661, -0.0134388, -0.0012555, -0.0105074, 0.0102622
7: -0.0005600, 0.0157808, -0.0008468, 0.0157458, -0.0139762, 0.0143102
8: 0.9888194, 1.0003302, 0.9886174, 1.0003055, -0.0098452, 0.0100804
9: -0.0161870, -0.0057383, -0.0161646, -0.0055549, -0.0091503, 0.0089368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065992, upper bound: 0.0065401
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065992, upper bound: 0.0066265
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0048609, 0.0103648, 0.0048399, 0.0100247, -0.0044600, 0.0048428
1: 0.0020246, 0.0028197, 0.0020215, 0.0027706, -0.0006443, 0.0006996
2: 0.0086294, 0.0116724, 0.0088174, 0.0116840, -0.0026774, 0.0024658
3: -0.0057555, -0.0026083, -0.0055610, -0.0025963, -0.0027691, 0.0025503
4: -0.0012133, 0.0021937, -0.0012263, 0.0019832, -0.0027608, 0.0029977
5: 0.0020374, 0.0052616, 0.0022366, 0.0052738, -0.0028369, 0.0026126
6: -0.0142167, -0.0014240, -0.0134263, -0.0013753, -0.0112559, 0.0103662
7: -0.0006174, 0.0168052, -0.0006836, 0.0157287, -0.0141179, 0.0153296
8: 0.9887789, 1.0010518, 0.9887324, 1.0002935, -0.0099449, 0.0107985
9: -0.0168420, -0.0057016, -0.0161537, -0.0056592, -0.0098021, 0.0090274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065293, upper bound: 0.0064897
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066079, upper bound: 0.0064897
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0048790, 0.0100412, 0.0045331, 0.0100150, -0.0044597, 0.0048476
1: 0.0020272, 0.0027730, 0.0019772, 0.0027692, -0.0006443, 0.0007003
2: 0.0088084, 0.0116624, 0.0088228, 0.0118536, -0.0026801, 0.0024656
3: -0.0055705, -0.0026187, -0.0055555, -0.0024209, -0.0027719, 0.0025501
4: -0.0012021, 0.0019934, -0.0014162, 0.0019772, -0.0027606, 0.0030008
5: 0.0022269, 0.0052509, 0.0022423, 0.0054536, -0.0028397, 0.0026125
6: -0.0134645, -0.0014661, -0.0134036, -0.0006622, -0.0112672, 0.0103655
7: -0.0005600, 0.0157808, -0.0016549, 0.0156978, -0.0141169, 0.0153450
8: 0.9888194, 1.0003302, 0.9880481, 1.0002718, -0.0099443, 0.0108093
9: -0.0161870, -0.0057383, -0.0161340, -0.0050382, -0.0098120, 0.0090268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0064913
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0065759
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0048609, 0.0103648, 0.0045911, 0.0100096, -0.0045047, 0.0051623
1: 0.0020246, 0.0028197, 0.0019856, 0.0027684, -0.0006508, 0.0007458
2: 0.0086294, 0.0116724, 0.0088258, 0.0118216, -0.0028541, 0.0024905
3: -0.0057555, -0.0026083, -0.0055524, -0.0024541, -0.0029518, 0.0025758
4: -0.0012133, 0.0021937, -0.0013803, 0.0019738, -0.0027885, 0.0031955
5: 0.0020374, 0.0052616, 0.0022455, 0.0054196, -0.0030241, 0.0026388
6: -0.0142167, -0.0014240, -0.0133910, -0.0007970, -0.0119986, 0.0104700
7: -0.0006174, 0.0168052, -0.0014712, 0.0156807, -0.0142593, 0.0163410
8: 0.9887789, 1.0010518, 0.9881775, 1.0002596, -0.0100445, 0.0115109
9: -0.0168420, -0.0057016, -0.0161230, -0.0051556, -0.0104489, 0.0091178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065450, upper bound: 0.0064273
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065953, upper bound: 0.0064273
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0046456, 0.0098834, 0.0046718, 0.0100412, -0.0045245, 0.0046256
1: 0.0019934, 0.0027502, 0.0019972, 0.0027730, -0.0006537, 0.0006683
2: 0.0088956, 0.0117915, 0.0088084, 0.0117770, -0.0025574, 0.0025015
3: -0.0054802, -0.0024852, -0.0055704, -0.0025002, -0.0026450, 0.0025871
4: -0.0013466, 0.0018957, -0.0013304, 0.0019934, -0.0028007, 0.0028633
5: 0.0023194, 0.0053877, 0.0022270, 0.0053723, -0.0027097, 0.0026504
6: -0.0130977, -0.0009235, -0.0134644, -0.0009845, -0.0107511, 0.0105161
7: -0.0012989, 0.0152812, -0.0012159, 0.0157807, -0.0143221, 0.0146421
8: 0.9882988, 0.9999782, 0.9883572, 1.0003301, -0.0100888, 0.0103142
9: -0.0158676, -0.0052658, -0.0161869, -0.0053189, -0.0093626, 0.0091579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066586, upper bound: 0.0066158
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066586, upper bound: 0.0066480
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0046197, 0.0102092, 0.0047227, 0.0100358, -0.0045757, 0.0049422
1: 0.0019897, 0.0027972, 0.0020046, 0.0027722, -0.0006611, 0.0007140
2: 0.0087154, 0.0118057, 0.0088113, 0.0117488, -0.0027324, 0.0025298
3: -0.0056665, -0.0024704, -0.0055674, -0.0025293, -0.0028260, 0.0026164
4: -0.0013626, 0.0020974, -0.0012989, 0.0019900, -0.0028324, 0.0030593
5: 0.0021285, 0.0054028, 0.0022301, 0.0053425, -0.0028951, 0.0026804
6: -0.0138551, -0.0008636, -0.0134519, -0.0011028, -0.0114871, 0.0106351
7: -0.0013806, 0.0163127, -0.0010548, 0.0157637, -0.0144841, 0.0156444
8: 0.9882413, 1.0007049, 0.9884709, 1.0003182, -0.0102029, 0.0110203
9: -0.0165271, -0.0052136, -0.0161761, -0.0054219, -0.0100035, 0.0092615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068104, upper bound: 0.0066158
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068104, upper bound: 0.0066480
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046456, 0.0098834, 0.0044171, 0.0100250, -0.0045607, 0.0049435
1: 0.0019934, 0.0027502, 0.0019604, 0.0027706, -0.0006589, 0.0007142
2: 0.0088956, 0.0117915, 0.0088173, 0.0119178, -0.0027331, 0.0025215
3: -0.0054802, -0.0024852, -0.0055612, -0.0023545, -0.0028267, 0.0026079
4: -0.0013466, 0.0018957, -0.0014880, 0.0019834, -0.0028232, 0.0030601
5: 0.0023194, 0.0053877, 0.0022364, 0.0055215, -0.0028959, 0.0026717
6: -0.0130977, -0.0009235, -0.0134269, -0.0003925, -0.0114901, 0.0106004
7: -0.0012989, 0.0152812, -0.0020222, 0.0157295, -0.0144368, 0.0156485
8: 0.9882988, 0.9999782, 0.9877895, 1.0002941, -0.0101696, 0.0110231
9: -0.0158676, -0.0052658, -0.0161542, -0.0048033, -0.0100060, 0.0092313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0065670
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066042
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0046197, 0.0102092, 0.0044766, 0.0100196, -0.0046122, 0.0052541
1: 0.0019897, 0.0027972, 0.0019690, 0.0027698, -0.0006663, 0.0007591
2: 0.0087154, 0.0118057, 0.0088203, 0.0118849, -0.0029048, 0.0025499
3: -0.0056665, -0.0024704, -0.0055581, -0.0023885, -0.0030043, 0.0026373
4: -0.0013626, 0.0020974, -0.0014512, 0.0019800, -0.0028550, 0.0032524
5: 0.0021285, 0.0054028, 0.0022396, 0.0054867, -0.0030778, 0.0027018
6: -0.0138551, -0.0008636, -0.0134144, -0.0005307, -0.0122119, 0.0107200
7: -0.0013806, 0.0163127, -0.0018339, 0.0157126, -0.0145996, 0.0166315
8: 0.9882413, 1.0007049, 0.9879221, 1.0002822, -0.0102843, 0.0117156
9: -0.0165271, -0.0052136, -0.0161434, -0.0049237, -0.0106347, 0.0093354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0065634
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066042
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0049427, 0.0100212, 0.0046643, 0.0102189, -0.0045508, 0.0046754
1: 0.0020364, 0.0027701, 0.0019962, 0.0027986, -0.0006575, 0.0006755
2: 0.0088194, 0.0116272, 0.0087101, 0.0117811, -0.0025849, 0.0025160
3: -0.0055590, -0.0026551, -0.0056721, -0.0024959, -0.0026734, 0.0026022
4: -0.0011627, 0.0019810, -0.0013350, 0.0021034, -0.0028170, 0.0028942
5: 0.0022386, 0.0052136, 0.0021228, 0.0053767, -0.0027388, 0.0026659
6: -0.0134181, -0.0016142, -0.0138776, -0.0009670, -0.0108669, 0.0105774
7: -0.0003583, 0.0157176, -0.0012397, 0.0163434, -0.0144055, 0.0147998
8: 0.9889615, 1.0002856, 0.9883406, 1.0007265, -0.0101475, 0.0104253
9: -0.0161466, -0.0058672, -0.0165468, -0.0053037, -0.0094634, 0.0092113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065032, upper bound: 0.0066877
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065032, upper bound: 0.0066877
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0049327, 0.0103330, 0.0047107, 0.0102128, -0.0045869, 0.0049891
1: 0.0020349, 0.0028151, 0.0020029, 0.0027978, -0.0006627, 0.0007208
2: 0.0086470, 0.0116327, 0.0087135, 0.0117555, -0.0027584, 0.0025360
3: -0.0057373, -0.0026494, -0.0056686, -0.0025224, -0.0028528, 0.0026228
4: -0.0011689, 0.0021740, -0.0013063, 0.0020996, -0.0028394, 0.0030883
5: 0.0020560, 0.0052195, 0.0021264, 0.0053495, -0.0029226, 0.0026870
6: -0.0141428, -0.0015909, -0.0138634, -0.0010749, -0.0115961, 0.0106613
7: -0.0003901, 0.0167045, -0.0010928, 0.0163240, -0.0145197, 0.0157929
8: 0.9889391, 1.0009809, 0.9884441, 1.0007129, -0.0102280, 0.0111248
9: -0.0167776, -0.0058469, -0.0165344, -0.0053976, -0.0100984, 0.0092843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065851, upper bound: 0.0066875
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065851, upper bound: 0.0066875
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0049776, 0.0102551, 0.0043851, 0.0100699, -0.0044623, 0.0051685
1: 0.0020414, 0.0028039, 0.0019558, 0.0027771, -0.0006447, 0.0007467
2: 0.0086901, 0.0116079, 0.0087925, 0.0119354, -0.0028575, 0.0024671
3: -0.0056928, -0.0026750, -0.0055869, -0.0023363, -0.0029554, 0.0025516
4: -0.0011411, 0.0021258, -0.0015078, 0.0020111, -0.0027623, 0.0031994
5: 0.0021016, 0.0051932, 0.0022101, 0.0055403, -0.0030277, 0.0026140
6: -0.0139616, -0.0016953, -0.0135312, -0.0003182, -0.0120130, 0.0103717
7: -0.0002479, 0.0164578, -0.0021233, 0.0158716, -0.0141253, 0.0163607
8: 0.9890393, 1.0008070, 0.9877181, 1.0003941, -0.0099502, 0.0115248
9: -0.0166199, -0.0059378, -0.0162451, -0.0047386, -0.0104615, 0.0090321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0070092
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0070092
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0047853, 0.0100322, 0.0043851, 0.0100699, -0.0044969, 0.0048192
1: 0.0020136, 0.0027717, 0.0019558, 0.0027771, -0.0006497, 0.0006962
2: 0.0088133, 0.0117142, 0.0087925, 0.0119354, -0.0026644, 0.0024862
3: -0.0055653, -0.0025651, -0.0055869, -0.0023363, -0.0027557, 0.0025714
4: -0.0012601, 0.0019878, -0.0015078, 0.0020111, -0.0027837, 0.0029832
5: 0.0022322, 0.0053058, 0.0022101, 0.0055403, -0.0028231, 0.0026343
6: -0.0134436, -0.0012484, -0.0135312, -0.0003182, -0.0112011, 0.0104520
7: -0.0008565, 0.0157524, -0.0021233, 0.0158716, -0.0142348, 0.0152549
8: 0.9886106, 1.0003102, 0.9877181, 1.0003941, -0.0100273, 0.0107459
9: -0.0161688, -0.0055487, -0.0162451, -0.0047386, -0.0097544, 0.0091021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0069784
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0069784
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0046072, 0.0098849, 0.0046643, 0.0102189, -0.0046493, 0.0043299
1: 0.0019879, 0.0027504, 0.0019962, 0.0027986, -0.0006717, 0.0006255
2: 0.0088948, 0.0118127, 0.0087101, 0.0117811, -0.0023939, 0.0025705
3: -0.0054811, -0.0024632, -0.0056721, -0.0024959, -0.0024759, 0.0026585
4: -0.0013704, 0.0018966, -0.0013350, 0.0021034, -0.0028780, 0.0026803
5: 0.0023185, 0.0054102, 0.0021228, 0.0053767, -0.0025365, 0.0027236
6: -0.0131012, -0.0008344, -0.0138776, -0.0009670, -0.0100639, 0.0108063
7: -0.0014204, 0.0152860, -0.0012397, 0.0163434, -0.0147172, 0.0137062
8: 0.9882133, 0.9999816, 0.9883406, 1.0007265, -0.0103671, 0.0096549
9: -0.0158706, -0.0051881, -0.0165468, -0.0053037, -0.0087641, 0.0094106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066376, upper bound: 0.0066492
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066376, upper bound: 0.0066492
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0045755, 0.0102106, 0.0047107, 0.0102128, -0.0047065, 0.0046385
1: 0.0019833, 0.0027974, 0.0020029, 0.0027978, -0.0006799, 0.0006701
2: 0.0087147, 0.0118302, 0.0087135, 0.0117555, -0.0025645, 0.0026021
3: -0.0056673, -0.0024451, -0.0056686, -0.0025224, -0.0026523, 0.0026912
4: -0.0013900, 0.0020983, -0.0013063, 0.0020996, -0.0029134, 0.0028713
5: 0.0021277, 0.0054287, 0.0021264, 0.0053495, -0.0027172, 0.0027570
6: -0.0138584, -0.0007607, -0.0138634, -0.0010749, -0.0107810, 0.0109391
7: -0.0015207, 0.0163172, -0.0010928, 0.0163240, -0.0148981, 0.0146828
8: 0.9881427, 1.0007080, 0.9884441, 1.0007129, -0.0104945, 0.0103429
9: -0.0165300, -0.0051240, -0.0165344, -0.0053976, -0.0093886, 0.0095262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067923, upper bound: 0.0066492
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067923, upper bound: 0.0066492
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046826, 0.0100546, 0.0043851, 0.0100699, -0.0045109, 0.0047545
1: 0.0019988, 0.0027749, 0.0019558, 0.0027771, -0.0006517, 0.0006869
2: 0.0088010, 0.0117710, 0.0087925, 0.0119354, -0.0026287, 0.0024939
3: -0.0055781, -0.0025064, -0.0055869, -0.0023363, -0.0027187, 0.0025794
4: -0.0013237, 0.0020017, -0.0015078, 0.0020111, -0.0027923, 0.0029431
5: 0.0022191, 0.0053660, 0.0022101, 0.0055403, -0.0027852, 0.0026425
6: -0.0134955, -0.0010097, -0.0135312, -0.0003182, -0.0110509, 0.0104845
7: -0.0011816, 0.0158231, -0.0021233, 0.0158716, -0.0142790, 0.0150503
8: 0.9883816, 1.0003599, 0.9877181, 1.0003941, -0.0100584, 0.0106018
9: -0.0162140, -0.0053408, -0.0162451, -0.0047386, -0.0096236, 0.0091304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068910
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068910
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0044215, 0.0098967, 0.0043851, 0.0100699, -0.0046237, 0.0044815
1: 0.0019611, 0.0027521, 0.0019558, 0.0027771, -0.0006680, 0.0006475
2: 0.0088882, 0.0119154, 0.0087925, 0.0119354, -0.0024777, 0.0025563
3: -0.0054878, -0.0023570, -0.0055869, -0.0023363, -0.0025626, 0.0026439
4: -0.0014853, 0.0019040, -0.0015078, 0.0020111, -0.0028621, 0.0027741
5: 0.0023116, 0.0055190, 0.0022101, 0.0055403, -0.0026253, 0.0027085
6: -0.0131287, -0.0004027, -0.0135312, -0.0003182, -0.0104163, 0.0107467
7: -0.0020083, 0.0153235, -0.0021233, 0.0158716, -0.0146361, 0.0141861
8: 0.9877992, 1.0000081, 0.9877181, 1.0003941, -0.0103100, 0.0099930
9: -0.0158946, -0.0048122, -0.0162451, -0.0047386, -0.0090710, 0.0093587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068960
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068961
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0046156, 0.0100166, 0.0049192, 0.0101402, -0.0049026, 0.0043997
1: 0.0019891, 0.0027694, 0.0020330, 0.0027873, -0.0007083, 0.0006356
2: 0.0088220, 0.0118080, 0.0087536, 0.0116402, -0.0024325, 0.0027105
3: -0.0055564, -0.0024681, -0.0056271, -0.0026417, -0.0025158, 0.0028033
4: -0.0013651, 0.0019782, -0.0011772, 0.0020547, -0.0030348, 0.0027235
5: 0.0022414, 0.0054052, 0.0021689, 0.0052274, -0.0025773, 0.0028719
6: -0.0134073, -0.0008539, -0.0136947, -0.0015596, -0.0102261, 0.0113950
7: -0.0013937, 0.0157029, -0.0004327, 0.0160942, -0.0155190, 0.0139270
8: 0.9882321, 1.0002754, 0.9889090, 1.0005510, -0.0109319, 0.0098105
9: -0.0161372, -0.0052052, -0.0163874, -0.0058197, -0.0089053, 0.0099232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0046218, 0.0103024, 0.0049192, 0.0101402, -0.0049144, 0.0047605
1: 0.0019900, 0.0028107, 0.0020330, 0.0027873, -0.0007100, 0.0006877
2: 0.0086640, 0.0118046, 0.0087536, 0.0116402, -0.0026319, 0.0027171
3: -0.0057198, -0.0024716, -0.0056271, -0.0026417, -0.0027221, 0.0028101
4: -0.0013613, 0.0021551, -0.0011772, 0.0020547, -0.0030421, 0.0029468
5: 0.0020739, 0.0054016, 0.0021689, 0.0052274, -0.0027887, 0.0028789
6: -0.0140715, -0.0008684, -0.0136947, -0.0015596, -0.0110646, 0.0114225
7: -0.0013740, 0.0166075, -0.0004327, 0.0160942, -0.0155564, 0.0150690
8: 0.9882460, 1.0009125, 0.9889090, 1.0005510, -0.0109583, 0.0106149
9: -0.0167156, -0.0052178, -0.0163874, -0.0058197, -0.0096355, 0.0099472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0047980, 0.0102156, 0.0049076, 0.0104256, -0.0050059, 0.0045135
1: 0.0020155, 0.0027982, 0.0020313, 0.0028285, -0.0007232, 0.0006521
2: 0.0087119, 0.0117072, 0.0085958, 0.0116466, -0.0024954, 0.0027677
3: -0.0056702, -0.0025723, -0.0057903, -0.0026350, -0.0025809, 0.0028624
4: -0.0012523, 0.0021014, -0.0011844, 0.0022313, -0.0030988, 0.0027940
5: 0.0021248, 0.0052984, 0.0020017, 0.0052342, -0.0026440, 0.0029325
6: -0.0138699, -0.0012778, -0.0143580, -0.0015326, -0.0104907, 0.0116352
7: -0.0008165, 0.0163329, -0.0004694, 0.0169977, -0.0158461, 0.0142874
8: 0.9886387, 1.0007192, 0.9888832, 1.0011873, -0.0111623, 0.0100644
9: -0.0165401, -0.0055743, -0.0169651, -0.0057962, -0.0091358, 0.0101324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045070, 0.0100232, 0.0049076, 0.0104256, -0.0054097, 0.0044471
1: 0.0019734, 0.0027704, 0.0020313, 0.0028285, -0.0007816, 0.0006425
2: 0.0088183, 0.0118681, 0.0085958, 0.0116466, -0.0024587, 0.0029909
3: -0.0055601, -0.0024059, -0.0057903, -0.0026350, -0.0025429, 0.0030933
4: -0.0014324, 0.0019822, -0.0011844, 0.0022313, -0.0033487, 0.0027529
5: 0.0022375, 0.0054689, 0.0020017, 0.0052342, -0.0026051, 0.0031690
6: -0.0134226, -0.0006014, -0.0143580, -0.0015326, -0.0103364, 0.0125737
7: -0.0017376, 0.0157237, -0.0004694, 0.0169977, -0.0171243, 0.0140773
8: 0.9879898, 1.0002899, 0.9888832, 1.0011873, -0.0120627, 0.0099163
9: -0.0161505, -0.0049853, -0.0169651, -0.0057962, -0.0090014, 0.0109497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043801, 0.0098754, 0.0049689, 0.0103629, -0.0053184, 0.0042936
1: 0.0019551, 0.0027490, 0.0020402, 0.0028194, -0.0007684, 0.0006203
2: 0.0089000, 0.0119382, 0.0086305, 0.0116127, -0.0023738, 0.0029404
3: -0.0054756, -0.0023334, -0.0057544, -0.0026701, -0.0024551, 0.0030411
4: -0.0015109, 0.0018907, -0.0011464, 0.0021925, -0.0032922, 0.0026578
5: 0.0023241, 0.0055432, 0.0020385, 0.0051983, -0.0025152, 0.0031155
6: -0.0130791, -0.0003066, -0.0142123, -0.0016751, -0.0099795, 0.0123613
7: -0.0021392, 0.0152559, -0.0002754, 0.0167992, -0.0168351, 0.0135912
8: 0.9877070, 0.9999604, 0.9890199, 1.0010476, -0.0118590, 0.0095739
9: -0.0158514, -0.0047285, -0.0168382, -0.0059203, -0.0086906, 0.0107648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066098, upper bound: 0.0066446
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066098, upper bound: 0.0066446
time: 1.01 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.81 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065992, upper bound: 0.0065401
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065992, upper bound: 0.0066265
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065293, upper bound: 0.0064897
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066079, upper bound: 0.0064897
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0064913
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0065759
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065450, upper bound: 0.0064273
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065953, upper bound: 0.0064273
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066586, upper bound: 0.0066158
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066586, upper bound: 0.0066480
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068104, upper bound: 0.0066158
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068104, upper bound: 0.0066480
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0065670
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066042
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0065634
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066042
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065032, upper bound: 0.0066877
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065032, upper bound: 0.0066877
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065851, upper bound: 0.0066875
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065851, upper bound: 0.0066875
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0070092
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0070092
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0069784
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0068838, upper bound: 0.0069784
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066376, upper bound: 0.0066492
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066376, upper bound: 0.0066492
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0067923, upper bound: 0.0066492
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0067923, upper bound: 0.0066492
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068910
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068910
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068960
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0069703, upper bound: 0.0068961
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066367, upper bound: 0.0065914
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0065717, upper bound: 0.0066460
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066098, upper bound: 0.0066446
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.81
Output dim: 8, lower bound: -0.0066098, upper bound: 0.0066446
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066290
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0066846
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0066809
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0067461
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0067430
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071290
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071356
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0067343
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0067343
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0070144
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0070246
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0068571
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0068562
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0065630, upper bound: 0.0068571
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066909, upper bound: 0.0068562
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0068808
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0068808
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066353, upper bound: 0.0068808
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067878, upper bound: 0.0068808
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0068752
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0068752
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071916
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0071956
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0068886
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0068886
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0071290
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0071343
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0067936
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066468, upper bound: 0.0067936
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0069075
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066331, upper bound: 0.0069075
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0068792
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0068775
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067208, upper bound: 0.0069255
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068369, upper bound: 0.0069252
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0065914, upper bound: 0.0069281
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066460, upper bound: 0.0069279
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0072918
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069964, upper bound: 0.0072968
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067484, upper bound: 0.0069511
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068731, upper bound: 0.0069511
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0072181
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070741, upper bound: 0.0072251
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066891, upper bound: 0.0065789
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068743, upper bound: 0.0065759
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066891, upper bound: 0.0065789
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068743, upper bound: 0.0065759
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067698, upper bound: 0.0066081
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069756, upper bound: 0.0066080
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067698, upper bound: 0.0066081
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069756, upper bound: 0.0066080
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067936, upper bound: 0.0066877
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069075, upper bound: 0.0066875
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0070092
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0070169
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068741, upper bound: 0.0066492
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070568, upper bound: 0.0066492
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0068910
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0069005
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067683, upper bound: 0.0065914
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067683, upper bound: 0.0065914
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068762, upper bound: 0.0066460
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068762, upper bound: 0.0066460
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068551, upper bound: 0.0066374
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070260, upper bound: 0.0066290
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068551, upper bound: 0.0066846
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070260, upper bound: 0.0066809
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067936, upper bound: 0.0067461
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069075, upper bound: 0.0067430
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0071290
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072005, upper bound: 0.0071356
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068741, upper bound: 0.0067343
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070568, upper bound: 0.0067343
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0070144
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072185, upper bound: 0.0070245
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066970, upper bound: 0.0066311
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068905, upper bound: 0.0066311
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0066970, upper bound: 0.0066311
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068905, upper bound: 0.0066311
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067772, upper bound: 0.0066441
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069822, upper bound: 0.0066441
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067772, upper bound: 0.0066441
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069822, upper bound: 0.0066441
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068161, upper bound: 0.0067050
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069333, upper bound: 0.0067050
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0069949
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0069982
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068768, upper bound: 0.0066628
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070599, upper bound: 0.0066628
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0068861
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0068907
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067863, upper bound: 0.0066404
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0067863, upper bound: 0.0066404
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069081, upper bound: 0.0067118
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069081, upper bound: 0.0067118
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068659, upper bound: 0.0066988
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070473, upper bound: 0.0066988
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068659, upper bound: 0.0067265
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070473, upper bound: 0.0067265
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068161, upper bound: 0.0067835
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0069333, upper bound: 0.0067835
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0071153
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072010, upper bound: 0.0071200
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0068768, upper bound: 0.0067538
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0070599, upper bound: 0.0067538
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0070000
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 8, lower bound: -0.0072190, upper bound: 0.0070093

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.93 + 596.65 = 600.58 seconds
