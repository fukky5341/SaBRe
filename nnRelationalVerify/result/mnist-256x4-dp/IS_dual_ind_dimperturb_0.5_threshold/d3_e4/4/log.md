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
Threshold: 0.00913976


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0026746, 0.0226208, 0.0026746, 0.0226208, -0.0199462, 0.0199462)
1: (-0.0052098, 0.0040924, -0.0052098, 0.0040924, -0.0093021, 0.0093021)
2: (-0.0007426, 0.0128812, -0.0007426, 0.0128812, -0.0136238, 0.0136238)
3: (-0.0066376, 0.0046767, -0.0066376, 0.0046767, -0.0113143, 0.0113143)
4: (-0.0034674, 0.0020224, -0.0034674, 0.0020224, -0.0054897, 0.0054897)
5: (-0.0027460, 0.0065423, -0.0027460, 0.0065423, -0.0092882, 0.0092882)
6: (-0.0172625, 0.0036575, -0.0172625, 0.0036575, -0.0209200, 0.0209200)
7: (-0.0148974, 0.0159290, -0.0148974, 0.0159290, -0.0308264, 0.0308264)
8: (0.9810172, 1.0015051, 0.9810172, 1.0015051, -0.0204880, 0.0204880)
9: (-0.0162818, 0.0024138, -0.0162818, 0.0024138, -0.0186956, 0.0186956)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.79 + 2.41 = 4.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0125867, upper bound: 0.0125867

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0117217, upper bound: 0.0119770
time: 1.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123270, upper bound: 0.0123270
time: 1.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 8, lower bound: -0.0117217, upper bound: 0.0119770
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 8, lower bound: -0.0123270, upper bound: 0.0123270

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0034823, 0.0163106, 0.0030370, 0.0197472, -0.0162648, 0.0132736
1: -0.0017891, 0.0029999, -0.0036907, 0.0036805, -0.0054696, 0.0066906
2: 0.0040429, 0.0124346, 0.0014049, 0.0126808, -0.0086378, 0.0110297
3: -0.0058662, 0.0012788, -0.0063034, 0.0031379, -0.0090042, 0.0075821
4: -0.0023670, 0.0021503, -0.0029580, 0.0020118, -0.0042966, 0.0050927
5: -0.0002814, 0.0060691, -0.0016106, 0.0063299, -0.0066113, 0.0076797
6: -0.0160421, 0.0017800, -0.0165060, 0.0028150, -0.0188571, 0.0182860
7: -0.0082704, 0.0165834, -0.0118675, 0.0158748, -0.0240433, 0.0284509
8: 0.9857051, 1.0014806, 0.9835197, 1.0012718, -0.0155667, 0.0179609
9: -0.0167002, -0.0017596, -0.0162471, 0.0004743, -0.0171746, 0.0142636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114575, upper bound: 0.0116199
time: 1.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114575, upper bound: 0.0117045
time: 1.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0028222, 0.0217519, 0.0027008, 0.0224674, -0.0196452, 0.0190511
1: -0.0047992, 0.0039781, -0.0051370, 0.0040721, -0.0088713, 0.0091151
2: -0.0001120, 0.0127995, -0.0006290, 0.0128667, -0.0129787, 0.0134286
3: -0.0065435, 0.0042536, -0.0066210, 0.0046002, -0.0111437, 0.0108746
4: -0.0033039, 0.0020175, -0.0034379, 0.0020216, -0.0051135, 0.0054554
5: -0.0023902, 0.0064558, -0.0026834, 0.0065269, -0.0089171, 0.0091392
6: -0.0170347, 0.0033144, -0.0172225, 0.0035966, -0.0206312, 0.0205369
7: -0.0139740, 0.0159042, -0.0147319, 0.0159249, -0.0291874, 0.0306361
8: 0.9818199, 1.0014328, 0.9811622, 1.0014926, -0.0196728, 0.0202706
9: -0.0162660, 0.0018090, -0.0162791, 0.0023042, -0.0185701, 0.0175665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120585, upper bound: 0.0119679
time: 1.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120585, upper bound: 0.0120586
time: 2.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.36
Output dim: 8, lower bound: -0.0114575, upper bound: 0.0116199
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.36
Output dim: 8, lower bound: -0.0114575, upper bound: 0.0117045
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.36
Output dim: 8, lower bound: -0.0120585, upper bound: 0.0119679
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.36
Output dim: 8, lower bound: -0.0120585, upper bound: 0.0120586

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037091, 0.0162205, 0.0035719, 0.0195296, -0.0158205, 0.0126486
1: -0.0017631, 0.0029768, -0.0036163, 0.0035984, -0.0053615, 0.0065931
2: 0.0040986, 0.0123092, 0.0015438, 0.0123851, -0.0082864, 0.0107654
3: -0.0058426, 0.0011879, -0.0062234, 0.0029646, -0.0088072, 0.0074113
4: -0.0022271, 0.0021431, -0.0026394, 0.0019917, -0.0041189, 0.0047553
5: -0.0002375, 0.0059362, -0.0014946, 0.0060166, -0.0062541, 0.0074308
6: -0.0159446, 0.0012530, -0.0162108, 0.0015719, -0.0175165, 0.0174638
7: -0.0075844, 0.0165462, -0.0103179, 0.0157722, -0.0231968, 0.0268641
8: 0.9862108, 1.0014305, 0.9847120, 1.0011240, -0.0149132, 0.0167185
9: -0.0166764, -0.0022188, -0.0161815, -0.0005494, -0.0161271, 0.0136828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0116198
time: 1.88 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0116198
time: 1.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0036696, 0.0162464, 0.0034206, 0.0210442, -0.0173746, 0.0128258
1: -0.0017746, 0.0029836, -0.0040582, 0.0038495, -0.0056240, 0.0070417
2: 0.0040802, 0.0123310, 0.0005353, 0.0124687, -0.0083885, 0.0117957
3: -0.0058476, 0.0011910, -0.0065286, 0.0033494, -0.0091970, 0.0077195
4: -0.0022492, 0.0021426, -0.0028018, 0.0021353, -0.0043845, 0.0049201
5: -0.0002483, 0.0059594, -0.0021939, 0.0061052, -0.0063536, 0.0081533
6: -0.0159560, 0.0013448, -0.0173134, 0.0019235, -0.0178795, 0.0186582
7: -0.0076677, 0.0165437, -0.0111707, 0.0165065, -0.0241742, 0.0277144
8: 0.9861227, 1.0014327, 0.9837726, 1.0018044, -0.0156816, 0.0176601
9: -0.0166748, -0.0021488, -0.0166511, 0.0000615, -0.0167363, 0.0145023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0117045
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0117045
time: 1.46 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0030364, 0.0216663, 0.0032180, 0.0222488, -0.0192124, 0.0184483
1: -0.0047702, 0.0039404, -0.0050606, 0.0039749, -0.0087451, 0.0090010
2: -0.0000605, 0.0126811, -0.0004907, 0.0125807, -0.0126412, 0.0131717
3: -0.0065077, 0.0041974, -0.0065285, 0.0044772, -0.0109848, 0.0107259
4: -0.0031765, 0.0020095, -0.0031292, 0.0020007, -0.0049619, 0.0051387
5: -0.0023448, 0.0063303, -0.0025678, 0.0062239, -0.0085687, 0.0088981
6: -0.0169173, 0.0028164, -0.0169198, 0.0023944, -0.0193116, 0.0197363
7: -0.0133500, 0.0158631, -0.0132285, 0.0158181, -0.0284313, 0.0290916
8: 0.9822774, 1.0013697, 0.9822674, 1.0013288, -0.0190514, 0.0191023
9: -0.0162396, 0.0014003, -0.0162108, 0.0013201, -0.0175597, 0.0170825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116116, upper bound: 0.0116987
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118278, upper bound: 0.0117285
time: 1.23 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0030254, 0.0216933, 0.0030861, 0.0236426, -0.0206173, 0.0186072
1: -0.0047830, 0.0039446, -0.0054404, 0.0041851, -0.0089681, 0.0093850
2: -0.0000804, 0.0126872, -0.0014144, 0.0126537, -0.0127341, 0.0141016
3: -0.0065107, 0.0042103, -0.0068013, 0.0048277, -0.0113384, 0.0110116
4: -0.0031885, 0.0020091, -0.0032763, 0.0021444, -0.0053152, 0.0052854
5: -0.0023556, 0.0063368, -0.0032437, 0.0063012, -0.0086568, 0.0095805
6: -0.0169239, 0.0028422, -0.0180355, 0.0027010, -0.0196249, 0.0208777
7: -0.0134123, 0.0158612, -0.0140075, 0.0165532, -0.0299655, 0.0298687
8: 0.9822299, 1.0013700, 0.9814024, 1.0020163, -0.0197864, 0.0199676
9: -0.0162384, 0.0014414, -0.0166809, 0.0018790, -0.0181174, 0.0181223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0116115, upper bound: 0.0117993
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0118279, upper bound: 0.0118278
time: 1.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0116198
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0116198
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0117045
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0113633, upper bound: 0.0117045
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0116116, upper bound: 0.0116987
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0118278, upper bound: 0.0117285
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0116115, upper bound: 0.0117993
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.53
Output dim: 8, lower bound: -0.0118279, upper bound: 0.0118278

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0040263, 0.0160803, 0.0035719, 0.0195296, -0.0155033, 0.0125084
1: -0.0017217, 0.0029421, -0.0036163, 0.0035984, -0.0053201, 0.0065584
2: 0.0041843, 0.0121338, 0.0015438, 0.0123851, -0.0082007, 0.0105900
3: -0.0058068, 0.0010562, -0.0062234, 0.0029646, -0.0087714, 0.0072796
4: -0.0020289, 0.0021319, -0.0026394, 0.0019917, -0.0039057, 0.0047445
5: -0.0001692, 0.0057505, -0.0014946, 0.0060166, -0.0061858, 0.0072450
6: -0.0157951, 0.0005159, -0.0162108, 0.0015719, -0.0173670, 0.0167267
7: -0.0066161, 0.0164890, -0.0103179, 0.0157722, -0.0221531, 0.0268069
8: 0.9869180, 1.0013543, 0.9847120, 1.0011240, -0.0142061, 0.0166423
9: -0.0166399, -0.0028693, -0.0161815, -0.0005494, -0.0160905, 0.0129818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0111497
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0113237
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038594, 0.0174761, 0.0035719, 0.0195296, -0.0156702, 0.0139042
1: -0.0021458, 0.0032862, -0.0036163, 0.0035984, -0.0057442, 0.0069025
2: 0.0032201, 0.0122261, 0.0015438, 0.0123851, -0.0091649, 0.0106823
3: -0.0062056, 0.0014183, -0.0062234, 0.0029646, -0.0091703, 0.0076418
4: -0.0022045, 0.0023025, -0.0026394, 0.0019917, -0.0041962, 0.0049419
5: -0.0007866, 0.0058482, -0.0014946, 0.0060166, -0.0068032, 0.0073428
6: -0.0168305, 0.0009038, -0.0162108, 0.0015719, -0.0184024, 0.0171146
7: -0.0075266, 0.0173615, -0.0103179, 0.0157722, -0.0232987, 0.0276793
8: 0.9865458, 1.0021138, 0.9847120, 1.0011240, -0.0145782, 0.0174018
9: -0.0171977, -0.0022170, -0.0161815, -0.0005494, -0.0166484, 0.0139645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0111497
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0113237
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0040263, 0.0160803, 0.0034206, 0.0210442, -0.0170180, 0.0126597
1: -0.0017217, 0.0029421, -0.0040582, 0.0038495, -0.0055712, 0.0070002
2: 0.0041843, 0.0121338, 0.0005353, 0.0124687, -0.0082844, 0.0115985
3: -0.0058068, 0.0010562, -0.0065286, 0.0033494, -0.0091562, 0.0075848
4: -0.0020289, 0.0021319, -0.0028018, 0.0021353, -0.0041606, 0.0049336
5: -0.0001692, 0.0057505, -0.0021939, 0.0061052, -0.0062745, 0.0079443
6: -0.0157951, 0.0005159, -0.0173134, 0.0019235, -0.0177186, 0.0178293
7: -0.0066161, 0.0164890, -0.0111707, 0.0165065, -0.0231226, 0.0276598
8: 0.9869180, 1.0013543, 0.9837726, 1.0018044, -0.0148864, 0.0175818
9: -0.0166399, -0.0028693, -0.0166511, 0.0000615, -0.0167014, 0.0137817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0112241
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0114145
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038594, 0.0174761, 0.0034206, 0.0210442, -0.0171849, 0.0140555
1: -0.0021458, 0.0032862, -0.0040582, 0.0038495, -0.0059953, 0.0073444
2: 0.0032201, 0.0122261, 0.0005353, 0.0124687, -0.0092486, 0.0116908
3: -0.0062056, 0.0014183, -0.0065286, 0.0033494, -0.0095551, 0.0079469
4: -0.0022045, 0.0023025, -0.0028018, 0.0021353, -0.0040911, 0.0049362
5: -0.0007866, 0.0058482, -0.0021939, 0.0061052, -0.0068919, 0.0080421
6: -0.0168305, 0.0009038, -0.0173134, 0.0019235, -0.0187540, 0.0182172
7: -0.0075266, 0.0173615, -0.0111707, 0.0165065, -0.0231642, 0.0280753
8: 0.9865458, 1.0021138, 0.9837726, 1.0018044, -0.0152586, 0.0183412
9: -0.0171977, -0.0022170, -0.0166511, 0.0000615, -0.0168829, 0.0137071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0111466
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0113236
time: 1.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0034659, 0.0190413, 0.0033821, 0.0212435, -0.0177776, 0.0156592
1: -0.0033960, 0.0035398, -0.0045395, 0.0038425, -0.0072385, 0.0080793
2: 0.0019042, 0.0124437, 0.0002557, 0.0124900, -0.0105858, 0.0121879
3: -0.0061823, 0.0027882, -0.0064208, 0.0039248, -0.0101071, 0.0092090
4: -0.0026672, 0.0019988, -0.0029318, 0.0019969, -0.0043324, 0.0049306
5: -0.0013025, 0.0060787, -0.0021668, 0.0061278, -0.0074303, 0.0082455
6: -0.0162163, 0.0018182, -0.0166553, 0.0020131, -0.0182293, 0.0184735
7: -0.0103893, 0.0158087, -0.0120770, 0.0157990, -0.0247660, 0.0278857
8: 0.9847451, 1.0011476, 0.9832456, 1.0012494, -0.0165043, 0.0179020
9: -0.0162049, -0.0005074, -0.0161987, 0.0005733, -0.0167782, 0.0147063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0110002
time: 1.79 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0114736
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033222, 0.0204609, 0.0033728, 0.0215995, -0.0182774, 0.0170881
1: -0.0041101, 0.0038140, -0.0047329, 0.0038846, -0.0079947, 0.0085469
2: 0.0008744, 0.0125231, -0.0000110, 0.0124951, -0.0116208, 0.0125341
3: -0.0065046, 0.0035320, -0.0064545, 0.0041134, -0.0106181, 0.0099865
4: -0.0028961, 0.0021410, -0.0029806, 0.0019974, -0.0045658, 0.0051216
5: -0.0018904, 0.0061629, -0.0023048, 0.0061333, -0.0080237, 0.0084678
6: -0.0169654, 0.0021524, -0.0167454, 0.0020347, -0.0190001, 0.0188978
7: -0.0117852, 0.0165358, -0.0123900, 0.0158014, -0.0262117, 0.0289258
8: 0.9836053, 1.0017016, 0.9829737, 1.0012736, -0.0176684, 0.0187279
9: -0.0166698, 0.0003766, -0.0162002, 0.0007762, -0.0174460, 0.0156389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0110128
time: 1.90 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0114998
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034504, 0.0190672, 0.0032535, 0.0226215, -0.0191711, 0.0158136
1: -0.0034083, 0.0035448, -0.0049152, 0.0040568, -0.0074652, 0.0084600
2: 0.0018851, 0.0124522, -0.0006515, 0.0125611, -0.0106760, 0.0131037
3: -0.0061859, 0.0027998, -0.0066968, 0.0042704, -0.0104563, 0.0094966
4: -0.0026765, 0.0019984, -0.0030744, 0.0021406, -0.0047087, 0.0050728
5: -0.0013129, 0.0060878, -0.0028302, 0.0062031, -0.0075160, 0.0089179
6: -0.0162223, 0.0018542, -0.0177582, 0.0023119, -0.0185342, 0.0196124
7: -0.0104409, 0.0158066, -0.0128564, 0.0165334, -0.0265954, 0.0286631
8: 0.9847029, 1.0011480, 0.9824037, 1.0019311, -0.0172282, 0.0187443
9: -0.0162035, -0.0004746, -0.0166683, 0.0011248, -0.0173283, 0.0159158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0110922
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0115704
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0204812, 0.0032415, 0.0229844, -0.0196759, 0.0172397
1: -0.0041198, 0.0038178, -0.0051061, 0.0040962, -0.0082160, 0.0089239
2: 0.0008594, 0.0125306, -0.0009223, 0.0125677, -0.0117083, 0.0134530
3: -0.0065072, 0.0035410, -0.0067283, 0.0044625, -0.0109697, 0.0102693
4: -0.0029092, 0.0021404, -0.0031274, 0.0021410, -0.0049284, 0.0052678
5: -0.0018986, 0.0061709, -0.0029752, 0.0062102, -0.0081088, 0.0091461
6: -0.0169699, 0.0021840, -0.0178521, 0.0023399, -0.0193098, 0.0200360
7: -0.0118617, 0.0165327, -0.0131742, 0.0165358, -0.0279881, 0.0297069
8: 0.9835630, 1.0017009, 0.9821138, 1.0019567, -0.0183937, 0.0195871
9: -0.0166678, 0.0004249, -0.0166698, 0.0013329, -0.0180008, 0.0168012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0111021
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0115959
time: 1.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.47 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0111497
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0113237
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0111497
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0113237
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0112241
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0114145
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0109977, upper bound: 0.0111466
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0110102, upper bound: 0.0113236
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0110002
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0114736
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0110128
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0114998
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0110922
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0112241, upper bound: 0.0115704
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0111021
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.47
Output dim: 8, lower bound: -0.0114144, upper bound: 0.0115959

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041727, 0.0151975, 0.0039643, 0.0170153, -0.0128427, 0.0112332
1: -0.0012252, 0.0028043, -0.0022902, 0.0031333, -0.0043585, 0.0050946
2: 0.0048696, 0.0120529, 0.0034363, 0.0121681, -0.0072985, 0.0086166
3: -0.0056946, 0.0005637, -0.0058473, 0.0016101, -0.0073047, 0.0064110
4: -0.0018817, 0.0021278, -0.0021629, 0.0019815, -0.0037030, 0.0041203
5: 0.0001764, 0.0056647, -0.0005120, 0.0057868, -0.0056103, 0.0061767
6: -0.0155463, 0.0001756, -0.0155540, 0.0006599, -0.0162062, 0.0157296
7: -0.0056678, 0.0164681, -0.0075009, 0.0157202, -0.0208787, 0.0234315
8: 0.9872444, 1.0012723, 0.9867798, 1.0009103, -0.0131918, 0.0144925
9: -0.0166265, -0.0033973, -0.0161483, -0.0023388, -0.0137975, 0.0122512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096492, upper bound: 0.0103142
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107681, upper bound: 0.0109344
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041875, 0.0154598, 0.0038738, 0.0183693, -0.0141817, 0.0115860
1: -0.0013780, 0.0028044, -0.0029630, 0.0034375, -0.0048155, 0.0057674
2: 0.0046591, 0.0120447, 0.0024367, 0.0122182, -0.0075590, 0.0096080
3: -0.0056949, 0.0006786, -0.0061927, 0.0022989, -0.0079939, 0.0068713
4: -0.0018804, 0.0021281, -0.0023578, 0.0021227, -0.0040032, 0.0043422
5: 0.0000723, 0.0056560, -0.0010646, 0.0058398, -0.0057675, 0.0067206
6: -0.0156218, 0.0001410, -0.0162858, 0.0008703, -0.0164921, 0.0164268
7: -0.0057214, 0.0164698, -0.0087397, 0.0164422, -0.0221636, 0.0247889
8: 0.9872776, 1.0012946, 0.9860262, 1.0014645, -0.0141869, 0.0152684
9: -0.0166276, -0.0033980, -0.0166099, -0.0015764, -0.0146783, 0.0132119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096492, upper bound: 0.0104669
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107716, upper bound: 0.0111012
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040014, 0.0165540, 0.0039643, 0.0170153, -0.0130139, 0.0125897
1: -0.0016524, 0.0030600, -0.0022902, 0.0031333, -0.0047857, 0.0053502
2: 0.0039178, 0.0121476, 0.0034363, 0.0121681, -0.0082503, 0.0087113
3: -0.0060231, 0.0009293, -0.0058473, 0.0016101, -0.0076331, 0.0067766
4: -0.0020425, 0.0022981, -0.0021629, 0.0019815, -0.0039973, 0.0044444
5: -0.0004267, 0.0057650, -0.0005120, 0.0057868, -0.0062135, 0.0062770
6: -0.0165717, 0.0005737, -0.0155540, 0.0006599, -0.0172317, 0.0161277
7: -0.0065548, 0.0173390, -0.0075009, 0.0157202, -0.0222751, 0.0248399
8: 0.9868625, 1.0020244, 0.9867798, 1.0009103, -0.0140477, 0.0152446
9: -0.0171834, -0.0028204, -0.0161483, -0.0023388, -0.0148445, 0.0132785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096042, upper bound: 0.0102686
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108646, upper bound: 0.0109342
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0040303, 0.0168580, 0.0038738, 0.0183693, -0.0143390, 0.0129842
1: -0.0018097, 0.0031353, -0.0029630, 0.0034375, -0.0052472, 0.0060982
2: 0.0036904, 0.0121316, 0.0024367, 0.0122182, -0.0085277, 0.0096950
3: -0.0060831, 0.0010530, -0.0061927, 0.0022989, -0.0083821, 0.0072457
4: -0.0020474, 0.0022986, -0.0023578, 0.0021227, -0.0041701, 0.0046564
5: -0.0005424, 0.0057481, -0.0010646, 0.0058398, -0.0063822, 0.0068127
6: -0.0166545, 0.0005066, -0.0162858, 0.0008703, -0.0175247, 0.0167924
7: -0.0066212, 0.0173418, -0.0087397, 0.0164422, -0.0230634, 0.0260815
8: 0.9869269, 1.0020509, 0.9860262, 1.0014645, -0.0145376, 0.0160247
9: -0.0171851, -0.0027839, -0.0166099, -0.0015764, -0.0156087, 0.0138260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096041, upper bound: 0.0104233
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108685, upper bound: 0.0111006
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041727, 0.0151975, 0.0038115, 0.0184377, -0.0142650, 0.0113860
1: -0.0012252, 0.0028043, -0.0027047, 0.0034217, -0.0046469, 0.0055091
2: 0.0048696, 0.0120529, 0.0024783, 0.0122526, -0.0073830, 0.0095746
3: -0.0056946, 0.0005637, -0.0061816, 0.0019778, -0.0076724, 0.0067453
4: -0.0018817, 0.0021278, -0.0023283, 0.0021247, -0.0039569, 0.0043913
5: 0.0001764, 0.0056647, -0.0011526, 0.0058763, -0.0056998, 0.0068173
6: -0.0155463, 0.0001756, -0.0165944, 0.0010150, -0.0165613, 0.0167700
7: -0.0056678, 0.0164681, -0.0083657, 0.0164522, -0.0221201, 0.0246929
8: 0.9872444, 1.0012723, 0.9860114, 1.0015724, -0.0143280, 0.0152609
9: -0.0166265, -0.0033973, -0.0166164, -0.0017213, -0.0147624, 0.0130812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096240, upper bound: 0.0103434
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107681, upper bound: 0.0110123
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041875, 0.0154598, 0.0037308, 0.0198020, -0.0156145, 0.0117289
1: -0.0013780, 0.0028044, -0.0033802, 0.0037018, -0.0050798, 0.0061846
2: 0.0046591, 0.0120447, 0.0014846, 0.0122972, -0.0076381, 0.0105601
3: -0.0056949, 0.0006786, -0.0065112, 0.0026757, -0.0083707, 0.0071897
4: -0.0018804, 0.0021281, -0.0025184, 0.0022702, -0.0041507, 0.0045902
5: 0.0000723, 0.0056560, -0.0017185, 0.0059235, -0.0058512, 0.0073745
6: -0.0156218, 0.0001410, -0.0173194, 0.0012025, -0.0168243, 0.0174604
7: -0.0057214, 0.0164698, -0.0095950, 0.0171964, -0.0229178, 0.0259653
8: 0.9872776, 1.0012946, 0.9851018, 1.0021290, -0.0148513, 0.0161929
9: -0.0166276, -0.0033980, -0.0170922, -0.0009678, -0.0155625, 0.0136942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096240, upper bound: 0.0105013
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107709, upper bound: 0.0111921
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0040014, 0.0165540, 0.0038115, 0.0184377, -0.0144363, 0.0127425
1: -0.0016524, 0.0030600, -0.0027047, 0.0034217, -0.0050741, 0.0057647
2: 0.0039178, 0.0121476, 0.0024783, 0.0122526, -0.0083348, 0.0096693
3: -0.0060231, 0.0009293, -0.0061816, 0.0019778, -0.0080008, 0.0071110
4: -0.0020425, 0.0022981, -0.0023283, 0.0021247, -0.0038577, 0.0043060
5: -0.0004267, 0.0057650, -0.0011526, 0.0058763, -0.0063030, 0.0069177
6: -0.0165717, 0.0005737, -0.0165944, 0.0010150, -0.0175868, 0.0168632
7: -0.0065548, 0.0173390, -0.0083657, 0.0164522, -0.0217991, 0.0244484
8: 0.9868625, 1.0020244, 0.9860114, 1.0015724, -0.0142361, 0.0160130
9: -0.0171834, -0.0028204, -0.0166164, -0.0017213, -0.0145226, 0.0128399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096042, upper bound: 0.0102680
time: 1.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108646, upper bound: 0.0109310
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0040303, 0.0168580, 0.0037308, 0.0198020, -0.0157717, 0.0131271
1: -0.0018097, 0.0031353, -0.0033802, 0.0037018, -0.0055115, 0.0065154
2: 0.0036904, 0.0121316, 0.0014846, 0.0122972, -0.0086068, 0.0106471
3: -0.0060831, 0.0010530, -0.0065112, 0.0026757, -0.0087589, 0.0075642
4: -0.0020474, 0.0022986, -0.0025184, 0.0022702, -0.0042236, 0.0045294
5: -0.0005424, 0.0057481, -0.0017185, 0.0059235, -0.0064660, 0.0074667
6: -0.0166545, 0.0005066, -0.0173194, 0.0012025, -0.0178570, 0.0178259
7: -0.0066212, 0.0173418, -0.0095950, 0.0171964, -0.0237587, 0.0258211
8: 0.9869269, 1.0020509, 0.9851018, 1.0021290, -0.0152020, 0.0169491
9: -0.0171851, -0.0027839, -0.0170922, -0.0009678, -0.0154093, 0.0140739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096041, upper bound: 0.0104231
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108685, upper bound: 0.0111006
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034659, 0.0190413, 0.0041727, 0.0151975, -0.0117316, 0.0148687
1: -0.0033960, 0.0035398, -0.0012252, 0.0028043, -0.0062004, 0.0047650
2: 0.0019042, 0.0124437, 0.0048696, 0.0120529, -0.0101487, 0.0075740
3: -0.0061823, 0.0027882, -0.0056946, 0.0005637, -0.0067460, 0.0084829
4: -0.0026672, 0.0019988, -0.0018817, 0.0021278, -0.0047571, 0.0037203
5: -0.0013025, 0.0060787, 0.0001764, 0.0056647, -0.0069672, 0.0059023
6: -0.0162163, 0.0018182, -0.0155463, 0.0001756, -0.0163919, 0.0173645
7: -0.0103893, 0.0158087, -0.0056678, 0.0164681, -0.0268574, 0.0209673
8: 0.9847451, 1.0011476, 0.9872444, 1.0012723, -0.0165272, 0.0134054
9: -0.0162049, -0.0005074, -0.0166265, -0.0033973, -0.0123078, 0.0161013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0100676
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0107681
time: 2.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034659, 0.0190413, 0.0035059, 0.0205211, -0.0170552, 0.0155354
1: -0.0033960, 0.0035398, -0.0041966, 0.0037446, -0.0071406, 0.0077364
2: 0.0019042, 0.0124437, 0.0007804, 0.0124216, -0.0105173, 0.0116633
3: -0.0061823, 0.0027882, -0.0063404, 0.0035638, -0.0097461, 0.0091287
4: -0.0026672, 0.0019988, -0.0027907, 0.0019931, -0.0043285, 0.0045176
5: -0.0013025, 0.0060787, -0.0018719, 0.0060553, -0.0073578, 0.0079506
6: -0.0162163, 0.0018182, -0.0164712, 0.0017253, -0.0179416, 0.0182894
7: -0.0103893, 0.0158087, -0.0112871, 0.0157795, -0.0247461, 0.0260186
8: 0.9847451, 1.0011476, 0.9839297, 1.0011929, -0.0164478, 0.0172179
9: -0.0162049, -0.0005074, -0.0161862, 0.0000541, -0.0155114, 0.0146935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0107177
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0112741
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033222, 0.0204609, 0.0041875, 0.0154598, -0.0121376, 0.0162734
1: -0.0041101, 0.0038140, -0.0013780, 0.0028044, -0.0069145, 0.0051920
2: 0.0008744, 0.0125231, 0.0046591, 0.0120447, -0.0111703, 0.0078640
3: -0.0065046, 0.0035320, -0.0056949, 0.0006786, -0.0071832, 0.0092269
4: -0.0028961, 0.0021410, -0.0018804, 0.0021281, -0.0050001, 0.0040215
5: -0.0018904, 0.0061629, 0.0000723, 0.0056560, -0.0075464, 0.0060906
6: -0.0169654, 0.0021524, -0.0156218, 0.0001410, -0.0171065, 0.0177742
7: -0.0117852, 0.0165358, -0.0057214, 0.0164698, -0.0282550, 0.0222572
8: 0.9836053, 1.0017016, 0.9872776, 1.0012946, -0.0176893, 0.0144240
9: -0.0166698, 0.0003766, -0.0166276, -0.0033980, -0.0132718, 0.0170042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0106610
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0107007
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033222, 0.0204609, 0.0034927, 0.0208937, -0.0175716, 0.0169681
1: -0.0041101, 0.0038140, -0.0044013, 0.0037923, -0.0079024, 0.0082152
2: 0.0008744, 0.0125231, 0.0004952, 0.0124288, -0.0115544, 0.0120279
3: -0.0065046, 0.0035320, -0.0063786, 0.0037578, -0.0102625, 0.0099106
4: -0.0028961, 0.0021410, -0.0028413, 0.0019936, -0.0045619, 0.0048862
5: -0.0018904, 0.0061629, -0.0020165, 0.0060630, -0.0079534, 0.0081794
6: -0.0169654, 0.0021524, -0.0165677, 0.0017559, -0.0187213, 0.0187200
7: -0.0117852, 0.0165358, -0.0115992, 0.0157821, -0.0261919, 0.0279690
8: 0.9836053, 1.0017016, 0.9836503, 1.0012188, -0.0176135, 0.0180513
9: -0.0166698, 0.0003766, -0.0161878, 0.0002582, -0.0167646, 0.0156262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0111696
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0112621
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034504, 0.0190672, 0.0040014, 0.0165540, -0.0131035, 0.0150658
1: -0.0034083, 0.0035448, -0.0016524, 0.0030600, -0.0064683, 0.0051972
2: 0.0018851, 0.0124522, 0.0039178, 0.0121476, -0.0102625, 0.0085344
3: -0.0061859, 0.0027998, -0.0060231, 0.0009293, -0.0071153, 0.0088228
4: -0.0026765, 0.0019984, -0.0020425, 0.0022981, -0.0049747, 0.0038539
5: -0.0013129, 0.0060878, -0.0004267, 0.0057650, -0.0070779, 0.0065145
6: -0.0162223, 0.0018542, -0.0165717, 0.0005737, -0.0167960, 0.0184259
7: -0.0104409, 0.0158066, -0.0065548, 0.0173390, -0.0277799, 0.0217801
8: 0.9847029, 1.0011480, 0.9868625, 1.0020244, -0.0173215, 0.0142369
9: -0.0162035, -0.0004746, -0.0171834, -0.0028204, -0.0128278, 0.0167088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0101469
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0108646
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034504, 0.0190672, 0.0033877, 0.0218350, -0.0183845, 0.0156795
1: -0.0034083, 0.0035448, -0.0045588, 0.0039553, -0.0073637, 0.0081036
2: 0.0018851, 0.0124522, -0.0000822, 0.0124869, -0.0106018, 0.0125344
3: -0.0061859, 0.0027998, -0.0066135, 0.0038996, -0.0100855, 0.0094133
4: -0.0026765, 0.0019984, -0.0029352, 0.0021367, -0.0047047, 0.0046622
5: -0.0013129, 0.0060878, -0.0025118, 0.0061245, -0.0074374, 0.0085996
6: -0.0162223, 0.0018542, -0.0175581, 0.0020001, -0.0182224, 0.0194123
7: -0.0104409, 0.0158066, -0.0120642, 0.0165137, -0.0265753, 0.0268324
8: 0.9847029, 1.0011480, 0.9831065, 1.0018685, -0.0171656, 0.0180415
9: -0.0162035, -0.0004746, -0.0166556, 0.0006057, -0.0160912, 0.0159029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0108115
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0113733
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0204812, 0.0040303, 0.0168580, -0.0135494, 0.0164509
1: -0.0041198, 0.0038178, -0.0018097, 0.0031353, -0.0072551, 0.0056275
2: 0.0008594, 0.0125306, 0.0036904, 0.0121316, -0.0112722, 0.0088402
3: -0.0065072, 0.0035410, -0.0060831, 0.0010530, -0.0075602, 0.0096241
4: -0.0029092, 0.0021404, -0.0020474, 0.0022986, -0.0052078, 0.0041878
5: -0.0018986, 0.0061709, -0.0005424, 0.0057481, -0.0076468, 0.0067133
6: -0.0169699, 0.0021840, -0.0166545, 0.0005066, -0.0174765, 0.0188384
7: -0.0118617, 0.0165327, -0.0066212, 0.0173418, -0.0292034, 0.0231540
8: 0.9835630, 1.0017009, 0.9869269, 1.0020509, -0.0184879, 0.0147740
9: -0.0166678, 0.0004249, -0.0171851, -0.0027839, -0.0138839, 0.0176100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0107506
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0107923
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033086, 0.0204812, 0.0033647, 0.0222055, -0.0188970, 0.0171164
1: -0.0041198, 0.0038178, -0.0047443, 0.0039994, -0.0081192, 0.0085620
2: 0.0008594, 0.0125306, -0.0003575, 0.0124996, -0.0116401, 0.0128881
3: -0.0065072, 0.0035410, -0.0066488, 0.0040871, -0.0105943, 0.0101898
4: -0.0029092, 0.0021404, -0.0029880, 0.0021372, -0.0049245, 0.0050295
5: -0.0018986, 0.0061709, -0.0026541, 0.0061380, -0.0080366, 0.0088250
6: -0.0169699, 0.0021840, -0.0176538, 0.0020534, -0.0190233, 0.0198378
7: -0.0118617, 0.0165327, -0.0123902, 0.0165163, -0.0279682, 0.0287718
8: 0.9835630, 1.0017009, 0.9828153, 1.0018947, -0.0183317, 0.0188856
9: -0.0166678, 0.0004249, -0.0166573, 0.0008161, -0.0173366, 0.0167885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0112673
time: 1.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0113576
time: 1.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.64 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096492, upper bound: 0.0103142
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0107681, upper bound: 0.0109344
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096492, upper bound: 0.0104669
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0107716, upper bound: 0.0111012
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096042, upper bound: 0.0102686
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0108646, upper bound: 0.0109342
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096041, upper bound: 0.0104233
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0108685, upper bound: 0.0111006
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096240, upper bound: 0.0103434
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0107681, upper bound: 0.0110123
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096240, upper bound: 0.0105013
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0107709, upper bound: 0.0111921
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096042, upper bound: 0.0102680
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0108646, upper bound: 0.0109310
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0096041, upper bound: 0.0104231
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0108685, upper bound: 0.0111006
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0100676
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0107681
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0107177
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0112741
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0106610
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0107007
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0111696
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0112621
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0101469
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0108646
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0099460, upper bound: 0.0108115
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0110123, upper bound: 0.0113733
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0107506
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0107923
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0109380, upper bound: 0.0112673
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 8, lower bound: -0.0111340, upper bound: 0.0113576

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0051620, 0.0124331, 0.0043750, 0.0155873, -0.0104252, 0.0077573
1: 0.0007900, 0.0028694, -0.0014151, 0.0027693, -0.0019793, 0.0042841
2: 0.0070961, 0.0115059, 0.0045490, 0.0119410, -0.0048065, 0.0069569
3: -0.0059520, -0.0016732, -0.0055484, 0.0006211, -0.0063297, 0.0035793
4: -0.0011252, 0.0024064, -0.0017621, 0.0019674, -0.0025979, 0.0035146
5: 0.0011525, 0.0050851, 0.0000408, 0.0055462, -0.0040752, 0.0049998
6: -0.0157374, -0.0021240, -0.0151007, -0.0002948, -0.0130130, 0.0116292
7: -0.0008056, 0.0178928, -0.0051018, 0.0156480, -0.0140523, 0.0200091
8: 0.9894507, 1.0020430, 0.9876956, 1.0007479, -0.0095873, 0.0118879
9: -0.0175375, -0.0059422, -0.0161021, -0.0037875, -0.0116407, 0.0085517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089690, upper bound: 0.0099772
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094844, upper bound: 0.0101385
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042934, 0.0147491, 0.0039643, 0.0170153, -0.0127219, 0.0107848
1: -0.0009510, 0.0028030, -0.0022902, 0.0031333, -0.0040843, 0.0050933
2: 0.0052260, 0.0119861, 0.0034363, 0.0121681, -0.0069421, 0.0085498
3: -0.0056895, 0.0002751, -0.0058473, 0.0016101, -0.0072996, 0.0061224
4: -0.0017928, 0.0021223, -0.0021629, 0.0019815, -0.0031026, 0.0041114
5: 0.0003567, 0.0055940, -0.0005120, 0.0057868, -0.0054301, 0.0061059
6: -0.0153989, -0.0001051, -0.0155540, 0.0006599, -0.0160588, 0.0137846
7: -0.0050875, 0.0164400, -0.0075009, 0.0157202, -0.0175780, 0.0233859
8: 0.9875138, 1.0012174, 0.9867798, 1.0009103, -0.0110844, 0.0144376
9: -0.0166085, -0.0036968, -0.0161483, -0.0023388, -0.0137683, 0.0102692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0099776
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0109344
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0051909, 0.0126148, 0.0042994, 0.0168721, -0.0116813, 0.0083154
1: 0.0006742, 0.0028695, -0.0020694, 0.0031105, -0.0024363, 0.0049389
2: 0.0069504, 0.0114900, 0.0035936, 0.0119828, -0.0050324, 0.0078964
3: -0.0059524, -0.0016005, -0.0059229, 0.0012800, -0.0070794, 0.0043224
4: -0.0011152, 0.0024069, -0.0019284, 0.0021083, -0.0029483, 0.0037457
5: 0.0010819, 0.0050682, -0.0004910, 0.0055904, -0.0045086, 0.0055593
6: -0.0158004, -0.0021910, -0.0158289, -0.0001191, -0.0138967, 0.0130486
7: -0.0008108, 0.0178954, -0.0062188, 0.0163685, -0.0159373, 0.0214306
8: 0.9895148, 1.0020636, 0.9875272, 1.0013047, -0.0108413, 0.0132903
9: -0.0175391, -0.0059709, -0.0165628, -0.0031354, -0.0125405, 0.0097037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089689, upper bound: 0.0101169
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094844, upper bound: 0.0102867
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043069, 0.0149960, 0.0038738, 0.0183693, -0.0140624, 0.0111222
1: -0.0010924, 0.0028031, -0.0029630, 0.0034375, -0.0045299, 0.0057661
2: 0.0050329, 0.0119787, 0.0024367, 0.0122182, -0.0071853, 0.0095420
3: -0.0056899, 0.0003817, -0.0061927, 0.0022989, -0.0079888, 0.0065745
4: -0.0017904, 0.0021227, -0.0023578, 0.0021227, -0.0034513, 0.0043333
5: 0.0002578, 0.0055861, -0.0010646, 0.0058398, -0.0055820, 0.0066507
6: -0.0154717, -0.0001363, -0.0162858, 0.0008703, -0.0163420, 0.0151517
7: -0.0051331, 0.0164419, -0.0087397, 0.0164422, -0.0194851, 0.0247432
8: 0.9875436, 1.0012391, 0.9860262, 1.0014645, -0.0123118, 0.0152129
9: -0.0166097, -0.0037011, -0.0166099, -0.0015764, -0.0146491, 0.0114194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0100440
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0111013
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0050710, 0.0134054, 0.0043750, 0.0155873, -0.0105163, 0.0090304
1: 0.0004384, 0.0029010, -0.0014151, 0.0027693, -0.0023309, 0.0043160
2: 0.0063864, 0.0115563, 0.0045490, 0.0119410, -0.0055546, 0.0070073
3: -0.0060771, -0.0013760, -0.0055484, 0.0006211, -0.0066683, 0.0040069
4: -0.0012025, 0.0025419, -0.0017621, 0.0019674, -0.0028677, 0.0038810
5: 0.0007254, 0.0051385, 0.0000408, 0.0055462, -0.0048205, 0.0050977
6: -0.0164873, -0.0019123, -0.0151007, -0.0002948, -0.0148952, 0.0125675
7: -0.0013347, 0.0185856, -0.0051018, 0.0156480, -0.0155413, 0.0218831
8: 0.9892474, 1.0026063, 0.9876956, 1.0007479, -0.0104875, 0.0133846
9: -0.0179805, -0.0056796, -0.0161021, -0.0037875, -0.0128390, 0.0094420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099292
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094328, upper bound: 0.0100987
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0041204, 0.0160361, 0.0039643, 0.0170153, -0.0128949, 0.0120718
1: -0.0013523, 0.0029077, -0.0022902, 0.0031333, -0.0044856, 0.0051979
2: 0.0043268, 0.0120818, 0.0034363, 0.0121681, -0.0078413, 0.0086455
3: -0.0058983, 0.0006193, -0.0058473, 0.0016101, -0.0075084, 0.0064666
4: -0.0019261, 0.0022926, -0.0021629, 0.0019815, -0.0034480, 0.0044361
5: -0.0002238, 0.0056953, -0.0005120, 0.0057868, -0.0060106, 0.0062073
6: -0.0164064, 0.0002970, -0.0155540, 0.0006599, -0.0170663, 0.0151940
7: -0.0058697, 0.0173108, -0.0075009, 0.0157202, -0.0193803, 0.0248117
8: 0.9871279, 1.0019611, 0.9867798, 1.0009103, -0.0125296, 0.0151813
9: -0.0171653, -0.0032406, -0.0161483, -0.0023388, -0.0148265, 0.0114213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0099451
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0109342
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0050966, 0.0135976, 0.0042994, 0.0168721, -0.0117755, 0.0092982
1: 0.0003309, 0.0029011, -0.0020694, 0.0031105, -0.0027795, 0.0049705
2: 0.0062399, 0.0115421, 0.0035936, 0.0119828, -0.0057429, 0.0079485
3: -0.0060774, -0.0013129, -0.0059229, 0.0012800, -0.0073574, 0.0046100
4: -0.0011933, 0.0025422, -0.0019284, 0.0021083, -0.0031858, 0.0041122
5: 0.0006552, 0.0051235, -0.0004910, 0.0055904, -0.0049352, 0.0056145
6: -0.0165488, -0.0019720, -0.0158289, -0.0001191, -0.0157795, 0.0138569
7: -0.0013306, 0.0185873, -0.0062188, 0.0163685, -0.0172520, 0.0233048
8: 0.9893047, 1.0026264, 0.9875272, 1.0013047, -0.0116349, 0.0147900
9: -0.0179816, -0.0057064, -0.0165628, -0.0031354, -0.0137389, 0.0104884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088092, upper bound: 0.0100452
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0102475
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0041470, 0.0163183, 0.0038738, 0.0183693, -0.0142222, 0.0124446
1: -0.0015003, 0.0029868, -0.0029630, 0.0034375, -0.0049378, 0.0059498
2: 0.0041110, 0.0120671, 0.0024367, 0.0122182, -0.0081072, 0.0096304
3: -0.0059614, 0.0007379, -0.0061927, 0.0022989, -0.0082604, 0.0069306
4: -0.0019298, 0.0022932, -0.0023578, 0.0021227, -0.0037676, 0.0046509
5: -0.0003335, 0.0056797, -0.0010646, 0.0058398, -0.0061733, 0.0067443
6: -0.0164860, 0.0002351, -0.0162858, 0.0008703, -0.0173563, 0.0165163
7: -0.0059246, 0.0173137, -0.0087397, 0.0164422, -0.0210997, 0.0260534
8: 0.9871873, 1.0019870, 0.9860262, 1.0014645, -0.0138322, 0.0159608
9: -0.0171672, -0.0032096, -0.0166099, -0.0015764, -0.0155908, 0.0124961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0100045
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0100046
time: 3.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0051620, 0.0124331, 0.0042476, 0.0169573, -0.0117953, 0.0081855
1: 0.0007900, 0.0028694, -0.0018247, 0.0030955, -0.0023055, 0.0046940
2: 0.0070961, 0.0115059, 0.0036091, 0.0120115, -0.0049154, 0.0078968
3: -0.0059520, -0.0016732, -0.0059126, 0.0009764, -0.0068204, 0.0042394
4: -0.0011252, 0.0024064, -0.0019158, 0.0021104, -0.0028521, 0.0038892
5: 0.0011525, 0.0050851, -0.0005750, 0.0056208, -0.0044683, 0.0056601
6: -0.0157374, -0.0021240, -0.0161023, 0.0000013, -0.0145091, 0.0132665
7: -0.0008056, 0.0178928, -0.0059396, 0.0163794, -0.0153524, 0.0218125
8: 0.9894507, 1.0020430, 0.9874116, 1.0013975, -0.0107532, 0.0138227
9: -0.0175375, -0.0059422, -0.0165698, -0.0032048, -0.0129574, 0.0093830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088822, upper bound: 0.0099675
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094587, upper bound: 0.0101668
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042934, 0.0147491, 0.0038115, 0.0184377, -0.0141442, 0.0109376
1: -0.0009510, 0.0028030, -0.0027047, 0.0034217, -0.0043727, 0.0055078
2: 0.0052260, 0.0119861, 0.0024783, 0.0122526, -0.0070266, 0.0095078
3: -0.0056895, 0.0002751, -0.0061816, 0.0019778, -0.0076673, 0.0064567
4: -0.0017928, 0.0021223, -0.0023283, 0.0021247, -0.0033747, 0.0043823
5: 0.0003567, 0.0055940, -0.0011526, 0.0058763, -0.0055196, 0.0067466
6: -0.0153989, -0.0001051, -0.0165944, 0.0010150, -0.0164139, 0.0155720
7: -0.0050875, 0.0164400, -0.0083657, 0.0164522, -0.0189691, 0.0246472
8: 0.9875138, 1.0012174, 0.9860114, 1.0015724, -0.0123507, 0.0152059
9: -0.0166085, -0.0036968, -0.0166164, -0.0017213, -0.0147332, 0.0111587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0099460
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0110123
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0051909, 0.0126148, 0.0041715, 0.0182756, -0.0130847, 0.0084433
1: 0.0006742, 0.0028695, -0.0024854, 0.0034020, -0.0027279, 0.0053549
2: 0.0069504, 0.0114900, 0.0026432, 0.0120535, -0.0051032, 0.0088468
3: -0.0059524, -0.0016005, -0.0062626, 0.0016475, -0.0075600, 0.0046620
4: -0.0011152, 0.0024069, -0.0020859, 0.0022553, -0.0031782, 0.0040837
5: 0.0010819, 0.0050682, -0.0011188, 0.0056654, -0.0045835, 0.0061871
6: -0.0158004, -0.0021910, -0.0168237, 0.0001782, -0.0152854, 0.0145835
7: -0.0008108, 0.0178954, -0.0070746, 0.0171203, -0.0171130, 0.0230885
8: 0.9895148, 1.0020636, 0.9869215, 1.0019569, -0.0118969, 0.0150973
9: -0.0175391, -0.0059709, -0.0170435, -0.0025356, -0.0137294, 0.0104555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088794, upper bound: 0.0100704
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094587, upper bound: 0.0103157
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043069, 0.0149960, 0.0037308, 0.0198020, -0.0154951, 0.0112652
1: -0.0010924, 0.0028031, -0.0033802, 0.0037018, -0.0047942, 0.0061833
2: 0.0050329, 0.0119787, 0.0014846, 0.0122972, -0.0072643, 0.0104941
3: -0.0056899, 0.0003817, -0.0065112, 0.0026757, -0.0083656, 0.0068929
4: -0.0017904, 0.0021227, -0.0025184, 0.0022702, -0.0036975, 0.0045813
5: 0.0002578, 0.0055861, -0.0017185, 0.0059235, -0.0056657, 0.0073046
6: -0.0154717, -0.0001363, -0.0173194, 0.0012025, -0.0166742, 0.0168254
7: -0.0051331, 0.0164419, -0.0095950, 0.0171964, -0.0207439, 0.0259196
8: 0.9875436, 1.0012391, 0.9851018, 1.0021290, -0.0134432, 0.0161373
9: -0.0166097, -0.0037011, -0.0170922, -0.0009678, -0.0155333, 0.0122244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0100158
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0111920
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0050710, 0.0134054, 0.0042476, 0.0169573, -0.0116194, 0.0091578
1: 0.0004384, 0.0029010, -0.0018247, 0.0030955, -0.0026572, 0.0047256
2: 0.0063864, 0.0115563, 0.0036091, 0.0120115, -0.0056250, 0.0078710
3: -0.0060771, -0.0013760, -0.0059126, 0.0009764, -0.0068676, 0.0044451
4: -0.0012025, 0.0025419, -0.0019158, 0.0021104, -0.0026764, 0.0037761
5: 0.0007254, 0.0051385, -0.0005750, 0.0056208, -0.0048654, 0.0054588
6: -0.0164873, -0.0019123, -0.0161023, 0.0000013, -0.0143347, 0.0122010
7: -0.0013347, 0.0185856, -0.0059396, 0.0163794, -0.0146574, 0.0214364
8: 0.9892474, 1.0026063, 0.9874116, 1.0013975, -0.0098654, 0.0134655
9: -0.0179805, -0.0056796, -0.0165698, -0.0032048, -0.0126147, 0.0088230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099201
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094328, upper bound: 0.0100948
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0041204, 0.0160361, 0.0038115, 0.0184377, -0.0143173, 0.0122246
1: -0.0013523, 0.0029077, -0.0027047, 0.0034217, -0.0047739, 0.0056124
2: 0.0043268, 0.0120818, 0.0024783, 0.0122526, -0.0079258, 0.0096035
3: -0.0058983, 0.0006193, -0.0061816, 0.0019778, -0.0078760, 0.0068009
4: -0.0019261, 0.0022926, -0.0023283, 0.0021247, -0.0032969, 0.0042976
5: -0.0002238, 0.0056953, -0.0011526, 0.0058763, -0.0061001, 0.0068479
6: -0.0164064, 0.0002970, -0.0165944, 0.0010150, -0.0174214, 0.0150004
7: -0.0058697, 0.0173108, -0.0083657, 0.0164522, -0.0187678, 0.0244051
8: 0.9871279, 1.0019611, 0.9860114, 1.0015724, -0.0120419, 0.0159497
9: -0.0171653, -0.0032406, -0.0166164, -0.0017213, -0.0144949, 0.0109414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0099177
time: 1.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0109310
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0050966, 0.0135976, 0.0041715, 0.0182756, -0.0129907, 0.0094261
1: 0.0003309, 0.0029011, -0.0024854, 0.0034020, -0.0030711, 0.0053865
2: 0.0062399, 0.0115421, 0.0026432, 0.0120535, -0.0058137, 0.0088548
3: -0.0060774, -0.0013129, -0.0062626, 0.0016475, -0.0076074, 0.0049497
4: -0.0011933, 0.0025422, -0.0020859, 0.0022553, -0.0030291, 0.0039968
5: 0.0006552, 0.0051235, -0.0011188, 0.0056654, -0.0050101, 0.0060582
6: -0.0165488, -0.0019720, -0.0168237, 0.0001782, -0.0152268, 0.0136092
7: -0.0013306, 0.0185873, -0.0070746, 0.0171203, -0.0165591, 0.0228027
8: 0.9893047, 1.0026264, 0.9869215, 1.0019569, -0.0111324, 0.0149026
9: -0.0179816, -0.0057064, -0.0170435, -0.0025356, -0.0134778, 0.0099831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088092, upper bound: 0.0100308
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0102454
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0041470, 0.0163183, 0.0037308, 0.0198020, -0.0156550, 0.0125875
1: -0.0015003, 0.0029868, -0.0033802, 0.0037018, -0.0052021, 0.0063670
2: 0.0041110, 0.0120671, 0.0014846, 0.0122972, -0.0081862, 0.0105825
3: -0.0059614, 0.0007379, -0.0065112, 0.0026757, -0.0086372, 0.0072490
4: -0.0019298, 0.0022932, -0.0025184, 0.0022702, -0.0036525, 0.0045209
5: -0.0003335, 0.0056797, -0.0017185, 0.0059235, -0.0062570, 0.0073983
6: -0.0164860, 0.0002351, -0.0173194, 0.0012025, -0.0176885, 0.0164271
7: -0.0059246, 0.0173137, -0.0095950, 0.0171964, -0.0206747, 0.0257777
8: 0.9871873, 1.0019870, 0.9851018, 1.0021290, -0.0134727, 0.0168852
9: -0.0171672, -0.0032096, -0.0170922, -0.0009678, -0.0153816, 0.0121348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0099866
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0111006
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0045211, 0.0156795, 0.0045672, 0.0139190, -0.0093979, 0.0111122
1: -0.0012390, 0.0028354, -0.0004031, 0.0028011, -0.0040401, 0.0032384
2: 0.0045785, 0.0118603, 0.0058596, 0.0118348, -0.0072562, 0.0060007
3: -0.0058175, 0.0003557, -0.0056819, -0.0003677, -0.0052610, 0.0060377
4: -0.0016588, 0.0022608, -0.0015749, 0.0021141, -0.0034284, 0.0033243
5: -0.0000789, 0.0054606, 0.0006963, 0.0054336, -0.0054465, 0.0047643
6: -0.0162484, -0.0006342, -0.0151193, -0.0007415, -0.0137775, 0.0133732
7: -0.0044738, 0.0171484, -0.0036548, 0.0163978, -0.0194796, 0.0184377
8: 0.9880214, 1.0018444, 0.9881242, 1.0011185, -0.0117802, 0.0118454
9: -0.0170615, -0.0041313, -0.0165816, -0.0044324, -0.0109735, 0.0113532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093299, upper bound: 0.0096227
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097779, upper bound: 0.0098949
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0035828, 0.0184948, 0.0041727, 0.0151975, -0.0116147, 0.0143222
1: -0.0031010, 0.0034380, -0.0012252, 0.0028043, -0.0059054, 0.0046632
2: 0.0023220, 0.0123790, 0.0048696, 0.0120529, -0.0097309, 0.0075094
3: -0.0060975, 0.0024716, -0.0056946, 0.0005637, -0.0066612, 0.0081662
4: -0.0025415, 0.0019933, -0.0018817, 0.0021278, -0.0041566, 0.0037117
5: -0.0010999, 0.0060102, 0.0001764, 0.0056647, -0.0067646, 0.0058338
6: -0.0160585, 0.0015465, -0.0155463, 0.0001756, -0.0160501, 0.0169961
7: -0.0096704, 0.0157804, -0.0056678, 0.0164681, -0.0237338, 0.0209231
8: 0.9853110, 1.0010893, 0.9872444, 1.0012723, -0.0159613, 0.0132656
9: -0.0161867, -0.0009692, -0.0166265, -0.0033973, -0.0122796, 0.0140611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0096240
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0107681
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0045211, 0.0156795, 0.0039404, 0.0189191, -0.0143980, 0.0117391
1: -0.0012390, 0.0028354, -0.0032734, 0.0034716, -0.0047106, 0.0061066
2: 0.0045785, 0.0118603, 0.0020039, 0.0121813, -0.0076028, 0.0098564
3: -0.0058175, 0.0003557, -0.0061135, 0.0025161, -0.0081778, 0.0064693
4: -0.0016588, 0.0022608, -0.0023431, 0.0019789, -0.0029655, 0.0039673
5: -0.0000789, 0.0054606, -0.0012514, 0.0058008, -0.0058796, 0.0067121
6: -0.0162484, -0.0006342, -0.0159810, 0.0007154, -0.0162507, 0.0139954
7: -0.0044738, 0.0171484, -0.0086955, 0.0157069, -0.0170724, 0.0229346
8: 0.9880214, 1.0018444, 0.9858667, 1.0010260, -0.0107050, 0.0155800
9: -0.0170615, -0.0041313, -0.0161398, -0.0015813, -0.0135154, 0.0098356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101148, upper bound: 0.0102926
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101379, upper bound: 0.0104308
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0035828, 0.0184948, 0.0035059, 0.0205211, -0.0169383, 0.0149889
1: -0.0031010, 0.0034380, -0.0041966, 0.0037446, -0.0068456, 0.0076346
2: 0.0023220, 0.0123790, 0.0007804, 0.0124216, -0.0100996, 0.0115987
3: -0.0060975, 0.0024716, -0.0063404, 0.0035638, -0.0096614, 0.0088120
4: -0.0025415, 0.0019933, -0.0027907, 0.0019931, -0.0036886, 0.0045089
5: -0.0010999, 0.0060102, -0.0018719, 0.0060553, -0.0071552, 0.0078821
6: -0.0160585, 0.0015465, -0.0164712, 0.0017253, -0.0177838, 0.0179948
7: -0.0096704, 0.0157804, -0.0112871, 0.0157795, -0.0213315, 0.0259742
8: 0.9853110, 1.0010893, 0.9839297, 1.0011929, -0.0150036, 0.0171596
9: -0.0161867, -0.0009692, -0.0161862, 0.0000541, -0.0154830, 0.0125265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110430, upper bound: 0.0108088
time: 1.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111541, upper bound: 0.0110357
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037109, 0.0169295, 0.0043214, 0.0143105, -0.0105996, 0.0126081
1: -0.0023319, 0.0031684, -0.0007441, 0.0028031, -0.0051350, 0.0039125
2: 0.0035096, 0.0123082, 0.0055447, 0.0119707, -0.0084611, 0.0067635
3: -0.0059779, 0.0017807, -0.0056898, 0.0000618, -0.0060396, 0.0074705
4: -0.0023258, 0.0021206, -0.0017514, 0.0021226, -0.0041704, 0.0037513
5: -0.0004919, 0.0059352, 0.0005263, 0.0055776, -0.0060695, 0.0054089
6: -0.0160370, 0.0012487, -0.0153054, -0.0001701, -0.0156306, 0.0162794
7: -0.0083471, 0.0164315, -0.0047352, 0.0164413, -0.0236992, 0.0208427
8: 0.9862149, 1.0013982, 0.9875761, 1.0011915, -0.0149766, 0.0133555
9: -0.0166031, -0.0018160, -0.0166094, -0.0038432, -0.0123879, 0.0139459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100124, upper bound: 0.0091777
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107057, upper bound: 0.0104220
time: 1.95 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0034901, 0.0186278, 0.0042941, 0.0147410, -0.0112509, 0.0143337
1: -0.0031318, 0.0035557, -0.0009962, 0.0028032, -0.0059350, 0.0045519
2: 0.0022706, 0.0124303, 0.0052036, 0.0119858, -0.0097152, 0.0072267
3: -0.0063847, 0.0026222, -0.0056903, 0.0002996, -0.0066843, 0.0083125
4: -0.0026231, 0.0022557, -0.0017888, 0.0021231, -0.0045497, 0.0039884
5: -0.0011968, 0.0060646, 0.0003617, 0.0055936, -0.0067904, 0.0057029
6: -0.0168557, 0.0017621, -0.0154229, -0.0001066, -0.0167491, 0.0171850
7: -0.0101193, 0.0171223, -0.0050743, 0.0164441, -0.0258656, 0.0221966
8: 0.9850622, 1.0019678, 0.9875151, 1.0012258, -0.0161636, 0.0142764
9: -0.0170448, -0.0006924, -0.0166111, -0.0037094, -0.0131746, 0.0153743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101489, upper bound: 0.0091777
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109069, upper bound: 0.0104581
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037109, 0.0169295, 0.0036484, 0.0194164, -0.0157054, 0.0132811
1: -0.0023319, 0.0031684, -0.0036785, 0.0035836, -0.0059154, 0.0068469
2: 0.0035096, 0.0123082, 0.0015962, 0.0123428, -0.0088332, 0.0107120
3: -0.0059779, 0.0017807, -0.0062078, 0.0030304, -0.0090083, 0.0079885
4: -0.0023258, 0.0021206, -0.0026024, 0.0019864, -0.0037524, 0.0045780
5: -0.0004919, 0.0059352, -0.0014209, 0.0059718, -0.0064637, 0.0073560
6: -0.0160370, 0.0012487, -0.0161810, 0.0013941, -0.0174311, 0.0170017
7: -0.0083471, 0.0164315, -0.0101826, 0.0157448, -0.0215432, 0.0261822
8: 0.9862149, 1.0013982, 0.9848713, 1.0011001, -0.0141492, 0.0165269
9: -0.0166031, -0.0018160, -0.0161640, -0.0006608, -0.0155855, 0.0125781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105658, upper bound: 0.0100805
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111522, upper bound: 0.0109772
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0034901, 0.0186278, 0.0036046, 0.0199672, -0.0164772, 0.0150232
1: -0.0031318, 0.0035557, -0.0039501, 0.0036653, -0.0067971, 0.0075059
2: 0.0022706, 0.0124303, 0.0011714, 0.0123670, -0.0100964, 0.0112589
3: -0.0063847, 0.0026222, -0.0062741, 0.0032979, -0.0096826, 0.0088963
4: -0.0026231, 0.0022557, -0.0026884, 0.0019884, -0.0041195, 0.0048696
5: -0.0011968, 0.0060646, -0.0016370, 0.0059975, -0.0071943, 0.0077016
6: -0.0168557, 0.0017621, -0.0163200, 0.0014959, -0.0183516, 0.0180821
7: -0.0101193, 0.0171223, -0.0107075, 0.0157553, -0.0236436, 0.0277605
8: 0.9850622, 1.0019678, 0.9844313, 1.0011433, -0.0160812, 0.0175365
9: -0.0170448, -0.0006924, -0.0161707, -0.0003254, -0.0166063, 0.0139636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107497, upper bound: 0.0100940
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113995, upper bound: 0.0110611
time: 1.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0045132, 0.0157040, 0.0044383, 0.0151161, -0.0106028, 0.0112657
1: -0.0012506, 0.0028353, -0.0007646, 0.0028409, -0.0040915, 0.0035999
2: 0.0045601, 0.0118646, 0.0050280, 0.0119060, -0.0073459, 0.0068366
3: -0.0058172, 0.0003665, -0.0058393, -0.0000574, -0.0057075, 0.0062058
4: -0.0016611, 0.0022605, -0.0016720, 0.0022845, -0.0037949, 0.0034430
5: -0.0000893, 0.0054652, 0.0001554, 0.0055091, -0.0055341, 0.0053098
6: -0.0162576, -0.0006160, -0.0160995, -0.0004419, -0.0141061, 0.0153098
7: -0.0044625, 0.0171466, -0.0042942, 0.0172693, -0.0213271, 0.0192827
8: 0.9880038, 1.0018462, 0.9878368, 1.0018487, -0.0132876, 0.0121137
9: -0.0170604, -0.0041245, -0.0171388, -0.0041058, -0.0113799, 0.0125489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093299, upper bound: 0.0096463
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097779, upper bound: 0.0099553
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0035678, 0.0185202, 0.0040014, 0.0165540, -0.0129862, 0.0145188
1: -0.0031132, 0.0034430, -0.0016524, 0.0030600, -0.0061732, 0.0050954
2: 0.0023032, 0.0123873, 0.0039178, 0.0121476, -0.0098444, 0.0084695
3: -0.0061012, 0.0024821, -0.0060231, 0.0009293, -0.0070305, 0.0085051
4: -0.0025506, 0.0019929, -0.0020425, 0.0022981, -0.0045614, 0.0038453
5: -0.0011096, 0.0060191, -0.0004267, 0.0057650, -0.0068746, 0.0064458
6: -0.0160645, 0.0015815, -0.0165717, 0.0005737, -0.0165541, 0.0181533
7: -0.0097115, 0.0157783, -0.0065548, 0.0173390, -0.0257813, 0.0217359
8: 0.9852686, 1.0010895, 0.9868625, 1.0020244, -0.0167558, 0.0140968
9: -0.0161854, -0.0009414, -0.0171834, -0.0028204, -0.0127995, 0.0153833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0096042
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0108646
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0045132, 0.0157040, 0.0038300, 0.0203137, -0.0158005, 0.0118740
1: -0.0012506, 0.0028353, -0.0036697, 0.0037052, -0.0049558, 0.0065050
2: 0.0045601, 0.0118646, 0.0010826, 0.0122423, -0.0076823, 0.0107820
3: -0.0058172, 0.0003665, -0.0064046, 0.0028754, -0.0086069, 0.0067711
4: -0.0016611, 0.0022605, -0.0024890, 0.0021223, -0.0033638, 0.0041149
5: -0.0000893, 0.0054652, -0.0019108, 0.0058654, -0.0059547, 0.0073760
6: -0.0162576, -0.0006160, -0.0170519, 0.0009719, -0.0168713, 0.0162175
7: -0.0044625, 0.0171466, -0.0094982, 0.0164402, -0.0190231, 0.0237657
8: 0.9880038, 1.0018462, 0.9850432, 1.0016963, -0.0124390, 0.0167215
9: -0.0170604, -0.0041245, -0.0166087, -0.0010125, -0.0141121, 0.0111300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101148, upper bound: 0.0103795
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101379, upper bound: 0.0105266
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0035678, 0.0185202, 0.0033877, 0.0218350, -0.0182672, 0.0151325
1: -0.0031132, 0.0034430, -0.0045588, 0.0039553, -0.0070686, 0.0080019
2: 0.0023032, 0.0123873, -0.0000822, 0.0124869, -0.0101837, 0.0124695
3: -0.0061012, 0.0024821, -0.0066135, 0.0038996, -0.0100008, 0.0090956
4: -0.0025506, 0.0019929, -0.0029352, 0.0021367, -0.0041159, 0.0046535
5: -0.0011096, 0.0060191, -0.0025118, 0.0061245, -0.0072341, 0.0085309
6: -0.0160645, 0.0015815, -0.0175581, 0.0020001, -0.0180645, 0.0191396
7: -0.0097115, 0.0157783, -0.0120642, 0.0165137, -0.0234226, 0.0267879
8: 0.9852686, 1.0010895, 0.9831065, 1.0018685, -0.0165999, 0.0179830
9: -0.0161854, -0.0009414, -0.0166556, 0.0006057, -0.0160628, 0.0139057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110430, upper bound: 0.0108983
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111541, upper bound: 0.0111344
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036941, 0.0169473, 0.0041771, 0.0155886, -0.0118945, 0.0127702
1: -0.0023407, 0.0031728, -0.0011397, 0.0028429, -0.0051836, 0.0043125
2: 0.0034962, 0.0123175, 0.0046511, 0.0120504, -0.0085543, 0.0076664
3: -0.0059810, 0.0017910, -0.0058472, 0.0004018, -0.0063828, 0.0076382
4: -0.0023376, 0.0021201, -0.0018627, 0.0022930, -0.0045269, 0.0039617
5: -0.0004987, 0.0059451, -0.0000417, 0.0056621, -0.0061608, 0.0059868
6: -0.0160411, 0.0012879, -0.0163081, 0.0001652, -0.0162064, 0.0175960
7: -0.0084240, 0.0164288, -0.0054440, 0.0173127, -0.0254937, 0.0218727
8: 0.9861773, 1.0013976, 0.9872543, 1.0019318, -0.0157545, 0.0140165
9: -0.0166013, -0.0017740, -0.0171665, -0.0034685, -0.0130939, 0.0151086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100124, upper bound: 0.0091563
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107056, upper bound: 0.0105211
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0034819, 0.0186443, 0.0041439, 0.0160348, -0.0125529, 0.0145004
1: -0.0031408, 0.0035593, -0.0013940, 0.0029196, -0.0060604, 0.0049533
2: 0.0022576, 0.0124348, 0.0043082, 0.0120688, -0.0098112, 0.0081266
3: -0.0063871, 0.0026322, -0.0059083, 0.0006430, -0.0070301, 0.0085406
4: -0.0026283, 0.0022552, -0.0019108, 0.0022934, -0.0049217, 0.0041660
5: -0.0012022, 0.0060694, -0.0002077, 0.0056816, -0.0068838, 0.0062771
6: -0.0168542, 0.0017811, -0.0164284, 0.0002424, -0.0170966, 0.0182095
7: -0.0101600, 0.0171195, -0.0057817, 0.0173149, -0.0274749, 0.0229012
8: 0.9850340, 1.0019648, 0.9871802, 1.0019696, -0.0169355, 0.0147846
9: -0.0170430, -0.0006730, -0.0171680, -0.0032903, -0.0137527, 0.0164950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101489, upper bound: 0.0091563
time: 1.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109069, upper bound: 0.0105556
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036941, 0.0169473, 0.0035299, 0.0207467, -0.0170526, 0.0134174
1: -0.0023407, 0.0031728, -0.0040324, 0.0038069, -0.0061476, 0.0072052
2: 0.0034962, 0.0123175, 0.0007186, 0.0124083, -0.0089121, 0.0115989
3: -0.0059810, 0.0017910, -0.0064908, 0.0033607, -0.0093416, 0.0082818
4: -0.0023376, 0.0021201, -0.0027520, 0.0021298, -0.0041265, 0.0047232
5: -0.0004987, 0.0059451, -0.0020595, 0.0060412, -0.0065399, 0.0080046
6: -0.0160411, 0.0012879, -0.0172431, 0.0016694, -0.0177105, 0.0185310
7: -0.0084240, 0.0164288, -0.0110049, 0.0164786, -0.0233944, 0.0269825
8: 0.9861773, 1.0013976, 0.9840369, 1.0017693, -0.0155920, 0.0173607
9: -0.0166013, -0.0017740, -0.0166332, -0.0000911, -0.0161665, 0.0137881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105658, upper bound: 0.0100940
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111522, upper bound: 0.0110810
time: 1.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0034819, 0.0186443, 0.0034747, 0.0212210, -0.0177391, 0.0151696
1: -0.0031408, 0.0035593, -0.0042753, 0.0038729, -0.0070137, 0.0078346
2: 0.0022576, 0.0124348, 0.0003669, 0.0124388, -0.0101813, 0.0120679
3: -0.0063871, 0.0026322, -0.0065446, 0.0036109, -0.0099981, 0.0091768
4: -0.0026283, 0.0022552, -0.0028308, 0.0021319, -0.0045542, 0.0050053
5: -0.0012022, 0.0060694, -0.0022472, 0.0060736, -0.0072758, 0.0083166
6: -0.0168542, 0.0017811, -0.0173750, 0.0017979, -0.0186521, 0.0191561
7: -0.0101600, 0.0171195, -0.0114526, 0.0164891, -0.0257449, 0.0285210
8: 0.9850340, 1.0019648, 0.9836360, 1.0018106, -0.0167765, 0.0183288
9: -0.0170430, -0.0006730, -0.0166399, 0.0002073, -0.0171489, 0.0153633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0107497, upper bound: 0.0101132
time: 1.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113995, upper bound: 0.0111632
time: 1.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.05 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0089690, upper bound: 0.0099772
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094844, upper bound: 0.0101385
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0099776
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0109344
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0089689, upper bound: 0.0101169
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094844, upper bound: 0.0102867
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0100440
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101124, upper bound: 0.0111013
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099292
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094328, upper bound: 0.0100987
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0099451
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0109342
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0088092, upper bound: 0.0100452
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0102475
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0100045
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0100046
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0088822, upper bound: 0.0099675
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094587, upper bound: 0.0101668
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0099460
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0110123
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0088794, upper bound: 0.0100704
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094587, upper bound: 0.0103157
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0100158
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0100677, upper bound: 0.0111920
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099201
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094328, upper bound: 0.0100948
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0099177
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0109310
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0088092, upper bound: 0.0100308
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0102454
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0099866
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101469, upper bound: 0.0111006
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0093299, upper bound: 0.0096227
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0097779, upper bound: 0.0098949
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0096240
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0107681
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101148, upper bound: 0.0102926
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101379, upper bound: 0.0104308
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0110430, upper bound: 0.0108088
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0111541, upper bound: 0.0110357
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0100124, upper bound: 0.0091777
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0107057, upper bound: 0.0104220
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101489, upper bound: 0.0091777
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0109069, upper bound: 0.0104581
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0105658, upper bound: 0.0100805
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0111522, upper bound: 0.0109772
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0107497, upper bound: 0.0100940
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0113995, upper bound: 0.0110611
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0093299, upper bound: 0.0096463
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0097779, upper bound: 0.0099553
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0096042
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0103434, upper bound: 0.0108646
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101148, upper bound: 0.0103795
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101379, upper bound: 0.0105266
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0110430, upper bound: 0.0108983
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0111541, upper bound: 0.0111344
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0100124, upper bound: 0.0091563
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0107056, upper bound: 0.0105211
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0101489, upper bound: 0.0091563
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0109069, upper bound: 0.0105556
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0105658, upper bound: 0.0100940
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0111522, upper bound: 0.0110810
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0107497, upper bound: 0.0101132
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.05
Output dim: 8, lower bound: -0.0113995, upper bound: 0.0111632

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0051840, 0.0121419, 0.0042775, 0.0154204, -0.0102364, 0.0074021
1: 0.0008711, 0.0028373, -0.0014571, 0.0026975, -0.0018225, 0.0042818
2: 0.0072743, 0.0114938, 0.0046048, 0.0119949, -0.0045855, 0.0068889
3: -0.0058252, -0.0017545, -0.0052719, 0.0006756, -0.0062193, 0.0031777
4: -0.0011057, 0.0022691, -0.0018216, 0.0016702, -0.0022491, 0.0033708
5: 0.0013080, 0.0050723, 0.0001793, 0.0056033, -0.0038833, 0.0047863
6: -0.0151981, -0.0021750, -0.0140848, -0.0000681, -0.0124269, 0.0104206
7: -0.0006655, 0.0171909, -0.0054066, 0.0141282, -0.0122173, 0.0192438
8: 0.9894994, 1.0015410, 0.9874781, 0.9997004, -0.0083736, 0.0113445
9: -0.0170886, -0.0060091, -0.0151303, -0.0035935, -0.0111698, 0.0074074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089688, upper bound: 0.0099772
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089688, upper bound: 0.0099772
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0051653, 0.0123861, 0.0043864, 0.0154472, -0.0102392, 0.0076851
1: 0.0008000, 0.0028642, -0.0013727, 0.0027481, -0.0019210, 0.0042364
2: 0.0071245, 0.0115041, 0.0046294, 0.0119347, -0.0047568, 0.0068747
3: -0.0059314, -0.0016838, -0.0054722, 0.0005743, -0.0062670, 0.0033397
4: -0.0011224, 0.0023841, -0.0017506, 0.0018870, -0.0023528, 0.0034909
5: 0.0011774, 0.0050832, 0.0001253, 0.0055395, -0.0040394, 0.0048114
6: -0.0156472, -0.0021316, -0.0147950, -0.0003211, -0.0129220, 0.0107348
7: -0.0007858, 0.0177790, -0.0050204, 0.0152367, -0.0127756, 0.0198559
8: 0.9894578, 1.0019586, 0.9877210, 1.0004575, -0.0087156, 0.0118055
9: -0.0174647, -0.0059517, -0.0158391, -0.0038277, -0.0115602, 0.0077475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094812, upper bound: 0.0101385
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094812, upper bound: 0.0101385
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042934, 0.0147491, 0.0049685, 0.0139764, -0.0096829, 0.0097805
1: -0.0009510, 0.0028030, -0.0002270, 0.0028312, -0.0037822, 0.0030301
2: 0.0052260, 0.0119861, 0.0058813, 0.0116129, -0.0063869, 0.0061049
3: -0.0056895, 0.0002751, -0.0058010, -0.0007184, -0.0047536, 0.0060760
4: -0.0017928, 0.0021223, -0.0013131, 0.0022430, -0.0037115, 0.0029132
5: 0.0003567, 0.0055940, 0.0005991, 0.0051985, -0.0048418, 0.0049949
6: -0.0153989, -0.0001051, -0.0156403, -0.0016742, -0.0121258, 0.0144749
7: -0.0050875, 0.0164400, -0.0022230, 0.0170570, -0.0207504, 0.0162243
8: 0.9875138, 1.0012174, 0.9890190, 1.0016162, -0.0128183, 0.0103577
9: -0.0166085, -0.0036968, -0.0170030, -0.0052952, -0.0096235, 0.0122665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0094294
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0098090
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042934, 0.0147491, 0.0040833, 0.0164865, -0.0121931, 0.0106657
1: -0.0009510, 0.0028030, -0.0019937, 0.0030070, -0.0039579, 0.0047738
2: 0.0052260, 0.0119861, 0.0038501, 0.0121023, -0.0068763, 0.0081361
3: -0.0056895, 0.0002751, -0.0057431, 0.0012946, -0.0066675, 0.0058894
4: -0.0017928, 0.0021223, -0.0020429, 0.0019760, -0.0030968, 0.0034043
5: 0.0003567, 0.0055940, -0.0003170, 0.0057170, -0.0053603, 0.0059110
6: -0.0153989, -0.0001051, -0.0153942, 0.0003832, -0.0138483, 0.0135048
7: -0.0050875, 0.0164400, -0.0067994, 0.0156918, -0.0175482, 0.0196205
8: 0.9875138, 1.0012174, 0.9870453, 1.0008519, -0.0110052, 0.0122007
9: -0.0166085, -0.0036968, -0.0161301, -0.0027822, -0.0113858, 0.0102502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0104348
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0106230
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0052132, 0.0123276, 0.0041980, 0.0165956, -0.0113824, 0.0081296
1: 0.0007545, 0.0028374, -0.0020439, 0.0029868, -0.0022323, 0.0048814
2: 0.0071251, 0.0114776, 0.0037381, 0.0120389, -0.0049137, 0.0077396
3: -0.0058256, -0.0016814, -0.0056010, 0.0012797, -0.0069013, 0.0039196
4: -0.0010954, 0.0022696, -0.0019765, 0.0018053, -0.0025768, 0.0035805
5: 0.0012369, 0.0050552, -0.0003017, 0.0056498, -0.0043380, 0.0053247
6: -0.0152630, -0.0022429, -0.0147520, 0.0001166, -0.0132526, 0.0117134
7: -0.0006704, 0.0171932, -0.0064516, 0.0148191, -0.0139845, 0.0205751
8: 0.9895646, 1.0015625, 0.9873009, 1.0002345, -0.0095325, 0.0126088
9: -0.0170901, -0.0060387, -0.0155721, -0.0029895, -0.0119832, 0.0084851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0085249, upper bound: 0.0097266
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0084745, upper bound: 0.0097266
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0051942, 0.0125680, 0.0043097, 0.0167444, -0.0115502, 0.0082583
1: 0.0006841, 0.0028643, -0.0020319, 0.0030646, -0.0023805, 0.0048962
2: 0.0069787, 0.0114881, 0.0036661, 0.0119771, -0.0049984, 0.0078220
3: -0.0059319, -0.0016112, -0.0058248, 0.0012381, -0.0070223, 0.0042054
4: -0.0011124, 0.0023847, -0.0019138, 0.0020248, -0.0026865, 0.0037168
5: 0.0011070, 0.0050663, -0.0004104, 0.0055844, -0.0044774, 0.0054123
6: -0.0157104, -0.0021987, -0.0155153, -0.0001430, -0.0137693, 0.0121021
7: -0.0007909, 0.0177816, -0.0061334, 0.0159413, -0.0145749, 0.0212687
8: 0.9895222, 1.0019795, 0.9875501, 1.0010095, -0.0099158, 0.0131347
9: -0.0174664, -0.0059807, -0.0162896, -0.0031904, -0.0124360, 0.0088463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091330, upper bound: 0.0099709
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091330, upper bound: 0.0102868
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043069, 0.0149960, 0.0049226, 0.0150924, -0.0107855, 0.0100734
1: -0.0010924, 0.0028031, -0.0008206, 0.0028585, -0.0039509, 0.0036237
2: 0.0050329, 0.0119787, 0.0050556, 0.0116383, -0.0066054, 0.0069231
3: -0.0056899, 0.0003817, -0.0059090, -0.0001481, -0.0054445, 0.0062908
4: -0.0017904, 0.0021227, -0.0013895, 0.0023599, -0.0040427, 0.0030461
5: 0.0002578, 0.0055861, 0.0001336, 0.0052254, -0.0049676, 0.0054525
6: -0.0154717, -0.0001363, -0.0163067, -0.0015675, -0.0124917, 0.0158957
7: -0.0051331, 0.0164419, -0.0029409, 0.0176551, -0.0225670, 0.0173070
8: 0.9875436, 1.0012391, 0.9889166, 1.0021025, -0.0140268, 0.0106486
9: -0.0166097, -0.0037011, -0.0173855, -0.0050220, -0.0100875, 0.0133591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0094824
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0098770
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043069, 0.0149960, 0.0039912, 0.0178512, -0.0135443, 0.0110048
1: -0.0010924, 0.0028031, -0.0026734, 0.0033313, -0.0044237, 0.0054762
2: 0.0050329, 0.0119787, 0.0028369, 0.0121533, -0.0071204, 0.0091418
3: -0.0056899, 0.0003817, -0.0061045, 0.0019917, -0.0074339, 0.0064862
4: -0.0017904, 0.0021227, -0.0022377, 0.0021172, -0.0034457, 0.0036340
5: 0.0002578, 0.0055861, -0.0008729, 0.0057710, -0.0055132, 0.0064590
6: -0.0154717, -0.0001363, -0.0161339, 0.0005974, -0.0147965, 0.0148932
7: -0.0051331, 0.0164419, -0.0080399, 0.0164138, -0.0194563, 0.0210326
8: 0.9875436, 1.0012391, 0.9865844, 1.0014093, -0.0122402, 0.0137134
9: -0.0166097, -0.0037011, -0.0165918, -0.0020176, -0.0122882, 0.0114010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0105491
time: 1.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0107282
time: 1.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0050930, 0.0131237, 0.0042775, 0.0154204, -0.0103274, 0.0088463
1: 0.0005196, 0.0028685, -0.0014571, 0.0026975, -0.0021779, 0.0043256
2: 0.0065618, 0.0115441, 0.0046048, 0.0119949, -0.0054331, 0.0069392
3: -0.0059487, -0.0014585, -0.0052719, 0.0006756, -0.0065853, 0.0036034
4: -0.0011826, 0.0024028, -0.0018216, 0.0016702, -0.0025169, 0.0037671
5: 0.0008785, 0.0051256, 0.0001793, 0.0056033, -0.0046524, 0.0049463
6: -0.0159603, -0.0019635, -0.0140848, -0.0000681, -0.0143867, 0.0113515
7: -0.0011922, 0.0178745, -0.0054066, 0.0141282, -0.0136945, 0.0212702
8: 0.9892966, 1.0021054, 0.9874781, 0.9997004, -0.0092667, 0.0129357
9: -0.0175258, -0.0057476, -0.0151303, -0.0035935, -0.0124655, 0.0082910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099292
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099292
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0050744, 0.0133576, 0.0043864, 0.0154472, -0.0103729, 0.0089712
1: 0.0004485, 0.0028958, -0.0013727, 0.0027481, -0.0022840, 0.0042686
2: 0.0064144, 0.0115544, 0.0046294, 0.0119347, -0.0055204, 0.0069250
3: -0.0060567, -0.0013870, -0.0054722, 0.0005743, -0.0065983, 0.0037814
4: -0.0011995, 0.0025198, -0.0017506, 0.0018870, -0.0026565, 0.0038495
5: 0.0007519, 0.0051365, 0.0001253, 0.0055395, -0.0047775, 0.0050112
6: -0.0163991, -0.0019202, -0.0147950, -0.0003211, -0.0147758, 0.0118009
7: -0.0013139, 0.0184725, -0.0050204, 0.0152367, -0.0144395, 0.0216896
8: 0.9892550, 1.0025264, 0.9877210, 1.0004575, -0.0097384, 0.0132720
9: -0.0179082, -0.0056897, -0.0158391, -0.0038277, -0.0127327, 0.0087494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100986
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100987
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041204, 0.0160361, 0.0049685, 0.0139764, -0.0098560, 0.0110676
1: -0.0013523, 0.0029077, -0.0002270, 0.0028312, -0.0041835, 0.0031347
2: 0.0043268, 0.0120818, 0.0058813, 0.0116129, -0.0072861, 0.0062005
3: -0.0058983, 0.0006193, -0.0058010, -0.0007184, -0.0051484, 0.0064203
4: -0.0019261, 0.0022926, -0.0013131, 0.0022430, -0.0040217, 0.0032332
5: -0.0002238, 0.0056953, 0.0005991, 0.0051985, -0.0054223, 0.0050963
6: -0.0164064, 0.0002970, -0.0156403, -0.0016742, -0.0139286, 0.0156942
7: -0.0058697, 0.0173108, -0.0022230, 0.0170570, -0.0224134, 0.0178607
8: 0.9871279, 1.0019611, 0.9890190, 1.0016162, -0.0140732, 0.0116967
9: -0.0171653, -0.0032406, -0.0170030, -0.0052952, -0.0106699, 0.0133057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0093684
time: 1.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0097751
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041204, 0.0160361, 0.0040833, 0.0164865, -0.0123661, 0.0119527
1: -0.0013523, 0.0029077, -0.0019937, 0.0030070, -0.0043592, 0.0049014
2: 0.0043268, 0.0120818, 0.0038501, 0.0121023, -0.0077755, 0.0082317
3: -0.0058983, 0.0006193, -0.0057431, 0.0012946, -0.0071162, 0.0063624
4: -0.0019261, 0.0022926, -0.0020429, 0.0019760, -0.0034402, 0.0037772
5: -0.0002238, 0.0056953, -0.0003170, 0.0057170, -0.0059408, 0.0060123
6: -0.0164064, 0.0002970, -0.0153942, 0.0003832, -0.0158842, 0.0148954
7: -0.0058697, 0.0173108, -0.0067994, 0.0156918, -0.0193402, 0.0215271
8: 0.9871279, 1.0019611, 0.9870453, 1.0008519, -0.0124172, 0.0137885
9: -0.0171653, -0.0032406, -0.0161301, -0.0027822, -0.0126050, 0.0113956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0104346
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0106230
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0051189, 0.0133237, 0.0041980, 0.0165956, -0.0114767, 0.0091257
1: 0.0004091, 0.0028686, -0.0020439, 0.0029868, -0.0025777, 0.0049125
2: 0.0064097, 0.0115298, 0.0037381, 0.0120389, -0.0056292, 0.0077917
3: -0.0059490, -0.0013931, -0.0056010, 0.0012797, -0.0072287, 0.0042079
4: -0.0011734, 0.0024032, -0.0019765, 0.0018053, -0.0028125, 0.0039767
5: 0.0008047, 0.0051104, -0.0003017, 0.0056498, -0.0048451, 0.0054121
6: -0.0160246, -0.0020237, -0.0147520, 0.0001166, -0.0152143, 0.0125352
7: -0.0011893, 0.0178763, -0.0064516, 0.0148191, -0.0152920, 0.0226015
8: 0.9893544, 1.0021268, 0.9873009, 1.0002345, -0.0103209, 0.0142031
9: -0.0175269, -0.0057744, -0.0155721, -0.0029895, -0.0132789, 0.0092639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0083827, upper bound: 0.0096443
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0083321, upper bound: 0.0096443
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0051001, 0.0135511, 0.0043097, 0.0167444, -0.0116443, 0.0092414
1: 0.0003407, 0.0028959, -0.0020319, 0.0030646, -0.0027239, 0.0049278
2: 0.0062669, 0.0115401, 0.0036661, 0.0119771, -0.0057102, 0.0078740
3: -0.0060570, -0.0013237, -0.0058248, 0.0012381, -0.0072951, 0.0045011
4: -0.0011903, 0.0025201, -0.0019138, 0.0020248, -0.0029625, 0.0040754
5: 0.0006814, 0.0051214, -0.0004104, 0.0055844, -0.0049030, 0.0055319
6: -0.0164607, -0.0019801, -0.0155153, -0.0001430, -0.0156250, 0.0130793
7: -0.0013098, 0.0184742, -0.0061334, 0.0159413, -0.0160793, 0.0231024
8: 0.9893125, 1.0025464, 0.9875501, 1.0010095, -0.0108532, 0.0146043
9: -0.0179093, -0.0057166, -0.0162896, -0.0031904, -0.0136085, 0.0097558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0098701
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0102475
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041470, 0.0163183, 0.0049226, 0.0150924, -0.0109453, 0.0113957
1: -0.0015003, 0.0029868, -0.0008206, 0.0028585, -0.0043588, 0.0038074
2: 0.0041110, 0.0120671, 0.0050556, 0.0116383, -0.0075273, 0.0070115
3: -0.0059614, 0.0007379, -0.0059090, -0.0001481, -0.0058133, 0.0066469
4: -0.0019298, 0.0022932, -0.0013895, 0.0023599, -0.0042897, 0.0033664
5: -0.0003335, 0.0056797, 0.0001336, 0.0052254, -0.0055588, 0.0055461
6: -0.0164860, 0.0002351, -0.0163067, -0.0015675, -0.0143084, 0.0165418
7: -0.0059246, 0.0173137, -0.0029409, 0.0176551, -0.0235796, 0.0189447
8: 0.9871873, 1.0019870, 0.9889166, 1.0021025, -0.0149152, 0.0119939
9: -0.0171672, -0.0032096, -0.0173855, -0.0050220, -0.0111347, 0.0141758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0093942
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0098378
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041470, 0.0163183, 0.0039912, 0.0178512, -0.0137041, 0.0123272
1: -0.0015003, 0.0029868, -0.0026734, 0.0033313, -0.0048316, 0.0056602
2: 0.0041110, 0.0120671, 0.0028369, 0.0121533, -0.0080423, 0.0092302
3: -0.0059614, 0.0007379, -0.0061045, 0.0019917, -0.0079531, 0.0068423
4: -0.0019298, 0.0022932, -0.0022377, 0.0021172, -0.0037608, 0.0040072
5: -0.0003335, 0.0056797, -0.0008729, 0.0057710, -0.0061045, 0.0065527
6: -0.0164860, 0.0002351, -0.0161339, 0.0005974, -0.0168455, 0.0162362
7: -0.0059246, 0.0173137, -0.0080399, 0.0164138, -0.0210651, 0.0229408
8: 0.9871873, 1.0019870, 0.9865844, 1.0014093, -0.0137262, 0.0153078
9: -0.0171672, -0.0032096, -0.0165918, -0.0020176, -0.0135084, 0.0124740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0105467
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0107280
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0051840, 0.0121419, 0.0041390, 0.0166825, -0.0114985, 0.0080029
1: 0.0008711, 0.0028373, -0.0017932, 0.0029754, -0.0021043, 0.0046305
2: 0.0072743, 0.0114938, 0.0037533, 0.0120715, -0.0047972, 0.0077405
3: -0.0058252, -0.0017545, -0.0056028, 0.0009709, -0.0066705, 0.0038483
4: -0.0011057, 0.0022691, -0.0019645, 0.0018200, -0.0025201, 0.0037653
5: 0.0013080, 0.0050723, -0.0003954, 0.0056844, -0.0043763, 0.0054677
6: -0.0151981, -0.0021750, -0.0151099, 0.0002537, -0.0140452, 0.0120868
7: -0.0006655, 0.0171909, -0.0061928, 0.0148939, -0.0136027, 0.0211744
8: 0.9894994, 1.0015410, 0.9871694, 1.0003841, -0.0095710, 0.0133076
9: -0.0170886, -0.0060091, -0.0156199, -0.0030566, -0.0125332, 0.0082932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088822, upper bound: 0.0099675
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088822, upper bound: 0.0099675
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0051653, 0.0123861, 0.0042629, 0.0167824, -0.0116171, 0.0081232
1: 0.0008000, 0.0028642, -0.0017595, 0.0030369, -0.0022369, 0.0046237
2: 0.0071245, 0.0115041, 0.0037236, 0.0120030, -0.0048785, 0.0077805
3: -0.0059314, -0.0016838, -0.0058025, 0.0009057, -0.0067301, 0.0040450
4: -0.0011224, 0.0023841, -0.0018937, 0.0020244, -0.0026244, 0.0038493
5: 0.0011774, 0.0050832, -0.0004754, 0.0056118, -0.0044344, 0.0055438
6: -0.0156472, -0.0021316, -0.0157681, -0.0000342, -0.0143576, 0.0124167
7: -0.0007858, 0.0177790, -0.0058033, 0.0159391, -0.0141647, 0.0215817
8: 0.9894578, 1.0019586, 0.9874456, 1.0010855, -0.0099239, 0.0136201
9: -0.0174647, -0.0059517, -0.0162883, -0.0032888, -0.0128123, 0.0086358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094567, upper bound: 0.0101668
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094567, upper bound: 0.0101668
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042934, 0.0147491, 0.0048824, 0.0151353, -0.0108418, 0.0098667
1: -0.0009510, 0.0028030, -0.0005854, 0.0028602, -0.0038112, 0.0033884
2: 0.0052260, 0.0119861, 0.0050649, 0.0116605, -0.0064345, 0.0069212
3: -0.0056895, 0.0002751, -0.0059158, -0.0004137, -0.0051500, 0.0061908
4: -0.0017928, 0.0021223, -0.0013814, 0.0023672, -0.0039766, 0.0031366
5: 0.0003567, 0.0055940, 0.0000947, 0.0052490, -0.0048923, 0.0054992
6: -0.0153989, -0.0001051, -0.0164517, -0.0014740, -0.0129210, 0.0160961
7: -0.0050875, 0.0164400, -0.0027011, 0.0176924, -0.0221062, 0.0174558
8: 0.9875138, 1.0012174, 0.9888270, 1.0021696, -0.0139974, 0.0111207
9: -0.0166085, -0.0036968, -0.0174093, -0.0050631, -0.0103593, 0.0131334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096227, upper bound: 0.0093299
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098950, upper bound: 0.0097779
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042934, 0.0147491, 0.0039325, 0.0178678, -0.0135743, 0.0108166
1: -0.0009510, 0.0028030, -0.0023953, 0.0033033, -0.0042543, 0.0051919
2: 0.0052260, 0.0119861, 0.0029161, 0.0121857, -0.0069596, 0.0090701
3: -0.0056895, 0.0002751, -0.0060836, 0.0016560, -0.0071283, 0.0063586
4: -0.0017928, 0.0021223, -0.0022038, 0.0021190, -0.0033688, 0.0037653
5: 0.0003567, 0.0055940, -0.0009403, 0.0058054, -0.0054487, 0.0065342
6: -0.0153989, -0.0001051, -0.0164173, 0.0007338, -0.0154233, 0.0152592
7: -0.0050875, 0.0164400, -0.0076459, 0.0164229, -0.0189392, 0.0213550
8: 0.9875138, 1.0012174, 0.9865701, 1.0015080, -0.0122615, 0.0141249
9: -0.0166085, -0.0036968, -0.0165976, -0.0021772, -0.0126508, 0.0111396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096227, upper bound: 0.0104682
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098950, upper bound: 0.0106368
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0052132, 0.0123276, 0.0040365, 0.0179432, -0.0127300, 0.0082911
1: 0.0007545, 0.0028374, -0.0024182, 0.0032615, -0.0025069, 0.0052556
2: 0.0071251, 0.0114776, 0.0028378, 0.0121282, -0.0050030, 0.0086398
3: -0.0058256, -0.0016814, -0.0059370, 0.0016145, -0.0073633, 0.0042556
4: -0.0010954, 0.0022696, -0.0021485, 0.0019655, -0.0028229, 0.0039531
5: 0.0012369, 0.0050552, -0.0009147, 0.0057445, -0.0045075, 0.0059699
6: -0.0152630, -0.0022429, -0.0158012, 0.0004921, -0.0147956, 0.0132962
7: -0.0006704, 0.0171932, -0.0073581, 0.0156380, -0.0152433, 0.0223804
8: 0.9895646, 1.0015625, 0.9867420, 1.0009423, -0.0106360, 0.0145038
9: -0.0170901, -0.0060387, -0.0160957, -0.0023516, -0.0132802, 0.0092900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0084556, upper bound: 0.0096590
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0083940, upper bound: 0.0096590
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0051942, 0.0125680, 0.0041866, 0.0181067, -0.0129125, 0.0083814
1: 0.0006841, 0.0028643, -0.0024218, 0.0033415, -0.0026574, 0.0052861
2: 0.0069787, 0.0114881, 0.0027545, 0.0120452, -0.0050665, 0.0087337
3: -0.0059319, -0.0016112, -0.0061479, 0.0015786, -0.0074719, 0.0045367
4: -0.0011124, 0.0023847, -0.0020637, 0.0021650, -0.0029300, 0.0040436
5: 0.0011070, 0.0050663, -0.0010214, 0.0056566, -0.0045495, 0.0060877
6: -0.0157104, -0.0021987, -0.0164776, 0.0001432, -0.0151294, 0.0136844
7: -0.0007909, 0.0177816, -0.0069400, 0.0166583, -0.0158199, 0.0228601
8: 0.9895222, 1.0019795, 0.9870265, 1.0016328, -0.0110148, 0.0148855
9: -0.0174664, -0.0059807, -0.0167481, -0.0026203, -0.0135834, 0.0096424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090594, upper bound: 0.0099242
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090594, upper bound: 0.0103157
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043069, 0.0149960, 0.0048216, 0.0162703, -0.0119635, 0.0101744
1: -0.0010924, 0.0028031, -0.0011731, 0.0029735, -0.0040659, 0.0039762
2: 0.0050329, 0.0119787, 0.0042252, 0.0116942, -0.0066613, 0.0077536
3: -0.0056899, 0.0003817, -0.0060962, 0.0001543, -0.0058149, 0.0064780
4: -0.0017904, 0.0021227, -0.0014800, 0.0024899, -0.0042803, 0.0032763
5: 0.0002578, 0.0055861, -0.0003750, 0.0052846, -0.0050268, 0.0059610
6: -0.0154717, -0.0001363, -0.0171147, -0.0013326, -0.0133616, 0.0169784
7: -0.0051331, 0.0164419, -0.0034822, 0.0183198, -0.0234530, 0.0185269
8: 0.9875436, 1.0012391, 0.9886914, 1.0026642, -0.0150830, 0.0116065
9: -0.0166097, -0.0037011, -0.0178105, -0.0047023, -0.0108686, 0.0141094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096202, upper bound: 0.0093467
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098949, upper bound: 0.0098490
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043069, 0.0149960, 0.0038493, 0.0192246, -0.0149177, 0.0111467
1: -0.0010924, 0.0028031, -0.0030645, 0.0035965, -0.0046889, 0.0058676
2: 0.0050329, 0.0119787, 0.0019244, 0.0122317, -0.0071988, 0.0100543
3: -0.0056899, 0.0003817, -0.0064235, 0.0023502, -0.0078815, 0.0068052
4: -0.0017904, 0.0021227, -0.0023939, 0.0022645, -0.0036920, 0.0039732
5: 0.0002578, 0.0055861, -0.0015013, 0.0058541, -0.0055963, 0.0070874
6: -0.0154717, -0.0001363, -0.0171419, 0.0009272, -0.0162776, 0.0165252
7: -0.0051331, 0.0164419, -0.0088719, 0.0171670, -0.0207160, 0.0226675
8: 0.9875436, 1.0012391, 0.9856811, 1.0020669, -0.0133638, 0.0155121
9: -0.0166097, -0.0037011, -0.0170734, -0.0014321, -0.0134735, 0.0122065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096202, upper bound: 0.0105916
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098949, upper bound: 0.0107560
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0050930, 0.0131237, 0.0041390, 0.0166825, -0.0111566, 0.0089847
1: 0.0005196, 0.0028685, -0.0017932, 0.0029754, -0.0024558, 0.0046617
2: 0.0065618, 0.0115441, 0.0037533, 0.0120715, -0.0055097, 0.0076071
3: -0.0059487, -0.0014585, -0.0056028, 0.0009709, -0.0066858, 0.0039714
4: -0.0011826, 0.0024028, -0.0019645, 0.0018200, -0.0023225, 0.0036392
5: 0.0008785, 0.0051256, -0.0003954, 0.0056844, -0.0046703, 0.0051877
6: -0.0159603, -0.0019635, -0.0151099, 0.0002537, -0.0138296, 0.0109479
7: -0.0011922, 0.0178745, -0.0061928, 0.0148939, -0.0127940, 0.0206748
8: 0.9892966, 1.0021054, 0.9871694, 1.0003841, -0.0086117, 0.0129033
9: -0.0175258, -0.0057476, -0.0156199, -0.0030566, -0.0121413, 0.0076618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099201
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099201
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0050744, 0.0133576, 0.0042629, 0.0167824, -0.0112522, 0.0090947
1: 0.0004485, 0.0028958, -0.0017595, 0.0030369, -0.0025884, 0.0046553
2: 0.0064144, 0.0115544, 0.0037236, 0.0120030, -0.0055887, 0.0076514
3: -0.0060567, -0.0013870, -0.0058025, 0.0009057, -0.0067766, 0.0041507
4: -0.0011995, 0.0025198, -0.0018937, 0.0020244, -0.0024253, 0.0037346
5: 0.0007519, 0.0051365, -0.0004754, 0.0056118, -0.0047907, 0.0052535
6: -0.0163991, -0.0019202, -0.0157681, -0.0000342, -0.0141707, 0.0112970
7: -0.0013139, 0.0184725, -0.0058033, 0.0159391, -0.0133479, 0.0212011
8: 0.9892550, 1.0025264, 0.9874456, 1.0010855, -0.0089712, 0.0132490
9: -0.0179082, -0.0056897, -0.0162883, -0.0032888, -0.0124636, 0.0079989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100948
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100948
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041204, 0.0160361, 0.0048824, 0.0151353, -0.0110149, 0.0110538
1: -0.0013523, 0.0029077, -0.0005854, 0.0028602, -0.0042125, 0.0034931
2: 0.0043268, 0.0120818, 0.0050649, 0.0116605, -0.0073337, 0.0070169
3: -0.0058983, 0.0006193, -0.0059158, -0.0004137, -0.0052626, 0.0065351
4: -0.0019261, 0.0022926, -0.0013814, 0.0023672, -0.0038821, 0.0029901
5: -0.0002238, 0.0056953, 0.0000947, 0.0052490, -0.0052937, 0.0055632
6: -0.0164064, 0.0002970, -0.0164517, -0.0014740, -0.0126903, 0.0154292
7: -0.0058697, 0.0173108, -0.0027011, 0.0176924, -0.0218368, 0.0168380
8: 0.9871279, 1.0019611, 0.9888270, 1.0021696, -0.0135933, 0.0106246
9: -0.0171653, -0.0032406, -0.0174093, -0.0050631, -0.0098909, 0.0128624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0093229
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0097516
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041204, 0.0160361, 0.0039325, 0.0178678, -0.0137474, 0.0121036
1: -0.0013523, 0.0029077, -0.0023953, 0.0033033, -0.0046556, 0.0053030
2: 0.0043268, 0.0120818, 0.0029161, 0.0121857, -0.0078589, 0.0091657
3: -0.0058983, 0.0006193, -0.0060836, 0.0016560, -0.0073014, 0.0067029
4: -0.0019261, 0.0022926, -0.0022038, 0.0021190, -0.0032892, 0.0036769
5: -0.0002238, 0.0056953, -0.0009403, 0.0058054, -0.0060292, 0.0065754
6: -0.0164064, 0.0002970, -0.0164173, 0.0007338, -0.0154218, 0.0146803
7: -0.0058697, 0.0173108, -0.0076459, 0.0164229, -0.0187284, 0.0210689
8: 0.9871279, 1.0019611, 0.9865701, 1.0015080, -0.0119231, 0.0139352
9: -0.0171653, -0.0032406, -0.0165976, -0.0021772, -0.0123880, 0.0109162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0104213
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0105986
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0051189, 0.0133237, 0.0040365, 0.0179432, -0.0124446, 0.0092872
1: 0.0004091, 0.0028686, -0.0024182, 0.0032615, -0.0028523, 0.0052868
2: 0.0064097, 0.0115298, 0.0028378, 0.0121282, -0.0057185, 0.0085375
3: -0.0059490, -0.0013931, -0.0059370, 0.0016145, -0.0073979, 0.0045439
4: -0.0011734, 0.0024032, -0.0021485, 0.0019655, -0.0026576, 0.0038487
5: 0.0008047, 0.0051104, -0.0009147, 0.0057445, -0.0049398, 0.0057552
6: -0.0160246, -0.0020237, -0.0158012, 0.0004921, -0.0146503, 0.0122376
7: -0.0011893, 0.0178763, -0.0073581, 0.0156380, -0.0146067, 0.0220059
8: 0.9893544, 1.0021268, 0.9867420, 1.0009423, -0.0098112, 0.0142343
9: -0.0175269, -0.0057744, -0.0160957, -0.0023516, -0.0129663, 0.0087647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0083827, upper bound: 0.0096266
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0083321, upper bound: 0.0096266
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0051001, 0.0135511, 0.0041866, 0.0181067, -0.0126372, 0.0093645
1: 0.0003407, 0.0028959, -0.0024218, 0.0033415, -0.0030008, 0.0053177
2: 0.0062669, 0.0115401, 0.0027545, 0.0120452, -0.0057783, 0.0086425
3: -0.0060570, -0.0013237, -0.0061479, 0.0015786, -0.0075187, 0.0048242
4: -0.0011903, 0.0025201, -0.0020637, 0.0021650, -0.0027648, 0.0039553
5: 0.0006814, 0.0051214, -0.0010214, 0.0056566, -0.0049751, 0.0058530
6: -0.0164607, -0.0019801, -0.0164776, 0.0001432, -0.0150589, 0.0126291
7: -0.0013098, 0.0184742, -0.0069400, 0.0166583, -0.0151759, 0.0225686
8: 0.9893125, 1.0025464, 0.9870265, 1.0016328, -0.0102001, 0.0146797
9: -0.0179093, -0.0057166, -0.0167481, -0.0026203, -0.0133265, 0.0091162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0098698
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0102454
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0041470, 0.0163183, 0.0048216, 0.0162703, -0.0121233, 0.0114968
1: -0.0015003, 0.0029868, -0.0011731, 0.0029735, -0.0044739, 0.0041599
2: 0.0041110, 0.0120671, 0.0042252, 0.0116942, -0.0075832, 0.0078419
3: -0.0059614, 0.0007379, -0.0060962, 0.0001543, -0.0060647, 0.0068341
4: -0.0019298, 0.0022932, -0.0014800, 0.0024899, -0.0042312, 0.0031476
5: -0.0003335, 0.0056797, -0.0003750, 0.0052846, -0.0055652, 0.0060547
6: -0.0164860, 0.0002351, -0.0171147, -0.0013326, -0.0132056, 0.0169421
7: -0.0059246, 0.0173137, -0.0034822, 0.0183198, -0.0237017, 0.0179993
8: 0.9871873, 1.0019870, 0.9886914, 1.0026642, -0.0150427, 0.0111895
9: -0.0171672, -0.0032096, -0.0178105, -0.0047023, -0.0104595, 0.0140369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0093444
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0098192
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041470, 0.0163183, 0.0038493, 0.0192246, -0.0150775, 0.0124691
1: -0.0015003, 0.0029868, -0.0030645, 0.0035965, -0.0050968, 0.0060513
2: 0.0041110, 0.0120671, 0.0019244, 0.0122317, -0.0081208, 0.0101427
3: -0.0059614, 0.0007379, -0.0064235, 0.0023502, -0.0081699, 0.0071613
4: -0.0019298, 0.0022932, -0.0023939, 0.0022645, -0.0036459, 0.0039016
5: -0.0003335, 0.0056797, -0.0015013, 0.0058541, -0.0061876, 0.0071810
6: -0.0164860, 0.0002351, -0.0171419, 0.0009272, -0.0163536, 0.0161126
7: -0.0059246, 0.0173137, -0.0088719, 0.0171670, -0.0206413, 0.0224636
8: 0.9871873, 1.0019870, 0.9856811, 1.0020669, -0.0133581, 0.0154317
9: -0.0171672, -0.0032096, -0.0170734, -0.0014321, -0.0132739, 0.0121134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0105396
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0107056
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045420, 0.0154326, 0.0044699, 0.0139247, -0.0093827, 0.0109627
1: -0.0011615, 0.0028042, -0.0005594, 0.0027316, -0.0038931, 0.0033636
2: 0.0047269, 0.0118487, 0.0057835, 0.0118885, -0.0071616, 0.0060652
3: -0.0056941, 0.0002730, -0.0054067, -0.0002099, -0.0052590, 0.0056796
4: -0.0016390, 0.0021272, -0.0016400, 0.0018161, -0.0030988, 0.0032013
5: 0.0000545, 0.0054483, 0.0007647, 0.0054906, -0.0052867, 0.0046836
6: -0.0157301, -0.0006829, -0.0141618, -0.0005154, -0.0131885, 0.0123305
7: -0.0043298, 0.0164650, -0.0040350, 0.0148740, -0.0177367, 0.0178648
8: 0.9880681, 1.0013638, 0.9879074, 1.0000936, -0.0106573, 0.0113302
9: -0.0166245, -0.0041991, -0.0156072, -0.0042170, -0.0105783, 0.0102711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096227
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096227
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045243, 0.0156383, 0.0045793, 0.0137583, -0.0092340, 0.0110590
1: -0.0012284, 0.0028303, -0.0003631, 0.0027821, -0.0040105, 0.0031934
2: 0.0046005, 0.0118585, 0.0059500, 0.0118281, -0.0072276, 0.0059085
3: -0.0057973, 0.0003439, -0.0056067, -0.0004121, -0.0051977, 0.0059313
4: -0.0016558, 0.0022389, -0.0015639, 0.0020326, -0.0032039, 0.0032980
5: -0.0000546, 0.0054587, 0.0007877, 0.0054265, -0.0054039, 0.0046710
6: -0.0161644, -0.0006417, -0.0148013, -0.0007695, -0.0136749, 0.0125560
7: -0.0044524, 0.0170364, -0.0035756, 0.0159815, -0.0182869, 0.0182749
8: 0.9880285, 1.0017632, 0.9881511, 1.0008229, -0.0109929, 0.0117585
9: -0.0169899, -0.0041415, -0.0163153, -0.0044703, -0.0108857, 0.0106151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0098949
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0098949
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0035828, 0.0184948, 0.0051620, 0.0124331, -0.0088503, 0.0133328
1: -0.0031010, 0.0034380, 0.0007900, 0.0028694, -0.0059704, 0.0026480
2: 0.0023220, 0.0123790, 0.0070961, 0.0115059, -0.0091839, 0.0052829
3: -0.0060975, 0.0024716, -0.0059520, -0.0016732, -0.0044243, 0.0084235
4: -0.0025415, 0.0019933, -0.0011252, 0.0024064, -0.0047021, 0.0026237
5: -0.0010999, 0.0060102, 0.0011525, 0.0050851, -0.0061850, 0.0048577
6: -0.0160585, 0.0015465, -0.0157374, -0.0021240, -0.0126295, 0.0172840
7: -0.0096704, 0.0157804, -0.0008056, 0.0178928, -0.0266052, 0.0141841
8: 0.9853110, 1.0010893, 0.9894507, 1.0020430, -0.0167320, 0.0099688
9: -0.0161867, -0.0009692, -0.0175375, -0.0059422, -0.0086359, 0.0158575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0096240
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0096240
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0035828, 0.0184948, 0.0042934, 0.0147491, -0.0111663, 0.0142014
1: -0.0031010, 0.0034380, -0.0009510, 0.0028030, -0.0059041, 0.0043890
2: 0.0023220, 0.0123790, 0.0052260, 0.0119861, -0.0096641, 0.0071530
3: -0.0060975, 0.0024716, -0.0056895, 0.0002751, -0.0063726, 0.0081611
4: -0.0025415, 0.0019933, -0.0017928, 0.0021223, -0.0041489, 0.0031150
5: -0.0010999, 0.0060102, 0.0003567, 0.0055940, -0.0066938, 0.0056536
6: -0.0160585, 0.0015465, -0.0153989, -0.0001051, -0.0141029, 0.0167161
7: -0.0096704, 0.0157804, -0.0050875, 0.0164400, -0.0236947, 0.0176411
8: 0.9853110, 1.0010893, 0.9875138, 1.0012174, -0.0159063, 0.0112440
9: -0.0161867, -0.0009692, -0.0166085, -0.0036968, -0.0103096, 0.0140361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0106059
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0106059
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0046412, 0.0144897, 0.0042700, 0.0155925, -0.0109513, 0.0102197
1: -0.0005898, 0.0028336, -0.0016012, 0.0027945, -0.0033844, 0.0043811
2: 0.0054865, 0.0117939, 0.0044998, 0.0119991, -0.0065126, 0.0072941
3: -0.0058105, -0.0002661, -0.0055632, 0.0008582, -0.0062434, 0.0050103
4: -0.0015412, 0.0022533, -0.0018457, 0.0019604, -0.0027964, 0.0032575
5: 0.0003853, 0.0053902, 0.0000712, 0.0056077, -0.0050840, 0.0053190
6: -0.0158937, -0.0009134, -0.0151238, -0.0000507, -0.0130167, 0.0122530
7: -0.0035711, 0.0171098, -0.0056222, 0.0156119, -0.0158218, 0.0187487
8: 0.9882892, 1.0017176, 0.9874615, 1.0007368, -0.0100119, 0.0112339
9: -0.0170368, -0.0045370, -0.0160790, -0.0035010, -0.0108160, 0.0092531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0102926
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0102926
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0046259, 0.0148899, 0.0040965, 0.0172142, -0.0125883, 0.0107934
1: -0.0008098, 0.0028342, -0.0023583, 0.0031860, -0.0039957, 0.0051719
2: 0.0051759, 0.0118023, 0.0033130, 0.0120950, -0.0069191, 0.0084894
3: -0.0058129, -0.0000708, -0.0059473, 0.0016382, -0.0071389, 0.0058765
4: -0.0015646, 0.0022558, -0.0020947, 0.0020603, -0.0030467, 0.0035757
5: 0.0002299, 0.0053992, -0.0006050, 0.0057093, -0.0054794, 0.0060042
6: -0.0160209, -0.0008779, -0.0159170, 0.0003525, -0.0143587, 0.0133855
7: -0.0037803, 0.0171227, -0.0071621, 0.0161231, -0.0172488, 0.0206631
8: 0.9882551, 1.0017626, 0.9870748, 1.0011908, -0.0109228, 0.0131718
9: -0.0170451, -0.0044535, -0.0164059, -0.0025534, -0.0120314, 0.0100834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0104309
time: 1.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0104309
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037309, 0.0170768, 0.0038740, 0.0169656, -0.0132347, 0.0132027
1: -0.0023866, 0.0031596, -0.0024410, 0.0031442, -0.0055308, 0.0056005
2: 0.0033901, 0.0122971, 0.0034254, 0.0122180, -0.0088279, 0.0088718
3: -0.0058712, 0.0017817, -0.0058504, 0.0018139, -0.0076851, 0.0075569
4: -0.0023217, 0.0019857, -0.0022337, 0.0019740, -0.0033235, 0.0037369
5: -0.0005396, 0.0059235, -0.0004503, 0.0058397, -0.0063793, 0.0063737
6: -0.0156859, 0.0012023, -0.0155619, 0.0008697, -0.0160077, 0.0148968
7: -0.0083488, 0.0157416, -0.0079364, 0.0156817, -0.0192756, 0.0214901
8: 0.9862594, 1.0009642, 0.9865786, 1.0008941, -0.0125366, 0.0139729
9: -0.0161619, -0.0018109, -0.0161237, -0.0020953, -0.0125486, 0.0111814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108088
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108088
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037001, 0.0175776, 0.0036858, 0.0186824, -0.0149823, 0.0138918
1: -0.0026450, 0.0032676, -0.0032380, 0.0034950, -0.0061400, 0.0065056
2: 0.0029982, 0.0123142, 0.0021736, 0.0123221, -0.0093238, 0.0101405
3: -0.0059584, 0.0020255, -0.0062027, 0.0026406, -0.0085990, 0.0082282
4: -0.0023865, 0.0019879, -0.0025158, 0.0020746, -0.0037126, 0.0040547
5: -0.0007342, 0.0059415, -0.0011694, 0.0059499, -0.0066841, 0.0071109
6: -0.0158222, 0.0012739, -0.0163896, 0.0013071, -0.0171292, 0.0165818
7: -0.0087478, 0.0157528, -0.0096282, 0.0161962, -0.0213126, 0.0233815
8: 0.9860564, 1.0010083, 0.9853988, 1.0013533, -0.0143128, 0.0156094
9: -0.0161691, -0.0015564, -0.0164526, -0.0010248, -0.0137799, 0.0125110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0110357
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0110357
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041191, 0.0155983, 0.0053091, 0.0116439, -0.0069613, 0.0102893
1: -0.0015004, 0.0028011, 0.0013589, 0.0028667, -0.0043671, 0.0014422
2: 0.0045481, 0.0120825, 0.0077408, 0.0114246, -0.0068765, 0.0041338
3: -0.0056764, 0.0008459, -0.0059415, -0.0022259, -0.0032831, 0.0066587
4: -0.0019352, 0.0021066, -0.0009952, 0.0023950, -0.0037577, 0.0028030
5: 0.0000239, 0.0056961, 0.0014463, 0.0049990, -0.0049225, 0.0038220
6: -0.0155947, 0.0003001, -0.0154042, -0.0024657, -0.0122811, 0.0134938
7: -0.0060690, 0.0163599, 0.0001125, 0.0178347, -0.0213179, 0.0147876
8: 0.9871250, 1.0012417, 0.9897784, 1.0019127, -0.0125993, 0.0104630
9: -0.0165573, -0.0032127, -0.0175003, -0.0063861, -0.0091984, 0.0124392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099531, upper bound: 0.0091777
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099531, upper bound: 0.0091777
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037109, 0.0169295, 0.0044377, 0.0138708, -0.0101598, 0.0124918
1: -0.0023319, 0.0031684, -0.0004606, 0.0028018, -0.0051337, 0.0036290
2: 0.0035096, 0.0123082, 0.0058957, 0.0119064, -0.0083967, 0.0064125
3: -0.0059779, 0.0017807, -0.0056847, -0.0002326, -0.0057453, 0.0074654
4: -0.0023258, 0.0021206, -0.0016623, 0.0021171, -0.0041655, 0.0032426
5: -0.0004919, 0.0059352, 0.0007043, 0.0055094, -0.0059530, 0.0052308
6: -0.0160370, 0.0012487, -0.0151542, -0.0004405, -0.0139271, 0.0160637
7: -0.0083471, 0.0164315, -0.0041570, 0.0164135, -0.0236745, 0.0180693
8: 0.9862149, 1.0013982, 0.9878355, 1.0011351, -0.0149202, 0.0116546
9: -0.0166031, -0.0018160, -0.0165916, -0.0041430, -0.0107112, 0.0139301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0104220
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0104220
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039141, 0.0171778, 0.0052856, 0.0119209, -0.0080068, 0.0118923
1: -0.0022520, 0.0032232, 0.0011380, 0.0028669, -0.0051189, 0.0020852
2: 0.0033951, 0.0121959, 0.0075155, 0.0114376, -0.0080425, 0.0046803
3: -0.0061106, 0.0016287, -0.0059422, -0.0020340, -0.0040766, 0.0075242
4: -0.0022015, 0.0022415, -0.0010242, 0.0023958, -0.0040715, 0.0030323
5: -0.0006248, 0.0058162, 0.0013357, 0.0050128, -0.0056030, 0.0044804
6: -0.0163919, 0.0007765, -0.0155223, -0.0024111, -0.0134825, 0.0147815
7: -0.0076583, 0.0170496, -0.0001313, 0.0178386, -0.0231735, 0.0161020
8: 0.9866680, 1.0018084, 0.9897260, 1.0019503, -0.0144743, 0.0113579
9: -0.0169983, -0.0022196, -0.0175028, -0.0062842, -0.0099582, 0.0136358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100770, upper bound: 0.0091777
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100770, upper bound: 0.0091777
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0034901, 0.0186278, 0.0044043, 0.0143032, -0.0108131, 0.0142234
1: -0.0031318, 0.0035557, -0.0007164, 0.0028020, -0.0059337, 0.0042722
2: 0.0022706, 0.0124303, 0.0055574, 0.0119248, -0.0096542, 0.0068729
3: -0.0063847, 0.0026222, -0.0056853, 0.0000092, -0.0063939, 0.0083075
4: -0.0026231, 0.0022557, -0.0017039, 0.0021177, -0.0045452, 0.0034885
5: -0.0011968, 0.0060646, 0.0005387, 0.0055290, -0.0067231, 0.0055259
6: -0.0168557, 0.0017621, -0.0152752, -0.0003629, -0.0151615, 0.0170373
7: -0.0101193, 0.0171223, -0.0045079, 0.0164164, -0.0258423, 0.0194712
8: 0.9850622, 1.0019678, 0.9877611, 1.0011710, -0.0161088, 0.0125948
9: -0.0170448, -0.0006924, -0.0165934, -0.0039965, -0.0115273, 0.0153594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108144, upper bound: 0.0104581
time: 1.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108144, upper bound: 0.0104581
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041191, 0.0155983, 0.0047041, 0.0159838, -0.0118648, 0.0108942
1: -0.0015004, 0.0028011, -0.0015036, 0.0029115, -0.0044119, 0.0043047
2: 0.0045481, 0.0120825, 0.0043086, 0.0117591, -0.0072109, 0.0077740
3: -0.0056764, 0.0008459, -0.0058694, 0.0005567, -0.0061963, 0.0063736
4: -0.0019352, 0.0021066, -0.0015847, 0.0022494, -0.0033376, 0.0032300
5: 0.0000239, 0.0056961, -0.0001669, 0.0053534, -0.0053294, 0.0058630
6: -0.0155947, 0.0003001, -0.0162468, -0.0010597, -0.0132100, 0.0139783
7: -0.0060690, 0.0163599, -0.0042454, 0.0170900, -0.0191367, 0.0185953
8: 0.9871250, 1.0012417, 0.9884295, 1.0018162, -0.0116422, 0.0113618
9: -0.0165573, -0.0032127, -0.0170242, -0.0043419, -0.0107419, 0.0110650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105065, upper bound: 0.0100805
time: 1.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105065, upper bound: 0.0100805
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037109, 0.0169295, 0.0037648, 0.0188969, -0.0151860, 0.0131647
1: -0.0023319, 0.0031684, -0.0033980, 0.0034930, -0.0058248, 0.0065664
2: 0.0035096, 0.0123082, 0.0019900, 0.0122784, -0.0087688, 0.0103182
3: -0.0059779, 0.0017807, -0.0061319, 0.0027241, -0.0087020, 0.0079126
4: -0.0023258, 0.0021206, -0.0024827, 0.0019809, -0.0037476, 0.0038874
5: -0.0004919, 0.0059352, -0.0012222, 0.0059036, -0.0063955, 0.0071574
6: -0.0160370, 0.0012487, -0.0160341, 0.0011235, -0.0163228, 0.0166896
7: -0.0083471, 0.0164315, -0.0094922, 0.0157169, -0.0215185, 0.0225859
8: 0.9862149, 1.0013982, 0.9854251, 1.0010453, -0.0140839, 0.0154458
9: -0.0166031, -0.0018160, -0.0161461, -0.0011052, -0.0132644, 0.0125624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0109772
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0109772
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039141, 0.0171778, 0.0046726, 0.0164415, -0.0125274, 0.0125052
1: -0.0022520, 0.0032232, -0.0017456, 0.0030303, -0.0052823, 0.0049688
2: 0.0033951, 0.0121959, 0.0039602, 0.0117765, -0.0083814, 0.0082357
3: -0.0061106, 0.0016287, -0.0059653, 0.0007727, -0.0068833, 0.0074502
4: -0.0022015, 0.0022415, -0.0016373, 0.0022518, -0.0036463, 0.0035034
5: -0.0006248, 0.0058162, -0.0003496, 0.0053718, -0.0059966, 0.0061657
6: -0.0163919, 0.0007765, -0.0163749, -0.0009865, -0.0146162, 0.0152818
7: -0.0076583, 0.0170496, -0.0045825, 0.0171021, -0.0209781, 0.0200681
8: 0.9866680, 1.0018084, 0.9883592, 1.0018609, -0.0135002, 0.0127001
9: -0.0169983, -0.0022196, -0.0170319, -0.0041355, -0.0116880, 0.0122410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106732, upper bound: 0.0100940
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106732, upper bound: 0.0100940
time: 1.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0034901, 0.0186278, 0.0037202, 0.0194421, -0.0159521, 0.0149076
1: -0.0031318, 0.0035557, -0.0036678, 0.0035787, -0.0067104, 0.0072235
2: 0.0022706, 0.0124303, 0.0015749, 0.0123031, -0.0100324, 0.0108554
3: -0.0063847, 0.0026222, -0.0062013, 0.0029895, -0.0093742, 0.0088235
4: -0.0026231, 0.0022557, -0.0025670, 0.0019829, -0.0041149, 0.0042444
5: -0.0011968, 0.0060646, -0.0014353, 0.0059298, -0.0071265, 0.0074999
6: -0.0168557, 0.0017621, -0.0161683, 0.0012271, -0.0180828, 0.0179304
7: -0.0101193, 0.0171223, -0.0100048, 0.0157274, -0.0236201, 0.0244714
8: 0.9850622, 1.0019678, 0.9849930, 1.0010883, -0.0160261, 0.0169747
9: -0.0170448, -0.0006924, -0.0161528, -0.0007736, -0.0144922, 0.0139486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113027, upper bound: 0.0110612
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113027, upper bound: 0.0110612
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045343, 0.0154571, 0.0043228, 0.0150310, -0.0104967, 0.0110747
1: -0.0011731, 0.0028041, -0.0008553, 0.0027750, -0.0039481, 0.0036594
2: 0.0047084, 0.0118530, 0.0050205, 0.0119699, -0.0072615, 0.0068325
3: -0.0056938, 0.0002837, -0.0055786, 0.0000499, -0.0055984, 0.0058623
4: -0.0016414, 0.0021269, -0.0017441, 0.0020022, -0.0034863, 0.0032660
5: 0.0000442, 0.0054529, 0.0002542, 0.0055768, -0.0053348, 0.0051987
6: -0.0157389, -0.0006649, -0.0151610, -0.0001733, -0.0133454, 0.0143235
7: -0.0043171, 0.0164636, -0.0046728, 0.0158256, -0.0196925, 0.0183596
8: 0.9880508, 1.0013654, 0.9875791, 1.0008680, -0.0122390, 0.0114727
9: -0.0166236, -0.0041922, -0.0162157, -0.0038694, -0.0108011, 0.0115357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096463
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096124
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045165, 0.0156629, 0.0044539, 0.0149283, -0.0104118, 0.0112090
1: -0.0012398, 0.0028302, -0.0007056, 0.0028199, -0.0040597, 0.0035358
2: 0.0045821, 0.0118628, 0.0051381, 0.0118974, -0.0073153, 0.0067248
3: -0.0057970, 0.0003546, -0.0057561, -0.0001206, -0.0055864, 0.0061107
4: -0.0016581, 0.0022386, -0.0016573, 0.0021943, -0.0035970, 0.0033613
5: -0.0000651, 0.0054633, 0.0002603, 0.0054999, -0.0054450, 0.0052030
6: -0.0161738, -0.0006235, -0.0157495, -0.0004782, -0.0138052, 0.0146005
7: -0.0044410, 0.0170348, -0.0041865, 0.0168084, -0.0202772, 0.0188122
8: 0.9880111, 1.0017650, 0.9878716, 1.0015242, -0.0126077, 0.0118753
9: -0.0169888, -0.0041347, -0.0168441, -0.0041564, -0.0111088, 0.0118969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0099553
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0098920
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0035678, 0.0185202, 0.0050710, 0.0134054, -0.0098376, 0.0134492
1: -0.0031132, 0.0034430, 0.0004384, 0.0029010, -0.0060142, 0.0030047
2: 0.0023032, 0.0123873, 0.0063864, 0.0115563, -0.0092531, 0.0060009
3: -0.0061012, 0.0024821, -0.0060771, -0.0013760, -0.0047252, 0.0085592
4: -0.0025506, 0.0019929, -0.0012025, 0.0025419, -0.0050924, 0.0026898
5: -0.0011096, 0.0060191, 0.0007254, 0.0051385, -0.0062481, 0.0052936
6: -0.0160645, 0.0015815, -0.0164873, -0.0019123, -0.0127654, 0.0180688
7: -0.0097115, 0.0157783, -0.0013347, 0.0185856, -0.0282970, 0.0147260
8: 0.9852686, 1.0010895, 0.9892474, 1.0026063, -0.0173377, 0.0100924
9: -0.0161854, -0.0009414, -0.0179805, -0.0056796, -0.0088669, 0.0170391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099025, upper bound: 0.0091582
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099900, upper bound: 0.0091582
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0035678, 0.0185202, 0.0041204, 0.0160361, -0.0124683, 0.0143998
1: -0.0031132, 0.0034430, -0.0013523, 0.0029077, -0.0060209, 0.0047953
2: 0.0023032, 0.0123873, 0.0043268, 0.0120818, -0.0097786, 0.0080605
3: -0.0061012, 0.0024821, -0.0058983, 0.0006193, -0.0067205, 0.0083803
4: -0.0025506, 0.0019929, -0.0019261, 0.0022926, -0.0045538, 0.0032929
5: -0.0011096, 0.0060191, -0.0002238, 0.0056953, -0.0068049, 0.0062429
6: -0.0160645, 0.0015815, -0.0164064, 0.0002970, -0.0147832, 0.0179879
7: -0.0097115, 0.0157783, -0.0058697, 0.0173108, -0.0257424, 0.0187475
8: 0.9852686, 1.0010895, 0.9871279, 1.0019611, -0.0166925, 0.0119584
9: -0.0161854, -0.0009414, -0.0171653, -0.0032406, -0.0109284, 0.0153584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099025, upper bound: 0.0102816
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099900, upper bound: 0.0103283
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0046339, 0.0145138, 0.0041905, 0.0168713, -0.0122374, 0.0103233
1: -0.0006014, 0.0028335, -0.0019453, 0.0030923, -0.0036937, 0.0047685
2: 0.0054691, 0.0117979, 0.0036366, 0.0120430, -0.0065740, 0.0081613
3: -0.0058102, -0.0002573, -0.0059049, 0.0011725, -0.0066547, 0.0056476
4: -0.0015434, 0.0022530, -0.0019696, 0.0021035, -0.0031988, 0.0034452
5: 0.0003752, 0.0053945, -0.0005068, 0.0056542, -0.0052790, 0.0059013
6: -0.0159026, -0.0008965, -0.0161000, 0.0001341, -0.0138258, 0.0144052
7: -0.0035588, 0.0171081, -0.0063490, 0.0163437, -0.0177942, 0.0197878
8: 0.9882730, 1.0017191, 0.9872842, 1.0013795, -0.0117041, 0.0124504
9: -0.0170357, -0.0045311, -0.0165469, -0.0030181, -0.0115357, 0.0105632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0103795
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0102904
time: 1.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0046132, 0.0149140, 0.0039902, 0.0183847, -0.0137715, 0.0109237
1: -0.0008213, 0.0028341, -0.0026668, 0.0034290, -0.0042503, 0.0055009
2: 0.0051583, 0.0118094, 0.0025398, 0.0121538, -0.0069955, 0.0092695
3: -0.0058125, -0.0000595, -0.0062549, 0.0019365, -0.0075118, 0.0061954
4: -0.0015691, 0.0022555, -0.0022235, 0.0022160, -0.0034189, 0.0037370
5: 0.0002199, 0.0054067, -0.0011463, 0.0057716, -0.0055517, 0.0065529
6: -0.0160295, -0.0008483, -0.0168961, 0.0005996, -0.0150269, 0.0154654
7: -0.0037809, 0.0171209, -0.0078739, 0.0169190, -0.0190689, 0.0215375
8: 0.9882266, 1.0017641, 0.9864346, 1.0018735, -0.0125111, 0.0142508
9: -0.0170439, -0.0044398, -0.0169148, -0.0020613, -0.0126545, 0.0112937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0105266
time: 2.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0104303
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037185, 0.0170996, 0.0037725, 0.0183333, -0.0146148, 0.0133271
1: -0.0023980, 0.0031647, -0.0028148, 0.0034172, -0.0058152, 0.0059796
2: 0.0033727, 0.0123040, 0.0025147, 0.0122742, -0.0089015, 0.0097893
3: -0.0058750, 0.0017932, -0.0061726, 0.0021629, -0.0080379, 0.0079658
4: -0.0023297, 0.0019853, -0.0023792, 0.0021173, -0.0038361, 0.0038834
5: -0.0005485, 0.0059307, -0.0010797, 0.0058991, -0.0064476, 0.0070104
6: -0.0156914, 0.0012311, -0.0165848, 0.0011057, -0.0166291, 0.0175688
7: -0.0083789, 0.0157396, -0.0087448, 0.0164143, -0.0217578, 0.0222998
8: 0.9862318, 1.0009644, 0.9859186, 1.0015510, -0.0147464, 0.0150458
9: -0.0161607, -0.0017858, -0.0165921, -0.0015360, -0.0131348, 0.0128415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108983
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108077
time: 2.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036807, 0.0176014, 0.0035618, 0.0199097, -0.0162290, 0.0140396
1: -0.0026568, 0.0032730, -0.0035761, 0.0037267, -0.0063834, 0.0068491
2: 0.0029802, 0.0123249, 0.0013738, 0.0123906, -0.0094104, 0.0109511
3: -0.0059624, 0.0020361, -0.0065015, 0.0029694, -0.0089318, 0.0085376
4: -0.0023978, 0.0019875, -0.0026581, 0.0022304, -0.0040957, 0.0042536
5: -0.0007435, 0.0059529, -0.0017462, 0.0060226, -0.0067660, 0.0076991
6: -0.0158280, 0.0013190, -0.0174023, 0.0015953, -0.0174233, 0.0187213
7: -0.0087959, 0.0157508, -0.0104018, 0.0169929, -0.0231961, 0.0244570
8: 0.9860035, 1.0010085, 0.9846002, 1.0020504, -0.0159964, 0.0164083
9: -0.0161678, -0.0015200, -0.0169621, -0.0004854, -0.0145298, 0.0137498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0111344
time: 1.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0110344
time: 1.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041047, 0.0156148, 0.0052163, 0.0124932, -0.0083885, 0.0103985
1: -0.0015089, 0.0028057, 0.0010036, 0.0028983, -0.0044072, 0.0018021
2: 0.0045356, 0.0120905, 0.0071246, 0.0114759, -0.0069403, 0.0049658
3: -0.0056797, 0.0008555, -0.0060666, -0.0019140, -0.0036906, 0.0069221
4: -0.0019483, 0.0021061, -0.0010752, 0.0025305, -0.0041672, 0.0028714
5: 0.0000172, 0.0057045, 0.0010717, 0.0050533, -0.0049651, 0.0046084
6: -0.0155986, 0.0003335, -0.0161454, -0.0022501, -0.0124264, 0.0155111
7: -0.0061463, 0.0163572, -0.0004476, 0.0185274, -0.0233534, 0.0153591
8: 0.9870930, 1.0012410, 0.9895716, 1.0024731, -0.0142543, 0.0105975
9: -0.0165556, -0.0031685, -0.0179433, -0.0061132, -0.0094377, 0.0137771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099527, upper bound: 0.0091563
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099527, upper bound: 0.0091414
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0036941, 0.0169473, 0.0042945, 0.0150680, -0.0113739, 0.0126528
1: -0.0023407, 0.0031728, -0.0008259, 0.0028416, -0.0051823, 0.0039987
2: 0.0034962, 0.0123175, 0.0050621, 0.0119856, -0.0084894, 0.0072554
3: -0.0059810, 0.0017910, -0.0058421, 0.0000847, -0.0060657, 0.0076331
4: -0.0023376, 0.0021201, -0.0017702, 0.0022875, -0.0045220, 0.0033650
5: -0.0004987, 0.0059451, 0.0001678, 0.0055934, -0.0060514, 0.0057773
6: -0.0160411, 0.0012879, -0.0161378, -0.0001075, -0.0142957, 0.0174257
7: -0.0084240, 0.0164288, -0.0048344, 0.0172847, -0.0254686, 0.0188878
8: 0.9861773, 1.0013976, 0.9875160, 1.0018661, -0.0156888, 0.0119986
9: -0.0166013, -0.0017740, -0.0171486, -0.0037805, -0.0111256, 0.0150925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0105211
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0104214
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038978, 0.0171927, 0.0051928, 0.0127547, -0.0088570, 0.0119999
1: -0.0022607, 0.0032273, 0.0008200, 0.0028985, -0.0051591, 0.0024073
2: 0.0033829, 0.0122049, 0.0069077, 0.0114889, -0.0081060, 0.0052972
3: -0.0061135, 0.0016403, -0.0060672, -0.0017608, -0.0043527, 0.0077075
4: -0.0022103, 0.0022409, -0.0011005, 0.0025311, -0.0044704, 0.0030927
5: -0.0006307, 0.0058257, 0.0009692, 0.0050671, -0.0056403, 0.0048566
6: -0.0163906, 0.0008145, -0.0162443, -0.0021956, -0.0136047, 0.0167762
7: -0.0077163, 0.0170467, -0.0006436, 0.0185304, -0.0251789, 0.0166109
8: 0.9866314, 1.0018055, 0.9895192, 1.0025051, -0.0158737, 0.0114731
9: -0.0169965, -0.0021875, -0.0179452, -0.0060255, -0.0101712, 0.0149324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100738, upper bound: 0.0091563
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100738, upper bound: 0.0091414
time: 1.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0034819, 0.0186443, 0.0042579, 0.0155221, -0.0120402, 0.0143864
1: -0.0031408, 0.0035593, -0.0010883, 0.0028417, -0.0059825, 0.0046477
2: 0.0022576, 0.0124348, 0.0047112, 0.0120058, -0.0097482, 0.0077237
3: -0.0063871, 0.0026322, -0.0058426, 0.0003317, -0.0067189, 0.0084748
4: -0.0026283, 0.0022552, -0.0018098, 0.0022880, -0.0049163, 0.0036042
5: -0.0012022, 0.0060694, -0.0000057, 0.0056148, -0.0068126, 0.0060750
6: -0.0168542, 0.0017811, -0.0162636, -0.0000224, -0.0155025, 0.0180447
7: -0.0101600, 0.0171195, -0.0051522, 0.0172871, -0.0274471, 0.0202757
8: 0.9850340, 1.0019648, 0.9874344, 1.0019063, -0.0168722, 0.0129123
9: -0.0170430, -0.0006730, -0.0171502, -0.0036437, -0.0119195, 0.0164772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096302, upper bound: 0.0097508
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096302, upper bound: 0.0105557
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0041047, 0.0156148, 0.0046063, 0.0172378, -0.0131331, 0.0110085
1: -0.0015089, 0.0028057, -0.0018437, 0.0032038, -0.0047127, 0.0046495
2: 0.0045356, 0.0120905, 0.0034534, 0.0118132, -0.0072776, 0.0086371
3: -0.0056797, 0.0008555, -0.0061929, 0.0008591, -0.0065388, 0.0070485
4: -0.0019483, 0.0021061, -0.0017066, 0.0023738, -0.0037685, 0.0033750
5: 0.0000172, 0.0057045, -0.0007234, 0.0054107, -0.0053935, 0.0064279
6: -0.0155986, 0.0003335, -0.0171072, -0.0008324, -0.0138200, 0.0163099
7: -0.0061463, 0.0163572, -0.0048769, 0.0177260, -0.0212803, 0.0194035
8: 0.9870930, 1.0012410, 0.9882114, 1.0023832, -0.0135379, 0.0124435
9: -0.0165556, -0.0031685, -0.0174309, -0.0038781, -0.0113246, 0.0124683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105064, upper bound: 0.0100940
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105064, upper bound: 0.0100571
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0036941, 0.0169473, 0.0036510, 0.0201388, -0.0164447, 0.0132963
1: -0.0023407, 0.0031728, -0.0037142, 0.0037099, -0.0060506, 0.0068870
2: 0.0034962, 0.0123175, 0.0011799, 0.0123413, -0.0088452, 0.0111376
3: -0.0059810, 0.0017910, -0.0064098, 0.0030233, -0.0090043, 0.0082007
4: -0.0023376, 0.0021201, -0.0026206, 0.0021243, -0.0041216, 0.0041102
5: -0.0004987, 0.0059451, -0.0018270, 0.0059703, -0.0064690, 0.0077720
6: -0.0160411, 0.0012879, -0.0170670, 0.0013881, -0.0172725, 0.0183549
7: -0.0084240, 0.0164288, -0.0102395, 0.0164500, -0.0233694, 0.0237585
8: 0.9861773, 1.0013976, 0.9846555, 1.0017062, -0.0155290, 0.0167421
9: -0.0166013, -0.0017740, -0.0166150, -0.0005789, -0.0140870, 0.0137721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0110809
time: 1.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0109761
time: 1.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038978, 0.0171927, 0.0045748, 0.0176103, -0.0137125, 0.0126179
1: -0.0022607, 0.0032273, -0.0020408, 0.0032882, -0.0055489, 0.0052681
2: 0.0033829, 0.0122049, 0.0031654, 0.0118306, -0.0084477, 0.0090395
3: -0.0061135, 0.0016403, -0.0062616, 0.0010411, -0.0071546, 0.0079019
4: -0.0022103, 0.0022409, -0.0017549, 0.0023761, -0.0040711, 0.0036351
5: -0.0006307, 0.0058257, -0.0008772, 0.0054291, -0.0060598, 0.0067029
6: -0.0163906, 0.0008145, -0.0172223, -0.0007591, -0.0151730, 0.0175482
7: -0.0077163, 0.0170467, -0.0051979, 0.0177379, -0.0230800, 0.0207899
8: 0.9866314, 1.0018055, 0.9881409, 1.0024233, -0.0153236, 0.0136646
9: -0.0169965, -0.0021875, -0.0174384, -0.0036854, -0.0122161, 0.0136191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106713, upper bound: 0.0101132
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106713, upper bound: 0.0100716
time: 1.45 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.01 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0089688, upper bound: 0.0099772
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0089688, upper bound: 0.0099772
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094812, upper bound: 0.0101385
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094812, upper bound: 0.0101385
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0094294
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0098090
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0104348
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0106230
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0085249, upper bound: 0.0097266
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0084745, upper bound: 0.0097266
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0091330, upper bound: 0.0099709
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0091330, upper bound: 0.0102868
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0094824
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0098770
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097066, upper bound: 0.0105491
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099354, upper bound: 0.0107282
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099292
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099292
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100986
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100987
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0093684
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0097751
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0104346
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0106230
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0083827, upper bound: 0.0096443
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0083321, upper bound: 0.0096443
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0098701
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0102475
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0093942
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0098378
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0105467
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0107280
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0088822, upper bound: 0.0099675
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0088822, upper bound: 0.0099675
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094567, upper bound: 0.0101668
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094567, upper bound: 0.0101668
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096227, upper bound: 0.0093299
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0098950, upper bound: 0.0097779
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096227, upper bound: 0.0104682
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0098950, upper bound: 0.0106368
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0084556, upper bound: 0.0096590
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0083940, upper bound: 0.0096590
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0090594, upper bound: 0.0099242
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0090594, upper bound: 0.0103157
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096202, upper bound: 0.0093467
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0098949, upper bound: 0.0098490
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096202, upper bound: 0.0105916
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0098949, upper bound: 0.0107560
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099201
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0088138, upper bound: 0.0099201
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100948
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0094325, upper bound: 0.0100948
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0093229
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0097516
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096463, upper bound: 0.0104213
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0105986
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0083827, upper bound: 0.0096266
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0083321, upper bound: 0.0096266
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0098698
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0089503, upper bound: 0.0102454
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0093444
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0098192
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096451, upper bound: 0.0105396
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099553, upper bound: 0.0107056
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096227
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096227
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0098949
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0098949
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0096240
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0096240
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0106059
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0102702, upper bound: 0.0106059
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0102926
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0102926
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0104309
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0104309
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108088
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108088
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0110357
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0110357
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099531, upper bound: 0.0091777
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099531, upper bound: 0.0091777
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0104220
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0104220
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100770, upper bound: 0.0091777
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100770, upper bound: 0.0091777
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0108144, upper bound: 0.0104581
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0108144, upper bound: 0.0104581
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0105065, upper bound: 0.0100805
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0105065, upper bound: 0.0100805
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0109772
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0109772
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106732, upper bound: 0.0100940
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106732, upper bound: 0.0100940
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0113027, upper bound: 0.0110612
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0113027, upper bound: 0.0110612
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096463
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0093229, upper bound: 0.0096124
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0099553
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0097516, upper bound: 0.0098920
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099025, upper bound: 0.0091582
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099900, upper bound: 0.0091582
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099025, upper bound: 0.0102816
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099900, upper bound: 0.0103283
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0103795
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100707, upper bound: 0.0102904
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0105266
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100933, upper bound: 0.0104303
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108983
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0109435, upper bound: 0.0108077
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0111344
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110593, upper bound: 0.0110344
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099527, upper bound: 0.0091563
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0099527, upper bound: 0.0091414
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0105211
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106213, upper bound: 0.0104214
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100738, upper bound: 0.0091563
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0100738, upper bound: 0.0091414
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096302, upper bound: 0.0097508
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0096302, upper bound: 0.0105557
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0105064, upper bound: 0.0100940
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0105064, upper bound: 0.0100571
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0110809
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0110673, upper bound: 0.0109761
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106713, upper bound: 0.0101132
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 8, lower bound: -0.0106713, upper bound: 0.0100716
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.01
Output dim: 8, lower bound: -0.0113995, upper bound: 0.0111632

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.20 + 596.75 = 600.96 seconds
