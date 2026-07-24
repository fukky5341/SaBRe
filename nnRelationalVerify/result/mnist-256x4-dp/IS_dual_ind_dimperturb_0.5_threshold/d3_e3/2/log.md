## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01546812


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0068954, 0.0052981, -0.0068954, 0.0052981, -0.0121935, 0.0121935)
1: (0.9890214, 1.0107067, 0.9890214, 1.0107067, -0.0216853, 0.0216853)
2: (-0.0138153, 0.0060032, -0.0138153, 0.0060032, -0.0194764, 0.0194764)
3: (-0.0002160, 0.0058974, -0.0002160, 0.0058974, -0.0061134, 0.0061134)
4: (-0.0072216, 0.0093359, -0.0072216, 0.0093359, -0.0165575, 0.0165575)
5: (-0.0023608, 0.0109224, -0.0023608, 0.0109224, -0.0132832, 0.0132832)
6: (-0.0095208, 0.0035485, -0.0095208, 0.0035485, -0.0130694, 0.0130694)
7: (-0.0114768, 0.0002483, -0.0114768, 0.0002483, -0.0117251, 0.0117251)
8: (-0.0137361, 0.0164032, -0.0137361, 0.0164032, -0.0299951, 0.0299951)
9: (-0.0096246, 0.0077213, -0.0096246, 0.0077213, -0.0173459, 0.0173459)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 3.52 = 5.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0169628, upper bound: 0.0169627

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0166937, upper bound: 0.0167520
time: 2.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0166937, upper bound: 0.0166929
time: 2.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.72
Output dim: 1, lower bound: -0.0166937, upper bound: 0.0167520
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.72
Output dim: 1, lower bound: -0.0166937, upper bound: 0.0166929

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0061068, 0.0042658, -0.0067304, 0.0050822, -0.0111890, 0.0109962
1: 0.9890585, 1.0091410, 0.9890286, 1.0103792, -0.0213207, 0.0201123
2: -0.0137507, 0.0051865, -0.0138027, 0.0058324, -0.0192127, 0.0186259
3: 0.0000048, 0.0058820, -0.0001698, 0.0058944, -0.0058896, 0.0060518
4: -0.0061837, 0.0092848, -0.0070045, 0.0093259, -0.0155096, 0.0162893
5: -0.0020016, 0.0108863, -0.0022857, 0.0109153, -0.0129169, 0.0131719
6: -0.0079175, 0.0034979, -0.0091854, 0.0035386, -0.0114562, 0.0126833
7: -0.0114508, -0.0003237, -0.0114718, 0.0001286, -0.0115795, 0.0111480
8: -0.0129367, 0.0163183, -0.0135689, 0.0163866, -0.0291725, 0.0297410
9: -0.0095774, 0.0072523, -0.0096153, 0.0076232, -0.0172006, 0.0168676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158999, upper bound: 0.0162651
time: 1.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165928, upper bound: 0.0166483
time: 2.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0059233, 0.0040256, -0.0065809, 0.0048865, -0.0108098, 0.0106065
1: 0.9889683, 1.0087765, 0.9890311, 1.0100822, -0.0211139, 0.0197455
2: -0.0139074, 0.0049965, -0.0137985, 0.0056775, -0.0192573, 0.0184555
3: 0.0000562, 0.0059194, -0.0001280, 0.0058934, -0.0058372, 0.0060473
4: -0.0059422, 0.0094087, -0.0068077, 0.0093226, -0.0152648, 0.0162164
5: -0.0019180, 0.0109739, -0.0022175, 0.0109130, -0.0128311, 0.0131914
6: -0.0075445, 0.0036208, -0.0088815, 0.0035354, -0.0110799, 0.0125023
7: -0.0115139, -0.0004568, -0.0114701, 0.0000202, -0.0115341, 0.0110133
8: -0.0127506, 0.0165243, -0.0134173, 0.0163812, -0.0289897, 0.0298134
9: -0.0096918, 0.0071431, -0.0096123, 0.0075343, -0.0172260, 0.0167555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158999, upper bound: 0.0162137
time: 1.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158999, upper bound: 0.0165928
time: 2.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 1, lower bound: -0.0158999, upper bound: 0.0162651
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 1, lower bound: -0.0165928, upper bound: 0.0166483
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 1, lower bound: -0.0158999, upper bound: 0.0162137
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 1, lower bound: -0.0158999, upper bound: 0.0165928

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0056210, 0.0036298, -0.0052853, 0.0031904, -0.0088113, 0.0089151
1: 0.9890715, 1.0081762, 0.9887094, 1.0075099, -0.0184384, 0.0194668
2: -0.0137280, 0.0046834, -0.0143578, 0.0043357, -0.0176403, 0.0185603
3: 0.0001408, 0.0058766, 0.0002347, 0.0060268, -0.0058860, 0.0056418
4: -0.0055442, 0.0092669, -0.0051024, 0.0097647, -0.0153089, 0.0143693
5: -0.0017803, 0.0108736, -0.0016275, 0.0112255, -0.0130059, 0.0125011
6: -0.0069298, 0.0034801, -0.0062474, 0.0039740, -0.0109038, 0.0097275
7: -0.0114417, -0.0006762, -0.0116951, -0.0009196, -0.0105221, 0.0110189
8: -0.0124441, 0.0162885, -0.0121038, 0.0171161, -0.0293949, 0.0282382
9: -0.0095609, 0.0069633, -0.0100204, 0.0067637, -0.0163245, 0.0169837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157052, upper bound: 0.0161660
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158809, upper bound: 0.0162455
time: 1.98 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0061068, 0.0042658, -0.0063593, 0.0045963, -0.0107032, 0.0106251
1: 0.9890585, 1.0091410, 0.9890375, 1.0096424, -0.0205839, 0.0201035
2: -0.0137507, 0.0051865, -0.0137872, 0.0054480, -0.0187644, 0.0185980
3: 0.0000048, 0.0058820, -0.0000659, 0.0058907, -0.0058859, 0.0059479
4: -0.0061837, 0.0092848, -0.0065160, 0.0093137, -0.0154974, 0.0158008
5: -0.0020016, 0.0108863, -0.0021166, 0.0109067, -0.0129084, 0.0130029
6: -0.0079175, 0.0034979, -0.0084309, 0.0035265, -0.0114441, 0.0119288
7: -0.0114508, -0.0003237, -0.0114655, -0.0001406, -0.0113102, 0.0111418
8: -0.0129367, 0.0163183, -0.0131927, 0.0163663, -0.0291522, 0.0293543
9: -0.0095774, 0.0072523, -0.0096041, 0.0074024, -0.0169798, 0.0168564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0159802
time: 2.27 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0166483
time: 2.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0054306, 0.0033805, -0.0051281, 0.0030733, -0.0085039, 0.0085086
1: 0.9889819, 1.0077982, 0.9887125, 1.0071976, -0.0182157, 0.0190857
2: -0.0138841, 0.0044862, -0.0143529, 0.0041729, -0.0176769, 0.0183840
3: 0.0001941, 0.0059138, 0.0002788, 0.0060256, -0.0058315, 0.0056350
4: -0.0052936, 0.0093902, -0.0048955, 0.0097607, -0.0150544, 0.0142858
5: -0.0016936, 0.0109608, -0.0015559, 0.0112228, -0.0129164, 0.0125167
6: -0.0065427, 0.0036025, -0.0059277, 0.0039701, -0.0105128, 0.0095302
7: -0.0115045, -0.0008143, -0.0116931, -0.0010337, -0.0104708, 0.0108788
8: -0.0122511, 0.0164936, -0.0119444, 0.0171095, -0.0292063, 0.0282973
9: -0.0096747, 0.0068501, -0.0100168, 0.0066702, -0.0163449, 0.0168669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157051, upper bound: 0.0161320
time: 2.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158808, upper bound: 0.0161938
time: 2.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0059233, 0.0040256, -0.0062121, 0.0044036, -0.0103270, 0.0102377
1: 0.9889683, 1.0087765, 0.9890400, 1.0093501, -0.0203817, 0.0197366
2: -0.0139074, 0.0049965, -0.0137830, 0.0052955, -0.0188124, 0.0184276
3: 0.0000562, 0.0059194, -0.0000247, 0.0058897, -0.0058335, 0.0059441
4: -0.0059422, 0.0094087, -0.0063222, 0.0093103, -0.0152525, 0.0157309
5: -0.0019180, 0.0109739, -0.0020496, 0.0109043, -0.0128224, 0.0130235
6: -0.0075445, 0.0036208, -0.0081316, 0.0035232, -0.0110677, 0.0117524
7: -0.0115139, -0.0004568, -0.0114639, -0.0002474, -0.0112665, 0.0110070
8: -0.0127506, 0.0165243, -0.0130434, 0.0163608, -0.0289692, 0.0294278
9: -0.0096918, 0.0071431, -0.0096010, 0.0073149, -0.0170067, 0.0167442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0158999
time: 2.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0165926
time: 1.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0157052, upper bound: 0.0161660
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0158809, upper bound: 0.0162455
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0159802
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0166483
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0157051, upper bound: 0.0161320
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0158808, upper bound: 0.0161938
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0158999
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.69
Output dim: 1, lower bound: -0.0162137, upper bound: 0.0165926

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0055591, 0.0035488, -0.0051383, 0.0030073, -0.0085664, 0.0086871
1: 0.9897196, 1.0080534, 0.9888788, 1.0072180, -0.0174984, 0.0191746
2: -0.0126007, 0.0046193, -0.0140634, 0.0041835, -0.0163807, 0.0181665
3: 0.0001581, 0.0056078, 0.0002759, 0.0059565, -0.0057984, 0.0053319
4: -0.0054628, 0.0083759, -0.0049089, 0.0095319, -0.0149948, 0.0132848
5: -0.0017522, 0.0102437, -0.0015605, 0.0110610, -0.0128132, 0.0118043
6: -0.0068040, 0.0025961, -0.0059485, 0.0037431, -0.0105471, 0.0085446
7: -0.0109882, -0.0007210, -0.0115767, -0.0010263, -0.0099619, 0.0108556
8: -0.0123814, 0.0148073, -0.0119548, 0.0167292, -0.0289396, 0.0266069
9: -0.0087383, 0.0069265, -0.0098056, 0.0066763, -0.0154146, 0.0167321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157052, upper bound: 0.0161660
time: 1.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157052, upper bound: 0.0161660
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0055146, 0.0034905, -0.0052853, 0.0031904, -0.0087049, 0.0087758
1: 0.9891497, 1.0079650, 0.9887094, 1.0075099, -0.0183602, 0.0192555
2: -0.0135920, 0.0045732, -0.0143578, 0.0043357, -0.0174191, 0.0184268
3: 0.0001706, 0.0058441, 0.0002347, 0.0060268, -0.0058562, 0.0056094
4: -0.0054042, 0.0091594, -0.0051024, 0.0097647, -0.0151688, 0.0142618
5: -0.0017319, 0.0107976, -0.0016275, 0.0112255, -0.0129574, 0.0124251
6: -0.0067134, 0.0033735, -0.0062474, 0.0039740, -0.0106874, 0.0096208
7: -0.0113870, -0.0007533, -0.0116951, -0.0009196, -0.0104674, 0.0109418
8: -0.0123362, 0.0161099, -0.0121038, 0.0171161, -0.0292857, 0.0280365
9: -0.0094617, 0.0069000, -0.0100204, 0.0067637, -0.0162254, 0.0169204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158809, upper bound: 0.0162455
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158809, upper bound: 0.0162449
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046133, 0.0030621, -0.0063593, 0.0045963, -0.0092097, 0.0094214
1: 0.9887407, 1.0061755, 0.9890375, 1.0096424, -0.0209017, 0.0171381
2: -0.0143037, 0.0036398, -0.0137872, 0.0054480, -0.0192812, 0.0170089
3: 0.0004229, 0.0060138, -0.0000659, 0.0058907, -0.0054678, 0.0060798
4: -0.0042180, 0.0097219, -0.0065160, 0.0093137, -0.0135317, 0.0162379
5: -0.0013214, 0.0111953, -0.0021166, 0.0109067, -0.0122281, 0.0133119
6: -0.0048812, 0.0039315, -0.0084309, 0.0035265, -0.0084077, 0.0123624
7: -0.0116733, -0.0014071, -0.0114655, -0.0001406, -0.0115327, 0.0100584
8: -0.0114225, 0.0170449, -0.0131927, 0.0163663, -0.0276307, 0.0300822
9: -0.0099809, 0.0063640, -0.0096041, 0.0074024, -0.0173834, 0.0159681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157010, upper bound: 0.0157540
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158707, upper bound: 0.0159584
time: 2.24 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0057315, 0.0037744, -0.0063593, 0.0045963, -0.0103278, 0.0101337
1: 0.9890676, 1.0083957, 0.9890375, 1.0096424, -0.0205747, 0.0193582
2: -0.0137349, 0.0047978, -0.0137872, 0.0054480, -0.0187391, 0.0181450
3: 0.0001099, 0.0058782, -0.0000659, 0.0058907, -0.0057808, 0.0059441
4: -0.0056897, 0.0092723, -0.0065160, 0.0093137, -0.0150033, 0.0157883
5: -0.0018307, 0.0108774, -0.0021166, 0.0109067, -0.0127374, 0.0129941
6: -0.0071545, 0.0034855, -0.0084309, 0.0035265, -0.0106810, 0.0119164
7: -0.0114445, -0.0005960, -0.0114655, -0.0001406, -0.0113038, 0.0108695
8: -0.0125561, 0.0162975, -0.0131927, 0.0163663, -0.0287598, 0.0293335
9: -0.0095659, 0.0070290, -0.0096041, 0.0074024, -0.0169683, 0.0166331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157010, upper bound: 0.0164368
time: 1.86 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158707, upper bound: 0.0166304
time: 1.94 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0053769, 0.0033102, -0.0049807, 0.0030062, -0.0083830, 0.0082908
1: 0.9896488, 1.0076914, 0.9888816, 1.0069047, -0.0172560, 0.0188098
2: -0.0127239, 0.0044305, -0.0140585, 0.0040202, -0.0163858, 0.0179945
3: 0.0002091, 0.0056372, 0.0003200, 0.0059554, -0.0057462, 0.0053171
4: -0.0052229, 0.0084733, -0.0047014, 0.0095281, -0.0147510, 0.0131747
5: -0.0016692, 0.0103126, -0.0014887, 0.0110583, -0.0127274, 0.0118013
6: -0.0064335, 0.0026928, -0.0056279, 0.0037392, -0.0101727, 0.0083207
7: -0.0110377, -0.0008532, -0.0115747, -0.0011406, -0.0098971, 0.0107214
8: -0.0121966, 0.0149692, -0.0117949, 0.0167227, -0.0287595, 0.0266246
9: -0.0088282, 0.0068181, -0.0098020, 0.0065825, -0.0154107, 0.0166201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0144123, upper bound: 0.0148104
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156803, upper bound: 0.0161134
time: 2.04 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0053149, 0.0032291, -0.0051281, 0.0030733, -0.0083882, 0.0083572
1: 0.9890607, 1.0075686, 0.9887125, 1.0071976, -0.0181369, 0.0188561
2: -0.0137470, 0.0043664, -0.0143529, 0.0041729, -0.0174543, 0.0182650
3: 0.0002265, 0.0058811, 0.0002788, 0.0060256, -0.0057991, 0.0056023
4: -0.0051414, 0.0092819, -0.0048955, 0.0097607, -0.0149021, 0.0141774
5: -0.0016409, 0.0108842, -0.0015559, 0.0112228, -0.0128637, 0.0124401
6: -0.0063075, 0.0034950, -0.0059277, 0.0039701, -0.0102776, 0.0094228
7: -0.0114494, -0.0008982, -0.0116931, -0.0010337, -0.0104157, 0.0107949
8: -0.0121338, 0.0163135, -0.0119444, 0.0171095, -0.0290894, 0.0280960
9: -0.0095747, 0.0067813, -0.0100168, 0.0066702, -0.0162449, 0.0167981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157088, upper bound: 0.0159303
time: 2.20 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157088, upper bound: 0.0161938
time: 1.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0043953, 0.0030951, -0.0062121, 0.0044036, -0.0087990, 0.0093072
1: 0.9886574, 1.0057428, 0.9890400, 1.0093501, -0.0206926, 0.0167028
2: -0.0144483, 0.0034140, -0.0137830, 0.0052955, -0.0193228, 0.0168371
3: 0.0004839, 0.0060483, -0.0000247, 0.0058897, -0.0054058, 0.0060730
4: -0.0039310, 0.0098362, -0.0063222, 0.0093103, -0.0132414, 0.0161584
5: -0.0012221, 0.0112761, -0.0020496, 0.0109043, -0.0121264, 0.0133257
6: -0.0044379, 0.0040449, -0.0081316, 0.0035232, -0.0079612, 0.0121765
7: -0.0117315, -0.0015652, -0.0114639, -0.0002474, -0.0114841, 0.0098986
8: -0.0112015, 0.0172349, -0.0130434, 0.0163608, -0.0274159, 0.0301364
9: -0.0100864, 0.0062344, -0.0096010, 0.0073149, -0.0174013, 0.0158354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157003, upper bound: 0.0157051
time: 1.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158705, upper bound: 0.0158808
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0055518, 0.0035392, -0.0062121, 0.0044036, -0.0099554, 0.0097513
1: 0.9889778, 1.0080388, 0.9890400, 1.0093501, -0.0203723, 0.0189988
2: -0.0138915, 0.0046117, -0.0137830, 0.0052955, -0.0187877, 0.0179780
3: 0.0001602, 0.0059155, -0.0000247, 0.0058897, -0.0057295, 0.0059402
4: -0.0054532, 0.0093961, -0.0063222, 0.0093103, -0.0147635, 0.0157183
5: -0.0017488, 0.0109650, -0.0020496, 0.0109043, -0.0126532, 0.0130146
6: -0.0067891, 0.0036083, -0.0081316, 0.0035232, -0.0103124, 0.0117399
7: -0.0115075, -0.0007263, -0.0114639, -0.0002474, -0.0112601, 0.0107375
8: -0.0123740, 0.0165033, -0.0130434, 0.0163608, -0.0285810, 0.0294068
9: -0.0096801, 0.0069222, -0.0096010, 0.0073149, -0.0169950, 0.0165232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157003, upper bound: 0.0163974
time: 2.29 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158705, upper bound: 0.0165755
time: 2.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.01 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157052, upper bound: 0.0161660
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157052, upper bound: 0.0161660
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0158809, upper bound: 0.0162455
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0158809, upper bound: 0.0162449
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157010, upper bound: 0.0157540
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0158707, upper bound: 0.0159584
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157010, upper bound: 0.0164368
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0158707, upper bound: 0.0166304
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0144123, upper bound: 0.0148104
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0156803, upper bound: 0.0161134
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157088, upper bound: 0.0159303
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157088, upper bound: 0.0161938
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157003, upper bound: 0.0157051
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0158705, upper bound: 0.0158808
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0157003, upper bound: 0.0163974
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.01
Output dim: 1, lower bound: -0.0158705, upper bound: 0.0165755

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0055591, 0.0035488, -0.0044929, 0.0029949, -0.0085540, 0.0080417
1: 0.9897196, 1.0080534, 0.9889099, 1.0059364, -0.0162168, 0.0191435
2: -0.0126007, 0.0046193, -0.0140091, 0.0035151, -0.0157020, 0.0180999
3: 0.0001581, 0.0056078, 0.0004566, 0.0059436, -0.0057855, 0.0051512
4: -0.0054628, 0.0083759, -0.0040595, 0.0094890, -0.0149519, 0.0124354
5: -0.0017522, 0.0102437, -0.0012666, 0.0110307, -0.0127828, 0.0115103
6: -0.0068040, 0.0025961, -0.0046363, 0.0037005, -0.0105046, 0.0072325
7: -0.0109882, -0.0007210, -0.0115548, -0.0014944, -0.0094937, 0.0108338
8: -0.0123814, 0.0148073, -0.0113005, 0.0166579, -0.0288685, 0.0259486
9: -0.0087383, 0.0069265, -0.0097660, 0.0062924, -0.0150307, 0.0166925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145933, upper bound: 0.0154038
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156802, upper bound: 0.0161472
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0055591, 0.0035488, -0.0042865, 0.0030251, -0.0085843, 0.0078353
1: 0.9897196, 1.0080534, 0.9888337, 1.0055264, -0.0158069, 0.0192197
2: -0.0126007, 0.0046193, -0.0141416, 0.0033013, -0.0155310, 0.0182746
3: 0.0001581, 0.0056078, 0.0005144, 0.0059752, -0.0058171, 0.0050934
4: -0.0054628, 0.0083759, -0.0037878, 0.0095937, -0.0150565, 0.0121637
5: -0.0017522, 0.0102437, -0.0011725, 0.0111047, -0.0128569, 0.0114163
6: -0.0068040, 0.0025961, -0.0042166, 0.0038044, -0.0106084, 0.0068128
7: -0.0109882, -0.0007210, -0.0116081, -0.0016442, -0.0093440, 0.0108871
8: -0.0123814, 0.0148073, -0.0110912, 0.0168319, -0.0290534, 0.0257523
9: -0.0087383, 0.0069265, -0.0098626, 0.0061696, -0.0149080, 0.0167891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145933, upper bound: 0.0154038
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156802, upper bound: 0.0161466
time: 2.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0055146, 0.0034905, -0.0046417, 0.0030621, -0.0085767, 0.0081321
1: 0.9891497, 1.0079650, 0.9887407, 1.0062318, -0.0170820, 0.0192243
2: -0.0135920, 0.0045732, -0.0143037, 0.0036691, -0.0167422, 0.0183615
3: 0.0001706, 0.0058441, 0.0004149, 0.0060138, -0.0058433, 0.0054292
4: -0.0054042, 0.0091594, -0.0042553, 0.0097219, -0.0151260, 0.0134147
5: -0.0017319, 0.0107976, -0.0013343, 0.0111953, -0.0129272, 0.0121319
6: -0.0067134, 0.0033735, -0.0049388, 0.0039315, -0.0106450, 0.0083122
7: -0.0113870, -0.0007533, -0.0116733, -0.0013865, -0.0100005, 0.0109200
8: -0.0123362, 0.0161099, -0.0114513, 0.0170449, -0.0292149, 0.0273806
9: -0.0094617, 0.0069000, -0.0099809, 0.0063809, -0.0158425, 0.0168809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0155918, upper bound: 0.0159463
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158614, upper bound: 0.0162251
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0055146, 0.0034905, -0.0044291, 0.0030951, -0.0086096, 0.0079196
1: 0.9891497, 1.0079650, 0.9886574, 1.0058095, -0.0166598, 0.0193076
2: -0.0135920, 0.0045732, -0.0144483, 0.0034490, -0.0165642, 0.0185433
3: 0.0001706, 0.0058441, 0.0004745, 0.0060483, -0.0058778, 0.0053697
4: -0.0054042, 0.0091594, -0.0039754, 0.0098362, -0.0152403, 0.0131348
5: -0.0017319, 0.0107976, -0.0012375, 0.0112761, -0.0130080, 0.0120351
6: -0.0067134, 0.0033735, -0.0045065, 0.0040449, -0.0107583, 0.0078800
7: -0.0113870, -0.0007533, -0.0117315, -0.0015407, -0.0098463, 0.0109782
8: -0.0123362, 0.0161099, -0.0112357, 0.0172349, -0.0294149, 0.0271776
9: -0.0094617, 0.0069000, -0.0100864, 0.0062544, -0.0157161, 0.0169864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0155918, upper bound: 0.0159463
time: 1.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158614, upper bound: 0.0162251
time: 2.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0044692, 0.0029949, -0.0062524, 0.0044563, -0.0089255, 0.0092473
1: 0.9889099, 1.0058893, 0.9896853, 1.0094299, -0.0205200, 0.0162040
2: -0.0140091, 0.0034905, -0.0126604, 0.0053372, -0.0188406, 0.0157570
3: 0.0004632, 0.0059436, -0.0000360, 0.0056220, -0.0051588, 0.0059796
4: -0.0040283, 0.0094890, -0.0063752, 0.0084231, -0.0124514, 0.0158643
5: -0.0012557, 0.0110307, -0.0020679, 0.0102771, -0.0115329, 0.0130986
6: -0.0045881, 0.0037005, -0.0082134, 0.0026430, -0.0072311, 0.0119140
7: -0.0115548, -0.0015116, -0.0110122, -0.0002182, -0.0113366, 0.0095006
8: -0.0112764, 0.0166579, -0.0130842, 0.0148858, -0.0260027, 0.0295776
9: -0.0097660, 0.0062783, -0.0087819, 0.0073388, -0.0171048, 0.0150602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0149247, upper bound: 0.0145724
time: 1.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0157238
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046133, 0.0030621, -0.0062475, 0.0044499, -0.0090633, 0.0093096
1: 0.9887407, 1.0061755, 0.9891158, 1.0094204, -0.0206797, 0.0170597
2: -0.0143037, 0.0036398, -0.0136511, 0.0053322, -0.0191421, 0.0167895
3: 0.0004229, 0.0060138, -0.0000346, 0.0058582, -0.0054354, 0.0060485
4: -0.0042180, 0.0097219, -0.0063688, 0.0092061, -0.0134241, 0.0160907
5: -0.0013214, 0.0111953, -0.0020657, 0.0108307, -0.0121521, 0.0132609
6: -0.0048812, 0.0039315, -0.0082035, 0.0034198, -0.0083010, 0.0121350
7: -0.0116733, -0.0014071, -0.0114108, -0.0002217, -0.0114516, 0.0100037
8: -0.0114225, 0.0170449, -0.0130793, 0.0161875, -0.0274295, 0.0299673
9: -0.0099809, 0.0063640, -0.0095048, 0.0073359, -0.0173168, 0.0158688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0157498
time: 1.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0159577
time: 1.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0055810, 0.0035774, -0.0062524, 0.0044563, -0.0100373, 0.0098298
1: 0.9892446, 1.0080968, 0.9896853, 1.0094299, -0.0201854, 0.0184115
2: -0.0134272, 0.0046419, -0.0126604, 0.0053372, -0.0182966, 0.0168584
3: 0.0001520, 0.0058048, -0.0000360, 0.0056220, -0.0054700, 0.0058408
4: -0.0054916, 0.0090291, -0.0063752, 0.0084231, -0.0139147, 0.0154043
5: -0.0017621, 0.0107055, -0.0020679, 0.0102771, -0.0120392, 0.0127734
6: -0.0068485, 0.0032442, -0.0082134, 0.0026430, -0.0094915, 0.0114576
7: -0.0113207, -0.0007052, -0.0110122, -0.0002182, -0.0111025, 0.0103070
8: -0.0124036, 0.0158933, -0.0130842, 0.0148858, -0.0271251, 0.0288157
9: -0.0093414, 0.0069395, -0.0087819, 0.0073388, -0.0166802, 0.0157215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162114, upper bound: 0.0161231
time: 2.45 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163698
time: 1.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0057315, 0.0037744, -0.0062475, 0.0044499, -0.0101814, 0.0100219
1: 0.9890676, 1.0083957, 0.9891158, 1.0094204, -0.0203528, 0.0192799
2: -0.0137349, 0.0047978, -0.0136511, 0.0053322, -0.0185980, 0.0179351
3: 0.0001099, 0.0058782, -0.0000346, 0.0058582, -0.0057484, 0.0059128
4: -0.0056897, 0.0092723, -0.0063688, 0.0092061, -0.0148957, 0.0156411
5: -0.0018307, 0.0108774, -0.0020657, 0.0108307, -0.0126613, 0.0129431
6: -0.0071545, 0.0034855, -0.0082035, 0.0034198, -0.0105743, 0.0116890
7: -0.0114445, -0.0005960, -0.0114108, -0.0002217, -0.0112227, 0.0108148
8: -0.0125561, 0.0162975, -0.0130793, 0.0161875, -0.0285606, 0.0292187
9: -0.0095659, 0.0070290, -0.0095048, 0.0073359, -0.0169018, 0.0165338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0166141
time: 2.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156696, upper bound: 0.0166296
time: 2.31 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0053769, 0.0033102, -0.0048836, 0.0029657, -0.0083426, 0.0081938
1: 0.9896488, 1.0076914, 0.9889837, 1.0067121, -0.0170633, 0.0187077
2: -0.0127239, 0.0044305, -0.0138808, 0.0039197, -0.0162863, 0.0177709
3: 0.0002091, 0.0056372, 0.0003472, 0.0059130, -0.0057039, 0.0052899
4: -0.0052229, 0.0084733, -0.0045737, 0.0093876, -0.0146105, 0.0130470
5: -0.0016692, 0.0103126, -0.0014445, 0.0109590, -0.0126282, 0.0117571
6: -0.0064335, 0.0026928, -0.0054306, 0.0035999, -0.0100334, 0.0081234
7: -0.0110377, -0.0008532, -0.0115032, -0.0012110, -0.0098267, 0.0106500
8: -0.0121966, 0.0149692, -0.0116965, 0.0164893, -0.0285168, 0.0265266
9: -0.0088282, 0.0068181, -0.0096724, 0.0065248, -0.0153530, 0.0164905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145456, upper bound: 0.0152773
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145456, upper bound: 0.0152773
time: 2.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053149, 0.0032291, -0.0051297, 0.0029866, -0.0083015, 0.0083588
1: 0.9890607, 1.0075686, 0.9893498, 1.0072007, -0.0181400, 0.0182188
2: -0.0137470, 0.0043664, -0.0132440, 0.0041746, -0.0175143, 0.0171649
3: 0.0002265, 0.0058811, 0.0002783, 0.0057612, -0.0055347, 0.0056028
4: -0.0051414, 0.0092819, -0.0048976, 0.0088843, -0.0140256, 0.0141795
5: -0.0016409, 0.0108842, -0.0015566, 0.0106032, -0.0122441, 0.0124408
6: -0.0063075, 0.0034950, -0.0059310, 0.0031005, -0.0094080, 0.0094260
7: -0.0114494, -0.0008982, -0.0112469, -0.0010325, -0.0104169, 0.0103488
8: -0.0121338, 0.0163135, -0.0119460, 0.0156525, -0.0276304, 0.0281218
9: -0.0095747, 0.0067813, -0.0092077, 0.0066711, -0.0162459, 0.0159890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150674, upper bound: 0.0151368
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0158910
time: 1.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053149, 0.0032291, -0.0050256, 0.0030383, -0.0083533, 0.0082547
1: 0.9890607, 1.0075686, 0.9888005, 1.0069941, -0.0179334, 0.0187681
2: -0.0137470, 0.0043664, -0.0141995, 0.0040668, -0.0173482, 0.0180386
3: 0.0002265, 0.0058811, 0.0003075, 0.0059890, -0.0057625, 0.0055736
4: -0.0051414, 0.0092819, -0.0047606, 0.0096395, -0.0147809, 0.0140425
5: -0.0016409, 0.0108842, -0.0015092, 0.0111370, -0.0127780, 0.0123934
6: -0.0063075, 0.0034950, -0.0057194, 0.0038498, -0.0101573, 0.0092144
7: -0.0114494, -0.0008982, -0.0116314, -0.0011080, -0.0103414, 0.0107333
8: -0.0121338, 0.0163135, -0.0118405, 0.0169080, -0.0288723, 0.0279923
9: -0.0095747, 0.0067813, -0.0099049, 0.0066092, -0.0161840, 0.0166862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150674, upper bound: 0.0157995
time: 2.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0160377
time: 2.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042447, 0.0030251, -0.0061120, 0.0042726, -0.0085173, 0.0091372
1: 0.9888337, 1.0054436, 0.9896877, 1.0091513, -0.0203176, 0.0157558
2: -0.0141416, 0.0032580, -0.0126561, 0.0051919, -0.0188803, 0.0155778
3: 0.0005261, 0.0059752, 0.0000033, 0.0056210, -0.0050949, 0.0059719
4: -0.0037328, 0.0095937, -0.0061905, 0.0084197, -0.0121524, 0.0157843
5: -0.0011535, 0.0111047, -0.0020040, 0.0102747, -0.0114282, 0.0131087
6: -0.0041317, 0.0038044, -0.0079281, 0.0026396, -0.0067713, 0.0117325
7: -0.0116081, -0.0016745, -0.0110105, -0.0003200, -0.0112881, 0.0093360
8: -0.0110488, 0.0168319, -0.0129420, 0.0148801, -0.0257809, 0.0296218
9: -0.0098626, 0.0061448, -0.0087788, 0.0072554, -0.0171180, 0.0149236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148104, upper bound: 0.0144122
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0156802
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0043953, 0.0030951, -0.0061008, 0.0042579, -0.0086532, 0.0091958
1: 0.9886574, 1.0057428, 0.9891183, 1.0091290, -0.0204716, 0.0166245
2: -0.0144483, 0.0034140, -0.0136469, 0.0051802, -0.0191809, 0.0166212
3: 0.0004839, 0.0060483, 0.0000065, 0.0058572, -0.0053733, 0.0060418
4: -0.0039310, 0.0098362, -0.0061757, 0.0092027, -0.0131338, 0.0160119
5: -0.0012221, 0.0112761, -0.0019989, 0.0108283, -0.0120504, 0.0132749
6: -0.0044379, 0.0040449, -0.0079052, 0.0034165, -0.0078544, 0.0119501
7: -0.0117315, -0.0015652, -0.0114091, -0.0003281, -0.0114034, 0.0098439
8: -0.0112015, 0.0172349, -0.0129305, 0.0161819, -0.0272168, 0.0300209
9: -0.0100864, 0.0062344, -0.0095017, 0.0072487, -0.0173351, 0.0157360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159303, upper bound: 0.0157088
time: 1.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0158808
time: 1.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0054049, 0.0033470, -0.0061120, 0.0042726, -0.0096776, 0.0094590
1: 0.9891547, 1.0077474, 0.9896877, 1.0091513, -0.0199966, 0.0180597
2: -0.0135835, 0.0044596, -0.0126561, 0.0051919, -0.0183490, 0.0167256
3: 0.0002013, 0.0058421, 0.0000033, 0.0056210, -0.0054197, 0.0058388
4: -0.0052599, 0.0091526, -0.0061905, 0.0084197, -0.0136795, 0.0153432
5: -0.0016819, 0.0107929, -0.0020040, 0.0102747, -0.0119566, 0.0127969
6: -0.0064906, 0.0033668, -0.0079281, 0.0026396, -0.0091301, 0.0112949
7: -0.0113836, -0.0008329, -0.0110105, -0.0003200, -0.0110636, 0.0101776
8: -0.0122251, 0.0160986, -0.0129420, 0.0148801, -0.0269510, 0.0288933
9: -0.0094554, 0.0068348, -0.0087788, 0.0072554, -0.0167108, 0.0156136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162112, upper bound: 0.0160729
time: 2.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163286
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0055518, 0.0035392, -0.0061008, 0.0042579, -0.0098097, 0.0096400
1: 0.9889778, 1.0080388, 0.9891183, 1.0091290, -0.0201513, 0.0189204
2: -0.0138915, 0.0046117, -0.0136469, 0.0051802, -0.0186432, 0.0177665
3: 0.0001602, 0.0059155, 0.0000065, 0.0058572, -0.0056971, 0.0059091
4: -0.0054532, 0.0093961, -0.0061757, 0.0092027, -0.0146559, 0.0155718
5: -0.0017488, 0.0109650, -0.0019989, 0.0108283, -0.0125771, 0.0129638
6: -0.0067891, 0.0036083, -0.0079052, 0.0034165, -0.0102056, 0.0115135
7: -0.0115075, -0.0007263, -0.0114091, -0.0003281, -0.0111794, 0.0106827
8: -0.0123740, 0.0165033, -0.0129305, 0.0161819, -0.0283821, 0.0292916
9: -0.0096801, 0.0069222, -0.0095017, 0.0072487, -0.0169288, 0.0164238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165553
time: 2.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165756
time: 2.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.00 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0145933, upper bound: 0.0154038
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0156802, upper bound: 0.0161472
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0145933, upper bound: 0.0154038
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0156802, upper bound: 0.0161466
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0155918, upper bound: 0.0159463
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0158614, upper bound: 0.0162251
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0155918, upper bound: 0.0159463
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0158614, upper bound: 0.0162251
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0149247, upper bound: 0.0145724
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0157238
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0157498
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0159577
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0162114, upper bound: 0.0161231
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163698
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0166141
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0156696, upper bound: 0.0166296
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0145456, upper bound: 0.0152773
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0145456, upper bound: 0.0152773
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0150674, upper bound: 0.0151368
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0158910
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0150674, upper bound: 0.0157995
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0160377
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0148104, upper bound: 0.0144122
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0156802
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0159303, upper bound: 0.0157088
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0158808
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0162112, upper bound: 0.0160729
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163286
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165553
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.00
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165756

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0054655, 0.0034262, -0.0044929, 0.0029949, -0.0084604, 0.0079191
1: 0.9898159, 1.0078675, 0.9889099, 1.0059364, -0.0161205, 0.0189576
2: -0.0124335, 0.0045223, -0.0140091, 0.0035151, -0.0154940, 0.0180043
3: 0.0001843, 0.0055679, 0.0004566, 0.0059436, -0.0057593, 0.0051113
4: -0.0053396, 0.0082438, -0.0040595, 0.0094890, -0.0148286, 0.0123032
5: -0.0017095, 0.0101503, -0.0012666, 0.0110307, -0.0127402, 0.0114169
6: -0.0066136, 0.0024651, -0.0046363, 0.0037005, -0.0103142, 0.0071014
7: -0.0109209, -0.0007890, -0.0115548, -0.0014944, -0.0094265, 0.0107659
8: -0.0122864, 0.0145877, -0.0113005, 0.0166579, -0.0287748, 0.0257228
9: -0.0086164, 0.0068708, -0.0097660, 0.0062924, -0.0149088, 0.0166368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147226, upper bound: 0.0151427
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0147226, upper bound: 0.0161688
time: 2.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0054655, 0.0034262, -0.0042865, 0.0030251, -0.0084906, 0.0077127
1: 0.9898159, 1.0078675, 0.9888337, 1.0055264, -0.0157105, 0.0190337
2: -0.0124335, 0.0045223, -0.0141416, 0.0033013, -0.0153267, 0.0181789
3: 0.0001843, 0.0055679, 0.0005144, 0.0059752, -0.0057909, 0.0050535
4: -0.0053396, 0.0082438, -0.0037878, 0.0095937, -0.0149333, 0.0120315
5: -0.0017095, 0.0101503, -0.0011725, 0.0111047, -0.0128142, 0.0113229
6: -0.0066136, 0.0024651, -0.0042166, 0.0038044, -0.0104180, 0.0066817
7: -0.0109209, -0.0007890, -0.0116081, -0.0016442, -0.0092768, 0.0108191
8: -0.0122864, 0.0145877, -0.0110912, 0.0168319, -0.0289597, 0.0255276
9: -0.0086164, 0.0068708, -0.0098626, 0.0061696, -0.0147860, 0.0167334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0144282, upper bound: 0.0148717
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0144282, upper bound: 0.0161472
time: 1.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0057372, 0.0037819, -0.0044872, 0.0030028, -0.0087400, 0.0082691
1: 0.9896498, 1.0084070, 0.9888903, 1.0059251, -0.0162752, 0.0195167
2: -0.0127221, 0.0048037, -0.0140435, 0.0035092, -0.0157064, 0.0183142
3: 0.0001083, 0.0056367, 0.0004582, 0.0059518, -0.0058436, 0.0051785
4: -0.0056972, 0.0084718, -0.0040520, 0.0095162, -0.0152134, 0.0125238
5: -0.0018333, 0.0103116, -0.0012640, 0.0110499, -0.0128831, 0.0115755
6: -0.0071660, 0.0026913, -0.0046248, 0.0037275, -0.0108935, 0.0073161
7: -0.0110370, -0.0005919, -0.0115686, -0.0014985, -0.0095384, 0.0109768
8: -0.0125619, 0.0149668, -0.0112947, 0.0167031, -0.0290992, 0.0260818
9: -0.0088269, 0.0070324, -0.0097911, 0.0062890, -0.0151159, 0.0168235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0151439, upper bound: 0.0152752
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151439, upper bound: 0.0158992
time: 2.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0054161, 0.0033615, -0.0046417, 0.0030621, -0.0084782, 0.0080032
1: 0.9892527, 1.0077693, 0.9887407, 1.0062318, -0.0169791, 0.0190287
2: -0.0134127, 0.0044711, -0.0143037, 0.0036691, -0.0165235, 0.0182609
3: 0.0001982, 0.0058014, 0.0004149, 0.0060138, -0.0058157, 0.0053864
4: -0.0052745, 0.0090176, -0.0042553, 0.0097219, -0.0149964, 0.0132729
5: -0.0016870, 0.0106974, -0.0013343, 0.0111953, -0.0128823, 0.0120317
6: -0.0065132, 0.0032328, -0.0049388, 0.0039315, -0.0104447, 0.0081716
7: -0.0113148, -0.0008248, -0.0116733, -0.0013865, -0.0099283, 0.0108485
8: -0.0122364, 0.0158742, -0.0114513, 0.0170449, -0.0291152, 0.0271331
9: -0.0093308, 0.0068415, -0.0099809, 0.0063809, -0.0157117, 0.0168224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157471, upper bound: 0.0159431
time: 1.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157471, upper bound: 0.0161316
time: 2.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057372, 0.0037819, -0.0042732, 0.0030361, -0.0087733, 0.0080551
1: 0.9896498, 1.0084070, 0.9888061, 1.0055000, -0.0158501, 0.0196009
2: -0.0127221, 0.0048037, -0.0141898, 0.0032875, -0.0155267, 0.0185029
3: 0.0001083, 0.0056367, 0.0005181, 0.0059867, -0.0058784, 0.0051186
4: -0.0056972, 0.0084718, -0.0037702, 0.0096318, -0.0153290, 0.0122421
5: -0.0018333, 0.0103116, -0.0011664, 0.0111316, -0.0129649, 0.0114780
6: -0.0071660, 0.0026913, -0.0041895, 0.0038422, -0.0110082, 0.0068809
7: -0.0110370, -0.0005919, -0.0116275, -0.0016538, -0.0093832, 0.0110356
8: -0.0125619, 0.0149668, -0.0110777, 0.0168953, -0.0292984, 0.0258789
9: -0.0088269, 0.0070324, -0.0098978, 0.0061617, -0.0149886, 0.0169302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150843, upper bound: 0.0152107
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150843, upper bound: 0.0158639
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0054161, 0.0033615, -0.0044291, 0.0030951, -0.0085111, 0.0077906
1: 0.9892527, 1.0077693, 0.9886574, 1.0058095, -0.0165569, 0.0191119
2: -0.0134127, 0.0044711, -0.0144483, 0.0034490, -0.0163501, 0.0184427
3: 0.0001982, 0.0058014, 0.0004745, 0.0060483, -0.0058502, 0.0053269
4: -0.0052745, 0.0090176, -0.0039754, 0.0098362, -0.0151107, 0.0129931
5: -0.0016870, 0.0106974, -0.0012375, 0.0112761, -0.0129631, 0.0119349
6: -0.0065132, 0.0032328, -0.0045065, 0.0040449, -0.0105581, 0.0077394
7: -0.0113148, -0.0008248, -0.0117315, -0.0015407, -0.0097741, 0.0109067
8: -0.0122364, 0.0158742, -0.0112357, 0.0172349, -0.0293151, 0.0269310
9: -0.0093308, 0.0068415, -0.0100864, 0.0062544, -0.0155852, 0.0169279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156918, upper bound: 0.0159200
time: 2.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156918, upper bound: 0.0160963
time: 2.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0043741, 0.0029544, -0.0062524, 0.0044563, -0.0088305, 0.0092068
1: 0.9890121, 1.0057006, 0.9896853, 1.0094299, -0.0204178, 0.0160153
2: -0.0138315, 0.0033920, -0.0126604, 0.0053372, -0.0186178, 0.0156597
3: 0.0004899, 0.0059013, -0.0000360, 0.0056220, -0.0051322, 0.0059372
4: -0.0039031, 0.0093487, -0.0063752, 0.0084231, -0.0123262, 0.0157239
5: -0.0012125, 0.0109315, -0.0020679, 0.0102771, -0.0114896, 0.0129994
6: -0.0043948, 0.0035613, -0.0082134, 0.0026430, -0.0070378, 0.0117747
7: -0.0114834, -0.0015806, -0.0110122, -0.0002182, -0.0112652, 0.0094316
8: -0.0111800, 0.0164245, -0.0130842, 0.0148858, -0.0259063, 0.0293363
9: -0.0096364, 0.0062217, -0.0087819, 0.0073388, -0.0169752, 0.0150037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0157238
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0157238
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046190, 0.0028091, -0.0062475, 0.0044499, -0.0090690, 0.0090566
1: 0.9893784, 1.0061867, 0.9891158, 1.0094204, -0.0200420, 0.0170709
2: -0.0131942, 0.0036457, -0.0136511, 0.0053322, -0.0180344, 0.0168511
3: 0.0004213, 0.0057493, -0.0000346, 0.0058582, -0.0054370, 0.0057839
4: -0.0042255, 0.0088450, -0.0063688, 0.0092061, -0.0134315, 0.0152138
5: -0.0013240, 0.0105754, -0.0020657, 0.0108307, -0.0121547, 0.0126411
6: -0.0048927, 0.0030615, -0.0082035, 0.0034198, -0.0083125, 0.0112651
7: -0.0112270, -0.0014029, -0.0114108, -0.0002217, -0.0110052, 0.0100078
8: -0.0114283, 0.0155872, -0.0130793, 0.0161875, -0.0274550, 0.0285078
9: -0.0091714, 0.0063674, -0.0095048, 0.0073359, -0.0165073, 0.0158722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0157505
time: 1.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0157505
time: 1.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045154, 0.0030272, -0.0062475, 0.0044499, -0.0089653, 0.0092747
1: 0.9888285, 1.0059811, 0.9891158, 1.0094204, -0.0205919, 0.0168653
2: -0.0141507, 0.0035383, -0.0136511, 0.0053322, -0.0189073, 0.0166885
3: 0.0004503, 0.0059774, -0.0000346, 0.0058582, -0.0054079, 0.0060120
4: -0.0040890, 0.0096009, -0.0063688, 0.0092061, -0.0132951, 0.0159698
5: -0.0012768, 0.0111098, -0.0020657, 0.0108307, -0.0121074, 0.0131755
6: -0.0046820, 0.0038115, -0.0082035, 0.0034198, -0.0081018, 0.0120150
7: -0.0116117, -0.0014781, -0.0114108, -0.0002217, -0.0113900, 0.0099326
8: -0.0113232, 0.0168439, -0.0130793, 0.0161875, -0.0273304, 0.0297470
9: -0.0098693, 0.0063058, -0.0095048, 0.0073359, -0.0172052, 0.0158105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0159203
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0159196
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0058098, 0.0038769, -0.0061002, 0.0042572, -0.0100669, 0.0099771
1: 0.9897419, 1.0085512, 0.9898250, 1.0091276, -0.0193857, 0.0187262
2: -0.0125618, 0.0048789, -0.0124176, 0.0051797, -0.0172739, 0.0168244
3: 0.0000879, 0.0055985, 0.0000066, 0.0055641, -0.0054762, 0.0055919
4: -0.0057927, 0.0083452, -0.0061750, 0.0082312, -0.0140239, 0.0145201
5: -0.0018663, 0.0102220, -0.0019986, 0.0101415, -0.0120078, 0.0122206
6: -0.0073137, 0.0025657, -0.0079041, 0.0024526, -0.0097662, 0.0104698
7: -0.0109725, -0.0005392, -0.0109145, -0.0003285, -0.0106440, 0.0103753
8: -0.0126355, 0.0147563, -0.0129300, 0.0145668, -0.0270371, 0.0275247
9: -0.0087100, 0.0070756, -0.0086048, 0.0072483, -0.0159583, 0.0156804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162115, upper bound: 0.0161231
time: 2.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162115, upper bound: 0.0161231
time: 2.15 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0054832, 0.0034494, -0.0062524, 0.0044563, -0.0099396, 0.0097018
1: 0.9893442, 1.0079029, 0.9896853, 1.0094299, -0.0200858, 0.0182176
2: -0.0132537, 0.0045407, -0.0126604, 0.0053372, -0.0180794, 0.0167581
3: 0.0001794, 0.0057635, -0.0000360, 0.0056220, -0.0054427, 0.0057994
4: -0.0053629, 0.0088920, -0.0063752, 0.0084231, -0.0137860, 0.0152672
5: -0.0017176, 0.0106086, -0.0020679, 0.0102771, -0.0119947, 0.0126765
6: -0.0066497, 0.0031082, -0.0082134, 0.0026430, -0.0092927, 0.0113216
7: -0.0112509, -0.0007761, -0.0110122, -0.0002182, -0.0110327, 0.0102361
8: -0.0123044, 0.0156653, -0.0130842, 0.0148858, -0.0270262, 0.0285784
9: -0.0092148, 0.0068814, -0.0087819, 0.0073388, -0.0165536, 0.0156633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163698
time: 2.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163698
time: 1.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0056559, 0.0036755, -0.0062475, 0.0044499, -0.0101058, 0.0099230
1: 0.9897161, 1.0082456, 0.9891158, 1.0094204, -0.0197043, 0.0191298
2: -0.0126071, 0.0047195, -0.0136511, 0.0053322, -0.0174902, 0.0178923
3: 0.0001310, 0.0056093, -0.0000346, 0.0058582, -0.0057272, 0.0056439
4: -0.0055902, 0.0083809, -0.0063688, 0.0092061, -0.0147963, 0.0147498
5: -0.0017962, 0.0102473, -0.0020657, 0.0108307, -0.0126269, 0.0123130
6: -0.0070008, 0.0026011, -0.0082035, 0.0034198, -0.0104206, 0.0108047
7: -0.0109907, -0.0006508, -0.0114108, -0.0002217, -0.0107690, 0.0107599
8: -0.0124795, 0.0148157, -0.0130793, 0.0161875, -0.0285050, 0.0277370
9: -0.0087430, 0.0069841, -0.0095048, 0.0073359, -0.0160789, 0.0164888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0166141
time: 3.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0166142
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0056245, 0.0036344, -0.0062475, 0.0044499, -0.0100745, 0.0098819
1: 0.9891459, 1.0081834, 0.9891158, 1.0094204, -0.0202745, 0.0190676
2: -0.0135988, 0.0046870, -0.0136511, 0.0053322, -0.0183786, 0.0177880
3: 0.0001398, 0.0058458, -0.0000346, 0.0058582, -0.0057185, 0.0058804
4: -0.0055489, 0.0091647, -0.0063688, 0.0092061, -0.0147550, 0.0155336
5: -0.0017819, 0.0108014, -0.0020657, 0.0108307, -0.0126126, 0.0128671
6: -0.0069370, 0.0033788, -0.0082035, 0.0034198, -0.0103568, 0.0115823
7: -0.0113897, -0.0006736, -0.0114108, -0.0002217, -0.0111680, 0.0107372
8: -0.0124477, 0.0161187, -0.0130793, 0.0161875, -0.0284481, 0.0290168
9: -0.0094666, 0.0069654, -0.0095048, 0.0073359, -0.0168025, 0.0164702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165126
time: 2.27 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165125
time: 2.29 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0052190, 0.0031035, -0.0051297, 0.0029866, -0.0082056, 0.0082332
1: 0.9891621, 1.0073781, 0.9893498, 1.0072007, -0.0180386, 0.0180283
2: -0.0135702, 0.0042670, -0.0132440, 0.0041746, -0.0172985, 0.0170664
3: 0.0002533, 0.0058390, 0.0002783, 0.0057612, -0.0055078, 0.0055606
4: -0.0050151, 0.0091422, -0.0048976, 0.0088843, -0.0138994, 0.0140398
5: -0.0015972, 0.0107855, -0.0015566, 0.0106032, -0.0122004, 0.0123421
6: -0.0061125, 0.0033564, -0.0059310, 0.0031005, -0.0092130, 0.0092874
7: -0.0113782, -0.0009677, -0.0112469, -0.0010325, -0.0103457, 0.0102792
8: -0.0120366, 0.0160813, -0.0119460, 0.0156525, -0.0275331, 0.0278777
9: -0.0094458, 0.0067242, -0.0092077, 0.0066711, -0.0161169, 0.0159319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0158909
time: 1.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0158910
time: 5.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0055674, 0.0035597, -0.0048698, 0.0029788, -0.0085462, 0.0084295
1: 0.9895711, 1.0080701, 0.9889506, 1.0066847, -0.0171136, 0.0191195
2: -0.0128591, 0.0046279, -0.0139384, 0.0039054, -0.0162923, 0.0180108
3: 0.0001558, 0.0056694, 0.0003511, 0.0059267, -0.0057710, 0.0053183
4: -0.0054737, 0.0085801, -0.0045555, 0.0094331, -0.0149069, 0.0131357
5: -0.0017560, 0.0103881, -0.0014382, 0.0109912, -0.0127471, 0.0118264
6: -0.0068209, 0.0027988, -0.0054026, 0.0036450, -0.0104660, 0.0082014
7: -0.0110921, -0.0007150, -0.0115264, -0.0012210, -0.0098711, 0.0108114
8: -0.0123898, 0.0151469, -0.0116826, 0.0165649, -0.0287791, 0.0266583
9: -0.0089269, 0.0069315, -0.0097144, 0.0065166, -0.0154435, 0.0166458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0155917, upper bound: 0.0157994
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0155917, upper bound: 0.0157994
time: 2.17 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0052190, 0.0031035, -0.0050256, 0.0030383, -0.0082573, 0.0081292
1: 0.9891621, 1.0073781, 0.9888005, 1.0069941, -0.0178320, 0.0185776
2: -0.0135702, 0.0042670, -0.0141995, 0.0040668, -0.0171334, 0.0179401
3: 0.0002533, 0.0058390, 0.0003075, 0.0059890, -0.0057357, 0.0055315
4: -0.0050151, 0.0091422, -0.0047606, 0.0096395, -0.0146547, 0.0139028
5: -0.0015972, 0.0107855, -0.0015092, 0.0111370, -0.0127343, 0.0122947
6: -0.0061125, 0.0033564, -0.0057194, 0.0038498, -0.0099623, 0.0090758
7: -0.0113782, -0.0009677, -0.0116314, -0.0011080, -0.0102702, 0.0106637
8: -0.0120366, 0.0160813, -0.0118405, 0.0169080, -0.0287751, 0.0277497
9: -0.0094458, 0.0067242, -0.0099049, 0.0066092, -0.0160550, 0.0166291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152272, upper bound: 0.0155007
time: 1.48 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152272, upper bound: 0.0160376
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041482, 0.0029847, -0.0061120, 0.0042726, -0.0084208, 0.0090967
1: 0.9889359, 1.0052520, 0.9896877, 1.0091513, -0.0202154, 0.0155643
2: -0.0139642, 0.0031581, -0.0126561, 0.0051919, -0.0186584, 0.0154789
3: 0.0005531, 0.0059329, 0.0000033, 0.0056210, -0.0050679, 0.0059296
4: -0.0036057, 0.0094535, -0.0061905, 0.0084197, -0.0120254, 0.0156441
5: -0.0011095, 0.0110056, -0.0020040, 0.0102747, -0.0113842, 0.0130096
6: -0.0039682, 0.0036653, -0.0079281, 0.0026396, -0.0066078, 0.0115934
7: -0.0115367, -0.0017445, -0.0110105, -0.0003200, -0.0112168, 0.0092660
8: -0.0109510, 0.0165988, -0.0129420, 0.0148801, -0.0256838, 0.0293830
9: -0.0097332, 0.0060874, -0.0087788, 0.0072554, -0.0169886, 0.0148661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0152773, upper bound: 0.0145456
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152773, upper bound: 0.0156803
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044176, 0.0028313, -0.0061008, 0.0042579, -0.0086755, 0.0089320
1: 0.9893225, 1.0057867, 0.9891183, 1.0091290, -0.0198066, 0.0166683
2: -0.0132913, 0.0034371, -0.0136469, 0.0051802, -0.0180384, 0.0166579
3: 0.0004777, 0.0057725, 0.0000065, 0.0058572, -0.0053796, 0.0057660
4: -0.0039603, 0.0089218, -0.0061757, 0.0092027, -0.0131631, 0.0150975
5: -0.0012322, 0.0106297, -0.0019989, 0.0108283, -0.0120605, 0.0126285
6: -0.0044832, 0.0031377, -0.0079052, 0.0034165, -0.0078997, 0.0110430
7: -0.0112661, -0.0015491, -0.0114091, -0.0003281, -0.0109379, 0.0098600
8: -0.0112241, 0.0157148, -0.0129305, 0.0161819, -0.0272548, 0.0284993
9: -0.0092423, 0.0062476, -0.0095017, 0.0072487, -0.0164910, 0.0157493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0157088
time: 2.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0157081
time: 2.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042835, 0.0030623, -0.0061008, 0.0042579, -0.0085414, 0.0091630
1: 0.9887401, 1.0055205, 0.9891183, 1.0091290, -0.0203889, 0.0164021
2: -0.0143045, 0.0032982, -0.0136469, 0.0051802, -0.0189503, 0.0164667
3: 0.0005152, 0.0060140, 0.0000065, 0.0058572, -0.0053420, 0.0060076
4: -0.0037838, 0.0097225, -0.0061757, 0.0092027, -0.0129865, 0.0158982
5: -0.0011712, 0.0111957, -0.0019989, 0.0108283, -0.0119995, 0.0131946
6: -0.0042105, 0.0039322, -0.0079052, 0.0034165, -0.0076270, 0.0118374
7: -0.0116736, -0.0016463, -0.0114091, -0.0003281, -0.0113455, 0.0097627
8: -0.0110881, 0.0170460, -0.0129305, 0.0161819, -0.0270994, 0.0298106
9: -0.0099815, 0.0061678, -0.0095017, 0.0072487, -0.0172302, 0.0156695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0158267
time: 1.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0158267
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0056500, 0.0036678, -0.0059592, 0.0040726, -0.0097226, 0.0096270
1: 0.9896704, 1.0082339, 0.9898275, 1.0088480, -0.0191776, 0.0184065
2: -0.0126862, 0.0047134, -0.0124132, 0.0050337, -0.0172970, 0.0167035
3: 0.0001327, 0.0056282, 0.0000461, 0.0055631, -0.0054304, 0.0055821
4: -0.0055825, 0.0084435, -0.0059894, 0.0082277, -0.0138101, 0.0144329
5: -0.0017936, 0.0102915, -0.0019344, 0.0101390, -0.0119325, 0.0122259
6: -0.0069888, 0.0026632, -0.0076175, 0.0024491, -0.0094379, 0.0102807
7: -0.0110226, -0.0006551, -0.0109127, -0.0004308, -0.0105918, 0.0102577
8: -0.0124736, 0.0149197, -0.0127871, 0.0145610, -0.0268753, 0.0275527
9: -0.0088007, 0.0069806, -0.0086015, 0.0071645, -0.0159652, 0.0155821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162112, upper bound: 0.0160729
time: 2.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162112, upper bound: 0.0160729
time: 2.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0053089, 0.0032212, -0.0061120, 0.0042726, -0.0095815, 0.0093332
1: 0.9892541, 1.0075566, 0.9896877, 1.0091513, -0.0198973, 0.0178688
2: -0.0134108, 0.0043601, -0.0126561, 0.0051919, -0.0181295, 0.0166268
3: 0.0002281, 0.0058009, 0.0000033, 0.0056210, -0.0053928, 0.0057976
4: -0.0051334, 0.0090162, -0.0061905, 0.0084197, -0.0135531, 0.0152067
5: -0.0016382, 0.0106964, -0.0020040, 0.0102747, -0.0119129, 0.0127004
6: -0.0062953, 0.0032314, -0.0079281, 0.0026396, -0.0089349, 0.0111595
7: -0.0113141, -0.0009025, -0.0110105, -0.0003200, -0.0109942, 0.0101079
8: -0.0121277, 0.0158718, -0.0129420, 0.0148801, -0.0268540, 0.0286578
9: -0.0093294, 0.0067777, -0.0087788, 0.0072554, -0.0165848, 0.0155565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161649, upper bound: 0.0158765
time: 2.26 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161649, upper bound: 0.0163286
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0054754, 0.0034391, -0.0061008, 0.0042579, -0.0097332, 0.0095399
1: 0.9896449, 1.0078871, 0.9891183, 1.0091290, -0.0194841, 0.0187688
2: -0.0127307, 0.0045325, -0.0136469, 0.0051802, -0.0175065, 0.0177229
3: 0.0001816, 0.0056388, 0.0000065, 0.0058572, -0.0056757, 0.0056323
4: -0.0053526, 0.0084786, -0.0061757, 0.0092027, -0.0145553, 0.0146543
5: -0.0017140, 0.0103163, -0.0019989, 0.0108283, -0.0125423, 0.0123152
6: -0.0066337, 0.0026981, -0.0079052, 0.0034165, -0.0100502, 0.0106033
7: -0.0110405, -0.0007818, -0.0114091, -0.0003281, -0.0107123, 0.0106273
8: -0.0122965, 0.0149781, -0.0129305, 0.0161819, -0.0283237, 0.0277673
9: -0.0088332, 0.0068767, -0.0095017, 0.0072487, -0.0160819, 0.0163784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160542, upper bound: 0.0162230
time: 2.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0165347
time: 2.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0054370, 0.0033889, -0.0061008, 0.0042579, -0.0096949, 0.0094897
1: 0.9890565, 1.0078113, 0.9891183, 1.0091290, -0.0200726, 0.0186930
2: -0.0137542, 0.0044929, -0.0136469, 0.0051802, -0.0184216, 0.0176481
3: 0.0001923, 0.0058828, 0.0000065, 0.0058572, -0.0056649, 0.0058763
4: -0.0053021, 0.0092876, -0.0061757, 0.0092027, -0.0145048, 0.0154633
5: -0.0016966, 0.0108882, -0.0019989, 0.0108283, -0.0125249, 0.0128871
6: -0.0065558, 0.0035006, -0.0079052, 0.0034165, -0.0099723, 0.0114059
7: -0.0114523, -0.0008096, -0.0114091, -0.0003281, -0.0111241, 0.0105995
8: -0.0122576, 0.0163229, -0.0129305, 0.0161819, -0.0282659, 0.0290891
9: -0.0095800, 0.0068539, -0.0095017, 0.0072487, -0.0168286, 0.0163556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160542, upper bound: 0.0161410
time: 2.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0163993
time: 2.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.31 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0147226, upper bound: 0.0151427
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0147226, upper bound: 0.0161688
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0144282, upper bound: 0.0148717
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0144282, upper bound: 0.0161472
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0151439, upper bound: 0.0152752
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0151439, upper bound: 0.0158992
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0157471, upper bound: 0.0159431
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0157471, upper bound: 0.0161316
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0150843, upper bound: 0.0152107
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0150843, upper bound: 0.0158639
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0156918, upper bound: 0.0159200
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0156918, upper bound: 0.0160963
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0157238
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0161141, upper bound: 0.0157238
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0157505
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0157505
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0159203
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159305, upper bound: 0.0159196
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0162115, upper bound: 0.0161231
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0162115, upper bound: 0.0161231
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163698
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0165165, upper bound: 0.0163698
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0166141
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0166142
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165126
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0163890, upper bound: 0.0165125
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0158909
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0156913, upper bound: 0.0158910
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0155917, upper bound: 0.0157994
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0155917, upper bound: 0.0157994
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0152272, upper bound: 0.0155007
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0152272, upper bound: 0.0160376
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0152773, upper bound: 0.0145456
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0152773, upper bound: 0.0156803
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0157088
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0157081
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0158267
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0159302, upper bound: 0.0158267
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0162112, upper bound: 0.0160729
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0162112, upper bound: 0.0160729
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0161649, upper bound: 0.0158765
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0161649, upper bound: 0.0163286
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0160542, upper bound: 0.0162230
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0165347
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0160542, upper bound: 0.0161410
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.31
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0163993

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0054655, 0.0034262, -0.0043978, 0.0029545, -0.0084199, 0.0078240
1: 0.9898159, 1.0078675, 0.9890121, 1.0057474, -0.0159315, 0.0188553
2: -0.0124335, 0.0045223, -0.0138315, 0.0034165, -0.0153958, 0.0177875
3: 0.0001843, 0.0055679, 0.0004832, 0.0059013, -0.0057170, 0.0050847
4: -0.0053396, 0.0082438, -0.0039342, 0.0093487, -0.0146882, 0.0121780
5: -0.0017095, 0.0101503, -0.0012232, 0.0109315, -0.0126410, 0.0113735
6: -0.0066136, 0.0024651, -0.0044429, 0.0035613, -0.0101749, 0.0069079
7: -0.0109209, -0.0007890, -0.0114834, -0.0015635, -0.0093575, 0.0106944
8: -0.0122864, 0.0145877, -0.0112040, 0.0164245, -0.0285323, 0.0256263
9: -0.0086164, 0.0068708, -0.0096364, 0.0062358, -0.0148522, 0.0165072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0143405, upper bound: 0.0159067
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0146164, upper bound: 0.0160084
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0054655, 0.0034262, -0.0041901, 0.0029847, -0.0084502, 0.0076162
1: 0.9898159, 1.0078675, 0.9889359, 1.0053350, -0.0155191, 0.0189315
2: -0.0124335, 0.0045223, -0.0139642, 0.0032014, -0.0152276, 0.0179622
3: 0.0001843, 0.0055679, 0.0005414, 0.0059329, -0.0057486, 0.0050265
4: -0.0053396, 0.0082438, -0.0036608, 0.0094535, -0.0147931, 0.0119046
5: -0.0017095, 0.0101503, -0.0011286, 0.0110056, -0.0127151, 0.0112789
6: -0.0066136, 0.0024651, -0.0040206, 0.0036653, -0.0102789, 0.0064857
7: -0.0109209, -0.0007890, -0.0115367, -0.0017141, -0.0092068, 0.0107478
8: -0.0122864, 0.0145877, -0.0109934, 0.0165988, -0.0287197, 0.0254305
9: -0.0086164, 0.0068708, -0.0097332, 0.0061123, -0.0147286, 0.0166040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0143996, upper bound: 0.0160918
time: 1.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0144204, upper bound: 0.0148678
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0057372, 0.0037819, -0.0043851, 0.0029677, -0.0087049, 0.0081670
1: 0.9896498, 1.0084070, 0.9889787, 1.0057223, -0.0160725, 0.0194283
2: -0.0127221, 0.0048037, -0.0138897, 0.0034034, -0.0156009, 0.0180782
3: 0.0001083, 0.0056367, 0.0004868, 0.0059151, -0.0058069, 0.0051499
4: -0.0056972, 0.0084718, -0.0039176, 0.0093947, -0.0150918, 0.0123894
5: -0.0018333, 0.0103116, -0.0012175, 0.0109639, -0.0127972, 0.0115290
6: -0.0071660, 0.0026913, -0.0044171, 0.0036069, -0.0107729, 0.0071084
7: -0.0110370, -0.0005919, -0.0115067, -0.0015726, -0.0094644, 0.0109149
8: -0.0125619, 0.0149668, -0.0111912, 0.0165009, -0.0288783, 0.0259783
9: -0.0088269, 0.0070324, -0.0096788, 0.0062283, -0.0150552, 0.0167113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0144818, upper bound: 0.0154996
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150194, upper bound: 0.0157833
time: 1.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0054161, 0.0033615, -0.0046296, 0.0028091, -0.0082252, 0.0079911
1: 0.9892527, 1.0077693, 0.9893784, 1.0062078, -0.0169551, 0.0183910
2: -0.0134127, 0.0044711, -0.0131942, 0.0036566, -0.0165738, 0.0171523
3: 0.0001982, 0.0058014, 0.0004183, 0.0057493, -0.0055511, 0.0053831
4: -0.0052745, 0.0090176, -0.0042393, 0.0088450, -0.0141195, 0.0132570
5: -0.0016870, 0.0106974, -0.0013288, 0.0105754, -0.0122624, 0.0120262
6: -0.0065132, 0.0032328, -0.0049142, 0.0030615, -0.0095747, 0.0081470
7: -0.0113148, -0.0008248, -0.0112270, -0.0013953, -0.0099196, 0.0104022
8: -0.0122364, 0.0158742, -0.0114390, 0.0155872, -0.0276558, 0.0271420
9: -0.0093308, 0.0068415, -0.0091714, 0.0063737, -0.0157045, 0.0160128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153542, upper bound: 0.0156981
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153447, upper bound: 0.0155976
time: 2.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0054161, 0.0033615, -0.0045381, 0.0030272, -0.0084433, 0.0078997
1: 0.9892527, 1.0077693, 0.9888285, 1.0060261, -0.0167735, 0.0189408
2: -0.0134127, 0.0044711, -0.0141507, 0.0035619, -0.0164171, 0.0180273
3: 0.0001982, 0.0058014, 0.0004439, 0.0059773, -0.0057792, 0.0053575
4: -0.0052745, 0.0090176, -0.0041190, 0.0096009, -0.0148755, 0.0131366
5: -0.0016870, 0.0106974, -0.0012872, 0.0111098, -0.0127968, 0.0119846
6: -0.0065132, 0.0032328, -0.0047283, 0.0038115, -0.0103247, 0.0079611
7: -0.0113148, -0.0008248, -0.0116118, -0.0014616, -0.0098532, 0.0107870
8: -0.0122364, 0.0158742, -0.0113463, 0.0168439, -0.0288946, 0.0270283
9: -0.0093308, 0.0068415, -0.0098693, 0.0063193, -0.0156501, 0.0167107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153542, upper bound: 0.0158531
time: 2.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153447, upper bound: 0.0157195
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0057372, 0.0037819, -0.0041727, 0.0030035, -0.0087407, 0.0079546
1: 0.9896498, 1.0084070, 0.9888883, 1.0053006, -0.0156508, 0.0195187
2: -0.0127221, 0.0048037, -0.0140467, 0.0031835, -0.0154227, 0.0182727
3: 0.0001083, 0.0056367, 0.0005462, 0.0059526, -0.0058443, 0.0050905
4: -0.0056972, 0.0084718, -0.0036380, 0.0095188, -0.0152159, 0.0121099
5: -0.0018333, 0.0103116, -0.0011207, 0.0110517, -0.0128850, 0.0114323
6: -0.0071660, 0.0026913, -0.0039968, 0.0037300, -0.0108961, 0.0066882
7: -0.0110370, -0.0005919, -0.0115700, -0.0017267, -0.0093103, 0.0109781
8: -0.0125619, 0.0149668, -0.0109758, 0.0167073, -0.0290893, 0.0257778
9: -0.0088269, 0.0070324, -0.0097934, 0.0061020, -0.0149289, 0.0168259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0142558, upper bound: 0.0154503
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149601, upper bound: 0.0157457
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0054161, 0.0033615, -0.0044495, 0.0028313, -0.0082474, 0.0078110
1: 0.9892527, 1.0077693, 0.9893225, 1.0058502, -0.0165975, 0.0184469
2: -0.0134127, 0.0044711, -0.0132914, 0.0034701, -0.0164228, 0.0172999
3: 0.0001982, 0.0058014, 0.0004688, 0.0057725, -0.0055743, 0.0053326
4: -0.0052745, 0.0090176, -0.0040023, 0.0089218, -0.0141963, 0.0130199
5: -0.0016870, 0.0106974, -0.0012468, 0.0106297, -0.0123167, 0.0119442
6: -0.0065132, 0.0032328, -0.0045480, 0.0031377, -0.0096509, 0.0077808
7: -0.0113148, -0.0008248, -0.0112661, -0.0015259, -0.0097889, 0.0104413
8: -0.0122364, 0.0158742, -0.0112564, 0.0157148, -0.0277936, 0.0269704
9: -0.0093308, 0.0068415, -0.0092423, 0.0062666, -0.0155974, 0.0160837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153363, upper bound: 0.0156881
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153284, upper bound: 0.0155894
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0054161, 0.0033615, -0.0043277, 0.0030623, -0.0084784, 0.0076893
1: 0.9892527, 1.0077693, 0.9887401, 1.0056083, -0.0163556, 0.0190292
2: -0.0134127, 0.0044711, -0.0143045, 0.0033440, -0.0162453, 0.0182137
3: 0.0001982, 0.0058014, 0.0005028, 0.0060140, -0.0058159, 0.0052986
4: -0.0052745, 0.0090176, -0.0038421, 0.0097225, -0.0149971, 0.0128597
5: -0.0016870, 0.0106974, -0.0011913, 0.0111957, -0.0128828, 0.0118888
6: -0.0065132, 0.0032328, -0.0043005, 0.0039321, -0.0104453, 0.0075333
7: -0.0113148, -0.0008248, -0.0116737, -0.0016142, -0.0097006, 0.0108489
8: -0.0122364, 0.0158742, -0.0111330, 0.0170460, -0.0291043, 0.0268288
9: -0.0093308, 0.0068415, -0.0099815, 0.0061942, -0.0155250, 0.0168230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153363, upper bound: 0.0158354
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153284, upper bound: 0.0157073
time: 2.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0043741, 0.0029544, -0.0056559, 0.0036755, -0.0080496, 0.0086103
1: 0.9890121, 1.0057006, 0.9897161, 1.0082456, -0.0192335, 0.0159845
2: -0.0138315, 0.0033920, -0.0126071, 0.0047195, -0.0179883, 0.0155888
3: 0.0004899, 0.0059013, 0.0001310, 0.0056093, -0.0051194, 0.0057703
4: -0.0039031, 0.0093487, -0.0055902, 0.0083809, -0.0122841, 0.0149388
5: -0.0012125, 0.0109315, -0.0017962, 0.0102473, -0.0114598, 0.0127277
6: -0.0043948, 0.0035613, -0.0070008, 0.0026011, -0.0069960, 0.0105620
7: -0.0114834, -0.0015806, -0.0109907, -0.0006508, -0.0108325, 0.0094101
8: -0.0111800, 0.0164245, -0.0124795, 0.0148157, -0.0258365, 0.0287312
9: -0.0096364, 0.0062217, -0.0087430, 0.0069841, -0.0166205, 0.0149647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158045, upper bound: 0.0155207
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157829, upper bound: 0.0153098
time: 2.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0043741, 0.0029544, -0.0054754, 0.0034391, -0.0078133, 0.0084298
1: 0.9890121, 1.0057006, 0.9896455, 1.0078871, -0.0188750, 0.0160550
2: -0.0138315, 0.0033920, -0.0127297, 0.0045326, -0.0178431, 0.0157435
3: 0.0004899, 0.0059013, 0.0001816, 0.0056385, -0.0051487, 0.0057197
4: -0.0039031, 0.0093487, -0.0053526, 0.0084778, -0.0123809, 0.0147012
5: -0.0012125, 0.0109315, -0.0017140, 0.0103158, -0.0115282, 0.0126455
6: -0.0043948, 0.0035613, -0.0066338, 0.0026973, -0.0070921, 0.0101950
7: -0.0114834, -0.0015806, -0.0110401, -0.0007818, -0.0107016, 0.0094595
8: -0.0111800, 0.0164245, -0.0122965, 0.0149768, -0.0260125, 0.0285583
9: -0.0096364, 0.0062217, -0.0088324, 0.0068767, -0.0165131, 0.0150542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158045, upper bound: 0.0155208
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157829, upper bound: 0.0153097
time: 2.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046190, 0.0028091, -0.0056245, 0.0036344, -0.0082534, 0.0084337
1: 0.9893784, 1.0061867, 0.9891459, 1.0081834, -0.0188050, 0.0170408
2: -0.0131942, 0.0036457, -0.0135988, 0.0046870, -0.0173788, 0.0167839
3: 0.0004213, 0.0057493, 0.0001398, 0.0058458, -0.0054245, 0.0056095
4: -0.0042255, 0.0088450, -0.0055489, 0.0091647, -0.0133902, 0.0143939
5: -0.0013240, 0.0105754, -0.0017819, 0.0108014, -0.0121254, 0.0123573
6: -0.0048927, 0.0030615, -0.0069370, 0.0033788, -0.0082715, 0.0099985
7: -0.0112270, -0.0014029, -0.0113897, -0.0006736, -0.0105534, 0.0099868
8: -0.0114283, 0.0155872, -0.0124477, 0.0161187, -0.0273862, 0.0278698
9: -0.0091714, 0.0063674, -0.0094666, 0.0069654, -0.0161368, 0.0158340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156269, upper bound: 0.0155855
time: 2.18 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0152334, upper bound: 0.0153559
time: 2.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046190, 0.0028091, -0.0054231, 0.0033707, -0.0079897, 0.0082322
1: 0.9893784, 1.0061867, 0.9890565, 1.0077832, -0.0184048, 0.0171303
2: -0.0131942, 0.0036457, -0.0137542, 0.0044784, -0.0172126, 0.0169724
3: 0.0004213, 0.0057493, 0.0001962, 0.0058828, -0.0054615, 0.0055531
4: -0.0042255, 0.0088450, -0.0052838, 0.0092876, -0.0135130, 0.0141287
5: -0.0013240, 0.0105754, -0.0016902, 0.0108883, -0.0122123, 0.0122656
6: -0.0048927, 0.0030615, -0.0065275, 0.0035006, -0.0083934, 0.0095890
7: -0.0112270, -0.0014029, -0.0114523, -0.0008197, -0.0104073, 0.0100493
8: -0.0114283, 0.0155872, -0.0122435, 0.0163229, -0.0276050, 0.0276765
9: -0.0091714, 0.0063674, -0.0095800, 0.0068456, -0.0160170, 0.0159474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156269, upper bound: 0.0155848
time: 2.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0152334, upper bound: 0.0153566
time: 2.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0045154, 0.0030272, -0.0056245, 0.0036344, -0.0081498, 0.0086518
1: 0.9888285, 1.0059811, 0.9891459, 1.0081834, -0.0193548, 0.0168352
2: -0.0141507, 0.0035383, -0.0135988, 0.0046870, -0.0182528, 0.0166204
3: 0.0004503, 0.0059774, 0.0001398, 0.0058458, -0.0053955, 0.0058376
4: -0.0040890, 0.0096009, -0.0055489, 0.0091647, -0.0132538, 0.0151498
5: -0.0012768, 0.0111098, -0.0017819, 0.0108014, -0.0120782, 0.0128917
6: -0.0046820, 0.0038115, -0.0069370, 0.0033788, -0.0080607, 0.0107485
7: -0.0116117, -0.0014781, -0.0113897, -0.0006736, -0.0109382, 0.0099116
8: -0.0113232, 0.0168439, -0.0124477, 0.0161187, -0.0272616, 0.0291089
9: -0.0098693, 0.0063058, -0.0094666, 0.0069654, -0.0168347, 0.0157723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156884, upper bound: 0.0153661
time: 1.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161730, upper bound: 0.0159010
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045154, 0.0030272, -0.0054231, 0.0033707, -0.0078861, 0.0084503
1: 0.9888285, 1.0059811, 0.9890565, 1.0077832, -0.0189546, 0.0169246
2: -0.0141507, 0.0035383, -0.0137542, 0.0044784, -0.0180848, 0.0168066
3: 0.0004503, 0.0059774, 0.0001962, 0.0058828, -0.0054325, 0.0057812
4: -0.0040890, 0.0096009, -0.0052838, 0.0092876, -0.0133766, 0.0148847
5: -0.0012768, 0.0111098, -0.0016902, 0.0108883, -0.0121650, 0.0128000
6: -0.0046820, 0.0038115, -0.0065275, 0.0035006, -0.0081826, 0.0103390
7: -0.0116117, -0.0014781, -0.0114523, -0.0008197, -0.0107920, 0.0099741
8: -0.0113232, 0.0168439, -0.0122435, 0.0163229, -0.0274806, 0.0289160
9: -0.0098693, 0.0063058, -0.0095800, 0.0068456, -0.0167149, 0.0158857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156884, upper bound: 0.0153661
time: 2.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161730, upper bound: 0.0159010
time: 1.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0058098, 0.0038769, -0.0055033, 0.0034758, -0.0092855, 0.0093803
1: 0.9897419, 1.0085512, 0.9898560, 1.0079428, -0.0182009, 0.0186952
2: -0.0125618, 0.0048789, -0.0123636, 0.0045615, -0.0166423, 0.0167535
3: 0.0000879, 0.0055985, 0.0001737, 0.0055512, -0.0054633, 0.0054248
4: -0.0057927, 0.0083452, -0.0053894, 0.0081885, -0.0139812, 0.0137345
5: -0.0018663, 0.0102220, -0.0017268, 0.0101112, -0.0119776, 0.0119488
6: -0.0073137, 0.0025657, -0.0066906, 0.0024102, -0.0097239, 0.0092563
7: -0.0109725, -0.0005392, -0.0108928, -0.0007615, -0.0102110, 0.0103536
8: -0.0126355, 0.0147563, -0.0123248, 0.0144958, -0.0269661, 0.0269190
9: -0.0087100, 0.0070756, -0.0085653, 0.0068933, -0.0156033, 0.0156409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159394, upper bound: 0.0159433
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159281, upper bound: 0.0158176
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0058098, 0.0038769, -0.0053267, 0.0032446, -0.0090544, 0.0092037
1: 0.9897419, 1.0085512, 0.9897907, 1.0075921, -0.0178502, 0.0187605
2: -0.0125618, 0.0048789, -0.0124769, 0.0043786, -0.0164982, 0.0169093
3: 0.0000879, 0.0055985, 0.0002232, 0.0055783, -0.0054903, 0.0053753
4: -0.0057927, 0.0083452, -0.0051569, 0.0082781, -0.0140708, 0.0135021
5: -0.0018663, 0.0102220, -0.0016463, 0.0101746, -0.0120409, 0.0118683
6: -0.0073137, 0.0025657, -0.0063316, 0.0024991, -0.0098128, 0.0088972
7: -0.0109725, -0.0005392, -0.0109384, -0.0008896, -0.0100829, 0.0103992
8: -0.0126355, 0.0147563, -0.0121458, 0.0146448, -0.0271279, 0.0267506
9: -0.0087100, 0.0070756, -0.0086480, 0.0067883, -0.0154983, 0.0157237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159394, upper bound: 0.0159433
time: 1.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159281, upper bound: 0.0158176
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0054832, 0.0034494, -0.0056559, 0.0036755, -0.0091587, 0.0091053
1: 0.9893442, 1.0079029, 0.9897161, 1.0082456, -0.0189014, 0.0181868
2: -0.0132537, 0.0045407, -0.0126071, 0.0047195, -0.0174509, 0.0166880
3: 0.0001794, 0.0057635, 0.0001310, 0.0056093, -0.0054299, 0.0056325
4: -0.0053629, 0.0088920, -0.0055902, 0.0083809, -0.0137438, 0.0144821
5: -0.0017176, 0.0106086, -0.0017962, 0.0102473, -0.0119649, 0.0124048
6: -0.0066497, 0.0031082, -0.0070008, 0.0026011, -0.0092508, 0.0101090
7: -0.0112509, -0.0007761, -0.0109907, -0.0006508, -0.0106001, 0.0102146
8: -0.0123044, 0.0156653, -0.0124795, 0.0148157, -0.0269563, 0.0279731
9: -0.0092148, 0.0068814, -0.0087430, 0.0069841, -0.0161989, 0.0156244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162430, upper bound: 0.0161865
time: 3.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162395, upper bound: 0.0161049
time: 1.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0054832, 0.0034494, -0.0054754, 0.0034391, -0.0089224, 0.0089248
1: 0.9893442, 1.0079029, 0.9896455, 1.0078871, -0.0185429, 0.0182573
2: -0.0132537, 0.0045407, -0.0127297, 0.0045326, -0.0173025, 0.0168457
3: 0.0001794, 0.0057635, 0.0001816, 0.0056385, -0.0054592, 0.0055819
4: -0.0053629, 0.0088920, -0.0053526, 0.0084778, -0.0138407, 0.0142445
5: -0.0017176, 0.0106086, -0.0017140, 0.0103158, -0.0120334, 0.0123226
6: -0.0066497, 0.0031082, -0.0066338, 0.0026973, -0.0093470, 0.0097419
7: -0.0112509, -0.0007761, -0.0110401, -0.0007818, -0.0104691, 0.0102640
8: -0.0123044, 0.0156653, -0.0122965, 0.0149768, -0.0271317, 0.0277993
9: -0.0092148, 0.0068814, -0.0088324, 0.0068767, -0.0160915, 0.0157138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162430, upper bound: 0.0161865
time: 1.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162395, upper bound: 0.0161049
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0056559, 0.0036755, -0.0056245, 0.0036344, -0.0092903, 0.0093000
1: 0.9897161, 1.0082456, 0.9891459, 1.0081834, -0.0184673, 0.0190997
2: -0.0126071, 0.0047195, -0.0135988, 0.0046870, -0.0168338, 0.0178276
3: 0.0001310, 0.0056093, 0.0001398, 0.0058458, -0.0057147, 0.0054695
4: -0.0055902, 0.0083809, -0.0055489, 0.0091647, -0.0147549, 0.0139298
5: -0.0017962, 0.0102473, -0.0017819, 0.0108014, -0.0125977, 0.0120292
6: -0.0070008, 0.0026011, -0.0069370, 0.0033788, -0.0103795, 0.0095381
7: -0.0109907, -0.0006508, -0.0113897, -0.0006736, -0.0103172, 0.0107389
8: -0.0124795, 0.0148157, -0.0124477, 0.0161187, -0.0284363, 0.0270995
9: -0.0087430, 0.0069841, -0.0094666, 0.0069654, -0.0157084, 0.0164506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158700, upper bound: 0.0162692
time: 2.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0165786
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0056559, 0.0036755, -0.0054231, 0.0033707, -0.0090266, 0.0090986
1: 0.9897161, 1.0082456, 0.9890565, 1.0077832, -0.0180671, 0.0191891
2: -0.0126071, 0.0047195, -0.0137542, 0.0044784, -0.0166674, 0.0180166
3: 0.0001310, 0.0056093, 0.0001962, 0.0058828, -0.0057518, 0.0054131
4: -0.0055902, 0.0083809, -0.0052838, 0.0092876, -0.0148777, 0.0136647
5: -0.0017962, 0.0102473, -0.0016902, 0.0108883, -0.0126845, 0.0119375
6: -0.0070008, 0.0026011, -0.0065275, 0.0035006, -0.0105014, 0.0091286
7: -0.0109907, -0.0006508, -0.0114523, -0.0008197, -0.0101710, 0.0108014
8: -0.0124795, 0.0148157, -0.0122435, 0.0163229, -0.0286545, 0.0269059
9: -0.0087430, 0.0069841, -0.0095800, 0.0068456, -0.0155886, 0.0165640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158700, upper bound: 0.0162701
time: 2.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0165786
time: 2.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0056245, 0.0036344, -0.0056245, 0.0036344, -0.0092589, 0.0092589
1: 0.9891459, 1.0081834, 0.9891459, 1.0081834, -0.0190375, 0.0190375
2: -0.0135988, 0.0046870, -0.0135988, 0.0046870, -0.0177227, 0.0177227
3: 0.0001398, 0.0058458, 0.0001398, 0.0058458, -0.0057060, 0.0057060
4: -0.0055489, 0.0091647, -0.0055489, 0.0091647, -0.0147136, 0.0147136
5: -0.0017819, 0.0108014, -0.0017819, 0.0108014, -0.0125834, 0.0125834
6: -0.0069370, 0.0033788, -0.0069370, 0.0033788, -0.0103158, 0.0103158
7: -0.0113897, -0.0006736, -0.0113897, -0.0006736, -0.0107161, 0.0107161
8: -0.0124477, 0.0161187, -0.0124477, 0.0161187, -0.0283793, 0.0283793
9: -0.0094666, 0.0069654, -0.0094666, 0.0069654, -0.0164320, 0.0164320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163084, upper bound: 0.0163305
time: 2.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165399, upper bound: 0.0164475
time: 1.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0056245, 0.0036344, -0.0054231, 0.0033707, -0.0089952, 0.0090575
1: 0.9891459, 1.0081834, 0.9890565, 1.0077832, -0.0186373, 0.0191269
2: -0.0135988, 0.0046870, -0.0137542, 0.0044784, -0.0175541, 0.0179085
3: 0.0001398, 0.0058458, 0.0001962, 0.0058828, -0.0057430, 0.0056496
4: -0.0055489, 0.0091647, -0.0052838, 0.0092876, -0.0148365, 0.0144485
5: -0.0017819, 0.0108014, -0.0016902, 0.0108883, -0.0126702, 0.0124916
6: -0.0069370, 0.0033788, -0.0065275, 0.0035006, -0.0104376, 0.0099062
7: -0.0113897, -0.0006736, -0.0114523, -0.0008197, -0.0105700, 0.0107787
8: -0.0124477, 0.0161187, -0.0122435, 0.0163229, -0.0285978, 0.0281859
9: -0.0094666, 0.0069654, -0.0095800, 0.0068456, -0.0163122, 0.0165454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163084, upper bound: 0.0163305
time: 2.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0165399, upper bound: 0.0164475
time: 2.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0052190, 0.0031035, -0.0046296, 0.0028091, -0.0080281, 0.0077331
1: 0.9891621, 1.0073781, 0.9893784, 1.0062078, -0.0170457, 0.0179997
2: -0.0135702, 0.0042670, -0.0131942, 0.0036566, -0.0167625, 0.0170192
3: 0.0002533, 0.0058390, 0.0004183, 0.0057493, -0.0054959, 0.0054206
4: -0.0050151, 0.0091422, -0.0042393, 0.0088450, -0.0138601, 0.0133815
5: -0.0015972, 0.0107855, -0.0013288, 0.0105754, -0.0121726, 0.0121143
6: -0.0061125, 0.0033564, -0.0049142, 0.0030615, -0.0091741, 0.0082706
7: -0.0113782, -0.0009677, -0.0112270, -0.0013953, -0.0099830, 0.0102592
8: -0.0120366, 0.0160813, -0.0114390, 0.0155872, -0.0274709, 0.0273633
9: -0.0094458, 0.0067242, -0.0091714, 0.0063737, -0.0158194, 0.0158956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153328, upper bound: 0.0156663
time: 2.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153257, upper bound: 0.0155835
time: 1.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0052190, 0.0031035, -0.0044495, 0.0028313, -0.0080503, 0.0075530
1: 0.9891621, 1.0073781, 0.9893225, 1.0058502, -0.0166880, 0.0180556
2: -0.0135702, 0.0042670, -0.0132914, 0.0034701, -0.0165579, 0.0171196
3: 0.0002533, 0.0058390, 0.0004688, 0.0057725, -0.0055191, 0.0053702
4: -0.0050151, 0.0091422, -0.0040023, 0.0089218, -0.0139369, 0.0131445
5: -0.0015972, 0.0107855, -0.0012468, 0.0106297, -0.0122269, 0.0120322
6: -0.0061125, 0.0033564, -0.0045480, 0.0031377, -0.0092502, 0.0079044
7: -0.0113782, -0.0009677, -0.0112661, -0.0015259, -0.0098523, 0.0102983
8: -0.0120366, 0.0160813, -0.0112564, 0.0157148, -0.0276006, 0.0271804
9: -0.0094458, 0.0067242, -0.0092423, 0.0062666, -0.0157123, 0.0159665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153328, upper bound: 0.0156663
time: 2.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153257, upper bound: 0.0155836
time: 2.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0055674, 0.0035597, -0.0043851, 0.0029677, -0.0085351, 0.0079448
1: 0.9895711, 1.0080701, 0.9889787, 1.0057223, -0.0161512, 0.0190914
2: -0.0128591, 0.0046279, -0.0138897, 0.0034034, -0.0157684, 0.0179714
3: 0.0001558, 0.0056694, 0.0004868, 0.0059151, -0.0057593, 0.0051826
4: -0.0054737, 0.0085801, -0.0039176, 0.0093947, -0.0148684, 0.0124977
5: -0.0017560, 0.0103881, -0.0012175, 0.0109639, -0.0127199, 0.0116056
6: -0.0068209, 0.0027988, -0.0044171, 0.0036069, -0.0104278, 0.0072159
7: -0.0110921, -0.0007150, -0.0115067, -0.0015726, -0.0095195, 0.0107918
8: -0.0123898, 0.0151469, -0.0111912, 0.0165009, -0.0287182, 0.0261621
9: -0.0089269, 0.0069315, -0.0096788, 0.0062283, -0.0151552, 0.0166103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150328, upper bound: 0.0154385
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0149844, upper bound: 0.0152757
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0055674, 0.0035597, -0.0041727, 0.0030035, -0.0085709, 0.0077324
1: 0.9895711, 1.0080701, 0.9888883, 1.0053006, -0.0157295, 0.0191818
2: -0.0128591, 0.0046279, -0.0140467, 0.0031835, -0.0155379, 0.0181152
3: 0.0001558, 0.0056694, 0.0005462, 0.0059526, -0.0057968, 0.0051232
4: -0.0054737, 0.0085801, -0.0036380, 0.0095188, -0.0149925, 0.0122182
5: -0.0017560, 0.0103881, -0.0011207, 0.0110517, -0.0128077, 0.0115089
6: -0.0068209, 0.0027988, -0.0039968, 0.0037300, -0.0105509, 0.0067956
7: -0.0110921, -0.0007150, -0.0115700, -0.0017267, -0.0093655, 0.0108550
8: -0.0123898, 0.0151469, -0.0109758, 0.0167073, -0.0289203, 0.0259518
9: -0.0089269, 0.0069315, -0.0097934, 0.0061020, -0.0150289, 0.0167249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150328, upper bound: 0.0154385
time: 3.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0149844, upper bound: 0.0152756
time: 1.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0052190, 0.0031035, -0.0053256, 0.0032431, -0.0084621, 0.0084291
1: 0.9891621, 1.0073781, 0.9893335, 1.0075898, -0.0184277, 0.0180446
2: -0.0135702, 0.0042670, -0.0132722, 0.0043775, -0.0174764, 0.0170316
3: 0.0002533, 0.0058390, 0.0002235, 0.0057679, -0.0055145, 0.0056155
4: -0.0050151, 0.0091422, -0.0051555, 0.0089066, -0.0139217, 0.0142977
5: -0.0015972, 0.0107855, -0.0016458, 0.0106189, -0.0122162, 0.0124313
6: -0.0061125, 0.0033564, -0.0063293, 0.0031226, -0.0092352, 0.0096857
7: -0.0113782, -0.0009677, -0.0112583, -0.0008904, -0.0104879, 0.0102906
8: -0.0120366, 0.0160813, -0.0121447, 0.0156896, -0.0275583, 0.0280666
9: -0.0094458, 0.0067242, -0.0092283, 0.0067877, -0.0162334, 0.0159525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150434, upper bound: 0.0150934
time: 2.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0150956, upper bound: 0.0153825
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0052190, 0.0031035, -0.0049276, 0.0029980, -0.0082170, 0.0080311
1: 0.9891621, 1.0073781, 0.9889023, 1.0067996, -0.0176374, 0.0184758
2: -0.0135702, 0.0042670, -0.0140224, 0.0039653, -0.0170327, 0.0177202
3: 0.0002533, 0.0058390, 0.0003349, 0.0059468, -0.0056934, 0.0055041
4: -0.0050151, 0.0091422, -0.0046316, 0.0094996, -0.0145147, 0.0137738
5: -0.0015972, 0.0107855, -0.0014645, 0.0110381, -0.0126354, 0.0122500
6: -0.0061125, 0.0033564, -0.0055201, 0.0037110, -0.0098235, 0.0088765
7: -0.0113782, -0.0009677, -0.0115602, -0.0011791, -0.0101991, 0.0105924
8: -0.0120366, 0.0160813, -0.0117412, 0.0166754, -0.0285311, 0.0276506
9: -0.0094458, 0.0067242, -0.0097757, 0.0065509, -0.0159967, 0.0164999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=40, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150434, upper bound: 0.0156298
time: 1.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150956, upper bound: 0.0158567
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041482, 0.0029847, -0.0060192, 0.0041511, -0.0082993, 0.0090039
1: 0.9889359, 1.0052520, 0.9897842, 1.0089670, -0.0200311, 0.0154678
2: -0.0139642, 0.0031581, -0.0124882, 0.0050957, -0.0185636, 0.0152696
3: 0.0005531, 0.0059329, 0.0000293, 0.0055809, -0.0050279, 0.0059036
4: -0.0036057, 0.0094535, -0.0060683, 0.0082870, -0.0118928, 0.0155219
5: -0.0011095, 0.0110056, -0.0019617, 0.0101809, -0.0112905, 0.0129673
6: -0.0039682, 0.0036653, -0.0077394, 0.0025080, -0.0064761, 0.0114047
7: -0.0115367, -0.0017445, -0.0109429, -0.0003873, -0.0111494, 0.0091985
8: -0.0109510, 0.0165988, -0.0128478, 0.0146596, -0.0254585, 0.0292892
9: -0.0097332, 0.0060874, -0.0086563, 0.0072001, -0.0169333, 0.0147437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152461, upper bound: 0.0156514
time: 1.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152741, upper bound: 0.0156514
time: 2.29 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0044176, 0.0028313, -0.0056245, 0.0036344, -0.0080520, 0.0084558
1: 0.9893225, 1.0057867, 0.9891459, 1.0081834, -0.0188609, 0.0166408
2: -0.0132913, 0.0034371, -0.0135988, 0.0046870, -0.0175260, 0.0166160
3: 0.0004777, 0.0057725, 0.0001398, 0.0058458, -0.0053681, 0.0056327
4: -0.0039603, 0.0089218, -0.0055489, 0.0091647, -0.0131251, 0.0144707
5: -0.0012322, 0.0106297, -0.0017819, 0.0108014, -0.0120337, 0.0124116
6: -0.0044832, 0.0031377, -0.0069370, 0.0033788, -0.0078620, 0.0100747
7: -0.0112661, -0.0015491, -0.0113897, -0.0006736, -0.0105925, 0.0098406
8: -0.0112241, 0.0157148, -0.0124477, 0.0161187, -0.0271946, 0.0280076
9: -0.0092423, 0.0062476, -0.0094666, 0.0069654, -0.0162077, 0.0157142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156262, upper bound: 0.0155580
time: 2.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156215, upper bound: 0.0153468
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0044176, 0.0028313, -0.0054231, 0.0033707, -0.0077883, 0.0082544
1: 0.9893225, 1.0057867, 0.9890565, 1.0077832, -0.0184607, 0.0167302
2: -0.0132913, 0.0034371, -0.0137542, 0.0044784, -0.0173080, 0.0167504
3: 0.0004777, 0.0057725, 0.0001962, 0.0058828, -0.0054051, 0.0055763
4: -0.0039603, 0.0089218, -0.0052838, 0.0092876, -0.0132479, 0.0142055
5: -0.0012322, 0.0106297, -0.0016902, 0.0108883, -0.0121205, 0.0123199
6: -0.0044832, 0.0031377, -0.0065275, 0.0035006, -0.0079838, 0.0096652
7: -0.0112661, -0.0015491, -0.0114523, -0.0008197, -0.0104464, 0.0099032
8: -0.0112241, 0.0157148, -0.0122435, 0.0163229, -0.0274022, 0.0278056
9: -0.0092423, 0.0062476, -0.0095800, 0.0068456, -0.0160879, 0.0158276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156262, upper bound: 0.0155580
time: 2.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156215, upper bound: 0.0153468
time: 1.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042835, 0.0030623, -0.0056245, 0.0036344, -0.0079179, 0.0086868
1: 0.9887401, 1.0055205, 0.9891459, 1.0081834, -0.0194432, 0.0163746
2: -0.0143045, 0.0032982, -0.0135988, 0.0046870, -0.0184389, 0.0164213
3: 0.0005152, 0.0060140, 0.0001398, 0.0058458, -0.0053305, 0.0058743
4: -0.0037838, 0.0097225, -0.0055489, 0.0091647, -0.0129485, 0.0152714
5: -0.0011712, 0.0111957, -0.0017819, 0.0108014, -0.0119726, 0.0129777
6: -0.0042105, 0.0039322, -0.0069370, 0.0033788, -0.0075893, 0.0108692
7: -0.0116736, -0.0016463, -0.0113897, -0.0006736, -0.0110001, 0.0097434
8: -0.0110881, 0.0170460, -0.0124477, 0.0161187, -0.0270386, 0.0293186
9: -0.0099815, 0.0061678, -0.0094666, 0.0069654, -0.0169469, 0.0156344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156774, upper bound: 0.0152035
time: 1.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161726, upper bound: 0.0158062
time: 3.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042835, 0.0030623, -0.0054231, 0.0033707, -0.0076542, 0.0084854
1: 0.9887401, 1.0055205, 0.9890565, 1.0077832, -0.0190430, 0.0164640
2: -0.0143045, 0.0032982, -0.0137542, 0.0044784, -0.0182208, 0.0165570
3: 0.0005152, 0.0060140, 0.0001962, 0.0058828, -0.0053676, 0.0058179
4: -0.0037838, 0.0097225, -0.0052838, 0.0092876, -0.0130714, 0.0150063
5: -0.0011712, 0.0111957, -0.0016902, 0.0108883, -0.0120594, 0.0128859
6: -0.0042105, 0.0039322, -0.0065275, 0.0035006, -0.0077112, 0.0104596
7: -0.0116736, -0.0016463, -0.0114523, -0.0008197, -0.0108539, 0.0098059
8: -0.0110881, 0.0170460, -0.0122435, 0.0163229, -0.0272469, 0.0291165
9: -0.0099815, 0.0061678, -0.0095800, 0.0068456, -0.0168271, 0.0157478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0156774, upper bound: 0.0152029
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161726, upper bound: 0.0158063
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0056500, 0.0036678, -0.0055033, 0.0034758, -0.0091258, 0.0091711
1: 0.9896704, 1.0082339, 0.9898560, 1.0079428, -0.0182724, 0.0183779
2: -0.0126862, 0.0047134, -0.0123636, 0.0045615, -0.0168004, 0.0166550
3: 0.0001327, 0.0056282, 0.0001737, 0.0055512, -0.0054186, 0.0054544
4: -0.0055825, 0.0084435, -0.0053894, 0.0081885, -0.0137709, 0.0138328
5: -0.0017936, 0.0102915, -0.0017268, 0.0101112, -0.0119048, 0.0120183
6: -0.0069888, 0.0026632, -0.0066906, 0.0024102, -0.0093991, 0.0093538
7: -0.0110226, -0.0006551, -0.0108928, -0.0007615, -0.0102611, 0.0102377
8: -0.0124736, 0.0149197, -0.0123248, 0.0144958, -0.0268121, 0.0270886
9: -0.0088007, 0.0069806, -0.0085653, 0.0068933, -0.0156941, 0.0155459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159371, upper bound: 0.0158935
time: 2.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159266, upper bound: 0.0157672
time: 2.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0056500, 0.0036678, -0.0053267, 0.0032446, -0.0088946, 0.0089945
1: 0.9896704, 1.0082339, 0.9897907, 1.0075921, -0.0179217, 0.0184432
2: -0.0126862, 0.0047134, -0.0124769, 0.0043786, -0.0166034, 0.0167634
3: 0.0001327, 0.0056282, 0.0002232, 0.0055783, -0.0054456, 0.0054050
4: -0.0055825, 0.0084435, -0.0051569, 0.0082781, -0.0138605, 0.0136004
5: -0.0017936, 0.0102915, -0.0016463, 0.0101746, -0.0119682, 0.0119378
6: -0.0069888, 0.0026632, -0.0063316, 0.0024991, -0.0094879, 0.0089948
7: -0.0110226, -0.0006551, -0.0109384, -0.0008896, -0.0101330, 0.0102833
8: -0.0124736, 0.0149197, -0.0121458, 0.0146448, -0.0269649, 0.0269096
9: -0.0088007, 0.0069806, -0.0086480, 0.0067883, -0.0155890, 0.0156286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159371, upper bound: 0.0158934
time: 2.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159266, upper bound: 0.0157672
time: 2.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053089, 0.0032212, -0.0063453, 0.0045780, -0.0098869, 0.0095665
1: 0.9892541, 1.0075566, 0.9901875, 1.0096145, -0.0203604, 0.0173691
2: -0.0134108, 0.0043601, -0.0117869, 0.0054335, -0.0183972, 0.0157582
3: 0.0002281, 0.0058009, -0.0000620, 0.0054137, -0.0051856, 0.0058629
4: -0.0051334, 0.0090162, -0.0064976, 0.0077327, -0.0128662, 0.0155138
5: -0.0016382, 0.0106964, -0.0021102, 0.0097891, -0.0114273, 0.0128066
6: -0.0062953, 0.0032314, -0.0084024, 0.0019581, -0.0082533, 0.0116338
7: -0.0113141, -0.0009025, -0.0106608, -0.0001507, -0.0111634, 0.0097582
8: -0.0121277, 0.0158718, -0.0131784, 0.0137382, -0.0257073, 0.0289021
9: -0.0093294, 0.0067777, -0.0081446, 0.0073941, -0.0167235, 0.0149223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158922, upper bound: 0.0156056
time: 1.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158877, upper bound: 0.0155262
time: 2.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053089, 0.0032212, -0.0060192, 0.0041511, -0.0094600, 0.0092404
1: 0.9892541, 1.0075566, 0.9897842, 1.0089670, -0.0197130, 0.0177723
2: -0.0134108, 0.0043601, -0.0124882, 0.0050957, -0.0180345, 0.0164120
3: 0.0002281, 0.0058009, 0.0000293, 0.0055809, -0.0053528, 0.0057716
4: -0.0051334, 0.0090162, -0.0060683, 0.0082870, -0.0134205, 0.0150845
5: -0.0016382, 0.0106964, -0.0019617, 0.0101809, -0.0118191, 0.0126581
6: -0.0062953, 0.0032314, -0.0077394, 0.0025080, -0.0088032, 0.0109708
7: -0.0113141, -0.0009025, -0.0109429, -0.0003873, -0.0109268, 0.0100404
8: -0.0121277, 0.0158718, -0.0128478, 0.0146596, -0.0266276, 0.0285639
9: -0.0093294, 0.0067777, -0.0086563, 0.0072001, -0.0165296, 0.0154340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158922, upper bound: 0.0160962
time: 2.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158877, upper bound: 0.0160225
time: 2.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0053267, 0.0032446, -0.0063377, 0.0045680, -0.0098948, 0.0095822
1: 0.9897907, 1.0075921, 0.9896187, 1.0095994, -0.0198087, 0.0179734
2: -0.0124769, 0.0043786, -0.0127764, 0.0054256, -0.0174796, 0.0167095
3: 0.0002232, 0.0055783, -0.0000599, 0.0056497, -0.0054265, 0.0056381
4: -0.0051569, 0.0082781, -0.0064875, 0.0085147, -0.0136717, 0.0147656
5: -0.0016463, 0.0101746, -0.0021068, 0.0103419, -0.0119882, 0.0122814
6: -0.0063316, 0.0024991, -0.0083869, 0.0027339, -0.0090655, 0.0108861
7: -0.0109384, -0.0008896, -0.0110588, -0.0001563, -0.0107821, 0.0101692
8: -0.0121458, 0.0146448, -0.0131707, 0.0150382, -0.0270311, 0.0276684
9: -0.0086480, 0.0067883, -0.0088665, 0.0073896, -0.0160376, 0.0156548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157604, upper bound: 0.0160001
time: 1.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0157580, upper bound: 0.0159352
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0054754, 0.0034391, -0.0060033, 0.0041303, -0.0096056, 0.0094424
1: 0.9896449, 1.0078871, 0.9892214, 1.0089353, -0.0192904, 0.0186657
2: -0.0127307, 0.0045325, -0.0134676, 0.0050793, -0.0174060, 0.0174992
3: 0.0001816, 0.0056388, 0.0000338, 0.0058145, -0.0056329, 0.0056050
4: -0.0053526, 0.0084786, -0.0060474, 0.0090611, -0.0144136, 0.0145260
5: -0.0017140, 0.0103163, -0.0019545, 0.0107281, -0.0124421, 0.0122708
6: -0.0066337, 0.0026981, -0.0077070, 0.0032759, -0.0099097, 0.0104051
7: -0.0110405, -0.0007818, -0.0113369, -0.0003989, -0.0106416, 0.0105552
8: -0.0122965, 0.0149781, -0.0128317, 0.0159464, -0.0280765, 0.0276685
9: -0.0088332, 0.0068767, -0.0093709, 0.0071907, -0.0160239, 0.0162476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158696, upper bound: 0.0161791
time: 2.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0158696, upper bound: 0.0165345
time: 2.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0052793, 0.0031824, -0.0063377, 0.0045680, -0.0098473, 0.0095201
1: 0.9892067, 1.0074977, 0.9896187, 1.0095994, -0.0203927, 0.0178790
2: -0.0134930, 0.0043294, -0.0127764, 0.0054256, -0.0183843, 0.0166210
3: 0.0002365, 0.0058205, -0.0000599, 0.0056497, -0.0054132, 0.0058804
4: -0.0050944, 0.0090811, -0.0064875, 0.0085147, -0.0136092, 0.0155686
5: -0.0016247, 0.0107423, -0.0021068, 0.0103419, -0.0119666, 0.0128490
6: -0.0062350, 0.0032958, -0.0083869, 0.0027339, -0.0089689, 0.0116827
7: -0.0113471, -0.0009240, -0.0110588, -0.0001563, -0.0111908, 0.0101348
8: -0.0120977, 0.0159797, -0.0131707, 0.0150382, -0.0269630, 0.0289790
9: -0.0093894, 0.0067601, -0.0088665, 0.0073896, -0.0167790, 0.0156266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160004, upper bound: 0.0158884
time: 3.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159949, upper bound: 0.0157744
time: 1.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0054370, 0.0033889, -0.0060033, 0.0041303, -0.0095673, 0.0093922
1: 0.9890565, 1.0078113, 0.9892214, 1.0089353, -0.0198789, 0.0185899
2: -0.0137542, 0.0044929, -0.0134676, 0.0050793, -0.0183215, 0.0174268
3: 0.0001923, 0.0058828, 0.0000338, 0.0058145, -0.0056222, 0.0058490
4: -0.0053021, 0.0092876, -0.0060474, 0.0090611, -0.0143632, 0.0153350
5: -0.0016966, 0.0108882, -0.0019545, 0.0107281, -0.0124247, 0.0128427
6: -0.0065558, 0.0035006, -0.0077070, 0.0032759, -0.0098317, 0.0112077
7: -0.0114523, -0.0008096, -0.0113369, -0.0003989, -0.0110534, 0.0105274
8: -0.0122576, 0.0163229, -0.0128317, 0.0159464, -0.0280192, 0.0289903
9: -0.0095800, 0.0068539, -0.0093709, 0.0071907, -0.0167707, 0.0162248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163082, upper bound: 0.0162686
time: 2.21 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0163082, upper bound: 0.0163994
time: 2.15 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.09 seconds
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0143405, upper bound: 0.0159067
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0146164, upper bound: 0.0160084
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0143996, upper bound: 0.0160918
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0144204, upper bound: 0.0148678
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0144818, upper bound: 0.0154996
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150194, upper bound: 0.0157833
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153542, upper bound: 0.0156981
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153447, upper bound: 0.0155976
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153542, upper bound: 0.0158531
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153447, upper bound: 0.0157195
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0142558, upper bound: 0.0154503
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0149601, upper bound: 0.0157457
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153363, upper bound: 0.0156881
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153284, upper bound: 0.0155894
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153363, upper bound: 0.0158354
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153284, upper bound: 0.0157073
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158045, upper bound: 0.0155207
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0157829, upper bound: 0.0153098
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158045, upper bound: 0.0155208
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0157829, upper bound: 0.0153097
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156269, upper bound: 0.0155855
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0152334, upper bound: 0.0153559
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156269, upper bound: 0.0155848
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0152334, upper bound: 0.0153566
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156884, upper bound: 0.0153661
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0161730, upper bound: 0.0159010
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156884, upper bound: 0.0153661
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0161730, upper bound: 0.0159010
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159394, upper bound: 0.0159433
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159281, upper bound: 0.0158176
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159394, upper bound: 0.0159433
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159281, upper bound: 0.0158176
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0162430, upper bound: 0.0161865
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0162395, upper bound: 0.0161049
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0162430, upper bound: 0.0161865
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0162395, upper bound: 0.0161049
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158700, upper bound: 0.0162692
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0165786
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158700, upper bound: 0.0162701
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0163198, upper bound: 0.0165786
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0163084, upper bound: 0.0163305
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0165399, upper bound: 0.0164475
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0163084, upper bound: 0.0163305
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0165399, upper bound: 0.0164475
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153328, upper bound: 0.0156663
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153257, upper bound: 0.0155835
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153328, upper bound: 0.0156663
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0153257, upper bound: 0.0155836
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150328, upper bound: 0.0154385
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0149844, upper bound: 0.0152757
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150328, upper bound: 0.0154385
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0149844, upper bound: 0.0152756
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150434, upper bound: 0.0150934
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150956, upper bound: 0.0153825
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150434, upper bound: 0.0156298
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0150956, upper bound: 0.0158567
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0152461, upper bound: 0.0156514
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0152741, upper bound: 0.0156514
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156262, upper bound: 0.0155580
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156215, upper bound: 0.0153468
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156262, upper bound: 0.0155580
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156215, upper bound: 0.0153468
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156774, upper bound: 0.0152035
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0161726, upper bound: 0.0158062
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0156774, upper bound: 0.0152029
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0161726, upper bound: 0.0158063
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159371, upper bound: 0.0158935
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159266, upper bound: 0.0157672
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159371, upper bound: 0.0158934
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159266, upper bound: 0.0157672
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158922, upper bound: 0.0156056
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158877, upper bound: 0.0155262
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158922, upper bound: 0.0160962
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158877, upper bound: 0.0160225
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0157604, upper bound: 0.0160001
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0157580, upper bound: 0.0159352
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158696, upper bound: 0.0161791
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0158696, upper bound: 0.0165345
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0160004, upper bound: 0.0158884
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0159949, upper bound: 0.0157744
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0163082, upper bound: 0.0162686
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.09
Output dim: 1, lower bound: -0.0163082, upper bound: 0.0163994

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 5.16 + 598.73 = 603.89 seconds
